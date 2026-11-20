// Host tests of strix/kv_storage.hpp: bytes, role / tier policy, the quantiser layouts behind the forms, and the draft-cache
// invariant (a drafter reading a quantised cache changes acceptance, never the ids emitted by exact verification).
#include "strix/kv_storage.hpp"
#include "strix/model_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

using strix::kv::Form;
using strix::kv::Role;

// ---- portable deterministic normals (std::normal_distribution is implementation-defined)
struct Rng {
    std::mt19937 g{12345U};
    double uniform() { return (static_cast<double>(g()) + 0.5) / 4294967296.0; }
    float normal() {
        const double u = uniform(), v = uniform();
        return static_cast<float>(std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
    }
};

float bf16_round(float x) {  // round to nearest even BF16 (finite inputs)
    std::uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    u = (u + 0x7FFFU + ((u >> 16) & 1U)) & 0xFFFF0000U;
    float y = 0.0F;
    std::memcpy(&y, &u, 4);
    return y;
}

// e4m3 (OCP FP8, bias 7, max 448, satfinite) round to nearest even, decoded back to float.
float e4m3_round(float x) {
    const float a = std::fabs(x);
    if (a == 0.0F) return 0.0F;
    const float s = x < 0.0F ? -1.0F : 1.0F;
    if (a >= 448.0F) return s * 448.0F;
    int e = 0;
    (void)std::frexp(a, &e);                 // a = m * 2^e, m in [0.5, 1)
    const int exp_unbiased = std::max(e - 1, -6);  // subnormals below 2^-6
    const float step = std::ldexp(1.0F, exp_unbiased - 3);  // 3 mantissa bits
    const float q = std::nearbyint(a / step) * step;
    return s * std::min(q, 448.0F);
}

// Uniform asymmetric b-bit quantise / dequantise of a strided group (per token: the head's channels; per channel: one channel
// over a group of tokens). Scale and minimum stay FP32.
void quant_group(float* x, std::size_t n, std::size_t stride, unsigned bits) {
    float lo = x[0], hi = x[0];
    for (std::size_t i = 1; i < n; ++i) {
        lo = std::min(lo, x[i * stride]);
        hi = std::max(hi, x[i * stride]);
    }
    const float levels = static_cast<float>((1U << bits) - 1U);
    const float sc = hi > lo ? (hi - lo) / levels : 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        float& v = x[i * stride];
        v = sc > 0.0F ? lo + std::nearbyint((v - lo) / sc) * sc : lo;
    }
}

enum class Lay { token, channel };
// rows [tokens][dim]: per token = one group per token row; per channel = one group per (channel, block of g tokens).
std::vector<float> quantise(std::vector<float> m, std::size_t tokens, std::size_t dim, unsigned bits, Lay lay, std::size_t g) {
    if (lay == Lay::token) {
        for (std::size_t t = 0; t < tokens; ++t) quant_group(&m[t * dim], dim, 1, bits);
    } else {
        for (std::size_t t0 = 0; t0 + g <= tokens; t0 += g)  // the tail (< g tokens) stays BF16, as a KIVI-style ring
            for (std::size_t c = 0; c < dim; ++c) quant_group(&m[t0 * dim + c], g, dim, bits);
    }
    return m;
}

std::vector<float> e4m3_token(std::vector<float> m, std::size_t tokens, std::size_t dim) {
    for (std::size_t t = 0; t < tokens; ++t) {
        float amax = 0.0F;
        for (std::size_t c = 0; c < dim; ++c) amax = std::max(amax, std::fabs(m[t * dim + c]));
        const float s = std::max(amax, 1e-30F) / 448.0F;
        for (std::size_t c = 0; c < dim; ++c) m[t * dim + c] = e4m3_round(m[t * dim + c] / s) * s;
    }
    return m;
}

// Softmax attention of each query over all keys: out[q][dim].
std::vector<double> attend(const std::vector<float>& q, std::size_t nq, const std::vector<float>& k, const std::vector<float>& v,
                           std::size_t tokens, std::size_t dim) {
    std::vector<double> out(nq * dim, 0.0), s(tokens);
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    for (std::size_t i = 0; i < nq; ++i) {
        double mx = -1e300;
        for (std::size_t t = 0; t < tokens; ++t) {
            double d = 0.0;
            for (std::size_t c = 0; c < dim; ++c) d += static_cast<double>(q[i * dim + c]) * k[t * dim + c];
            s[t] = d * scale;
            mx = std::max(mx, s[t]);
        }
        double z = 0.0;
        for (std::size_t t = 0; t < tokens; ++t) z += (s[t] = std::exp(s[t] - mx));
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t c = 0; c < dim; ++c) out[i * dim + c] += s[t] / z * v[t * dim + c];
    }
    return out;
}

double rel_err(const std::vector<double>& ref, const std::vector<double>& x) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        num += (x[i] - ref[i]) * (x[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

void test_bytes_and_policy() {
    const strix::kv::Shape dense27{4, 256};
    check(strix::kv::bytes_per_token(Form::bf16, dense27) == 4096U, "bf16 dense 27B layer = 4,096 B/token");
    check(strix::kv::bytes_per_token(Form::e4m3_token, dense27) == 2080U, "e4m3 shadow = 1,024 x 2 codes + 4 x 2 FP32 scales");
    // kv-int8-tok-v1 row: 2 x kv_width + 8 x kv_heads (hip_indexed_attention_launcher.hip kv_int8_row_bytes; memory_plan.cpp)
    check(strix::kv::bytes_per_token(Form::int8_token, {2, 256}) == 2U * 512U + 8U * 2U, "int8 token row = QSA kv_int8_row_bytes");
    for (const Form f : {Form::bf16, Form::int8_token, Form::e4m3_token}) {
        check(!strix::kv::changes_output(f, Role::draft), "a draft cache form never changes the output");
        check(strix::kv::allowed(f, Role::draft, false), "every draft cache form is allowed in an exact tier");
        check(strix::kv::allowed(f, Role::target, true), "every target form is allowed in a FAST tier");
    }
    check(!strix::kv::allowed(Form::e4m3_token, Role::target, false), "an 8-bit target cache is not exact");
    check(strix::kv::allowed(Form::bf16, Role::target, false), "the BF16 target cache is exact");
    check(strix::kv::parse("kv8", Form::e4m3_token) == Form::e4m3_token, "kv8 = the backend's 8-bit form (CUDA)");
    check(strix::kv::parse("kv8", Form::int8_token) == Form::int8_token, "kv8 = the backend's 8-bit form (HIP)");
    check(strix::kv::parse("bf16", Form::e4m3_token) == Form::bf16, "bf16 word");
    check(!strix::kv::parse("kv4", Form::e4m3_token).has_value(), "kv4 has no reader: refused");
    check(strix::kv::name(Form::int8_token) == "kv-int8-tok-v1", "int8 name = the QSA identity word");
}

// One table: the model contract's KV storage is this Form, its identity word comes from kv::name.
void test_contract_binding() {
    static_assert(std::is_same_v<strix::AttentionKvStorage, Form>, "the contract's KV storage is kv::Form");
    static_assert(static_cast<std::uint32_t>(strix::AttentionKvStorage::bf16) == 0U &&
                      static_cast<std::uint32_t>(strix::AttentionKvStorage::int8_token) == 1U,
                  "contract enumerator values unchanged");
    for (Form f : {Form::bf16, Form::int8_token, Form::e4m3_token})
        check(std::string_view(strix::attention_kv_storage_name(f)) == strix::kv::name(f),
              "attention_kv_storage_name = kv::name for every form");
    check(std::string_view(strix::attention_kv_storage_name(strix::AttentionKvStorage::bf16)) == "bf16",
          "contract bf16 word");
    check(std::string_view(strix::attention_kv_storage_name(strix::AttentionKvStorage::int8_token)) == "kv-int8-tok-v1",
          "contract int8 word");
}

// The layout lesson of the source: keys carry fixed outlier channels (per-token scales are set by them, so 4-bit per-token
// keys flatten every other channel); values have per-token magnitude and no channel structure (either layout serves).
void test_layouts() {
    const std::size_t T = 512, D = 128, NQ = 16, G = 32;
    Rng r;
    std::vector<float> k(T * D), v(T * D), q(NQ * D);
    for (std::size_t t = 0; t < T; ++t) {
        const float vt = static_cast<float>(std::exp(0.5 * r.normal()));
        for (std::size_t c = 0; c < D; ++c) {
            float x = r.normal();
            if (c % 32 == 3) x += (c % 64 == 3 ? 12.0F : -9.0F) + 0.5F * r.normal();  // 4 outlier channels
            k[t * D + c] = bf16_round(x);
            v[t * D + c] = bf16_round(vt * r.normal());
        }
    }
    for (auto& x : q) x = r.normal();
    const auto ref = attend(q, NQ, k, v, T, D);
    const auto err = [&](const std::vector<float>& kq, const std::vector<float>& vq) { return rel_err(ref, attend(q, NQ, kq, vq, T, D)); };
    const double k4t = err(quantise(k, T, D, 4, Lay::token, G), v);
    const double k4c = err(quantise(k, T, D, 4, Lay::channel, G), v);
    const double v4t = err(k, quantise(v, T, D, 4, Lay::token, G));
    const double v4c = err(k, quantise(v, T, D, 4, Lay::channel, G));
    const double k8t = err(quantise(k, T, D, 8, Lay::token, G), v);
    const double e4 = err(e4m3_token(k, T, D), e4m3_token(v, T, D));
    const double k4v4 = err(quantise(k, T, D, 4, Lay::channel, G), quantise(v, T, D, 4, Lay::token, G));
    std::printf("attention output rel err: K4 tok %.4f | K4 chan(g%zu) %.4f | V4 tok %.4f | V4 chan %.4f | K8 tok %.4f | "
                "e4m3 tok K+V %.4f | K4chan+V4tok %.4f\n", k4t, G, k4c, v4t, v4c, k8t, e4, k4v4);
    check(k4c * 2.0 < k4t, "4-bit keys: per-channel groups beat per-token by > 2x on outlier-channel keys");
    // values carry no channel outliers: the two layouts tie (the key layout is the one that matters)
    check(v4t < v4c * 1.1 && v4c < v4t * 1.1, "4-bit values: per-token and per-channel within 10 %");
    check(k8t * 4.0 < k4t, "8-bit per-token keys stay far below 4-bit per-token");
    check(e4 < k4v4 * 2.0, "the e4m3 token form is within 2x of the paper's K4V4 layout");
}

// Fake target (greedy next id = hash of the last three ids) and a fake drafter that attends over its own cache in a given
// form and proposes the target's id only when its attention output clears a threshold. Speculative greedy decode with exact
// verification must emit the serial ids for every form; only the acceptance moves.
struct Fake {
    static constexpr std::uint32_t V = 61, D = 32;
    static std::uint32_t target(const std::vector<std::uint32_t>& ctx) {
        const std::size_t n = ctx.size();
        const std::uint32_t a = ctx[n - 1], b = n > 1 ? ctx[n - 2] : 7U, c = n > 2 ? ctx[n - 3] : 11U;
        return (a * 31U + b * 17U + c * 7U + 3U) % V;
    }
    Form form;
    std::vector<float> keys;  // drafter cache [positions][D], stored in `form`
    explicit Fake(Form f) : form(f) {}
    static float emb(std::uint32_t id, std::uint32_t c) {
        return std::sin(static_cast<float>(id * 13U + c * 7U)) * (c == 5 ? 9.0F : 1.0F);
    }
    void store(std::uint32_t id) {  // append one position in the cache form
        std::vector<float> row(D);
        for (std::uint32_t c = 0; c < D; ++c) row[c] = bf16_round(emb(id, c));
        if (form == Form::e4m3_token) row = e4m3_token(row, 1, D);
        if (form == Form::int8_token) {
            float amax = 0.0F;
            for (const float x : row) amax = std::max(amax, std::fabs(x));
            const float s = amax / 127.0F;
            for (auto& x : row) x = s > 0.0F ? std::clamp(std::nearbyint(x / s), -127.0F, 127.0F) * s : 0.0F;
        }
        keys.insert(keys.end(), row.begin(), row.end());
    }
    void truncate(std::size_t positions) { keys.resize(positions * D); }
    std::uint32_t draft(const std::vector<std::uint32_t>& ctx) const {
        const std::uint32_t t = target(ctx);  // the drafter "knows" the answer unless its attention score misses
        double s = 0.0;
        const std::size_t n = keys.size() / D;
        for (std::size_t p = 0; p < n; ++p)
            for (std::uint32_t c = 0; c < D; ++c) s += keys[p * D + c] * emb(ctx.back(), c) / static_cast<double>(n);
        const double frac = s - std::floor(s);
        return frac < 0.62 ? t : (t + 1U) % V;
    }
};

void test_draft_cache_exact() {
    const std::vector<std::uint32_t> prompt{3, 1, 4, 1, 5, 9, 2, 6};
    const std::size_t N = 400;
    std::vector<std::uint32_t> serial = prompt;
    while (serial.size() < prompt.size() + N) serial.push_back(Fake::target(serial));
    for (const Form f : {Form::bf16, Form::int8_token, Form::e4m3_token}) {
        Fake dr(f);
        std::vector<std::uint32_t> out = prompt;
        for (const std::uint32_t id : prompt) dr.store(id);
        std::size_t proposed = 0, accepted = 0;
        while (out.size() < prompt.size() + N) {
            const std::size_t base = out.size();
            std::vector<std::uint32_t> ctx = out, drafts;
            for (int i = 0; i < 4; ++i) {  // chain: each draft id is stored in the drafter cache before the next step
                drafts.push_back(dr.draft(ctx));
                ctx.push_back(drafts.back());
                dr.store(drafts.back());
            }
            dr.truncate(base);  // drafter positions past the committed prefix are rewritten after verify
            proposed += drafts.size();
            for (const std::uint32_t d : drafts) {  // exact greedy verify on the target's own choice
                const std::uint32_t t = Fake::target(out);
                out.push_back(t);
                dr.store(t);
                if (d != t) break;
                ++accepted;
            }
        }
        out.resize(prompt.size() + N);
        check(out == serial, "speculative ids == serial ids for every draft cache form");
        std::printf("draft cache %s: ids == serial %s, acceptance %.4f\n", std::string(strix::kv::name(f)).c_str(),
                    out == serial ? "yes" : "NO", static_cast<double>(accepted) / static_cast<double>(proposed));
    }
}

}  // namespace

int main() {
    test_bytes_and_policy();
    test_contract_binding();
    test_layouts();
    test_draft_cache_exact();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("kv_storage_tests: PASS");
    return 0;
}
