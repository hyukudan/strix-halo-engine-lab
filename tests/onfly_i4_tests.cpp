// TURBO+ onfly INT4 g64: converter bit-exactness against an independent scalar reference, tile layout, resolver
// (CLI and server resolve one request through one function), cache identity and the q6p::iu4 registry.
#include "../apps/qwen_effective_config.hpp"
#include "core/iu4_onfly.hpp"
#include "strix/q6p_registry.hpp"

#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace strix;
namespace {
int failures = 0;
void expect(bool ok, const char* message) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
std::uint32_t lcg(std::uint32_t& s) { s = s * 1664525U + 1013904223U; return s >> 8U; }

// Oracle: nearest binary16 by exhaustive search over the finite values (ties to the even mantissa).
std::uint16_t oracle_f16(float f) {
    std::uint16_t best = 0;
    double best_err = 1e300;
    for (std::uint32_t h = 0; h < 0x7c00U; ++h) {
        const double v = core::onfly::f16_bits_to_f32(static_cast<std::uint16_t>(h));
        const double err = std::fabs(v - static_cast<double>(f));
        if (err < best_err || (err == best_err && !(h & 1U))) { best_err = err; best = static_cast<std::uint16_t>(h); }
    }
    return best;
}
int clamp_rne(float x, int lo) {  // independent round-half-even
    const double fl = std::floor(static_cast<double>(x));
    const double fr = static_cast<double>(x) - fl;
    int q = static_cast<int>(fl);
    if (fr > 0.5 || (fr == 0.5 && (q & 1))) ++q;
    return q < lo ? lo : (q > 7 ? 7 : q);
}
}  // namespace

int main() {
    // binary16 software conversion: all finite halves round-trip; hand-checked rounding cases; oracle on random floats.
    for (std::uint32_t h = 0; h < 0x7c00U; ++h)
        if (core::onfly::f32_to_f16_bits(core::onfly::f16_bits_to_f32(static_cast<std::uint16_t>(h))) != h) { expect(false, "f16 round trip"); break; }
    expect(core::onfly::f32_to_f16_bits(1.0f) == 0x3c00U, "f16 one");
    expect(core::onfly::f32_to_f16_bits(0x1p-25f) == 0U, "f16 tie to even (zero)");
    expect(core::onfly::f32_to_f16_bits(1.5f * 0x1p-24f) == 2U, "f16 tie to even (two)");
    expect(core::onfly::f32_to_f16_bits(65504.0f) == 0x7bffU, "f16 max");
    std::uint32_t seed = 7;
    for (int i = 0; i < 300; ++i) {
        const float f = (static_cast<float>(lcg(seed) % 20001U) - 10000.0f) * 0x1p-12f * static_cast<float>(1 + lcg(seed) % 4U);
        const float a = std::fabs(f);
        if (core::onfly::f32_to_f16_bits(a) != oracle_f16(a)) { expect(false, "f16 versus exhaustive oracle"); break; }
    }

    // Converter: N = 70 (padded to 128), K = 192 (3 groups), both scale rules.
    constexpr std::size_t N = 70, K = 192;
    std::vector<std::uint8_t> q8(N * K / 32U * 34U);
    for (std::size_t c = 0; c < N; ++c)
        for (std::size_t b = 0; b < K / 32U; ++b) {
            auto* blk = q8.data() + (c * (K / 32U) + b) * 34U;
            const std::uint16_t d = core::onfly::f32_to_f16_bits(0.001f * static_cast<float>(1 + lcg(seed) % 40U));
            blk[0] = static_cast<std::uint8_t>(d & 255U); blk[1] = static_cast<std::uint8_t>(d >> 8U);
            for (int j = 0; j < 32; ++j) blk[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(lcg(seed) % 255U) - 127));
        }
    for (std::size_t g = 0; g < K / 64U; ++g) {  // an all-zero group and a one-sided group
        for (int b = 0; b < 2; ++b) std::memset(q8.data() + (3U * (K / 32U) + 2U * g + b) * 34U + 2U, 0, 32);
        break;
    }
    for (int j = 0; j < 32; ++j) q8[(5U * (K / 32U) + 2U) * 34U + 2U + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(j + 1));
    for (const int rule : {core::iu4_scale_rule_legacy, core::iu4_scale_rule_fast_full}) {
        const auto copy = core::onfly::convert_q8_0_host(q8.data(), N, K, rule);
        const std::size_t padded = core::onfly::padded_outputs(N);
        expect(padded == 128U && copy.packed.size() == padded * K / 2U && copy.scales.size() == padded * (K / 64U), "copy sizes");
        bool exact = true, pad_zero = true;
        for (std::size_t c = 0; c < N && exact; ++c)
            for (std::size_t g = 0; g < K / 64U && exact; ++g) {
                float w[64];
                float pos = 0, neg = 0, amax = 0;
                for (int j = 0; j < 64; ++j) {
                    const std::uint8_t* blk = q8.data() + (c * (K / 32U) + 2U * g + j / 32) * 34U;
                    const float d = core::onfly::f16_bits_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
                    w[j] = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + j % 32]));
                    pos = std::fmax(pos, w[j]); neg = std::fmax(neg, -w[j]); amax = std::fmax(amax, std::fabs(w[j]));
                }
                float s;
                if (rule == core::iu4_scale_rule_fast_full) {
                    const float a = pos / 7.0f, b = neg / 8.0f, base = a > b ? a : b;
                    s = base == 0.0f ? 1.0f : std::fmax(0.9f * base, 0x1p-24f);
                } else s = amax == 0.0f ? 1.0f : std::fmax(amax / 7.0f, 0x1p-24f);
                const std::uint16_t sb = oracle_f16(s);
                const float inv = 1.0f / core::onfly::f16_bits_to_f32(sb);
                if (copy.scales[c * (K / 64U) + g] != sb) { exact = false; expect(false, "scale bits"); }
                for (int j = 0; j < 64; ++j) {
                    const int q = clamp_rne(w[j] * inv, rule == core::iu4_scale_rule_fast_full ? -8 : -7);
                    const std::size_t k = g * 64U + static_cast<std::size_t>(j);
                    // [K/16][N/16][column within tile][K within tile / 2]
                    const std::size_t byte = ((k / 16U * (padded / 16U) + c / 16U) * 16U + c % 16U) * 8U + (k % 16U) / 2U;
                    const int got = core::iu4_unpack(copy.packed[byte], static_cast<int>(k & 1U));
                    if (got != q) { exact = false; expect(false, "nibble"); break; }
                }
            }
        expect(exact, "converter equals the independent scalar reference");
        for (std::size_t c = N; c < padded; ++c)
            for (std::size_t g = 0; g < K / 64U; ++g) pad_zero = pad_zero && copy.scales[c * (K / 64U) + g] == 0;
        expect(pad_zero, "pad columns have scale 0");
    }

    // MSE clip search: equals an independent search over the same candidates and never has more squared error than the
    // fast full range rule summed over every group (0.90 is one of the candidates).
    {
        const auto mse = core::onfly::convert_q8_0_host(q8.data(), N, K, core::iu4_scale_rule_mse);
        const auto full = core::onfly::convert_q8_0_host(q8.data(), N, K, core::iu4_scale_rule_fast_full);
        const std::size_t padded = core::onfly::padded_outputs(N);
        double err_mse = 0, err_full = 0;
        bool same = true;
        for (std::size_t c = 0; c < N; ++c)
            for (std::size_t g = 0; g < K / 64U; ++g) {
                float w[64], pos = 0, neg = 0;
                core::onfly::dequant_q8_group(q8.data() + c * core::onfly::q8_row_bytes(K), static_cast<unsigned>(g), w);
                for (float x : w) { pos = std::fmax(pos, x); neg = std::fmax(neg, -x); }
                const float base = std::fmax(pos / 7.0f, neg / 8.0f);
                std::uint16_t want = oracle_f16(1.0f);
                float best = -1.0f;
                for (int i = 0; base != 0.0f && i < 21; ++i) {
                    const std::uint16_t cb = oracle_f16(std::fmax((1.0f - 0.02f * static_cast<float>(i)) * base, 0x1p-24f));
                    const float cs = core::onfly::f16_bits_to_f32(cb);
                    float e = 0.0f;
                    for (float x : w) { const float d = x - cs * static_cast<float>(clamp_rne(x * (1.0f / cs), -8)); e += d * d; }
                    if (best < 0.0f || e < best) { best = e; want = cb; }
                }
                same = same && mse.scales[c * (K / 64U) + g] == want;
                const auto group_err = [&](const core::onfly::HostCopy& cp) {
                    const float s = core::onfly::f16_bits_to_f32(cp.scales[c * (K / 64U) + g]);
                    double e = 0;
                    for (int j = 0; j < 64; ++j) {
                        const std::size_t k = g * 64U + static_cast<std::size_t>(j);
                        const std::size_t byte = ((k / 16U * (padded / 16U) + c / 16U) * 16U + c % 16U) * 8U + (k % 16U) / 2U;
                        const double d = static_cast<double>(w[j]) -
                                         static_cast<double>(s) * static_cast<double>(core::iu4_unpack(cp.packed[byte], static_cast<int>(k & 1U)));
                        e += d * d;
                    }
                    return e;
                };
                err_mse += group_err(mse);
                err_full += group_err(full);
            }
        expect(same, "mse rule equals the independent clip search");
        expect(err_mse <= err_full, "mse rule error <= fast full range error");
        std::printf("onfly mse/full squared error ratio %.4f\n", err_full > 0 ? err_mse / err_full : 1.0);
    }

    // Sub-switch resolver: native default, onfly (decided by the profile weights) via CLI or environment gives the same value.
    const auto no_env = [](const char*) -> const char* { return nullptr; };
    expect(!onfly_i4::resolve({}, no_env, false).active() && onfly_i4::resolve({}, no_env, false).identity().empty(), "native default has no identity");
    const auto via_cli = onfly_i4::resolve({"gdn_in,qsa_io", "legacy"}, no_env, true);
    const auto via_env = onfly_i4::resolve({}, [](const char* n) -> const char* {
        const std::string name = n;
        if (name == "STRIX_TURBO_ONFLY_GROUPS") return "gdn_in,qsa_io";
        if (name == "STRIX_TURBO_ONFLY_SCALE") return "legacy";
        return nullptr;
    }, true);
    expect(via_cli.identity() == via_env.identity() && via_cli.active() && via_cli.groups == 5U, "CLI and environment resolve to the same value");
    expect(via_cli.identity() != onfly_i4::resolve({}, no_env, true).identity(), "groups and rule enter the identity");
    for (const auto& bad : {onfly_i4::Request{"nope", ""}, onfly_i4::Request{"", "fast"}}) {
        bool refused = false;
        try { (void)onfly_i4::resolve(bad, no_env, true); } catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "unsupported request refused with a reason");
    }
    expect(onfly_i4::classify("blk.3.attn_qkv.weight") == onfly_i4::gdn_in && onfly_i4::classify("blk.3.ssm_out.weight") == onfly_i4::gdn_out &&
           onfly_i4::classify("blk.3.attn_output.weight") == onfly_i4::qsa_io && onfly_i4::classify("blk.3.ffn_down_shexp.weight") == onfly_i4::shared_expert &&
           onfly_i4::classify("blk.3.ffn_gate_inp.weight") == 0U && onfly_i4::classify("token_embd.weight") == 0U, "tensor groups");

    // One resolver for the CLI and the server: the same request through Options + environment (what both build) agrees,
    // the identity enters the cache argv, native keeps the historical argv.
    const auto options_for = [](std::vector<std::string> extra) {
        std::vector<std::string> args{"decoder", "target.gguf", "1", "2", "3", "--generate", "4"};
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        return decode_cli::parse_options(static_cast<int>(argv.size()), argv.data());
    };
    const auto cli = decode_cli::initialize_prefill_config({options_for({"--profile", "turbo", "--turbo-weights", "onfly", "--turbo-onfly-groups=gdn_in,gdn_out"}), {}});
    const auto serve = decode_cli::initialize_prefill_config({options_for({"--profile", "turbo", "--turbo-weights", "onfly"}), {{"STRIX_TURBO_ONFLY_GROUPS", "gdn_in,gdn_out"}}});
    const auto native = decode_cli::initialize_prefill_config({options_for({}), {}});
    expect(cli.options.turbo.identity() == serve.options.turbo.identity() && cli.options.turbo.active(), "CLI and server resolve the same effective turbo value");
    expect(cli.options.product_profile.onfly.identity() == cli.options.turbo.identity() &&
           serve.options.product_profile.onfly.identity() == serve.options.turbo.identity(), "capabilities retain the resolved request");
    expect(decode_cli::effective_prefill_argv(cli.options) == decode_cli::effective_prefill_argv(serve.options), "CLI and server give the same cache argv");
    expect(decode_cli::effective_prefill_argv(cli.options) != decode_cli::effective_prefill_argv(native.options), "onfly never shares a cache key with native");
    expect(!native.options.turbo.active(), "native default");

    // Registry.
    int tensor = 0;
    const auto backing = std::make_shared<std::array<unsigned char, 8>>();
    auto* copy = backing->data();
    expect(q6p::iu4::find(&tensor) == nullptr && q6p::iu4::count() == 0, "registry empty");
    q6p::iu4::register_tensor(&tensor, {copy, 70, 192, backing});
    const auto found = q6p::iu4::find(&tensor);
    expect(found && found->rows == 70 && found->cols == 192 && q6p::iu4::count() == 1, "registered copy found");
    q6p::iu4::unregister_tensor(&tensor, copy + 1);
    expect(q6p::iu4::find(&tensor) != nullptr, "unregister of another owner's copy is a no-op");
    q6p::iu4::unregister_tensor(&tensor, copy);
    expect(q6p::iu4::find(&tensor) == nullptr && q6p::iu4::count() == 0, "retired");

    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "onfly i4 host tests passed\n";
    return 0;
}
