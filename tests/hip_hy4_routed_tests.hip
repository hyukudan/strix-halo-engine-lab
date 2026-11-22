// GPU fixture of the shared Engram row kernel and the shared MoE route / combine kernels against their host oracles,
// synthetic data only (no weights), every comparison bitwise:
//   rows    src/engram/hip/row_kernel.hpp   decode (DS41: E4M3 x E8M0, 264-byte rows) and head_rms_bf16 (Hy4-flash:
//           96 E4M3 codes, 16 heads, eps 2^-7)                              == engram::row_prep_reference
//   route   src/moe_core/route_wave_kernel.hpp on the Hy4 preset (strix/hy4/presets.hpp route: 512 / 16 / x 3.998, ascending ids) and
//           the GLM-5 policy value (288 / 8 / x 2.5, = glm_math.hpp kRoutePolicy), incl. rows of tied keys
//                                                                           == route_scores_ref + route_select_ref
//   combine fma_f32 (GLM: 8 x 4096 + shared), bf16_steps (Hy4: 16 x 1536 latent merge; unit-weight shared add 3072)
//                                                                            == moe::combine_ref
#pragma clang fp contract(off)  // declared arithmetic: the oracles' bits
#include "strix/hy4/presets.hpp"
#include "engram/hip/row_kernel.hpp"
#include "moe_core/route_wave_kernel.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/engram/row_prep.hpp"
#include "strix/moe/combine.hpp"
#include "strix/moe/route_policy.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#define HIPCHECK(x)                                                                                   \
    do {                                                                                              \
        const hipError_t e_ = (x);                                                                    \
        if (e_ != hipSuccess) {                                                                       \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__);       \
            std::exit(2);                                                                             \
        }                                                                                             \
    } while (0)

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
std::uint32_t bits(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
template <class T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    HIPCHECK(hipMalloc(&d, v.size() * sizeof(T) + 16));
    if (!v.empty()) HIPCHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <class T>
std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> v(n);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}
template <class T>
unsigned mismatches(const std::vector<T>& a, const std::vector<T>& b) {
    unsigned n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if constexpr (sizeof(T) == 4 && !std::is_integral_v<T>) n += bits(a[i]) != bits(b[i]) ? 1U : 0U;
        else n += a[i] != b[i] ? 1U : 0U;
    }
    return n;
}

// ------------------------------------------------------------------ Engram rows
template <strix::engram::RowPrepForm Form, strix::engram::RowScale Scale, std::uint32_t Elements>
void rows_case(std::mt19937& rng, std::uint32_t tokens, std::uint32_t heads, const char* what) {
    using namespace strix::engram;
    const RowFormat f{Elements, Scale};
    const std::uint32_t count = tokens * heads;
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(count) * f.bytes());
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint8_t* r = rows.data() + static_cast<std::size_t>(i) * f.bytes();
        for (std::uint32_t j = 0; j < Elements; ++j) {
            auto c = static_cast<std::uint8_t>(rng() & 0xFFU);
            if ((c & 0x7FU) == 0x7FU) c = static_cast<std::uint8_t>(c ^ 1U);  // NaN codes kept out of the bitwise compare
            r[j] = c;
        }
        if constexpr (Scale == RowScale::e8m0_per32)
            for (std::uint32_t b = 0; b < Elements / 32U; ++b) r[Elements + b] = static_cast<std::uint8_t>(100U + rng() % 50U);
    }
    std::vector<float> w(static_cast<std::size_t>(heads) * Elements);
    std::uniform_real_distribution<float> u(-2.0F, 2.0F);
    for (auto& x : w) x = strix::ds41::round_to_bf16(u(rng));
    std::vector<float> want(static_cast<std::size_t>(count) * Elements);
    RowPrepView v;
    v.form = Form;
    v.fmt = f;
    v.rows = rows.data();
    v.row_count = count;
    v.out = want.data();
    v.norm_weight = w.data();
    v.heads = heads;
    v.eps = 0.0078125F;
    row_prep_reference(v);
    auto* drows = upload(rows);
    auto* dw = upload(w);
    float* dout = nullptr;
    HIPCHECK(hipMalloc(&dout, want.size() * sizeof(float)));
    v.rows = drows;
    v.norm_weight = dw;
    v.out = dout;
    hip::enqueue_row_prep<Form, Scale, Elements>(v, nullptr);
    HIPCHECK(hipGetLastError());
    const auto got = download(dout, want.size());
    const unsigned bad = mismatches(got, want);
    std::printf("rows %s: %u rows x %u, bitwise mismatches %u\n", what, count, Elements, bad);
    check(bad == 0U, what);
    (void)hipFree(drows);
    (void)hipFree(dw);
    (void)hipFree(dout);
}

// ------------------------------------------------------------------ route
struct DeclaredMath {
    __host__ __device__ static float exp(float x) { return strix::ds41::declared_exp(x); }
};

// The GLM-5 policy value (sigmoid + bias, ordered keys, ascending ids, 288 / 8 / x 2.5); static storage: route_case
// takes the policy by reference.
constexpr strix::moe::RoutePolicy kSigmoid288{strix::moe::RouteScore::sigmoid, true, strix::moe::RouteKey::ordered_bits,
                                              strix::moe::RouteOrder::ascending_id, 288U, 8U, 1U, 1U, true, true, 1e-20F,
                                              2.5F};
template <const strix::moe::RoutePolicy& P>
void route_case(std::mt19937& rng, unsigned rows, const char* what) {
    using namespace strix::moe;
    std::normal_distribution<float> n(0.0F, 2.0F);
    std::vector<float> z(static_cast<std::size_t>(rows) * P.experts), b(P.experts);
    for (auto& x : z) x = n(rng);
    for (auto& x : b) x = 0.01F * static_cast<float>(rng() % 3U);  // few bias values: tied keys in the coarse rows
    for (unsigned e = 0; e < P.experts; ++e)  // tied keys: row 1 coarse logits (ties at the cut), row 2 all equal
        for (unsigned r = 1; r < 3 && r < rows; ++r) z[static_cast<std::size_t>(r) * P.experts + e] = r == 1 ? static_cast<float>(e % 7U) : 0.25F;
    std::vector<std::uint32_t> want_ids(static_cast<std::size_t>(rows) * P.top_k);
    std::vector<float> want_w(want_ids.size());
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<float> zr(z.begin() + static_cast<std::ptrdiff_t>(r) * P.experts, z.begin() + static_cast<std::ptrdiff_t>(r + 1) * P.experts);
        std::vector<float> s(P.experts);
        route_scores_ref<DeclaredMath>(P.score, zr, s);
        route_select_ref(P, s, b, std::span<std::uint32_t>(want_ids.data() + static_cast<std::size_t>(r) * P.top_k, P.top_k),
                         std::span<float>(want_w.data() + static_cast<std::size_t>(r) * P.top_k, P.top_k));
    }
    auto* dz = upload(z);
    auto* db = upload(b);
    std::uint32_t* dids = nullptr;
    float* dw = nullptr;
    HIPCHECK(hipMalloc(&dids, want_ids.size() * sizeof(std::uint32_t)));
    HIPCHECK(hipMalloc(&dw, want_w.size() * sizeof(float)));
    strix::moe_core::enqueue_route_wave<P, DeclaredMath>(dz, db, dids, dw, rows, nullptr);
    HIPCHECK(hipGetLastError());
    const auto ids = download(dids, want_ids.size());
    const auto w = download(dw, want_w.size());
    const unsigned bad_ids = mismatches(ids, want_ids), bad_w = mismatches(w, want_w);
    std::printf("route %s: %u rows, id mismatches %u, weight bit mismatches %u\n", what, rows, bad_ids, bad_w);
    check(bad_ids == 0U && bad_w == 0U, what);
    (void)hipFree(dz);
    (void)hipFree(db);
    (void)hipFree(dids);
    (void)hipFree(dw);
}

// ------------------------------------------------------------------ combine
template <strix::moe::CombineForm Form, unsigned TopK, unsigned Width>
void combine_case(std::mt19937& rng, unsigned rows, bool weights, bool base, const char* what) {
    using namespace strix::moe;
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::uniform_real_distribution<float> uw(0.0F, 0.6F);
    std::vector<float> w(weights ? static_cast<std::size_t>(rows) * TopK : 0U), eo(static_cast<std::size_t>(rows) * TopK * Width),
        sh(base ? static_cast<std::size_t>(rows) * Width : 0U), want(static_cast<std::size_t>(rows) * Width);
    for (auto& x : w) x = uw(rng);
    for (auto& x : eo) x = Form == CombineForm::bf16_steps ? combine_round_bf16(n(rng)) : n(rng);  // Hy4 expert outputs are BF16 values
    for (auto& x : sh) x = Form == CombineForm::bf16_steps ? combine_round_bf16(n(rng)) : n(rng);
    combine_ref(Form, rows, TopK, Width, w, eo, sh, want);
    float* dw = weights ? upload(w) : nullptr;
    auto* deo = upload(eo);
    float* dsh = base ? upload(sh) : nullptr;
    float* dy = nullptr;
    HIPCHECK(hipMalloc(&dy, want.size() * sizeof(float)));
    strix::moe_core::enqueue_combine<Form, TopK, Width>(dw, deo, dsh, dy, rows, nullptr);
    HIPCHECK(hipGetLastError());
    const auto got = download(dy, want.size());
    const unsigned bad = mismatches(got, want);
    std::printf("combine %s: %u rows x %u, bitwise mismatches %u\n", what, rows, Width, bad);
    check(bad == 0U, what);
    if (dw) (void)hipFree(dw);
    (void)hipFree(deo);
    if (dsh) (void)hipFree(dsh);
    (void)hipFree(dy);
}

}  // namespace

int main() {
    using strix::engram::RowPrepForm;
    using strix::engram::RowScale;
    using strix::moe::CombineForm;
    std::mt19937 rng(20261118U);
    rows_case<RowPrepForm::decode, RowScale::e8m0_per32, 256U>(rng, 37U, 12U, "decode ds41 e8m0_per32 x 256 == reference");
    rows_case<RowPrepForm::head_rms_bf16, RowScale::none, 96U>(rng, 41U, 16U, "head_rms_bf16 hy4 96 x 16 heads == reference");
    route_case<strix::hy4::preset::route>(rng, 64U, "hy4 preset 512 / 16 == route_select_ref");
    route_case<kSigmoid288>(rng, 64U, "sigmoid 288 / 8 x 2.5 (GLM-5 policy) == route_select_ref");
    combine_case<CombineForm::fma_f32, 8U, 4096U>(rng, 5U, true, true, "fma_f32 glm 8 x 4096 + shared == combine_ref");
    combine_case<CombineForm::bf16_steps, 16U, 1536U>(rng, 5U, true, false, "bf16_steps hy4 16 x 1536 latent merge == combine_ref");
    combine_case<CombineForm::bf16_steps, 1U, 3072U>(rng, 5U, false, true, "bf16_steps hy4 shared add 3072 == combine_ref");
    std::printf("%s\n", failures == 0 ? "PASS hy4 routed fixtures" : "FAIL hy4 routed fixtures");
    return failures == 0 ? 0 : 1;
}
