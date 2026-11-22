// No-model bench of the packed-operand tile schedules (core::f16_tile_gemm::TileSchedule) on the grouped expert projection
// at Inkling 2 Mini shapes: 128 experts, 8192 tokens x top-8 = 65,536 routed rows with a skewed expert load; gate-up
// K = 2304, N = 2304 through IndirectRows<ActivatedTileBinding> (paired SwiGLU epilogue), down K = 1152, N = 2304 through
// TileBinding. Synthetic NVFP4 and MXFP8 weights. Every schedule's outputs are compared bitwise with the incumbent
// (TileSchedule<> on 128-row tiles); times are medians of interleaved repetitions (hipEvent, one launch each).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/grouped_microscaled.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace gm = strix::moe_core::grouped_microscaled;
namespace gemm = strix::core::f16_tile_gemm;
namespace packed = strix::weight::microscaled;

constexpr unsigned kTokens = 8192U, kTopK = 8U, kExperts = 128U, kHidden = 2304U, kInter = 1152U;
constexpr unsigned kRouted = kTokens * kTopK;

void check(hipError_t e, const char* what) {
    if (e == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
}

__host__ __device__ inline std::uint32_t mix(std::uint64_t i, std::uint32_t seed) {
    auto x = static_cast<std::uint32_t>(i * 0x9E3779B1ULL) ^ seed ^ static_cast<std::uint32_t>(i >> 32U);
    x ^= x >> 16U; x *= 0x85EBCA6BU; x ^= x >> 13U; x *= 0xC2B2AE35U; x ^= x >> 16U;
    return x;
}

// kind 0: any byte (NVFP4 codes); 1: E4M3 without NaN (MXFP8 codes); 2: NVFP4 E4M3 block scale 0x28..0x3f;
// 3: E8M0 scale 124..128; 4: BF16 in [-1, 1] (u16 pairs written as bytes of the u16 image: index = u16 element).
__global__ void fill(std::uint8_t* p, std::size_t n, std::uint32_t seed, int kind) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        const std::uint32_t h = mix(i, seed);
        if (kind == 0) p[i] = static_cast<std::uint8_t>(h);
        else if (kind == 1) { auto c = static_cast<std::uint8_t>(h); if ((c & 0x7fU) == 0x7fU) c = static_cast<std::uint8_t>(c - 1U); p[i] = c; }
        else if (kind == 2) p[i] = static_cast<std::uint8_t>(0x28U + h % 24U);
        else if (kind == 3) p[i] = static_cast<std::uint8_t>(124U + h % 5U);
        else {
            const float f = static_cast<float>(static_cast<int>(h % 2001U) - 1000) * 1e-3F;
            const std::uint32_t bits = __float_as_uint(f);
            reinterpret_cast<std::uint16_t*>(p)[i] = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
        }
    }
}

__global__ void diff(const std::uint32_t* a, const std::uint32_t* b, std::size_t n, unsigned long long* count) {
    unsigned local = 0U;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x)
        local += a[i] != b[i] ? 1U : 0U;
    if (local) atomicAdd(count, static_cast<unsigned long long>(local));
}

template <class T> T* dalloc(std::size_t n) {
    void* p = nullptr;
    check(hipMalloc(&p, n * sizeof(T)), "hipMalloc");
    return static_cast<T*>(p);
}

void fill_dev(void* p, std::size_t n, std::uint32_t seed, int kind) {
    hipLaunchKernelGGL(fill, dim3(4096), dim3(256), 0, nullptr, static_cast<std::uint8_t*>(p), n, seed, kind);
    check(hipGetLastError(), "fill");
}

unsigned long long count_diff(const void* a, const void* b, std::size_t words, unsigned long long* scratch) {
    check(hipMemset(scratch, 0, sizeof(unsigned long long)), "memset");
    hipLaunchKernelGGL(diff, dim3(4096), dim3(256), 0, nullptr, static_cast<const std::uint32_t*>(a),
                       static_cast<const std::uint32_t*>(b), words, scratch);
    unsigned long long n = 0;
    check(hipMemcpy(&n, scratch, sizeof n, hipMemcpyDeviceToHost), "diff copy");
    return n;
}

// Skewed expert load: lognormal weights (sigma 0.8), every expert >= 1 row, sum = kRouted.
std::vector<unsigned> expert_rows() {
    std::vector<double> w(kExperts);
    std::uint64_t state = 0x1234567ULL;
    const auto uniform = [&] { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return ((state >> 11U) + 0.5) / 9007199254740992.0; };
    for (auto& x : w) x = std::exp(0.8 * std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307179586 * uniform()));
    double sum = 0; for (double x : w) sum += x;
    std::vector<unsigned> n(kExperts);
    unsigned total = 0;
    for (unsigned e = 0; e < kExperts; ++e) total += n[e] = std::max(1U, static_cast<unsigned>(w[e] / sum * kRouted));
    for (unsigned e = 0; total != kRouted; e = (e + 1U) % kExperts) {
        if (total < kRouted) { ++n[e]; ++total; } else if (n[e] > 1U) { --n[e]; --total; }
    }
    return n;
}

struct Weights {
    std::uint8_t *gu_codes, *gu_scales, *dn_codes, *dn_scales;
    std::size_t gu_code_row, gu_scale_row, dn_code_row, dn_scale_row;
};

struct Plan {
    unsigned rows_per_tile;
    std::vector<gm::TileRows> gu, pair, down;
    gm::TileRows *d_gu, *d_pair, *d_down;
    unsigned count;
};

template <packed::MatrixFormat F>
Plan make_plan(unsigned per_tile, const std::vector<unsigned>& n, const Weights& w, const std::uint16_t* tokens,
               std::uint16_t* act, float* out) {
    constexpr bool nvfp4 = F == packed::MatrixFormat::nvfp4;
    const float outer = nvfp4 ? 0.0625F : 1.0F;
    Plan p{per_tile, {}, {}, {}, nullptr, nullptr, nullptr, 0U};
    unsigned row0 = 0;
    for (unsigned e = 0; e < kExperts; ++e) {
        const packed::PackedMatrix gu{w.gu_codes + e * kHidden * w.gu_code_row, w.gu_scales + e * kHidden * w.gu_scale_row,
                                      2U * kInter, kHidden, outer, packed::RowOrder::interleaved_halves};
        const packed::PackedMatrix dn{w.dn_codes + e * kHidden * w.dn_code_row, w.dn_scales + e * kHidden * w.dn_scale_row,
                                      kHidden, kInter, outer, packed::RowOrder::contiguous};
        for (unsigned t = 0; t * per_tile < n[e]; ++t) {
            const unsigned r = row0 + t * per_tile, rows = std::min(per_tile, n[e] - t * per_tile);
            p.gu.push_back({gu, tokens + std::size_t{r} * kHidden, out, kHidden, 2U * kInter, rows});  // output unused (paired)
            p.pair.push_back({dn, act + std::size_t{r} * kInter, nullptr, kInter, 0U, rows});
            p.down.push_back({dn, act + std::size_t{r} * kInter, out + std::size_t{r} * kHidden, kInter, kHidden, rows});
            if (!gm::fits_rows<F>(p.gu.back(), kHidden, 2U * kInter, 32U, per_tile) || !gm::fits_rows<F>(p.down.back(), kInter, kHidden, 32U, per_tile)) {
                std::fprintf(stderr, "tile refused\n");
                std::exit(1);
            }
        }
        row0 += n[e];
    }
    p.count = static_cast<unsigned>(p.gu.size());
    for (auto [host, dev] : {std::pair{&p.gu, &p.d_gu}, std::pair{&p.pair, &p.d_pair}, std::pair{&p.down, &p.d_down}}) {
        *dev = dalloc<gm::TileRows>(host->size());
        check(hipMemcpy(*dev, host->data(), host->size() * sizeof(gm::TileRows), hipMemcpyHostToDevice), "plan copy");
    }
    return p;
}

struct Buffers {
    const std::uint16_t* tokens;
    const std::uint32_t* gather;
    std::uint16_t* act_ref;  // pair-row base of every plan (rebase_from); down input
};

template <packed::MatrixFormat F, class Sched>
hipError_t gate_up(const Plan& p, const Buffers& b, std::uint16_t* act_out, hipStream_t s) {
    gm::IndirectArgs a{};
    a.tiles = p.d_gu; a.count = p.count; a.inputs = kHidden; a.outputs = 2U * kInter; a.max_tile_rows = p.rows_per_tile;
    a.rebase_from = b.act_ref; a.rebase_to = act_out; a.pair_rows = p.d_pair; a.gate_cap = 10.0F; a.up_cap = 10.0F;
    a.gather_rows = b.gather; a.gather_from = b.tokens;
    const unsigned slices = (p.rows_per_tile + Sched::kRows - 1U) / Sched::kRows;
    return gemm::gemm_packed_bf16<gm::IndirectRows<gm::ActivatedTileBinding<F>>, Sched>(a, a.outputs, a.count, s, slices);
}

// The down tiles write `out_ref`-relative rows; `shift` (elements) moves them to the variant's buffer.
template <packed::MatrixFormat F, class Sched>
hipError_t down(const Plan& p, const gm::TileRows* tiles, const Buffers& b, hipStream_t s) {
    gm::Args a{};
    a.tiles = tiles; a.count = p.count; a.inputs = kInter; a.outputs = kHidden; a.max_tile_rows = p.rows_per_tile;
    a.rebase_from = b.act_ref; a.rebase_to = b.act_ref;
    const unsigned slices = (p.rows_per_tile + Sched::kRows - 1U) / Sched::kRows;
    return gemm::gemm_packed_bf16<gm::TileBinding<F>, Sched>(a, a.outputs, a.count, s, slices);
}

struct Variant {
    const char* name;
    unsigned plan;  // 128 or 256 rows per descriptor tile
    hipError_t (*gu)(const Plan&, const Buffers&, std::uint16_t*, hipStream_t);
    hipError_t (*dn)(const Plan&, const gm::TileRows*, const Buffers&, hipStream_t);
};

template <packed::MatrixFormat F, class Sched> Variant variant(const char* name, unsigned plan) {
    return {name, plan, gate_up<F, Sched>, down<F, Sched>};
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2U];
}

template <packed::MatrixFormat F>
bool run(const std::vector<unsigned>& n, int reps, const std::string& only) {
    constexpr bool nvfp4 = F == packed::MatrixFormat::nvfp4;
    const char* format = nvfp4 ? "nvfp4" : "mxfp8";
    Weights w{};
    w.gu_code_row = nvfp4 ? kHidden / 2U : kHidden; w.gu_scale_row = kHidden / (nvfp4 ? 16U : 32U);
    w.dn_code_row = nvfp4 ? kInter / 2U : kInter; w.dn_scale_row = kInter / (nvfp4 ? 16U : 32U);
    const std::size_t gu_codes = std::size_t{kExperts} * 2U * kInter * w.gu_code_row, gu_scales = std::size_t{kExperts} * 2U * kInter * w.gu_scale_row;
    const std::size_t dn_codes = std::size_t{kExperts} * kHidden * w.dn_code_row, dn_scales = std::size_t{kExperts} * kHidden * w.dn_scale_row;
    w.gu_codes = dalloc<std::uint8_t>(gu_codes); w.gu_scales = dalloc<std::uint8_t>(gu_scales);
    w.dn_codes = dalloc<std::uint8_t>(dn_codes); w.dn_scales = dalloc<std::uint8_t>(dn_scales);
    fill_dev(w.gu_codes, gu_codes, 11U, nvfp4 ? 0 : 1); fill_dev(w.dn_codes, dn_codes, 12U, nvfp4 ? 0 : 1);
    fill_dev(w.gu_scales, gu_scales, 13U, nvfp4 ? 2 : 3); fill_dev(w.dn_scales, dn_scales, 14U, nvfp4 ? 2 : 3);

    auto* tokens = dalloc<std::uint16_t>(std::size_t{kTokens} * kHidden);
    fill_dev(tokens, std::size_t{kTokens} * kHidden, 21U, 4);
    std::vector<std::uint32_t> gather(kRouted);
    for (unsigned i = 0; i < kRouted; ++i) gather[i] = mix(i, 77U) % kTokens;
    auto* d_gather = dalloc<std::uint32_t>(kRouted);
    check(hipMemcpy(d_gather, gather.data(), kRouted * 4U, hipMemcpyHostToDevice), "gather");
    const std::size_t act_n = std::size_t{kRouted} * kInter, out_n = std::size_t{kRouted} * kHidden;
    auto* act_ref = dalloc<std::uint16_t>(act_n);
    auto* act_test = dalloc<std::uint16_t>(act_n);
    auto* out_ref = dalloc<float>(out_n);
    auto* out_test = dalloc<float>(out_n);
    fill_dev(act_ref, act_n, 31U, 4);  // down input until the reference gate-up overwrites it
    auto* scratch = dalloc<unsigned long long>(1);
    const Buffers b{tokens, d_gather, act_ref};

    Plan plans[2] = {make_plan<F>(128U, n, w, tokens, act_ref, out_ref), make_plan<F>(256U, n, w, tokens, act_ref, out_ref)};
    // Down tiles writing out_test: same rows, shifted output base.
    gm::TileRows* test_down[2];
    for (int i = 0; i < 2; ++i) {
        auto tiles = plans[i].down;
        for (auto& t : tiles) t.output = out_test + (t.output - out_ref);
        test_down[i] = dalloc<gm::TileRows>(tiles.size());
        check(hipMemcpy(test_down[i], tiles.data(), tiles.size() * sizeof(gm::TileRows), hipMemcpyHostToDevice), "tiles");
    }
    std::printf("%s tiles: plan128 %u, plan256 %u\n", format, plans[0].count, plans[1].count);

    using S0 = gemm::TileSchedule<2, 4, 32>;
    const std::vector<Variant> all{
        variant<F, S0>("S0 incumbent r128 k32", 128U),
        variant<F, S0>("S0 on plan256 (2 slices)", 256U),
        variant<F, gemm::TileSchedule<2, 4, 64>>("S4 r128 k64", 128U),
        variant<F, gemm::TileSchedule<4, 4, 32>>("S2 r256 w8 64x64 k32", 256U),
        variant<F, gemm::TileSchedule<2, 8, 64>>("S3 r256 w16 k64", 256U),
    };
    hipStream_t s;
    check(hipStreamCreate(&s), "stream");
    // Reference outputs (incumbent on plan128).
    check(all[0].gu(plans[0], b, act_ref, s), "ref gate_up");
    check(all[0].dn(plans[0], plans[0].d_down, b, s), "ref down");
    check(hipStreamSynchronize(s), "ref sync");

    bool ok = true;
    std::vector<std::vector<double>> tg(all.size()), td(all.size());
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
    const auto timed = [&](auto&& launch) {
        check(hipEventRecord(e0, s), "rec");
        check(launch(), "launch");
        check(hipEventRecord(e1, s), "rec");
        check(hipEventSynchronize(e1), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return static_cast<double>(ms);
    };
    std::vector<bool> selected(all.size());
    for (std::size_t v = 0; v < all.size(); ++v)
        selected[v] = only.empty() || v == 0 || only.find(std::string(all[v].name).substr(0, 3)) != std::string::npos;
    // Bitwise check (outputs poisoned first), one untimed warm launch.
    for (std::size_t v = 0; v < all.size(); ++v) {
        if (!selected[v]) continue;
        const Plan& p = plans[all[v].plan == 128U ? 0 : 1];
        check(hipMemsetAsync(act_test, 0xff, act_n * 2U, s), "poison");
        check(hipMemsetAsync(out_test, 0xff, out_n * 4U, s), "poison");
        check(all[v].gu(p, b, act_test, s), "gate_up");
        check(all[v].dn(p, test_down[all[v].plan == 128U ? 0 : 1], b, s), "down");
        check(hipStreamSynchronize(s), "sync");
        const auto dg = count_diff(act_ref, act_test, act_n / 2U, scratch);
        const auto dd = count_diff(out_ref, out_test, out_n, scratch);
        std::printf("%s %-26s bitwise gate_up %s (%llu words differ) down %s (%llu differ)\n", format, all[v].name,
                    dg ? "DIFF" : "EQUAL", dg, dd ? "DIFF" : "EQUAL", dd);
        ok = ok && dg == 0 && dd == 0;
    }
    for (int r = 0; r < reps; ++r)
        for (std::size_t v = 0; v < all.size(); ++v) {
            if (!selected[v]) continue;
            const int pi = all[v].plan == 128U ? 0 : 1;
            tg[v].push_back(timed([&] { return all[v].gu(plans[pi], b, act_test, s); }));
            td[v].push_back(timed([&] { return all[v].dn(plans[pi], test_down[pi], b, s); }));
        }
    const double g0 = median(tg[0]), d0 = median(td[0]);
    for (std::size_t v = 0; v < all.size(); ++v) {
        if (!selected[v]) continue;
        const double g = median(tg[v]), d = median(td[v]);
        std::printf("%s %-26s gate_up %7.3f ms (%+5.1f %%, min %7.3f)  down %7.3f ms (%+5.1f %%, min %7.3f)\n", format,
                    all[v].name, g, 100.0 * (g / g0 - 1.0), *std::min_element(tg[v].begin(), tg[v].end()), d,
                    100.0 * (d / d0 - 1.0), *std::min_element(td[v].begin(), td[v].end()));
    }
    check(hipStreamDestroy(s), "stream");
    for (void* p : {static_cast<void*>(w.gu_codes), static_cast<void*>(w.gu_scales), static_cast<void*>(w.dn_codes),
                    static_cast<void*>(w.dn_scales), static_cast<void*>(tokens), static_cast<void*>(d_gather),
                    static_cast<void*>(act_ref), static_cast<void*>(act_test), static_cast<void*>(out_ref),
                    static_cast<void*>(out_test), static_cast<void*>(scratch)})
        check(hipFree(p), "free");
    for (int i = 0; i < 2; ++i) {
        check(hipFree(plans[i].d_gu), "free"); check(hipFree(plans[i].d_pair), "free");
        check(hipFree(plans[i].d_down), "free"); check(hipFree(test_down[i]), "free");
    }
    return ok;
}

}  // namespace

// Usage: strix-hip-grouped-tile-schedules [reps=7] [nvfp4|mxfp8|both] [variant prefixes, e.g. "S2 S3"]
int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 7;
    const std::string formats = argc > 2 ? argv[2] : "both";
    const std::string only = argc > 3 ? argv[3] : "";
    const auto n = expert_rows();
    unsigned lo = n[0], hi = n[0];
    for (unsigned x : n) { lo = std::min(lo, x); hi = std::max(hi, x); }
    std::printf("experts %u routed rows %u (per expert min %u max %u mean %u)\n", kExperts, kRouted, lo, hi, kRouted / kExperts);
    bool ok = true;
    if (formats != "mxfp8") ok = run<packed::MatrixFormat::nvfp4>(n, reps, only) && ok;
    if (formats != "nvfp4") ok = run<packed::MatrixFormat::mxfp8>(n, reps, only) && ok;
    std::printf("%s\n", ok ? "ALL BITWISE EQUAL" : "BITWISE MISMATCH");
    return ok ? 0 : 1;
}
