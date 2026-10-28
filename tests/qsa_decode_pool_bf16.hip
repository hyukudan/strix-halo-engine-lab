// Standalone production pooled-scorer storage gate and bandwidth timing fixture.
// Powers of two, boundary counts and seeded random counts span 1..262144 blocks.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace {
template <unsigned Rows> struct QsaRowsScoreViews {
    const float* index_query[Rows];
    float* block_scores[Rows];
    std::uint32_t blocks[Rows];
};
__device__ float bf16_to_float(std::uint16_t u) {
    return __uint_as_float(static_cast<std::uint32_t>(u) << 16U);
}
__device__ std::uint16_t float_to_bf16(float v) {
    const auto bits = __float_as_uint(v);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
__device__ float round_bf16(float v) { return bf16_to_float(float_to_bf16(v)); }
enum class SmallWeightKind { f32 };
template <SmallWeightKind Kind> __device__ float load_small_weight(const void* w, std::uint32_t i) {
    static_assert(Kind == SmallWeightKind::f32); return static_cast<const float*>(w)[i];
}
#include "../src/hip_qsa_prefill_stages.inc"
#include "../src/hip_qsa_scatter.inc"
#include "../src/hip_qsa_pooled_score.inc"

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
template <class T> T* device(std::size_t n) {
    void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "allocate"); return static_cast<T*>(p);
}
struct Rng {
    std::uint64_t s = 0x9e3779b97f4a7c15ULL;
    std::uint32_t next() { s ^= s << 13U; s ^= s >> 7U; s ^= s << 17U; return static_cast<std::uint32_t>(s >> 16U); }
    std::uint16_t bf16() {
        // Finite BF16 values of both signs, including zero and subnormals.
        return static_cast<std::uint16_t>((next() & 0x8000U) | (next() % 0x4080U));
    }
};
float widen(std::uint16_t u) { return std::bit_cast<float>(static_cast<std::uint32_t>(u) << 16U); }
float median(std::vector<float> v) {
    std::sort(v.begin(), v.end()); return (v[(v.size() - 1U) / 2U] + v[v.size() / 2U]) * 0.5F;
}
bool packed_only_store(Rng& rng) {
    constexpr unsigned blocks = 33U, dim = 128U, ratio = 4U, cells = blocks * dim;
    std::vector<std::uint16_t> raw(cells * ratio), reference(cells), packed(cells);
    for (auto& v : raw) v = static_cast<std::uint16_t>(0x3e00U + rng.next() % 0x200U);
    std::vector<float> norm(dim, 1.0F), pooled(cells);
    auto* d_raw = device<std::uint16_t>(raw.size()); auto* d_norm = device<float>(dim);
    auto* d_f = device<float>(cells); auto* d_ref = device<std::uint16_t>(cells); auto* d_p = device<std::uint16_t>(cells);
    check(hipMemcpy(d_raw, raw.data(), raw.size() * sizeof(raw[0]), hipMemcpyHostToDevice), "raw keys");
    check(hipMemcpy(d_norm, norm.data(), norm.size() * sizeof(norm[0]), hipMemcpyHostToDevice), "norm");
    check(hipMemset(d_ref, 0xa5, cells * sizeof(std::uint16_t)), "pool canary");
    check(hipMemset(d_p, 0xa5, cells * sizeof(std::uint16_t)), "pool canary");
    // Nonzero frontier and an untouched block at either end check incremental writes.
    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<SmallWeightKind::f32>), dim3(31), dim3(dim), 2U * dim * sizeof(float), nullptr,
        d_raw, 1U, dim, ratio, d_norm, 1e-6F, 64U, 10000.0F, d_f, nullptr, 0U, d_ref);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<SmallWeightKind::f32,false,true>), dim3(31), dim3(dim), 2U * dim * sizeof(float), nullptr,
        d_raw, 1U, dim, ratio, d_norm, 1e-6F, 64U, 10000.0F, nullptr, nullptr, 0U, d_p);
    check(hipGetLastError(), "pool launch");
    check(hipMemcpy(reference.data(), d_ref, cells * sizeof(reference[0]), hipMemcpyDeviceToHost), "reference pool");
    check(hipMemcpy(packed.data(), d_p, cells * sizeof(packed[0]), hipMemcpyDeviceToHost), "packed-only pool");
    check(hipMemcpy(pooled.data(), d_f, cells * sizeof(pooled[0]), hipMemcpyDeviceToHost), "FP32 pool");
    bool ok = reference == packed;
    for (unsigned i = 0; i < cells; ++i) {
        if (i < dim || i >= 32U * dim) ok = ok && packed[i] == 0xa5a5U;
        else ok = ok && std::bit_cast<std::uint32_t>(pooled[i]) == (static_cast<std::uint32_t>(packed[i]) << 16U);
    }
    check(hipFree(d_raw), "free"); check(hipFree(d_norm), "free"); check(hipFree(d_f), "free");
    check(hipFree(d_ref), "free"); check(hipFree(d_p), "free");
    std::printf("EXACT packed-only store with null FP32 output %s\n", ok ? "PASS" : "FAIL"); return ok;
}
} // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::puts("SKIP: no HIP device; native BF16 exactness and TIME pending"); return 77;
    }
    hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, 0), "device");
    std::printf("device %s arch %s\n", prop.name, prop.gcnArchName);
    constexpr unsigned dim = 128U, heads = 4U, maximum = 262144U, guard = 8U;
    Rng rng;
    if (!packed_only_store(rng)) return 1;
    std::vector<unsigned> counts{1U,31U,32U,33U,255U,256U,257U,65535U,65536U,65537U};
    for (unsigned n = 1U; n <= maximum; n *= 2U) counts.push_back(n);
    for (unsigned i = 0; i < 16U; ++i) counts.push_back(1U + rng.next() % maximum);
    std::sort(counts.begin(), counts.end()); counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    std::vector<std::uint16_t> packed(std::size_t{maximum} * dim);
    std::vector<float> keys(packed.size());
    for (std::size_t i = 0; i < keys.size(); ++i) { packed[i] = rng.bf16(); keys[i] = widen(packed[i]); }
    auto* d_p = device<std::uint16_t>(packed.size()); auto* d_k = device<float>(keys.size());
    auto* d_q = device<float>(heads * dim);
    auto* d_f = device<float>(maximum + guard); auto* d_b = device<float>(maximum + guard);
    check(hipMemcpy(d_p, packed.data(), packed.size() * sizeof(packed[0]), hipMemcpyHostToDevice), "packed keys");
    check(hipMemcpy(d_k, keys.data(), keys.size() * sizeof(keys[0]), hipMemcpyHostToDevice), "float keys");
    hipEvent_t begin{}, end{}; check(hipEventCreate(&begin), "event"); check(hipEventCreate(&end), "event");
    std::vector<std::uint32_t> fp32(maximum + guard), bf16(maximum + guard);
    unsigned cases = 0U;
    for (const auto blocks : counts) {
        std::array<float, heads * dim> query{};
        for (auto& q : query) q = widen(rng.bf16());
        // Exact signed zero and cancellation cases among ordinary random queries.
        if (blocks == 1U) query.fill(-0.0F);
        check(hipMemcpy(d_q, query.data(), sizeof(query), hipMemcpyHostToDevice), "query");
        const auto launch = [&]<class Key>(const Key* input, float* output) {
            QsaRowsScoreViews<1U> views{};
            views.index_query[0] = d_q; views.block_scores[0] = output; views.blocks[0] = blocks;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<1U,1U,Key>),
                dim3(((blocks + 31U) / 32U + 7U) / 8U), dim3(256), 0, nullptr, views, input, blocks, dim);
            check(hipGetLastError(), "score launch");
        };
        check(hipMemset(d_f, 0xa5, (blocks + guard) * sizeof(float)), "canary");
        check(hipMemset(d_b, 0xa5, (blocks + guard) * sizeof(float)), "canary");
        launch(d_k, d_f); launch(d_p, d_b);
        check(hipMemcpy(fp32.data(), d_f, (blocks + guard) * sizeof(float), hipMemcpyDeviceToHost), "FP32 scores");
        check(hipMemcpy(bf16.data(), d_b, (blocks + guard) * sizeof(float), hipMemcpyDeviceToHost), "BF16 scores");
        for (unsigned i = 0; i < blocks + guard; ++i) {
            if (fp32[i] != bf16[i] || (i >= blocks && fp32[i] != 0xa5a5a5a5U) || (i < blocks && fp32[i] == 0xa5a5a5a5U)) {
                std::printf("FAIL blocks=%u word=%u FP32=%08x BF16=%08x\n", blocks, i, fp32[i], bf16[i]); return 1;
            }
        }
        ++cases;
        std::printf("EXACT blocks=%u scores=%u PASS\n", blocks, blocks);
        if (blocks == 65536U || blocks == 131072U || blocks == 262144U) {
            const auto timed = [&](bool packed_keys) {
                check(hipEventRecord(begin), "begin");
                if (packed_keys) launch(d_p, d_b); else launch(d_k, d_f);
                check(hipEventRecord(end), "end"); check(hipEventSynchronize(end), "sync");
                float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "elapsed"); return ms;
            };
            timed(false); timed(true); timed(false); timed(true);
            std::vector<float> f, b;
            for (unsigned r = 0; r < 10U; ++r) { // interleaved ABBA; 20 samples per arm
                f.push_back(timed(false)); b.push_back(timed(true)); b.push_back(timed(true)); f.push_back(timed(false));
            }
            std::printf("TIME blocks=%u FP32_median_ms=%.6f BF16_median_ms=%.6f samples=%zu\n", blocks, median(f), median(b), f.size());
        }
    }
    check(hipEventDestroy(begin), "destroy event"); check(hipEventDestroy(end), "destroy event");
    check(hipFree(d_p), "free"); check(hipFree(d_k), "free"); check(hipFree(d_q), "free");
    check(hipFree(d_f), "free"); check(hipFree(d_b), "free");
    std::printf("RESULT PASS cases=%u\n", cases); return 0;
}
