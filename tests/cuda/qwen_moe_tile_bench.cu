// No-model bench of the CUDA tensor-core expert route (gsq-iq-bf16mma-cuda-v1, src/cuda/core/moe_a8_mma.cuh) that
// separates the per-tile fixed cost from the per-m16-block cost of today's 64-row tiles. Synthetic IQ3_S gate/up and
// IQ4_NL down weights for all 512 experts (random codes, finite block scales); 500 experts are each routed exactly c rows
// (rows = 50 c, the ten experts of a row are distinct). Each config prints the median ms of the routed experts
// (enqueue_routed minus enqueue_csr), the tile count read back from the device and the m16-block count; a least-squares
// fit t = a + tiles D + mblocks M over all configs then prints D, M and the share of the 8K expert time that is per-tile.
// Each config also runs the wide 128-row form (Weights::wide_tile, forced at every load) and prints its time, its tile
// count and whether its rawdots / intermediates hash equals the 64-row form's bit for bit (both buffers NaN-filled before
// each checked run, so a skipped output cannot match).
//   usage: qwen_moe_tile_bench [reps]      (needs the card; never loads a model)
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "strix/cuda/amd_math_tables.hpp"
#include "strix/cuda/qwen_moe.hpp"

namespace mo = strix::cuda::qwen_moe;
using u8 = std::uint8_t; using u16 = std::uint16_t; using u32 = std::uint32_t; using u64 = std::uint64_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

namespace {
constexpr u32 kHidden = mo::hidden, kFfn = mo::ffn, kTop = mo::top_k, kExperts = mo::experts;
constexpr u32 kUsed = 500;                         // experts routed per config (kUsed * c / kTop rows, integer for any c)
constexpr u32 kGateType = 21, kGateBlock = 110;    // IQ3_S: 256 values, fp16 d at byte 0
constexpr u32 kDownType = 20, kDownBlock = 18;     // IQ4_NL: 32 values, fp16 d at byte 0
constexpr u16 kScaleBits = 0x0C00;                 // fp16 2^-12: every product and sum stays finite
constexpr u32 kCases[] = {16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320};
constexpr u32 kMaxRows = kUsed / kTop * 320;

u64 g_state = 0x9E3779B97F4A7C15ULL;
u64 next() { g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17; return g_state; }

std::vector<u8> weights(std::size_t blocks, u32 block_bytes) {
    std::vector<u8> m(blocks * block_bytes);
    for (std::size_t i = 0; i + 8 <= m.size(); i += 8) { const u64 r = next(); for (u32 j = 0; j < 8; ++j) m[i + j] = u8(r >> (8 * j)); }
    for (std::size_t b = 0; b < blocks; ++b) { m[b * block_bytes] = u8(kScaleBits & 255U); m[b * block_bytes + 1] = u8(kScaleBits >> 8); }
    return m;
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; CK(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T))); CK(cudaMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T))); return p; }
template <class T> T* upload(const std::vector<T>& v) { T* p = dev<T>(v.size()); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
u32 ceil_div(u32 a, u32 b) { return (a + b - 1U) / b; }
// Order-independent 64-bit hash of n words (bitwise equality check of two runs' outputs).
__global__ void hash_kernel(const u32* w, std::size_t n, unsigned long long* out) {
    unsigned long long h = 0ULL;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        unsigned long long x = (static_cast<unsigned long long>(i) << 32U) ^ w[i];
        x *= 0x9E3779B97F4A7C15ULL; x ^= x >> 29U; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 32U;
        h += x;
    }
    atomicAdd(out, h);
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::max(3, std::atoi(argv[1])) : 9;
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("TILEBENCH device %s sm_%d%d SMs %d reps %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, reps);

    const std::size_t gate_blocks = std::size_t{kExperts} * kFfn * (kHidden / 256U), down_blocks = std::size_t{kExperts} * kHidden * (kFfn / 32U);
    u8* gate = upload(weights(gate_blocks, kGateBlock));
    u8* up = upload(weights(gate_blocks, kGateBlock));
    u8* down = upload(weights(down_blocks, kDownBlock));
    mo::Weights w;
    w.gate_type = kGateType; w.gate = gate; w.up = up; w.down_type = kDownType; w.down = down;
    w.arithmetic = mo::ExpertArithmetic::gsq_bf16_mma;

    std::vector<u16> input(std::size_t{kMaxRows} * kHidden);
    for (auto& x : input) x = u16(0x3C00U + (next() & 0x7FU) + ((next() & 1U) << 15));   // BF16 in +-[0.0078, 0.0156)
    const auto sz = mo::csr_sizes(kMaxRows);
    mo::Rows v;
    u32* routes = dev<u32>(std::size_t{kMaxRows} * kTop);
    v.input = {upload(input), kHidden * 2}; v.routes = {routes, kTop * 4};
    std::vector<float> rw(std::size_t{kMaxRows} * kTop, 0.1F);
    v.route_weights = {upload(rw), kTop * 4};
    v.codes = {dev<std::int8_t>(std::size_t{kMaxRows} * kHidden), kHidden};
    v.deltas = {dev<float>(std::size_t{kMaxRows} * kHidden / 32), kHidden / 32 * 4};
    v.sums = {dev<std::int32_t>(std::size_t{kMaxRows} * kHidden / 32), kHidden / 32 * 4};
    v.status = {dev<u32>(std::size_t{kMaxRows} * 4), 16};
    v.intermediates = {dev<float>(std::size_t{kMaxRows} * kTop * 2 * kFfn), kTop * 2 * kFfn * 4};
    v.accumulation = {dev<float>(std::size_t{kMaxRows} * kHidden), kHidden * 4};
    v.down_codes = {dev<std::int8_t>(std::size_t{kMaxRows} * kTop * kFfn), kTop * kFfn};
    v.down_deltas = {dev<float>(std::size_t{kMaxRows} * kTop * kFfn / 32), kTop * kFfn / 32 * 4};
    v.down_sums = {dev<std::int32_t>(std::size_t{kMaxRows} * kTop * kFfn / 32), kTop * kFfn / 32 * 4};
    v.rawdots = {dev<float>(std::size_t{kMaxRows} * kTop * kHidden), kTop * kHidden * 4};
    v.output = {dev<u16>(std::size_t{kMaxRows} * kHidden), kHidden * 2};
    v.csr.summary = dev<u32>(32); v.csr.offsets = dev<u32>(kExperts + 1); v.csr.touched = dev<u32>(kExperts);
    v.csr.rows = dev<u32>(sz.assignments); v.csr.slots = dev<u32>(sz.assignments);
    v.csr.batch_experts = dev<u32>(sz.max_batches); v.csr.batch_begins = dev<u32>(sz.max_batches + 1);
    v.csr.tile_count = dev<u32>(1); v.csr.tile_experts = dev<u32>(sz.max_tiles); v.csr.tile_begins = dev<u32>(sz.max_tiles + 1);
    const strix::cuda::amdmath::Tables tables{};   // unused by the tensor-core GSQ route
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    unsigned long long* hash_dev = dev<unsigned long long>(1);
    const auto hash = [&](const void* p, std::size_t bytes) {
        CK(cudaMemsetAsync(hash_dev, 0, 8, s));
        hash_kernel<<<1024, 256, 0, s>>>(static_cast<const u32*>(p), bytes / 4U, hash_dev);
        CK(cudaGetLastError());
        unsigned long long h = 0ULL;
        CK(cudaMemcpyAsync(&h, hash_dev, 8, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        return h;
    };
    std::vector<double> T, M, Y;
    std::vector<std::pair<u32, double>> wide_ratio;   // (rows per expert, wide / tile time)
    int bad = 0;
    for (const u32 c : kCases) {
        const u32 rows = kUsed / kTop * c;
        std::vector<u32> r(std::size_t{rows} * kTop);
        for (u32 i = 0; i < rows; ++i)
            for (u32 j = 0; j < kTop; ++j) r[std::size_t{i} * kTop + j] = j * (kUsed / kTop) + i / c;   // slot-major: expert gets [e c, (e+1) c)
        CK(cudaMemcpy(routes, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
        const std::size_t raw_bytes = std::size_t{rows} * kTop * kHidden * 4U, mid_bytes = std::size_t{rows} * kTop * 2U * kFfn * 4U;
        double form_ms[2] = {};
        unsigned long long form_hash[2][2] = {};
        for (u32 form = 0; form < 2U; ++form) {   // 0: 64-row tile form, 1: wide 128-row form (forced at every load)
            w.wide_tile = form == 1U; w.wide_min_avg = 1U;
            const u32 tile_rows = form ? 128U : 64U;
            auto timed = [&](bool whole) {
                mo::enqueue_prologue(v, rows, kHidden, s);
                CK(cudaEventRecord(e0, s));
                if (whole) (void)mo::enqueue_routed(v, rows, w, tables, s); else mo::enqueue_csr(v, rows, s, tile_rows);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms = 0.0F;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                return ms;
            };
            timed(true); timed(true);
            std::vector<float> routed, csr;
            for (int k = 0; k < reps; ++k) { routed.push_back(timed(true)); csr.push_back(timed(false)); }
            CK(cudaMemset(v.rawdots.base, 0xFF, raw_bytes)); CK(cudaMemset(v.intermediates.base, 0xFF, mid_bytes));
            (void)timed(true);
            form_hash[form][0] = hash(v.rawdots.base, raw_bytes); form_hash[form][1] = hash(v.intermediates.base, mid_bytes);
            u32 summary = 0, tiles = 0;
            CK(cudaMemcpy(&summary, v.csr.summary, 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&tiles, v.csr.tile_count, 4, cudaMemcpyDeviceToHost));
            const u32 want_tiles = kUsed * ceil_div(c, tile_rows), mblocks = kUsed * ceil_div(c, 16);
            const double expert_ms = double(median(routed)) - double(median(csr));
            const bool ok = summary == 0U && tiles == want_tiles;
            bad += ok ? 0 : 1;
            form_ms[form] = expert_ms;
            std::printf("TILEBENCH%s c=%u rows=%u tiles=%u mblocks=%u routed_ms=%.3f csr_ms=%.3f expert_ms=%.3f us_per_tile=%.3f summary=%u %s\n",
                        form ? "W" : "", c, rows, tiles, mblocks, median(routed), median(csr), expert_ms, 1000.0 * expert_ms / tiles, summary,
                        ok ? "OK" : "BAD");
            if (form == 0U) { T.push_back(want_tiles); M.push_back(mblocks); Y.push_back(expert_ms); }
        }
        const bool equal = form_hash[0][0] == form_hash[1][0] && form_hash[0][1] == form_hash[1][1];
        bad += equal ? 0 : 1;
        std::printf("TILECMP c=%u tile_ms=%.3f wide_ms=%.3f wide_over_tile=%.3f rawdots_hash=%016llx intermediates_hash=%016llx bits %s\n",
                    c, form_ms[0], form_ms[1], form_ms[1] / form_ms[0], form_hash[1][0], form_hash[1][1], equal ? "EQUAL" : "DIFFER");
        wide_ratio.emplace_back(c, form_ms[1] / form_ms[0]);
    }
    // The planner parameter of the wide form on this device (`--expert-wide-tile N`): the smallest rows-per-expert case
    // from which the wide form is faster than the 64-row tile form at that case and every larger one (none: keep it off).
    {
        u32 from = 0U;
        for (std::size_t i = wide_ratio.size(); i-- > 0;) {
            if (wide_ratio[i].second >= 1.0) break;
            from = wide_ratio[i].first;
        }
        if (from != 0U) std::printf("TILEWIDE min_rows=%u (use --expert-wide-tile %u on this device)\n", from, from);
        else std::printf("TILEWIDE min_rows=none (the wide form is not faster at the largest case: keep it off)\n");
    }
    // Least squares y = a + D T + M m (normal equations, Cramer's rule).
    double ne[3][3] = {}, b[3] = {};
    for (std::size_t i = 0; i < Y.size(); ++i) {
        const double x[3] = {1.0, T[i], M[i]};
        for (int p = 0; p < 3; ++p) { b[p] += x[p] * Y[i]; for (int q = 0; q < 3; ++q) ne[p][q] += x[p] * x[q]; }
    }
    auto det = [](const double (&m)[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };
    const double d0 = det(ne);
    double sol[3];
    for (int k = 0; k < 3; ++k) {
        double m[3][3];
        for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) m[p][q] = q == k ? b[p] : ne[p][q];
        sol[k] = det(m) / d0;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < Y.size(); ++i) worst = std::max(worst, std::abs(sol[0] + sol[1] * T[i] + sol[2] * M[i] - Y[i]) / Y[i]);
    // The 8K receipt (qmoe_pf32-STATE.md:20-28): 1,775 tiles, ~5,300 m16 blocks, 8.227 ms experts per layer block.
    const double tile8k = 1775.0 * sol[1], mb8k = 5300.0 * sol[2];
    std::printf("TILEFIT a_ms=%.3f D_us_per_tile=%.3f M_us_per_mblock=%.3f worst_rel_resid=%.3f\n", sol[0], 1000.0 * sol[1], 1000.0 * sol[2], worst);
    std::printf("TILEFIT model_8k_ms=%.3f (receipt 8.227) f8k=%.3f equal_occupancy_save128_8k_ms_per_layer=%.3f save128_32k_ms=%.1f\n",
                sol[0] + tile8k + mb8k, tile8k / (tile8k + mb8k), 700.0 * sol[1], 192.0 * 700.0 * sol[1]);
    std::printf("TILEBENCH rc=%d\n", bad ? 3 : 0);
    return bad ? 3 : 0;
}
