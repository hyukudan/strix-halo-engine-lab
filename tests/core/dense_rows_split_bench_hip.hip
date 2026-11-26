// No-model bench of the decode BF16 rows GEMV (rows 1) against the declared split tree of the shared rows plan
// (src/core/rows_plan.hpp SplitTree: floor-divided contiguous chunk ranges, each partial from +0, partials merged
// from the first one by rounded FP32 adds in ascending split order).
// Arms per shape, each checked bitwise against its CPU reference before it is timed:
//   prod  core::dense_linear::enqueue_bf16_tight_rows (one output per wave, lane-strided 16-element chunks,
//         xor 16..1 butterfly): the incumbent arithmetic;
//   sS    a bench-only split body: 4 / S outputs per 128-thread block, wave (output, split) runs the prod lane order
//         over the split's chunk range, partials merge in LDS in ascending split order. s1 must equal prod bitwise
//         (same per-output order); s2 / s4 are a different FP32 order (a new identity, not EXACT to prod).
// Timing: cold weights (rotating copies, >= 256 MiB per shape), back-to-back launches, median of 5 rounds.
// GB/s = (weights + activation + output bytes) / time. The split arm is a measurement draft, not a kernel family.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/dense_linear.hpp"
#include "core/rows_plan.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

namespace {

namespace dl = strix::core::dense_linear;
using strix::core::rows::SplitTree;

constexpr unsigned kChunk = 16U;  // BF16 elements per lane chunk (weight_decode Chunk<bf16>)
constexpr unsigned kMaxSplits = 4U;
constexpr std::string_view kDraftIdentity = "bench.bf16-f32-lane-chunk16-splitk-draft";

bool ok(hipError_t status) { return status == hipSuccess; }

struct Bounds {
    unsigned begin[kMaxSplits + 1U];
};

float bf16(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t{h} << 16;
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}
float add(float a, float b) { const volatile float value = a + b; return value; }
float mul(float a, float b) { const volatile float value = a * b; return value; }

// One partial of one output over chunks [begin, end): lane l visits chunks begin + l, begin + l + 32, ...; elements in
// ascending order, acc = acc + w * x; then the xor 16..1 butterfly, read on lane 0.
float partial(const std::uint16_t* row, const float* x, unsigned begin, unsigned end) {
    float lanes[32];
    for (unsigned l = 0; l < 32U; ++l) {
        float acc = 0.0F;
        for (unsigned c = begin + l; c < end; c += 32U)
            for (unsigned e = 0; e < kChunk; ++e) acc = add(acc, mul(bf16(row[c * kChunk + e]), x[c * kChunk + e]));
        lanes[l] = acc;
    }
    for (unsigned m = 16U; m > 0U; m >>= 1U) {
        float next[32];
        for (unsigned l = 0; l < 32U; ++l) next[l] = add(lanes[l], lanes[l ^ m]);
        std::memcpy(lanes, next, sizeof lanes);
    }
    return lanes[0];
}

float reference(const SplitTree& tree, const std::uint16_t* row, const float* x) {
    float sum = 0.0F;
    for (unsigned s = 0; s < tree.splits; ++s) {
        const auto r = tree.range(s);
        const float part = partial(row, x, r.begin, r.end);
        sum = s == 0U ? part : add(sum, part);
    }
    return sum;
}

template <unsigned S>
__global__ void __launch_bounds__(128) split_kernel(const std::uint16_t* w, const float* x, float* out, unsigned inputs,
                                                    unsigned outputs, Bounds bounds) {
    __shared__ float parts[4];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, split = wave % S;
    const unsigned o = blockIdx.x * (4U / S) + wave / S;
    const std::uint16_t* row = w + std::size_t(min(o, outputs - 1U)) * inputs;
    float acc = 0.0F;
    for (unsigned c = bounds.begin[split] + lane; c < bounds.begin[split + 1U]; c += 32U) {
        const uint4 lo = *reinterpret_cast<const uint4*>(row + c * kChunk);
        const uint4 hi = *reinterpret_cast<const uint4*>(row + c * kChunk + 8U);
        const unsigned words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (unsigned e = 0; e < kChunk; ++e)
            acc = acc + __uint_as_float((words[e / 2U] >> ((e % 2U) * 16U)) << 16) * x[c * kChunk + e];
    }
#pragma unroll
    for (unsigned m = 16U; m > 0U; m >>= 1U) acc += __shfl_xor(acc, int(m), 32);
    if (lane == 0U) parts[wave] = acc;
    __syncthreads();
    if (lane == 0U && split == 0U && o < outputs) {
        float sum = parts[wave];
#pragma unroll
        for (unsigned s = 1; s < S; ++s) sum = sum + parts[wave + s];
        out[o] = sum;
    }
}

struct Shape {
    unsigned in, out;
    const char* note;
};

struct Device {
    std::uint16_t* w = nullptr;
    float *x = nullptr, *y = nullptr;
    unsigned copies = 0;
};

enum class Arm { prod, s1, s2, s4 };
constexpr unsigned splits_of(Arm a) { return a == Arm::s2 ? 2U : a == Arm::s4 ? 4U : 1U; }

bool launch(Arm arm, const Shape& s, const std::uint16_t* w, const float* x, float* y, const Bounds& b) {
    const unsigned S = splits_of(arm), blocks = (s.out + 4U / S - 1U) / (4U / S);
    switch (arm) {
    case Arm::prod: {
        const dl::View view{reinterpret_cast<const std::uint8_t*>(w), s.in, s.out, std::size_t{s.in} * 2U};
        return ok(dl::enqueue_bf16_tight_rows(view, x, y, 1U, nullptr));
    }
    case Arm::s1: split_kernel<1><<<blocks, 128>>>(w, x, y, s.in, s.out, b); break;
    case Arm::s2: split_kernel<2><<<blocks, 128>>>(w, x, y, s.in, s.out, b); break;
    case Arm::s4: split_kernel<4><<<blocks, 128>>>(w, x, y, s.in, s.out, b); break;
    }
    return ok(hipGetLastError());
}

int failures = 0;

bool run_shape(const Shape& s, std::mt19937& random, unsigned units) {
    const unsigned chunks = s.in / kChunk;
    const std::size_t wbytes = std::size_t{s.in} * s.out * 2U;
    std::vector<std::uint16_t> w(std::size_t{s.in} * s.out);
    std::vector<float> x(s.in);
    std::uniform_real_distribution<float> value(-1.0F, 1.0F);
    for (auto& v : x) v = value(random);
    for (auto& v : w) {
        const float f = value(random);
        std::uint32_t bits;
        std::memcpy(&bits, &f, sizeof bits);
        v = static_cast<std::uint16_t>(bits >> 16);
    }
    Device d;
    d.copies = static_cast<unsigned>(std::clamp<std::size_t>((std::size_t{256} << 20) / wbytes + 1U, 2U, 4096U));
    if (!ok(hipMalloc(&d.w, wbytes * d.copies)) || !ok(hipMalloc(&d.x, x.size() * 4U)) ||
        !ok(hipMalloc(&d.y, std::size_t{s.out} * 4U)) ||
        !ok(hipMemcpy(d.w, w.data(), wbytes, hipMemcpyHostToDevice)) ||
        !ok(hipMemcpy(d.x, x.data(), x.size() * 4U, hipMemcpyHostToDevice)))
        return false;
    for (unsigned c = 1; c < d.copies; ++c)
        if (!ok(hipMemcpy(reinterpret_cast<std::uint8_t*>(d.w) + wbytes * c, d.w, wbytes, hipMemcpyDeviceToDevice))) return false;

    const double bytes = double(wbytes) + 4.0 * s.in + 4.0 * s.out;
    std::printf("shape in=%u out=%u (%s) bytes=%.3f MB prod_grid=%u plan_target_blocks=%u\n", s.in, s.out, s.note,
                bytes / 1e6, (s.out + 3U) / 4U, units * 4U);
    bool all = true;
    for (const Arm arm : {Arm::prod, Arm::s1, Arm::s2, Arm::s4}) {
        const unsigned S = splits_of(arm);
        static const char* const names[] = {"prod", "s1", "s2", "s4"};
        const char* name = names[static_cast<int>(arm)];
        const SplitTree tree{kDraftIdentity, kChunk, chunks, S};
        if (!tree.valid()) { std::printf("  %-4s skipped (splits > chunks)\n", name); continue; }
        Bounds b{};
        for (unsigned i = 0; i < S; ++i) b.begin[i] = tree.range(i).begin;
        b.begin[S] = chunks;
        // Bits: copy 0, every output vs the CPU reference of the arm's declared tree.
        if (!ok(hipMemset(d.y, 0x5A, std::size_t{s.out} * 4U)) || !launch(arm, s, d.w, d.x, d.y, b) ||
            !ok(hipDeviceSynchronize()))
            return false;
        std::vector<float> got(s.out);
        if (!ok(hipMemcpy(got.data(), d.y, got.size() * 4U, hipMemcpyDeviceToHost))) return false;
        unsigned mismatches = 0;
        for (unsigned o = 0; o < s.out; ++o) {
            const float want = reference(tree, w.data() + std::size_t{o} * s.in, x.data());
            if (std::memcmp(&want, &got[o], sizeof want) != 0) ++mismatches;
        }
        // Time: rotate through cold copies.
        const unsigned launches = std::max(d.copies, 200U);
        std::vector<float> us;
        hipEvent_t start, stop;
        if (!ok(hipEventCreate(&start)) || !ok(hipEventCreate(&stop))) return false;
        for (unsigned round = 0; round < 6U; ++round) {
            if (!ok(hipEventRecord(start, nullptr))) return false;
            for (unsigned i = 0; i < launches; ++i)
                if (!launch(arm, s, d.w + std::size_t{i % d.copies} * (wbytes / 2U), d.x, d.y, b)) return false;
            float ms = 0.0F;
            if (!ok(hipEventRecord(stop, nullptr)) || !ok(hipEventSynchronize(stop)) ||
                !ok(hipEventElapsedTime(&ms, start, stop)))
                return false;
            if (round != 0U) us.push_back(1000.0F * ms / float(launches));  // round 0 warms code and TLB
        }
        static_cast<void>(hipEventDestroy(start));
        static_cast<void>(hipEventDestroy(stop));
        std::sort(us.begin(), us.end());
        const double median = us[us.size() / 2U];
        std::printf("  %-4s splits=%u grid=%u %8.2f us %7.1f GB/s bits=%s (%u/%u differ vs CPU tree)\n", name, S,
                    (s.out + 4U / S - 1U) / (4U / S) , median, bytes / (median * 1e3), mismatches ? "MISMATCH" : "EQUAL",
                    mismatches, s.out);
        all = all && mismatches == 0U;
    }
    static_cast<void>(hipFree(d.w));
    static_cast<void>(hipFree(d.x));
    static_cast<void>(hipFree(d.y));
    if (!all) ++failures;
    return true;
}

}  // namespace

int main() {
    int device = 0, units = 0;
    hipDeviceProp_t prop{};
    if (!ok(hipGetDevice(&device)) || !ok(hipGetDeviceProperties(&prop, device)) ||
        !ok(hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device)) || units <= 0) {
        std::fprintf(stderr, "dense-rows-split-bench: no HIP device\n");
        return 2;
    }
    std::printf("device %s CUs=%d (device facts read at run time)\n", prop.gcnArchName, units);
    // Decode rows-1 BF16 projection classes (in x out). Inputs of the small classes are assumed = hidden 2304.
    const Shape shapes[] = {
        {2304U, 3072U, "big, at roof in model"},  {3072U, 2304U, "wo, sub-roof in model"},
        {2304U, 1024U, "k/v class"},              {1024U, 2304U, "2304-out class, in 1024"},
        {64U, 2304U, "2304-out class, rank-64 in"}, {2304U, 384U, "small grid 96"},
        {2304U, 128U, "small grid 32"},           {2304U, 64U, "small grid 16"},
        {2304U, 32U, "small grid 8"},             {2304U, 24U, "small grid 6"},
    };
    std::mt19937 random(0xB2B2U);
    for (const auto& s : shapes)
        if (!run_shape(s, random, static_cast<unsigned>(units))) {
            std::fprintf(stderr, "dense-rows-split-bench: HIP error on in=%u out=%u\n", s.in, s.out);
            return 2;
        }
    std::printf("dense-rows-split-bench: %s\n", failures ? "FAIL (bits)" : "PASS (bits of every arm equal its CPU tree)");
    return failures ? 1 : 0;
}
