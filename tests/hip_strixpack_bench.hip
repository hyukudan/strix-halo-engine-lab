// No-model fixture of candidate weight block layouts (core/strixpack_layout.hpp) on the existing dense kernels.
// Kernels (the candidates are traits of the existing bodies, core/strixpack_operand.hpp):
//   sweep  the FP32 rows sweep (gemv_rows_direct.inc) with a Chunk trait; incumbents bf16 / q8_0 / q4_g64 through
//          core::detail::launch_q8_tight_rows
//   small  the small-M FP16 WMMA schedule (f16_tile_gemm_small.hpp run) with an Operand trait; incumbent q4_g64
//   a8     a probe of the integer class: int8 activations, sudot4 per 32-block, one FP32 fma per block and row
//          (the arithmetic class of the dense A8 rows kernel; a measuring instrument, not a production kernel)
//   tile   the 128-row projection tile (q8_core) with a Staging trait (tilef: the scale folded into the FP16 operand);
//          incumbents q8_0 and q4_g64
// Checks: every candidate's unpack bit for bit against the host decode (FP16 pair words and scales of Operand and
// Staging, FP32 weights of Chunk, code bytes and scales of A8Block), each kernel against a double reference, the tile
// route bit-equal to the small route.
// Timing: every launch of a pass reads a distinct copy of the matrix (>= 512 MiB per pass, dram=1), arms interleaved
// in two rounds. Lines:
//   check <what> PASS|FAIL
//   bench kind=K pad=P kernel=sweep|small|a8|tile|tilef N= K= rows= round= bytes= bpw= dram= us= GBps= Gwps=
// Usage: strix-hip-strixpack-bench [--check] [--decode N K ...] [--prefill N K ...]   (K % 128 == 0; tile N % 128 == 0)
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/resident_q8_rows.hpp"
#include "core/f16_tile_gemm_small.hpp"
#include "core/strixpack_operand.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace core = strix::core;
namespace sp = strix::core::strixpack;
namespace tg = strix::core::f16_tile_gemm;
namespace wd = strix::core::weight_decode;
namespace wei = strix::weight;
using sp::Kind;

constexpr unsigned kMaxRows = 16U;

void ok(hipError_t e, const char* what) {
    if (e == hipSuccess) return;
    std::cerr << "hip error at " << what << ": " << hipGetErrorString(e) << "\n";
    std::exit(2);
}

// ---- kernels -------------------------------------------------------------------------------------------------------

template <unsigned Rows, unsigned T, Kind F>
__global__ void __launch_bounds__(128)
sweep_kernel(const std::uint8_t* anchor0, std::size_t pitch, unsigned inputs, unsigned outputs, const float* x, float* out,
             unsigned nrows) {
    const unsigned first = (blockIdx.x * 4U + threadIdx.x / 32U) * T, lane = threadIdx.x % 32U;
    constexpr unsigned Streams = 1U;
#define GEMV_ROWS_WEIGHT_ROW(stream, o) (anchor0 + std::size_t(o) * pitch)
#define GEMV_ROWS_CHUNK sp::Chunk<F>
#define GEMV_ROWS_LOAD_CHUNK(row, start) sp::Chunk<F>(row, start)
#define GEMV_ROWS_LANE_SUM(partial) core::dense_linear::wave_sum32(partial)
#define GEMV_ROWS_INPUT_ROW(row) ((row) != 0U && (row) >= nrows ? 0U : (row))
#define GEMV_ROWS_STORE(row, o, sa, sb)                                                                               \
    if (lane == 0 && (o) < outputs && (row) < nrows) out[std::size_t(row) * outputs + (o)] = (static_cast<void>(sb), (sa))
#include "core/gemv_rows_direct.inc"
#undef GEMV_ROWS_WEIGHT_ROW
#undef GEMV_ROWS_CHUNK
#undef GEMV_ROWS_LOAD_CHUNK
#undef GEMV_ROWS_LANE_SUM
#undef GEMV_ROWS_INPUT_ROW
#undef GEMV_ROWS_STORE
}

template <Kind F>
__global__ void __launch_bounds__(128)
small_kernel(const float* a, unsigned rows, unsigned K, std::size_t pitch, tg::Q8ProjectionGroup<1> group) {
    const tg::small_m::ProjectionTile tile{a, rows, K, pitch, &group.projections[0]};
    tg::small_m::run<sp::Operand<F>, 1U>(tile, blockIdx.y, blockIdx.x * tg::kProjectionPanel);
}

template <Kind F, bool Folded>
__global__ void __launch_bounds__(tg::kThreads)
tile_kernel(const float* a, unsigned rows, unsigned K, std::size_t pitch, tg::Q8ProjectionGroup<1> group) {
    const tg::Args p{a, nullptr, K, 0U, nullptr, pitch, 0U, nullptr, 0U, 0U, rows, K, 0U, 1U};
    __shared__ tg::Storage lds;
    if constexpr (Folded) tg::q8_core<1U, sp::StagingFolded<F>>(p, lds, group);
    else tg::q8_core<1U, sp::Staging<F>>(p, lds, group);
}

// Integer class probe. acodes [kMaxRows][K] int8 (element order), adelta / asum [K / 32][kMaxRows]: the FP32 step and
// the integer sum of the 32 activation codes of a block. out[r][o] = sum over blocks b (lane b % 32 takes blocks b,
// b + 32, ... in ascending order from +0, then the wave butterfly) of fma(float(level dot), d[o][b] * delta[b][r], acc).
template <Kind F, unsigned Rows, unsigned Outs>
__global__ void __launch_bounds__(128)
a8_kernel(const std::uint8_t* anchor0, std::size_t pitch, unsigned K, unsigned outputs, const std::int8_t* acodes,
          const float* adelta, const int* asum, float* out, unsigned nrows) {
    using Block = sp::A8Block<F>;
    const unsigned lane = threadIdx.x % 32U;
    const unsigned first = (blockIdx.x * 4U + threadIdx.x / 32U) * Outs;
    if (first >= outputs) return;
    const unsigned blocks = K / 32U;
    const std::uint8_t* row[Outs];
#pragma unroll
    for (unsigned o = 0; o < Outs; ++o) row[o] = anchor0 + std::size_t(min(first + o, outputs - 1U)) * pitch;
    float acc[Outs][Rows] = {};
    constexpr unsigned RG = Rows > 8U ? 8U : Rows;
    for (unsigned b = lane; b < blocks; b += 32U) {
        Block w[Outs];
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) w[o].load(row[o], b);
#pragma unroll
        for (unsigned r0 = 0; r0 < Rows; r0 += RG) {
            int4 x[RG][2];
            float delta[RG];
            int sa[RG];
#pragma unroll
            for (unsigned r = 0; r < RG; ++r) {
                const auto* p = reinterpret_cast<const int4*>(acodes + std::size_t(r0 + r) * K + 32U * b);
                x[r][0] = p[0];
                x[r][1] = p[1];
                delta[r] = adelta[std::size_t(b) * kMaxRows + r0 + r];
                sa[r] = asum[std::size_t(b) * kMaxRows + r0 + r];
            }
#pragma unroll
            for (unsigned o = 0; o < Outs; ++o) {
#pragma unroll
                for (unsigned r = 0; r < RG; ++r) {
                    const int a[8]{x[r][0].x, x[r][0].y, x[r][0].z, x[r][0].w, x[r][1].x, x[r][1].y, x[r][1].z, x[r][1].w};
                    int dot = 0;
#pragma unroll
                    for (unsigned k = 0; k < 8U; ++k)
                        dot = STRIX_HIP_SUDOT4(true, a[k], Block::kSigned, static_cast<int>(w[o].q[k]), dot, false);
                    const int level_dot = Block::kMul * dot - Block::kOff * sa[r];
                    acc[o][r0 + r] = __builtin_fmaf(static_cast<float>(level_dot), w[o].d * delta[r], acc[o][r0 + r]);
                }
            }
            if constexpr (Rows > 8U) __builtin_amdgcn_sched_barrier(0);
        }
    }
#pragma unroll
    for (unsigned o = 0; o < Outs; ++o) {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const float s = core::dense_linear::wave_sum32(acc[o][r]);
            if (lane == 0U && first + o < outputs && r < nrows) out[std::size_t(r) * outputs + first + o] = s;
        }
    }
}

// Unpack dumps (one thread per output row).
template <Kind F>
__global__ void dump_operand_kernel(const std::uint8_t* anchor0, std::size_t pitch, unsigned K, unsigned outputs,
                                    std::uint32_t* words, std::uint32_t* staged, float* scales, float* staged_scales) {
    using Op = sp::Operand<F>;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= outputs) return;
    const std::uint8_t* row = anchor0 + std::size_t(col) * pitch;
    constexpr unsigned groups = sp::kSuper / Op::kGroup;
    for (unsigned j = 0; j < K / sp::kSuper; ++j) {
        const auto w = Op::load(row, j);
        for (unsigned gi = 0; gi < groups; ++gi) {
            for (unsigned u = 0; u < Op::kUnits; ++u) {
                const unsigned run = j * 8U + gi * Op::kUnits + u;
                std::uint32_t ws[8];
                Op::unit(w, gi, u, ws);
                std::uint32_t dst[20] = {};
                const auto raw = sp::Staging<F>::load(row, 16U * run, 0U);
                sp::Staging<F>::publish(raw, 0U, dst);
                for (unsigned i = 0; i < 8U; ++i) {
                    words[(std::size_t(col) * (K / 16U) + run) * 8U + i] = ws[i];
                    staged[(std::size_t(col) * (K / 16U) + run) * 8U + i] = dst[i];
                }
                staged_scales[std::size_t(col) * (K / 16U) + run] = sp::Staging<F>::scale(dst);
            }
            scales[std::size_t(col) * (K / Op::kGroup) + j * groups + gi] = Op::scale(w, gi);
        }
    }
}
template <Kind F>
__global__ void dump_chunk_kernel(const std::uint8_t* anchor0, std::size_t pitch, unsigned K, unsigned outputs, float* values) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= outputs) return;
    const std::uint8_t* row = anchor0 + std::size_t(col) * pitch;
    for (unsigned c = 0; c < K / 16U; ++c) {
        const sp::Chunk<F> chunk(row, 16U * c);
        for (unsigned e = 0; e < 16U; ++e) values[std::size_t(col) * K + 16U * c + e] = chunk.at(e);
    }
}
template <Kind F>
__global__ void dump_a8_kernel(const std::uint8_t* anchor0, std::size_t pitch, unsigned K, unsigned outputs,
                               std::uint32_t* bytes, float* scales) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= outputs) return;
    const std::uint8_t* row = anchor0 + std::size_t(col) * pitch;
    for (unsigned b = 0; b < K / 32U; ++b) {
        sp::A8Block<F> w;
        w.load(row, b);
        for (unsigned i = 0; i < 8U; ++i) bytes[std::size_t(col) * (K / 4U) + 8U * b + i] = w.q[i];
        scales[std::size_t(col) * (K / 32U) + b] = w.d;
    }
}

// ---- host data -----------------------------------------------------------------------------------------------------

unsigned rnd(std::mt19937& rng) { return static_cast<unsigned>(rng()); }
// Normal FP16 bits of (1 + m / 1024) 2^e and their value.
std::uint16_t half_bits(int e, unsigned m) { return static_cast<std::uint16_t>((static_cast<unsigned>(e + 15) << 10U) | m); }
double half_value(unsigned bits) {
    return std::ldexp(1.0 + double(bits & 1023U) / 1024.0, static_cast<int>((bits >> 10U) & 31U) - 15);
}
// FP16 bits of an integer |v| < 1024.
std::uint16_t half_of_int(int v) {
    if (v == 0) return 0U;
    const unsigned a = static_cast<unsigned>(v < 0 ? -v : v);
    unsigned e = 0;
    while ((a >> (e + 1U)) != 0U) ++e;
    return static_cast<std::uint16_t>((v < 0 ? 0x8000U : 0U) | ((e + 15U) << 10U) | ((a << (10U - e)) & 1023U));
}

struct Matrix {
    Kind kind{};
    unsigned n{}, k{};
    std::size_t pad{}, pitch{}, front{};
    std::vector<std::uint8_t> bytes;
    const std::uint8_t* anchor(unsigned row) const { return bytes.data() + std::size_t(row) * pitch + front; }
    std::uint8_t* anchor(unsigned row) { return bytes.data() + std::size_t(row) * pitch + front; }
    int level(unsigned row, unsigned k_) const { return sp::level_of(kind, sp::raw_code(kind, anchor(row), k_)); }
    double scale(unsigned row, unsigned k_) const {
        const unsigned bits = sp::scale_bits(kind, anchor(row), k_);
        if (sp::group_of(kind) == 64U) return half_value(bits);
        float f;
        const std::uint32_t fb = sp::byte_scale_bits(bits);
        std::memcpy(&f, &fb, 4U);
        return double(f);
    }
    float scale_f(unsigned row, unsigned k_) const { return static_cast<float>(scale(row, k_)); }
    std::size_t read_bytes() const { return std::size_t(n) * sp::read_bytes(kind, k); }
};

Matrix make_matrix(Kind kind, unsigned n, unsigned k, std::size_t pad, std::mt19937& rng) {
    Matrix m;
    m.kind = kind;
    m.n = n;
    m.k = k;
    m.pad = pad;
    m.front = sp::front_bytes(k, pad);
    m.pitch = sp::pitch(kind, k, pad);
    m.bytes.resize(std::size_t(n) * m.pitch);
    for (auto& b : m.bytes) b = static_cast<std::uint8_t>(rnd(rng) & 255U);
    for (unsigned row = 0; row < n; ++row)
        for (unsigned g = 0; g < k / sp::group_of(kind); ++g) {
            const unsigned k0 = g * sp::group_of(kind);
            if (sp::group_of(kind) == 64U) sp::set_scale_bits(kind, m.anchor(row), k0, half_bits(-7 + int(rnd(rng) % 3U), rnd(rng) & 1023U));
            else sp::set_scale_bits(kind, m.anchor(row), k0, ((24U + rnd(rng) % 3U) << 3U) | (rnd(rng) & 7U));
        }
    return m;
}

// `copies` distinct 64-byte aligned copies of a byte image, enough that a pass reads DRAM.
struct DeviceCopies {
    std::uint8_t* w = nullptr;
    std::size_t stride{}, copies{}, dram_copies{};
    bool dram() const { return copies >= dram_copies; }
};
DeviceCopies upload(const std::vector<std::uint8_t>& image, bool many) {
    DeviceCopies d;
    d.stride = (image.size() + 63U) / 64U * 64U;
    d.dram_copies = (std::size_t(512) << 20U) / d.stride + 1U;
    d.copies = many ? std::max<std::size_t>(8U, std::min<std::size_t>(d.dram_copies, 4096U)) : 1U;
    ok(hipMalloc(&d.w, d.copies * d.stride), "malloc w");
    ok(hipMemcpy(d.w, image.data(), image.size(), hipMemcpyHostToDevice), "copy w");
    for (std::size_t done = 1; done < d.copies;) {
        const std::size_t step = std::min(done, d.copies - done);
        ok(hipMemcpy(d.w + done * d.stride, d.w, step * d.stride, hipMemcpyDeviceToDevice), "replicate w");
        done += step;
    }
    return d;
}

template <class T>
T* to_device(const std::vector<T>& v) {
    T* p = nullptr;
    ok(hipMalloc(&p, v.size() * sizeof(T)), "malloc");
    ok(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "copy");
    return p;
}
template <class T>
std::vector<T> from_device(const T* p, std::size_t count) {
    std::vector<T> v(count);
    ok(hipDeviceSynchronize(), "sync");
    ok(hipMemcpy(v.data(), p, count * sizeof(T), hipMemcpyDeviceToHost), "copy back");
    return v;
}

// Mean microseconds per launch of `run(copy)` over three passes of all copies (after one warm pass).
template <class Run>
double time_us(std::size_t copies, const Run& run) {
    hipEvent_t a, b;
    ok(hipEventCreate(&a), "event");
    ok(hipEventCreate(&b), "event");
    for (std::size_t c = 0; c < copies; ++c) run(c);
    ok(hipGetLastError(), "warm");
    constexpr int passes = 3;
    ok(hipEventRecord(a, nullptr), "record");
    for (int it = 0; it < passes; ++it)
        for (std::size_t c = 0; c < copies; ++c) run(c);
    ok(hipEventRecord(b, nullptr), "record");
    ok(hipEventSynchronize(b), "sync");
    ok(hipGetLastError(), "run");
    float ms = 0.0F;
    ok(hipEventElapsedTime(&ms, a, b), "elapsed");
    ok(hipEventDestroy(a), "event");
    ok(hipEventDestroy(b), "event");
    return 1000.0 * double(ms) / (double(passes) * double(copies));
}

// ---- launchers -----------------------------------------------------------------------------------------------------

template <class Fn>
void by_rows(unsigned rows, Fn&& fn) {
    if (rows == 1U) fn(std::integral_constant<unsigned, 1U>{});
    else if (rows == 2U) fn(std::integral_constant<unsigned, 2U>{});
    else if (rows <= 4U) fn(std::integral_constant<unsigned, 4U>{});
    else if (rows <= 8U) fn(std::integral_constant<unsigned, 8U>{});
    else fn(std::integral_constant<unsigned, 16U>{});
}

template <Kind F>
void launch_sweep(const std::uint8_t* anchor0, std::size_t pitch, unsigned k, unsigned n, const float* x, float* y, unsigned rows) {
    by_rows(rows, [&](auto tag) {
        constexpr unsigned R = decltype(tag)::value, T = R <= 8U ? 2U : 4U;
        sweep_kernel<R, T, F><<<(n + 4U * T - 1U) / (4U * T), 128, 0, nullptr>>>(anchor0, pitch, k, n, x, y, rows);
    });
}
template <Kind F>
void launch_small(const std::uint8_t* anchor0, std::size_t pitch, unsigned k, unsigned n, const float* x, float* y, unsigned rows) {
    tg::Q8ProjectionGroup<1> group{};
    group.append({anchor0, n, y});
    const dim3 grid(group.first_panel[group.count], (rows + 15U) / 16U);
    STRIX_HIP_LAUNCH(Rdna3Layout, small_kernel<F><<<grid, 128, 0, nullptr>>>(x, rows, k, pitch, group));
}
template <Kind F, bool Folded = false>
void launch_tile(const std::uint8_t* anchor0, std::size_t pitch, unsigned k, unsigned n, const float* x, float* y, unsigned rows) {
    tg::Q8ProjectionGroup<1> group{};
    group.append({anchor0, n, y});
    const dim3 grid((group.first_panel[group.count] + 1U) / 2U, (rows + tg::kBM - 1U) / tg::kBM);
    STRIX_HIP_LAUNCH(Rdna3Layout, tile_kernel<F, Folded><<<grid, tg::kThreads, 0, nullptr>>>(x, rows, k, pitch, group));
}
template <Kind F>
void launch_a8(const std::uint8_t* anchor0, std::size_t pitch, unsigned k, unsigned n, const std::int8_t* codes,
               const float* delta, const int* sum, float* y, unsigned rows) {
    by_rows(rows, [&](auto tag) {
        constexpr unsigned R = decltype(tag)::value, Outs = 2U;
        a8_kernel<F, R, Outs><<<(n + 4U * Outs - 1U) / (4U * Outs), 128, 0, nullptr>>>(anchor0, pitch, k, n, codes, delta, sum, y, rows);
    });
}
template <wd::Format Format>
void launch_incumbent_sweep(const std::uint8_t* w, unsigned k, unsigned n, const float* x, float* y, unsigned rows) {
    const core::dense_linear::View view{w, k, n, wd::row_bytes(Format, k)};
    by_rows(rows, [&](auto tag) { core::detail::launch_q8_tight_rows<decltype(tag)::value, Format>(view, x, y, rows, nullptr); });
}

// ---- shared inputs -------------------------------------------------------------------------------------------------

struct Inputs {
    unsigned k{};
    std::vector<float> x;             // [kMaxRows][k]
    std::vector<std::int8_t> codes;   // [kMaxRows][k]
    std::vector<float> delta;         // [k / 32][kMaxRows]
    std::vector<int> sum;             // [k / 32][kMaxRows]
    float* dx = nullptr;
    std::int8_t* dcodes = nullptr;
    float* ddelta = nullptr;
    int* dsum = nullptr;
};
Inputs make_inputs(unsigned k, std::mt19937& rng) {
    Inputs in;
    in.k = k;
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    in.x.resize(std::size_t(kMaxRows) * k);
    for (auto& v : in.x) v = uniform(rng);
    in.codes.resize(std::size_t(kMaxRows) * k);
    for (auto& v : in.codes) v = static_cast<std::int8_t>(static_cast<int>(rnd(rng) % 255U) - 127);
    in.delta.resize(std::size_t(k / 32U) * kMaxRows);
    for (auto& v : in.delta) v = 0.004F + 0.004F * (uniform(rng) + 1.0F);
    in.sum.assign(std::size_t(k / 32U) * kMaxRows, 0);
    for (unsigned r = 0; r < kMaxRows; ++r)
        for (unsigned i = 0; i < k; ++i) in.sum[std::size_t(i / 32U) * kMaxRows + r] += in.codes[std::size_t(r) * k + i];
    in.dx = to_device(in.x);
    in.dcodes = to_device(in.codes);
    in.ddelta = to_device(in.delta);
    in.dsum = to_device(in.sum);
    return in;
}
void free_inputs(Inputs& in) {
    ok(hipFree(in.dx), "free");
    ok(hipFree(in.dcodes), "free");
    ok(hipFree(in.ddelta), "free");
    ok(hipFree(in.dsum), "free");
}

// ---- checks --------------------------------------------------------------------------------------------------------

bool g_pass = true;
void report(const std::string& what, bool pass) {
    std::cout << "check " << what << (pass ? " PASS" : " FAIL") << std::endl;
    g_pass = g_pass && pass;
}

// y[r][o] against the double reference of sum_k act(r, k) * scale * level, within `rel` of the sum of magnitudes.
template <class Act>
bool close_to_reference(const Matrix& m, const std::vector<float>& y, unsigned rows, double rel, const Act& act) {
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned o = 0; o < m.n; ++o) {
            double ref = 0.0, mag = 0.0;
            for (unsigned i = 0; i < m.k; ++i) {
                const double term = act(r, i) * m.scale(o, i) * double(m.level(o, i));
                ref += term;
                mag += std::fabs(term);
            }
            const double got = double(y[std::size_t(r) * m.n + o]);
            if (!std::isfinite(got) || std::fabs(got - ref) > rel * mag + 1e-30) return false;
        }
    return true;
}

template <Kind F>
void check_kind(unsigned n, unsigned k, std::size_t pad, const Inputs& in, std::mt19937& rng) {
    const Matrix m = make_matrix(F, n, k, pad, rng);
    const std::string tag = std::string(sp::name_of(F)) + " pad=" + std::to_string(pad) + " N=" + std::to_string(n) + " K=" + std::to_string(k);
    const DeviceCopies dev = upload(m.bytes, false);
    const std::uint8_t* anchor0 = dev.w + m.front;
    const unsigned grid = (n + 31U) / 32U;
    const auto fx = [&](unsigned r, unsigned i) { return double(in.x[std::size_t(r) * k + i]); };
    const auto fa = [&](unsigned r, unsigned i) {
        return double(in.delta[std::size_t(i / 32U) * kMaxRows + r]) * double(in.codes[std::size_t(r) * k + i]);
    };
    static_cast<void>(fx);
    constexpr unsigned many = 130U;  // tile vs small route: two row tiles, a partial one
    float* dy = nullptr;
    float* dy2 = nullptr;
    ok(hipMalloc(&dy, std::size_t(many) * n * 4U), "malloc y");
    ok(hipMalloc(&dy2, std::size_t(many) * n * 4U), "malloc y2");

    if constexpr (F != Kind::q8) {
        // Operand and Staging: FP16 pair words and scales.
        std::uint32_t *dwords = nullptr, *dstaged = nullptr;
        float *dscales = nullptr, *dstaged_scales = nullptr;
        const std::size_t words = std::size_t(n) * (k / 16U) * 8U, groups = std::size_t(n) * (k / sp::group_of(F));
        ok(hipMalloc(&dwords, words * 4U), "malloc");
        ok(hipMalloc(&dstaged, words * 4U), "malloc");
        ok(hipMalloc(&dscales, groups * 4U), "malloc");
        ok(hipMalloc(&dstaged_scales, std::size_t(n) * (k / 16U) * 4U), "malloc");
        dump_operand_kernel<F><<<grid, 32, 0, nullptr>>>(anchor0, m.pitch, k, n, dwords, dstaged, dscales, dstaged_scales);
        const auto got = from_device(dwords, words), staged = from_device(dstaged, words);
        const auto scales = from_device(dscales, groups), staged_scales = from_device(dstaged_scales, std::size_t(n) * (k / 16U));
        bool pairs = true, staging = true, scale = true;
        for (unsigned o = 0; o < n; ++o) {
            for (unsigned i = 0; i < k; i += 2U) {
                const std::uint32_t want = std::uint32_t(half_of_int(m.level(o, i))) | (std::uint32_t(half_of_int(m.level(o, i + 1U))) << 16U);
                pairs = pairs && got[std::size_t(o) * (k / 2U) + i / 2U] == want;
                staging = staging && staged[std::size_t(o) * (k / 2U) + i / 2U] == want;
            }
            for (unsigned g = 0; g < k / sp::group_of(F); ++g) {
                const float want = m.scale_f(o, g * sp::group_of(F));
                scale = scale && std::memcmp(&want, &scales[std::size_t(o) * (k / sp::group_of(F)) + g], 4U) == 0;
            }
            for (unsigned run = 0; run < k / 16U; ++run) {
                const float want = m.scale_f(o, 16U * run);
                staging = staging && std::memcmp(&want, &staged_scales[std::size_t(o) * (k / 16U) + run], 4U) == 0;
            }
        }
        report(tag + " operand pairs", pairs);
        report(tag + " operand scales", scale);
        report(tag + " staging pairs+scales", staging);
        ok(hipFree(dwords), "free");
        ok(hipFree(dstaged), "free");
        ok(hipFree(dscales), "free");
        ok(hipFree(dstaged_scales), "free");

        // The small route against the reference (FP16 activations: 2^-10 relative per term), and the tile bit-equal.
        for (const unsigned rows : {1U, 16U}) {
            ok(hipMemset(dy, 0xFF, std::size_t(many) * n * 4U), "memset");
            launch_small<F>(anchor0, m.pitch, k, n, in.dx, dy, rows);
            report(tag + " small rows=" + std::to_string(rows), close_to_reference(m, from_device(dy, std::size_t(rows) * n), rows, 2.5e-3, fx));
        }
        if (n % tg::kBN == 0U) {
            std::vector<float> xs(std::size_t(many) * k);
            for (std::size_t i = 0; i < xs.size(); ++i) xs[i] = in.x[i % in.x.size()] * (1.0F + 0.01F * float(i / in.x.size()));
            float* dxs = to_device(xs);
            launch_small<F>(anchor0, m.pitch, k, n, dxs, dy, many);
            launch_tile<F>(anchor0, m.pitch, k, n, dxs, dy2, many);
            const auto a = from_device(dy, std::size_t(many) * n), b = from_device(dy2, std::size_t(many) * n);
            report(tag + " tile == small bits rows=" + std::to_string(many), std::memcmp(a.data(), b.data(), a.size() * 4U) == 0);
            launch_tile<F, true>(anchor0, m.pitch, k, n, dxs, dy2, many);
            const auto fxs = [&](unsigned r, unsigned i) { return double(xs[std::size_t(r) * k + i]); };
            report(tag + " folded tile rows=" + std::to_string(many), close_to_reference(m, from_device(dy2, std::size_t(many) * n), many, 2.5e-3, fxs));
            ok(hipFree(dxs), "free");
        }
        // Chunk: FP32 weights, then the sweep.
        float* dvalues = nullptr;
        ok(hipMalloc(&dvalues, std::size_t(n) * k * 4U), "malloc");
        dump_chunk_kernel<F><<<grid, 32, 0, nullptr>>>(anchor0, m.pitch, k, n, dvalues);
        const auto values = from_device(dvalues, std::size_t(n) * k);
        bool chunk = true;
        for (unsigned o = 0; o < n; ++o)
            for (unsigned i = 0; i < k; ++i) {
                const float want = m.scale_f(o, i) * float(m.level(o, i));
                chunk = chunk && std::memcmp(&want, &values[std::size_t(o) * k + i], 4U) == 0;
            }
        report(tag + " chunk weights", chunk);
        ok(hipFree(dvalues), "free");
        for (const unsigned rows : {1U, 2U, 4U, 8U, 16U}) {
            ok(hipMemset(dy, 0xFF, std::size_t(many) * n * 4U), "memset");
            launch_sweep<F>(anchor0, m.pitch, k, n, in.dx, dy, rows);
            report(tag + " sweep rows=" + std::to_string(rows), close_to_reference(m, from_device(dy, std::size_t(rows) * n), rows, 1e-4, fx));
        }
    }
    // A8Block: code bytes and scales, then the probe.
    std::uint32_t* dbytes = nullptr;
    float* dscales = nullptr;
    ok(hipMalloc(&dbytes, std::size_t(n) * k), "malloc");
    ok(hipMalloc(&dscales, std::size_t(n) * (k / 32U) * 4U), "malloc");
    dump_a8_kernel<F><<<grid, 32, 0, nullptr>>>(anchor0, m.pitch, k, n, dbytes, dscales);
    const auto bytes = from_device(dbytes, std::size_t(n) * (k / 4U));
    const auto scales = from_device(dscales, std::size_t(n) * (k / 32U));
    bool a8 = true;
    for (unsigned o = 0; o < n; ++o) {
        for (unsigned i = 0; i < k; ++i) {
            const unsigned raw = sp::raw_code(F, m.anchor(o), i);
            const unsigned want = sp::A8Block<F>::kSigned ? static_cast<unsigned>(m.level(o, i)) & 255U : raw;
            a8 = a8 && ((bytes[std::size_t(o) * (k / 4U) + i / 4U] >> (8U * (i % 4U))) & 255U) == want;
        }
        for (unsigned b = 0; b < k / 32U; ++b) {
            const float want = m.scale_f(o, 32U * b);
            a8 = a8 && std::memcmp(&want, &scales[std::size_t(o) * (k / 32U) + b], 4U) == 0;
        }
    }
    report(tag + " a8 bytes+scales", a8);
    ok(hipFree(dbytes), "free");
    ok(hipFree(dscales), "free");
    for (const unsigned rows : {1U, 2U, 4U, 8U, 16U}) {
        ok(hipMemset(dy, 0xFF, std::size_t(many) * n * 4U), "memset");
        launch_a8<F>(anchor0, m.pitch, k, n, in.dcodes, in.ddelta, in.dsum, dy, rows);
        report(tag + " a8 rows=" + std::to_string(rows), close_to_reference(m, from_device(dy, std::size_t(rows) * n), rows, 1e-4, fa));
    }
    ok(hipFree(dy), "free");
    ok(hipFree(dy2), "free");
    ok(hipFree(dev.w), "free");
}

// ---- timing --------------------------------------------------------------------------------------------------------

struct Arm {
    std::string kind, kernel;
    std::size_t pad, bytes, copies;
    bool dram;
    std::function<void(std::size_t copy, unsigned rows)> run;
};

void print_bench(const Arm& arm, unsigned n, unsigned k, unsigned rows, unsigned round, double us) {
    const double weights = double(n) * double(k);
    std::cout << "bench kind=" << arm.kind << " pad=" << arm.pad << " kernel=" << arm.kernel << " N=" << n << " K=" << k
              << " rows=" << rows << " round=" << round << " bytes=" << arm.bytes << " bpw=" << 8.0 * double(arm.bytes) / weights
              << " dram=" << (arm.dram ? 1 : 0) << " us=" << us << " GBps=" << double(arm.bytes) / us * 1e-3
              << " Gwps=" << weights / us * 1e-3 << std::endl;
}

// Random valid images of the incumbent storages (finite scales and weights).
std::vector<std::uint8_t> incumbent_image(wd::Format format, unsigned n, unsigned k, std::mt19937& rng) {
    std::vector<std::uint8_t> image(std::size_t(n) * wd::row_bytes(format, k));
    for (auto& b : image) b = static_cast<std::uint8_t>(rnd(rng) & 255U);
    const auto put16 = [&](std::size_t at, std::uint16_t v) { std::memcpy(image.data() + at, &v, 2U); };
    for (unsigned row = 0; row < n; ++row) {
        const std::size_t base = std::size_t(row) * wd::row_bytes(format, k);
        if (format == wd::Format::bf16) {
            // BF16 = the high half of an FP32: sign, exponent 2^-8 .. 2^-6, 7 mantissa bits.
            for (unsigned i = 0; i < k; ++i)
                put16(base + 2U * i, static_cast<std::uint16_t>(((rnd(rng) & 1U) << 15U) | ((119U + rnd(rng) % 3U) << 7U) | (rnd(rng) & 127U)));
        } else if (format == wd::Format::q8_0) {
            for (unsigned b = 0; b < k / 32U; ++b) put16(base + 34U * b, half_bits(-7 + int(rnd(rng) % 3U), rnd(rng) & 1023U));
        } else {
            for (unsigned b = 0; b < k / 128U; ++b)
                for (unsigned g = 0; g < 2U; ++g)
                    put16(base + std::size_t(wei::Q4G64Storage::bytes) * b + wei::Q4G64Storage::scale_offset + 2U * g,
                          half_bits(-7 + int(rnd(rng) % 3U), rnd(rng) & 1023U));
        }
    }
    return image;
}

template <Kind F>
void add_pack_arms(std::vector<Arm>& arms, std::vector<DeviceCopies>& owned, unsigned n, unsigned k, std::size_t pad,
                   const Inputs& in, float* dy, std::mt19937& rng, bool decode) {
    const Matrix m = make_matrix(F, n, k, pad, rng);
    const DeviceCopies dev = upload(m.bytes, decode);
    owned.push_back(dev);
    const std::size_t pitch = m.pitch, front = m.front, bytes = m.read_bytes();
    const auto anchor = [dev, front](std::size_t c) { return dev.w + c * dev.stride + front; };
    const std::string name = sp::name_of(F);
    const float* dx = in.dx;
    if (!decode) {
        if constexpr (F != Kind::q8) {
            arms.push_back({name, "tile", pad, bytes, dev.copies, false,
                            [=](std::size_t c, unsigned rows) { launch_tile<F>(anchor(c), pitch, k, n, dx, dy, rows); }});
            arms.push_back({name, "tilef", pad, bytes, dev.copies, false,
                            [=](std::size_t c, unsigned rows) { launch_tile<F, true>(anchor(c), pitch, k, n, dx, dy, rows); }});
        }
        return;
    }
    if constexpr (F != Kind::q8) {
        if constexpr (F != Kind::p2t && F != Kind::p4b)
            arms.push_back({name, "sweep", pad, bytes, dev.copies, dev.dram(),
                            [=](std::size_t c, unsigned rows) { launch_sweep<F>(anchor(c), pitch, k, n, dx, dy, rows); }});
        arms.push_back({name, "small", pad, bytes, dev.copies, dev.dram(),
                        [=](std::size_t c, unsigned rows) { launch_small<F>(anchor(c), pitch, k, n, dx, dy, rows); }});
    }
    const std::int8_t* dcodes = in.dcodes;
    const float* ddelta = in.ddelta;
    const int* dsum = in.dsum;
    arms.push_back({name, "a8", pad, bytes, dev.copies, dev.dram(),
                    [=](std::size_t c, unsigned rows) { launch_a8<F>(anchor(c), pitch, k, n, dcodes, ddelta, dsum, dy, rows); }});
}

template <wd::Format Format>
void add_incumbent_arms(std::vector<Arm>& arms, std::vector<DeviceCopies>& owned, unsigned n, unsigned k, const Inputs& in,
                        float* dy, std::mt19937& rng, bool decode) {
    const auto image = incumbent_image(Format, n, k, rng);
    const DeviceCopies dev = upload(image, decode);
    owned.push_back(dev);
    const std::size_t bytes = image.size();
    const char* name = Format == wd::Format::bf16 ? "bf16" : Format == wd::Format::q8_0 ? "q8_0" : "q4_g64";
    const float* dx = in.dx;
    const auto w = [dev](std::size_t c) { return dev.w + c * dev.stride; };
    if (decode) {
        arms.push_back({name, "sweep", 16U, bytes, dev.copies, dev.dram(),
                        [=](std::size_t c, unsigned rows) { launch_incumbent_sweep<Format>(w(c), k, n, dx, dy, rows); }});
        if constexpr (Format == wd::Format::q4_g64)
            arms.push_back({name, "small", 16U, bytes, dev.copies, dev.dram(), [=](std::size_t c, unsigned rows) {
                                tg::Q8ProjectionGroup<1> group{};
                                group.append({w(c), n, dy});
                                tg::small_m::launch<wei::Q4G64Storage, 1U>(dx, rows, k, group, nullptr);
                            }});
    } else if constexpr (Format != wd::Format::bf16) {
        arms.push_back({name, "tile", 16U, bytes, dev.copies, false, [=](std::size_t c, unsigned rows) {
                            tg::Q8ProjectionGroup<1> group{};
                            group.append({w(c), n, dy});
                            if constexpr (Format == wd::Format::q8_0) ok(tg::gemm_q8_projections(dx, rows, k, group, nullptr), "q8 tile");
                            else ok(tg::gemm_lowbit_projections<wei::Q4G64Storage>(dx, rows, k, group, nullptr), "q4 tile");
                        }});
    }
}

void bench_shape(unsigned n, unsigned k, bool decode, std::mt19937& rng) {
    const std::vector<unsigned> row_counts = decode ? std::vector<unsigned>{1U, 2U, 4U, 8U, 16U} : std::vector<unsigned>{2048U, 8192U};
    const unsigned max_rows = row_counts.back();
    Inputs in = make_inputs(k, rng);
    if (!decode) {  // wide activations for the tile
        std::vector<float> xs(std::size_t(max_rows) * k);
        for (std::size_t i = 0; i < xs.size(); ++i) xs[i] = in.x[i % in.x.size()];
        ok(hipFree(in.dx), "free");
        in.dx = to_device(xs);
    }
    float* dy = nullptr;
    ok(hipMalloc(&dy, std::size_t(max_rows) * n * 4U), "malloc y");
    std::vector<Arm> arms;
    std::vector<DeviceCopies> owned;
    add_incumbent_arms<wd::Format::bf16>(arms, owned, n, k, in, dy, rng, decode);
    add_incumbent_arms<wd::Format::q8_0>(arms, owned, n, k, in, dy, rng, decode);
    add_incumbent_arms<wd::Format::q4_g64>(arms, owned, n, k, in, dy, rng, decode);
    add_pack_arms<Kind::q8>(arms, owned, n, k, 16U, in, dy, rng, decode);
    add_pack_arms<Kind::p4>(arms, owned, n, k, 16U, in, dy, rng, decode);
    add_pack_arms<Kind::p4>(arms, owned, n, k, 64U, in, dy, rng, decode);
    add_pack_arms<Kind::p4b>(arms, owned, n, k, 16U, in, dy, rng, decode);
    add_pack_arms<Kind::p3>(arms, owned, n, k, 16U, in, dy, rng, decode);
    add_pack_arms<Kind::p2>(arms, owned, n, k, 16U, in, dy, rng, decode);
    add_pack_arms<Kind::p2t>(arms, owned, n, k, 16U, in, dy, rng, decode);
    for (const unsigned rows : row_counts)
        for (unsigned round = 0; round < 2U; ++round)
            for (const auto& arm : arms) {
                const double us = time_us(arm.copies, [&](std::size_t c) { arm.run(c, rows); });
                print_bench(arm, n, k, rows, round, us);
            }
    for (auto& d : owned) ok(hipFree(d.w), "free");
    ok(hipFree(dy), "free");
    free_inputs(in);
}

std::vector<std::pair<unsigned, unsigned>> shapes_of(int argc, char** argv, int& i) {
    std::vector<std::pair<unsigned, unsigned>> shapes;
    while (i + 2 < argc && argv[i + 1][0] != '-') {
        const auto n = static_cast<unsigned>(std::strtoul(argv[i + 1], nullptr, 10));
        const auto k = static_cast<unsigned>(std::strtoul(argv[i + 2], nullptr, 10));
        if (n == 0U || k == 0U || k % 128U != 0U) {
            std::cerr << "shape needs N > 0 and K a positive multiple of 128\n";
            std::exit(64);
        }
        shapes.emplace_back(n, k);
        i += 2;
    }
    return shapes;
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(20261125U);
    bool check = argc == 1;
    std::vector<std::pair<unsigned, unsigned>> decode, prefill;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--check") check = true;
        else if (arg == "--decode") decode = shapes_of(argc, argv, i);
        else if (arg == "--prefill") prefill = shapes_of(argc, argv, i);
        else {
            std::cerr << "usage: strix-hip-strixpack-bench [--check] [--decode N K ...] [--prefill N K ...]\n";
            return 64;
        }
    }
    if (check) {
        // 1152 = 9 superblocks: 36 blocks (two lane steps of the a8 probe); N 136 has an output tail, N 128 runs the tile.
        for (const auto& [n, k] : {std::pair<unsigned, unsigned>{136U, 1152U}, std::pair<unsigned, unsigned>{128U, 1152U}}) {
            Inputs in = make_inputs(k, rng);
            check_kind<Kind::q8>(n, k, 16U, in, rng);
            check_kind<Kind::p4>(n, k, 16U, in, rng);
            check_kind<Kind::p4>(n, k, 64U, in, rng);
            check_kind<Kind::p4b>(n, k, 16U, in, rng);
            check_kind<Kind::p3>(n, k, 16U, in, rng);
            check_kind<Kind::p2>(n, k, 16U, in, rng);
            check_kind<Kind::p2t>(n, k, 16U, in, rng);
            free_inputs(in);
        }
        std::cout << "strixpack bench checks " << (g_pass ? "PASS" : "FAIL") << std::endl;
        if (!g_pass) return 1;
    }
    for (const auto& [n, k] : decode) bench_shape(n, k, true, rng);
    for (const auto& [n, k] : prefill) {
        if (n % tg::kBN != 0U) {
            std::cerr << "prefill shape needs N a multiple of " << tg::kBN << "\n";
            return 64;
        }
        bench_shape(n, k, false, rng);
    }
    return 0;
}
