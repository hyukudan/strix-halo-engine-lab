// qpf2k/dense: the whole-T register-blocked dense FP16 projections (q8_rows::tiled, STRIX_QPF2K_DENSE=1) against
// the production fast-prefill-f16wmma-v1 launches (q8_rows::launch_dense_f16, 256-row launches), byte for byte.
//   default:                    gates (exhaustive dequant identity + every engine shape x rows x flavor, bitwise,
//                               unwritten/out-of-range outputs, overflow counts).
//   STRIX_QPF2K_DENSE_BENCH=1:  timing of the engine schedule (512-row launches for GDN in, 256 elsewhere) vs the
//                               tiled launch at T 2048/4096/8192, cold (weights evicted) and warm, plus resources,
//                               probes and a raster sweep.
//   STRIX_QPF2K_DENSE_AB=1:     form 3 vs form 4 interleaved (complete FP16-scratch linears at T 2048/4096/8192, and
//                               a K sweep with probes), wall-clock stamped.
#include "../src/hip_q8_rows_gemv.hpp"
#include "../src/hip_dense_f16_tiled.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#pragma clang fp contract(off)

using namespace strix;
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }

struct Buffer {
    void* p = nullptr; std::size_t bytes = 0;
    explicit Buffer(std::size_t size) : bytes(size) { check(hipMalloc(&p, size), "allocate"); }
    ~Buffer() { if (hipFree(p) != hipSuccess) std::_Exit(3); }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    template <class T> T* as() const { return static_cast<T*>(p); }
    template <class T> void upload(const std::vector<T>& v) const {
        require(v.size() * sizeof(T) <= bytes, "upload size"); check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
};

// ---- exhaustive dequant identity: every FP16 scale x every int8 code ----
__global__ void dequant_reference_kernel(std::uint16_t* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x;  // scale bits = i / 64, codes 4 (i % 64) .. + 3
    const unsigned scale_bits = i / 64U, c0 = 4U * (i % 64U);
    const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits)));
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        const _Float16 h = static_cast<_Float16>(scale * static_cast<float>(static_cast<signed char>(c0 + j)));
        out[4U * i + j] = __builtin_bit_cast(std::uint16_t, h);
    }
}
__global__ void dequant_tiled_kernel(std::uint16_t* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x;
    const unsigned scale_bits = i / 64U, c0 = 4U * (i % 64U);
    const unsigned codes = c0 | (c0 + 1U) << 8U | (c0 + 2U) << 16U | (c0 + 3U) << 24U;
    const _Float16 sh = __builtin_bit_cast(_Float16, static_cast<unsigned short>(scale_bits));
    const auto d = q8_rows::tiled::dequant4(codes, q8_rows::tiled::h2{sh, sh});
    reinterpret_cast<unsigned*>(out)[2U * i] = d.x;
    reinterpret_cast<unsigned*>(out)[2U * i + 1U] = d.y;
}
void dequant_gate(hipStream_t stream) {
    constexpr unsigned items = 65536U * 64U;
    Buffer a(items * 8U), b(items * 8U);
    hipLaunchKernelGGL(dequant_reference_kernel, dim3(items / 256U), dim3(256U), 0, stream, a.as<std::uint16_t>());
    hipLaunchKernelGGL(dequant_tiled_kernel, dim3(items / 256U), dim3(256U), 0, stream, b.as<std::uint16_t>());
    check(hipStreamSynchronize(stream), "dequant gate");
    std::vector<std::uint16_t> x(items * 4U), y(items * 4U);
    check(hipMemcpy(x.data(), a.p, a.bytes, hipMemcpyDeviceToHost), "download"); check(hipMemcpy(y.data(), b.p, b.bytes, hipMemcpyDeviceToHost), "download");
    std::uint64_t differ = 0, differ_nan = 0;
    for (std::size_t k = 0; k < x.size(); ++k)
        if (x[k] != y[k]) {
            const bool nan = (x[k] & 0x7c00U) == 0x7c00U && (x[k] & 0x3ffU) && (y[k] & 0x7c00U) == 0x7c00U && (y[k] & 0x3ffU);
            (nan ? differ_nan : differ) += 1U;
            if (differ + differ_nan <= 8U)
                std::cout << "dequant differs: scale 0x" << std::hex << (k / 256U) << " code " << std::dec << static_cast<int>(static_cast<signed char>(k % 256U))
                          << " reference 0x" << std::hex << x[k] << " tiled 0x" << y[k] << std::dec << '\n';
        }
    std::cout << "dequant identity: 65536 scales x 256 codes, differ " << differ << " (NaN payload only " << differ_nan << ")\n";
    require(differ == 0U && differ_nan == 0U, "tiled dequant differs from fp16(float(d) * q)");
}

// ---- inputs and weights ----
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
std::uint16_t float_to_half(float v) { return std::bit_cast<std::uint16_t>(static_cast<_Float16>(v)); }
// Flavors: 0 random (realistic scales, outlier channels); 1 edges (zero / subnormal / near-FP32-max activations,
// FP16 overflow, -128/127 codes, edge scales: +-0, subnormal, max); 2 sparse Inf/NaN activations and Inf/NaN scales.
std::vector<std::uint8_t> make_weights(std::uint32_t width, std::uint32_t outputs, unsigned flavor, std::mt19937& rng) {
    const std::size_t blocks = static_cast<std::size_t>(width / 32U) * outputs;
    std::vector<std::uint8_t> w(blocks * 34U);
    std::normal_distribution<float> code(0.0F, 40.0F);
    std::uniform_real_distribution<float> scale(0.0004F, 0.004F);
    constexpr std::array<std::uint16_t, 8> edge_scales{0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0xa000, 0x7bff, 0x1c00};
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint16_t d = float_to_half(scale(rng) * (b % 7U == 3U ? -1.0F : 1.0F));
        if (flavor == 1U && b % 5U == 0U) d = edge_scales[(b / 5U) % edge_scales.size()];
        if (flavor == 2U && b % 211U == 17U) d = (b / 211U) % 3U == 0U ? 0x7c00 : (b / 211U) % 3U == 1U ? 0xfc00 : 0x7e01;
        w[b * 34U] = static_cast<std::uint8_t>(d & 255U); w[b * 34U + 1U] = static_cast<std::uint8_t>(d >> 8U);
        for (unsigned k = 0; k < 32U; ++k) {
            int c = static_cast<int>(std::lround(code(rng)));
            c = std::clamp(c, -128, 127);
            if (flavor == 1U && b % 3U == 1U) c = k % 2U ? -128 : 127;
            w[b * 34U + 2U + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(c));
        }
    }
    return w;
}
std::vector<float> make_inputs(std::uint32_t rows, std::uint32_t width, bool bf, unsigned flavor, std::mt19937& rng) {
    std::vector<float> x(static_cast<std::size_t>(rows) * width);
    std::normal_distribution<float> g(0.0F, 1.0F);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < width; ++i) {
            float v = g(rng) * (i % 97U == 5U ? 30.0F : 1.0F);
            const std::uint32_t b = i / 32U, k = i % 32U;
            if (flavor == 1U) {
                switch ((b + r) % 9U) {
                case 0: v = 0.0F; break;
                case 1: v = k == 7U ? 0x1p-120F : 0.0F; break;
                case 2: v = std::ldexp(static_cast<float>(k) - 16.0F, -133); break;
                case 3: v = (k % 2U ? -1.0F : 1.0F) * 3.0e38F; break;
                case 4: v = (static_cast<float>(k) - 15.5F) * 4100.0F; break;  // around the FP16 max (65504 / 65520)
                case 5: v = k == 0U ? -0.0F : (k == 1U ? 65519.0F : 1.0F); break;
                case 6: v = k % 2U ? -0.0F : 0.0F; break;
                case 7:  // FP16 overflow in the first half only (group 0), second half only (group 1), both halves (group 2)
                    if (k == 3U || k == 12U || k == 16U || k == 21U) v = (r % 2U ? -1.0F : 1.0F) * 70000.0F;
                    break;
                case 8: {  // FP16 range / subnormal / tie boundaries, signed signaling and quiet NaNs
                    constexpr std::array<std::uint32_t, 16> special{0x477fe000U /*65504*/, 0x477fe100U /*65505*/, 0x477fef00U /*65519*/,
                        0x477ff000U /*65520*/, 0x47800000U /*65536*/, 0xc77fef00U, 0x38800000U /*2^-14*/, 0x33800000U /*2^-24*/,
                        0x33000000U /*2^-25 tie*/, 0x33c00000U /*1.5 x 2^-24 tie*/, 0x3f801000U /*1 + 2^-11 tie*/, 0x3f803000U,
                        0x7f810000U /*sNaN*/, 0xff810000U, 0x7fc10000U /*qNaN*/, 0xffc30000U};
                    if (k < special.size()) v = std::bit_cast<float>(special[k] | (bf ? 0U : (k >= 12U ? 0x1234U : 0U)));
                    break;
                }
                default: break;
                }
            }
            if (flavor == 2U && r % 5U == 2U && i == (r * 37U) % width)
                v = std::bit_cast<float>(i % 3U == 0U ? 0x7f800000U : i % 3U == 1U ? 0xff800000U : 0x7fc01234U + r);
            if (bf) v = std::bit_cast<float>(std::isnan(v) ? std::bit_cast<std::uint32_t>(v) & 0xffff0000U  // NaN bits kept (BF16 truncation)
                                                    : static_cast<std::uint32_t>(bf16(v)) << 16U);
            x[static_cast<std::size_t>(r) * width + i] = v;
        }
    return x;
}

// Rows at input0 + r * input_stride, FP32 outputs [r][outputs] (the production sites store the same accumulator
// through their own rounding; the raw FP32 value is the strictest comparison).
template <class In> struct TiledSite {
    using Input = In;
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output;
    __device__ void store(std::uint32_t out, std::uint32_t r, float value) const { output[static_cast<std::uint64_t>(r) * outputs + out] = value; }
};
template <class In, unsigned Rows> struct RefSite {
    using Input = In;
    static constexpr unsigned rows = Rows;
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output;
    __device__ const In* row_input(unsigned r) const { return reinterpret_cast<const In*>(input0 + r * input_stride); }
    __device__ static float coefficient(unsigned scale, std::int8_t code) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(scale))) * static_cast<float>(code);
    }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) { return lower + upper; }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[static_cast<std::uint64_t>(r) * outputs + out] = value; }
};

struct Shape { const char* name; std::uint32_t width; std::vector<std::uint32_t> segs; bool bf16; unsigned engine_rows; };
const std::vector<Shape>& shapes() {
    static const std::vector<Shape> s{
        {"GDN in (qkv|z)", 2560U, {10240U, 6144U}, true, 512U},
        {"GDN out", 6144U, {2560U}, false, 256U},
        {"QSA q|k|v", 2560U, {12288U, 512U, 512U}, true, 256U},
        {"QSA out", 6144U, {2560U}, false, 64U},
        {"shared gate|up", 2560U, {640U, 640U}, true, 256U},
        {"shared down", 640U, {2560U}, false, 256U},
        {"tail bf16 2560->1000", 2560U, {1000U}, true, 256U},
        {"tail f32 640->1000", 640U, {1000U}, false, 256U},
    };
    return s;
}
std::uint32_t outputs_of(const Shape& s) { std::uint32_t n = 0; for (auto v : s.segs) n += v; return n; }

struct Problem {
    std::uint32_t rows, rows_up, width, outputs;
    bool bf;
    std::unique_ptr<Buffer> weights, input, tight, out;
    std::vector<std::uint32_t> segs;
    unsigned expected_overflow = 0;  // groups of 8 with a finite value beyond the FP16 range, real rows (host count)
    q8_rows::Segments segments() const {
        q8_rows::Segments s{};
        const auto* base = weights->as<unsigned char>();
        std::uint64_t off = 0;
        for (auto n : segs) {
            s.weights[s.count] = base + off;
            s.first[s.count + 1U] = s.first[s.count] + n;
            ++s.count;
            off += static_cast<std::uint64_t>(n) * (width / 32U) * 34U;
        }
        return s;
    }
    std::uint64_t stride() const { return static_cast<std::uint64_t>(width) * (bf ? 2U : 4U) + (bf ? 128U : 256U); }  // padded rows
};
Problem make_problem(const Shape& shape, std::uint32_t rows, unsigned flavor, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Problem p{rows, (rows + 255U) / 256U * 256U, shape.width, outputs_of(shape), shape.bf16, {}, {}, {}, {}, shape.segs, 0U};
    const auto w = make_weights(p.width, p.outputs, flavor, rng);
    p.weights = std::make_unique<Buffer>(w.size()); p.weights->upload(w);
    // Inputs for rows_up rows (rows past `rows` are zeros: the reference runs whole 256-row launches); the tiled form
    // reads a tight copy of exactly `rows` rows (a missing clamp reads past the allocation there).
    const auto x = make_inputs(rows, p.width, p.bf, flavor, rng);
    const std::uint64_t stride = p.stride();
    std::vector<unsigned char> host(stride * p.rows_up, 0);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < p.width; ++i) {
            const float v = x[static_cast<std::size_t>(r) * p.width + i];
            if (p.bf) { const auto h = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16U); std::memcpy(&host[r * stride + 2U * i], &h, 2U); }
            else std::memcpy(&host[r * stride + 4U * i], &v, 4U);
        }
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t g = 0; g < p.width / 8U; ++g) {
            bool bad = false;
            for (unsigned k = 0; k < 8U; ++k) {
                const float v = x[static_cast<std::size_t>(r) * p.width + 8U * g + k];
                bad |= std::isfinite(v) && !(std::fabs(v) <= 65504.0F);
            }
            p.expected_overflow += bad ? 1U : 0U;
        }
    p.input = std::make_unique<Buffer>(host.size()); p.input->upload(host);
    host.resize(stride * rows);
    p.tight = std::make_unique<Buffer>(host.size()); p.tight->upload(host);
    p.out = std::make_unique<Buffer>(static_cast<std::size_t>(p.rows_up) * p.outputs * 4U);
    return p;
}
template <class In> TiledSite<In> tiled_site(const Problem& p, float* out) {
    return TiledSite<In>{p.tight->as<unsigned char>(), p.stride(), p.segments(), p.width, p.outputs, out};
}
// The engine schedule: launch_dense_f16 over EngineRows-row launches (convert + projection each).
template <class In, unsigned EngineRows> void reference(const Problem& p, float* out, hipStream_t stream) {
    for (std::uint32_t r0 = 0; r0 < p.rows_up; r0 += EngineRows) {
        RefSite<In, EngineRows> site{p.input->as<unsigned char>() + r0 * p.stride(), p.stride(), p.segments(), p.width, p.outputs,
            out + static_cast<std::uint64_t>(r0) * p.outputs};
        q8_rows::launch_dense_f16(site, stream, false);
    }
}
unsigned overflow_now() { return q8_rows::dense_f16_overflow_count(); }

std::vector<std::uint32_t> download_u32(const Buffer& b, std::size_t n) {
    std::vector<std::uint32_t> v(n); check(hipMemcpy(v.data(), b.p, n * 4U, hipMemcpyDeviceToHost), "download"); return v;
}
// rows x outputs of `got` == `ref`, and nothing written at rows >= rows (poisoned 0xa5).
void compare(const std::vector<std::uint32_t>& ref, const std::vector<std::uint32_t>& got, std::uint32_t rows, std::uint32_t outputs, const std::string& what) {
    std::uint64_t differ = 0, first = UINT64_MAX, beyond = 0;
    const std::size_t n = static_cast<std::size_t>(rows) * outputs;
    for (std::size_t i = 0; i < n; ++i) if (ref[i] != got[i]) { ++differ; if (first == UINT64_MAX) first = i; }
    for (std::size_t i = n; i < got.size(); ++i) beyond += got[i] != 0xa5a5a5a5U;
    if (!differ && !beyond) return;
    std::cout << "FAIL " << what << ": differ " << differ << " beyond-rows writes " << beyond;
    if (first != UINT64_MAX) std::cout << " first row " << first / outputs << " out " << first % outputs << " ref 0x" << std::hex << ref[first] << " got 0x" << got[first] << std::dec;
    std::cout << '\n';
    throw std::runtime_error("tiled dense FP16 differs from the production launches: " + what);
}
template <class In> void gate_case(const Shape& shape, std::uint32_t rows, unsigned flavor, hipStream_t stream, unsigned& cases) {
    auto p = make_problem(shape, rows, flavor, 1000U + rows * 7U + flavor * 13U + shape.width);
    const std::size_t n = static_cast<std::size_t>(p.rows_up) * p.outputs;
    Buffer ref(n * 4U);
    check(hipMemsetAsync(ref.p, 0xa5, ref.bytes, stream), "poison");
    const unsigned o0 = overflow_now();
    reference<In, 256U>(p, ref.as<float>(), stream);
    const unsigned o1 = overflow_now();
    const auto a = download_u32(ref, n);
    require(o1 - o0 == p.expected_overflow, std::string(shape.name) + ": reference overflow count " + std::to_string(o1 - o0) + " != host " +
        std::to_string(p.expected_overflow));
    const auto site = tiled_site<In>(p, p.out->as<float>());
    const unsigned inline_mode = sizeof(In) == 4U ? q8_rows::tiled::kF32 : q8_rows::tiled::kBf16;
    // Both A modes (separate convert, inline) and rasters 1 / 3 / 64 besides the default on the larger cases.
    // Every kernel form x A mode; rasters 1 / 3 / 64 besides the defaults on the larger cases.
    std::vector<std::array<unsigned, 3>> runs;
    for (const unsigned form : {0U, 1U, 2U, 3U, 4U}) {
        runs.push_back({q8_rows::tiled::kPre, 0U, form}); runs.push_back({inline_mode, 0U, form});
        if (rows == 1000U || rows == 2048U)
            for (const unsigned r : {1U, 3U, 64U}) { runs.push_back({inline_mode, r, form}); runs.push_back({q8_rows::tiled::kPre, r, form}); }
    }
    unsigned before = o1;
    for (const auto& [mode, raster, form] : runs) {
        require(q8_rows::tiled::supported(site, rows, mode), "tiled form unsupported for a gate shape");
        check(hipMemsetAsync(p.out->p, 0xa5, p.out->bytes, stream), "poison");
        q8_rows::tiled::launch(site, rows, stream, false, mode, raster, form);
        check(hipGetLastError(), "tiled launch");
        const unsigned after = overflow_now();
        const std::string what = std::string(shape.name) + " rows " + std::to_string(rows) + " flavor " + std::to_string(flavor) + " mode " +
            std::to_string(mode) + " raster " + std::to_string(raster) + " form " + std::to_string(form);
        compare(a, download_u32(*p.out, n), rows, p.outputs, what);
        require(after - before == p.expected_overflow, what + ": overflow count " + std::to_string(after - before) + " != " + std::to_string(p.expected_overflow));
        before = after;
    }
    ++cases;
    std::cout << "gate " << shape.name << " rows " << rows << " flavor " << flavor << ": " << static_cast<std::size_t>(rows) * p.outputs
              << " outputs bitwise equal in " << runs.size() << " mode/raster runs, overflow groups " << p.expected_overflow << '\n';
}
// Isolated activation conversion: identity Q8_0 weights (scale 1.0, code 1 at k == n) make output n of row r the FP16
// operand A[r][n] itself (one exact product, zeros elsewhere: -0 inputs give +0), in every A mode. Inputs: every BF16
// value that converts to a finite FP16 (exhaustive), and FP32 values around every FP16 exponent incl. subnormals, RNE ties
// (+-1 ulp) and the 65504 / 65520 edge; compared bitwise with the host RNE conversion.
template <class In> void conversion_gate_mode(const std::vector<float>& values, hipStream_t stream) {
    constexpr std::uint32_t width = 640U;
    const auto rows = static_cast<std::uint32_t>((values.size() + width - 1U) / width);
    std::vector<std::uint8_t> w(static_cast<std::size_t>(width) * (width / 32U) * 34U, 0);
    for (std::uint32_t n = 0; n < width; ++n)
        for (std::uint32_t b = 0; b < width / 32U; ++b) {
            auto* block = &w[(static_cast<std::size_t>(n) * (width / 32U) + b) * 34U];
            block[0] = 0x00; block[1] = 0x3c;  // 1.0
            if (b == n / 32U) block[2U + n % 32U] = 1U;
        }
    Buffer weights(w.size()); weights.upload(w);
    const std::uint64_t stride = width * sizeof(In);
    std::vector<In> host(static_cast<std::size_t>(rows) * width);
    for (std::size_t i = 0; i < host.size(); ++i) {
        const float v = i < values.size() ? values[i] : 0.0F;
        if constexpr (sizeof(In) == 2U) host[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16U); else host[i] = v;
    }
    Buffer input(host.size() * sizeof(In)); input.upload(host);
    Buffer out(static_cast<std::size_t>(rows) * width * 4U);
    q8_rows::Segments seg{}; seg.weights[0] = weights.as<unsigned char>(); seg.first[1] = width; seg.count = 1U;
    const TiledSite<In> site{input.as<unsigned char>(), stride, seg, width, width, out.as<float>()};
    std::vector<std::uint32_t> pre;
    for (const unsigned run : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U}) {
        const unsigned mode = run % 2U ? (sizeof(In) == 4U ? q8_rows::tiled::kF32 : q8_rows::tiled::kBf16) : q8_rows::tiled::kPre, form = run / 2U;
        check(hipMemsetAsync(out.p, 0xa5, out.bytes, stream), "poison");
        q8_rows::tiled::launch(site, rows, stream, false, mode, 0U, form);
        check(hipStreamSynchronize(stream), "conversion gate");
        const auto got = download_u32(out, static_cast<std::size_t>(rows) * width);
        std::uint64_t differ = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const float x = sizeof(In) == 2U ? std::bit_cast<float>(std::bit_cast<std::uint32_t>(values[i]) & 0xffff0000U) : values[i];
            float expect = static_cast<float>(static_cast<_Float16>(x));
            if (expect == 0.0F) expect = 0.0F;  // +0 (the accumulation starts at +0)
            // FP16 subnormal operands: the WMMA's own FP32 result is the reference (production's), not the host value.
            const bool subnormal = std::fabs(expect) < 0x1p-14F && expect != 0.0F;
            if (subnormal) continue;
            if (std::bit_cast<std::uint32_t>(expect) != got[i] && ++differ <= 8U)
                std::cout << "conversion vs host RNE: mode " << mode << " x 0x" << std::hex << std::bit_cast<std::uint32_t>(x) << " expect 0x"
                          << std::bit_cast<std::uint32_t>(expect) << " got 0x" << got[i] << std::dec << '\n';
        }
        std::cout << "conversion gate " << (sizeof(In) == 2U ? "BF16" : "FP32") << " mode " << mode << ": " << values.size()
                  << " values, differ from host RNE " << differ << '\n';
        // Binding: the inline conversion == the convert-kernel path (production's operand) bitwise.
        if (run == 0U) {
            pre = got;
            // The convert kernel's FP16 operands themselves (the scratch slot of this stream) == host RNE, bitwise.
            const std::uint32_t pitch = q8_rows::tiled::scratch_pitch(width);
            const _Float16* a16 = q8_rows::tiled::scratch(stream, static_cast<std::uint64_t>((rows + 255U) / 256U * 256U) * pitch);
            std::vector<std::uint16_t> h(static_cast<std::size_t>(rows) * pitch);
            check(hipMemcpy(h.data(), a16, h.size() * 2U, hipMemcpyDeviceToHost), "scratch download");
            std::uint64_t bad = 0;
            for (std::size_t i = 0; i < values.size(); ++i) {
                const float x = sizeof(In) == 2U ? std::bit_cast<float>(std::bit_cast<std::uint32_t>(values[i]) & 0xffff0000U) : values[i];
                const auto expect = std::bit_cast<std::uint16_t>(static_cast<_Float16>(x));
                if (h[(i / width) * pitch + i % width] != expect && ++bad <= 4U)
                    std::cout << "FP16 operand differs: x 0x" << std::hex << std::bit_cast<std::uint32_t>(x) << " expect 0x" << expect << " got 0x"
                              << h[(i / width) * pitch + i % width] << std::dec << '\n';
            }
            std::cout << "conversion gate operands (convert kernel) vs host RNE: " << values.size() << " values, differ " << bad << '\n';
            require(bad == 0U, "convert kernel FP16 operands differ from host RNE");
        }
        else require(got == pre, "inline activation conversion differs from the convert kernel");
        // Informational only: the WMMA output of a single exact product can differ from the host value in its last FP32
        // bit (t2: x = -2^-14 -> 0xb8800001 already on the production convert path), so the host value is no oracle.
    }
}
void conversion_gate(hipStream_t stream) {
    std::vector<float> bf;  // every BF16 pattern that converts to a finite FP16 (|x| <= 65280 = 0x477f)
    for (std::uint32_t h = 0; h < 65536U; ++h)
        if ((h & 0x7fffU) <= 0x477fU) bf.push_back(std::bit_cast<float>(h << 16U));
    conversion_gate_mode<std::uint16_t>(bf, stream);
    std::vector<float> f;
    std::mt19937 rng(5U);
    for (std::uint32_t e = 100U; e <= 141U; ++e)          // 2^-27 .. 2^15: below the FP16 subnormals up to the binade under the top
        for (const std::uint32_t sign : {0U, 0x80000000U}) {
            for (std::uint32_t m = 0; m < 8192U; m += 1U) {   // mantissa tops (RNE ties at bit 12 / subnormal ties) +-1 ulp
                const std::uint32_t base = sign | e << 23U | m << 10U;
                for (const std::uint32_t d : {0U, 1U, 0x1ffU, 0x200U, 0x201U, 0x3ffU}) f.push_back(std::bit_cast<float>(base + d));
            }
            for (unsigned k = 0; k < 2048U; ++k) f.push_back(std::bit_cast<float>(sign | e << 23U | (static_cast<std::uint32_t>(rng()) & 0x7fffffU)));
        }
    for (std::uint32_t m = 0; m < (1U << 23U); m += 97U) {  // the top binade up to 65519.996 (finite FP16 after RNE)
        const float v = std::bit_cast<float>(142U << 23U | m);
        if (v < 65520.0F) { f.push_back(v); f.push_back(-v); }
    }
    conversion_gate_mode<float>(f, stream);
}
// Back-to-back tiled calls without host synchronization (one stream: the convert scratch reused in order, grown between
// calls), then the same calls alternating over two streams at once (per-stream scratch).
void queue_gate(hipStream_t stream) {
    const auto& all = shapes();
    struct Item { const Shape* shape; std::uint32_t rows; Problem p; std::vector<std::uint32_t> ref; };
    std::vector<Item> items;
    const std::vector<std::pair<unsigned, std::uint32_t>> list{{0U, 1000U}, {2U, 2048U}, {4U, 768U}, {0U, 256U}, {6U, 4096U}, {2U, 512U}};
    for (const auto& [index, rows] : list) {
        Item it{&all[index], rows, make_problem(all[index], rows, rows % 3U, 77U + rows), {}};
        const std::size_t n = static_cast<std::size_t>(it.p.rows_up) * it.p.outputs;
        Buffer ref(n * 4U);
        check(hipMemsetAsync(ref.p, 0xa5, ref.bytes, stream), "poison");
        reference<std::uint16_t, 256U>(it.p, ref.as<float>(), stream);
        check(hipStreamSynchronize(stream), "queue reference");
        it.ref = download_u32(ref, n);
        items.push_back(std::move(it));
    }
    // Fresh streams: their scratch slots start empty, so pass 0 grows its slot between queued calls (1000 -> 2048 -> 4096 rows).
    hipStream_t first = nullptr, second = nullptr;
    check(hipStreamCreateWithFlags(&first, hipStreamNonBlocking), "first stream");
    check(hipStreamCreateWithFlags(&second, hipStreamNonBlocking), "second stream");
    for (unsigned pass = 0; pass < 5U; ++pass) {  // pass p: form p (pass 0 one stream, passes 1-4 two streams)
        for (auto& it : items) check(hipMemsetAsync(it.p.out->p, 0xa5, it.p.out->bytes, stream), "poison");
        check(hipStreamSynchronize(stream), "poison");
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto& it = items[i];
            const hipStream_t s = pass && i % 2U ? second : first;
            q8_rows::tiled::launch(tiled_site<std::uint16_t>(it.p, it.p.out->as<float>()), it.rows, s, false, q8_rows::tiled::kPre, 0U, pass);
        }
        check(hipStreamSynchronize(first), "queue"); check(hipStreamSynchronize(second), "queue");
        for (const auto& it : items)
            compare(it.ref, download_u32(*it.p.out, it.ref.size()), it.rows, it.p.outputs,
                std::string("queued ") + it.shape->name + " rows " + std::to_string(it.rows) + " pass " + std::to_string(pass));
    }
    check(hipStreamDestroy(first), "first stream"); check(hipStreamDestroy(second), "second stream");
    std::cout << "queued / two-stream tiled calls: PASS\n";
}
void gates(hipStream_t stream) {
    dequant_gate(stream);
    unsigned cases = 0;
    for (const auto& shape : shapes()) {
        const auto run = [&]<class In>() {
            for (const std::uint32_t rows : {17U, 256U, 1000U, 2048U})
                for (unsigned flavor = 0; flavor < 3U; ++flavor) gate_case<In>(shape, rows, flavor, stream, cases);
            for (unsigned flavor = 0; flavor < 3U; ++flavor) gate_case<In>(shape, 8192U, flavor, stream, cases);
        };
        if (shape.bf16) run.template operator()<std::uint16_t>(); else run.template operator()<float>();
    }
    queue_gate(stream);
    conversion_gate(stream);
    std::cout << "tiled dense FP16 == production launch_dense_f16 bitwise: " << cases << " cases PASS\n";
}

// ---- timing ----
float time_us(const std::function<void()>& enqueue, hipStream_t stream, Buffer* flush, unsigned reps) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    std::vector<float> samples;
    for (unsigned it = 0; it < reps + 2U; ++it) {
        if (flush) check(hipMemsetAsync(flush->p, static_cast<int>(it), flush->bytes, stream), "flush");
        check(hipEventRecord(start, stream), "record"); enqueue(); check(hipEventRecord(stop, stream), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        if (it >= 2U) samples.push_back(ms * 1000.0F);
    }
    check(hipEventDestroy(start), "event"); check(hipEventDestroy(stop), "event");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}
template <class Kernel> void resources(const char* name, Kernel kernel, int threads = 256) {
    hipFuncAttributes attr{}; int blocks = 0;
    check(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)), "attributes");
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0), "occupancy");
    std::cout << "resources " << name << ": vgpr " << attr.numRegs << " lds " << attr.sharedSizeBytes << " scratch " << attr.localSizeBytes
              << " blocks/CU " << blocks << '\n';
}
constexpr double kPeak = 47.6e12;
// Raw-FP32-store fixture sites (not the engine epilogues): "linear" = convert + projection as launched (the complete
// dense linear of the fixture), "projection" = the projection kernel alone.
template <class In, unsigned EngineRows> void bench_shape(const Shape& shape, std::uint32_t rows, hipStream_t stream, Buffer& flush, bool extra) {
    auto p = make_problem(shape, rows, 0U, 7U);
    const double flop = 2.0 * rows * p.width * p.outputs;
    const auto site = tiled_site<In>(p, p.out->as<float>());
    const auto pct = [&](float us) { return 100.0 * flop / (us * 1e-6) / kPeak; };
    const unsigned reps = 15U;  // median of 15
    constexpr unsigned Inline = sizeof(In) == 4U ? q8_rows::tiled::kF32 : q8_rows::tiled::kBf16;
    const float base_cold = time_us([&] { reference<In, EngineRows>(p, p.out->as<float>(), stream); }, stream, &flush, reps);
    const float base_warm = time_us([&] { reference<In, EngineRows>(p, p.out->as<float>(), stream); }, stream, nullptr, reps);
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    std::cout << "bench " << shape.name << " " << p.width << "->" << p.outputs << " T=" << rows << ": engine linear (" << EngineRows
              << "-row launch_dense_f16) cold " << base_cold << " us (" << pct(base_cold) << " %) warm " << base_warm << " us (" << pct(base_warm) << " %)";
    for (const unsigned form : {0U, 3U, 4U, 1U, 2U})
        for (const unsigned mode : {q8_rows::tiled::kPre, Inline}) {
            const float c = time_us([&] { q8_rows::tiled::launch(site, rows, stream, false, mode, 0U, form); }, stream, &flush, reps);
            const float w = time_us([&] { q8_rows::tiled::launch(site, rows, stream, false, mode, 0U, form); }, stream, nullptr, reps);
            std::cout << "\n    tiled linear form " << form << " " << (mode == q8_rows::tiled::kPre ? "convert+proj" : "inline") << " cold " << c << " us ("
                      << pct(c) << " %) warm " << w << " us (" << pct(w) << " %) cold B/A " << c / base_cold;
        }
    std::cout << " | default mode " << q8_rows::tiled::mode_of(site) << '\n';
    if (!extra) return;
    _Float16* a16 = q8_rows::tiled::scratch(stream, static_cast<std::uint64_t>(p.rows_up) * q8_rows::tiled::scratch_pitch(p.width));
    const auto items = static_cast<std::uint64_t>(rows) * (p.width / 8U);
    const auto convert = [&] {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::tiled::convert_kernel<TiledSite<In>>), dim3(static_cast<unsigned>((items + 255U) / 256U)), dim3(256U), 0,
            stream, site, rows, a16, q8_rows::dense_f16_overflow_counter());
    };
    convert();
    const float conv = time_us(convert, stream, &flush, reps);
    std::cout << "  convert alone cold " << conv << " us\n";
    unsigned pf = 0U;  // projection form of the sweep below
    const auto project = [&]<unsigned Mode, unsigned Probe>(unsigned raster) {
        q8_rows::tiled::launch_projection<TiledSite<In>, Mode, Probe>(site, rows, stream, false, pf, raster, Mode == q8_rows::tiled::kPre ? a16 : nullptr);
    };
    resources((std::string(shape.name) + " pre").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, q8_rows::tiled::kPre, 0U>);
    resources((std::string(shape.name) + " inline").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, Inline, 0U>);
    resources((std::string(shape.name) + " probe1").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, q8_rows::tiled::kPre, 1U>);
    resources((std::string(shape.name) + " probe2").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, q8_rows::tiled::kPre, 2U>);
    resources((std::string(shape.name) + " probe3").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, q8_rows::tiled::kPre, 3U>);
    resources((std::string(shape.name) + " form1 pre").c_str(), q8_rows::tiled::project2_kernel<TiledSite<In>, q8_rows::tiled::kPre, 512U, 32U, 2U, 0U>, 512);
    resources((std::string(shape.name) + " form1 inline").c_str(), q8_rows::tiled::project2_kernel<TiledSite<In>, Inline, 512U, 32U, 2U, 0U>, 512);
    resources((std::string(shape.name) + " form2 pre").c_str(), q8_rows::tiled::project2_kernel<TiledSite<In>, q8_rows::tiled::kPre, 512U, 32U, 1U, 0U>, 512);
    resources((std::string(shape.name) + " form3 pre").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, q8_rows::tiled::kPre, 0U, 256U>, 512);
    resources((std::string(shape.name) + " form3 inline").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, Inline, 0U, 256U>, 512);
    resources((std::string(shape.name) + " form4 pre").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, q8_rows::tiled::kPre, 0U, 256U, true>, 512);
    if constexpr (Inline != q8_rows::tiled::kF32)
        resources((std::string(shape.name) + " form4 inline").c_str(), q8_rows::tiled::project_kernel<TiledSite<In>, Inline, 0U, 256U, true>, 512);
    for (const unsigned pff : {0U, 3U, 4U})
    if ((pf = pff), true)
    for (const unsigned raster : {1U, 2U, 4U, 8U}) {
        const float c = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 0U>(raster); }, stream, &flush, reps);
        const float w = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 0U>(raster); }, stream, nullptr, reps);
        const float ci = time_us([&] { project.template operator()<Inline, 0U>(raster); }, stream, &flush, reps);
        std::cout << "  projection form " << pf << " raster " << raster << ": FP16-A cold " << c << " us (" << pct(c) << " %) warm " << w << " us (" << pct(w)
                  << " %) | inline-A cold " << ci << " us (" << pct(ci) << " %)\n";
    }
    const unsigned r = q8_rows::tiled::raster(q8_rows::tiled::kPre);
    for (const unsigned pff : {0U, 3U, 4U}) {
    pf = pff;
    const float full = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 0U>(r); }, stream, nullptr, reps);
    const float p1 = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 1U>(r); }, stream, nullptr, reps);
    const float p2 = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 2U>(r); }, stream, nullptr, reps);
    const float p3 = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 3U>(r); }, stream, nullptr, reps);
    const float p4 = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 4U>(r); }, stream, nullptr, reps);
    const float p5 = time_us([&] { project.template operator()<q8_rows::tiled::kPre, 5U>(r); }, stream, nullptr, reps);
    std::cout << "  probes form " << pf << " warm (FP16-A, not additive): full " << full << " us (" << pct(full) << " %), no-dequant " << p1 << " us ("
              << pct(p1) << " %), no-WMMA " << p2 << " us, WMMA+fragments only " << p3 << " us (" << pct(p3) << " %), no-epilogue " << p4
              << " us (" << pct(p4) << " %), WMMA+fragments no-epilogue " << p5 << " us (" << pct(p5) << " %)\n";
    }
}
// ---- STRIX_QPF2K_DENSE_AB=1: form 3 vs form 4 interleaved (rev 6) ----
double wall() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
struct Stat { float med, p25, p75; };
Stat stat(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2U], v[v.size() / 4U], v[(3U * v.size()) / 4U]};
}
// Repetition i runs A then B (even i) or B then A (odd i): AB / BA pairs, balanced over an even repetition count
// (ABBA across two repetitions); 2 warm-up repetitions; cold = flush before every timed call. Returns the A and B samples
// and the paired ratios B / A.
std::array<std::vector<float>, 3> time_ab(const std::function<void()>& a, const std::function<void()>& b, hipStream_t stream, Buffer* flush, unsigned reps) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    const auto one = [&](const std::function<void()>& f, unsigned it) {
        if (flush) check(hipMemsetAsync(flush->p, static_cast<int>(it), flush->bytes, stream), "flush");
        check(hipEventRecord(start, stream), "record"); f(); check(hipEventRecord(stop, stream), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        return ms * 1000.0F;
    };
    std::array<std::vector<float>, 3> r;
    for (unsigned it = 0; it < reps + 2U; ++it) {
        float ta, tb;
        if (it % 2U == 0U) { ta = one(a, it); tb = one(b, it); } else { tb = one(b, it); ta = one(a, it); }
        if (it >= 2U) { r[0].push_back(ta); r[1].push_back(tb); r[2].push_back(tb / ta); }
    }
    check(hipEventDestroy(start), "event"); check(hipEventDestroy(stop), "event");
    return r;
}
void print_ab(const std::string& what, double flop, const std::array<std::vector<float>, 3>& r) {
    const auto pct = [&](float us) { return 100.0 * flop / (us * 1e-6) / kPeak; };
    const Stat a = stat(r[0]), b = stat(r[1]), q = stat(r[2]);
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    std::cout << "ab " << what << " | form3 " << a.med << " us [" << a.p25 << ", " << a.p75 << "] (" << pct(a.med) << " %) | form4 " << b.med << " us ["
              << b.p25 << ", " << b.p75 << "] (" << pct(b.med) << " %) | form4/form3 paired median ";
    std::cout.precision(4);
    std::cout << q.med << " [" << q.p25 << ", " << q.p75 << "] n " << r[2].size();
    std::cout.precision(3);
    std::cout << " | t " << wall() << '\n';
}
// STRIX_QPF2K_DENSE_AB=2 (t9 root cause of the form-4 slowdowns: QSA q|k|v at T 8192, K 5120): form 3 (raster 1) vs
// form 4 at rasters 1/2/4/8, complete linear warm and the projection without epilogue stores (probe 4).
void bench_ab_raster(hipStream_t stream) {
    struct Item { Shape shape; std::uint32_t rows; };
    const std::vector<Item> items{{shapes()[2], 8192U}, {shapes()[0], 8192U}, {Shape{"ksweep", 5120U, {16384U}, true, 512U}, 8192U}};
    for (const auto& it : items) {
        auto p = make_problem(it.shape, it.rows, 0U, 9U);
        const auto site = tiled_site<std::uint16_t>(p, p.out->as<float>());
        const double flop = 2.0 * p.rows * p.width * p.outputs;
        _Float16* a16 = q8_rows::tiled::scratch(stream, static_cast<std::uint64_t>(p.rows_up) * q8_rows::tiled::scratch_pitch(p.width));
        const auto items_c = static_cast<std::uint64_t>(p.rows) * (p.width / 8U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::tiled::convert_kernel<TiledSite<std::uint16_t>>), dim3(static_cast<unsigned>((items_c + 255U) / 256U)), dim3(256U), 0,
            stream, site, p.rows, a16, q8_rows::dense_f16_overflow_counter());
        const std::string name = std::string(it.shape.name) + " K " + std::to_string(p.width) + " T " + std::to_string(p.rows);
        for (const unsigned r : {1U, 2U, 4U, 8U}) {
            const auto f3 = [&] { q8_rows::tiled::launch(site, p.rows, stream, false, q8_rows::tiled::kPre, 1U, 3U); };
            const auto f4 = [&] { q8_rows::tiled::launch(site, p.rows, stream, false, q8_rows::tiled::kPre, r, 4U); };
            print_ab(name + " linear warm form4-raster " + std::to_string(r), flop, time_ab(f3, f4, stream, nullptr, 10U));
            const auto p3 = [&] { q8_rows::tiled::launch_projection<TiledSite<std::uint16_t>, q8_rows::tiled::kPre, 4U>(site, p.rows, stream, false, 3U, 1U, a16); };
            const auto p4 = [&] { q8_rows::tiled::launch_projection<TiledSite<std::uint16_t>, q8_rows::tiled::kPre, 4U>(site, p.rows, stream, false, 4U, r, a16); };
            print_ab(name + " probe4 no-epilogue warm form4-raster " + std::to_string(r), flop, time_ab(p3, p4, stream, nullptr, 10U));
        }
    }
}
// STRIX_QPF2K_DENSE_AB=3 (rev 6c default check): complete linears of the BF16 >= 4096-output sites, interleaved,
// T 2048/4096/8192 cold + warm: "A" = rev 5 (FP16 scratch + form 3, raster 1) vs "B" = rev 6c (inline BF16 + form 4,
// raster 2), and "A" = rev 6b GDN (FP16 scratch + form 4, raster 1) vs "B" = rev 6c. Labels: form3 column = A, form4 = B.
void bench_ab_inline(hipStream_t stream) {
    Buffer flush(256U << 20U);
    for (const auto& shape : shapes()) {
        if (!shape.bf16 || outputs_of(shape) < 4096U) continue;
        for (const std::uint32_t rows : {2048U, 4096U, 8192U}) {
            auto p = make_problem(shape, rows, 0U, 7U);
            const auto site = tiled_site<std::uint16_t>(p, p.out->as<float>());
            const double flop = 2.0 * rows * p.width * p.outputs;
            const auto r5 = [&] { q8_rows::tiled::launch(site, rows, stream, false, q8_rows::tiled::kPre, 1U, 3U); };
            const auto r6b = [&] { q8_rows::tiled::launch(site, rows, stream, false, q8_rows::tiled::kPre, 1U, 4U); };
            const auto r6c = [&] { q8_rows::tiled::launch(site, rows, stream, false, q8_rows::tiled::kBf16, 2U, 4U); };
            const std::string n = std::string(shape.name) + " T " + std::to_string(rows);
            print_ab(n + " linear cold A=rev5(pre,f3) B=rev6c(inline,f4)", flop, time_ab(r5, r6c, stream, &flush, 16U));
            print_ab(n + " linear warm A=rev5(pre,f3) B=rev6c(inline,f4)", flop, time_ab(r5, r6c, stream, nullptr, 16U));
            print_ab(n + " linear cold A=rev6b(pre,f4) B=rev6c(inline,f4)", flop, time_ab(r6b, r6c, stream, &flush, 16U));
            print_ab(n + " linear warm A=rev6b(pre,f4) B=rev6c(inline,f4)", flop, time_ab(r6b, r6c, stream, nullptr, 16U));
        }
    }
}
// STRIX_QPF2K_DENSE_AB=4 ("producers emit the consumer's format"): the FP32-A sites (GDN out,
// QSA out, shared down) as they run today (inline FP32 A, form 3, default raster) against the projection reading FP16 A
// rows that a producer wrote directly (the kPre scratch layout, convert done before timing: what the projection costs
// once the producer stores fp16(x) itself) at forms 3 and 4 / rasters 1 and 2, and against convert + kPre (no producer
// change). Bits: kPre forms 3/4 on FP32 sites are already in the fail-closed gates (fp16(x) RNE, same K16 order).
void bench_ab_fp16a(hipStream_t stream) {
    Buffer flush(256U << 20U);
    for (const auto& shape : shapes()) {
        if (shape.bf16 || std::string(shape.name).rfind("tail", 0) == 0) continue;
        for (const std::uint32_t rows : {8192U, 4096U, 2048U, 1316U}) {
            auto p = make_problem(shape, rows, 0U, 11U);
            const auto site = tiled_site<float>(p, p.out->as<float>());
            const double flop = 2.0 * rows * p.width * p.outputs;
            _Float16* a16 = q8_rows::tiled::scratch(stream, static_cast<std::uint64_t>(p.rows_up) * q8_rows::tiled::scratch_pitch(p.width));
            const auto items_c = static_cast<std::uint64_t>(p.rows) * (p.width / 8U);
            const auto convert = [&] {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::tiled::convert_kernel<TiledSite<float>>), dim3(static_cast<unsigned>((items_c + 255U) / 256U)), dim3(256U), 0,
                    stream, site, p.rows, a16, q8_rows::dense_f16_overflow_counter());
            };
            convert();
            const auto today = [&] { q8_rows::tiled::launch(site, rows, stream, false, q8_rows::tiled::kF32, 2U, 3U); };  // the engine default, env-independent
            const auto pre = [&](unsigned form, unsigned r) {
                return [&, form, r] { q8_rows::tiled::launch_projection<TiledSite<float>, q8_rows::tiled::kPre>(site, rows, stream, false, form, r, a16); };
            };
            const auto conv_pre4 = [&] { q8_rows::tiled::launch(site, rows, stream, false, q8_rows::tiled::kPre, 1U, 4U); };
            const std::string n = std::string(shape.name) + " T " + std::to_string(rows);
            for (const bool cold : {true, false}) {
                Buffer* fl = cold ? &flush : nullptr;
                const std::string c = cold ? " cold" : " warm";
                print_ab(n + c + " A=today(inline f32,f3) B=fp16A f4 r1", flop, time_ab(today, pre(4U, 1U), stream, fl, 10U));
                print_ab(n + c + " A=today(inline f32,f3) B=fp16A f4 r2", flop, time_ab(today, pre(4U, 2U), stream, fl, 10U));
                print_ab(n + c + " A=today(inline f32,f3) B=fp16A f3 r1", flop, time_ab(today, pre(3U, 1U), stream, fl, 10U));
                print_ab(n + c + " A=today(inline f32,f3) B=convert+pre f4", flop, time_ab(today, conv_pre4, stream, fl, 10U));
            }
        }
    }
}
// STRIX_QPF2K_DENSE_AB=5 (K-5120 collapse, falsifier): fine K sweep at raster 1, N 16384, T 8192, FP16 rows in place,
// projection only, warm: form 3 vs form 4 at K 2560 .. 5120 step 512 (+ 4096 + 64 and 4608 + 64 to probe address aliasing).
void bench_ab_ksweep(hipStream_t stream) {
    for (const std::uint32_t width : {2560U, 3072U, 3584U, 4096U, 4160U, 4608U, 4672U, 5120U}) {
        const Shape shape{"ksweep", width, {16384U}, true, 512U};
        auto p = make_problem(shape, 8192U, 0U, 9U);
        const auto site = tiled_site<std::uint16_t>(p, p.out->as<float>());
        _Float16* a16 = q8_rows::tiled::scratch(stream, static_cast<std::uint64_t>(p.rows_up) * q8_rows::tiled::scratch_pitch(p.width));
        const auto items = static_cast<std::uint64_t>(p.rows) * (p.width / 8U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::tiled::convert_kernel<TiledSite<std::uint16_t>>), dim3(static_cast<unsigned>((items + 255U) / 256U)), dim3(256U), 0,
            stream, site, p.rows, a16, q8_rows::dense_f16_overflow_counter());
        const double flop = 2.0 * p.rows * p.width * p.outputs;
        const auto f = [&](unsigned form) { return [&, form] { q8_rows::tiled::launch_projection<TiledSite<std::uint16_t>, q8_rows::tiled::kPre>(site, p.rows, stream, false, form, 1U, a16); }; };
        print_ab("ksweep K " + std::to_string(width) + " T 8192 N 16384 r1 projection warm", flop, time_ab(f(3U), f(4U), stream, nullptr, 6U));
    }
}
void bench_ab(hipStream_t stream) {
    if (std::getenv("STRIX_QPF2K_DENSE_AB")[0] == '5') { bench_ab_ksweep(stream); return; }
    if (std::getenv("STRIX_QPF2K_DENSE_AB")[0] == '4') { bench_ab_fp16a(stream); return; }
    if (std::getenv("STRIX_QPF2K_DENSE_AB")[0] == '2') { bench_ab_raster(stream); return; }
    if (std::getenv("STRIX_QPF2K_DENSE_AB")[0] == '3') { bench_ab_inline(stream); return; }
    Buffer flush(256U << 20U);
    const unsigned reps = 16U;  // even: balanced AB / BA order
    // 1. Complete linears (convert + projection) of the FP16-scratch sites, forms 3 and 4.
    for (const auto& shape : shapes()) {
        if (!shape.bf16 || outputs_of(shape) < 4096U) continue;  // mode_of: BF16 rows with >= 4096 outputs take the FP16 scratch
        for (const std::uint32_t rows : {2048U, 4096U, 8192U}) {
            auto p = make_problem(shape, rows, 0U, 7U);
            const auto site = tiled_site<std::uint16_t>(p, p.out->as<float>());
            const double flop = 2.0 * rows * p.width * p.outputs;
            const auto f3 = [&] { q8_rows::tiled::launch(site, rows, stream, false, q8_rows::tiled::kPre, 0U, 3U); };
            const auto f4 = [&] { q8_rows::tiled::launch(site, rows, stream, false, q8_rows::tiled::kPre, 0U, 4U); };
            print_ab(std::string(shape.name) + " T " + std::to_string(rows) + " linear cold", flop, time_ab(f3, f4, stream, &flush, reps));
            print_ab(std::string(shape.name) + " T " + std::to_string(rows) + " linear warm", flop, time_ab(f3, f4, stream, nullptr, reps));
        }
    }
    // 2. K sweep at T 8192, 16384 outputs (one segment), FP16 scratch: projection alone (full, probe 3 = WMMA + fragments,
    //    probe 4 = no epilogue stores, probe 5 = both) of forms 3 and 4. time(K) = fixed per-tile cost + per-stage cost.
    for (const std::uint32_t width : {640U, 1280U, 2560U, 5120U}) {
        const Shape shape{"ksweep", width, {16384U}, true, 512U};
        auto p = make_problem(shape, 8192U, 0U, 9U);
        const auto site = tiled_site<std::uint16_t>(p, p.out->as<float>());
        _Float16* a16 = q8_rows::tiled::scratch(stream, static_cast<std::uint64_t>(p.rows_up) * q8_rows::tiled::scratch_pitch(p.width));
        const auto items = static_cast<std::uint64_t>(p.rows) * (p.width / 8U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::tiled::convert_kernel<TiledSite<std::uint16_t>>), dim3(static_cast<unsigned>((items + 255U) / 256U)), dim3(256U), 0,
            stream, site, p.rows, a16, q8_rows::dense_f16_overflow_counter());
        const double flop = 2.0 * p.rows * p.width * p.outputs;
        const auto run = [&]<unsigned Probe>(unsigned form) {
            q8_rows::tiled::launch_projection<TiledSite<std::uint16_t>, q8_rows::tiled::kPre, Probe>(site, p.rows, stream, false, form, 1U, a16);
        };
        const std::string k = "K " + std::to_string(width) + " T 8192 N 16384 projection";
        print_ab(k + " full warm", flop, time_ab([&] { run.template operator()<0U>(3U); }, [&] { run.template operator()<0U>(4U); }, stream, nullptr, 10U));
        print_ab(k + " probe3 WMMA+fragments warm", flop, time_ab([&] { run.template operator()<3U>(3U); }, [&] { run.template operator()<3U>(4U); }, stream, nullptr, 10U));
        print_ab(k + " probe4 no-epilogue warm", flop, time_ab([&] { run.template operator()<4U>(3U); }, [&] { run.template operator()<4U>(4U); }, stream, nullptr, 10U));
        print_ab(k + " probe5 WMMA+fragments no-epilogue warm", flop, time_ab([&] { run.template operator()<5U>(3U); }, [&] { run.template operator()<5U>(4U); }, stream, nullptr, 10U));
    }
}
void bench(hipStream_t stream) {
    Buffer flush(256U << 20U);
    const char* only = std::getenv("STRIX_QPF2K_DENSE_BENCH_SHAPE");
    for (const auto& shape : shapes()) {
        if (std::string(shape.name).rfind("tail", 0) == 0) continue;
        if (only && std::string(shape.name).find(only) == std::string::npos) continue;
        // STRIX_QPF2K_DENSE_BENCH_ROWS=8192: that row count only; a comma list (e.g. 512,1024,1536) benches those
        // counts (short calls such as the tail chunk of a 17.7K prompt also take the tiled form from min_rows 512).
        std::vector<std::uint32_t> row_list{2048U, 4096U, 8192U};
        if (const char* only_rows = std::getenv("STRIX_QPF2K_DENSE_BENCH_ROWS")) {
            row_list.clear();
            for (const char* s = only_rows; *s;) {
                char* end = nullptr;
                const auto v = std::strtoul(s, &end, 10);
                if (end == s) break;
                if (v >= 17U && v <= 8192U) row_list.push_back(static_cast<std::uint32_t>(v));
                s = *end == ',' ? end + 1 : end;
            }
        }
        for (const std::uint32_t rows : row_list) {
            const bool extra = rows == 8192U;
            if (shape.bf16) {
                if (shape.engine_rows == 512U) bench_shape<std::uint16_t, 512U>(shape, rows, stream, flush, extra);
                else bench_shape<std::uint16_t, 256U>(shape, rows, stream, flush, extra);
            } else if (shape.engine_rows == 64U) bench_shape<float, 64U>(shape, rows, stream, flush, extra);
            else bench_shape<float, 256U>(shape, rows, stream, flush, extra);
        }
    }
}
}  // namespace

int main() try {
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
    if (std::getenv("STRIX_QPF2K_DENSE_AB")) bench_ab(stream);
    else if (std::getenv("STRIX_QPF2K_DENSE_BENCH")) bench(stream);
    else gates(stream);
    check(hipStreamDestroy(stream), "destroy stream");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "tiled dense FP16 fixture failed: " << error.what() << '\n';
    return 1;
}
