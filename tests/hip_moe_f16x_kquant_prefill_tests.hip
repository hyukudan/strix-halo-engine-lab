// K-quant envelope of the shared f16x grouped MoE prefill (known_rows enqueue_prefill): Q4_K gate/up, Q4_K / Q6_K down,
// top-16, I up to 1536, SwiGLU clamp (first binder model-B). Synthetic weights, no model.
// Checks (default mode):
// - gate/up h of sampled rows vs a host double sum over the same FP16 operands (device A image and B decode emulated), with the
//   clamp trait applied in the reference; FP32-accumulation tolerance;
// - down raw slot dots vs a host double sum over the device h (scaled and rounded as the kernel stages it);
// - clamp entry with limit 1e30 == unclamped entry, bitwise, on every h and raw dot;
// - fail-closed admission (top_k 17, form 4, clamp with a Q8_0 down, route BF16 view).
// Bench mode (`bench [REPS [ROWS]]`): E512 T=ROWS (default 4096; 2048 = one 8K-prompt window of the first binder) K16 H1536 I1536,
// Q6_K and Q4_K down with clamp 10; prologue+CSR / gate-up / down ms per layer.
// Check mode also prints a DIGEST line per case (FNV-1a 64 of the h and raw-dot bits): equal across builds = bitwise equal.
// Real mode (`real GGUF GATE_AT UP_AT DOWN_AT DOWN_TYPE`): one model-B MoE layer's experts read from the pack at the absolute
// tensor offsets (synthetic BF16 rows, uniform random top-16 routes, clamp 10): windows of 16 / 512 / 2048 / 4096 rows, stage
// times, sampled rows vs the FP16-operand reference (pass/fail) and the FAST-class drift vs an FP32-weight reference.
#include <fstream>
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "strix/q51_known_rows.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace a8 = strix::moe_q4k_a8;
namespace kr = strix::moe_q4k_a8::known_rows;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Buffer {
    void* base = nullptr; std::uint64_t stride = 0, bytes = 0;
    Buffer() = default;
    Buffer(std::uint64_t rows, std::uint64_t row_bytes) : stride((row_bytes + 255U) / 256U * 256U), bytes(rows * stride + 256U) {
        check(hipMalloc(&base, bytes), "alloc"); check(hipMemset(base, 0, bytes), "clear");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer& operator=(Buffer&& o) noexcept { std::swap(base, o.base); std::swap(stride, o.stride); std::swap(bytes, o.bytes); return *this; }
    ~Buffer() { if (base) (void)hipFree(base); }
    template <class T> kr::StridedRows<T> rows() const { return {static_cast<T*>(base), stride}; }
    template <class T> std::vector<T> read() const {
        std::vector<T> out(bytes / sizeof(T));
        check(hipMemcpy(out.data(), base, out.size() * sizeof(T), hipMemcpyDeviceToHost), "read");
        return out;
    }
};
std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33U; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33U; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33U; return x;
}
__device__ std::uint64_t dmix(std::uint64_t x) {
    x ^= x >> 33U; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33U; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33U; return x;
}
// Random K-quant blocks; scale words in bounded ranges (Q4_K d, dmin in [2^-12, 2^-10); Q6_K d in [2^-13, 2^-11)).
__global__ void fill_kquant(unsigned char* w, std::uint64_t blocks, unsigned type, std::uint64_t seed) {
    const unsigned bytes = type == 12U ? 144U : 210U;
    for (std::uint64_t b = blockIdx.x * 256ULL + threadIdx.x; b < blocks; b += 256ULL * gridDim.x) {
        unsigned char* p = w + b * bytes;
        std::uint64_t r = dmix(seed ^ (b * 0x9e3779b97f4a7c15ULL));
        for (unsigned i = 0; i < bytes; ++i) { if (i % 8U == 0U) r = dmix(r + i); p[i] = static_cast<unsigned char>(r >> (8U * (i % 8U))); }
        r = dmix(r ^ 77U);
        const auto half_bits = [&](unsigned exponent) {  // FP16 with biased exponent `exponent`, random mantissa
            return static_cast<unsigned short>((exponent << 10U) | static_cast<unsigned>((r >>= 10U) & 1023U));
        };
        if (type == 12U) {
            const unsigned short d = half_bits(3U + static_cast<unsigned>(r & 1U)), m = half_bits(3U + static_cast<unsigned>((r >> 1U) & 1U));
            p[0] = static_cast<unsigned char>(d); p[1] = static_cast<unsigned char>(d >> 8U);
            p[2] = static_cast<unsigned char>(m); p[3] = static_cast<unsigned char>(m >> 8U);
        } else {
            const unsigned short d = half_bits(2U + static_cast<unsigned>(r & 1U));
            p[208] = static_cast<unsigned char>(d); p[209] = static_cast<unsigned char>(d >> 8U);
        }
    }
}
float half_to_float(unsigned short h) { return static_cast<float>(std::bit_cast<_Float16>(h)); }
float round_half(float v) { return static_cast<float>(static_cast<_Float16>(v)); }
// One FP16 weight as the kernel decodes it (row = one output row of width k superblocks).
float q4k_weight(const unsigned char* row, unsigned k, bool round = true) {
    const unsigned char* b = row + (k / 256U) * 144U;
    const unsigned g = (k % 256U) / 32U, l = k % 32U;
    const auto s = [&](unsigned i) { return static_cast<unsigned>(b[4U + i]); };
    const unsigned sc = g < 4U ? s(g) & 63U : (s(g + 4U) & 15U) | ((s(g - 4U) >> 6U) << 4U);
    const unsigned m = g < 4U ? s(g + 4U) & 63U : (s(g + 4U) >> 4U) | ((s(g) >> 6U) << 4U);
    const float d = half_to_float(static_cast<unsigned short>(b[0] | b[1] << 8U));
    const float dm = half_to_float(static_cast<unsigned short>(b[2] | b[3] << 8U));
    const unsigned q = (static_cast<unsigned>(b[16U + 32U * (g / 2U) + l]) >> (4U * (g % 2U))) & 15U;
    const float v = std::fma(d * static_cast<float>(sc), static_cast<float>(q), -(dm * static_cast<float>(m)));
    return round ? round_half(v) : v;
}
float q6k_weight(const unsigned char* row, unsigned k, bool round = true) {  // ggml dequantize_row_q6_K order
    const unsigned char* b = row + (k / 256U) * 210U;
    const unsigned i = k % 256U, n = i / 128U, j = (i % 128U) / 32U, l = i % 32U;
    const unsigned ql = (static_cast<unsigned>(b[64U * n + 32U * (j & 1U) + l]) >> (4U * (j >> 1U))) & 15U;
    const unsigned qh = (static_cast<unsigned>(b[128U + 32U * n + l]) >> (2U * j)) & 3U;
    const int q = static_cast<int>(ql | qh << 4U) - 32;
    const float d = half_to_float(static_cast<unsigned short>(b[208] | b[209] << 8U));
    const float sc = static_cast<float>(static_cast<signed char>(b[192U + 8U * n + 2U * j + l / 16U]));
    return round ? round_half(d * sc * static_cast<float>(q)) : d * sc * static_cast<float>(q);
}
int f16_exponent(float peak) {
    if (!(peak > 0.0F) || !std::isfinite(peak)) return 0;
    int e; (void)std::frexp(peak, &e);
    return std::min(120, std::max(-120, 14 - e));
}
struct Case { const char* name; unsigned H, I, E, T, K, down; float limit, x_scale; };
struct RealWeights { const char* gguf; std::uint64_t gate_at, up_at, down_at; };  // one layer of the pack (absolute offsets)
void read_file(const char* path, std::uint64_t at, std::uint64_t bytes, void* device) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("open ") + path);
    std::vector<char> host(bytes);
    f.seekg(static_cast<std::streamoff>(at));
    f.read(host.data(), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("short read of GGUF tensor");
    check(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice), "upload real weights");
}
struct Run { std::vector<float> h, raw; unsigned tune = 0; };
struct Fixture {
    Case c; hipStream_t stream = nullptr; a8::Request request;
    Buffer gate, up, down, input, routes, weights, codes, deltas, sums, status, output, clear, dcodes, ddeltas, dsums, rawdots, csr;
    std::vector<unsigned char> hg, hu, hd; std::vector<std::uint16_t> x; std::vector<std::uint32_t> rt;
    kr::PrefillRows view;
    std::uint64_t gate_row = 0, down_row = 0;
    explicit Fixture(const Case& cs, bool host_copy, const RealWeights* pack = nullptr) : c(cs) {
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        request.enabled = true; request.input = a8::InputKind::bf16; request.shape = {c.H, c.I, c.E, c.K, c.T};
        request.gate_type = request.up_type = 12U; request.swiglu_limit = c.limit;
        gate_row = c.H / 256U * 144U; down_row = c.I / 256U * (c.down == 12U ? 144U : 210U);
        const std::uint64_t gb = static_cast<std::uint64_t>(c.E) * c.I * (c.H / 256U), db = static_cast<std::uint64_t>(c.E) * c.H * (c.I / 256U);
        gate = Buffer(1, gb * 144U); up = Buffer(1, gb * 144U); down = Buffer(1, db * (c.down == 12U ? 144U : 210U));
        const std::uint64_t seed = mix(c.H * 7U + c.I * 13U + c.E * 31U + c.T + c.down);
        if (pack) {
            read_file(pack->gguf, pack->gate_at, gb * 144U, gate.base); read_file(pack->gguf, pack->up_at, gb * 144U, up.base);
            read_file(pack->gguf, pack->down_at, db * (c.down == 12U ? 144U : 210U), down.base);
        } else {
        hipLaunchKernelGGL(fill_kquant, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(gate.base), gb, 12U, seed ^ 1U);
        hipLaunchKernelGGL(fill_kquant, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(up.base), gb, 12U, seed ^ 2U);
        hipLaunchKernelGGL(fill_kquant, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(down.base), db, c.down, seed ^ 3U);
        }
        check(hipGetLastError(), "fill");
        const unsigned T = c.T, K = c.K, H = c.H, I = c.I;
        input = Buffer(T, 2U * H); routes = Buffer(T, 4U * K); weights = Buffer(T, 4U * K);
        codes = Buffer(T, H); deltas = Buffer(T, 4U * (H / 32U)); sums = Buffer(T, 4U * (H / 32U)); status = Buffer(T, 4U);
        output = Buffer(T, 4U * K * 2U * I); clear = Buffer(T, 4U * H);
        dcodes = Buffer(T, K * I); ddeltas = Buffer(T, 4U * K * (I / 32U)); dsums = Buffer(T, 4U * K * (I / 32U)); rawdots = Buffer(T, 4U * K * H);
        const auto layout = strix::q51_known_rows::scratch_layout(T, c.E, K, true);
        csr = Buffer(1, layout.bytes);
        x.assign(static_cast<std::size_t>(input.stride / 2U) * T, 0U);
        rt.assign(static_cast<std::size_t>(routes.stride / 4U) * T, 0U);
        std::vector<float> rw(static_cast<std::size_t>(weights.stride / 4U) * T, 0.0F);
        std::uint64_t r = seed;
        const auto bf16 = [](float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U); };
        for (unsigned t = 0; t < T; ++t) {
            for (unsigned k = 0; k < H; ++k) {
                r = mix(r);
                float v = static_cast<float>(static_cast<std::int64_t>(r % 2001U) - 1000) / 1000.0F * c.x_scale;
                if (k % 97U == 5U) v *= 40.0F;
                if ((k + t) % 211U == 3U) v = 0.0F;
                x[static_cast<std::size_t>(t) * (input.stride / 2U) + k] = bf16(v);
            }
            for (unsigned s = 0; s < K; ++s) {
                for (;;) {
                    r = mix(r);
                    const auto e = static_cast<std::uint32_t>(r % c.E);
                    bool dup = false;
                    for (unsigned q = 0; q < s; ++q) dup |= rt[static_cast<std::size_t>(t) * (routes.stride / 4U) + q] == e;
                    if (!dup) { rt[static_cast<std::size_t>(t) * (routes.stride / 4U) + s] = e; break; }
                }
                rw[static_cast<std::size_t>(t) * (weights.stride / 4U) + s] = 0.05F + static_cast<float>((r >> 20U) % 1000U) / 1000.0F;
            }
        }
        check(hipMemcpy(input.base, x.data(), x.size() * 2U, hipMemcpyHostToDevice), "input");
        check(hipMemcpy(routes.base, rt.data(), rt.size() * 4U, hipMemcpyHostToDevice), "routes");
        check(hipMemcpy(weights.base, rw.data(), rw.size() * 4U, hipMemcpyHostToDevice), "route weights");
        view.input = input.rows<const void>(); view.routes = routes.rows<const unsigned>(); view.route_weights = weights.rows<const float>();
        view.codes = codes.rows<signed char>(); view.deltas = deltas.rows<float>(); view.sums = sums.rows<int>(); view.status = status.rows<unsigned>();
        view.output = output.rows<float>(); view.clear = clear.rows<float>();
        view.down_codes = dcodes.rows<signed char>(); view.down_deltas = ddeltas.rows<float>(); view.down_sums = dsums.rows<int>();
        view.rawdots = rawdots.rows<float>();
        const auto ptr = [&](std::uint64_t o) { return reinterpret_cast<unsigned*>(static_cast<unsigned char*>(csr.base) + o); };
        view.csr = {ptr(layout.header), ptr(layout.offsets), ptr(layout.touched), ptr(layout.rows), ptr(layout.slots), ptr(layout.batch_experts),
            ptr(layout.batch_begins)};
        check(hipStreamSynchronize(stream), "fixture setup");
        if (host_copy) { hg = gate.read<unsigned char>(); hu = up.read<unsigned char>(); hd = down.read<unsigned char>(); }
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
    ~Fixture() { if (stream) (void)hipStreamDestroy(stream); }
    void enqueue(float limit, unsigned down_type = 0U, int form = 3, unsigned rows = 0U) {
        a8::Request q = request; q.swiglu_limit = limit; q.shape.logical_rows = rows ? rows : c.T;
        kr::enqueue_prefill(q, view, rows ? rows : c.T, c.H, static_cast<const unsigned char*>(gate.base), static_cast<const unsigned char*>(up.base),
            down_type ? down_type : c.down, static_cast<const unsigned char*>(down.base), stream, form);
    }
    Run run(float limit, unsigned rows = 0U) {
        check(hipMemsetAsync(output.base, 0xa5, output.bytes, stream), "poison h");
        check(hipMemsetAsync(rawdots.base, 0xa5, rawdots.bytes, stream), "poison raw");
        enqueue(limit, 0U, 3, rows);
        check(hipStreamSynchronize(stream), "run");
        Run out; out.tune = kr::prefill_last_tune(); out.h = output.read<float>(); out.raw = rawdots.read<float>();
        unsigned summary = 0; check(hipMemcpy(&summary, view.csr.summary, 4U, hipMemcpyDeviceToHost), "summary");
        const auto st = status.read<unsigned>();
        unsigned bad = 0; for (unsigned t = 0; t < (rows ? rows : c.T); ++t) bad |= st[t * (status.stride / 4U)];
        if (summary != 0U || bad != 0U) throw std::runtime_error("CSR summary / row status not zero: " + std::to_string(summary) + " / " + std::to_string(bad));
        return out;
    }
};
// Sampled rows vs the host double reference; returns the number of failures.
struct Drift { double e16 = 0.0, e32 = 0.0, n32 = 0.0; };  // squared error vs the FP16-operand / FP32-weight reference, |ref32|^2
unsigned compare(Fixture& f, const Run& run, float limit, unsigned& clamped, unsigned& checked, unsigned rows = 0U, unsigned npick = 6U,
    Drift* drift = nullptr) {
    const unsigned H = f.c.H, I = f.c.I, K = f.c.K;
    const std::uint64_t hs = f.output.stride / 4U, rs = f.rawdots.stride / 4U;
    unsigned fails = 0;
    const unsigned T = rows ? rows : f.c.T;
    const unsigned picks[6]{0U, T - 1U, T / 3U, T / 2U + 1U, T - 2U, 1U};
    std::vector<float> a(H), a2(I);
    std::vector<double> x32(H), h32(I);
    for (unsigned pi = 0; pi < npick; ++pi) {
        const unsigned t = picks[pi];
        float peak = 0.0F;
        for (unsigned k = 0; k < H; ++k) peak = std::max(peak, std::fabs(std::bit_cast<float>(static_cast<std::uint32_t>(f.x[t * (f.input.stride / 2U) + k]) << 16U)));
        const int s = f16_exponent(peak);
        for (unsigned k = 0; k < H; ++k) {
            x32[k] = std::bit_cast<float>(static_cast<std::uint32_t>(f.x[t * (f.input.stride / 2U) + k]) << 16U);
            a[k] = round_half(static_cast<float>(x32[k]) * std::ldexp(1.0F, s));
        }
        for (unsigned slot = 0; slot < K; ++slot) {
            const unsigned e = f.rt[t * (f.routes.stride / 4U) + slot];
            const float* hdev = run.h.data() + t * hs + static_cast<std::uint64_t>(slot) * 2U * I;
            for (unsigned o = 0; o < I; ++o) {
                double g = 0.0, u = 0.0, sg = 0.0, su = 0.0, g32 = 0.0, u32 = 0.0;
                const unsigned char* gr = f.hg.data() + (static_cast<std::uint64_t>(e) * I + o) * f.gate_row;
                const unsigned char* ur = f.hu.data() + (static_cast<std::uint64_t>(e) * I + o) * f.gate_row;
                for (unsigned k = 0; k < H; ++k) {
                    const double wg = q4k_weight(gr, k), wu = q4k_weight(ur, k);
                    g += a[k] * wg; u += a[k] * wu; sg += std::fabs(a[k] * wg); su += std::fabs(a[k] * wu);
                    if (drift) { g32 += x32[k] * q4k_weight(gr, k, false); u32 += x32[k] * q4k_weight(ur, k, false); }
                }
                const double un = std::ldexp(1.0, -s);
                g *= un; u *= un; sg *= un; su *= un;
                const double gc = std::isfinite(limit) ? std::min(g, static_cast<double>(limit)) : g;
                const double uc = std::isfinite(limit) ? std::max(-static_cast<double>(limit), std::min(u, static_cast<double>(limit))) : u;
                clamped += (gc != g || uc != u) ? 1U : 0U; ++checked;
                const double silu = gc / (1.0 + std::exp(-gc)), ref = silu * uc;
                if (drift) {
                    const double l = std::isfinite(limit) ? static_cast<double>(limit) : 1e300;
                    const double gc32 = std::min(g32, l), uc32 = std::max(-l, std::min(u32, l));
                    h32[o] = gc32 / (1.0 + std::exp(-gc32)) * uc32;
                }
                const double tg = 2e-5 * sg + 1e-30, tu = 2e-5 * su + 1e-30;
                const double tol = (1.1 * std::fabs(uc) + tu) * tg + (std::fabs(silu) + 1.1 * tg) * tu + 1e-5 * std::fabs(ref) + 1e-30;
                if (!(std::fabs(hdev[o] - ref) <= tol)) {
                    if (fails < 5U) std::printf("FAIL h row %u slot %u out %u: device %.9g ref %.9g tol %.3g\n", t, slot, o, static_cast<double>(hdev[o]), ref, tol);
                    ++fails;
                }
            }
            // Down over the device h, staged as the kernel does (row/slot max |h| exponent, FP16 rounding).
            float hp = 0.0F;
            for (unsigned o = 0; o < I; ++o) hp = std::max(hp, std::fabs(hdev[o]));
            const int s2 = f16_exponent(hp);
            for (unsigned k = 0; k < I; ++k) a2[k] = round_half(hdev[k] * std::ldexp(1.0F, s2));
            const float* rdev = run.raw.data() + t * rs + static_cast<std::uint64_t>(slot) * H;
            for (unsigned o = 0; o < H; ++o) {
                const unsigned char* dr = f.hd.data() + (static_cast<std::uint64_t>(e) * H + o) * f.down_row;
                double sum = 0.0, mag = 0.0;
                for (unsigned k = 0; k < I; ++k) {
                    const double w = f.c.down == 12U ? q4k_weight(dr, k) : q6k_weight(dr, k);
                    sum += a2[k] * w; mag += std::fabs(a2[k] * w);
                }
                const double un = std::ldexp(1.0, -s2), ref = sum * un, tol = 2e-5 * mag * un + 1e-30;
                if (drift) {
                    double r32 = 0.0;
                    for (unsigned k = 0; k < I; ++k) r32 += h32[k] * (f.c.down == 12U ? q4k_weight(dr, k, false) : q6k_weight(dr, k, false));
                    drift->e16 += (rdev[o] - ref) * (rdev[o] - ref); drift->e32 += (rdev[o] - r32) * (rdev[o] - r32); drift->n32 += r32 * r32;
                }
                if (!(std::fabs(rdev[o] - ref) <= tol)) {
                    if (fails < 5U) std::printf("FAIL raw row %u slot %u out %u: device %.9g ref %.9g tol %.3g\n", t, slot, o, static_cast<double>(rdev[o]), ref, tol);
                    ++fails;
                }
            }
        }
    }
    return fails;
}
template <class F> bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
struct BenchHook { hipEvent_t ev[4]; };
void hook2(void* user, hipStream_t s) { (void)hipEventRecord(static_cast<BenchHook*>(user)->ev[2], s); }
void hook1(void* user, hipStream_t s) {
    (void)hipEventRecord(static_cast<BenchHook*>(user)->ev[1], s);
    kr::set_prefill_stage_hook(hook2, user, 2U);
}
void stage_times(Fixture& f, float limit, unsigned rows, unsigned reps, double (&out)[3]) {
    BenchHook hk{};
    for (auto& e : hk.ev) check(hipEventCreate(&e), "event");
    std::vector<double> v[3];
    for (unsigned i = 0; i <= reps; ++i) {
        check(hipEventRecord(hk.ev[0], f.stream), "ev0");
        kr::set_prefill_stage_hook(hook1, &hk, 1U);
        f.enqueue(limit, 0U, 3, rows);
        check(hipEventRecord(hk.ev[3], f.stream), "ev3");
        check(hipStreamSynchronize(f.stream), "timed run");
        for (unsigned j = 0; j < 3U; ++j) { float m = 0.0F; check(hipEventElapsedTime(&m, hk.ev[j], hk.ev[j + 1U]), "elapsed"); if (i) v[j].push_back(m); }
    }
    for (unsigned j = 0; j < 3U; ++j) { std::sort(v[j].begin(), v[j].end()); out[j] = v[j][v[j].size() / 2U]; }
    for (auto& e : hk.ev) (void)hipEventDestroy(e);
}
int real(const char* gguf, std::uint64_t gate_at, std::uint64_t up_at, std::uint64_t down_at, unsigned down) {
    const Case c{"real", 1536U, 1536U, 512U, 4096U, 16U, down, 10.0F, 1.0F};
    const RealWeights w{gguf, gate_at, up_at, down_at};
    Fixture f(c, true, &w);
    unsigned fails = 0;
    for (const unsigned rows : {16U, 512U, 2048U, 4096U}) {
        const Run r = f.run(c.limit, rows);
        unsigned clamped = 0, checked = 0, bad = 0;
        Drift dr;
        if (rows == 512U || rows == 2048U) bad = compare(f, r, c.limit, clamped, checked, rows, 2U, &dr);
        double ms[3];
        stage_times(f, c.limit, rows, rows >= 2048U ? 5U : 10U, ms);
        const double mac = 3.0 * rows * c.K * c.I * static_cast<double>(c.H);
        std::printf("REAL down=%s rows=%u tune=%u: prologue+csr %.3f ms, gate/up %.3f ms, down %.3f ms, layer %.3f ms (%.2f TMAC/s, %.4f ms/row)",
            down == 14U ? "Q6_K" : "Q4_K", rows, r.tune, ms[0], ms[1], ms[2], ms[0] + ms[1] + ms[2], mac / (ms[1] + ms[2]) / 1e9,
            (ms[0] + ms[1] + ms[2]) / rows);
        if (checked) std::printf("; %u h checked, %u clamped, %u failures, rel L2 vs FP16-operand ref %.2e, vs FP32-weight ref %.2e",
            checked, clamped, bad, std::sqrt(dr.e16 / dr.n32), std::sqrt(dr.e32 / dr.n32));
        std::printf("\n");
        fails += bad;
    }
    std::printf("%s real layer (%u failures)\n", fails == 0U ? "PASS" : "FAIL", fails);
    return fails == 0U ? 0 : 1;
}
int bench(unsigned reps, unsigned rows) {
    int rc = 0;
    for (const unsigned down : {14U, 12U}) {
        const Case c{down == 14U ? "bench-q6k" : "bench-q4k", 1536U, 1536U, 512U, rows, 16U, down, 10.0F, 1.0F};
        Fixture f(c, false);
        BenchHook hk{};
        for (auto& e : hk.ev) check(hipEventCreate(&e), "event");
        std::vector<double> pc, gu, dn;
        for (unsigned i = 0; i <= reps; ++i) {
            check(hipEventRecord(hk.ev[0], f.stream), "ev0");
            kr::set_prefill_stage_hook(hook1, &hk, 1U);
            f.enqueue(c.limit);
            check(hipEventRecord(hk.ev[3], f.stream), "ev3");
            check(hipStreamSynchronize(f.stream), "bench run");
            float m[3]{};
            for (unsigned j = 0; j < 3U; ++j) check(hipEventElapsedTime(&m[j], hk.ev[j], hk.ev[j + 1U]), "elapsed");
            if (i == 0U) continue;  // warm-up (cold) run
            pc.push_back(m[0]); gu.push_back(m[1]); dn.push_back(m[2]);
        }
        unsigned summary = 0; check(hipMemcpy(&summary, f.view.csr.summary, 4U, hipMemcpyDeviceToHost), "summary");
        const auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2U]; };
        const double mac_gu = 2.0 * c.T * c.K * c.I * static_cast<double>(c.H), mac_dn = static_cast<double>(c.T) * c.K * c.I * c.H;
        const double g = med(gu), d = med(dn), p = med(pc);
        std::printf("BENCH %s T=%u E=%u K=%u H=%u I=%u tune=%u clamp=%g: prologue+csr %.3f ms, gate/up %.3f ms (%.2f TMAC/s), "
            "down %.3f ms (%.2f TMAC/s), layer %.3f ms, summary=%u\n", c.name, c.T, c.E, c.K, c.H, c.I, kr::prefill_last_tune(),
            static_cast<double>(c.limit), p, g, mac_gu / g / 1e9, d, mac_dn / d / 1e9, p + g + d, summary);
        if (summary != 0U || g + d > 45.0) { std::printf("BENCH FAIL %s (summary or gate/up + down > 45 ms)\n", c.name); rc = 1; }
        for (auto& e : hk.ev) (void)hipEventDestroy(e);
    }
    return rc;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "check";
        kr::reserve_prefill_pools(6560U, 1536U);  // ceil(4096 * 16 / 10) rows of CSR capacity, rounded to 8; DA16 image for I 1536
        if (mode == "bench")
            return bench(argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 10U, argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 4096U);
        if (mode == "real" && argc == 7)
            return real(argv[2], std::strtoull(argv[3], nullptr, 10), std::strtoull(argv[4], nullptr, 10), std::strtoull(argv[5], nullptr, 10),
                static_cast<unsigned>(std::atoi(argv[6])));
        unsigned fails = 0;
        const Case cases[]{
            {"q6k-I1536-tune82", 1536U, 1536U, 64U, 512U, 16U, 14U, a8::Request{}.swiglu_limit, 1.0F},
            {"q4k-I1536-clamp10", 1536U, 1536U, 64U, 512U, 16U, 12U, 10.0F, 1.0F},
            {"q6k-I768-odd-rows-tune80-clamp10", 512U, 768U, 32U, 64U, 16U, 14U, 10.0F, 2.0F},
        };
        for (const auto& c : cases) {
            Fixture f(c, true);
            const Run r = f.run(c.limit);
            unsigned clamped = 0, checked = 0;
            const unsigned bad = compare(f, r, c.limit, clamped, checked);
            const auto fnv = [](const std::vector<float>& v) {
                std::uint64_t x = 0xcbf29ce484222325ULL;
                for (const float f32 : v) { const auto b = std::bit_cast<std::uint32_t>(f32); for (unsigned k = 0; k < 32U; k += 8U) { x ^= (b >> k) & 255U; x *= 0x100000001b3ULL; } }
                return static_cast<unsigned long long>(x);
            };
            std::printf("DIGEST %s h %016llx raw %016llx\n", c.name, fnv(r.h), fnv(r.raw));
            const bool coverage = !std::isfinite(c.limit) || clamped > 0U;
            std::printf("%s %s: tune %u, %u h checked, %u clamped, %u failures\n", bad == 0U && coverage ? "PASS" : "FAIL", c.name, r.tune,
                checked, clamped, bad);
            fails += bad + (coverage ? 0U : 1U);
            {  // DA16 image down (h converted once per batch into the shared FP16 image, poisoned first) == the plain down
               // (FP32 h converted per column tile), bitwise on h and raw dots; each arm must have taken its path.
                kr::prefill_test_poison_da16(f.stream, 6560U);
                kr::set_prefill_da16(1);
                const Run img = f.run(c.limit);
                const bool img_ran = kr::prefill_da16_ran();
                kr::set_prefill_da16(0);
                const Run plain = f.run(c.limit);
                const bool plain_ran = kr::prefill_da16_ran();
                kr::set_prefill_da16(-1);
                const bool same = img.h.size() == plain.h.size() && img.raw.size() == plain.raw.size() &&
                    std::memcmp(img.h.data(), plain.h.data(), img.h.size() * 4U) == 0 &&
                    std::memcmp(img.raw.data(), plain.raw.data(), img.raw.size() * 4U) == 0 &&
                    std::memcmp(img.raw.data(), r.raw.data(), r.raw.size() * 4U) == 0;
                const bool ok = same && img_ran && !plain_ran;
                std::printf("%s %s: DA16 image down (ran %d) == plain down (ran %d), bitwise (h and raw dots)\n", ok ? "PASS" : "FAIL",
                    c.name, img_ran ? 1 : 0, plain_ran ? 1 : 0);
                fails += ok ? 0U : 1U;
            }
            if (!std::isfinite(c.limit)) {  // clamp entry with a limit no value reaches == the unclamped entry, bitwise
                const Run big = f.run(1e30F);
                const bool same = big.h.size() == r.h.size() && std::memcmp(big.h.data(), r.h.data(), r.h.size() * 4U) == 0 &&
                    std::memcmp(big.raw.data(), r.raw.data(), r.raw.size() * 4U) == 0;
                std::printf("%s %s: clamp entry (limit 1e30) == unclamped entry, bitwise (h and raw dots)\n", same ? "PASS" : "FAIL", c.name);
                fails += same ? 0U : 1U;
                const bool refused = throws([&] { a8::Request q = f.request; q.shape.top_k = 17U; kr::enqueue_prefill(q, f.view, f.c.T, f.c.H,
                        static_cast<const unsigned char*>(f.gate.base), static_cast<const unsigned char*>(f.up.base), 14U,
                        static_cast<const unsigned char*>(f.down.base), f.stream, 3); }) &&
                    throws([&] { f.enqueue(f.request.swiglu_limit, 0U, 4); }) &&
                    throws([&] { f.enqueue(10.0F, 8U); }) &&
                    throws([&] { f.view.route_bf16 = true; struct Reset { kr::PrefillRows& v; ~Reset() { v.route_bf16 = false; } } reset{f.view};
                        f.enqueue(f.request.swiglu_limit); });
                std::printf("%s fail-closed admission (top_k 17, form 4, clamp with Q8_0 down, route BF16)\n", refused ? "PASS" : "FAIL");
                fails += refused ? 0U : 1U;
            }
        }
        std::printf("%s hip_moe_f16x_kquant_prefill_tests (%u failures)\n", fails == 0U ? "PASS" : "FAIL", fails);
        return fails == 0U ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
}
