// Trellis attention projections micro-benchmark (not a ctest): q_b / o_a / o_b of one tensor-parallel rank read from
// a trellis pack (dense.L<layer>.safetensors, the C API library of strix_trellis.h) against today's site
// kernel (enqueue_q8_0_a8 on U6 = c-u6 rows of the same shape, random bytes) at the decode window rows 1..8.
// Rank slices as the engine takes them (ds41-trellis-k5-v1, attention section):
//   q_b  rank r: output rows [16384 r, 16384 r + 16384) of wq_b (1280 -> 32768): N tiles [1024 r, 1024 r + 1024), svh slice;
//   o_a  rank r: wo_a.slice.g for g in [4 r, 4 r + 4) (each 4096 -> 1024, the pack stores the groups separately);
//   o_b  rank r: K-half [4096 r, 4096 r + 4096) of wo_b (8192 -> 5120): I tiles [256 r, 256 r + 256), suh slice, svh whole.
// Checks: every row of a call with R rows against the same row submitted alone (R-invariance), and the slices against
// the whole tensors (q_b rows, o_b h0 + h1) numerically. Timing cold: rotating prepared copies over >= 640 MB.
// usage: strix-hip-ds41-trellis-attn-bench <dense.L<n>.safetensors> <layer> [rank 0|1] [iterations] [sites]

#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix_trellis.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
void check(strix_status s, const char* what) {
    if (s != STRIX_SUCCESS) { std::fprintf(stderr, "%s: %s\n", what, strix_status_string(s)); std::exit(2); }
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc"); return p; }
template <class T> T* dev(const std::vector<T>& h) { T* p = dev<T>(h.size()); check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    check(hipDeviceSynchronize(), "sync");
    check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}

double time_ms(int iterations, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) launch(i);
    std::vector<float> t;
    for (int i = 0; i < iterations; ++i) {
        check(hipEventRecord(a, nullptr), "record");
        launch(i + 3);
        check(hipEventRecord(b, nullptr), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms);
    }
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// ---- minimal safetensors reader ----
struct Tensor { std::string dtype; std::vector<std::uint64_t> shape; const std::uint8_t* data = nullptr; std::uint64_t bytes = 0; };
struct File {
    std::vector<std::uint8_t> raw;
    std::map<std::string, Tensor> tensors;
    const Tensor& at(const std::string& n) const {
        const auto it = tensors.find(n);
        if (it == tensors.end()) { std::fprintf(stderr, "missing tensor %s\n", n.c_str()); std::exit(2); }
        return it->second;
    }
};
File read_safetensors(const char* path) {
    File f;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    f.raw.resize(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(f.raw.data()), static_cast<std::streamsize>(f.raw.size()));
    std::uint64_t n = 0;
    std::memcpy(&n, f.raw.data(), 8);
    const std::string h(reinterpret_cast<const char*>(f.raw.data() + 8), n);
    const std::uint8_t* base = f.raw.data() + 8 + n;
    std::size_t p = 0;
    while ((p = h.find("\"dtype\"", p)) != std::string::npos) {
        // the tensor name is the last quoted key before this object's '{'
        const std::size_t brace = h.rfind('{', p), q1 = h.rfind('"', h.rfind(':', brace) - 1), q0 = h.rfind('"', q1 - 1);
        Tensor t;
        const std::string name = h.substr(q0 + 1, q1 - q0 - 1);
        const std::size_t d0 = h.find('"', p + 7) + 1;
        t.dtype = h.substr(d0, h.find('"', d0) - d0);
        const std::size_t s0 = h.find('[', h.find("\"shape\"", p)) + 1, s1 = h.find(']', s0);
        for (std::size_t k = s0; k < s1;) {
            char* e = nullptr;
            const auto v = std::strtoull(h.c_str() + k, &e, 10);
            const auto used = static_cast<std::size_t>(e - (h.c_str() + k));
            if (used == 0) { ++k; continue; }
            t.shape.push_back(v);
            k += used;
        }
        const std::size_t o0 = h.find('[', h.find("\"data_offsets\"", p)) + 1;
        char* e = nullptr;
        const auto a = std::strtoull(h.c_str() + o0, &e, 10);
        const auto b = std::strtoull(e + 1, nullptr, 10);
        t.data = base + a;
        t.bytes = b - a;
        f.tensors[name] = t;
        p = s1;
    }
    return f;
}

// ---- transports (host blob: trellis, suh, svh) of a tile range of one tensor ----
struct Pack {
    std::vector<std::uint8_t> blob;
    strix_exl3_metadata_v1 meta{};
    std::uint64_t rows = 0, columns = 0;
    strix_exl3_transport_desc desc() const {
        strix_exl3_transport_desc d{};
        d.struct_size = sizeof(d);
        d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.format = STRIX_TRANSPORT_EXL3_CANONICAL;
        d.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        d.data = blob.data();
        d.data_bytes = blob.size();
        d.metadata = &meta;
        d.metadata_bytes = sizeof(meta);
        d.rows = rows;
        d.columns = columns;
        return d;
    }
};
// I tiles [i0, i0 + ni), N tiles [n0, n0 + nn) of prefix's trellis [I/16][N/16][16 bits] (I = input, N = output).
Pack make_pack(const File& f, const std::string& prefix, std::uint64_t i0, std::uint64_t ni, std::uint64_t n0, std::uint64_t nn) {
    const auto& tr = f.at(prefix + ".trellis");
    const auto& suh = f.at(prefix + ".suh");
    const auto& svh = f.at(prefix + ".svh");
    const auto& mul = f.at(prefix + ".mul1");
    std::uint32_t marker = 0;
    std::memcpy(&marker, mul.data, 4);
    if (tr.dtype != "I16" || tr.shape.size() != 3 || marker != STRIX_EXL3_MARKER_MUL1) { std::fprintf(stderr, "bad %s\n", prefix.c_str()); std::exit(2); }
    const std::uint64_t IT = tr.shape[0], NT = tr.shape[1], tile = tr.shape[2] * 2U;  // bytes per 16 x 16 tile
    if (i0 + ni > IT || n0 + nn > NT || suh.bytes != IT * 32U || svh.bytes != NT * 32U) { std::fprintf(stderr, "bad range %s\n", prefix.c_str()); std::exit(2); }
    Pack p;
    p.rows = nn * 16U;
    p.columns = ni * 16U;
    const std::uint64_t tb = ni * nn * tile, sb = p.columns * 2U, vb = p.rows * 2U;
    p.blob.resize(tb + sb + vb);
    for (std::uint64_t i = 0; i < ni; ++i)
        std::memcpy(p.blob.data() + i * nn * tile, tr.data + ((i0 + i) * NT + n0) * tile, nn * tile);
    std::memcpy(p.blob.data() + tb, suh.data + i0 * 32U, sb);
    std::memcpy(p.blob.data() + tb + sb, svh.data + n0 * 32U, vb);
    auto& m = p.meta;
    m.struct_size = sizeof(m);
    m.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    m.rows = p.rows;
    m.columns = p.columns;
    m.bits_x2 = static_cast<std::uint32_t>(tr.shape[2] / 16U * 2U);
    m.codebook = STRIX_CODEBOOK_MUL1;
    m.marker = STRIX_EXL3_MARKER_MUL1;
    m.trellis_offset = 0;
    m.trellis_bytes = tb;
    m.suh_offset = tb;
    m.suh_bytes = sb;
    m.svh_offset = tb + sb;
    m.svh_bytes = vb;
    return p;
}

strix_context g_ctx{};
strix_stream g_stream{};
void* g_ws = nullptr;
std::uint64_t g_ws_bytes = 0;

strix_weights prepare(const Pack& p) {
    strix_weights w{};
    const auto d = p.desc();
    check(strix_weights_prepare(g_ctx, &d, &g_stream, &w), "weights_prepare");
    return w;
}
strix_matmul_desc linear(strix_weights w, const Pack& p, const float* x, float* y, std::uint32_t rows, float beta = 0.0F) {
    strix_matmul_desc d{};
    strix_matmul_desc_init(&d);
    d.weights = w;
    d.input.data = const_cast<float*>(x);
    d.input.rank = 2;
    d.input.shape[0] = rows;
    d.input.shape[1] = p.columns;
    d.input.strides[0] = p.columns;
    d.input.strides[1] = 1;
    d.input.buffer_bytes = rows * p.columns * 4U;
    d.output.data = y;
    d.output.rank = 2;
    d.output.shape[0] = rows;
    d.output.shape[1] = p.rows;
    d.output.strides[0] = p.rows;
    d.output.strides[1] = 1;
    d.output.buffer_bytes = rows * p.rows * 4U;
    d.beta = beta;
    std::uint64_t need = 0;
    check(strix_matmul_workspace_bytes(g_ctx, &d, &need), "workspace_bytes");
    if (need > g_ws_bytes) { std::fprintf(stderr, "workspace %llu > %llu\n", (unsigned long long)need, (unsigned long long)g_ws_bytes); std::exit(2); }
    d.workspace = g_ws;
    d.workspace_bytes = g_ws_bytes;
    return d;
}
void gemv(strix_weights w, const Pack& p, const float* x, float* y, std::uint32_t rows, float beta = 0.0F) {
    const auto d = linear(w, p, x, y, rows, beta);
    check(strix_gemv(g_ctx, &d, &g_stream), "gemv");
}

std::uint64_t pack_bytes(const Pack& p) { return p.meta.trellis_bytes + p.meta.suh_bytes + p.meta.svh_bytes; }

// A site of one rank: one or more linear parts reading one input [R][in_width] and writing [R][out_width].
struct Part { Pack pack; std::uint32_t in_col = 0, out_col = 0; std::vector<strix_weights> copies; };
struct SiteT { std::string name; std::uint32_t in_width = 0, out_width = 0; std::vector<Part> parts; bool sum_parts = false; };

// gather columns [c, c + w) of [R][stride] into [R][w] (pure copies)
__global__ void gather_cols(const float* __restrict__ in, float* __restrict__ out, std::uint32_t rows, std::uint32_t stride, std::uint32_t c, std::uint32_t w) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * w) return;
    out[i] = in[(i / w) * stride + c + i % w];
}
__global__ void scatter_cols(const float* __restrict__ in, float* __restrict__ out, std::uint32_t rows, std::uint32_t stride, std::uint32_t c, std::uint32_t w) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * w) return;
    out[(i / w) * stride + c + i % w] = in[i];
}

struct Bufs { float *x, *xg, *yg, *y; };

// The site as the engine runs it (copy = which prepared copy of every part).
void run_site(const SiteT& s, int copy, const Bufs& b, std::uint32_t rows) {
    for (std::size_t k = 0; k < s.parts.size(); ++k) {
        const auto& p = s.parts[k];
        const auto w = p.copies[static_cast<std::size_t>(copy) % p.copies.size()];
        const bool direct_in = rows == 1U || p.pack.columns == s.in_width;
        const bool direct_out = rows == 1U || p.pack.rows == s.out_width;
        const float* x = b.x + p.in_col;
        if (!direct_in) {
            const std::uint32_t n = rows * static_cast<std::uint32_t>(p.pack.columns);
            gather_cols<<<(n + 255U) / 256U, 256>>>(b.x, b.xg, rows, s.in_width, p.in_col, static_cast<std::uint32_t>(p.pack.columns));
            x = b.xg;
        }
        float* y = direct_out ? b.y + p.out_col : b.yg;
        gemv(w, p.pack, x, y, rows, s.sum_parts && k > 0 ? 1.0F : 0.0F);
        if (!direct_out) {
            const std::uint32_t n = rows * static_cast<std::uint32_t>(p.pack.rows);
            scatter_cols<<<(n + 255U) / 256U, 256>>>(b.yg, b.y, rows, s.out_width, p.out_col, static_cast<std::uint32_t>(p.pack.rows));
        }
    }
}

void bench_site(SiteT& s, int iterations, std::mt19937& rng) {
    std::uint64_t bytes = 0;
    for (const auto& p : s.parts) bytes += pack_bytes(p.pack);
    const std::size_t copies = std::max<std::size_t>(4, (640ULL << 20) / bytes + 1);
    for (auto& p : s.parts)
        for (std::size_t c = 0; c < copies; ++c) p.copies.push_back(prepare(p.pack));
    check(hipStreamSynchronize(nullptr), "prepare");
    const std::uint32_t W = 8;
    std::vector<float> x(std::size_t{W} * s.in_width);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (auto& v : x) v = nd(rng);
    Bufs b{};
    b.x = dev(x);
    b.xg = dev<float>(std::size_t{W} * s.in_width);
    b.yg = dev<float>(std::size_t{W} * s.out_width);
    b.y = dev<float>(std::size_t{W} * s.out_width);
    float* x1 = dev<float>(s.in_width);
    const std::size_t on = s.out_width;
    // R1 reference of every row (copy 0)
    std::vector<std::vector<float>> ref(W);
    for (std::uint32_t r = 0; r < W; ++r) {
        Bufs b1 = b;
        check(hipMemcpy(x1, b.x + std::size_t{r} * s.in_width, s.in_width * 4U, hipMemcpyDeviceToDevice), "row");
        b1.x = x1;
        check(hipMemset(b.y, 0, on * 4U), "clear");
        run_site(s, 0, b1, 1);
        ref[r] = down(b.y, on);
    }
    {  // FNV-1a of the R1 rows (compare library versions)
        std::uint64_t h = 1469598103934665603ULL;
        for (const auto& row : ref)
            for (const float v : row) {
                std::uint32_t u = 0;
                std::memcpy(&u, &v, 4);
                for (int k = 0; k < 4; ++k) h = (h ^ ((u >> (8 * k)) & 0xffU)) * 1099511628211ULL;
            }
        std::printf("{\"site\":\"%s\",\"r1_fnv\":\"%016llx\"}\n", s.name.c_str(), (unsigned long long)h);
    }
    std::vector<std::vector<float>> r8;
    for (std::uint32_t rows = 1; rows <= W; ++rows) {
        check(hipMemset(b.y, 0, rows * on * 4U), "clear");
        run_site(s, 1, b, rows);  // another copy: same weights
        const auto got = down(b.y, rows * on);
        std::uint32_t eq_r1 = 0;
        for (std::uint32_t r = 0; r < rows; ++r) eq_r1 += std::memcmp(got.data() + r * on, ref[r].data(), on * 4U) == 0 ? 1U : 0U;
        double maxrel = 0;
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::size_t i = 0; i < on; ++i) {
                const double a = got[r * on + i], c = ref[r][i];
                maxrel = std::max(maxrel, std::fabs(a - c) / std::max(1e-6, std::fabs(c)));
            }
        if (rows == W) {
            r8.resize(W);
            for (std::uint32_t r = 0; r < W; ++r) r8[r].assign(got.begin() + r * on, got.begin() + (r + 1) * on);
        }
        const double ms = time_ms(iterations, [&](int i) { run_site(s, i, b, rows); });
        const double warm = time_ms(std::max(5, iterations / 4), [&](int) { run_site(s, 0, b, rows); });
        std::printf("{\"site\":\"%s\",\"kernel\":\"trellis\",\"rows\":%u,\"bytes\":%llu,\"ms\":%.4f,\"gbs\":%.1f,\"warm_gbs\":%.1f,"
                    "\"rows_eq_r1\":%u,\"max_rel_vs_r1\":%.3g}\n",
                    s.name.c_str(), rows, (unsigned long long)bytes, ms, static_cast<double>(bytes) / (ms * 1e-3) / 1e9,
                    static_cast<double>(bytes) / (warm * 1e-3) / 1e9, eq_r1, maxrel);
        std::fflush(stdout);
    }
    // Rows of smaller calls against the R8 call (an "always 8 rows" mode)
    for (std::uint32_t rows = 1; rows < W; ++rows) {
        check(hipMemset(b.y, 0, rows * on * 4U), "clear");
        run_site(s, 2, b, rows);
        const auto got = down(b.y, rows * on);
        std::uint32_t eq = 0;
        for (std::uint32_t r = 0; r < rows; ++r) eq += std::memcmp(got.data() + r * on, r8[r].data(), on * 4U) == 0 ? 1U : 0U;
        std::printf("{\"site\":\"%s\",\"check\":\"vs_r8\",\"rows\":%u,\"rows_eq_r8\":%u}\n", s.name.c_str(), rows, eq);
    }
    for (auto& p : s.parts) {
        for (auto w : p.copies) check(strix_weights_destroy(w), "destroy");
        p.copies.clear();
    }
    (void)hipFree(b.x); (void)hipFree(b.xg); (void)hipFree(b.yg); (void)hipFree(b.y); (void)hipFree(x1);
}

// Slices vs the whole tensor (numerical, R1): q_b rows of the full output, o_b h0 + h1 vs the full K.
void slice_check(const File& f, const std::string& L, std::uint32_t rank, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    {
        const auto full = make_pack(f, L + "attn.wq_b", 0, 80, 0, 2048), part = make_pack(f, L + "attn.wq_b", 0, 80, 1024U * rank, 1024);
        auto wf = prepare(full), wp = prepare(part);
        std::vector<float> x(1280);
        for (auto& v : x) v = nd(rng);
        auto* dx = dev(x);
        auto* yf = dev<float>(32768);
        auto* yp = dev<float>(16384);
        gemv(wf, full, dx, yf, 1);
        gemv(wp, part, dx, yp, 1);
        const auto a = down(yf, 32768), c = down(yp, 16384);
        double md = 0, mx = 0;
        std::uint32_t same = 0;
        for (std::uint32_t i = 0; i < 16384; ++i) {
            md = std::max(md, static_cast<double>(std::fabs(a[16384U * rank + i] - c[i])));
            mx = std::max(mx, static_cast<double>(std::fabs(a[16384U * rank + i])));
            same += a[16384U * rank + i] == c[i] ? 1U : 0U;
        }
        std::printf("{\"check\":\"q_b slice vs full\",\"rank\":%u,\"max_abs_diff\":%.3g,\"max_abs\":%.3g,\"bitwise_equal\":%u,\"of\":16384}\n", rank, md, mx, same);
        (void)strix_weights_destroy(wf); (void)strix_weights_destroy(wp);
        (void)hipFree(dx); (void)hipFree(yf); (void)hipFree(yp);
    }
    {
        const auto full = make_pack(f, L + "attn.wo_b", 0, 512, 0, 320);
        const auto h0 = make_pack(f, L + "attn.wo_b", 0, 256, 0, 320), h1 = make_pack(f, L + "attn.wo_b", 256, 256, 0, 320);
        auto wf = prepare(full), w0 = prepare(h0), w1 = prepare(h1);
        std::vector<float> x(8192);
        for (auto& v : x) v = nd(rng);
        auto* dx = dev(x);
        auto* yf = dev<float>(5120);
        auto* y0 = dev<float>(5120);
        auto* y1 = dev<float>(5120);
        gemv(wf, full, dx, yf, 1);
        gemv(w0, h0, dx, y0, 1);
        gemv(w1, h1, dx + 4096, y1, 1);
        const auto a = down(yf, 5120), c0 = down(y0, 5120), c1 = down(y1, 5120);
        double md = 0, mx = 0;
        for (std::uint32_t i = 0; i < 5120; ++i) {
            md = std::max(md, static_cast<double>(std::fabs(a[i] - (c0[i] + c1[i]))));
            mx = std::max(mx, static_cast<double>(std::fabs(a[i])));
        }
        std::printf("{\"check\":\"o_b h0+h1 vs full\",\"max_abs_diff\":%.3g,\"max_abs\":%.3g}\n", md, mx);
        (void)strix_weights_destroy(wf); (void)strix_weights_destroy(w0); (void)strix_weights_destroy(w1);
        (void)hipFree(dx); (void)hipFree(yf); (void)hipFree(y0); (void)hipFree(y1);
    }
}

// Today's site kernel on U6 rows of the same shape (random bytes), the engine's enqueue_q8_0_a8 (+ its A8 launch).
void bench_today(const char* name, std::uint32_t width, std::uint32_t row_width, std::uint32_t count, std::uint32_t per_group,
                 std::uint32_t x_stride, int iterations, std::mt19937& rng) {
    using namespace strix::ds41;
    const std::size_t row_bytes = std::size_t{row_width} / 32U * 26U, bytes = row_bytes * count;
    const std::size_t read = std::size_t{width} / 32U * 26U * count;
    const std::size_t copies = std::max<std::size_t>(4, (640ULL << 20) / bytes + 1);
    std::vector<std::uint8_t> host(bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& v : host) v = static_cast<std::uint8_t>(byte(rng));
    for (std::size_t blk = 0; blk < bytes / 26U; ++blk) {  // finite f16 scales
        const auto d = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(0.01F + 0.0001F * static_cast<float>(blk % 100U)));
        host[blk * 26U] = static_cast<std::uint8_t>(d & 0xffU);
        host[blk * 26U + 1U] = static_cast<std::uint8_t>(d >> 8U);
    }
    auto* w = dev<std::uint8_t>(bytes * copies);
    for (std::size_t c = 0; c < copies; ++c) check(hipMemcpy(w + c * bytes, host.data(), bytes, hipMemcpyHostToDevice), "upload");
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(std::size_t{8} * x_stride);
    for (auto& v : x) v = nd(rng);
    auto* dx = dev(x);
    auto* qc = dev<std::int8_t>(x.size());
    auto* qd = dev<float>(x.size() / 32U);
    auto* qs = dev<std::int32_t>(x.size() / 32U);
    auto* out = dev<float>(std::size_t{8} * count);
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        enqueue_quantize_a8(dx, x_stride, rows, qc, qd, qs, nullptr);
        const auto view = [&](int i) {
            Q8A8View v;
            v.weight = w + (static_cast<std::size_t>(i) % copies) * bytes;
            v.width = width; v.row_width = row_width == width ? 0U : row_width; v.count = count; v.per_group = per_group; v.first = 0;
            v.codes = qc; v.deltas = qd; v.rows = rows; v.x_stride = x_stride; v.out_stride = count; v.out = out; v.packed = 6;
            return v;
        };
        const double ms = time_ms(iterations, [&](int i) { enqueue_q8_0_a8(view(i), nullptr); });
        const double qa8 = time_ms(iterations, [&](int) { enqueue_quantize_a8(dx, x_stride, rows, qc, qd, qs, nullptr); });
        std::printf("{\"site\":\"%s\",\"kernel\":\"today-u6\",\"rows\":%u,\"bytes\":%zu,\"ms\":%.4f,\"gbs\":%.1f,\"a8_ms\":%.4f}\n", name, rows,
                    read, ms, static_cast<double>(read) / (ms * 1e-3) / 1e9, qa8);
        std::fflush(stdout);
    }
    (void)hipFree(w); (void)hipFree(dx); (void)hipFree(qc); (void)hipFree(qd); (void)hipFree(qs); (void)hipFree(out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s dense.L<n>.safetensors layer [rank] [iterations] [sites]\n", argv[0]); return 1; }
    const File f = read_safetensors(argv[1]);
    const std::string L = "layers." + std::string(argv[2]) + ".";
    const std::uint32_t rank = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 0U;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 40;
    const std::string filter = argc > 5 ? argv[5] : "all";
    const auto want = [&](const char* s) { return filter == "all" || filter.find(s) != std::string::npos; };
    check(hipSetDevice(0), "device");
    g_stream.struct_size = sizeof(g_stream);
    g_stream.abi_version = STRIX_TRELLIS_ABI_VERSION;
    g_stream.hip_stream = nullptr;
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    cd.device_ordinal = 0;
    check(strix_context_create(&cd, &g_ctx), "context");
    g_ws_bytes = 64ULL << 20;
    check(hipMalloc(&g_ws, g_ws_bytes), "workspace");
    std::mt19937 rng(20261021);
    std::printf("{\"layer\":\"%s\",\"rank\":%u,\"abi\":%u}\n", L.c_str(), rank, strix_get_abi_version());
    if (want("check")) slice_check(f, L, rank, rng);
    if (want("q_b")) {
        SiteT s{"q_b", 1280, 16384, {}, false};
        s.parts.push_back({make_pack(f, L + "attn.wq_b", 0, 80, 1024U * rank, 1024), 0, 0, {}});
        bench_site(s, iterations, rng);
        if (want("today")) bench_today("q_b", 1280, 1280, 16384, 16384, 1280, iterations, rng);
    }
    if (want("o_a")) {
        SiteT s{"o_a", 16384, 4096, {}, false};
        for (std::uint32_t g = 0; g < 4; ++g)
            s.parts.push_back({make_pack(f, L + "attn.wo_a.slice." + std::to_string(4U * rank + g), 0, 256, 0, 64), 4096U * g, 1024U * g, {}});
        bench_site(s, iterations, rng);
        if (want("today")) bench_today("o_a", 4096, 4096, 4096, 1024, 16384, iterations, rng);
    }
    if (want("o_b")) {
        SiteT s{"o_b", 4096, 5120, {}, false};
        s.parts.push_back({make_pack(f, L + "attn.wo_b", 256U * rank, 256, 0, 320), 0, 0, {}});
        bench_site(s, iterations, rng);
        if (want("today")) bench_today("o_b", 4096, 8192, 5120, 5120, 4096, iterations, rng);
    }
    if (want("full")) {  // the whole tensors (the library's measured shapes), for reference only
        SiteT q{"q_b-full", 1280, 32768, {}, false};
        q.parts.push_back({make_pack(f, L + "attn.wq_b", 0, 80, 0, 2048), 0, 0, {}});
        bench_site(q, iterations, rng);
        SiteT o{"o_b-full", 8192, 5120, {}, false};
        o.parts.push_back({make_pack(f, L + "attn.wo_b", 0, 512, 0, 320), 0, 0, {}});
        bench_site(o, iterations, rng);
    }
    (void)hipFree(g_ws);
    (void)strix_context_destroy(g_ctx);
    return 0;
}
