// Shared expert in tensor-parallel form on the trellis library (identity ds41-trellis-k5-v1:shexp-tp): rank r runs gate /
// up rows [1152 r, +1152) of w1 / w3 (5120 -> 1152) and the K-half columns [1152 r, +1152) of w2 (1152 -> 5120); its
// partial product is added onto its half sum. This bench, on real pack tensors (cold: a 256 MB buffer is touched
// between reps, as a layer of the engine sees its weights):
//   - exactness of the library calls the engine wants to use: beta = 1 (y = L(x) + y in the library) vs a separate FP32
//     add; strix_grouped_moe {gate, up} vs two GEMVs; the lean chain vs the chain through enqueue_trellis_linear;
//   - time of one shared expert per rank and row count: replicated (today's whole w1 / w3 / w2 through the entry point),
//     TP through the entry point, and TP lean (padding in place, no staging copies, beta = 1 when exact).
// usage: strix-hip-ds41-shexp-tp-bench PACK_DIR LAYER [LAYER...]
#include "strix/ds41_moe.hpp"
#include "strix/ds41_trellis.hpp"

#include <hip/hip_runtime.h>
#include <strix_trellis.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::ds41;

namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void lib(strix_status s, const char* what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error(std::string(what) + ": " + strix_status_string(s));
}
struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(std::size_t bytes) { check(hipMalloc(&p, bytes), "hipMalloc"); }
    ~DevBuf() { (void)hipFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    float* f() const { return static_cast<float*>(p); }
};
std::uint32_t lcg(std::uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(s >> 33U);
}
std::vector<float> make_x(std::size_t n, std::uint64_t seed) {
    std::vector<float> x(n);
    for (auto& v : x) {
        float a = 0;
        for (int k = 0; k < 4; ++k) a += static_cast<float>(lcg(seed) % 20001U) / 10000.0F - 1.0F;
        if (lcg(seed) % 997U == 0U) a *= 16.0F;
        v = a;
    }
    return x;
}
strix_stream lib_stream(hipStream_t s) {
    strix_stream st{};
    st.struct_size = sizeof(st);
    st.abi_version = STRIX_TRELLIS_ABI_VERSION;
    st.hip_stream = s;
    return st;
}
void set_tensor(strix_tensor_desc& t, const void* data, std::uint32_t rows, std::uint32_t width) {
    t.data = const_cast<void*>(data);
    t.buffer_bytes = std::uint64_t{rows} * width * 4U;
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = width;
    t.strides[0] = width;
    t.strides[1] = 1;
}
__global__ void touch_kernel(float* p, std::size_t n) {
    for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x)
        p[i] = p[i] + 1.0F;
}
__global__ void add_kernel(float* y, const float* t, std::size_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] + t[i];
}
__global__ void swiglu_kernel(const float* g, const float* u, std::size_t n, float limit, float* a) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) a[i] = swiglu_clamped(g[i], u[i], limit);
}
std::vector<float> down(const float* p, std::size_t n, hipStream_t s) {
    std::vector<float> v(n);
    check(hipMemcpyAsync(v.data(), p, n * 4U, hipMemcpyDeviceToHost, s), "download");
    check(hipStreamSynchronize(s), "sync");
    return v;
}
std::size_t differ(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::memcmp(&a[i], &b[i], 4) != 0 ? 1U : 0U;
    return d;
}
int failures = 0;
void report(const char* what, std::size_t d, std::size_t n, bool required) {
    std::printf("  %-52s %s", what, d == 0 ? "bitwise" : "DIFFERS");
    if (d != 0) std::printf(" (%zu / %zu values)", d, n);
    std::printf("%s\n", required || d == 0 ? "" : " [informative]");
    if (d != 0 && required) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s PACK_DIR LAYER [LAYER...]\n", argv[0]);
        return 2;
    }
    try {
        TrellisPack pack(argv[1]);
        TrellisContext ctx;
        hipStream_t st;
        check(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
        const auto c = static_cast<strix_context>(ctx.library_context());
        constexpr std::uint32_t H = 5120, I = 2304, hI = 1152, W = 8;
        constexpr float limit = 10.0F;
        struct Layer { int l; const TrellisSite *w1, *w3, *w2; const TrellisSite* t1[2]; const TrellisSite* t3[2]; const TrellisSite* t2[2]; };
        std::vector<Layer> layers;
        for (int a = 2; a < argc; ++a) {
            Layer L{};
            L.l = std::atoi(argv[a]);
            const std::string p = "layers." + std::to_string(L.l) + ".ffn.shared_experts.";
            L.w1 = &ctx.prepare(pack, p + "w1");
            L.w3 = &ctx.prepare(pack, p + "w3");
            L.w2 = &ctx.prepare(pack, p + "w2");
            for (std::uint32_t r = 0; r < 2U; ++r) {
                L.t1[r] = &ctx.prepare(pack, p + "w1", {TrellisSlice::Kind::rows, hI * r, hI});
                L.t3[r] = &ctx.prepare(pack, p + "w3", {TrellisSlice::Kind::rows, hI * r, hI});
                L.t2[r] = &ctx.prepare(pack, p + "w2", {TrellisSlice::Kind::columns, hI * r, hI});
            }
            layers.push_back(L);
        }
        ctx.reserve(st, W);
        std::printf("library %s; shapes: whole w1 %ux%u sizes 0x%02x, slice w1 %ux%u sizes 0x%02x, slice w2 %ux%u sizes 0x%02x\n",
                    TrellisContext::library_version(), layers[0].w1->columns, layers[0].w1->rows, trellis_submit_sizes(*layers[0].w1),
                    layers[0].t1[0]->columns, layers[0].t1[0]->rows, trellis_submit_sizes(*layers[0].t1[0]), layers[0].t2[0]->columns,
                    layers[0].t2[0]->rows, trellis_submit_sizes(*layers[0].t2[0]));
        DevBuf dws(64ULL << 20U), dx(std::size_t{2U * W} * H * 4U), dg(std::size_t{W} * I * 4U), du(std::size_t{W} * I * 4U),
            da(std::size_t{W} * I * 4U), dy(std::size_t{2U * W} * H * 4U), dy2(std::size_t{2U * W} * H * 4U), dt(std::size_t{W} * H * 4U),
            dflush(256ULL << 20U);
        check(hipMemset(dflush.p, 0, 256ULL << 20U), "memset");
        const auto x = make_x(std::size_t{W} * H, 0x5eedULL);
        const auto y0 = make_x(std::size_t{W} * H, 0xabcULL);
        const auto upload = [&](const DevBuf& b, const std::vector<float>& v) {
            check(hipMemcpyAsync(b.p, v.data(), v.size() * 4U, hipMemcpyHostToDevice, st), "upload");
        };
        const auto gemv = [&](const TrellisSite& s, const float* in, float* out, std::uint32_t rows, float beta) {
            strix_matmul_desc d{};
            strix_matmul_desc_init(&d);
            d.weights = static_cast<strix_weights>(s.weights);
            set_tensor(d.input, in, rows, s.columns);
            set_tensor(d.output, out, rows, s.rows);
            d.beta = beta;
            d.workspace = dws.p;
            d.workspace_bytes = 64ULL << 20U;
            const auto ls = lib_stream(st);
            lib(strix_gemv(c, &d, &ls), "gemv");
        };
        bool beta_exact = true, grouped_ok = true;
        // ---- exactness of the candidate calls (layer 0 of the list, both ranks, R 1..8)
        for (const auto& L : layers)
            for (std::uint32_t r = 0; r < 2U; ++r)
                for (std::uint32_t R = 1; R <= W; ++R) {
                    upload(dx, x);
                    // beta = 1 on w2's slice: y0 + L(a) in the library vs t = L(a), y = y0 + t
                    upload(da, make_x(std::size_t{W} * I, 77U + R));
                    upload(dy, y0);
                    upload(dy2, y0);
                    gemv(*L.t2[r], da.f(), dy.f(), R, 1.0F);
                    gemv(*L.t2[r], da.f(), dt.f(), R, 0.0F);
                    hipLaunchKernelGGL(add_kernel, dim3((R * H + 255U) / 256U), dim3(256), 0, st, dy2.f(), dt.f(), std::size_t{R} * H);
                    const auto a = down(dy.f(), std::size_t{R} * H, st), b = down(dy2.f(), std::size_t{R} * H, st);
                    const auto d = differ(a, b);
                    if (d != 0) beta_exact = false;
                    if (d != 0 || (R == 1U || R == 6U)) {
                        const std::string what = "L" + std::to_string(L.l) + " r" + std::to_string(r) + " w2 beta=1 vs add, R" + std::to_string(R);
                        report(what.c_str(), d, a.size(), false);
                    }
                    // grouped {w1 slice, w3 slice} vs two GEMVs
                    std::vector<strix_matmul_desc> g(2);
                    for (std::size_t k = 0; k < 2U; ++k) {
                        strix_matmul_desc_init(&g[k]);
                        g[k].weights = static_cast<strix_weights>((k == 0 ? L.t1[r] : L.t3[r])->weights);
                        set_tensor(g[k].input, dx.p, R, H);
                        set_tensor(g[k].output, k == 0 ? dg.p : du.p, R, hI);
                    }
                    strix_grouped_moe_desc gd{};
                    gd.struct_size = sizeof(gd);
                    gd.abi_version = STRIX_TRELLIS_ABI_VERSION;
                    gd.groups = g.data();
                    gd.group_count = 2;
                    gd.workspace = dws.p;
                    gd.workspace_bytes = 64ULL << 20U;
                    const auto ls = lib_stream(st);
                    const auto gs = strix_grouped_moe(c, &gd, &ls);
                    if (gs != STRIX_SUCCESS) {
                        if (R == 1U && r == 0U && &L == &layers[0]) std::printf("  grouped_moe refused: %s\n", strix_status_string(gs));
                        grouped_ok = false;
                    } else {
                        const auto gg = down(dg.f(), std::size_t{R} * hI, st), gu = down(du.f(), std::size_t{R} * hI, st);
                        gemv(*L.t1[r], dx.f(), dg.f(), R, 0.0F);
                        gemv(*L.t3[r], dx.f(), du.f(), R, 0.0F);
                        const auto rg = down(dg.f(), std::size_t{R} * hI, st), ru = down(du.f(), std::size_t{R} * hI, st);
                        const auto d2 = differ(gg, rg) + differ(gu, ru);
                        if (d2 != 0) grouped_ok = false;
                        if (d2 != 0 || R == 1U || R == 6U) {
                            const std::string what = "L" + std::to_string(L.l) + " r" + std::to_string(r) + " grouped gate/up vs gemv, R" + std::to_string(R);
                            report(what.c_str(), d2, gg.size() + gu.size(), false);
                        }
                    }
                }
        std::printf("beta=1 exact on every case: %s; grouped gate/up exact and accepted: %s\n", beta_exact ? "yes" : "NO", grouped_ok ? "yes" : "NO");

        // ---- the lean TP chain of one rank (what the engine hook does) and its reference through the entry point
        // lean: x rows [n, m) := x row n - 1 in place; gate / up GEMV of m rows; SwiGLU of m rows (pad rows are copies of
        // row n - 1: rows are independent within a call); w2 GEMV of m2 rows (m2 <= m) accumulated onto y (beta = 1 if
        // exact, else into t then an add of the n rows). y needs room for m2 rows.
        const auto lean = [&](const Layer& L, std::uint32_t r, float* xin, float* y, std::uint32_t R) {
            const std::uint32_t chunk = trellis_chunk_rows(*L.t1[r]);
            for (std::uint32_t b = 0; b < R; b += chunk) {
                const std::uint32_t n = std::min(chunk, R - b);
                const std::uint32_t m = trellis_padded_rows(*L.t1[r], n), m2 = trellis_padded_rows(*L.t2[r], n);
                if (m2 > m) throw std::logic_error("lean: w2 needs more rows than w1");
                float* xr = xin + std::size_t{b} * H;
                for (std::uint32_t k = n; k < m; ++k)
                    check(hipMemcpyAsync(xr + std::size_t{k} * H, xr + std::size_t{n - 1U} * H, H * 4U, hipMemcpyDeviceToDevice, st), "pad");
                gemv(*L.t1[r], xr, dg.f(), m, 0.0F);
                gemv(*L.t3[r], xr, du.f(), m, 0.0F);
                hipLaunchKernelGGL(swiglu_kernel, dim3((m * hI + 255U) / 256U), dim3(256), 0, st, dg.f(), du.f(), std::size_t{m} * hI, limit, da.f());
                float* yr = y + std::size_t{b} * H;
                if (beta_exact) {
                    gemv(*L.t2[r], da.f(), yr, m2, 1.0F);
                } else {
                    gemv(*L.t2[r], da.f(), dt.f(), m2, 0.0F);
                    hipLaunchKernelGGL(add_kernel, dim3((n * H + 255U) / 256U), dim3(256), 0, st, yr, dt.f(), std::size_t{n} * H);
                }
            }
        };
        const auto entry_tp = [&](const Layer& L, std::uint32_t r, const float* xin, float* y, std::uint32_t R) {
            enqueue_trellis_linear(*L.t1[r], xin, dg.f(), R, H, hI, false, st);
            enqueue_trellis_linear(*L.t3[r], xin, du.f(), R, H, hI, false, st);
            enqueue_swiglu_f32(dg.f(), du.f(), R, hI, limit, da.f(), st);
            enqueue_trellis_linear(*L.t2[r], da.f(), y, R, hI, H, true, st);
        };
        const auto entry_whole = [&](const Layer& L, const float* xin, float* y, std::uint32_t R) {
            enqueue_trellis_linear(*L.w1, xin, dg.f(), R, H, I, false, st);
            enqueue_trellis_linear(*L.w3, xin, du.f(), R, H, I, false, st);
            enqueue_swiglu_f32(dg.f(), du.f(), R, I, limit, da.f(), st);
            enqueue_trellis_linear(*L.w2, da.f(), y, R, I, H, false, st);
        };
        for (const auto& L : layers)
            for (std::uint32_t r = 0; r < 2U; ++r)
                for (std::uint32_t R = 1; R <= W; ++R) {
                    // reference: every row alone through the entry point
                    std::vector<float> ref(std::size_t{R} * H);
                    for (std::uint32_t k = 0; k < R; ++k) {
                        upload(dy, y0);
                        upload(dx, x);
                        entry_tp(L, r, dx.f() + std::size_t{k} * H, dy.f() + std::size_t{k} * H, 1);
                        const auto v = down(dy.f() + std::size_t{k} * H, H, st);
                        std::copy(v.begin(), v.end(), ref.begin() + static_cast<std::ptrdiff_t>(std::size_t{k} * H));
                    }
                    upload(dy, y0);
                    upload(dx, x);
                    entry_tp(L, r, dx.f(), dy.f(), R);
                    const auto e = down(dy.f(), std::size_t{R} * H, st);
                    upload(dy, y0);
                    upload(dx, x);
                    lean(L, r, dx.f(), dy.f(), R);
                    const auto l = down(dy.f(), std::size_t{R} * H, st);
                    const std::string w1 = "L" + std::to_string(L.l) + " r" + std::to_string(r) + " entry TP R" + std::to_string(R) + " vs rows alone";
                    const std::string w2 = "L" + std::to_string(L.l) + " r" + std::to_string(r) + " lean TP R" + std::to_string(R) + " vs rows alone";
                    report(w1.c_str(), differ(e, ref), e.size(), true);
                    report(w2.c_str(), differ(l, ref), l.size(), true);
                }

        // ---- timing, cold: touch 256 MB between reps; per layer of the list, sum over the layers
        hipEvent_t e0, e1;
        check(hipEventCreate(&e0), "event");
        check(hipEventCreate(&e1), "event");
        const auto time = [&](const std::function<void(const Layer&)>& f) {
            constexpr int reps = 10;
            double total = 0;
            for (int k = 0; k < reps; ++k)
                for (const auto& L : layers) {
                    hipLaunchKernelGGL(touch_kernel, dim3(1024), dim3(256), 0, st, dflush.f(), (256ULL << 20U) / 4U);
                    check(hipEventRecord(e0, st), "record");
                    f(L);
                    check(hipEventRecord(e1, st), "record");
                    check(hipEventSynchronize(e1), "sync");
                    float ms = 0;
                    check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                    total += ms;
                }
            return 1000.0 * total / reps / static_cast<double>(layers.size());  // us per layer
        };
        const double whole_bytes = static_cast<double>(layers[0].w1->device_bytes + layers[0].w3->device_bytes + layers[0].w2->device_bytes);
        const double half_bytes = static_cast<double>(layers[0].t1[0]->device_bytes + layers[0].t3[0]->device_bytes + layers[0].t2[0]->device_bytes);
        std::printf("cold time per layer (us) and GB/s on the weights read (whole %.1f MB, rank half %.1f MB)\n", whole_bytes / 1e6, half_bytes / 1e6);
        std::printf("  R   replicated(entry)   TP r0 entry   TP r0 lean   TP r1 lean\n");
        for (std::uint32_t R : {1U, 2U, 4U, 6U, 8U}) {
            upload(dx, x);
            const double tw = time([&](const Layer& L) { entry_whole(L, dx.f(), dy.f(), R); });
            const double te = time([&](const Layer& L) { entry_tp(L, 0, dx.f(), dy.f(), R); });
            const double tl0 = time([&](const Layer& L) { lean(L, 0, dx.f(), dy.f(), R); });
            const double tl1 = time([&](const Layer& L) { lean(L, 1, dx.f(), dy.f(), R); });
            std::printf("  %u   %7.1f (%5.1f)    %7.1f (%5.1f)  %7.1f (%5.1f)  %7.1f (%5.1f)\n", R, tw, whole_bytes / tw / 1e3, te,
                        half_bytes / te / 1e3, tl0, half_bytes / tl0 / 1e3, tl1, half_bytes / tl1 / 1e3);
        }
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
        std::printf("%s\n", failures == 0 ? "ALL BITWISE" : "FAILURES");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
