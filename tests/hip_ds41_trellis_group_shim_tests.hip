// Wiring test of TrellisGroupRunner's grouped-library path (not a ctest; run explicitly on a GPU): the library does not have the grouped
// strided GEMV yet, so this test compiles hip_ds41_trellis_group.hip against a SHIM of the requested entry
// (strix_grouped_gemv: strided input/output, every row the bits of strix_gemv R1) built from strix_gemv R1 calls, and
// checks: the runner selects the grouped path, its rows == the per-group enqueue_trellis_linear path for R = 1..16; a
// shim with R-dependent bits (submits the chunk's rows together: R2 differs, ds41-trellis-k5-v1 o_a classes) is
// rejected by the self-check and the runner falls back to the staged path (still == the per-group path); a shim that
// refuses the shape (STRIX_ERROR_NOT_IMPLEMENTED) falls back too.
// usage: strix-hip-ds41-trellis-group-shim-tests PACK_DIR
#include <hip/hip_runtime.h>
#include <strix_trellis.h>

#include <cstdint>
#include <cstring>
#include <vector>

// ---- shim of the requested API (see ds41_trellis_group.hpp) ----
#define STRIX_TRELLIS_HAS_GROUPED_GEMV 1
#define STRIX_GEMV_ROW_INVARIANT_R1 UINT32_C(1)
typedef struct strix_grouped_gemv_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    const strix_weights* weights;
    uint64_t group_count;
    uint64_t rows;
    const float* input;
    uint64_t input_row_stride, input_group_stride, input_buffer_bytes;
    float* output;
    uint64_t output_row_stride, output_group_stride, output_buffer_bytes;
    void* workspace;
    uint64_t workspace_bytes;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved[4];
} strix_grouped_gemv_desc;

namespace shim {
int mode = 0;  // 0 good (per row R1), 1 R-dependent (rows of a chunk together, contiguous scratch), 2 refuses
std::uint32_t I = 4096, N = 1024;
float *xs = nullptr, *ys = nullptr;
strix_matmul_desc desc(strix_weights w, const float* x, float* y, std::uint64_t rows, void* ws, std::uint64_t wsb) {
    strix_matmul_desc d{};
    strix_matmul_desc_init(&d);
    d.weights = w;
    d.input.data = const_cast<float*>(x);
    d.input.rank = 2;
    d.input.shape[0] = rows;
    d.input.shape[1] = I;
    d.input.strides[0] = I;
    d.input.strides[1] = 1;
    d.input.buffer_bytes = rows * I * 4U;
    d.output.data = y;
    d.output.rank = 2;
    d.output.shape[0] = rows;
    d.output.shape[1] = N;
    d.output.strides[0] = N;
    d.output.strides[1] = 1;
    d.output.buffer_bytes = rows * N * 4U;
    d.workspace = ws;
    d.workspace_bytes = wsb;
    return d;
}
}  // namespace shim

extern "C" strix_status strix_grouped_gemv_workspace_bytes(strix_context c, const strix_grouped_gemv_desc* d, uint64_t* out) {
    if (shim::mode == 2) return STRIX_ERROR_NOT_IMPLEMENTED;
    std::uint64_t m = 0;
    for (std::uint64_t r = 1; r <= 8U; ++r) {
        const auto md = shim::desc(d->weights[0], reinterpret_cast<const float*>(std::uintptr_t{1} << 40U), reinterpret_cast<float*>(std::uintptr_t{1} << 41U), r, nullptr, 0);
        std::uint64_t b = 0;
        const auto s = strix_matmul_workspace_bytes(c, &md, &b);
        if (s != STRIX_SUCCESS) return s;
        m = b > m ? b : m;
    }
    *out = m;
    return STRIX_SUCCESS;
}
extern "C" strix_status strix_grouped_gemv(strix_context c, const strix_grouped_gemv_desc* d, const strix_stream* st) {
    if (shim::mode == 2) return STRIX_ERROR_NOT_IMPLEMENTED;
    const auto hs = static_cast<hipStream_t>(st->hip_stream);
    for (std::uint64_t g = 0; g < d->group_count; ++g) {
        if (shim::mode == 0) {
            for (std::uint64_t r = 0; r < d->rows; ++r) {  // one row: contiguous in place
                const auto md = shim::desc(d->weights[g], d->input + r * d->input_row_stride + g * d->input_group_stride,
                                           d->output + r * d->output_row_stride + g * d->output_group_stride, 1, d->workspace, d->workspace_bytes);
                const auto s = strix_gemv(c, &md, st);
                if (s != STRIX_SUCCESS) return s;
            }
        } else {
            for (std::uint64_t r = 0; r < d->rows; ++r)
                (void)hipMemcpyAsync(shim::xs + r * shim::I, d->input + r * d->input_row_stride + g * d->input_group_stride, shim::I * 4U,
                                     hipMemcpyDeviceToDevice, hs);
            const auto md = shim::desc(d->weights[g], shim::xs, shim::ys, d->rows, d->workspace, d->workspace_bytes);
            const auto s = strix_gemv(c, &md, st);
            if (s != STRIX_SUCCESS) return s;
            for (std::uint64_t r = 0; r < d->rows; ++r)
                (void)hipMemcpyAsync(d->output + r * d->output_row_stride + g * d->output_group_stride, shim::ys + r * shim::N, shim::N * 4U,
                                     hipMemcpyDeviceToDevice, hs);
        }
    }
    return STRIX_SUCCESS;
}

// the runner, compiled against the shim
#ifndef STRIX_DS41_TRELLIS_LIB
#define STRIX_DS41_TRELLIS_LIB 1
#endif
#include "../src/hip_ds41_trellis_group.hip"

#include <cstdio>
#include <string>

using namespace strix::ds41;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s PACK_DIR\n", argv[0]); return 1; }
    hipStream_t s = nullptr;
    (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    (void)hipMalloc(reinterpret_cast<void**>(&shim::xs), 8U * 4096U * 4U);
    (void)hipMalloc(reinterpret_cast<void**>(&shim::ys), 8U * 1024U * 4U);
    TrellisPack pack(argv[1]);
    TrellisContext ctx;
    std::vector<const TrellisSite*> sites;
    for (int g = 4; g < 8; ++g) sites.push_back(&ctx.prepare(pack, *trellis_pack_name("blk.0.attn_output_a.weight", g)));
    ctx.reserve(s, 8);
    constexpr std::uint32_t RM = 16, XW = 4 * 4096, YW = 4 * 1024;
    std::vector<float> hx(std::size_t{RM} * XW);
    std::uint64_t seed = 99;
    for (auto& v : hx) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<float>(static_cast<std::int32_t>(seed >> 40U) % 30001) / 7000.0F;
    }
    float *x = nullptr, *ya = nullptr, *yb = nullptr;
    (void)hipMalloc(reinterpret_cast<void**>(&x), hx.size() * 4U);
    (void)hipMalloc(reinterpret_cast<void**>(&ya), std::size_t{RM} * YW * 4U);
    (void)hipMalloc(reinterpret_cast<void**>(&yb), std::size_t{RM} * YW * 4U);
    (void)hipMemcpy(x, hx.data(), hx.size() * 4U, hipMemcpyHostToDevice);
    int failures = 0;
    for (const int m : {0, 1, 2}) {
        shim::mode = m;
        TrellisGroupRunner run(sites, 1, s);
        const std::string d = run.describe();
        const bool grouped = d.find("grouped library GEMV") != std::string::npos;
        std::uint32_t differ = 0, rows = 0;
        for (std::uint32_t R = 1; R <= RM; ++R) {
            (void)hipMemsetAsync(ya, 0xff, std::size_t{R} * YW * 4U, s);
            (void)hipMemsetAsync(yb, 0xfe, std::size_t{R} * YW * 4U, s);
            for (std::size_t g = 0; g < 4; ++g) enqueue_trellis_linear(*sites[g], x + g * 4096U, ya + g * 1024U, R, XW, YW, false, s);
            run.enqueue(sites, x, XW, yb, YW, R, s);
            std::vector<float> a(std::size_t{R} * YW), b(std::size_t{R} * YW);
            (void)hipStreamSynchronize(s);
            (void)hipMemcpy(a.data(), ya, a.size() * 4U, hipMemcpyDeviceToHost);
            (void)hipMemcpy(b.data(), yb, b.size() * 4U, hipMemcpyDeviceToHost);
            for (std::uint32_t r = 0; r < R; ++r, ++rows) differ += std::memcmp(a.data() + std::size_t{r} * YW, b.data() + std::size_t{r} * YW, YW * 4U) != 0 ? 1U : 0U;
        }
        const bool want_grouped = m == 0;
        const bool ok = grouped == want_grouped && differ == 0U;
        failures += ok ? 0 : 1;
        std::printf("shim mode %d: %s; rows %u, differ %u -> %s\n", m, d.c_str(), rows, differ, ok ? "PASS" : "FAIL");
    }
    std::printf("GROUP SHIM %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
