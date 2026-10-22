// R-invariance of the trellis library on real pack tensors (identity ds41-trellis-k5-v1): does a row get the same bits
// whatever the number of rows submitted with it? For each linear: per-row R=1 GEMV references, then GEMV R=2..8 (two
// row offsets), GEMM R=1..16/32, the chunked enqueue_trellis_linear (contiguous, strided, accumulate) and a repeat.
// usage: strix-hip-ds41-trellis-tests PACK_DIR NAME [NAME...]   (NAME = pack name, e.g. layers.0.ffn.shared_experts.w1;
//        NAME@rows:FIRST:COUNT / NAME@cols:FIRST:COUNT prepares a slice). Exit 0 iff every check is bitwise.
#include "strix/ds41_trellis.hpp"

#include <hip/hip_runtime.h>
#include <strix_trellis.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::ds41;

namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(std::size_t bytes) { check(hipMalloc(&p, bytes), "hipMalloc"); }
    ~DevBuf() { (void)hipFree(p); }
    float* f() const { return static_cast<float*>(p); }
};
std::uint32_t lcg(std::uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(s >> 33U);
}
// activations like a normalized hidden state: mostly |x| < 3, a few outliers
std::vector<float> make_x(std::size_t rows, std::size_t width, std::uint64_t seed) {
    std::vector<float> x(rows * width);
    for (auto& v : x) {
        float a = 0;
        for (int k = 0; k < 4; ++k) a += static_cast<float>(lcg(seed) % 20001U) / 10000.0F - 1.0F;
        if (lcg(seed) % 997U == 0U) a *= 16.0F;
        v = a;
    }
    return x;
}
std::uint32_t ulp_diff(float a, float b) {
    std::int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = std::int32_t(0x80000000U) - ia;
    if (ib < 0) ib = std::int32_t(0x80000000U) - ib;
    const auto d = static_cast<std::int64_t>(ia) - ib;
    return static_cast<std::uint32_t>(d < 0 ? -d : d);
}
strix_stream lib_stream(hipStream_t s) {
    strix_stream st{};
    st.struct_size = sizeof(st);
    st.abi_version = STRIX_TRELLIS_ABI_VERSION;
    st.hip_stream = s;
    return st;
}
void set_tensor(strix_tensor_desc& t, void* data, std::uint32_t rows, std::uint32_t width) {
    t.data = data;
    t.buffer_bytes = std::uint64_t{rows} * width * 4U;
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = width;
    t.strides[0] = width;
    t.strides[1] = 1;
}

int failures = 0;
bool informative = false;  // raw library calls against the R1 reference: reported, not failures (R classes)
// rows [0, rows) of got (stride n) against ref rows [first, first + rows)
void compare(const std::string& what, const std::vector<float>& got, const std::vector<float>& ref, std::uint32_t first,
             std::uint32_t rows, std::uint32_t n) {
    std::uint32_t bad_rows = 0, max_ulp = 0;
    std::size_t bad = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        bool row_bad = false;
        for (std::uint32_t c = 0; c < n; ++c) {
            const float a = got[std::size_t{r} * n + c], b = ref[std::size_t{first + r} * n + c];
            if (std::memcmp(&a, &b, 4) != 0) {
                ++bad;
                row_bad = true;
                max_ulp = std::max(max_ulp, ulp_diff(a, b));
            }
        }
        bad_rows += row_bad ? 1U : 0U;
    }
    std::printf("  %-34s %s", what.c_str(), bad == 0 ? "bitwise" : "DIFFERS");
    if (bad != 0) std::printf(" (%u/%u rows, %zu values, max %u ulp)", bad_rows, rows, bad, max_ulp);
    std::printf("\n");
    if (bad != 0 && !informative) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s PACK_DIR NAME[@rows:F:C|@cols:F:C] ...\n", argv[0]);
        return 2;
    }
    try {
        TrellisPack pack(argv[1]);
        TrellisContext ctx;
        hipStream_t stream;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        std::vector<const TrellisSite*> sites;
        std::vector<std::string> labels;
        for (int a = 2; a < argc; ++a) {
            std::string arg = argv[a], name = arg;
            TrellisSlice slice;
            if (const auto at = arg.find('@'); at != std::string::npos) {
                name = arg.substr(0, at);
                unsigned f = 0, c = 0;
                char kind[8] = {};
                if (std::sscanf(arg.c_str() + at + 1, "%4[a-z]:%u:%u", kind, &f, &c) != 3) throw std::invalid_argument("bad slice " + arg);
                slice.kind = std::string(kind) == "rows" ? TrellisSlice::Kind::rows : TrellisSlice::Kind::columns;
                slice.first = f;
                slice.count = c;
            }
            const auto t0 = std::chrono::steady_clock::now();
            sites.push_back(&ctx.prepare(pack, name, slice));
            labels.push_back(arg);
            std::printf("prepared %s: N %u x I %u, K %u, %.1f MB, %.2f s\n", arg.c_str(), sites.back()->rows, sites.back()->columns,
                        sites.back()->bits, static_cast<double>(sites.back()->device_bytes) / 1e6,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        constexpr std::uint32_t max_rows = 32;
        ctx.reserve(stream, max_rows);
        for (std::size_t si = 0; si < sites.size(); ++si) {
            const auto& s = *sites[si];
            const std::uint32_t I = s.columns, N = s.rows;
            std::printf("%s (N %u, I %u, K %u)\n", labels[si].c_str(), N, I, s.bits);
            const auto x = make_x(max_rows + 8U, I, 0x1234567ULL + si);
            DevBuf dx(x.size() * 4U), dy(std::size_t{max_rows + 8U} * N * 4U), dws(64ULL << 20U);
            check(hipMemcpy(dx.p, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "x upload");
            // direct library calls on the site's weights (they belong to the TrellisContext's library context)
            const auto call = [&](bool gemm, std::uint32_t first, std::uint32_t rows) {
                strix_matmul_desc d{};
                strix_matmul_desc_init(&d);
                d.weights = static_cast<strix_weights>(s.weights);
                set_tensor(d.input, dx.f() + std::size_t{first} * I, rows, I);
                set_tensor(d.output, dy.p, rows, N);
                std::uint64_t ws = 0;
                const auto st = lib_stream(stream);
                const auto c = static_cast<strix_context>(ctx.library_context());
                if (strix_matmul_workspace_bytes(c, &d, &ws) != STRIX_SUCCESS || ws > (64ULL << 20U)) throw std::runtime_error("workspace");
                d.workspace = dws.p;
                d.workspace_bytes = 64ULL << 20U;
                const auto r = gemm ? strix_gemm(c, &d, &st) : strix_gemv(c, &d, &st);
                if (r != STRIX_SUCCESS) throw std::runtime_error(std::string("library call: ") + strix_status_string(r));
                std::vector<float> out(std::size_t{rows} * N);
                check(hipMemcpyAsync(out.data(), dy.p, out.size() * 4U, hipMemcpyDeviceToHost, stream), "y download");
                check(hipStreamSynchronize(stream), "sync");
                return out;
            };
            // reference: every row alone (GEMV R=1)
            std::vector<float> ref(std::size_t{max_rows + 8U} * N);
            for (std::uint32_t r = 0; r < max_rows + 8U; ++r) {
                const auto y = call(false, r, 1);
                std::copy(y.begin(), y.end(), ref.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * N));
            }
            informative = true;
            {
                const auto again = call(false, 5, 1);
                compare("gemv R1 repeat", again, ref, 5, 1, N);
            }
            for (std::uint32_t R = 2; R <= 8; ++R) {
                compare("gemv R" + std::to_string(R) + " @0", call(false, 0, R), ref, 0, R, N);
                compare("gemv R" + std::to_string(R) + " @3", call(false, 3, R), ref, 3, R, N);
            }
            for (std::uint32_t R : {1U, 2U, 6U, 8U, 9U, 16U, 32U}) compare("gemm R" + std::to_string(R), call(true, 0, R), ref, 0, R, N);
            // R classes: refR[r] = row 0 of a GEMV of R rows starting at row r (neighbours r+1.. differ per offset);
            // equal[R][R'] iff refR == refR' on every row; cross-row independence: GEMV R @0 row j == refR[j]
            {
                constexpr std::uint32_t rows_c = 24;
                std::vector<std::vector<float>> refR(9);
                bool independent = true;
                for (std::uint32_t R = 1; R <= 8; ++R) {
                    refR[R].resize(std::size_t{rows_c} * N);
                    for (std::uint32_t r = 0; r < rows_c; ++r) {
                        const auto y = call(false, r, R);
                        std::copy(y.begin(), y.begin() + N, refR[R].begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * N));
                    }
                    const auto y0 = call(false, 0, R);
                    for (std::uint32_t j = 1; j < R; ++j)
                        if (std::memcmp(y0.data() + std::size_t{j} * N, refR[R].data() + std::size_t{j} * N, std::size_t{N} * 4U) != 0) independent = false;
                }
                std::printf("  R classes (row-independent within a call: %s):", independent ? "yes" : "NO");
                std::vector<int> cls(9, 0);
                int next = 0;
                for (std::uint32_t R = 1; R <= 8; ++R) {
                    for (std::uint32_t Q = 1; Q < R; ++Q)
                        if (std::memcmp(refR[Q].data(), refR[R].data(), refR[R].size() * 4U) == 0) {
                            cls[R] = cls[Q];
                            break;
                        }
                    if (cls[R] == 0) cls[R] = ++next;
                    std::printf(" R%u:%c", R, 'A' + cls[R] - 1);
                }
                std::printf("\n");
            }
            informative = false;
            // the engine entry point (the identity): every row alone through enqueue_trellis_linear is the reference of
            // windows / batches of any R (chunks, padded tails), strided rows and accumulate
            for (std::uint32_t r = 0; r < max_rows + 8U; ++r) {
                enqueue_trellis_linear(s, dx.f() + std::size_t{r} * I, dy.f(), 1, I, N, false, stream);
                check(hipMemcpyAsync(ref.data() + std::size_t{r} * N, dy.p, std::size_t{N} * 4U, hipMemcpyDeviceToHost, stream), "y download");
            }
            check(hipStreamSynchronize(stream), "sync");
            std::printf("  entry point: sizes 0x%02x, chunk %u\n", trellis_submit_sizes(s), trellis_chunk_rows(s));
            for (std::uint32_t R : {2U, 3U, 4U, 5U, 6U, 7U, 8U, 13U, 32U}) {
                check(hipMemsetAsync(dy.p, 0, std::size_t{R} * N * 4U, stream), "memset");
                enqueue_trellis_linear(s, dx.f(), dy.f(), R, I, N, false, stream);
                std::vector<float> out(std::size_t{R} * N);
                check(hipMemcpyAsync(out.data(), dy.p, out.size() * 4U, hipMemcpyDeviceToHost, stream), "y download");
                check(hipStreamSynchronize(stream), "sync");
                compare("linear R" + std::to_string(R), out, ref, 0, R, N);
            }
            {
                // strided: x rows 2 apart (every other row), y rows with a stride of N + 128; accumulate onto 0.5
                constexpr std::uint32_t R = 6;
                const std::size_t ys = std::size_t{N} + 128U;
                DevBuf dys(std::size_t{R} * ys * 4U);
                std::vector<float> half(std::size_t{R} * ys, 0.5F);
                check(hipMemcpy(dys.p, half.data(), half.size() * 4U, hipMemcpyHostToDevice), "half upload");
                enqueue_trellis_linear(s, dx.f(), dys.f(), R, std::size_t{2U} * I, ys, true, stream);
                std::vector<float> out(std::size_t{R} * ys);
                check(hipMemcpyAsync(out.data(), dys.p, out.size() * 4U, hipMemcpyDeviceToHost, stream), "y download");
                check(hipStreamSynchronize(stream), "sync");
                std::vector<float> got(std::size_t{R} * N), want(std::size_t{R} * N);
                for (std::uint32_t r = 0; r < R; ++r)
                    for (std::uint32_t c = 0; c < N; ++c) {
                        got[std::size_t{r} * N + c] = out[r * ys + c];
                        want[std::size_t{r} * N + c] = 0.5F + ref[std::size_t{2U * r} * N + c];
                    }
                compare("linear R6 strided x/y + accumulate", got, want, 0, R, N);
            }
            // timing: R1 and R6 GEMV through the entry point (warm, weights from DRAM: tensor >> L2 for the big sites)
            hipEvent_t e0, e1;
            check(hipEventCreate(&e0), "event");
            check(hipEventCreate(&e1), "event");
            for (std::uint32_t R : {1U, 6U, 8U}) {
                for (int w = 0; w < 3; ++w) enqueue_trellis_linear(s, dx.f(), dy.f(), R, I, N, false, stream);
                constexpr int reps = 20;
                check(hipEventRecord(e0, stream), "record");
                for (int k = 0; k < reps; ++k) enqueue_trellis_linear(s, dx.f(), dy.f(), R, I, N, false, stream);
                check(hipEventRecord(e1, stream), "record");
                check(hipEventSynchronize(e1), "sync");
                float ms = 0;
                check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                const double us = 1000.0 * ms / reps;
                std::printf("  time R%u: %.1f us, %.1f GB/s (weights)\n", R, us, static_cast<double>(s.device_bytes) / us / 1e3);
            }
            (void)hipEventDestroy(e0);
            (void)hipEventDestroy(e1);
        }
        std::printf("%s\n", failures == 0 ? "ALL BITWISE" : "FAILURES");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
