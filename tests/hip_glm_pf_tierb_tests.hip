// Gate PF-B: the pinned-split WMMA contract of glm-pf-wmma-v1 (design notes section 1.1) on the
// real GLM EXL3 trellis fixtures (L0 dense K5, L3 experts K4 / shared + MLA + indexer K6, L4 KDA K6, TP2 slices).
// Per tensor, with s = the library's own split at M = 1024 (frozen):
//   * invariance: strix_gemm_pinned on rows [off, off + M) == rows [off, off + M) of the M = 2048 call, bit for bit,
//     for M in {1, 2, 7, 8, 9, 15, 16, 17, 64, 127, 128, 129, 1000, 1024, 1025} (every row moves to another position
//     of the call and is surrounded by other rows);
//   * at M = 1024 the pinned call == plain strix_gemm (same split: the pinned table is the library's operating point);
//   * recorded (not gated): deviation of the pinned WMMA output vs strix_gemv R = 1 (max ulp, rel-L2);
//   * timing (informational, only after every shape passed): pinned vs default strix_gemm at M = 1024 / 2048 / 4096, TOPS,
//     interleaved A B A B A B, median of 3.
// usage: hip_glm_pf_tierb_tests TRELLIS_REAL_DIR [--no-timing]
#include "strix_trellis.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
void sok(strix_status s, const std::string& w) {
    if (s != STRIX_SUCCESS) {
        std::fprintf(stderr, "%s: %s\n", w.c_str(), strix_status_string(s));
        std::exit(3);
    }
}
[[noreturn]] void die(const std::string& w) {
    std::fprintf(stderr, "FATAL %s\n", w.c_str());
    std::exit(4);
}
std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) die("missing " + p);
    std::vector<char> b(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    f.read(b.data(), static_cast<std::streamsize>(b.size()));
    if (static_cast<std::size_t>(f.gcount()) != b.size()) die("short read " + p);
    return b;
}
bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

struct Shape {
    const char* name;  // fixture stem
    const char* kind;
    unsigned bits;
    unsigned I, N;      // full tensor geometry (asserted against the files)
    char cut;           // 0 = full; 'n' = output rows [n0, n1); 'k' = input columns [k0, k1) (TP2 rank slice)
    unsigned c0, c1;
};
// Fixture stems (trellis-real/metadata.json); N / I come from svh / suh sizes.
// Rank slices are cut from the full tensors on whole 16-tiles / 128-blocks (the TP2 split is bit-identical to such
// cuts: design notes); r0/r1 = the two halves.
const Shape kShapes[] = {
    {"layers.0.mlp.gate_proj", "dense gate r0 K5", 5, 4096, 12288, 'n', 0, 6144},
    {"layers.0.mlp.gate_proj", "dense gate r1 K5", 5, 4096, 12288, 'n', 6144, 12288},
    {"layers.0.mlp.down_proj", "dense down r0 K5", 5, 12288, 4096, 'k', 0, 6144},
    {"layers.0.mlp.down_proj", "dense down r1 K5", 5, 12288, 4096, 'k', 6144, 12288},
    {"layers.3.mlp.experts.0.gate_proj", "expert gate r0 K4", 4, 4096, 2048, 'n', 0, 1024},
    {"layers.3.mlp.experts.287.up_proj", "expert up r1 K4", 4, 4096, 2048, 'n', 1024, 2048},
    {"layers.3.mlp.experts.0.down_proj", "expert down r0 K4", 4, 2048, 4096, 'k', 0, 1024},
    {"layers.3.mlp.experts.287.down_proj", "expert down r1 K4", 4, 2048, 4096, 'k', 1024, 2048},
    {"layers.3.mlp.shared_experts.up_proj", "shared up r0 K6", 6, 4096, 2048, 'n', 0, 1024},
    {"layers.3.mlp.shared_experts.down_proj", "shared down r1 K6", 6, 2048, 4096, 'k', 1024, 2048},
    {"layers.3.self_attn.q_a_proj", "MLA q_a (replicated) K6", 6, 4096, 1536, 0, 0, 0},
    {"layers.3.self_attn.kv_a_proj_with_mqa", "MLA kv_a (replicated) K6", 6, 4096, 512, 0, 0, 0},
    {"layers.3.self_attn.q_b_proj", "MLA q_b r0 K6", 6, 1536, 16384, 'n', 0, 8192},
    {"layers.3.self_attn.q_b_proj", "MLA q_b r1 K6", 6, 1536, 16384, 'n', 8192, 16384},
    {"layers.3.self_attn.indexer.wq_b", "indexer wq_b (replicated) K6", 6, 1536, 4096, 0, 0, 0},
    {"layers.3.self_attn.o_proj_r0", "MLA o_proj r0 (file slice) K6", 6, 8192, 4096, 0, 0, 0},
    {"layers.3.self_attn.o_proj_r1", "MLA o_proj r1 (file slice) K6", 6, 8192, 4096, 0, 0, 0},
    {"layers.4.self_attn.qkv_proj", "KDA qkv q-part r0 K6", 6, 4096, 24576, 'n', 0, 4096},
    {"layers.4.self_attn.qkv_proj", "KDA qkv v-part r1 K6", 6, 4096, 24576, 'n', 20480, 24576},
    {"layers.4.self_attn.q_r0", "KDA q r0 (file slice) K6", 6, 4096, 4096, 0, 0, 0},
    {"layers.4.self_attn.o_proj", "KDA o_proj r1 K6", 6, 8192, 4096, 'k', 4096, 8192},
};

struct Ctx {
    strix_context c{};
    strix_stream st{};
    hipStream_t hs{};
};

struct Lin {
    strix_weights w{};
    unsigned N = 0, I = 0;
};
Lin prepare(Ctx& cx, const std::string& dir, const Shape& sh) {
    const std::string stem = dir + "/" + sh.name;
    auto tr = slurp(stem + ".trellis"), suh = slurp(stem + ".suh"), svh = slurp(stem + ".svh");
    const std::size_t tile = std::size_t{16} * sh.bits * 2;  // bytes of one 16x16 trellis tile
    if (suh.size() != std::size_t{2} * sh.I || svh.size() != std::size_t{2} * sh.N ||
        tr.size() != static_cast<std::size_t>(sh.I / 16) * (sh.N / 16) * tile)
        die(std::string("geometry mismatch: ") + sh.name);
    if (sh.cut == 'n') {  // output rows [c0, c1): tile columns
        std::vector<char> t2;
        for (unsigned kt = 0; kt < sh.I / 16; ++kt)
            t2.insert(t2.end(), tr.begin() + (static_cast<std::size_t>(kt) * (sh.N / 16) + sh.c0 / 16) * tile,
                      tr.begin() + (static_cast<std::size_t>(kt) * (sh.N / 16) + sh.c1 / 16) * tile);
        tr = t2;
        svh = std::vector<char>(svh.begin() + 2 * sh.c0, svh.begin() + 2 * sh.c1);
    } else if (sh.cut == 'k') {  // input columns [c0, c1): tile rows, suh slice
        tr = std::vector<char>(tr.begin() + static_cast<std::size_t>(sh.c0 / 16) * (sh.N / 16) * tile, tr.begin() + static_cast<std::size_t>(sh.c1 / 16) * (sh.N / 16) * tile);
        suh = std::vector<char>(suh.begin() + 2 * sh.c0, suh.begin() + 2 * sh.c1);
    }
    strix_exl3_parts_desc p{};
    p.struct_size = sizeof(p);
    p.abi_version = STRIX_TRELLIS_ABI_VERSION;
    p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    p.bits_x2 = 2 * sh.bits;
    p.rows = svh.size() / 2;
    p.columns = suh.size() / 2;
    p.codebook = STRIX_CODEBOOK_MUL1;
    p.marker = STRIX_EXL3_MARKER_MUL1;
    p.trellis = {tr.data(), tr.size()};
    p.suh = {suh.data(), suh.size()};
    p.svh = {svh.data(), svh.size()};
    Lin l;
    sok(strix_weights_prepare_parts(cx.c, &p, &cx.st, &l.w), std::string("prepare ") + sh.name);
    l.N = static_cast<unsigned>(p.rows);
    l.I = static_cast<unsigned>(p.columns);
    return l;
}

strix_matmul_desc desc(const Lin& l, float* x, float* y, unsigned rows, void* ws, std::uint64_t ws_bytes) {
    strix_matmul_desc d;
    strix_matmul_desc_init(&d);
    d.weights = l.w;
    d.input.data = x;
    d.input.rank = 2;
    d.input.shape[0] = rows;
    d.input.shape[1] = l.I;
    d.input.strides[0] = l.I;
    d.input.strides[1] = 1;
    d.input.buffer_bytes = static_cast<std::uint64_t>(rows) * l.I * 4;
    d.output.data = y;
    d.output.rank = 2;
    d.output.shape[0] = rows;
    d.output.shape[1] = l.N;
    d.output.strides[0] = l.N;
    d.output.strides[1] = 1;
    d.output.buffer_bytes = static_cast<std::uint64_t>(rows) * l.N * 4;
    d.workspace = ws;
    d.workspace_bytes = ws_bytes;
    return d;
}

std::uint64_t align256(std::uint64_t v) { return (v + 255) & ~std::uint64_t{255}; }

// The library's split at M rows, read back from its workspace size: align256(M * I * 2) + M * N * 4 * splits.
unsigned library_splits(Ctx& cx, const Lin& l, unsigned M, float* x, float* y) {
    auto d = desc(l, x, y, M, nullptr, 0);
    std::uint64_t ws = 0;
    sok(strix_matmul_workspace_bytes(cx.c, &d, &ws), "workspace");
    const std::uint64_t part = ws - align256(static_cast<std::uint64_t>(M) * l.I * 2);
    return static_cast<unsigned>(part / (static_cast<std::uint64_t>(M) * l.N * 4));
}

std::vector<float> down(const float* p, std::size_t n) {
    std::vector<float> h(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(h.data(), p, n * 4, hipMemcpyDeviceToHost), "d2h");
    return h;
}

void test_shape(Ctx& cx, const std::string& dir, const Shape& sh, bool timing, std::mt19937& g) {
    // timing == false: the correctness phase; timing == true: the timing phase only (run after a clean gate)
    Lin l = prepare(cx, dir, sh);
    constexpr unsigned MF = 2048, MT = 4096;
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> xh(static_cast<std::size_t>(MT) * l.I);
    for (auto& v : xh) v = nd(g);
    float *x = nullptr, *y = nullptr, *yr = nullptr;
    hok(hipMalloc(&x, xh.size() * 4), "x");
    hok(hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice), "x");
    hok(hipMalloc(&y, static_cast<std::size_t>(MT) * l.N * 4), "y");
    hok(hipMalloc(&yr, static_cast<std::size_t>(MF) * l.N * 4), "yr");
    const unsigned s = library_splits(cx, l, 1024, x, y);
    std::uint64_t ws_bytes = 0;
    {
        auto d = desc(l, x, y, MT, nullptr, 0);
        std::uint64_t a = 0, b = 0;
        sok(strix_gemm_pinned_workspace_bytes(cx.c, &d, s, &a), "pinned ws");
        sok(strix_matmul_workspace_bytes(cx.c, &d, &b), "ws");
        ws_bytes = a > b ? a : b;
    }
    void* ws = nullptr;
    hok(hipMalloc(&ws, ws_bytes), "ws");
    auto pinned = [&](unsigned off, unsigned M, float* out) {
        auto d = desc(l, x + static_cast<std::size_t>(off) * l.I, out, M, ws, ws_bytes);
        sok(strix_gemm_pinned(cx.c, &d, s, &cx.st), "gemm_pinned");
    };
    const std::string tag = std::string(sh.kind) + " " + std::to_string(l.I) + "->" + std::to_string(l.N) + " s=" + std::to_string(s);
    if (!timing) {
    // reference: the M = 2048 call
    hok(hipMemset(yr, 0xFF, static_cast<std::size_t>(MF) * l.N * 4), "poison");
    pinned(0, MF, yr);
    const auto ref = down(yr, static_cast<std::size_t>(MF) * l.N);
    check(all_finite(ref), "PF-B reference (M 2048) finite: " + tag);
    bool inv = true;
    std::string bad;
    for (unsigned M : {1U, 2U, 7U, 8U, 9U, 15U, 16U, 17U, 64U, 127U, 128U, 129U, 1000U, 1024U, 1025U}) {
        const unsigned off = (M * 37U + 11U) % (MF - M + 1);
        hok(hipMemset(y, 0xFF, static_cast<std::size_t>(M) * l.N * 4), "poison");
        pinned(off, M, y);
        const auto got = down(y, static_cast<std::size_t>(M) * l.N);
        if (!all_finite(got) || std::memcmp(got.data(), ref.data() + static_cast<std::size_t>(off) * l.N, got.size() * 4) != 0) {
            inv = false;
            bad += " M=" + std::to_string(M);
        }
    }
    check(inv, "PF-B invariance " + tag + " (M 1..1025 at shifted offsets == M 2048 rows)" + bad);
    {  // pinned == default strix_gemm at M = 1024
        auto d = desc(l, x, y, 1024, ws, ws_bytes);
        sok(strix_gemm(cx.c, &d, &cx.st), "gemm");
        const auto got = down(y, static_cast<std::size_t>(1024) * l.N);
        check(all_finite(got) && std::memcmp(got.data(), ref.data(), got.size() * 4) == 0, "PF-B pinned == strix_gemm at M 1024: " + tag);
    }
    {  // deviation vs strix_gemv R = 1 (recorded)
        double se = 0, sr = 0;
        std::uint32_t maxulp = 0;
        for (unsigned r = 0; r < 16; ++r) {
            auto d = desc(l, x + static_cast<std::size_t>(r) * l.I, y, 1, ws, ws_bytes);
            sok(strix_gemv(cx.c, &d, &cx.st), "gemv");
            const auto gv = down(y, l.N);
            if (!all_finite(gv)) {
                check(false, "PF-B strix_gemv R=1 output finite: " + tag);
                break;
            }
            for (unsigned o = 0; o < l.N; ++o) {
                const float a = ref[static_cast<std::size_t>(r) * l.N + o], b = gv[o];
                se += (static_cast<double>(a) - b) * (static_cast<double>(a) - b);
                sr += static_cast<double>(b) * b;
                std::int32_t ia, ib;
                std::memcpy(&ia, &a, 4);
                std::memcpy(&ib, &b, 4);
                if (ia < 0) ia = INT32_MIN - ia;
                if (ib < 0) ib = INT32_MIN - ib;
                const std::uint32_t u = static_cast<std::uint32_t>(ia > ib ? static_cast<std::int64_t>(ia) - ib : static_cast<std::int64_t>(ib) - ia);
                maxulp = u > maxulp ? u : maxulp;
            }
        }
        if (sr > 0) std::printf("DEV %s: pinned WMMA vs strix_gemv R=1 (16 rows): rel-L2 %.3e, max ulp %u\n", tag.c_str(), std::sqrt(se / sr), maxulp);
        else std::printf("DEV %s: gemv reference norm is zero (absolute L2 difference %.3e)\n", tag.c_str(), std::sqrt(se));
    }
    } else {
        hipEvent_t a, b;
        hok(hipEventCreate(&a), "ev");
        hok(hipEventCreate(&b), "ev");
        for (unsigned M : {1024U, 2048U, 4096U}) {
            const unsigned sm = library_splits(cx, l, M, x, y);
            float t[2][3];
            for (int rep = 0; rep < 3; ++rep)
                for (int arm = 0; arm < 2; ++arm) {  // A = pinned, B = default, interleaved x3 (A B A B A B)
                    auto d = desc(l, x, y, M, ws, ws_bytes);
                    if (arm == 0) sok(strix_gemm_pinned(cx.c, &d, s, &cx.st), "warm");
                    else sok(strix_gemm(cx.c, &d, &cx.st), "warm");
                    hok(hipEventRecord(a, cx.hs), "ev");
                    for (int it = 0; it < 10; ++it) {
                        if (arm == 0) sok(strix_gemm_pinned(cx.c, &d, s, &cx.st), "time");
                        else sok(strix_gemm(cx.c, &d, &cx.st), "time");
                    }
                    hok(hipEventRecord(b, cx.hs), "ev");
                    hok(hipEventSynchronize(b), "ev");
                    float ms = 0;
                    hok(hipEventElapsedTime(&ms, a, b), "ev");
                    t[arm][rep] = ms / 10;
                }
            auto med = [](float* v) {
                float x0 = v[0], x1 = v[1], x2 = v[2];
                return std::fmax(std::fmin(x0, x1), std::fmin(std::fmax(x0, x1), x2));
            };
            const double flop = 2.0 * M * l.I * l.N;
            std::printf("TIME %s M=%u: pinned(s=%u) %.3f ms %.1f TOPS | default(s=%u) %.3f ms %.1f TOPS\n", tag.c_str(), M, s, med(t[0]),
                        flop / med(t[0]) / 1e9, sm, med(t[1]), flop / med(t[1]) / 1e9);
        }
        hok(hipEventDestroy(a), "ev");
        hok(hipEventDestroy(b), "ev");
    }
    hok(hipFree(x), "f");
    hok(hipFree(y), "f");
    hok(hipFree(yr), "f");
    hok(hipFree(ws), "f");
    sok(strix_weights_destroy(l.w), "destroy");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s TRELLIS_REAL_DIR [--no-timing]\n", argv[0]);
        return 2;
    }
    const bool timing = !(argc > 2 && std::string(argv[2]) == "--no-timing");
    Ctx cx;
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    cd.device_ordinal = 0;
    sok(strix_context_create(&cd, &cx.c), "context");
    hok(hipStreamCreate(&cx.hs), "stream");
    cx.st.struct_size = sizeof(cx.st);
    cx.st.abi_version = STRIX_TRELLIS_ABI_VERSION;
    cx.st.hip_stream = cx.hs;
    std::mt19937 g(20261026);
    for (const auto& sh : kShapes) test_shape(cx, argv[1], sh, false, g);
    std::printf("checks: %d failures: %d\n", checks, failures);
    if (timing && failures == 0)
        for (const auto& sh : kShapes) test_shape(cx, argv[1], sh, true, g);
    else if (timing) std::printf("timing skipped: gate failed\n");
    return failures ? 1 : 0;
}
