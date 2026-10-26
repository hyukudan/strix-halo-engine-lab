// Exact tiled DOT2 gate against the linked rows=1 library, plus reuse timing.
#include "strix_trellis.h"
#include "glm/prefill/glm_pf_exact.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>

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
    {"layers.4.self_attn.qkv_proj", "KDA qkv k-part r0 K6", 6, 4096, 24576, 'n', 8192, 12288},
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
    unsigned N = 0, I = 0, bits = 0;
};
Lin prepare(Ctx& cx, const std::string& dir, const Shape& sh, bool synthetic) {
    const std::string stem = dir + "/" + sh.name;
    std::vector<char> tr, suh, svh;
    if (synthetic) {
        std::mt19937 gen(711U + sh.bits + sh.I + sh.N + sh.c0);
        tr.resize(std::size_t{sh.I} * sh.N / 8U * sh.bits);
        suh.resize(std::size_t{sh.I} * 2U); svh.resize(std::size_t{sh.N} * 2U);
        for (std::size_t q = 0; q < tr.size(); q += 4U) {
            const std::uint32_t v = static_cast<std::uint32_t>(gen());
            std::memcpy(tr.data() + q, &v, 4);
        }
        for (auto* a : {&suh, &svh})
            for (std::size_t q = 0; q < a->size(); q += 2U) {
                const auto v = static_cast<std::uint16_t>((gen() & 1U ? 0x8000U : 0U) | (0x3800U + gen() % 2048U));
                std::memcpy(a->data() + q, &v, 2);
            }
    } else {
        tr = slurp(stem + ".trellis"); suh = slurp(stem + ".suh"); svh = slurp(stem + ".svh");
    }
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
    l.bits = sh.bits;
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


namespace E = strix::glm::pf::exact;
using Linear = strix::glm::engine::Linear;
void tm_env(unsigned tm) {
    const std::string v = std::to_string(tm);
#ifdef _WIN32
    if (_putenv_s("STRIX_GLM_PF_EXACT_TM", v.c_str()) != 0) die("set TM");
#else
    if (setenv("STRIX_GLM_PF_EXACT_TM", v.c_str(), 1) != 0) die("set TM");
#endif
}
std::uint32_t bits(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}
float from_bits(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
std::uint32_t ordered(std::uint32_t u) { return u & 0x80000000U ? ~u : (u ^ 0x80000000U); }
struct Allocation {
    void* p = nullptr;
    explicit Allocation(std::size_t bytes) { hok(hipMalloc(&p, bytes), "alloc"); }
    ~Allocation() { if (p) (void)hipFree(p); }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    float* f() const { return static_cast<float*>(p); }
};
Linear engine_linear(const Lin& l, bool raw) {
    Linear e;
    e.prepared = l.w;
    e.raw.input = l.I; e.raw.output = l.N; e.raw.bits = l.bits;
    if (raw) {
        const void* data = nullptr;
        std::uint64_t bytes = 0;
        sok(strix_weights_device_span(l.w, &data, &bytes), "span");
        const auto* b = static_cast<const unsigned char*>(data);
        const auto so = align256(std::uint64_t{l.I} * l.N / 8U * l.bits);
        const auto vo = align256(so + std::uint64_t{l.I} * 2U);
        e.raw.trellis = reinterpret_cast<const std::uint32_t*>(b);
        e.raw.suh = reinterpret_cast<const std::uint16_t*>(b + so);
        e.raw.svh = reinterpret_cast<const std::uint16_t*>(b + vo);
        e.prepared = nullptr;
    }
    return e;
}
void test_shape(Ctx& cx, const std::string& dir, const Shape& sh, bool synthetic, bool timing) {
    const Lin l = prepare(cx, dir, sh, synthetic);
    const Linear prepared = engine_linear(l, false), raw = engine_linear(l, true);
    constexpr unsigned Mmax = 2048, extra = 16;
    const std::size_t xs = l.I + 19U, ys = l.N + 17U;
    Allocation xb((Mmax + extra) * xs * 4U), yb((Mmax + extra) * ys * 4U), rb(std::size_t{Mmax} * l.N * 4U);
    const std::string tag = std::string(synthetic ? "random " : "fixture ") + sh.kind + " " +
        std::to_string(l.I) + "->" + std::to_string(l.N);
    if (!timing) check(E::supported(prepared) && E::supported(raw), "supported " + tag);
    if (!E::supported(prepared) || !E::supported(raw)) die("unsupported fixture");
    const unsigned splits = E::vector_splits(l.I, l.N, l.bits);
    if (!timing) check(library_splits(cx, l, 1, xb.f(), yb.f()) == splits, "R1 splits=" + std::to_string(splits) + " " + tag);
    std::uint64_t library_ws = 0;
    for (unsigned r : {1U, 8U}) {
        auto q = desc(l, xb.f(), yb.f(), r, nullptr, 0);
        std::uint64_t b = 0;
        sok(strix_matmul_workspace_bytes(cx.c, &q, &b), "ws");
        library_ws = std::max(library_ws, b);
    }
    Allocation ws(library_ws), ew(E::workspace_bytes(Mmax, l.I));
    std::mt19937 g(812U + l.I + l.N);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> input(std::size_t{Mmax} * l.I);
    for (float& v : input) v = nd(g);
    for (unsigned r = 0; r < Mmax; ++r) {
        float* row = input.data() + std::size_t{r} * l.I;
        row[0] = from_bits(0x80000000U); row[1] = 0.0F;
        row[2] = from_bits(1); row[3] = from_bits(0x80000001U);
        row[4] = 8192.0F; row[5] = -8192.0F;
        if (r % 127U == 0)
            for (unsigned c = 0; c < l.I; ++c) row[c] = from_bits(c & 1U ? 0x80000000U : 0U);
        if (r % 127U == 1)
            for (unsigned c = 0; c < l.I; ++c) row[c] = from_bits(c & 1U ? 0x80000001U : 1U);
    }
    auto reference = [&](float* x, float* y, unsigned m, unsigned block) {
        for (unsigned r = 0; r < m; r += block) {
            const unsigned nr = std::min(block, m - r);
            auto q = desc(l, x + std::size_t{r} * l.I, y + std::size_t{r} * l.N, nr, ws.p, library_ws);
            sok(strix_gemv(cx.c, &q, &cx.st), "gemv");
        }
    };
    if (!timing) {
        for (unsigned tm : {16U, 32U, 64U}) {
            const auto scratch = E::scratch_bytes(prepared, tm);
            std::printf("RESOURCE %s TM=%u private=%zu B/lane\n", tag.c_str(), tm, scratch);
            check(scratch == 0, "zero spills TM=" + std::to_string(tm) + " " + tag);
        }
        Linear bad = raw; bad.raw.codebook = 0;
        check(!E::supported(bad), "reject non-mul1 " + tag);
        bad = raw; --bad.raw.input;
        check(!E::supported(bad), "reject unaligned shape " + tag);
        hok(hipMemcpyAsync(xb.p, input.data(), input.size() * 4U, hipMemcpyHostToDevice, cx.hs), "input");
        reference(xb.f(), rb.f(), Mmax, 1);
        const auto ref = down(rb.f(), std::size_t{Mmax} * l.N);
        const float poison = from_bits(0xffffffffU);
        std::vector<float> staged((Mmax + extra) * xs, poison);
        for (unsigned m : {1U,2U,7U,8U,9U,15U,16U,17U,31U,32U,33U,63U,64U,65U,100U,1024U,2048U}) {
            const unsigned source = (m * 37U + 11U) % (Mmax - m + 1U);
            const unsigned xo = 1U + m % 11U, yo = 1U + m % 7U;
            std::fill(staged.begin(), staged.end(), poison);
            for (unsigned r = 0; r < m; ++r)
                std::copy_n(input.data() + std::size_t{source + r} * l.I, l.I, staged.data() + (xo + r) * xs + 3U);
            hok(hipMemcpyAsync(xb.p, staged.data(), staged.size() * 4U, hipMemcpyHostToDevice, cx.hs), "padded input");
            for (bool raw_form : {false, true})
                for (unsigned tm : {16U,32U,64U}) {
                    tm_env(tm);
                    hok(hipMemsetAsync(yb.p, 0xff, (Mmax + extra) * ys * 4U, cx.hs), "poison output");
                    hok(hipMemsetAsync(ew.p, 0xff, E::workspace_bytes(Mmax, l.I), cx.hs), "poison workspace");
                    E::linear_rows(raw_form ? raw : prepared, xb.f() + xo * xs + 3U, xs,
                                   yb.f() + yo * ys + 5U, ys, m, ew.p, E::workspace_bytes(Mmax, l.I), cx.hs);
                    const auto got = down(yb.f(), (Mmax + extra) * ys);
                    bool same = true, guards = true;
                    std::uint32_t maxulp = 0;
                    std::size_t first = 0;
                    std::uint32_t first_got = 0, first_ref = 0;
                    for (std::size_t q = 0; q < got.size(); ++q) {
                        const std::size_t row = q / ys, c = q % ys;
                        if (row >= yo && row < yo + m && c >= 5U && c < 5U + l.N) {
                            const std::uint32_t a = bits(got[q]), b = bits(ref[(source + row - yo) * l.N + c - 5U]);
                            if (a != b) {
                                if (same) { first = (row - yo) * l.N + c - 5U; first_got = a; first_ref = b; }
                                same = false;
                                const auto oa = ordered(a), ob = ordered(b);
                                maxulp = std::max(maxulp, oa > ob ? oa - ob : ob - oa);
                            }
                        } else if (bits(got[q]) != 0xffffffffU) guards = false;
                    }
                    const std::string suffix = " M=" + std::to_string(m) + " TM=" + std::to_string(tm) +
                        (raw_form ? " raw " : " prepared ") + tag;
                    if (!same) std::printf("MISMATCH first=%zu got=%08x ref=%08x max_ulp=%u\n", first, first_got, first_ref, maxulp);
                    check(same && guards, "bitwise+guards" + suffix);
                }
        }
    } else {
        hok(hipMemcpyAsync(xb.p, input.data(), input.size() * 4U, hipMemcpyHostToDevice, cx.hs), "input");
        hipEvent_t a{}, b{};
        hok(hipEventCreate(&a), "event"); hok(hipEventCreate(&b), "event");
        for (unsigned tm : {16U,32U,64U}) {
            tm_env(tm);
            auto exact = [&] { E::linear_rows(prepared, xb.f(), l.I, yb.f(), l.N, Mmax,
                                             ew.p, E::workspace_bytes(Mmax, l.I), cx.hs); };
            exact(); reference(xb.f(), yb.f(), Mmax, 8);
            float ms[2][3]{};
            for (unsigned rep = 0; rep < 3; ++rep)
                for (unsigned arm = 0; arm < 2; ++arm) {
                    hok(hipEventRecord(a, cx.hs), "event");
                    if (arm == 0) exact(); else reference(xb.f(), yb.f(), Mmax, 8);
                    hok(hipEventRecord(b, cx.hs), "event"); hok(hipEventSynchronize(b), "event");
                    hok(hipEventElapsedTime(&ms[arm][rep], a, b), "event");
                }
            for (unsigned arm = 0; arm < 2; ++arm) {
                std::sort(ms[arm], ms[arm] + 3);
                const double seconds = ms[arm][1] * 1e-3;
                const unsigned reuse = arm == 0 ? tm : 8U;
                const double weights = static_cast<double>(l.I) * l.N * l.bits / 8.0 * ((Mmax + reuse - 1U) / reuse);
                const double flop = 2.0 * Mmax * l.I * l.N;
                std::printf("TIME %s TM=%u %s M=2048 %.4f us/row weight-GB/s=%.3f dot2-TFLOP/s=%.3f\n",
                            tag.c_str(), tm, arm == 0 ? "exact" : "gemv8", seconds * 1e6 / Mmax,
                            weights / seconds / 1e9, flop / seconds / 1e12);
            }
        }
        hok(hipEventDestroy(a), "event"); hok(hipEventDestroy(b), "event");
    }
    hok(hipStreamSynchronize(cx.hs), "finish");
    sok(strix_weights_destroy(l.w), "destroy");
}
}  // namespace

int main(int argc, char** argv) {
    std::string dir;
    bool timing = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-timing") timing = false;
        else if (dir.empty()) dir = argv[i];
        else { std::fprintf(stderr, "usage: %s [TRELLIS_REAL_DIR] [--no-timing]\n", argv[0]); return 2; }
    }
    if (dir.empty()) if (const char* p = std::getenv("TRELLIS_REAL_DIR")) dir = p;
    try {
        Ctx cx;
        strix_context_desc cd{};
        cd.struct_size = sizeof(cd); cd.abi_version = STRIX_TRELLIS_ABI_VERSION; cd.device_ordinal = 0;
        sok(strix_context_create(&cd, &cx.c), "context");
        hok(hipStreamCreate(&cx.hs), "stream");
        cx.st.struct_size = sizeof(cx.st); cx.st.abi_version = STRIX_TRELLIS_ABI_VERSION; cx.st.hip_stream = cx.hs;
        for (const auto& sh : kShapes) test_shape(cx, dir, sh, true, false);
        // Small generic cases exercise every decoder and splits that cut H128.
        for (unsigned bits = 2; bits <= 8; ++bits) {
            const Shape sh{"generic", "generic", bits, 384, 128, 0, 0, 0};
            test_shape(cx, dir, sh, true, false);
        }
        // Original-input fusion's H128 partition differs from the staged K16 policy.
        for (const Shape sh : {Shape{"fused", "fused input", 5, 5120, 2304, 0, 0, 0},
                               Shape{"fused", "fused both", 5, 2304, 5120, 0, 0, 0},
                               Shape{"policy", "K5 split exception", 5, 4096, 1024, 0, 0, 0},
                               Shape{"fused", "fused K6", 6, 5120, 1280, 0, 0, 0},
                               Shape{"fused", "fused wide", 5, 8192, 5120, 0, 0, 0},
                               Shape{"pipeline", "K2 wide", 2, 8192, 5120, 0, 0, 0},
                               Shape{"pipeline", "K2 head", 2, 1280, 32768, 0, 0, 0}})
            test_shape(cx, dir, sh, true, false);
        if (!dir.empty()) for (const auto& sh : kShapes) test_shape(cx, dir, sh, false, false);
        else std::printf("SKIP real fixtures: set TRELLIS_REAL_DIR or pass its path\n");
        if (timing && failures == 0) {
            for (const auto& sh : kShapes) test_shape(cx, dir, sh, true, true);
            if (!dir.empty()) for (const auto& sh : kShapes) test_shape(cx, dir, sh, false, true);
        } else if (timing) std::printf("timing skipped: gate failed\n");
        std::printf("checks: %d failures: %d\n", checks, failures);
        hok(hipStreamDestroy(cx.hs), "stream destroy");
        sok(strix_context_destroy(cx.c), "context destroy");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\nchecks: %d failures: %d\n", e.what(), checks, failures + 1);
        return 1;
    }
    return failures ? 1 : 0;
}
