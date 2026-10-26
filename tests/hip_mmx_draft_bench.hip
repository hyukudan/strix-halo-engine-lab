// MiniMax DSpark drafter EXL3 linears (src/minimax/engine/draft_lin.hpp): bitwise gates against an
// independent strix-trellis reference and ABA timing of the library path (A) against the grouped opt path (B).
//
// usage: strix-hip-mmx-draft-bench --exl3 DIR --dspark DIR [--test] [--t1] [--bench] [--linears] [--reps N] [--iters N]
//                                  [--ctx N,...]
//   Reference R: its own strix-trellis context and prepared weights (the 9bc47241 engine hook: Lib::mm per linear,
//   strix_gemv for <= 8 rows, strix_gemm above), independent of DraftLinears.
//   --t1     T1 production DraftLinears (Mode::opt) vs R, bitwise: 35 block matrices x rows 1..8; groups q+k+v, k+v,
//            gate+up x 5 layers x rows 1..8; k / v at 32 rows; fc at rows 1..8 and 32; Mode::load with opt off / on vs R.
//   --test   T1 + T2: the drafter (real sidecar non-linear weights, random taps and anchor rows) with R's per-linear hook
//            (A) and with the production hooks incl. the group hook (B): append + forward_block, y bits A == B at
//            ctx 37 / 300 / 4095, at position 4400 after incremental 7-row appends across the 4096-entry ring, and at
//            ctx 300 on a non-null stream.
//   --bench  per ctx: forward_block A B A B A B A (Mode::load, opt off / on), mean of --iters blocks per arm; then --linears.
//   --linears the 35 linears alone in block order, same ABA order (ms/pass, useful-byte GB/s), then per arm and triplet
//            the per-site sums (q+k+v, o, gate+up, down over 5 layers) from evented passes (diagnostic only).
#include "minimax/engine/draft_lin.hpp"
#include "minimax/engine/mm_common.hpp"
#include "minimax/dspark/hip_minimax_dspark.hpp"

#include <strix_trellis.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace ds = strix::minimax::dspark;
using strix::minimax::engine::DraftLinears;

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void lok(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("trellis " + what + ": " + strix_status_string(s));
}
std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
// Roughly N(0, scale): sum of 4 uniforms.
std::vector<float> random_floats(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) {
        float s = 0.0F;
        for (int k = 0; k < 4; ++k) s += static_cast<float>(splitmix(seed) >> 40) / 16777216.0F - 0.5F;
        x = s * 1.7320508F * scale;
    }
    return v;
}
template <class T> T* dev_copy(const std::vector<T>& v) {
    T* d = nullptr;
    hok(hipMalloc(&d, std::max<std::size_t>(1, v.size()) * sizeof(T)), "malloc");
    hok(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return d;
}
float* dev_zeros(std::size_t n) {
    float* d = nullptr;
    hok(hipMalloc(&d, std::max<std::size_t>(1, n) * sizeof(float)), "malloc");
    hok(hipMemset(d, 0, std::max<std::size_t>(1, n) * sizeof(float)), "memset");
    return d;
}
std::vector<float> host_copy(const float* d, std::size_t n) {
    std::vector<float> v(n);
    hok(hipMemcpy(v.data(), d, n * sizeof(float), hipMemcpyDeviceToHost), "d2h");
    return v;
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b, std::size_t* first = nullptr) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            if (first != nullptr) *first = i;
            return false;
        }
    return true;
}
std::vector<unsigned> parse_list(const std::string& s) {
    std::vector<unsigned> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(static_cast<unsigned>(std::stoul(t)));
    return v;
}

const char* const kKind[8] = {"q", "k", "v", "o", "gate", "up", "down", "fc"};

// The independent reference: the 9bc47241 engine hook (Lib::mm, per linear, null-stream default), own context.
class RefLib {
public:
    explicit RefLib(const std::string& dir) : pack_(dir) {
        strix_context_desc cd{};
        cd.struct_size = sizeof(cd);
        cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
        cd.device_ordinal = 0;
        lok(strix_context_create(&cd, &ctx_), "context");
        static const char* const names[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                             "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
        for (unsigned s = 0; s <= 35; ++s) {
            const std::string p = s == 35 ? std::string("language_model.model.dspark.fc")
                                          : "language_model.model.dspark.layers." + std::to_string(s / 7) + ".decoder_layer." + names[s % 7];
            if (s == 35) {
                try {
                    (void)pack_.info(p + ".trellis");
                } catch (const std::exception&) {
                    break;
                }
                has_fc = true;
            }
            const auto &tr = pack_.info(p + ".trellis"), &suh = pack_.info(p + ".suh"), &svh = pack_.info(p + ".svh");
            std::vector<std::uint8_t> h(tr.bytes + suh.bytes + svh.bytes);
            pack_.read(tr, 0, tr.bytes, h.data());
            pack_.read(suh, 0, suh.bytes, h.data() + tr.bytes);
            pack_.read(svh, 0, svh.bytes, h.data() + tr.bytes + suh.bytes);
            strix_exl3_parts_desc pd{};
            pd.struct_size = sizeof(pd);
            pd.abi_version = STRIX_TRELLIS_ABI_VERSION;
            pd.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
            pd.bits_x2 = static_cast<std::uint32_t>(2 * (tr.shape[2] / 16));
            pd.rows = svh.shape[0];
            pd.columns = suh.shape[0];
            pd.codebook = STRIX_CODEBOOK_MUL1;
            pd.marker = STRIX_EXL3_MARKER_MUL1;
            pd.trellis = {h.data(), tr.bytes};
            pd.suh = {h.data() + tr.bytes, suh.bytes};
            pd.svh = {h.data() + tr.bytes + suh.bytes, svh.bytes};
            strix_stream st{};
            st.struct_size = sizeof(st);
            st.abi_version = STRIX_TRELLIS_ABI_VERSION;
            strix_weights w = nullptr;
            lok(strix_weights_prepare_parts(ctx_, &pd, &st, &w), "prepare " + p);
            hok(hipDeviceSynchronize(), "prepare");
            w_.push_back(w);
            rows_.push_back(svh.shape[0]);
            cols_.push_back(suh.shape[0]);
        }
    }
    ~RefLib() {
        (void)hipDeviceSynchronize();
        for (auto w : w_) (void)strix_weights_destroy(w);
        if (ws_) (void)hipFree(ws_);
        (void)strix_context_destroy(ctx_);
    }
    // kind 7 = fc; Lib::mm with gemv = rows <= 8, workspace grown on demand (synchronising), like the old engine.
    void mm(unsigned layer, unsigned kind, const float* x, float* out, std::uint64_t t, hipStream_t stream) {
        const unsigned s = kind == 7 ? 35 : layer * 7 + kind;
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = w_.at(s);
        set(d.input, const_cast<float*>(x), t, cols_[s]);
        set(d.output, out, t, rows_[s]);
        std::uint64_t bytes = 0;
        lok(strix_matmul_workspace_bytes(ctx_, &d, &bytes), "workspace");
        if (bytes > ws_bytes_) {
            hok(hipDeviceSynchronize(), "sync");
            if (ws_) hok(hipFree(ws_), "free");
            hok(hipMalloc(&ws_, bytes), "workspace");
            ws_bytes_ = bytes;
        }
        d.workspace = bytes ? ws_ : nullptr;
        d.workspace_bytes = bytes ? ws_bytes_ : 0;
        strix_stream st{};
        st.struct_size = sizeof(st);
        st.abi_version = STRIX_TRELLIS_ABI_VERSION;
        st.hip_stream = stream;
        lok(t <= 8 ? strix_gemv(ctx_, &d, &st) : strix_gemm(ctx_, &d, &st), t <= 8 ? "gemv" : "gemm");
    }
    bool has_fc = false;

private:
    static void set(strix_tensor_desc& t, float* p, std::uint64_t rows, std::uint64_t width) {
        t.data = p;
        t.buffer_bytes = rows * width * sizeof(float);
        t.dtype = STRIX_DTYPE_F32;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = width;
        t.strides[0] = width;
        t.strides[1] = 1;
    }
    strix::minimax::engine::detail::PackReader pack_;
    strix_context ctx_ = nullptr;
    std::vector<strix_weights> w_;
    std::vector<std::uint64_t> rows_, cols_;
    void* ws_ = nullptr;
    std::uint64_t ws_bytes_ = 0;
};

// Linear buffers of one block pass (DeviceDrafter::forward_block's call sites).
struct Bufs {
    float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *att = nullptr, *o = nullptr, *g = nullptr, *u = nullptr, *o2 = nullptr;
};

}  // namespace

int main(int argc, char** argv) try {
    std::string exl3, dspark_dir;
    bool t1 = false, t2 = false, bench = false, linears = false;
    unsigned reps = 3, iters = 20;
    std::vector<unsigned> ctxs = {37, 300, 4095}, attn_variants = {2, 3}, attn_ctxs = {300, 1000, 4095};
    bool attn_only = false;  // only the attention arms (profiling)
    unsigned block_attn = 0;  // attention variant of the BLOCK (linears) A/B arms
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--exl3") exl3 = next();
        else if (a == "--dspark") dspark_dir = next();
        else if (a == "--test") t1 = t2 = true;
        else if (a == "--t1") t1 = true;
        else if (a == "--bench") bench = linears = true;
        else if (a == "--linears") linears = true;
        else if (a == "--reps") reps = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--iters") iters = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--ctx") ctxs = parse_list(next());
        else if (a == "--attn-variants") attn_variants = parse_list(next());
        else if (a == "--attn-ctx") attn_ctxs = parse_list(next());
        else if (a == "--attn-only") attn_only = true;
        else if (a == "--block-attn") block_attn = static_cast<unsigned>(std::stoul(next()));
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (exl3.empty() || (dspark_dir.empty() && (t2 || bench || attn_only))) {
        std::fprintf(stderr, "usage: %s --exl3 DIR --dspark DIR [--test] [--t1] [--bench] [--linears] [--reps N] [--iters N] [--ctx N,...]\n",
                     argv[0]);
        return 2;
    }
    const char* lv = std::getenv("STRIX_MMX_LINEAR_V");
    std::printf("draft bench: exl3 %s, STRIX_MMX_LINEAR_V=%s\n", exl3.c_str(), lv ? lv : "(unset: schedule)");
    RefLib ref(exl3);
    DraftLinears prod(exl3, DraftLinears::Mode::opt);
    DraftLinears load(exl3, DraftLinears::Mode::load);
    prod.warm_up();
    load.warm_up();
    std::uint64_t block_bytes = 0;  // trellis bytes of the 35 block linears (inputs x outputs x bits / 8)
    for (unsigned L = 0; L < DraftLinears::kLayers; ++L)
        for (unsigned k = 0; k < 7; ++k) block_bytes += std::uint64_t{prod.input(L, k)} * prod.output(L, k) * prod.bits(L, k) / 8;
    std::printf("device bytes: opt %.3f GB, load %.3f GB; block trellis %.3f GB (floor %.3f ms at 246.7 GB/s); fc %s\n",
                static_cast<double>(prod.device_bytes()) / 1e9, static_cast<double>(load.device_bytes()) / 1e9,
                static_cast<double>(block_bytes) / 1e9, static_cast<double>(block_bytes) / 246.7e6, prod.has_fc() ? "yes" : "no");
    if (prod.has_fc() != ref.has_fc) throw std::runtime_error("fc presence differs");

    const unsigned max_w = 30720;  // fc input
    float* dx = dev_copy(random_floats(std::size_t{32} * max_w, 1, 1.0F));
    std::vector<float*> ya(4), yb(4);
    for (int i = 0; i < 4; ++i) {
        ya[i] = dev_zeros(std::size_t{32} * 12288);
        yb[i] = dev_zeros(std::size_t{32} * 12288);
    }
    auto cmp = [&](const float* a, const float* b, std::size_t n, std::size_t* first) { return same_bits(host_copy(a, n), host_copy(b, n), first); };

    // ---------------------------------------------------------------- T1
    if (t1) {
        unsigned cells = 0, bad = 0;
        for (unsigned L = 0; L < DraftLinears::kLayers; ++L)
            for (unsigned k = 0; k < 7; ++k)
                for (unsigned r = 1; r <= 8; ++r) {
                    ref.mm(L, k, dx, ya[0], r, nullptr);
                    prod.run(L, k, dx, r, yb[0], nullptr);
                    hok(hipDeviceSynchronize(), "t1");
                    std::size_t first = 0;
                    ++cells;
                    if (!cmp(ya[0], yb[0], std::size_t{r} * prod.output(L, k), &first)) {
                        ++bad;
                        std::printf("  T1 mismatch L%u %s rows %u first %zu\n", L, kKind[k], r, first);
                    }
                }
        check(bad == 0, "T1a opt single linears == R (" + std::to_string(cells) + " cells: 35 matrices x rows 1..8)");
        struct G {
            std::vector<std::uint32_t> kinds;
            const char* name;
        };
        const G groups[3] = {{{0, 1, 2}, "q+k+v"}, {{1, 2}, "k+v"}, {{4, 5}, "gate+up"}};
        cells = bad = 0;
        for (const G& g : groups)
            for (unsigned L = 0; L < DraftLinears::kLayers; ++L)
                for (unsigned r = 1; r <= 8; ++r) {
                    const auto c = static_cast<unsigned>(g.kinds.size());
                    for (unsigned i = 0; i < c; ++i) ref.mm(L, g.kinds[i], dx, ya[i], r, nullptr);
                    prod.run_group(L, g.kinds.data(), c, dx, r, yb.data(), nullptr);
                    hok(hipDeviceSynchronize(), "t1g");
                    for (unsigned i = 0; i < c; ++i) {
                        ++cells;
                        if (!cmp(ya[i], yb[i], std::size_t{r} * prod.output(L, g.kinds[i]), nullptr)) {
                            ++bad;
                            std::printf("  T1 group mismatch %s L%u rows %u member %s\n", g.name, L, r, kKind[g.kinds[i]]);
                        }
                    }
                }
        check(bad == 0, "T1b opt groups == R per member (" + std::to_string(cells) + " cells: q+k+v, k+v, gate+up x 5 layers x rows 1..8)");
        cells = bad = 0;
        for (unsigned L = 0; L < DraftLinears::kLayers; ++L) {
            const std::uint32_t kv[2] = {1, 2};
            for (unsigned i = 0; i < 2; ++i) ref.mm(L, kv[i], dx, ya[i], 32, nullptr);
            prod.run_group(L, kv, 2, dx, 32, yb.data(), nullptr);
            hok(hipDeviceSynchronize(), "t1c");
            for (unsigned i = 0; i < 2; ++i) {
                ++cells;
                if (!cmp(ya[i], yb[i], std::size_t{32} * 512, nullptr)) {
                    ++bad;
                    std::printf("  T1 32-row mismatch L%u %s\n", L, kKind[kv[i]]);
                }
            }
        }
        check(bad == 0, "T1c k / v at 32 rows (strix_gemm) == R (" + std::to_string(cells) + " cells)");
        if (prod.has_fc()) {
            cells = bad = 0;
            for (const unsigned r : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 32U}) {
                ref.mm(0, 7, dx, ya[0], r, nullptr);
                prod.run(0, DraftLinears::kKindFc, dx, r, yb[0], nullptr);
                hok(hipDeviceSynchronize(), "t1d");
                ++cells;
                std::size_t first = 0;
                if (!cmp(ya[0], yb[0], std::size_t{r} * 6144, &first)) {
                    ++bad;
                    std::printf("  T1 fc mismatch rows %u first %zu\n", r, first);
                }
            }
            check(bad == 0, "T1d fc (kind 7) rows 1..8 and 32 == R (" + std::to_string(cells) + " cells)");
        }
        cells = bad = 0;
        for (const bool on : {false, true}) {
            load.set_opt(on);
            for (unsigned L = 0; L < DraftLinears::kLayers; ++L)
                for (unsigned k = 0; k < 7; ++k) {
                    ref.mm(L, k, dx, ya[0], 7, nullptr);
                    load.run(L, k, dx, 7, yb[0], nullptr);
                    hok(hipDeviceSynchronize(), "t1e");
                    ++cells;
                    if (!cmp(ya[0], yb[0], std::size_t{7} * load.output(L, k), nullptr)) {
                        ++bad;
                        std::printf("  T1 load(opt=%d) mismatch L%u %s\n", on ? 1 : 0, L, kKind[k]);
                    }
                }
        }
        check(bad == 0, "T1e Mode::load opt off / on == R at 7 rows (" + std::to_string(cells) + " cells)");
    }

    // ---------------------------------------------------------------- the drafter on the real non-linear weights
    std::unique_ptr<ds::DraftFile> file;
    std::unique_ptr<ds::DeviceDrafter> dd;
    float* d_taps = nullptr;
    std::uint16_t* d_rows = nullptr;
    const unsigned wrap_pos = 4400;
    auto hooks_ref = [&] {  // A: the old engine hook (per linear, fc on the library when the pack has it)
        const bool fc = ref.has_fc;
        dd->set_linear_hook([&ref, fc](std::uint32_t L, std::uint32_t kind, const float* x, std::size_t rows, float* out, hipStream_t st) {
            if (kind == ds::DeviceDrafter::lin_fc && !fc) return false;
            ref.mm(L, kind, x, out, rows, st);  // the old hook used the null stream; the drafter's stream keeps A ordered
            return true;
        });
        dd->set_linear_group_hook(nullptr);
    };
    auto hooks_lin = [&](DraftLinears& lin) {  // B: the production hooks (engine.hip attach_draft_exl3)
        const bool fc = lin.has_fc();
        dd->set_linear_hook([&lin, fc](std::uint32_t L, std::uint32_t kind, const float* x, std::size_t rows, float* out, hipStream_t st) {
            if (kind == ds::DeviceDrafter::lin_fc && !fc) return false;
            lin.run(L, kind, x, rows, out, st);
            return true;
        });
        dd->set_linear_group_hook([&lin](std::uint32_t L, const std::uint32_t* kinds, std::uint32_t count, const float* x, std::size_t rows,
                                         float* const* outs, hipStream_t st) {
            lin.run_group(L, kinds, count, x, rows, outs, st);
            return true;
        });
    };
    if (t2 || bench || attn_only) {
        file = ds::DraftFile::open(dspark_dir, false);
        ds::DeviceDrafter::Options dopt;
        dopt.slots = 2;
        dd = std::make_unique<ds::DeviceDrafter>(file->weights(), dopt);
        const unsigned max_ctx = std::max(wrap_pos, *std::max_element(ctxs.begin(), ctxs.end()));
        d_taps = dev_copy(random_floats(std::size_t{max_ctx} * ds::tap_width, 7, 1.0F));
        std::vector<std::uint16_t> rows16(2 * ds::hidden);
        const auto f = random_floats(rows16.size(), 11, 0.02F);
        for (std::size_t i = 0; i < f.size(); ++i) {
            std::uint32_t b;
            std::memcpy(&b, &f[i], 4);
            rows16[i] = static_cast<std::uint16_t>(b >> 16);
        }
        hok(hipMalloc(&d_rows, rows16.size() * 2), "malloc");
        hok(hipMemcpy(d_rows, rows16.data(), rows16.size() * 2, hipMemcpyHostToDevice), "h2d");
    }
    // append() splits a long append into 32-row chunks (k + v and fc on strix_gemm) and <= 8-row chunks (k + v grouped on
    // opt in B); `step` > 0 appends the rows after `first` in calls of `step` rows (decode-like accepted + 1 appends).
    auto build = [&](unsigned slot, unsigned first, unsigned total, unsigned step, hipStream_t st) {
        dd->append(slot, d_taps, first, 0, st);
        for (unsigned p = first; p < total; p += step) {
            const unsigned n = std::min(step, total - p);
            dd->append(slot, d_taps + std::size_t{p} * ds::tap_width, n, p, st);
        }
    };
    if (t2) {
        struct Case {
            unsigned first, total, step;
            bool stream;
            const char* what;
        };
        std::vector<Case> cases;
        for (const unsigned c : ctxs) cases.push_back({c, c, 7, false, "single append"});
        cases.push_back({4095, wrap_pos, 7, false, "4095 + 7-row appends, ring wrapped"});
        cases.push_back({300, 300, 7, true, "non-null stream"});
        for (const Case& c : cases) {
            hipStream_t st = nullptr;
            if (c.stream) hok(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
            hooks_ref();
            build(0, c.first, c.total, c.step, st);
            dd->forward_block(0, d_rows, d_rows + ds::hidden, c.total, st);
            hok(hipStreamSynchronize(st), "t2 a");
            hok(hipDeviceSynchronize(), "t2 a");
            const auto ya_h = host_copy(dd->y(), std::size_t{ds::block} * ds::hidden);
            hooks_lin(prod);
            build(1, c.first, c.total, c.step, st);
            dd->forward_block(1, d_rows, d_rows + ds::hidden, c.total, st);
            hok(hipStreamSynchronize(st), "t2 b");
            const auto yb_h = host_copy(dd->y(), std::size_t{ds::block} * ds::hidden);
            if (st) hok(hipStreamDestroy(st), "stream");
            std::size_t first = 0;
            const bool same = same_bits(ya_h, yb_h, &first);
            bool finite = true;
            for (const float v : yb_h) finite = finite && std::isfinite(v);
            check(same && finite, "T2 position " + std::to_string(c.total) + " (" + c.what + "): y bits B == A (7 x 6144)" +
                                      (same ? "" : ", first diff " + std::to_string(first)) + (finite ? "" : ", NOT FINITE"));
        }
    }

    if (t2) {  // T3: every attention variant (DeviceDrafter::Options::attn_variant) vs variant 0, production linears
        hooks_lin(prod);
        struct Case {
            unsigned first, total;
        };
        const Case cases[] = {{60, 60}, {100, 100}, {300, 300}, {1000, 1000}, {4095, 4095}, {4095, wrap_pos}};
        for (const Case& c : cases) {
            build(0, c.first, c.total, 7, nullptr);
            dd->set_attn_variant(0);
            dd->forward_block(0, d_rows, d_rows + ds::hidden, c.total, nullptr);
            hok(hipDeviceSynchronize(), "t3 a");
            const auto ya_h = host_copy(dd->y(), std::size_t{ds::block} * ds::hidden);
            for (const unsigned v : {1U, 2U, 3U}) {
                dd->set_attn_variant(v);
                dd->forward_block(0, d_rows, d_rows + ds::hidden, c.total, nullptr);
                hok(hipDeviceSynchronize(), "t3 b");
                const auto yb_h = host_copy(dd->y(), std::size_t{ds::block} * ds::hidden);
                std::size_t first = 0;
                const bool same = same_bits(ya_h, yb_h, &first);
                check(same, "T3 position " + std::to_string(c.total) + ": attention variant " + std::to_string(v) + " y bits == variant 0" +
                                (same ? "" : ", first diff " + std::to_string(first)));
            }
            dd->set_attn_variant(0);
        }
    }

    hipEvent_t e0, e1;
    hok(hipEventCreate(&e0), "event");
    hok(hipEventCreate(&e1), "event");
    auto timed = [&](auto&& f) {
        f();  // untimed first call after each arm switch
        hok(hipEventRecord(e0, nullptr), "event");
        for (unsigned i = 0; i < iters; ++i) f();
        hok(hipEventRecord(e1, nullptr), "event");
        hok(hipEventSynchronize(e1), "event");
        float ms = 0.0F;
        hok(hipEventElapsedTime(&ms, e0, e1), "event");
        return static_cast<double>(ms) / iters;
    };
    auto ratios = [&](const std::vector<double>& a, const std::vector<double>& b) {
        std::printf(" | B/A");
        for (unsigned r = 0; r < reps; ++r) std::printf(" %.4f", b[r] / (0.5 * (a[r] + a[r + 1])));
    };
    if (bench) {
        hooks_lin(load);
        // the linears A/B runs on one fixed attention baseline (Options' default is variant 2 since 0b8b5473): pin it
        dd->set_attn_variant(block_attn);
        std::printf("BLOCK timing: attention variant %u in both arms\n", block_attn);
        for (const unsigned ctx : ctxs) {
            load.set_opt(false);
            build(0, ctx, ctx, 7, nullptr);
            hok(hipDeviceSynchronize(), "bench ctx");
            std::printf("BLOCK ctx %u ms/block:", ctx);
            std::vector<double> a, b;
            for (unsigned r = 0; r < 2 * reps + 1; ++r) {
                const bool opt = r % 2 == 1;
                load.set_opt(opt);
                const double ms = timed([&] { dd->forward_block(0, d_rows, d_rows + ds::hidden, ctx, nullptr); });
                (opt ? b : a).push_back(ms);
                std::printf(" %s=%.3f", opt ? "B" : "A", ms);
            }
            ratios(a, b);
            std::printf("\n");
        }
    }
    if (bench || attn_only) {  // attention arms: production linears, A = variant 0, B = each of --attn-variants
        hooks_lin(prod);
        for (const unsigned v : attn_variants)
            for (const unsigned ctx : attn_ctxs) {
                build(0, ctx, ctx, 7, nullptr);
                hok(hipDeviceSynchronize(), "bench ctx");
                std::printf("ATTN v%u ctx %u ms/block:", v, ctx);
                std::vector<double> a, b;
                for (unsigned r = 0; r < 2 * reps + 1; ++r) {
                    const bool on = r % 2 == 1;
                    dd->set_attn_variant(on ? v : 0U);
                    const double ms = timed([&] { dd->forward_block(0, d_rows, d_rows + ds::hidden, ctx, nullptr); });
                    (on ? b : a).push_back(ms);
                    std::printf(" %s=%.3f", on ? "B" : "A", ms);
                }
                dd->set_attn_variant(0);
                ratios(a, b);
                std::printf("\n");
            }
    }
    if (linears) {
        Bufs bf;
        bf.x = dx;
        bf.q = dev_zeros(7 * 8192);
        bf.k = dev_zeros(7 * 512);
        bf.v = dev_zeros(7 * 512);
        bf.att = dev_copy(random_floats(7 * 8192, 3, 1.0F));
        bf.o = dev_zeros(7 * 6144);
        bf.g = dev_copy(random_floats(7 * 12288, 5, 1.0F));
        bf.u = dev_zeros(7 * 12288);
        bf.o2 = dev_zeros(7 * 6144);
        // the down input must not be overwritten by gate+up in the chain (fixed input, same work): gate writes a copy
        float* g_out = dev_zeros(7 * 12288);
        Bufs bc = bf;
        bc.g = g_out;  // run_group writes gate here; down reads bf.g
        constexpr unsigned kEv = 4 * DraftLinears::kLayers + 1;
        std::vector<hipEvent_t> ev(kEv);
        for (auto& e : ev) hok(hipEventCreate(&e), "event");
        auto pass = [&](hipEvent_t* evp) {
            int e = 0;
            if (evp) hok(hipEventRecord(evp[e++], nullptr), "event");
            for (unsigned L = 0; L < DraftLinears::kLayers; ++L) {
                const std::uint32_t qkv[3] = {0, 1, 2}, gu[2] = {4, 5};
                float* const o_qkv[3] = {bf.q, bf.k, bf.v};
                float* const o_gu[2] = {bc.g, bf.u};
                load.run_group(L, qkv, 3, bf.x, 7, o_qkv, nullptr);
                if (evp) hok(hipEventRecord(evp[e++], nullptr), "event");
                load.run(L, 3, bf.att, 7, bf.o, nullptr);
                if (evp) hok(hipEventRecord(evp[e++], nullptr), "event");
                load.run_group(L, gu, 2, bf.x, 7, o_gu, nullptr);
                if (evp) hok(hipEventRecord(evp[e++], nullptr), "event");
                load.run(L, 6, bf.g, 7, bf.o2, nullptr);
                if (evp) hok(hipEventRecord(evp[e++], nullptr), "event");
            }
        };
        std::printf("LINEARS ms/pass (35 linears, block order):");
        std::vector<double> a, b;
        std::vector<std::array<double, 4>> seg(2 * reps + 1, std::array<double, 4>{});
        for (unsigned r = 0; r < 2 * reps + 1; ++r) {
            const bool opt = r % 2 == 1;
            load.set_opt(opt);
            const double ms = timed([&] { pass(nullptr); });
            (opt ? b : a).push_back(ms);
            std::printf(" %s=%.3f", opt ? "B" : "A", ms);
            for (unsigned i = 0; i < iters; ++i) {  // evented passes (diagnostic attribution)
                pass(ev.data());
                hok(hipEventSynchronize(ev[kEv - 1]), "event");
                for (unsigned e = 0; e + 1 < kEv; ++e) {
                    float t = 0.0F;
                    hok(hipEventElapsedTime(&t, ev[e], ev[e + 1]), "event");
                    seg[r][e % 4] += static_cast<double>(t) / iters;
                }
            }
        }
        ratios(a, b);
        std::vector<double> sa = a, sb = b;
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        std::printf(" | useful GB/s A %.1f B %.1f (medians)\n", static_cast<double>(block_bytes) / 1e6 / sa[sa.size() / 2],
                    static_cast<double>(block_bytes) / 1e6 / sb[sb.size() / 2]);
        const char* const sn[4] = {"q+k+v", "o", "gate+up", "down"};
        for (unsigned s = 0; s < 4; ++s) {
            std::printf("  SEG %-8s ms/pass:", sn[s]);
            for (unsigned r = 0; r < 2 * reps + 1; ++r) std::printf(" %s=%.3f", r % 2 ? "B" : "A", seg[r][s]);
            std::printf(" | B/A");
            for (unsigned r = 0; r < reps; ++r) std::printf(" %.4f", seg[2 * r + 1][s] / (0.5 * (seg[2 * r][s] + seg[2 * r + 2][s])));
            std::printf("\n");
        }
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 3;
}
