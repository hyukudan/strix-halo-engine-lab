// Window entries of the CUDA Qwen main session, compiled verbatim from src/qwen35/cuda/engine.cu (prepare.py -> window*.inc)
// against the mock runtime. Invariant: forward and forward_fed run the same input prologue, so with decode_attn8 the e4m3
// shadow mirrors the committed prefix (after a big prefill, which advances committed alone, and after reset) before any
// window reads it, outside graph capture and before the window's input kernel; run_rows refuses an entry that skipped it.
#include "strix/qwen35/cuda_engine.hpp"
#include "../../src/core/dense_linear.hpp"
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cc = strix::cuda_core;
namespace q = strix::qwen35;
namespace core = strix::core;
#define CK(expr) do { if ((expr) != cudaSuccess) throw std::runtime_error("mock CUDA failure"); } while (0)

// Call log: A shadow allocated, Q one layer quantised into the shadow, I / F host / fed input kernel,
// B / E graph capture begin / end, T trunk body, L graph launch.
static std::string calls;
static bool capturing = false;
static int failures = 0;
static void check(bool ok, const char* name) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s (calls %s)\n", name, calls.c_str()); }
}

using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
constexpr int cudaStreamCaptureModeThreadLocal = 1;
static char graph_token, exec_token;
int cudaStreamBeginCapture(cudaStream_t, int) { capturing = true; calls += 'B'; return cudaSuccess; }
int cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) { capturing = false; calls += 'E'; *g = &graph_token; return cudaSuccess; }
int cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
int cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t, unsigned long long) { *e = &exec_token; return cudaSuccess; }
int cudaGraphGetNodes(cudaGraph_t, void*, std::size_t* n) { *n = 3; return cudaSuccess; }
int cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { calls += 'L'; return cudaSuccess; }
int cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
int cudaGetLastError() { return cudaSuccess; }
int cudaMemsetAsync(void*, int, std::size_t, cudaStream_t) { return cudaSuccess; }

struct ShadowWrite { int layer; std::uint32_t from, count; bool operator==(const ShadowWrite&) const = default; };
static std::vector<ShadowWrite> shadow_writes;
namespace strix::cuda_core {
DenseArith dense_arith() { return DenseArith::mma3; }  // EngineOptions default (process policy); not used by the window entries
struct KvShadowE4m3 { int layer = -1; };
struct GdnCommitLayer {};
cudaError_t kv_shadow_e4m3(int, const std::uint16_t*, const std::uint16_t*, const KvShadowE4m3& sh, std::uint32_t from,
                           std::uint32_t count, cudaStream_t) {
    check(!capturing, "shadow catch-up must run outside graph capture");
    calls += 'Q';
    shadow_writes.push_back({sh.layer, from, count});
    return cudaSuccess;
}
template <class G>
cudaError_t gdn_commit(const G&, const GdnCommitLayer*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t*, cudaStream_t) {
    return cudaSuccess;
}
}  // namespace strix::cuda_core

#include "window_types.inc"
void set_inputs_kernel(InputPack in, std::uint32_t* pos, std::uint32_t* ids) {
    calls += 'I';
    *pos = in.pos0;
    for (std::uint32_t i = 0; i < cc::kMaxRows; ++i) ids[i] = in.ids[i];
}
void fed_rows_kernel(FedPack in, const std::uint32_t* feed, std::uint32_t* pos, std::uint32_t* ids) {
    calls += 'F';
    *pos = in.pos0;
    for (std::uint32_t i = 0; i < cc::kMaxRows; ++i) ids[i] = in.slot[i] == kFeedHostRow ? in.ids[i] : feed[in.slot[i]];
}

struct Impl {
    using LogitRows = q::LogitRows;
    using RowsOut = q::RowsOut;
    using CudaEngine = q::CudaEngine;
    static constexpr std::uint32_t RM = 32;
    q::EngineOptions opt;
    struct Config { std::uint32_t layers = 4, vocab = 64; } c;
    struct Gdn {
        std::uint32_t key_dim = 2, value_dim = 2, value_heads = 1;
        std::uint32_t conv_channels() const { return 2; }
    } gd;
    struct LayerW { bool attn = false; };
    std::vector<LayerW> L{{false}, {true}, {false}, {true}};
    std::vector<float*> conv = std::vector<float*>(4), delta = std::vector<float*>(4);
    std::vector<std::uint16_t*> kc = std::vector<std::uint16_t*>(4), vc = std::vector<std::uint16_t*>(4);
    std::vector<cc::GdnCommitLayer> commit_layers = std::vector<cc::GdnCommitLayer>(2);
    int ad = 0;
    cudaStream_t s = nullptr;
    bool mtp_ok = true;
    std::uint32_t pos_word = 0, ids_words[cc::kMaxRows] = {}, feed_words[CudaEngine::kMtpFeedSlots] = {}, parity_word = 0, top_words[cc::kMaxRows] = {};
    std::uint32_t *d_pos = &pos_word, *d_ids = ids_words, *d_feed = feed_words, *d_parity = &parity_word, *d_top = top_words;
    float logit_words[4] = {}, x_words[4] = {}, xn_words[4] = {};
    float *logits = logit_words, *x = x_words, *xn = xn_words;
    std::uint32_t committed = 0, last_rows = 0, d8_end = 0;
    bool graphs_on = true;
    struct Graph {
        cc::UniqueHandle<cudaGraphExec_t, cudaGraphExecDestroy> exec;
        std::uint64_t nodes = 0;
    };
    std::unordered_map<std::uint32_t, Graph> graphs;
    std::uint64_t last_nodes = 0;
    struct Pf { std::vector<cc::KvShadowE4m3> sh; } pf;
    bool trunk_saw_stale = false;
    void pf_shadow_init() {  // engine.cu allocates one shadow per attention layer on first use; the mock tags each layer
        if (!pf.sh.empty()) return;
        calls += 'A';
        for (std::uint32_t l = 0; l < c.layers; ++l) pf.sh.push_back({L[l].attn ? static_cast<int>(l) : -1});
    }
    void trunk(std::uint32_t, LogitRows) {  // the window's attention reads &pf.sh[l] and shadow positions < committed + row
        calls += 'T';
        if (decode8_on() && (pf.sh.size() != c.layers || d8_end != committed)) trunk_saw_stale = true;
    }
    void prefill_big(std::uint32_t n) { committed += n; }  // engine.cu prefill(): `committed += n;`, the decode shadow is not touched
#include "window.inc"
};

template <class F> static bool rejects(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
static std::vector<ShadowWrite> attn_catch_up(std::uint32_t from, std::uint32_t count) { return {{1, from, count}, {3, from, count}}; }
static void clear() { calls.clear(); shadow_writes.clear(); }

static void run() {
    const std::uint32_t ids[4] = {11, 12, 13, 14};
    const std::uint32_t host_slot[3] = {kFeedHostRow, kFeedHostRow, kFeedHostRow};
    {  // forward_fed as the very first window: the shadow is allocated before the trunk takes &pf.sh[l]
        Impl e;
        e.opt.max_positions = 512;
        e.opt.decode_attn8 = true;
        clear();
        e.feed_words[5] = 21;
        const std::uint32_t slot[2] = {kFeedHostRow, 5}, vals[2] = {12, 0};
        e.forward_fed(ids, 1, slot, vals, 2, q::LogitRows::all);
        check(calls == "AFBTEL" && shadow_writes.empty() && !e.trunk_saw_stale, "first fed window allocates the shadow first");
        check(e.ids_words[0] == 11 && e.ids_words[1] == 12 && e.ids_words[2] == 21 && e.pos_word == 0, "fed inputs gathered");
        e.commit_rows(3);
        check(e.committed == 3 && e.d8_end == 3, "commit keeps the shadow current");
        clear();  // big prefill, then the first (pre-launched) verify: catch-up before the fed input kernel and the graph
        e.prefill_big(100);
        e.forward_fed(ids, 1, host_slot, ids + 1, 3, q::LogitRows::all);
        check(calls == "QQFBTEL" && shadow_writes == attn_catch_up(3, 100) && !e.trunk_saw_stale, "fed verify after big prefill catches up");
        e.commit_rows(4);
        check(e.committed == 107 && e.d8_end == 107, "commit after the caught-up fed verify");
        clear();  // reset: a new sequence re-quantises from position 0, no second allocation
        e.reset();
        check(e.committed == 0 && e.d8_end == 0, "reset rewinds the shadow mirror");
        e.prefill_big(40);
        e.forward_fed(ids, 1, host_slot, ids + 1, 3, q::LogitRows::all);
        check(calls == "QQFL" && shadow_writes == attn_catch_up(0, 40) && !e.trunk_saw_stale, "fed verify after reset + big prefill");
        e.commit_rows(0);
        clear();  // graphs off: same prologue, the trunk runs eagerly after it
        e.graphs_on = false;
        e.prefill_big(10);
        e.forward_fed(ids, 1, host_slot, ids + 1, 3, q::LogitRows::all);
        check(calls == "QQFT" && shadow_writes == attn_catch_up(40, 10) && !e.trunk_saw_stale, "graphs off: catch-up before the trunk");
        e.commit_rows(4);
        clear();  // an entry that skips the prologue is refused before any trunk work
        e.prefill_big(5);
        check(rejects([&] { (void)e.run_rows(1, q::LogitRows::all); }) && calls.empty(), "run_rows refuses a stale shadow");
        clear();  // bounds stay ahead of the catch-up: an over-capacity fed window launches nothing
        e.committed = 510;
        check(rejects([&] { (void)e.forward_fed(ids, 1, host_slot, ids + 1, 3, q::LogitRows::all); }) && calls.empty(),
              "capacity checked before any launch");
    }
    {  // fed and unfed verify after the same big prefill: same catch-up, same window inputs, same graph key
        Impl a, b;
        for (Impl* e : {&a, &b}) {
            e->opt.max_positions = 512;
            e->opt.decode_attn8 = true;
            e->prefill_big(64);
        }
        clear();
        a.forward(ids, 4, q::LogitRows::all);
        const std::string calls_a = calls;
        const auto writes_a = shadow_writes;
        clear();
        b.forward_fed(ids, 1, host_slot, ids + 1, 3, q::LogitRows::all);
        check(calls_a == "AQQIBTEL" && calls == "AQQFBTEL" && writes_a == shadow_writes, "fed and unfed run the same prologue");
        bool same = a.pos_word == b.pos_word;
        for (std::uint32_t i = 0; i < cc::kMaxRows; ++i) same = same && a.ids_words[i] == b.ids_words[i];
        check(same && a.graphs.size() == 1 && b.graphs.count(a.graphs.begin()->first) == 1, "fed and unfed window inputs identical");
    }
    {  // decode_attn8 off (default): no shadow, no catch-up, no guard
        Impl e;
        e.opt.max_positions = 512;
        clear();
        e.prefill_big(50);
        e.forward_fed(ids, 1, host_slot, ids + 1, 3, q::LogitRows::all);
        e.commit_rows(4);
        e.forward(ids, 4, q::LogitRows::last);
        check(calls == "FBTELIBTEL" && shadow_writes.empty() && e.pf.sh.empty() && e.d8_end == 0, "default path untouched");
    }
}

int main() {
    try {
        run();
    } catch (const std::exception& ex) {  // a window refused where the prologue should have run: a failure, not terminate
        ++failures;
        std::fprintf(stderr, "FAIL: unexpected exception: %s (calls %s)\n", ex.what(), calls.c_str());
    }
    std::printf(failures ? "window prologue: FAILED (%d)\n" : "window prologue: forward and forward_fed share the decode_attn8 catch-up: PASS\n", failures);
    return failures ? 1 : 0;
}
