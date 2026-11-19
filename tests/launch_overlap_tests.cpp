// Host tests of the backend-neutral early-launch rule (include/strix/launch_overlap.hpp): the seeded chains of the GPU probe
// harness (tests/cuda_core/pdl_harness.cuh) as data, the placement planner, and the Qwen verify layer chain of the CUDA
// core (NVFP4 GEMV / reduce, norms, GDN, attention) as it is launched today.
#include "strix/launch_overlap.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace ov = strix::overlap;

namespace {

int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr ov::Access R = ov::Access::read, W = ov::Access::write;
constexpr ov::Phase PRE = ov::Phase::pre_wait, POST = ov::Phase::post_wait;

ov::Launch k(const char* name, bool early, bool release_after_wait, std::vector<ov::Use> uses, bool waits = true) {
    ov::Launch l;
    l.kernel = name;
    l.early = early;
    l.waits_every_launch = waits;
    l.release_after_wait = release_after_wait;
    l.uses = std::move(uses);
    return l;
}

bool has(const std::vector<ov::Violation>& v, std::size_t consumer, std::size_t producer, std::uint32_t buffer) {
    for (const ov::Violation& x : v)
        if (x.consumer == consumer && x.producer == producer && x.buffer == buffer) return true;
    return false;
}

void seeded() {
    enum : std::uint32_t { X = 1, S = 2, STATE = 3 };
    // R1: the writer releases before its last write, the reader reads after its wait: ordered.
    expect(ov::check({k("writer", true, false, {{S, W, POST}, {X, W, POST}}), k("reader", true, false, {{X, R, POST}})}).empty(),
           "R1 release before the last write, read after the wait: no violation");
    // S1: read before the wait.
    std::vector<ov::Launch> s1 = {k("writer", true, false, {{X, W, POST}}), k("reader", true, false, {{X, R, PRE}})};
    expect(has(ov::check(s1), 1, 0, X), "S1 read before the wait: violation (reader, writer, X)");
    expect(ov::plan(s1) == 1 && !s1[1].early && ov::check(s1).empty(), "S1 plan: reader launched without early launch");
    // S2: a pre-wait write into the scratch an overlapping kernel uses.
    expect(has(ov::check({k("writer", true, false, {{S, W, POST}, {S, R, POST}}), k("reader", true, false, {{S, W, PRE}})}), 1, 0,
               S),
           "S2 pre-wait write into a shared scratch: violation");
    // S3: a middle kernel that never waits breaks the completion chain for a post-wait read two launches later.
    std::vector<ov::Launch> s3 = {k("writer", true, false, {{X, W, POST}}), k("middle", true, false, {}, false),
                                  k("reader", true, false, {{X, R, POST}})};
    expect(has(ov::check(s3), 2, 0, X), "S3 middle kernel never waits: violation (reader, writer, X)");
    expect(ov::plan(s3) == 1 && !s3[1].early && s3[2].early && ov::check(s3).empty(),
           "S3 plan: the non-waiting middle kernel is demoted, the reader keeps early launch");
    // T: the same chain with a waiting middle kernel.
    expect(ov::check({k("writer", true, false, {{X, W, POST}}), k("middle", true, false, {}), k("reader", true, false, {{X, R, POST}})})
               .empty(),
           "T middle kernel waits: no violation at depth 2");
    // A launch without early launch is a barrier: state written before it may be read before a later kernel's wait.
    expect(ov::check({k("commit", true, false, {{STATE, W, POST}}), k("plain", false, false, {}), k("next", true, false, {}),
                      k("conv_pre", true, false, {{STATE, R, PRE}})})
               .empty(),
           "barrier: a normal launch orders earlier writes before every later pre-wait read");
    expect(has(ov::check({k("commit", true, false, {{STATE, W, POST}}), k("next", true, false, {}),
                          k("conv_pre", true, false, {{STATE, R, PRE}})}),
               2, 0, STATE),
           "no barrier: pre-wait read of state written two early launches before: violation");
    // Release after the wait bounds the overlap to depth 1; release at entry lets depth 2 overlap.
    const std::vector<ov::Launch> after = {k("a", true, true, {}), k("b", true, true, {}), k("c", true, true, {}), k("d", true, true, {})};
    const std::vector<ov::Launch> entry = {k("a", true, false, {}), k("b", true, false, {}), k("c", true, false, {}),
                                           k("d", true, false, {})};
    std::size_t d2_after = 0, d2_entry = 0;
    for (const ov::Pair& p : ov::overlapping(after, 2)) d2_after += p.second - p.first == 2 ? 1u : 0u;
    for (const ov::Pair& p : ov::overlapping(entry, 2)) d2_entry += p.second - p.first == 2 ? 1u : 0u;
    expect(d2_after == 0 && d2_entry == 2, "release after the wait: no depth-2 overlap; release at entry: depth-2 pairs exist");
}

// Buffers of the Qwen verify layer chain (CUDA core, rows = verify rows). Weights are per GEMV and never written.
enum : std::uint32_t {
    H = 1, N1, PART, QKVZ, CONV, CONV_STATE, GDN_STATE, GDN_OUT, N2, GU, QKV, QN, KN, KV, SC, VAL, AOUT,
    WQKVZ = 100, WOUT, WGU, WDOWN, WQKV, WO, WGU2, WDOWN2
};

// Facts per kernel (read in the kernel sources): fp4_gemv pre-wait = weight records only, waits in block (x,0,0) of
// every launch (nvfp4.cu fp4_gemv), release after the wait; fp4_reduce waits first, releases after it; norm_kernel
// releases at entry, waits, then reads (norms.cu); conv_pre reads conv_state before its wait (gdn.cu:41); the other
// attention / GDN / rope kernels touch only constant data before their wait and release at entry.
void gemv(std::vector<ov::Launch>& c, std::uint32_t w, std::uint32_t x, bool reduce, std::uint32_t y, std::uint32_t resid) {
    c.push_back(k("fp4_gemv", true, true, {{w, R, PRE}, {x, R, POST}, {PART, W, POST}}));
    if (!reduce) return;
    std::vector<ov::Use> u = {{PART, R, POST}, {y, W, POST}};
    if (resid) u.push_back({resid, R, POST});
    c.push_back(k("fp4_reduce", true, true, u));
}
std::vector<ov::Launch> qwen_verify_layers() {
    std::vector<ov::Launch> c;
    // GDN layer
    c.push_back(k("norm", true, false, {{H, R, POST}, {N1, W, POST}}));
    gemv(c, WQKVZ, N1, true, QKVZ, 0);
    c.push_back(k("conv_pre", true, false, {{CONV_STATE, R, PRE}, {QKVZ, R, POST}, {CONV, W, POST}}));
    c.push_back(k("delta_window", true, false, {{GDN_STATE, R, POST}, {CONV, R, POST}, {QKVZ, R, POST}, {GDN_OUT, W, POST}}));
    gemv(c, WOUT, GDN_OUT, true, H, H);
    c.push_back(k("norm", true, false, {{H, R, POST}, {N2, W, POST}}));
    gemv(c, WGU, N2, true, GU, 0);
    gemv(c, WDOWN, GU, true, H, H);
    // attention layer
    c.push_back(k("norm", true, false, {{H, R, POST}, {N1, W, POST}}));
    gemv(c, WQKV, N1, true, QKV, 0);
    c.push_back(k("norm_q", true, false, {{QKV, R, POST}, {QN, W, POST}}));
    c.push_back(k("norm_k", true, false, {{QKV, R, POST}, {KN, W, POST}}));
    c.push_back(k("rope_q", true, false, {{QN, R, POST}, {QN, W, POST}}));
    c.push_back(k("rope_k", true, false, {{KN, R, POST}, {KN, W, POST}}));
    c.push_back(k("kv_store", true, false, {{KN, R, POST}, {QKV, R, POST}, {KV, W, POST}}));
    c.push_back(k("attn_scores", true, false, {{QN, R, POST}, {KV, R, POST}, {SC, W, POST}}));
    c.push_back(k("attn_values", true, false, {{SC, R, POST}, {KV, R, POST}, {VAL, W, POST}}));
    c.push_back(k("attn_combine", true, false, {{VAL, R, POST}, {SC, R, POST}, {AOUT, W, POST}}));
    gemv(c, WO, AOUT, true, H, H);
    c.push_back(k("norm", true, false, {{H, R, POST}, {N2, W, POST}}));
    gemv(c, WGU2, N2, true, GU, 0);
    gemv(c, WDOWN2, GU, true, H, H);
    return c;
}

void qwen_chain() {
    std::vector<ov::Launch> c = qwen_verify_layers();
    expect(ov::check(c).empty(), "Qwen verify GDN + attention layer chain as launched: no unordered access");
    const std::vector<ov::Pair> p = ov::overlapping(c, 2);
    std::size_t d1 = 0, d2 = 0, d2_behind_fp4 = 0;
    for (const ov::Pair& x : p) {
        if (x.second - x.first == 1) ++d1;
        if (x.second - x.first == 2) {
            ++d2;
            const std::string& mid = c[x.first + 1].kernel;
            if (mid == "fp4_gemv" || mid == "fp4_reduce") ++d2_behind_fp4;
        }
    }
    std::printf("-- Qwen verify layer table: %zu launches, overlap pairs depth 1: %zu, depth 2: %zu\n", c.size(), d1, d2);
    for (const ov::Pair& x : p)
        if (x.second - x.first == 2)
            std::printf("   depth 2: %s(%zu) .. %s(%zu)\n", c[x.first].kernel.c_str(), x.first, c[x.second].kernel.c_str(),
                        x.second);
    expect(d1 == c.size() - 1, "every consecutive pair may overlap (all early)");
    expect(d2_behind_fp4 == 0, "no depth-2 overlap through an NVFP4 GEMV / reduce (release after the wait)");
    // Seeded variants of the same chain.
    std::vector<ov::Launch> bad = c;
    bad[1].uses.push_back({N1, R, PRE});  // fp4_gemv reading its activation before the wait
    expect(has(ov::check(bad), 1, 0, N1), "seeded: GEMV reads its activation before the wait: violation");
    std::vector<ov::Launch> commit = c;
    commit.insert(commit.begin() + 3, k("conv_commit", true, false, {{CONV_STATE, W, POST}}));
    expect(!ov::check(commit).empty(), "seeded: an early conv_state writer before conv_pre in the segment: violation");
    const std::size_t demoted = ov::plan(commit);
    expect(demoted == 1 && !commit[4].early && ov::check(commit).empty(), "plan: conv_pre launched normally fixes it");
    std::vector<ov::Launch> nowait = c;
    nowait[2].waits_every_launch = false;  // a reduce in which no block waits
    expect(!ov::check(nowait).empty(), "seeded: a reduce that never waits breaks the chain: violation");
}

}  // namespace

int main() {
    seeded();
    qwen_chain();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
