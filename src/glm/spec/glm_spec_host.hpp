#pragma once
// GLM-5.3-Flash speculative decoding, host reference (design notes): greedy acceptance, the verify
// state rules for R rows of ONE sequence slot (KDA deferred replay, DSA multi-position append + frontier rollback) and
// the MTP draft layer (layer 45) as a composition of the Tier A host ops with the EXL3 sites as callbacks (Tier B).
// Exactness rests on: every verify row's ops are the row-wise ops of the plain decode step (batch invariance R <= 8),
// KDA/conv advance through exactly the same kda_step sequence as a plain run, and a DSA row at position t reads only
// positions <= t. The drafter only proposes tokens; it never changes the emitted stream.
#include "glm/host/glm_host.hpp"

#include <functional>
#include <span>
#include <vector>

namespace strix::glm::spec {

using host::ArgMax;
using host::DsaCache;
using host::IndexerWeights;
using host::KdaInputs;
using host::KdaState;
using host::KdaWeights;
using host::Mat;
using host::Vec;

// ---- greedy acceptance of one cycle. Verify rows are [x0, d1 .. dK] at positions p .. p+K; target[j] = argmax of
// row j (the target's next token after position p+j). Accepted m = longest prefix with d_{j+1} == target[j]; the
// cycle emits d1 .. dm and then bonus = target[m]. Rows 0 .. m are committed, rows m+1 .. K are rolled back.
// Termination is applied first: plain greedy processes row j only if every token emitted before it (x0's emission
// included) left the request running; emission stops after a token equal to `eos` or when `budget` tokens of this cycle
// have been emitted. A draft accepted as the terminal token ends the cycle with no bonus (stop, rows = m); a terminal
// bonus ends it after the bonus (stop, rows = m + 1).
// Non-finite checkpoints (C1-C5) are per row: first_flagged = the first row whose checkpoint fired (rows from it on have
// no defined outputs; target holds at least min(K + 1, first_flagged) entries). A flagged row aborts the request only if
// plain greedy would process it (it is within the terminal frontier): abort = true, the cycle emits d1 .. dm (m = that
// row) and stops, exactly where plain greedy fails. Flagged rows after the frontier are ignored.
struct Accepted {
    unsigned m = 0;              // drafts accepted and emitted (d1 .. dm)
    std::uint32_t bonus = 0;     // valid iff has_bonus
    bool has_bonus = false;
    bool abort = false;
    bool stop = false;           // the request terminates after this cycle's emission (EOS or budget)
    unsigned rows = 0;           // committed verify rows (0 .. rows-1); the plain run processes exactly these
};
inline constexpr unsigned kNoFlag = 0xffffffffU;
struct Terminal {
    std::uint32_t eos = kNoFlag;       // one EOS id (a set of ids: call per id class or map them before)
    unsigned budget = kNoFlag;         // tokens this cycle may still emit before the output limit
};
Accepted accept_greedy(std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> target,
                       unsigned first_flagged = kNoFlag, Terminal term = {});

// ---- replicated window control for TP2 (design notes section 2.1). Rank 0 owns the timing EMAs and publishes them as a
// CostTable inside its half of the verify argmax exchange of cycle c (version-stamped); from the draft phase of cycle c
// on, BOTH ranks decide with that same table and the merged MTP confidences, so every decision is a pure function of
// replicated data and is known to both ranks before any dependent launch (no extra exchange).
struct CostTable {
    std::uint32_t version = 0;
    float plain_ms = 0.0F;
    float verify_ms[9] = {};   // verify of R = 1..8 new rows (index R), replay averaged in by rank 0
    float step_ms = 0.0F;      // one MTP extension step
    float catchup_ms = 0.0F;   // mean MTP catch-up (wire table[10])
    // Publisher only (not on the wire): verify samples per R. Unsampled rows follow the sampled ones by the fitted
    // per-row slope, so a window that only ever verifies R rows cannot leave R-1 at a stale seed (which made the
    // marginal cost of the last draft look ~4 ms instead of ~10 ms and locked adaptive K at Kmax).
    std::uint32_t verify_samples[9] = {};
};
// After draft step k (1-based) with cumulative merged survival estimate s_k (product of confidences so far): draft
// step k+1 iff s_k * conf_next_estimate is worth its marginal cost. The rule used: extend iff
//   s_k * plain_ms > (verify_ms[k+2] - verify_ms[k+1]) + step_ms   and k < kmax.
// Gate (no draft at all this cycle, verify = plain row): draft iff conf_1 * plain_ms > verify_ms[2] - verify_ms[1].
bool window_extend(const CostTable& t, float survival_k, unsigned k, unsigned kmax);
bool window_gate_open(const CostTable& t, float conf1);

// Adaptive state is replicated; only the publisher observes local wall times. Wire table[0] is plain,
// [1..8] is verify R, [9] is extension, [10] is catch-up. Invalid packets select the runner's fixed-K fallback.
// Extend when S_next * (C_k / E_k) exceeds 1.03 times the incremental verify + extension cost.
struct WindowEstimate { unsigned k = 0; float survival = 1, expected = 1, cycle_ms = 0; };
bool window_extend(const CostTable&, const WindowEstimate&, float p_next, unsigned kmax);
bool valid_cost_table(const CostTable&);
CostTable seeded_cost_table();
void store_cost_table(const CostTable&, std::span<float, 11>);
bool load_cost_table(std::span<const float, 11>, std::uint32_t version, CostTable&);
void observe_cost(CostTable&, unsigned wire_index, double elapsed_ms);
unsigned draft_limit(unsigned kmax, unsigned remaining, std::uint64_t p, std::uint64_t capacity);
class Calibration {
public:
    float probability(unsigned depth, float confidence) const;
    void observe(unsigned depth, float confidence, bool accepted);
private:
    static unsigned bin(float confidence);
    std::uint32_t trials_[8][8] = {}, successes_[8][8] = {};
};

// ---- KDA rows of one slot with deferred replay (design notes section 3.2).
// The committed state lags the committed text by the `pending` rows of the last verify (rows 0 .. m). The next verify
// first applies the pending rows to the committed state (kda_step each, outputs discarded), stores that state once,
// then runs its new rows on a working copy and discards the copy: one state read and one write per cycle, no
// per-row snapshots, and on every rejection pattern the state sequence is the plain run's kda_step sequence.
struct KdaRow {  // one row's rank-local pre-conv inputs (KdaInputs storage)
    std::vector<float> qkv, f, gate, b;
    KdaInputs in() const { return {qkv, f, gate, b}; }
};
class KdaVerify {
public:
    explicit KdaVerify(KdaState s) : st_(std::move(s)) {}
    // Replays the pending rows into the committed state, then runs `rows` (in order) from it; y[r] = kda output.
    // Returns the first row whose C2 checkpoint fired (rows.size() if none); y holds the rows before it. A replay row
    // that fires throws (it is committed: plain greedy fails there too).
    unsigned run(std::span<const KdaRow> rows, const KdaWeights& w, std::vector<std::vector<float>>& y);
    void commit(unsigned n_rows);  // the first n rows of the last run become pending (n <= rows run)
    void flush(const KdaWeights& w);  // pending -> committed (session end, prefix snapshot, switch to plain decode)
    const KdaState& committed() const { return st_; }
    unsigned pending() const { return static_cast<unsigned>(pending_.size()); }
    unsigned replays() const { return replays_; }  // rows replayed so far (diagnostic)

private:
    KdaState st_;
    std::vector<KdaRow> pending_, cand_;
    unsigned replays_ = 0;
};

// ---- DSA rows of one slot: appends R positions c.tokens() .. +R-1 in order (= R sequential dsa_append; the device
// form stores all rows first, then completes the pools they close). Rollback is DsaCache::truncate(frontier).
// latents [R][512] (normed), x [R][in]. Returns the first row whose C3/C4 checkpoint fired (R if none); that row and the
// later ones are not appended (the device stores them and flags them; they are never committed without an abort).
unsigned dsa_append_rows(DsaCache& c, std::span<const float> latents, std::span<const float> x, unsigned in,
                         const IndexerWeights& w, unsigned R);

// ---- MTP draft layer (design notes): per row the pair (h_p, x_{p+1}),
//   z = eh_proj([enorm(embed(x)) | hnorm(h)]); r = z + o_proj(MLA(input_ln(z))); y = r + MoE(post_ln(r));
//   h' = shared_head.norm(y); logits = lm_head(h'). No mHC. MTP row j (pair (h_j, x_{j+1})) sits at MTP cache index j.
// EXL3 sites are callbacks (Tier B: device strix_gemv). `reduce` is the TP sum R17 of a rank partial (identity on one
// rank). Heads/experts/vocab are the rank's slice; the indexer, q_a, kv_a and eh_proj are replicated.
using Linear = std::function<void(std::span<const float> x, std::span<float> y)>;
struct MtpWeights {
    std::function<void(std::uint32_t tok, std::span<float> e)> embed;  // [4096] row of the shared embedding
    Vec enorm, hnorm, input_ln, post_ln, head_norm;                     // BF16 [4096]
    Linear eh_proj;                                                     // 8192 -> 4096 (K5)
    Linear q_a, q_b, kv_a, o_proj, idx_wq_b;                            // 4096->1536, 1536->heads*256, 4096->512, heads*256->4096, 1536->4096
    Vec q_a_norm, kv_a_norm;                                            // BF16 [1536], [512]
    Mat kv_b;                                                           // [heads * 512][512] F16, rank rows
    IndexerWeights idx;
    Mat router;                                                         // [288][4096] F16
    const float* router_bias = nullptr;                                 // [288]
    std::function<void(std::uint32_t e, std::span<const float> x, std::span<float> y)> expert;  // rank partial of expert e
    Linear shared;                                                      // rank partial of the shared expert
    Linear head;                                                        // rank vocab slice logits
    std::uint32_t vocab0 = 0, vocab_n = kVocab;                         // the rank's vocab slice [vocab0, +vocab_n)
    unsigned heads = kMlaHeads;
    std::function<void(std::span<float>)> reduce;                       // R17 over ranks (nullptr = one rank)
};
struct MtpOut {
    std::vector<float> h;  // h' [4096]: recycled as the next step's hidden
    ArgMax top;            // rank-local argmax (merge across ranks with argmax_merge)
};
// Indexer sharing across draft steps (config index_share_for_mtp_iteration = true, vLLM/SGLang; design notes 6.1 default):
// the catch-up pass (step 0) selects per row; each later single-row draft step reuses the pools selected by the last
// catch-up row (P0 complete pools at its position) and attends to them plus every position from 4 * P0 to its own
// (the tail grown by the draft rows: <= 3 + K - 1 positions). Up to 2,051 visible tokens this equals fresh selection.
struct MtpShare {
    bool on = true;                    // false: fresh selection per row (llama.cpp variant)
    unsigned P0 = 0;                   // set by the last row of a catch-up pass
    std::vector<std::uint32_t> pools;  // its selected pool tokens (ascending, 4 per pool)
    bool valid = false;
};
// One MTP pass over R rows of one slot: rows appended at MTP positions c.tokens() .. +R-1, each row attends to MTP
// positions <= its own. share == nullptr or !share->on: fresh selection per row. With sharing, a pass with
// draft_step = false (catch-up) selects per row and records the last row's pools; draft_step = true (R == 1) reuses them.
std::vector<MtpOut> mtp_forward(DsaCache& c, std::span<const float> h /*[R][4096]*/, std::span<const std::uint32_t> x,
                                const MtpWeights& w, MtpShare* share = nullptr, bool draft_step = false);

}  // namespace strix::glm::spec
