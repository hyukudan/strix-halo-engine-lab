#pragma once

// Native-MTP ChainDrafter glue shared by every backend that has an engine with an MTP layer (CUDA CudaEngine, HIP HipEngine).
// Home: shared (host spec layer): the ring / carry / catch-up logic is device-agnostic; a backend supplies only
//   Engine::Rows / Engine::MtpOut   output structs (device pointers: top, hidden_pre, hidden_post)
//   Engine::config().hidden, Engine::read_top(Rows, n, host) (synchronizes), Engine::mtp_forward(next_tokens, hidden_dev, rows, pos0, want_top)
//   Dev::alloc(engine, floats) / Dev::release(engine, ptr) / Dev::copy(engine, dst, src, floats)   (async device->device on the engine stream)
//
// LAZY catch-up: the hidden rows of every committed position are kept in a device ring (with their successor tokens) and
// written into the MTP KV only when a draft needs them, so copy-only stretches pay no MTP cost. Chain steps feed the MTP
// layer's own hidden back ('post' = after shared_head_norm, 'pre' = block output). Exactness never depends on the drafts
// (BlockSpeculator verifies every id); only acceptance does.
//
// DEVICE FEED (optional backend hooks, detected by MtpFeedEngine): a chain step's input id is the previous step's draft, so
// it can stay on the device. The backend then supplies
//   Engine::kMtpFeedSlots, Engine::mtp_step_fed(in_host | nullptr, in_slot, out_slot, hidden_dev, pos, want_top),
//   Engine::mtp_feed_put(slot, top_dev) (async device copy), Engine::mtp_feed_read(n, host) (synchronizes),
// and a draft of k positions is enqueued without host round trips and read back once, instead of one synchronizing
// read per position. Same calls in the same order with the same operands: the drafted ids are bitwise those of the
// host-fed chain (tests/spec_mtp_chain_tests.cpp). Feed slot layout per draft: 0 = the flushed prediction (d1),
// 1 = the last prefix step's prediction, 1 + j = chain step j's draft.
// VERIFY PRE-LAUNCH (optional, detected by MtpPrelaunchEngine: mtp_feed_read_async(n) + mtp_feed_wait(n, host)):
// draft_prelaunch() enqueues the same draft, queues the ids' copy to host staging, hands the slot map to the caller's
// launch (which enqueues the verify behind the draft), and only then waits for the copy.
#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "strix/spec/block_drafter.hpp"

namespace strix::spec {

template <class E>
concept MtpFeedEngine = requires(E& e, const std::uint32_t* t, std::uint32_t* h, const float* x, std::uint32_t u) {
    { E::kMtpFeedSlots } -> std::convertible_to<std::uint32_t>;
    e.mtp_step_fed(t, u, u, x, u, true);
    e.mtp_feed_put(u, t);
    e.mtp_feed_read(u, h);
};
// VERIFY PRE-LAUNCH hooks (strix/spec/block_drafter.hpp): mtp_feed_read_async(n) enqueues the copy of slots 0..n-1 to host
// staging (no wait); mtp_feed_wait(n, host) waits for that copy only (not for work enqueued after it) and fills host[0..n).
template <class E>
concept MtpPrelaunchEngine = MtpFeedEngine<E> && requires(E& e, std::uint32_t* h, std::uint32_t u) {
    e.mtp_feed_read_async(u);
    e.mtp_feed_wait(u, h);
};

// Optional runner-up hook (token-tree siblings): the feed slots' second ids, read with the ids in one synchronization.
template <class E>
concept MtpRunnerUpEngine = MtpFeedEngine<E> && requires(E& e, std::uint32_t* h, std::uint32_t u) { e.mtp_feed_read2(u, h, h); };
// ... and the runner-ups' drafter logit gaps (top-1 minus runner-up; the sibling selector), in the same synchronization.
template <class E>
concept MtpGapEngine = MtpRunnerUpEngine<E> && requires(E& e, std::uint32_t* h, float* g, std::uint32_t u) { e.mtp_feed_read2(u, h, h, g); };

// KV-ONLY ROWS (optional backend hook, detected by MtpKvRowsEngine): Engine::mtp_forward(next_tokens, hidden_dev, rows, pos0,
// want_top, outputs) where only the trailing `outputs` rows' top / hidden are read. A catch-up flush reads nothing but the
// last row of its final chunk (d1 + carry), so every other row only has to write the MTP layer's KV: a one-layer block's
// cache row depends on the row's input alone, so a backend can skip the query side and the FFN of those rows and keep
// its caches bitwise the full form's (the drafts and every proposal unchanged). Engines without the hook run every row.
template <class E>
concept MtpKvRowsEngine = requires(E& e, const std::uint32_t* t, const float* x, std::uint32_t u) {
    e.mtp_forward(t, x, u, u, true, u);
};

// K/V-ONLY CATCH-UP (optional backend hook, detected by MtpKvEngine): Engine::mtp_kv(next_tokens, hidden_dev, rows, pos0)
// writes exactly the MTP K/V that mtp_forward of the same rows writes and nothing else. A single-layer MTP's K/V at a
// position depend only on that row's layer input, so catch-up chunks whose block output and draft are never read skip the
// attention, o_proj, MLP and head (a long prompt: every chunk but the last).
template <class E>
concept MtpKvEngine = requires(E& e, const std::uint32_t* t, const float* x, std::uint32_t u) { e.mtp_kv(t, x, u, u); };

template <class Engine, class Dev>
class MtpChain final : public ChainDrafter {
public:
    using Rows = typename Engine::Rows;
    using MtpOut = typename Engine::MtpOut;

    // mtp_rows = widest mtp_forward call (engine limit); ring = positions buffered before a forced flush.
    MtpChain(Engine& e, bool pre_hidden, std::uint32_t mtp_rows, std::uint32_t ring = 512)
        : e_(e), pre_(pre_hidden), hid_(e.config().hidden), rows_max_(std::min<std::uint32_t>(mtp_rows, 32U)), ring_n_(ring) {
        ring_ = Dev::alloc(e_, std::size_t(ring_n_) * hid_);
        carry_ = Dev::alloc(e_, hid_);
        next_.reserve(ring_n_);
    }
    ~MtpChain() override {
        if (ring_) Dev::release(e_, ring_);
        if (carry_) Dev::release(e_, carry_);
    }
    MtpChain(const MtpChain&) = delete;
    MtpChain& operator=(const MtpChain&) = delete;
    void set_row_rule(const RowRule& rule) override { rule_ = rule; }

    // New sequence (the engine is reset by the caller): forget the MTP positions.
    void reset() {
        next_.clear();
        mtp_pos_ = 0;
        d1_ = 0;
        d1_ru_ = kNoToken;
        d1_gap_ = -1.0f;
        ru_.clear();
        gap_.clear();
        last_rows_ = 0;
    }
    // The last verify pass (rows of [anchor, drafts...]); accept() needs its hidden rows.
    void observe_verify(const Rows& o, std::uint32_t rows) {
        last_ = o;
        last_rows_ = rows;
    }
    // A prefill window of `rows` committed rows: row i's successor = tokens[i + 1]; the last row's = next_after.
    void observe_prefill(const Rows& o, std::uint32_t rows, const std::uint32_t* tokens, std::uint32_t next_after) {
        succ_.resize(rows);
        for (std::uint32_t i = 0; i + 1 < rows; ++i) succ_[i] = tokens[i + 1];
        succ_[rows - 1] = next_after;
        push_hidden(o, rows, succ_.data());
    }

    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        const auto t0 = std::chrono::steady_clock::now();
        out.clear();
        ru_.clear();  // runner-ups: device-fed drafts on an engine with the hook only
        gap_.clear();  // their gaps: an engine with the gap hook only
        if constexpr (MtpFeedEngine<Engine>) {
            if (device_feed() && std::size_t(k) + 1 <= Engine::kMtpFeedSlots) {
                draft_fed(prefix, k, out);
                draft_wall_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ++drafts_;
                return;
            }
        }
        if (!next_.empty()) flush();
        // carry_ = MTP hidden after the last committed row; d1_ predicts the token immediately after the anchor.
        std::uint32_t pos = mtp_pos_;  // speculative rows from here on are overwritten by the next flush
        std::uint32_t tok = d1_;       // chain token for the next step when there is no prefix
        for (std::size_t i = 0; i < prefix.size(); ++i)  // feed the copy tokens: only the last step's prediction is used
            tok = step(prefix[i], pos++, i + 1 == prefix.size());
        for (std::uint32_t j = 0; j < k; ++j) {
            out.push_back(tok);
            if (j + 1 < k) tok = step(tok, pos++, true);
        }
        draft_wall_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++drafts_;
    }
    void accept(std::span<const std::uint32_t> emitted) override {
        if (last_rows_ == 0) return;
        // rows 0..emitted-1 of the last verify are committed; emitted[i] follows row i
        push_hidden(last_, static_cast<std::uint32_t>(emitted.size()), emitted.data());
    }
    // Tree verify: the committed rows are the path's rows (emitted[i] follows verify row rows[i]); each run of consecutive
    // rows is one copy, so a linear path (0, 1, ..) is the same single copy as accept().
    void accept_path(std::span<const std::uint32_t> emitted, std::span<const std::uint32_t> rows) override {
        if (last_rows_ == 0) return;
        if (rows.size() != emitted.size()) throw std::invalid_argument("MtpChain: path rows != emitted tokens");
        for (std::size_t i = 0; i < rows.size();) {
            std::size_t j = i + 1;
            while (j < rows.size() && rows[j] == rows[j - 1] + 1) ++j;
            if (rows[j - 1] >= last_rows_) throw std::invalid_argument("MtpChain: path row outside the last verify");
            push_hidden(last_, static_cast<std::uint32_t>(j - i), emitted.data() + i, rows[i]);
            i = j;
        }
    }

    [[nodiscard]] std::uint64_t calls() const { return calls_; }    // mtp_forward calls (flush chunks + chain steps)
    [[nodiscard]] std::uint64_t rows() const { return rows_; }
    [[nodiscard]] std::uint64_t drafts() const { return drafts_; }  // draft() invocations
    [[nodiscard]] double draft_wall_ms() const { return draft_wall_ms_; }
    // Device feed (on by default where the engine has the hooks); false = one synchronizing read per position (rollback).
    void set_device_feed(bool on) { feed_ = on; }
    [[nodiscard]] bool device_feed() const { return MtpFeedEngine<Engine> && feed_ && !coupled_host_draft(); }
    [[nodiscard]] std::uint64_t host_reads() const { return reads_; }  // synchronizing id reads issued by the chain
    // Verify pre-launch: draft_fed() (same calls, same ids), with launch(prefix, feed) called after the draft is enqueued
    // and the ids' copy to host staging is queued, before waiting for that copy.
    bool draft_prelaunch(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out,
                         const PrelaunchFn& launch) override {
        if constexpr (MtpPrelaunchEngine<Engine>) {
            if (!device_feed() || k == 0 || std::size_t(k) + 1 > Engine::kMtpFeedSlots || k > DraftFeed::kMax) return false;
            const auto t0 = std::chrono::steady_clock::now();
            out.clear();
            draft_fed(prefix, k, out, &launch);
            draft_wall_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++drafts_;
            ++prelaunched_;
            return true;
        } else {
            (void)prefix;
            (void)k;
            (void)out;
            (void)launch;
            return false;
        }
    }
    [[nodiscard]] std::uint64_t prelaunched() const { return prelaunched_; }  // draft_prelaunch() drafts
    // K/V-only catch-up (engines with MtpKvEngine): off = the incumbent calls (rollback); on = catch-up chunks whose outputs
    // nobody reads only write the MTP K/V (same K/V bits, so the same drafts). On by default where the engine has the hook;
    // its device gate compares the MTP K/V bytes of the prompt rows, ids, steps and the k histogram with the full catch-up
    // under a wall-independent controller (CUDA sub15d --cost-kv off: an online context cost prices k from the run's own
    // verify walls, so k may differ between any two runs; the K/V and the drafts do not).
    void set_kv_catchup(bool on) { kv_catchup_ = on; }
    [[nodiscard]] bool kv_catchup() const { return MtpKvEngine<Engine> && kv_catchup_; }
    [[nodiscard]] std::uint64_t kv_calls() const { return kv_calls_; }  // K/V-only catch-up calls (not in calls())

private:
    // Optional synchronized host read of one logical output row. ids is empty for the full vocabulary, or maps
    // each compact logit to its token. Id-only engines keep deterministic drafts under sampling. Device-side
    // coupled drafting belongs to the sampler owner's kernel (phase 2).
    static constexpr bool has_draft_logits = requires(Engine& e, const MtpOut& out, std::uint32_t row,
                                                     std::vector<float>& logits, std::vector<std::uint32_t>& ids) {
        e.read_mtp_logits(out, row, logits, ids);
    };
    [[nodiscard]] bool coupled_host_draft() const { return has_draft_logits && rule_.sampled(); }

    std::uint32_t read_draft(const MtpOut& out, std::uint32_t rows, std::uint64_t position) {
        ++reads_;
        if constexpr (has_draft_logits) {
            if (rule_.sampled()) {
                draft_logits_.clear();
                draft_ids_.clear();
                e_.read_mtp_logits(out, rows - 1, draft_logits_, draft_ids_);
                return rule_.draft_token(draft_logits_, draft_ids_, position);
            }
        }
        Rows ro{};
        ro.top = out.top;
        e_.read_top(ro, rows, top_tmp_);
        return top_tmp_[rows - 1];
    }

    // draft() over the device feed: the same mtp calls in the same order, ids read once at the end (DEVICE FEED above).
    // launch (pre-launch only): called once after the copy of the ids to host staging is queued, before the wait.
    void draft_fed(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out,
                   const PrelaunchFn* launch = nullptr) {
        const bool flushed = !next_.empty();
        if (flushed) flush(true);  // d1 -> feed slot 0
        std::uint32_t pos = mtp_pos_;
        bool first_dev = flushed;  // draft 0 lives in feed slot first_slot, or in host d1_
        std::uint32_t first_slot = 0;
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            const bool last = i + 1 == prefix.size();
            fed_step(&prefix[i], 0, 1, pos++, last);
            if (last) {
                first_dev = true;
                first_slot = 1;
            }
        }
        for (std::uint32_t j = 1; j < k; ++j)
            fed_step(j == 1 && !first_dev ? &d1_ : nullptr, j == 1 ? first_slot : j, 1 + j, pos++, true);
        std::uint32_t ids[Engine::kMtpFeedSlots]{}, ru[Engine::kMtpFeedSlots];
        float gp[Engine::kMtpFeedSlots];
        for (auto& u : ru) u = kNoToken;  // runner-ups / gaps come only from mtp_feed_read2 (the pre-launch read carries none)
        for (auto& g : gp) g = -1.0f;
        const bool read = first_dev || k > 1;
        if constexpr (MtpPrelaunchEngine<Engine>) {
            if (launch) {
                // Draft j: slot first_slot (j == 0 on the device) or host d1_ (j == 0 not on the device); slot 1 + j after.
                DraftFeed f;
                f.k = k;
                f.slot[0] = first_dev ? first_slot : DraftFeed::kHost;
                f.host_id[0] = first_dev ? 0U : d1_;
                for (std::uint32_t j = 1; j < k; ++j) f.slot[j] = 1 + j;
                if (read) e_.mtp_feed_read_async(k + 1);
                (*launch)(prefix, f);
                if (read) {
                    e_.mtp_feed_wait(k + 1, ids);
                    ++reads_;
                }
            }
        }
        if (read && launch == nullptr) {
            if constexpr (MtpGapEngine<Engine>) e_.mtp_feed_read2(k + 1, ids, ru, gp);
            else if constexpr (MtpRunnerUpEngine<Engine>) e_.mtp_feed_read2(k + 1, ids, ru);
            else e_.mtp_feed_read(k + 1, ids);
            ++reads_;
        }
        if (flushed) {
            d1_ = ids[0];
            d1_ru_ = MtpRunnerUpEngine<Engine> ? ru[0] : kNoToken;
            d1_gap_ = MtpGapEngine<Engine> ? gp[0] : -1.0f;
        }
        out.push_back(first_dev ? ids[first_slot] : d1_);
        for (std::uint32_t j = 1; j < k; ++j) out.push_back(ids[1 + j]);
        if constexpr (MtpRunnerUpEngine<Engine>) {
            ru_.push_back(first_dev ? ru[first_slot] : d1_ru_);
            for (std::uint32_t j = 1; j < k; ++j) ru_.push_back(ru[1 + j]);
        }
        if constexpr (MtpGapEngine<Engine>) {
            gap_.push_back(first_dev ? gp[first_slot] : d1_gap_);
            for (std::uint32_t j = 1; j < k; ++j) gap_.push_back(gp[1 + j]);
        }
    }

public:
    [[nodiscard]] std::span<const std::uint32_t> runner_ups() const override { return ru_; }
    [[nodiscard]] std::span<const float> runner_up_gaps() const override { return gap_; }

private:
    void fed_step(const std::uint32_t* in_host, std::uint32_t in_slot, std::uint32_t out_slot, std::uint32_t pos, bool want_top) {
        if constexpr (MtpFeedEngine<Engine>) {
            const auto m = e_.mtp_step_fed(in_host, in_slot, out_slot, carry_, pos, want_top);
            ++calls_;
            ++rows_;
            Dev::copy(e_, carry_, pre_ ? m.hidden_pre : m.hidden_post, hid_);
        }
    }
    void push_hidden(const Rows& o, std::uint32_t rows, const std::uint32_t* next_tokens, std::uint32_t first = 0) {
        const float* src = (pre_ ? o.hidden_pre : o.hidden_post) + std::size_t(first) * hid_;
        std::uint32_t done = 0;
        while (done < rows) {
            if (next_.size() == ring_n_) flush(false, false);  // forced: its last-row outputs are overwritten by a later flush
            const std::uint32_t n = std::min<std::uint32_t>(rows - done, ring_n_ - static_cast<std::uint32_t>(next_.size()));
            Dev::copy(e_, ring_ + next_.size() * hid_, src + std::size_t(done) * hid_, std::size_t(n) * hid_);
            next_.insert(next_.end(), next_tokens + done, next_tokens + done + n);
            done += n;
        }
    }
    // Write the ring into the MTP KV (rows_max_ per call, causal inside each call). Leaves the last row's top / hidden in
    // d1_ / carry_ (fed: d1 goes to feed slot 0 without a host read; draft_fed() reads it with the drafts).
    // outputs = false: nobody reads this flush's last-row outputs (a ring-full flush inside push_hidden; d1_ / carry_ are
    // rewritten by the flush draft() runs first). With K/V-only catch-up (MtpKvEngine) every chunk whose outputs have no
    // reader writes only the MTP K/V (same K/V bits); without it the calls are exactly the incumbent ones.
    void flush(bool fed = false, bool outputs = true) {
        std::uint32_t off = 0;
        const auto total = static_cast<std::uint32_t>(next_.size());
        const bool kv = kv_catchup();
        while (off < total) {
            const std::uint32_t n = std::min<std::uint32_t>(rows_max_, total - off);
            const bool final_chunk = off + n == total;  // only the last row's draft is needed: earlier chunks skip lm_head
            if (kv && !(final_chunk && outputs)) {
                if constexpr (MtpKvEngine<Engine>) e_.mtp_kv(next_.data() + off, ring_ + std::size_t(off) * hid_, n, mtp_pos_);
                ++kv_calls_;
                rows_ += n;
                mtp_pos_ += n;
                off += n;
                continue;
            }
            const auto m = forward_rows(next_.data() + off, ring_ + std::size_t(off) * hid_, n, mtp_pos_, final_chunk);
            ++calls_;
            rows_ += n;
            mtp_pos_ += n;
            off += n;
            if (final_chunk) {
                if constexpr (MtpFeedEngine<Engine>) {
                    if (fed) e_.mtp_feed_put(0, m.top + (n - 1));
                }
                if (!fed) {
                    // The final hidden row is mtp_pos_ - 1. MTP consumes its successor and predicts the following id.
                    d1_ = read_draft(m, n, std::uint64_t(mtp_pos_) + 1);
                    d1_ru_ = kNoToken;  // host-read flush: no runner-up
                    d1_gap_ = -1.0f;
                }
                Dev::copy(e_, carry_, (pre_ ? m.hidden_pre : m.hidden_post) + std::size_t(n - 1) * hid_, hid_);
            }
        }
        next_.clear();
    }
    // A flush chunk: the final chunk's last row is the only output read (outputs 1), earlier chunks read none (outputs 0).
    MtpOut forward_rows(const std::uint32_t* next, const float* hidden, std::uint32_t n, std::uint32_t pos0, bool final_chunk) {
        if constexpr (MtpKvRowsEngine<Engine>) return e_.mtp_forward(next, hidden, n, pos0, final_chunk, final_chunk ? 1U : 0U);
        else return e_.mtp_forward(next, hidden, n, pos0, final_chunk);
    }
    std::uint32_t step(std::uint32_t input_token, std::uint32_t pos, bool want_top) {
        const auto m = e_.mtp_forward(&input_token, carry_, 1, pos, want_top);
        ++calls_;
        ++rows_;
        std::uint32_t t = 0;
        if (want_top) t = read_draft(m, 1, std::uint64_t(pos) + 2);
        Dev::copy(e_, carry_, pre_ ? m.hidden_pre : m.hidden_post, hid_);
        return t;
    }

    Engine& e_;
    RowRule rule_;
    std::vector<float> draft_logits_;
    std::vector<std::uint32_t> draft_ids_;
    bool pre_;
    std::uint32_t hid_, rows_max_, ring_n_;
    Rows last_{};
    std::uint32_t last_rows_ = 0;
    float* ring_ = nullptr;
    float* carry_ = nullptr;
    std::vector<std::uint32_t> next_, succ_;
    std::uint32_t mtp_pos_ = 0, d1_ = 0, d1_ru_ = kNoToken;  // d1_ru_: runner-up of d1_ (device-fed flush only)
    std::vector<std::uint32_t> ru_;                           // runner-ups of the last draft (empty: none)
    float d1_gap_ = -1.0f;                                    // gap of d1_ru_ (device-fed flush on a gap engine only)
    std::vector<float> gap_;                                  // gaps of ru_ (empty: the engine has no gap hook)
    std::uint32_t top_tmp_[32]{};
    std::uint64_t calls_ = 0, rows_ = 0, drafts_ = 0, reads_ = 0, kv_calls_ = 0, prelaunched_ = 0;
    bool feed_ = true;
    bool kv_catchup_ = true;
    double draft_wall_ms_ = 0.0;
};

// Dev glue for an engine that owns its device memory and stream through three members:
//   float* alloc_f32(std::size_t floats), void free_f32(float*), void copy_f32_async(float* dst, const float* src, std::size_t floats)
// (async device -> device on the engine's stream). Any backend whose engine has them binds MtpChain<Engine, EngineChainDev>.
struct EngineChainDev {
    template <class E> static float* alloc(E& e, std::size_t n) { return e.alloc_f32(n); }
    template <class E> static void release(E& e, float* p) { e.free_f32(p); }
    template <class E> static void copy(E& e, float* d, const float* s, std::size_t n) { e.copy_f32_async(d, s, n); }
};

}  // namespace strix::spec
