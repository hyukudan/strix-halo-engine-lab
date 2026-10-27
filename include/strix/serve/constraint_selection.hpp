#pragma once

#include <stdexcept>
#include <unordered_map>
#include "strix/mtp_selection_transaction.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/json_constraint.hpp"

namespace strix::serve {

class ConstraintWindowSelection;
class ConstraintWindowReceipt;

// A branch is a copy of the committed output frontier. Selection advances only
// that branch; publish it with commit only after the model prefix commits.
class ConstraintSelection : public std::enable_shared_from_this<ConstraintSelection> {
public:
    struct Frontier {
        std::shared_ptr<Frontier> parent;
    };
    struct State {
        std::unordered_map<std::uint32_t, std::uint64_t> reply_counts;
        bool reasoning;
        ConstraintCursor cursor;
        bool terminal = false;
        std::uint32_t reasoning_tokens = 0;
        std::vector<std::uint32_t> forced_ids{};
        std::shared_ptr<Frontier> frontier = std::make_shared<Frontier>();
    };
    struct Prepared {
        State state;
        AllowedTokens mask;
    };
    struct Staged {
        std::uint32_t token;
        State state;
        bool forced_control = false;
        TokenScore score;
    };
    ConstraintSelection(std::shared_ptr<const ConstraintSpec>, std::shared_ptr<const TokenByteTrie>,
        std::shared_ptr<const BpeTokenizer>);
    [[nodiscard]] Prepared prepare() const;
    [[nodiscard]] Prepared prepare(const State&) const;
    [[nodiscard]] Staged select(std::span<const float>, const Prepared&) const;
    // Argmax-first (set_argmax_first(true)): `token` / `status` are the row's unmasked lowest-ID maximum and
    // its finite status (0: every logit finite). When the result provably equals select() on the full row -
    // finite row, no scores or transforms, and the token is the forced/unmasked choice or admitted by the
    // cursor - this stages it without the row; otherwise nullopt and the caller selects on the full row.
    // Every route selects through it (select_row), so serial and window rows behave the same, including
    // mask-walk evaluation limits that only a full mask walk reaches.
    [[nodiscard]] std::optional<Staged> select_known(const State&, std::uint32_t token, std::uint32_t status) const;
    // One row on any route: argmax-first when enabled, else (or on decline) prepare() + select().
    [[nodiscard]] Staged select_row(std::span<const float>, const State&) const;
    // prepare(state).mask.contains(id) for an answer-phase program state (not reasoning, not terminal) and an ID
    // inside the vocabulary, without building the mask; a full walk may additionally stop on evaluation limits.
    [[nodiscard]] bool admits(const State&, std::uint32_t id) const;
    void set_argmax_first(bool on) noexcept { argmax_first_ = on; }
    void commit(const Staged&);
    [[nodiscard]] std::unique_ptr<MtpSelectionTransaction> window_transaction();
    [[nodiscard]] const State& state() const { return state_; }
    [[nodiscard]] ConstraintVerdict verdict(std::string finish, std::string execution) const;
private:
    friend class ConstraintWindowReceipt;
    friend class ConstraintWindowSelection;
    struct Profile { std::uint64_t rows = 0, window_rows = 0, known_rows = 0, mask_ns = 0, select_ns = 0, window_row_ns = 0; };
    void advance(Staged&, const State& prior, bool force) const;
    mutable Profile profile_;
    bool argmax_first_ = false;
    void publish_prefix(const std::shared_ptr<Frontier>&, const std::vector<Staged>&);
    std::shared_ptr<const ConstraintSpec> spec_;
    std::shared_ptr<const TokenByteTrie> trie_;
    std::shared_ptr<const BpeTokenizer> tokenizer_;
    State state_;
};

// One request-owned attempt. Sealing transfers all records to a receipt; reset
// starts a new attempt at the current committed frontier, including after replay.
class ConstraintWindowSelection final : public MtpSelectionTransaction {
public:
    explicit ConstraintWindowSelection(std::shared_ptr<ConstraintSelection>);
    void reset() override;
    void discard() noexcept override;
    std::uint32_t select(std::span<const float>) override;
    std::optional<std::uint32_t> select_known(std::uint32_t token, std::uint32_t status) override;
    std::unique_ptr<MtpSelectionReceipt> seal() override;
    void note_row_ns(std::uint64_t ns) noexcept override { ++selection_->profile_.window_rows; selection_->profile_.window_row_ns += ns; }
private:
    std::shared_ptr<ConstraintSelection> selection_;
    std::optional<ConstraintSelection::State> branch_;
    std::shared_ptr<ConstraintSelection::Frontier> base_;
    std::vector<ConstraintSelection::Staged> records_;
};

// One serial output transaction. A per-call target callback stages the token;
// the runner commits it only after receiving the actual model ticket. Destruction
// drops any uncommitted selection (cancellation, failure or slot reuse).
class ConstraintRowSelection {
public:
    explicit ConstraintRowSelection(std::shared_ptr<ConstraintSelection> selection) : selection_(std::move(selection)) {}
    // Starts a row; the mask (when needed) is built with the row, after target execution.
    void prepare() { staged_.reset(); }
    std::uint32_t select(std::span<const float> row) {
        staged_ = std::make_unique<ConstraintSelection::Staged>(selection_->select_row(row, selection_->state()));
        return staged_->token;
    }
    void commit() {
        if (!staged_) throw std::logic_error("constraint row did not select a token");
        selection_->commit(*staged_); staged_.reset();
    }
    [[nodiscard]] std::unique_ptr<MtpSelectionTransaction> window_transaction() {return selection_->window_transaction();}
    void set_execution(std::string route,std::string fallback={}) {execution_=std::move(route);fallback_=std::move(fallback);}
    ConstraintVerdict verdict(const std::string& finish) const {auto v=selection_->verdict(finish,execution_);v.execution_fallback=fallback_;return v;}
private:
    std::shared_ptr<ConstraintSelection> selection_;
    std::string execution_="serial-host",fallback_;
    // Heap-held: GCC 13 -O2 flags an optional<Staged> member as maybe-uninitialized.
    std::unique_ptr<ConstraintSelection::Staged> staged_;
};

}  // namespace strix::serve
