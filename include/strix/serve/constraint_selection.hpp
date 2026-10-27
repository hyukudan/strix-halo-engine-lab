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
    void commit(const Staged&);
    [[nodiscard]] std::unique_ptr<MtpSelectionTransaction> window_transaction();
    [[nodiscard]] const State& state() const { return state_; }
    [[nodiscard]] ConstraintVerdict verdict(std::string finish, std::string execution) const;
private:
    friend class ConstraintWindowReceipt;
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
    std::unique_ptr<MtpSelectionReceipt> seal() override;
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
    void prepare() { staged_.reset(); prepared_ = selection_->prepare(); }
    std::uint32_t select(std::span<const float> row) {
        if (!prepared_) prepare();
        staged_ = selection_->select(row, *prepared_);
        return staged_->token;
    }
    void commit() {
        if (!staged_) throw std::logic_error("constraint row did not select a token");
        selection_->commit(*staged_); staged_.reset(); prepared_.reset();
    }
    [[nodiscard]] std::unique_ptr<MtpSelectionTransaction> window_transaction() {return selection_->window_transaction();}
    void set_execution(std::string route,std::string fallback={}) {execution_=std::move(route);fallback_=std::move(fallback);}
    ConstraintVerdict verdict(const std::string& finish) const {auto v=selection_->verdict(finish,execution_);v.execution_fallback=fallback_;return v;}
private:
    std::shared_ptr<ConstraintSelection> selection_;
    std::string execution_="serial-host",fallback_;
    std::optional<ConstraintSelection::Prepared> prepared_;
    std::optional<ConstraintSelection::Staged> staged_;
};

}  // namespace strix::serve
