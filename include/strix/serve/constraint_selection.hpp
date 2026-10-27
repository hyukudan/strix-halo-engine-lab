#pragma once

#include <stdexcept>
#include <unordered_map>
#include "strix/serve/engine.hpp"
#include "strix/serve/json_constraint.hpp"

namespace strix::serve {

// A branch is a copy of the committed output frontier. Selection advances only
// that branch; publish it with commit only after the model prefix commits.
class ConstraintSelection {
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
    [[nodiscard]] const State& state() const { return state_; }
    [[nodiscard]] ConstraintVerdict verdict(std::string finish, std::string execution) const;
private:
    std::shared_ptr<const ConstraintSpec> spec_;
    std::shared_ptr<const TokenByteTrie> trie_;
    std::shared_ptr<const BpeTokenizer> tokenizer_;
    State state_;
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
    ConstraintVerdict verdict(const std::string& finish) const { return selection_->verdict(finish, "serial-host"); }
private:
    std::shared_ptr<ConstraintSelection> selection_;
    std::optional<ConstraintSelection::Prepared> prepared_;
    std::optional<ConstraintSelection::Staged> staged_;
};

}  // namespace strix::serve
