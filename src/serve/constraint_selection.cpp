#include "strix/serve/constraint_selection.hpp"

#include <algorithm>
#include <cmath>

namespace strix::serve {

ConstraintSelection::ConstraintSelection(std::shared_ptr<const ConstraintSpec> spec,
    std::shared_ptr<const TokenByteTrie> trie, std::shared_ptr<const BpeTokenizer> tokenizer)
    : spec_(std::move(spec)), trie_(std::move(trie)), tokenizer_(std::move(tokenizer)),
      state_{spec_->starts_in_reasoning, ConstraintCursor(*spec_->program)} {}

ConstraintSelection::Prepared ConstraintSelection::prepare() const { return prepare(state_); }
ConstraintSelection::Prepared ConstraintSelection::prepare(const State& state) const {
    if (state.terminal) throw ConstraintError("selection after constraint termination");
    Prepared p{state, {}};
    if (!state.reasoning) {
        p.mask = allowed(*spec_->program, state.cursor, *trie_, spec_->eos_ids);
        if (spec_->end_thinking_id) p.mask.set(*spec_->end_thinking_id, false);
        if (std::none_of(p.mask.words.begin(), p.mask.words.end(), [](auto word) { return word != 0; }))
            throw ConstraintError("empty constraint mask");
    }
    return p;
}
ConstraintSelection::Staged ConstraintSelection::select(std::span<const float> logits, const Prepared& p) const {
    if (logits.size() != trie_->vocab_size()) throw ConstraintError("constraint logits vocabulary mismatch");
    // Preserve the target's all-vocabulary finite-row contract, including masked entries.
    for (const auto value : logits) if (!std::isfinite(value)) throw ConstraintError("nonfinite target logits");
    std::uint32_t id = 0;
    if (p.state.reasoning) {
        if (logits.empty()) throw ConstraintError("empty target logits");
        for (std::size_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[id]) id = static_cast<std::uint32_t>(i);
    } else id = constrained_greedy(logits, p.mask);
    Staged staged{id, p.state};
    staged.state.frontier = std::make_shared<Frontier>(Frontier{p.state.frontier});
    const bool eos = std::find(spec_->eos_ids.begin(), spec_->eos_ids.end(), id) != spec_->eos_ids.end();
    if (eos) staged.state.terminal = true;
    else if (p.state.reasoning) {
        if (spec_->end_thinking_id == id) staged.state.reasoning = false;
    } else staged.state.cursor = advance_token(p.state.cursor, tokenizer_->token_bytes(id));
    return staged;
}
void ConstraintSelection::commit(const Staged& staged) {
    auto ancestor = staged.state.frontier->parent;
    while (ancestor && ancestor != state_.frontier) ancestor = ancestor->parent;
    if (!ancestor) throw ConstraintError("stale or already committed constraint branch");
    state_ = staged.state;
    // Descendants still identify this frontier, but serial requests do not retain
    // a token-count-sized ancestry chain (or recursively destroy one at finish).
    state_.frontier->parent.reset();
}

ConstraintVerdict ConstraintSelection::verdict(std::string finish, std::string execution) const {
    ConstraintVerdict v;
    v.execution = std::move(execution);
    v.phase = state_.reasoning ? "reasoning" : "answer";
    v.accepting = !state_.reasoning && state_.cursor.complete();
    v.status = "incomplete";
    if (finish == "constraint_failure" || finish == "execution_failure") { v.status = "failed"; v.reason = finish; }
    else if (finish == "cancelled" || finish == "context") v.reason = finish;
    else if (state_.terminal && state_.reasoning) v.reason = "reasoning_eos";
    else if (state_.terminal && v.accepting) v.status = "complete";
    else v.reason = "max_output_tokens";
    return v;
}

}  // namespace strix::serve
