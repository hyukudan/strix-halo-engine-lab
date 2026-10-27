#include "strix/serve/constraint_selection.hpp"

#include <algorithm>
#include <cmath>

namespace strix::serve {

ConstraintSelection::ConstraintSelection(std::shared_ptr<const ConstraintSpec> spec,
    std::shared_ptr<const TokenByteTrie> trie, std::shared_ptr<const BpeTokenizer> tokenizer)
    : spec_(std::move(spec)), trie_(std::move(trie)), tokenizer_(std::move(tokenizer)),
      state_{{}, spec_->starts_in_reasoning, ConstraintCursor(spec_->program ? *spec_->program : compile_json_object())} {}

ConstraintSelection::Prepared ConstraintSelection::prepare() const { return prepare(state_); }
ConstraintSelection::Prepared ConstraintSelection::prepare(const State& state) const {
    if (state.terminal) throw ConstraintError("selection after constraint termination");
    Prepared p{state, {}};
    if (!state.reasoning && spec_->program) {
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
    std::vector<float> transformed;
    if (spec_->transform.active()) {
        transformed.assign(logits.begin(), logits.end());
        const auto& t = spec_->transform;
        // logit-transform-v1: sign-sensitive repetition on the raw logit of every token seen in the prompt or the
        // reply, then presence and frequency over reply counts, then bias.
        if (t.repetition != 1) {
            std::vector<bool> seen(logits.size(), false);
            for (auto token : spec_->prompt_ids) seen.at(token) = true;
            for (const auto& [token, count] : p.state.reply_counts) { (void)count; seen.at(token) = true; }
            for (std::size_t i = 0; i < transformed.size(); ++i) if (seen[i])
                transformed[i] = static_cast<float>(transformed[i] > 0 ? transformed[i] / t.repetition : transformed[i] * t.repetition);
        }
        for (const auto& [token, count] : p.state.reply_counts) {
            transformed.at(token) -= static_cast<float>(t.presence);
            transformed.at(token) -= static_cast<float>(t.frequency * static_cast<double>(count));
        }
        for (const auto& [token, bias] : t.bias) transformed.at(token) += static_cast<float>(bias);
        for (auto value : transformed) if (!std::isfinite(value)) throw ConstraintError("nonfinite transformed logits");
        logits = transformed;
    }
    std::uint32_t id = 0;
    const bool force = p.state.reasoning && spec_->thinking &&
        p.state.reasoning_tokens >= spec_->thinking->reasoning_limit;
    if (force) id = spec_->thinking->closure_ids.at(p.state.forced_ids.size());
    else if (p.state.reasoning || !spec_->program) {
        if (logits.empty()) throw ConstraintError("empty target logits");
        for (std::size_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[id]) id = static_cast<std::uint32_t>(i);
    } else id = constrained_greedy(logits, p.mask);
    Staged staged{id, p.state, force, {id, {}, {}, force}};
    if (spec_->logprobs && !force) {
        std::vector<std::uint32_t> order;
        for (std::uint32_t i = 0; i < logits.size(); ++i)
            if (p.mask.words.empty() || p.mask.contains(i)) order.push_back(i);
        double maximum = logits[id], sum = 0;
        for (auto token : order) maximum = std::max(maximum, static_cast<double>(logits[token]));
        for (auto token : order) sum += std::exp(static_cast<double>(logits[token]) - maximum);
        const double normalizer = std::log(sum);
        const auto score = [&](std::uint32_t token) { return (static_cast<double>(logits[token]) - maximum) - normalizer; };
        staged.score.logprob = score(id);
        const auto n = std::min<std::size_t>(spec_->top_logprobs, order.size());
        std::partial_sort(order.begin(), order.begin() + n, order.end(), [&](auto a, auto b) {
            const auto pa = score(a), pb = score(b);
            return pa != pb ? pa > pb : a < b;
        });
        for (std::size_t i = 0; i < n; ++i) staged.score.top.emplace_back(order[i], score(order[i]));
    }
    if (spec_->transform.active()) ++staged.state.reply_counts[id];
    staged.state.frontier = std::make_shared<Frontier>(Frontier{p.state.frontier});
    const bool eos = std::find(spec_->eos_ids.begin(), spec_->eos_ids.end(), id) != spec_->eos_ids.end();
    if (eos) staged.state.terminal = true;
    else if (p.state.reasoning) {
        if (force) {
            staged.state.forced_ids.push_back(id);
            if (staged.state.forced_ids.size() == spec_->thinking->closure_ids.size()) staged.state.reasoning = false;
        } else if (spec_->end_thinking_id == id) staged.state.reasoning = false;
        else ++staged.state.reasoning_tokens;
    } else if (spec_->program) staged.state.cursor = advance_token(p.state.cursor, tokenizer_->token_bytes(id));
    return staged;
}
void ConstraintSelection::commit(const Staged& staged) {
    auto ancestor = staged.state.frontier->parent;
    while (ancestor && ancestor != state_.frontier) ancestor = ancestor->parent;
    if (!ancestor) throw ConstraintError("stale or already committed constraint branch");
    if (spec_->logprobs) {
        std::lock_guard lock(spec_->scores->mutex);
        spec_->scores->tokens.push_back(staged.score);
    }
    state_ = staged.state;
    // Descendants still identify this frontier, but serial requests do not retain
    // a token-count-sized ancestry chain (or recursively destroy one at finish).
    state_.frontier->parent.reset();
}

ConstraintVerdict ConstraintSelection::verdict(std::string finish, std::string execution) const {
    ConstraintVerdict v;
    v.reasoning_tokens = state_.reasoning_tokens;
    v.forced_ids = state_.forced_ids;
    v.closure = state_.forced_ids.empty() ? "natural" : "forced";
    v.execution = std::move(execution);
    v.phase = state_.reasoning ? "reasoning" : "answer";
    v.accepting = !state_.reasoning && (!spec_->program || state_.cursor.complete());
    v.status = "incomplete";
    if (finish == "constraint_failure" || finish == "execution_failure") { v.status = "failed"; v.reason = finish; }
    else if (finish == "cancelled" || finish == "context") v.reason = finish;
    else if (state_.terminal && state_.reasoning) v.reason = "reasoning_eos";
    else if (state_.terminal && v.accepting) v.status = "complete";
    else v.reason = "max_output_tokens";
    return v;
}

}  // namespace strix::serve
