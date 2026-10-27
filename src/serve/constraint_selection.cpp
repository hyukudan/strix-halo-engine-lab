#include "strix/serve/constraint_selection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <type_traits>

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
        const auto t0 = std::chrono::steady_clock::now();
        struct Timed { const std::chrono::steady_clock::time_point& t0; std::uint64_t& ns;
            ~Timed() { ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()); } }
            timed{t0, profile_.mask_ns};
        p.mask = allowed(*spec_->program, state.cursor, *trie_, spec_->eos_ids);
        if (spec_->end_thinking_id) p.mask.set(*spec_->end_thinking_id, false);
        if (std::none_of(p.mask.words.begin(), p.mask.words.end(), [](auto word) { return word != 0; }))
            throw ConstraintError("empty constraint mask");
    }
    return p;
}
ConstraintSelection::Staged ConstraintSelection::select(std::span<const float> logits, const Prepared& p) const {
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed { const std::chrono::steady_clock::time_point& t0; Profile& p;
        ~Timed() { ++p.rows; p.select_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()); } }
        timed{t0, profile_};
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
    advance(staged, p.state, force);
    return staged;
}
void ConstraintSelection::advance(Staged& staged, const State& prior, bool force) const {
    const auto id = staged.token;
    if (spec_->transform.active()) ++staged.state.reply_counts[id];
    staged.state.frontier = std::make_shared<Frontier>(Frontier{prior.frontier});
    const bool eos = std::find(spec_->eos_ids.begin(), spec_->eos_ids.end(), id) != spec_->eos_ids.end();
    if (eos) staged.state.terminal = true;
    else if (prior.reasoning) {
        if (force) {
            staged.state.forced_ids.push_back(id);
            if (staged.state.forced_ids.size() == spec_->thinking->closure_ids.size()) staged.state.reasoning = false;
        } else if (spec_->end_thinking_id == id) staged.state.reasoning = false;
        else ++staged.state.reasoning_tokens;
    } else if (spec_->program) staged.state.cursor = advance_token(prior.cursor, tokenizer_->token_bytes(id));
}
bool ConstraintSelection::admits(const State& state, std::uint32_t id) const {
    // prepare(): the walk admits trie tokens whose bytes all advance the cursor, EOS IDs are then set to
    // cursor.complete(), and the end-of-thinking ID is cleared last.
    if (spec_->end_thinking_id && *spec_->end_thinking_id == id) return false;
    if (std::find(spec_->eos_ids.begin(), spec_->eos_ids.end(), id) != spec_->eos_ids.end()) return state.cursor.complete();
    if (!trie_->contains(id)) return false;
    auto cursor = state.cursor;
    for (unsigned char b : tokenizer_->token_bytes(id)) if (!cursor.advance(b)) return false;
    return true;
}
std::optional<ConstraintSelection::Staged> ConstraintSelection::select_known(const State& state, std::uint32_t token, std::uint32_t status) const {
    // A nonfinite row, a terminal state and score or transform requests take the full-row path (its errors and
    // their order stay those of prepare()/select()).
    if (!argmax_first_ || status != 0 || token >= trie_->vocab_size() || state.terminal || spec_->logprobs || spec_->transform.active())
        return std::nullopt;
    const bool force = state.reasoning && spec_->thinking && state.reasoning_tokens >= spec_->thinking->reasoning_limit;
    std::uint32_t id = token;
    if (force) id = spec_->thinking->closure_ids.at(state.forced_ids.size());
    else if (!state.reasoning && spec_->program && !admits(state, token)) return std::nullopt;
    ++profile_.known_rows;
    Staged staged{id, state, force, {id, {}, {}, force}};
    advance(staged, state, force);
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

void ConstraintSelection::publish_prefix(const std::shared_ptr<Frontier>& base, const std::vector<Staged>& records) {
    if (base != state_.frontier) throw ConstraintError("stale or empty constraint window");
    if(records.empty())return;
    auto parent = base;
    for (const auto& record : records) {
        if (!record.state.frontier || record.state.frontier->parent != parent)
            throw ConstraintError("noncontiguous constraint window");
        parent = record.state.frontier;
    }
    // All copying/allocation and lock acquisition precede publication. Each score
    // move and the final state installation are nonthrowing after reserve.
    auto final_state = records.back().state;
    std::vector<TokenScore> scores;
    if (spec_->logprobs) {
        scores.reserve(records.size());
        for (const auto& record : records) scores.push_back(record.score);
    }
    static_assert(std::is_nothrow_move_constructible_v<TokenScore>);
    static_assert(std::is_nothrow_move_assignable_v<State>);
    const auto install = [&]() noexcept {
        for (auto& score : scores) spec_->scores->tokens.push_back(std::move(score));
        state_ = std::move(final_state);
        state_.frontier->parent.reset();
    };
    if (spec_->logprobs) {
        std::lock_guard lock(spec_->scores->mutex);
        auto& tokens = spec_->scores->tokens;
        if (scores.size() > tokens.max_size() - tokens.size()) throw std::length_error("constraint score storage overflow");
        tokens.reserve(tokens.size() + scores.size());
        install();
    } else install();
}

class ConstraintWindowReceipt final : public MtpSelectionReceipt {
public:
    explicit ConstraintWindowReceipt(std::shared_ptr<ConstraintSelection> selection,
        std::shared_ptr<ConstraintSelection::Frontier> base)
        : selection_(std::move(selection)), base_(std::move(base)) {}
    void publish() override {
        if (published_) throw ConstraintError("already published constraint window");
        selection_->publish_prefix(base_, records_);
        published_ = true;
    }
    std::size_t size() const noexcept override { return records_.size(); }
private:
    friend class ConstraintWindowSelection;
    // Binding the request owner, not a slot address, is the request-generation
    // identity: a receipt can never publish into a replacement slot's selector.
    std::shared_ptr<ConstraintSelection> selection_;
    std::shared_ptr<ConstraintSelection::Frontier> base_;
    std::vector<ConstraintSelection::Staged> records_;
    bool published_ = false;
};

ConstraintSelection::Staged ConstraintSelection::select_row(std::span<const float> row, const State& state) const {
    if (argmax_first_ && row.size() == trie_->vocab_size() && !row.empty()) {
        // The host form of the device reduction: lowest-ID maximum (both zeros equal), status 1 on any nonfinite.
        std::uint32_t best = 0, status = 0;
        for (std::uint32_t i = 0; i < row.size(); ++i) {
            if (!std::isfinite(row[i])) { status = 1; break; }
            if (row[i] > row[best]) best = i;
        }
        if (auto known = select_known(state, best, status)) return std::move(*known);
    }
    return select(row, prepare(state));
}
std::unique_ptr<MtpSelectionTransaction> ConstraintSelection::window_transaction() {
    return std::make_unique<ConstraintWindowSelection>(shared_from_this());
}
ConstraintWindowSelection::ConstraintWindowSelection(std::shared_ptr<ConstraintSelection> selection)
    : selection_(std::move(selection)) {
    if (!selection_) throw std::invalid_argument("null constraint selection");
    reset();
}
void ConstraintWindowSelection::reset() {
    auto state = selection_->state();
    discard();
    base_ = state.frontier;
    branch_ = std::move(state);
}
void ConstraintWindowSelection::discard() noexcept { records_.clear(); branch_.reset(); base_.reset(); }
std::uint32_t ConstraintWindowSelection::select(std::span<const float> row) try {
    if (!branch_) throw ConstraintError("inactive constraint window");
    auto staged = selection_->select_row(row, *branch_);
    auto next = staged.state;
    const auto id = staged.token;
    records_.push_back(std::move(staged));
    *branch_ = std::move(next);
    return id;
} catch(const ConstraintError& e) {throw MtpSelectionError(e.what());}
std::optional<std::uint32_t> ConstraintWindowSelection::select_known(std::uint32_t token, std::uint32_t status) try {
    if (!branch_) throw ConstraintError("inactive constraint window");
    auto staged = selection_->select_known(*branch_, token, status);
    if (!staged) return std::nullopt;
    auto next = staged->state;
    const auto id = staged->token;
    records_.push_back(std::move(*staged));
    *branch_ = std::move(next);
    return id;
} catch(const ConstraintError& e) {throw MtpSelectionError(e.what());}
std::unique_ptr<MtpSelectionReceipt> ConstraintWindowSelection::seal() {
    if (!branch_) throw ConstraintError("inactive or empty constraint window");
    auto receipt = std::make_unique<ConstraintWindowReceipt>(selection_, base_);
    receipt->records_.swap(records_);
    discard();
    return receipt;
}

ConstraintVerdict ConstraintSelection::verdict(std::string finish, std::string execution) const {
    ConstraintVerdict v;
    v.profile_rows = profile_.rows; v.profile_window_rows = profile_.window_rows; v.profile_known_rows = profile_.known_rows;
    v.profile_mask_ms = static_cast<double>(profile_.mask_ns) / 1e6; v.profile_select_ms = static_cast<double>(profile_.select_ns) / 1e6;
    v.profile_window_row_ms = static_cast<double>(profile_.window_row_ns) / 1e6;
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
