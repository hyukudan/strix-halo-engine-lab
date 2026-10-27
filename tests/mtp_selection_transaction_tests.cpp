#include "strix/mtp_selection_transaction.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/serve/constraint_selection.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>

// Fail individual allocation sites without introducing production test hooks.
thread_local std::ptrdiff_t allocation_countdown = -1;
#if defined(_MSC_VER)
#define TEST_NOINLINE __declspec(noinline)
#else
#define TEST_NOINLINE __attribute__((noinline))
#endif
TEST_NOINLINE void free_test_allocation(void* memory) noexcept { std::free(memory); }
void* operator new(std::size_t size) {
    if (allocation_countdown >= 0 && allocation_countdown-- == 0) { allocation_countdown = -1; throw std::bad_alloc(); }
    if (auto* memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { free_test_allocation(memory); }
void operator delete[](void* memory) noexcept { ::operator delete(memory); }
void operator delete(void* memory, std::size_t) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::size_t) noexcept { ::operator delete(memory); }
#undef TEST_NOINLINE

namespace {
using namespace strix;
using namespace strix::serve;
std::size_t checks = 0, runs = 0, replays = 0;
std::array<bool, 4> rejection_rows{}, eos_rows{}, closure_rows{};
bool crossing_closure = false, truncated_closure = false, bonus_seen = false;
bool natural_transition = false, reasoning_eos = false, divergent_suffix = false;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}
template<class F> void refuses(F&& f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    check(failed, "expected refusal");
}
bool same_double(double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }
void scores_equal(const std::vector<TokenScore>& a, const std::vector<TokenScore>& b) {
    check(a.size() == b.size(), "score count");
    for (std::size_t i = 0; i < a.size(); ++i) {
        check(a[i].id == b[i].id && a[i].forced_control == b[i].forced_control &&
            a[i].logprob.has_value() == b[i].logprob.has_value() && a[i].top.size() == b[i].top.size(), "score record");
        if (a[i].logprob) check(same_double(*a[i].logprob, *b[i].logprob), "chosen score bits");
        for (std::size_t j = 0; j < a[i].top.size(); ++j)
            check(a[i].top[j].first == b[i].top[j].first && same_double(a[i].top[j].second, b[i].top[j].second), "top score bits");
    }
}
struct Fixture {
    std::shared_ptr<const BpeTokenizer> tokenizer;
    std::shared_ptr<const TokenByteTrie> trie;
    explicit Fixture(std::vector<std::string> tokens) {
        for (auto& token : tokens) if (token == " ") token = "\xC4\xA0";
        tokenizer = BpeTokenizer::from_parts(tokens, std::vector<std::int32_t>(tokens.size(), 1), {}, "qwen35");
        trie = std::make_shared<TokenByteTrie>(*tokenizer);
    }
    std::shared_ptr<ConstraintSelection> selection(const std::shared_ptr<ConstraintSpec>& spec) const {
        return std::make_shared<ConstraintSelection>(spec, trie, tokenizer);
    }
};
std::shared_ptr<ConstraintSpec> copy_spec(const ConstraintSpec& spec) {
    auto result = std::make_shared<ConstraintSpec>(spec);
    result->scores = std::make_shared<SelectionScores>();
    return result;
}
void state_equal(const ConstraintSelection& a, const ConstraintSelection& b, const Fixture& f, const ConstraintSpec& spec) {
    const auto& x = a.state(); const auto& y = b.state();
    check(x.reply_counts == y.reply_counts && x.reasoning == y.reasoning && x.terminal == y.terminal &&
        x.reasoning_tokens == y.reasoning_tokens && x.forced_ids == y.forced_ids, "selection state");
    check(x.cursor.complete() == y.cursor.complete() && x.cursor.dead() == y.cursor.dead(), "cursor status");
    if (spec.program) {
        const auto mx = allowed(*spec.program, x.cursor, *f.trie, spec.eos_ids);
        const auto my = allowed(*spec.program, y.cursor, *f.trie, spec.eos_ids);
        check(mx.words == my.words, "cursor vocabulary behavior");
    }
    for (std::uint32_t id = 0; id < f.tokenizer->vocab_size(); ++id) {
        auto cx = x.cursor, cy = y.cursor;
        bool ax = true, ay = true;
        try { cx = advance_token(cx, f.tokenizer->token_bytes(id)); } catch (const ConstraintError&) { ax = false; }
        try { cy = advance_token(cy, f.tokenizer->token_bytes(id)); } catch (const ConstraintError&) { ay = false; }
        check(ax == ay && cx.complete() == cy.complete() && cx.dead() == cy.dead(), "cursor transitions");
    }
    for (const auto* finish : {"length", "stop", "cancelled", "execution_failure"}) {
        const auto vx = a.verdict(finish, "cpu"), vy = b.verdict(finish, "cpu");
        check(vx.status == vy.status && vx.accepting == vy.accepting && vx.phase == vy.phase && vx.reason == vy.reason &&
            vx.closure == vy.closure && vx.forced_ids == vy.forced_ids && vx.reasoning_tokens == vy.reasoning_tokens, "verdict");
    }
}

// Every consumed input contributes to the hash and to a common exactly representable
// offset. Even when an ordering collides, different consumed prefixes change row bytes.
struct Target {
    std::vector<float> ordering;
    std::optional<std::size_t> eos_at, natural_at;
    std::uint32_t eos = 0, natural = 0;
    std::optional<std::uint32_t> preferred = {};
    std::vector<float> row(const std::vector<std::uint32_t>& consumed) const {
        std::uint64_t hash = 0, code = 0;
        for (auto id : consumed) { hash = hash * 17 + id + 1; code = code * (ordering.size() + 1) + id + 1; }
        std::vector<float> logits(ordering.size());
        for (std::size_t i = 0; i < logits.size(); ++i)
            logits[i] = ordering[(i + hash % logits.size()) % logits.size()] + static_cast<float>(code % 65536) * 0.0000152587890625F;
        if (preferred) logits[*preferred] = 20;
        if (eos_at && consumed.size() == *eos_at) logits[eos] = 100;
        if (natural_at && consumed.size() == *natural_at) logits[natural] = 101;
        return logits;
    }
};
struct Model {
    std::vector<std::uint32_t> consumed{1};
    std::uint32_t pending = 0;
    void consume_pending() { consumed.push_back(pending); }
};
std::vector<std::uint32_t> serial_ids(const Fixture& f, const ConstraintSpec& spec, const Target& target, std::size_t budget) {
    auto s = f.selection(copy_spec(spec)); Model m;
    const auto first = s->select(target.row(m.consumed), s->prepare());
    s->commit(first); m.pending = first.token;
    std::vector<std::uint32_t> ids;
    while (ids.size() + 1 < budget && !s->state().terminal) {
        m.consume_pending();
        auto staged = s->select(target.row(m.consumed), s->prepare());
        m.pending = staged.token; ids.push_back(staged.token); s->commit(staged);
    }
    if (ids.empty()) ids.push_back(first.token); // Only used to fill unreachable proposals.
    return ids;
}

// The serial oracle never uses the acceptance walker, staged window states or
// provisional IDs. Its own consumed prefix drives each fresh target invocation.
void compare_run(const Fixture& f, const ConstraintSpec& spec, const Target& target, std::size_t rows,
    std::size_t budget, const std::vector<std::uint32_t>& proposals, bool checkpoint) {
    ++runs;
    auto ws = copy_spec(spec), ss = copy_spec(spec);
    auto window = f.selection(ws), serial = f.selection(ss);
    Model model, reference;
    // The terminal-prompt selection is already emitted but has not been consumed
    // by target execution. Its counts, score and closure/cursor state are live.
    const auto first = window->select(target.row(model.consumed), window->prepare());
    window->commit(first); model.pending = first.token;
    const auto serial_first = serial->select(target.row(reference.consumed), serial->prepare());
    serial->commit(serial_first); reference.pending = serial_first.token;
    std::vector<std::uint32_t> outputs{first.token}, expected{serial_first.token};
    check(outputs == expected, "terminal-prompt IDs");
    state_equal(*window, *serial, f, spec);
    scores_equal(ws->scores->tokens, ss->scores->tokens);
    while (outputs.size() < budget && !window->state().terminal) {
        auto attempt = window->window_transaction();
        const auto base_frontier = window->state().frontier;
        const auto old_scores = ws->scores->tokens;
        const auto snapshot = model;
        // Capture physical rows on the proposed model branch, including rejected
        // suffixes. The reader must never inspect that suffix.
        std::vector<std::vector<float>> physical;
        std::vector<Model> physical_states;
        auto speculative = model;
        for (std::size_t i = 0; i < rows; ++i) {
            speculative.consume_pending(); physical.push_back(target.row(speculative.consumed));
            physical_states.push_back(speculative);
            if (i < proposals.size()) speculative.pending = proposals[i];
        }
        std::size_t reads = 0;
        std::uint32_t previous = 0;
        const auto remaining = budget - outputs.size();
        const auto plan = walk_mtp_selection(proposals, rows, remaining, spec.eos_ids, [&](std::size_t row) {
            check(row == reads, "ordered reader");
            if (row) check(proposals[row - 1] == previous, "reader reached only accepted inputs");
            ++reads;
            previous = attempt->select(physical.at(row));
            return previous;
        });
        check(reads == plan.decided_rows && reads == plan.selected_ids.size() && reads > 0, "explicit decided rows");
        check(window->state().frontier == base_frontier, "selection remains private");
        scores_equal(old_scores, ws->scores->tokens);
        check(plan.accepted_draft_inputs == reads - 1, "accepted consumed draft inputs");
        const auto last = plan.selected_ids.back();
        const bool eos = std::find(spec.eos_ids.begin(), spec.eos_ids.end(), last) != spec.eos_ids.end();
        const auto stop = eos ? MtpSelectionStop::eos : reads == remaining ? MtpSelectionStop::output_limit :
            reads == rows ? MtpSelectionStop::row_limit : reads > proposals.size() ? MtpSelectionStop::no_draft : MtpSelectionStop::rejection;
        check(plan.stop == stop, "stop reason precedence");
        if (plan.stop == MtpSelectionStop::rejection) {
            check(proposals[reads - 1] != last, "correction retained");
            rejection_rows[reads - 1] = true;
        }
        if (eos) eos_rows[reads - 1] = true;
        if (reads == rows && reads == proposals.size() + 1) bonus_seen = true;
        if (window->state().reasoning && !window->state().forced_ids.empty()) crossing_closure = true;
        if (checkpoint) model = speculative;
        const bool replay = checkpoint && (reads != rows || plan.accepted_draft_inputs + 1 != rows);
        std::unique_ptr<MtpSelectionReceipt> receipt;
        if (replay) {
            ++replays;
            model = snapshot;
            attempt->discard();
            scores_equal(old_scores, ws->scores->tokens);
            check(window->state().frontier == base_frontier, "restore before replay");
            attempt->reset();
            for (std::size_t i = 0; i < reads; ++i) {
                model.consume_pending();
                model.pending = attempt->select(target.row(model.consumed));
                check(model.pending == plan.selected_ids[i], "corrected serial replay");
            }
            if (plan.stop == MtpSelectionStop::rejection && reads < rows) {
                auto corrected = model; corrected.consume_pending();
                check(target.row(corrected.consumed) != physical[reads], "rejected suffix differs from corrected continuation");
                divergent_suffix = true;
            }
            receipt = attempt->seal();
        } else {
            if (checkpoint) check(model.consumed == physical_states.back().consumed, "complete checkpoint chunk retained");
            else model = physical_states[reads - 1]; // Commit the captured model input prefix.
            model.pending = plan.selected_ids.back();
            receipt = attempt->seal();
        }
        check(receipt->size() == reads, "receipt length");
        scores_equal(old_scores, ws->scores->tokens);
        for (std::size_t i = 0; i < reads; ++i) {
            reference.consume_pending();
            const auto staged = serial->select(target.row(reference.consumed), serial->prepare());
            if (staged.forced_control && serial->state().forced_ids.empty()) closure_rows[i] = true;
            if (!staged.forced_control && serial->state().reasoning && !staged.state.reasoning) natural_transition = true;
            if (staged.state.terminal && staged.state.reasoning) reasoning_eos = true;
            reference.pending = staged.token; expected.push_back(staged.token); serial->commit(staged);
        }
        receipt->publish();
        outputs.insert(outputs.end(), plan.selected_ids.begin(), plan.selected_ids.end());
        check(outputs == expected && model.pending == reference.pending && model.consumed == reference.consumed, "complete output and model frontier");
        state_equal(*window, *serial, f, spec);
        if (outputs.size() == budget && window->state().reasoning && !window->state().forced_ids.empty()) truncated_closure = true;
        scores_equal(ws->scores->tokens, ss->scores->tokens);
        refuses([&] { receipt->publish(); });
        scores_equal(ws->scores->tokens, ss->scores->tokens);
    }
}

void enumeration() {
    const Fixture f({"a", "b", "c"});
    constexpr std::array<float, 4> values{-1, -0.0F, 0.0F, 1};
    for (unsigned ordering = 0; ordering < 64; ++ordering) {
        Target target{{values[ordering % 4], values[(ordering / 4) % 4], values[ordering / 16]}, {}, {}, 0, 0};
        for (unsigned policy = 0; policy < 4; ++policy) {
            ConstraintSpec spec;
            spec.logprobs = policy != 1; spec.top_logprobs = 3;
            if (policy & 1U) { spec.transform.frequency = .75; spec.transform.presence = .25;
                spec.transform.repetition = 2; spec.transform.bias = {{1, .5}}; spec.prompt_ids = {2}; }
            if (policy & 2U) spec.eos_ids = {2};
            for (std::size_t rows = 1; rows <= 4; ++rows) {
                std::size_t count = 1;
                for (std::size_t i = 1; i < rows; ++i) count *= 3;
                for (std::size_t code = 0; code < count; ++code) {
                    auto digits = code;
                    std::vector<std::uint32_t> proposals(rows - 1);
                    for (auto& id : proposals) { id = static_cast<std::uint32_t>(digits % 3); digits /= 3; }
                    for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, 6, proposals, checkpoint);
                }
                // Enumerate all match/reject bit patterns independently, including
                // deliberately unreachable suffix bits after the first rejection.
                const auto ids = serial_ids(f, spec, target, 6);
                for (unsigned bits = 0; bits < (1U << (rows - 1)); ++bits) {
                    std::vector<std::uint32_t> proposals;
                    for (std::size_t i = 0; i + 1 < rows; ++i) {
                        auto id = ids[i % ids.size()];
                        proposals.push_back((bits & (1U << i)) ? id : (id + 1) % 3);
                    }
                    for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, 6, proposals, checkpoint);
                }
            }
        }
    }
}

void grammar_and_thinking() {
    const Fixture f({"a", "b", "{", "}", "x", "y", "z", " "});
    for (unsigned mask = 0; mask < 3; ++mask) for (unsigned policy = 0; policy < 3; ++policy) {
        ConstraintSpec spec; spec.eos_ids = {6}; spec.end_thinking_id = 4;
        spec.logprobs = policy != 0; spec.top_logprobs = 8;
        if (policy != 1) { spec.transform.presence = .5; spec.transform.frequency = 1; spec.transform.repetition = 2; spec.prompt_ids = {0}; }
        if (mask == 1) spec.program = std::make_shared<ConstraintProgram>(compile_json_object());
        if (mask == 2) spec.program = std::make_shared<ConstraintProgram>(compile_json_schema(parse_json("{\"const\":{}}")));
        for (std::size_t rows = 1; rows <= 4; ++rows) for (std::size_t start = 0; start <= 4; ++start)
            for (std::size_t budget = 1; budget <= 8; ++budget) for (unsigned event = 0; event < 4; ++event) {
                spec.starts_in_reasoning = true;
                spec.thinking = ThinkingPolicy{0, 0, static_cast<std::uint32_t>(start), {4, 5, 5}};
                Target target{{2, 1, 0, -1, -2, -3, -4, -5}, {}, {}, 6, 4};
                target.preferred = 0;
                if (event == 1) target.eos_at = std::max<std::size_t>(1, start);
                if (event == 2) target.natural_at = std::max<std::size_t>(1, start);
                if (event == 3) { target.eos_at = start + 2; target.natural_at = 2; }
                const auto ids = serial_ids(f, spec, target, budget);
                for (std::size_t reject = 0; reject <= rows; ++reject) {
                    std::vector<std::uint32_t> proposals;
                    for (std::size_t i = 0; i + 1 < rows; ++i) {
                        const auto id = ids[i % ids.size()];
                        proposals.push_back(i == reject ? (id + 1) % 8 : id);
                    }
                    for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, budget, proposals, checkpoint);
                }
            }
    }
}

void masked_enumeration() {
    const Fixture f({"{", "}", " ", "z"});
    constexpr std::array<float, 4> values{-1, -0.0F, 0.0F, 1};
    for (unsigned mask = 0; mask < 2; ++mask) for (unsigned ordering = 0; ordering < 64; ++ordering) {
        ConstraintSpec spec; spec.eos_ids = {3}; spec.logprobs = true; spec.top_logprobs = 4;
        spec.program = std::make_shared<ConstraintProgram>(mask ? compile_json_object() : compile_json_schema(parse_json("{\"const\":{}}")));
        spec.transform.bias = {{3, 5}};
        Target target{{values[ordering % 4], values[(ordering / 4) % 4], values[ordering / 16], 10}, {}, {}, 3, 0};
        for (std::size_t rows = 1, count = 1; rows <= 4; ++rows, count *= 4) {
            for (std::size_t code = 0; code < count; ++code) {
                std::vector<std::uint32_t> proposals(rows - 1);
                auto digits = code;
                for (auto& id : proposals) { id = static_cast<std::uint32_t>(digits % 4); digits /= 4; }
                for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, 6, proposals, checkpoint);
            }
        }
    }
}

void allocation_failures() {
    const Fixture f({"a", "b", "c"});
    const std::array<float, 3> row{2, 1, 0};
    for (bool scoring : {false, true}) {
        auto spec = std::make_shared<ConstraintSpec>(); spec->logprobs = scoring; spec->top_logprobs = 3; spec->transform.frequency = 1;
        auto s = f.selection(spec);
        s->commit(s->select(row, s->prepare()));
        auto attempt = s->window_transaction();
        for (unsigned i = 0; i < 4; ++i) attempt->select(row);
        auto receipt = attempt->seal();
        const auto frontier = s->state().frontier;
        const auto before = spec->scores->tokens;
        bool published = false;
        std::size_t failures = 0;
        for (std::ptrdiff_t allocation = 0; allocation < 100 && !published; ++allocation) {
            allocation_countdown = allocation;
            try { receipt->publish(); published = true; } catch (const std::bad_alloc&) { ++failures; }
            allocation_countdown = -1;
            if (!published) {
                check(s->state().frontier == frontier && s->state().reply_counts.size() == 1, "publication allocation failure preserves state");
                scores_equal(before, spec->scores->tokens);
            }
        }
        check(published && failures > 0 && spec->scores->tokens.size() == (scoring ? 5U : 0U), "allocation failures then atomic publication");
        // Reset, row staging and sealing also leave the live request untouched on
        // setup failure; the caller discards any surviving private attempt.
        for (std::ptrdiff_t allocation = 0; allocation < 30; ++allocation) {
            auto tx = s->window_transaction();
            const auto base = s->state().frontier;
            const auto scores = spec->scores->tokens;
            allocation_countdown = allocation;
            try { tx->reset(); tx->select(row); auto unused = tx->seal(); } catch (const std::bad_alloc&) {}
            allocation_countdown = -1;
            check(s->state().frontier == base, "setup failure preserves state");
            scores_equal(scores, spec->scores->tokens);
            tx->discard();
        }
    }
}

void boundaries() {
    const Fixture f({"a", "b", "c"});
    auto spec = std::make_shared<ConstraintSpec>(); spec->logprobs = true; spec->top_logprobs = 3;
    spec->transform.frequency = 1;
    auto selection = f.selection(spec);
    for (const auto& zeros : {std::array<float, 3>{-0.0F, 0.0F, -1}, std::array<float, 3>{0.0F, -0.0F, -1}}) {
        auto signed_spec = copy_spec(*spec); signed_spec->transform = {};
        auto signed_selection = f.selection(signed_spec); auto t = signed_selection->window_transaction();
        check(t->select(zeros) == 0, "signed zero tie lowest ID"); auto r = t->seal(); r->publish();
        auto serial = f.selection(copy_spec(*signed_spec)); serial->commit(serial->select(zeros, serial->prepare()));
        check(signed_spec->scores->tokens[0].top[0].first == 0 && signed_spec->scores->tokens[0].top[1].first == 1, "signed zero top ordering");
    }
    const std::vector<float> row{2, 1.5F, 0};
    auto a = selection->window_transaction(), b = selection->window_transaction();
    check(a->select(row) == 0 && a->select(row) == 1, "penalty counts selected outputs only");
    check(spec->scores->tokens.empty(), "private multirow scores");
    auto receipt = a->seal();
    check(b->select(row) == 0, "independent attempt starts at live frontier");
    auto stale = b->seal(); receipt->publish();
    check(spec->scores->tokens.size() == 2 && spec->scores->tokens[0].id == 0 && spec->scores->tokens[1].id == 1, "every score exactly once");
    auto scores = spec->scores->tokens;
    refuses([&] { stale->publish(); }); refuses([&] { receipt->publish(); });
    scores_equal(scores, spec->scores->tokens);
    refuses([&] { a->select(row); }); refuses([&] { a->seal(); });
    b->reset(); b->select(row); b->discard();
    scores_equal(scores, spec->scores->tokens);
    b->reset();
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        auto invalid = row; invalid[2] = bad;
        refuses([&] { b->select(invalid); });
        scores_equal(scores, spec->scores->tokens);
    }
    refuses([&] { b->select(std::span<const float>(row.data(), 2)); });
    b->select(row); auto valid = b->seal(); valid->publish();
    check(spec->scores->tokens.size() == 3, "failed rows stage no scores");
    // Setup failure and cancellation drop attempts/receipts without publication.
    scores = spec->scores->tokens;
    try { auto setup = selection->window_transaction(); setup->select(row); throw std::runtime_error("setup"); } catch (const std::runtime_error&) {}
    { auto cancel = selection->window_transaction(); cancel->select(row); auto cancelled = cancel->seal(); }
    scores_equal(scores, spec->scores->tokens);
    auto reused = f.selection(copy_spec(*spec));
    auto old = selection->window_transaction(); old->select(row); auto old_receipt = old->seal();
    selection.reset(); old_receipt->publish();
    check(reused->state().reply_counts.empty(), "slot reuse binds receipt to original request");

    const Target target{{2, 1, 0}, {}, {}, 2, 0};
    ConstraintSpec short_spec; short_spec.logprobs = true; short_spec.top_logprobs = 3;
    for (bool checkpoint : {false, true}) compare_run(f, short_spec, target, 4, 5, {}, checkpoint);
    auto interrupted = reused->window_transaction();
    const auto live = reused->state().frontier;
    refuses([&] { (void)walk_mtp_selection(std::array<std::uint32_t, 2>{0, 0}, 3, 3, {}, [&](std::size_t row_index) {
        if (row_index == 1) throw std::runtime_error("row read failure");
        return interrupted->select(std::array<float, 3>{10, 0, -1});
    }); });
    interrupted->discard();
    check(reused->state().frontier == live && reused->state().reply_counts.empty(), "interrupted walker discards private branch");

    std::size_t reads = 0;
    for (std::size_t rows = 0; rows <= 4; ++rows) {
        auto plan = walk_mtp_selection({}, rows, 0, {}, [&](auto) { ++reads; return 0U; });
        check(plan.decided_rows == 0 && plan.stop == MtpSelectionStop::output_limit, "zero budget");
    }
    auto zero = walk_mtp_selection({}, 0, 5, {}, [&](auto) { ++reads; return 0U; });
    check(zero.decided_rows == 0 && reads == 0, "zero rows never read");
    auto bonus = walk_mtp_selection({}, 4, 5, {}, [&](auto) { ++reads; return 0U; });
    check(bonus.decided_rows == 1 && bonus.stop == MtpSelectionStop::no_draft && reads == 1, "no proposals bonus");
    // Poison every unreachable row, for each termination boundary.
    for (std::size_t stop = 0; stop < 4; ++stop) for (unsigned reason = 0; reason < 4; ++reason) {
        const std::array<std::uint32_t, 3> proposals{0, 0, 0};
        const std::array<std::uint32_t, 1> eos{1};
        reads = 0;
        auto plan = walk_mtp_selection(proposals, reason == 2 ? stop + 1 : 4, reason == 1 ? stop + 1 : 8,
            reason == 0 ? std::span<const std::uint32_t>(eos) : std::span<const std::uint32_t>{}, [&](std::size_t i) {
                check(i <= stop, "unreachable poison row reader"); ++reads;
                return i == stop && (reason == 0 || reason == 3) ? 1U : 0U;
            });
        check(plan.decided_rows == stop + 1 && reads == stop + 1, "reachability boundary");
    }
}

void routes_and_reuse() {
    CompiledKnownPrefixWindow owner;
    for(const auto rows:{2U,4U,8U}) {
        owner.logical_rows=rows;
        for(unsigned variant=0;variant<=33U;++variant) {
            owner.r2_variant={static_cast<KnownWindowR2Variant>(variant)};
            check(known_window_policy_serial_rows(owner)==(variant!=2U && variant<=32U),"closed arithmetic recipe admission");
        }
    }
    owner.r2_variant={};owner.logical_rows=16;
    check(!known_window_policy_serial_rows(owner),"tiled owner is not a decode policy owner");
    owner.logical_rows=2;owner.output_policy=KnownWindowOutputPolicy::state_only;
    check(!known_window_policy_serial_rows(owner),"policy needs every reachable full-logit row");
    for(bool enabled:{false,true})for(bool window:{false,true})for(bool serial:{false,true}) {
        const auto route=resolve_mtp_selection_route(enabled,window,serial);
        check(route.mtp==(enabled && window && serial),"route admission");
        const std::string expected=!enabled?"selection-mtp-off":!window?"no-window-owner":!serial?"window-arithmetic-not-serial":"";
        check(std::string(route.fallback?route.fallback:"")==expected,"route fallback precedence");
    }
    const Fixture f({"a","b","c"});
    for(unsigned policy=0;policy<3;++policy) {
        auto spec=std::make_shared<ConstraintSpec>();spec->logprobs=policy!=1;
        if(policy)spec->transform.presence=1;
        auto old=f.selection(spec);ConstraintRowSelection adapter(old);
        adapter.prepare();check(adapter.select(std::array<float,3>{3,2,1})==0,"terminal prompt selection");adapter.commit();
        const auto committed=old->state().frontier;
        adapter.set_execution("mtp-host");
        auto attempt=adapter.window_transaction();attempt->select(std::array<float,3>{1,3,2});
        auto cancelled=attempt->seal();cancelled.reset();
        check(old->state().frontier==committed,"cancelled receipt leaves committed prompt only");
        auto replacement_spec=copy_spec(*spec);auto replacement=f.selection(replacement_spec);
        ConstraintRowSelection reused(replacement);reused.set_execution("serial-host","selection-mtp-off");
        auto stale=adapter.window_transaction();stale->select(std::array<float,3>{1,2,3});auto receipt=stale->seal();
        const auto frontier=replacement->state().frontier;receipt->publish();
        check(replacement->state().frontier==frontier && replacement_spec->scores->tokens.empty(),"old receipt cannot target reused request");
        refuses([&]{receipt->publish();});
        check(adapter.verdict("length").execution=="mtp-host" && reused.verdict("length").execution_fallback=="selection-mtp-off","request route lifecycle");
        auto next=reused.window_transaction();next->select(std::array<float,3>{1,3,2});next->seal()->publish();
        check(replacement->state().frontier!=frontier && (!policy || replacement->state().reply_counts.at(1)==1),"reuse starts from its own frontier");
    }
}

void selection_failures() {
    const Fixture f({"a","b","c"});
    for(std::size_t fail=0;fail<4;++fail)for(unsigned kind=0;kind<4;++kind) {
        auto spec=std::make_shared<ConstraintSpec>();spec->logprobs=true;spec->top_logprobs=3;
        if(kind==2)spec->transform.bias.emplace_back(0U,std::numeric_limits<float>::max());
        auto serial_spec=copy_spec(*spec);auto serial=f.selection(serial_spec);
        auto selection=f.selection(spec);auto tx=selection->window_transaction();
        const auto base=selection->state().frontier;
        const std::array<float,3> good{3,2,1};
        std::size_t reads=0;
        const auto plan=walk_mtp_selection(std::array<std::uint32_t,3>{0,0,0},4,4,{},[&](std::size_t row) {
            check(row<=fail,"failure suffix unreachable");++reads;
            if(row==fail) {
                if(kind==0) {auto bad=good;bad[1]=std::numeric_limits<float>::infinity();return tx->select(bad);}
                if(kind==1) {
                    auto empty_spec=std::make_shared<ConstraintSpec>();
                    empty_spec->program=std::make_shared<ConstraintProgram>(compile_json_object());
                    return f.selection(empty_spec)->window_transaction()->select(good);
                }
                if(kind==2) {auto bad=good;bad[0]=std::numeric_limits<float>::max();return tx->select(bad);}
                throw MtpSelectionError("");
            }
            return tx->select(good);
        });
        check(plan.stop==MtpSelectionStop::selection_failure && plan.selection_failure.has_value(),"explicit selection failure");
        if(kind==3)check(plan.selection_failure->empty(),"empty error text still carries failure");
        check(reads==fail+1 && plan.decided_rows==fail && plan.selected_ids==std::vector<std::uint32_t>(fail,0),"failure retains only decided prefix");
        check(spec->scores->tokens.empty() && selection->state().frontier==base,"failure never partially publishes");
        auto receipt=tx->seal();check(receipt->size()==fail,"failure receipt prefix length");
        for(std::size_t i=0;i<fail;++i)serial->commit(serial->select(good,serial->prepare()));
        receipt->publish();scores_equal(spec->scores->tokens,serial_spec->scores->tokens);
        state_equal(*selection,*serial,f,*spec);
        tx->reset();check(tx->select(good)==0,"failed transaction reusable after reset");
        tx->seal()->publish();serial->commit(serial->select(good,serial->prepare()));
        scores_equal(spec->scores->tokens,serial_spec->scores->tokens);
        state_equal(*selection,*serial,f,*spec);
    }
    // A failed member neither throws out of the shared dispatch nor changes peers.
    for(std::size_t member=0;member<3;++member) {
        const auto plan=walk_mtp_selection(std::array<std::uint32_t,3>{0,0,0},4,4,{},[&](std::size_t row) {
            if(member==1 && row==2)throw MtpSelectionError("member failure");
            return 0U;
        });
        check(plan.decided_rows==(member==1?2U:4U),"cohort failure isolation");
    }
    refuses([&]{walk_mtp_selection({},1,1,{},[](auto)->std::uint32_t{throw std::logic_error("execution invariant");});});
}

void masks_and_cohorts() {
    const Fixture f({"a", "{", "}", "z"});
    auto spec = std::make_shared<ConstraintSpec>(); spec->program = std::make_shared<ConstraintProgram>(compile_json_object());
    spec->logprobs = true; spec->eos_ids = {3};
    auto s = f.selection(spec); auto tx = s->window_transaction();
    std::vector<float> row{100, 2, 1, 90};
    check(tx->select(row) == 1, "masked maximum");
    auto bad = row; bad[0] = std::numeric_limits<float>::quiet_NaN();
    refuses([&] { tx->select(bad); });
    auto receipt = tx->seal(); receipt->publish();
    check(spec->scores->tokens.size() == 1, "reachable nonfinite masked logit stages nothing");
    const Fixture empty({"a", "b"});
    auto es = std::make_shared<ConstraintSpec>(); es->program = spec->program;
    auto empty_selection = empty.selection(es); auto et = empty_selection->window_transaction();
    const auto frontier = empty_selection->state().frontier;
    refuses([&] { et->select(std::array<float, 2>{1, 0}); });
    check(empty_selection->state().frontier == frontier && es->scores->tokens.empty(), "empty mask atomic refusal");

    std::array<std::shared_ptr<ConstraintSelection>, 4> members;
    std::array<std::unique_ptr<MtpSelectionTransaction>, 4> attempts;
    std::array<std::unique_ptr<MtpSelectionReceipt>, 4> receipts;
    for (std::size_t i = 0; i < 3; ++i) { members[i] = f.selection(copy_spec(*spec)); attempts[i] = members[i]->window_transaction(); }
    check(!attempts[3], "ordinary cohort member stays null");
    for (std::size_t i = 0; i < 3; ++i) {
        const auto plan = walk_mtp_selection({}, 1, 1, spec->eos_ids, [&](auto) { return attempts[i]->select(row); });
        check(plan.selected_ids == std::vector<std::uint32_t>{1}, "independent cohort plan");
        receipts[i] = attempts[i]->seal();
    }
    for (std::size_t i : {2U, 0U, 1U}) {
        receipts[i]->publish();
        for (std::size_t j = 0; j < 3; ++j) if (j != i && attempts[j])
            check(!members[j]->state().cursor.complete(), "cohort cursor isolated");
        attempts[i].reset();
    }
    row = {100, 1, 2, 90};
    for (auto& member : members) if (member) {
        auto t = member->window_transaction(); check(t->select(row) == 2, "independent advancing cursor"); auto r = t->seal(); r->publish();
        check(member->state().cursor.complete(), "cohort completed object");
    }
}
}  // namespace

int main() try {
    selection_failures(); routes_and_reuse(); boundaries(); allocation_failures(); masks_and_cohorts(); enumeration(); masked_enumeration(); grammar_and_thinking();
    // A four-row window can reject comparisons at rows 0..2; row 3 is bonus.
    check(rejection_rows[0] && rejection_rows[1] && rejection_rows[2], "rejection coverage");
    check(std::all_of(eos_rows.begin(), eos_rows.end(), [](bool x) { return x; }), "EOS row coverage");
    check(std::all_of(closure_rows.begin(), closure_rows.end(), [](bool x) { return x; }), "closure start row coverage");
    check(crossing_closure, "closure crosses windows");
    check(truncated_closure, "closure truncated by budget");
    check(bonus_seen && replays && divergent_suffix, "bonus and divergent replay coverage");
    check(natural_transition, "natural end-thinking coverage");
    check(reasoning_eos, "early reasoning EOS coverage");
    std::cout << "mtp selection transaction: " << runs << " complete runs, " << replays << " checkpoint replays, " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL after " << runs << " runs / " << checks << " checks: " << e.what() << '\n';
    return 1;
}
