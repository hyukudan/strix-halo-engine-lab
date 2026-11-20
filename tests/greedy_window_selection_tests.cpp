// Device-greedy selection (strix::select_greedy_window + serve::ArgmaxConstraintRequests): a thinking budget applied to
// windows of verified rows of which the host sees only each row's argmax (the RTX serve engine).
// Invariants: (1) with nothing forced, the selected prefix is the greedy rule argmax[0, 1 + accepted drafts);
// (2) the emitted ids do not depend on the window shape (drafts, their errors, window length): every windowed run
// equals the serial run; (3) the budget emits exactly reasoning_limit reasoning tokens, then the closure ids, then
// the answer; (4) a forced token equal to the next draft keeps the window going; (5) admission refuses every
// policy select_known cannot decide; (6) the end-thinking predicate (one for the API and every engine).
#include "strix/mtp_selection_transaction.hpp"
#include "strix/serve/constraint_selection.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix;
using namespace strix::serve;

namespace {
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}

// Vocabulary: 0..3 ordinary, 4 = </think> (control), 5 = separator, 6 = EOS (control).
std::shared_ptr<const BpeTokenizer> tokenizer(std::int32_t think_type = 3, const char* think = "</think>") {
    std::vector<std::string> tokens{"a", "b", "c", "d", think, "\xC4\x8A", "<eos>"};
    return BpeTokenizer::from_parts(tokens, {1, 1, 1, 1, think_type, 1, 3}, {}, "qwen35");
}
constexpr std::uint32_t kEnd = 4, kSep = 5, kEos = 6;

// The "model": a row's argmax depends only on its input token (never EOS, never </think> by itself).
std::uint32_t next_of(std::uint32_t t) { return t < 4 ? (t + 1) % 4 : (t == kEnd ? kSep : 2U); }

std::shared_ptr<ConstraintSpec> budget_spec(std::uint32_t limit) {
    auto spec = std::make_shared<ConstraintSpec>();
    spec->eos_ids = {kEos};
    spec->end_thinking_id = kEnd;
    spec->starts_in_reasoning = true;
    spec->thinking = ThinkingPolicy{limit, 8, limit, {kEnd, kSep}};
    return spec;
}

// Runs `total` emitted tokens from pending `first`. Window w drafts `depth(w)` tokens: the model's own next tokens,
// with draft `wrong(w)` (if < depth) replaced by a wrong one. Returns the emitted ids.
template <class Depth, class Wrong>
std::vector<std::uint32_t> run(ArgmaxConstraintRequests& requests, std::uint32_t limit, std::size_t total,
                               Depth depth, Wrong wrong, bool* forced = nullptr) {
    auto selection = requests.make(budget_spec(limit));
    auto tx = selection->window_transaction();
    std::vector<std::uint32_t> out;
    std::uint32_t pending = 0;
    for (std::size_t w = 0; out.size() < total; ++w) {
        std::vector<std::uint32_t> rows{pending};
        for (std::uint32_t d = 0; d < depth(w); ++d) {
            auto t = next_of(rows.back());
            if (d == wrong(w)) t = (t + 2) % 4;
            rows.push_back(t);
        }
        std::vector<std::uint32_t> argmax;
        for (const auto t : rows) argmax.push_back(next_of(t));
        auto g = select_greedy_window(*tx, std::span<const std::uint32_t>(rows).subspan(1), argmax);
        check(!g.ids.empty() && g.ids.size() <= rows.size(), "prefix within the window");
        for (std::size_t r = 1; r < g.ids.size() && r < rows.size(); ++r) check(g.ids[r - 1] == rows[r], "every kept row's input was the draft");
        g.receipt->publish();  // after the "commit" of g.ids.size() rows
        pending = g.ids.back();
        out.insert(out.end(), g.ids.begin(), g.ids.end());
    }
    out.resize(total);
    const auto v = selection->verdict("length", "mtp-host");
    if (forced) *forced = v.closure == "forced";
    check(v.reasoning_tokens == std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(total)), "reasoning tokens counted");
    return out;
}

void greedy_rule_when_nothing_forced() {
    ArgmaxConstraintRequests requests(tokenizer());
    auto selection = requests.make(budget_spec(1000));
    auto tx = selection->window_transaction();
    // Every draft pattern of a 5-row window over a 4-token alphabet prefix: the selected prefix is the greedy rule.
    for (std::uint32_t pattern = 0; pattern < 256; ++pattern) {
        std::vector<std::uint32_t> drafts, argmax;
        for (std::uint32_t r = 0; r < 4; ++r) drafts.push_back((pattern >> (2 * r)) & 3U);
        for (std::uint32_t r = 0; r < 5; ++r) argmax.push_back((pattern * 7U + r * 3U) % 4U);
        std::size_t keep = 1;
        while (keep < 5 && drafts[keep - 1] == argmax[keep - 1]) ++keep;
        auto g = select_greedy_window(*tx, drafts, argmax);
        check(g.ids == std::vector<std::uint32_t>(argmax.begin(), argmax.begin() + static_cast<std::ptrdiff_t>(keep)),
              "greedy rule");
        g.receipt.reset();  // not published: the next attempt starts at the same committed frontier
    }
    check(selection->state().reasoning_tokens == 0, "unpublished receipts change nothing");
}

void window_invariance_and_closure() {
    ArgmaxConstraintRequests requests(tokenizer());
    for (std::uint32_t limit : {1U, 2U, 5U, 9U, 17U}) {
        bool forced = false;
        const auto serial = run(requests, limit, 40, [](std::size_t) { return 0U; }, [](std::size_t) { return 99U; }, &forced);
        check(forced, "closure forced");
        // reasoning: the model's own chain from pending 0, then the closure ids, then the answer chain from kSep
        std::vector<std::uint32_t> expect;
        std::uint32_t t = 0;
        for (std::uint32_t i = 0; i < limit; ++i) expect.push_back(t = next_of(t));
        expect.push_back(kEnd);
        expect.push_back(kSep);
        t = kSep;
        while (expect.size() < 40) expect.push_back(t = next_of(t));
        check(serial == expect, "serial run: limit reasoning tokens, closure, answer");
        for (std::uint32_t seed = 0; seed < 64; ++seed) {
            const auto windowed = run(requests, limit, 40,
                [seed](std::size_t w) { return static_cast<std::uint32_t>((seed + w * 5U) % 8U); },
                [seed](std::size_t w) { return static_cast<std::uint32_t>((seed * 3U + w) % 11U); });
            check(windowed == serial, "window shape does not change the ids");
        }
    }
}

void forced_token_equal_to_draft() {
    // limit 2: rows [p=0, 1, 2, </think>, sep, ...]: drafts that predict the closure are accepted through it.
    ArgmaxConstraintRequests requests(tokenizer());
    auto selection = requests.make(budget_spec(2));
    auto tx = selection->window_transaction();
    const std::vector<std::uint32_t> drafts{1, 2, kEnd, kSep}, argmax{1, 2, 3, kSep, 2};
    auto g = select_greedy_window(*tx, drafts, argmax);
    check((g.ids == std::vector<std::uint32_t>{1, 2, kEnd, kSep, 2}), "forced ids equal to drafts keep the window");
    g.receipt->publish();
    check(!selection->state().reasoning && selection->state().forced_ids.size() == 2, "closure complete");
    // and a draft that misses the forced token ends the window at the forced row
    auto other = requests.make(budget_spec(2));
    auto t2 = other->window_transaction();
    auto h = select_greedy_window(*t2, std::vector<std::uint32_t>{1, 2, 3}, std::vector<std::uint32_t>{1, 2, 3, 0});
    check((h.ids == std::vector<std::uint32_t>{1, 2, kEnd}), "forced row ends the window");
}

void admission_and_shapes() {
    ArgmaxConstraintRequests requests(tokenizer());
    auto spec = budget_spec(4);
    check(ArgmaxConstraintRequests::admits(*spec), "thinking-only admitted");
    auto with_scores = std::make_shared<ConstraintSpec>(*spec);
    with_scores->logprobs = true;
    auto with_bias = std::make_shared<ConstraintSpec>(*spec);
    with_bias->transform.bias.emplace_back(1U, 2.0);
    for (const auto& bad : {with_scores, with_bias}) {
        check(!ArgmaxConstraintRequests::admits(*bad), "argmax-undecidable refused");
        bool threw = false;
        try { (void)requests.make(bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "make refuses");
    }
    check(!requests.make(nullptr), "no spec, no selection");
    auto tx = requests.make(spec)->window_transaction();
    bool threw = false;
    try { (void)select_greedy_window(*tx, std::vector<std::uint32_t>{1, 2}, std::vector<std::uint32_t>{1, 2}); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "one argmax per input row");
}

void closure_predicate() {
    const std::vector<std::uint32_t> none, stops{kEnd};
    check(thinking_closure_usable(*tokenizer(), none), "control </think> usable");
    check(thinking_closure_usable(*tokenizer(4), none), "user-defined </think> usable");
    check(!thinking_closure_usable(*tokenizer(), stops), "a stop token cannot close thinking");
    check(!thinking_closure_usable(*tokenizer(1), none), "ordinary text is not a closure");
    check(!thinking_closure_usable(*tokenizer(3, "</thinking>"), none), "missing </think>");
}
}  // namespace

int main() {
    try {
        greedy_rule_when_nothing_forced();
        window_invariance_and_closure();
        forced_token_equal_to_draft();
        admission_and_shapes();
        closure_predicate();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL after %d checks: %s\n", checks, e.what());
        return 1;
    }
    std::printf("greedy window selection tests: %d checks passed\n", checks);
    return 0;
}
