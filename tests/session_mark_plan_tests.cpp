// Shared prompt-state mark plan (include/strix/session/mark_plan.hpp):
// 1. without a root it returns exactly the previous CUDA Qwen capture points (reference copied below), exhaustively;
// 2. every plan is ascending, unique, block-aligned, above the restore, at most the end mark; the root mark is
//    strictly inside the root prefix;
// 3. no-model replay: conversations sharing a system + tools prefix, a token-exact checkpoint store with the same
//    restore rule as the CUDA Qwen cache (longest checkpoint whose prefix and boundary token match, at most n - 2);
//    restored tokens per request with and without the root mark.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <iterator>
#include <string>
#include <vector>

#include "strix/cuda/qwen_prefix_plan.hpp"
#include "strix/session/mark_plan.hpp"

namespace {

using Marks = std::vector<std::uint64_t>;

void check(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

// The capture points before the shared plan (include/strix/cuda/qwen_prefix_plan.hpp at c14e1ed1d).
Marks reference_points(std::uint64_t tokens, std::uint32_t block, std::uint64_t restored, std::uint64_t stride) {
    const std::uint64_t limit = tokens >= 2 ? tokens - 2 : 0;
    const auto last = limit / block * block;
    Marks points;
    if (last <= restored) return points;
    if (stride) {
        for (auto p = (restored / stride + 1) * stride; p < last; p += stride) points.push_back(p);
    }
    points.push_back(last);
    return points;
}

strix::session::MarkPlanRequest request(std::uint64_t tokens, std::uint64_t restored, std::uint32_t block,
                                        std::uint64_t stride, std::uint64_t root) {
    strix::session::MarkPlanRequest r;
    r.tokens = tokens;
    r.restored = restored;
    r.block = block;
    r.stride = stride;
    r.root = root;
    r.tail = 2;
    return r;
}

void no_root_equals_reference() {
    for (const std::uint32_t block : {4U, 8U}) {
        for (std::uint64_t stride = 0; stride <= 24; stride += block) {
            for (std::uint64_t tokens = 0; tokens <= 90; ++tokens) {
                for (std::uint64_t restored = 0; restored <= tokens; ++restored) {
                    const auto want = reference_points(tokens, block, restored, stride);
                    check(strix::session::plan_marks(request(tokens, restored, block, stride, 0)) == want,
                          "shared plan without root differs from the previous points");
                    check(strix::cuda::qwen_prefix::capture_points(tokens, block, restored, stride) == want,
                          "CUDA capture_points without root differs from the previous points");
                }
            }
        }
    }
}

void plan_properties() {
    std::uint64_t with_root = 0;
    for (const std::uint32_t block : {4U, 8U}) {
        for (std::uint64_t stride = 0; stride <= 24; stride += block) {
            for (std::uint64_t tokens = 0; tokens <= 70; ++tokens) {
                for (std::uint64_t restored = 0; restored <= tokens; ++restored) {
                    const auto base = reference_points(tokens, block, restored, stride);
                    for (std::uint64_t root = 0; root <= tokens + 2; ++root) {
                        const auto marks = strix::session::plan_marks(request(tokens, restored, block, stride, root));
                        check(std::is_sorted(marks.begin(), marks.end()) &&
                                  std::adjacent_find(marks.begin(), marks.end()) == marks.end(), "marks not ascending/unique");
                        for (const auto m : marks)
                            check(m % block == 0 && m > restored && m + 2 <= tokens, "mark not aligned / above restore / within limit");
                        // the plan is the previous points plus at most the root mark
                        Marks extra;
                        std::set_difference(marks.begin(), marks.end(), base.begin(), base.end(), std::back_inserter(extra));
                        check(std::includes(marks.begin(), marks.end(), base.begin(), base.end()), "a previous point was dropped");
                        check(extra.size() <= 1, "more than one extra mark");
                        const auto rm = strix::session::root_mark(root, tokens, block, 2);
                        if (!extra.empty()) {
                            ++with_root;
                            check(extra[0] == rm && rm < root, "extra mark is not the root mark strictly inside the root");
                        }
                        if (rm > restored && !base.empty()) check(std::binary_search(marks.begin(), marks.end(), rm), "root mark missing");
                    }
                }
            }
        }
    }
    check(with_root > 0, "no plan used a root mark");
    check(strix::session::root_mark(6000, 9000, 256, 2) == 5888, "root 6000 / block 256");
    check(strix::session::root_mark(6144, 9000, 256, 2) == 5888, "aligned root stays strictly inside");
    check(strix::session::root_mark(6000, 5000, 256, 2) == 4864, "root clipped to the restore limit");
    check(strix::session::root_mark(0, 9000, 256, 2) == 0 && strix::session::root_mark(200, 9000, 256, 2) == 0, "no root");
    bool threw = false;
    try { (void)strix::session::plan_marks(request(40, 0, 0, 0, 0)); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "block 0 accepted");
    threw = false;
    try { (void)strix::session::plan_marks(request(40, 0, 4, 6, 0)); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "stride not a multiple of block accepted");
}

// Token-exact checkpoint store with the CUDA Qwen restore rule.
struct Store {
    struct Checkpoint {
        std::vector<std::uint32_t> prefix;  // tokens [0, p)
        std::uint32_t boundary = 0;         // token p (recorded by checkpoints with a draft head)
    };
    std::vector<Checkpoint> checkpoints;

    std::uint64_t restore(const std::vector<std::uint32_t>& prompt) const {
        const auto limit = strix::session::mark_limit(prompt.size(), 2);
        std::uint64_t best = 0;
        for (const auto& c : checkpoints) {
            const auto p = c.prefix.size();
            if (p <= best || p > limit || prompt[p] != c.boundary) continue;
            if (std::equal(c.prefix.begin(), c.prefix.end(), prompt.begin())) best = p;
        }
        return best;
    }
    void capture(const std::vector<std::uint32_t>& prompt, std::uint64_t p) {
        for (const auto& c : checkpoints)
            if (c.prefix.size() == p && std::equal(c.prefix.begin(), c.prefix.end(), prompt.begin())) return;
        Checkpoint c;
        c.prefix.assign(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(p));
        c.boundary = prompt[p];
        checkpoints.push_back(std::move(c));
    }
};

struct ReplayResult {
    std::uint64_t prompt_tokens = 0, restored_tokens = 0, first_turn_restored = 0, marks = 0;
};

// `conversations` conversations, `turns` turns each, all starting with the same `system` tokens; the turns of all
// conversations interleave (agent sub-sessions). User and reply ids are private to each conversation.
ReplayResult replay(bool root_on, std::uint64_t system, std::uint32_t conversations, std::uint32_t turns,
                    std::vector<std::uint64_t>* per_request = nullptr) {
    constexpr std::uint32_t block = 256;
    constexpr std::uint64_t stride = 16384, user = 1500, reply = 800;
    Store store;
    ReplayResult out;
    std::vector<std::vector<std::uint32_t>> history(conversations);
    std::uint32_t next_id = 1000000;
    for (auto& h : history)
        for (std::uint64_t i = 0; i < system; ++i) h.push_back(static_cast<std::uint32_t>(i + 1));
    for (std::uint32_t t = 0; t < turns; ++t) {
        for (std::uint32_t c = 0; c < conversations; ++c) {
            auto& h = history[c];
            for (std::uint64_t i = 0; i < user; ++i) h.push_back(next_id++);
            auto prompt = h;
            prompt.push_back(7);  // generation prompt tokens: assistant header
            prompt.push_back(8);
            const auto restored = store.restore(prompt);
            const auto marks = strix::session::plan_marks(request(prompt.size(), restored, block, stride, root_on ? system : 0));
            for (const auto m : marks) store.capture(prompt, m);
            out.prompt_tokens += prompt.size();
            out.restored_tokens += restored;
            out.marks += marks.size();
            if (t == 0) out.first_turn_restored += restored;
            if (per_request) per_request->push_back(restored);
            h.push_back(7);
            h.push_back(8);
            for (std::uint64_t i = 0; i < reply; ++i) h.push_back(next_id++);
        }
    }
    return out;
}

void sibling_replay() {
    for (const std::uint64_t system : {6000ULL, 20000ULL}) {
        std::vector<std::uint64_t> off_requests, on_requests;
        const auto off = replay(false, system, 4, 3, &off_requests);
        const auto on = replay(true, system, 4, 3, &on_requests);
        for (std::size_t i = 0; i < off_requests.size(); ++i)
            check(on_requests[i] >= off_requests[i], "the root mark restored fewer tokens for some request");
        // first turns of conversations 2..4 restore the root: (S - 1) rounded down to 256
        const auto root = strix::session::root_mark(system, system + 1502, 256, 2);
        check(on.first_turn_restored == 3 * root, "siblings did not resume at the root mark");
        std::printf("mark-plan replay: system=%llu conversations=4 turns=3 prompt_tokens=%llu restored root_off=%llu "
                    "root_on=%llu (re-read %llu -> %llu, first turns %llu -> %llu) marks %llu -> %llu\n",
                    static_cast<unsigned long long>(system), static_cast<unsigned long long>(on.prompt_tokens),
                    static_cast<unsigned long long>(off.restored_tokens), static_cast<unsigned long long>(on.restored_tokens),
                    static_cast<unsigned long long>(off.prompt_tokens - off.restored_tokens),
                    static_cast<unsigned long long>(on.prompt_tokens - on.restored_tokens),
                    static_cast<unsigned long long>(off.first_turn_restored),
                    static_cast<unsigned long long>(on.first_turn_restored),
                    static_cast<unsigned long long>(off.marks), static_cast<unsigned long long>(on.marks));
    }
}

}  // namespace

int main() {
    try {
        no_root_equals_reference();
        plan_properties();
        sibling_replay();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "session mark plan tests FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("session mark plan tests passed\n");
    return 0;
}
