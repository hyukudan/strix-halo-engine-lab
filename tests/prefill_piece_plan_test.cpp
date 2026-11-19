// Host test: the shared prompt piece planner (strix/prefill_piece_plan.hpp) and its two engine wrappers.
// Invariant: plan_blocks (Qwen CUDA MoE block plan) and plan_chunks (CUDA dense chunk plan) return exactly the plans of
// their previous own bodies (kept below as reference oracles) for every input of the sweep, and every plan covers each
// row once with pieces in [1, schedule maximum].
#include "strix/cuda_core/submit_window.hpp"
#include "strix/prefill_piece_plan.hpp"
#include "strix/qwen_prefill_schedule.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
int fails = 0;
long checks = 0;
void check(bool ok, const char* what, std::uint64_t a = 0, std::uint64_t b = 0, std::uint64_t c = 0) {
    ++checks;
    if (!ok && fails++ < 20)
        std::fprintf(stderr, "FAIL: %s (%llu, %llu, %llu)\n", what, static_cast<unsigned long long>(a), static_cast<unsigned long long>(b),
                     static_cast<unsigned long long>(c));
}

using V = std::vector<std::uint32_t>;

// Reference: the previous body of qwen_prefill::plan_blocks (validation is shared and unchanged, so it is called as is).
V ref_plan_blocks(std::uint64_t position, const V& blocks, bool tail_large, std::uint32_t final_rows) {
    strix::cuda::qwen_prefill::validate_blocks(blocks);
    const auto cap = *std::max_element(blocks.begin(), blocks.end());
    if (final_rows > cap) throw std::invalid_argument("final_block_rows exceeds loaded block capacity");
    V plan;
    for (std::uint64_t at = 0; at < position;) {
        auto n = static_cast<std::uint32_t>(std::min<std::uint64_t>(position - at, blocks[std::min(plan.size(), blocks.size() - 1)]));
        plan.push_back(n);
        at += n;
    }
    if (tail_large && plan.size() > 1 && plan.back() < plan[plan.size() - 2]) std::rotate(plan.begin(), plan.end() - 1, plan.end());
    if (final_rows && plan.size() > 1) {
        const auto target = static_cast<std::uint32_t>(std::min<std::uint64_t>(final_rows, position - plan.size() + 1));
        if (target > plan.back()) {
            auto need = target - plan.back();
            for (std::size_t i = plan.size() - 1; i-- > 0 && need;) {
                auto take = std::min(need, plan[i] - 1);
                plan[i] -= take;
                plan.back() += take;
                need -= take;
            }
        } else if (target < plan.back()) {
            auto extra = plan.back() - target;
            for (std::size_t i = 0; i + 1 < plan.size() && extra; ++i) {
                auto take = std::min(extra, cap - plan[i]);
                plan[i] += take;
                plan.back() -= take;
                extra -= take;
            }
        }
    }
    return plan;
}

// Reference: the previous body of cuda_core::plan_chunks.
std::vector<strix::prefill::PieceSpan> ref_plan_chunks(std::uint32_t n, std::uint32_t C, std::uint32_t ramp0, std::uint32_t align) {
    std::vector<strix::prefill::PieceSpan> out;
    if (n == 0 || C == 0) return out;
    if (align == 0) align = 1;
    std::uint32_t p = 0;
    if (ramp0 > 0 && ramp0 < C) {
        std::uint32_t r = ramp0 / align * align;
        if (r == 0) r = align;
        while (r < C && p < n) {
            const std::uint32_t rows = n - p < r ? n - p : r;
            out.push_back({p, rows});
            p += rows;
            r *= 2;
        }
    }
    while (p < n) {
        const std::uint32_t rows = n - p < C ? n - p : C;
        out.push_back({p, rows});
        p += rows;
    }
    return out;
}

std::optional<V> run_blocks(bool reference, std::uint64_t position, const V& blocks, bool tail_large, std::uint32_t final_rows) {
    try {
        return reference ? ref_plan_blocks(position, blocks, tail_large, final_rows)
                         : strix::cuda::qwen_prefill::plan_blocks(position, blocks, tail_large, final_rows);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

bool same_spans(const std::vector<strix::prefill::PieceSpan>& a, const std::vector<strix::prefill::PieceSpan>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].pos != b[i].pos || a[i].rows != b[i].rows) return false;
    return true;
}

bool covers(const V& plan, std::uint64_t rows, std::uint32_t cap) {
    std::uint64_t sum = 0;
    for (const auto n : plan) {
        if (n == 0 || n > cap) return false;
        sum += n;
    }
    return sum == rows;
}

void sweep_blocks() {
    const std::vector<V> schedules{{1024, 2048, 4096, 8192}, {8192}, {3968}, {16384}, {7}, {1, 3}, {5, 2}, {65520}};
    const V finals{0, 1, 2, 100, 1000, 3968, 8192};
    std::vector<std::uint64_t> positions;
    for (std::uint64_t p = 1; p <= 1200; ++p) positions.push_back(p);
    for (std::uint64_t p = 1201; p <= 140000; p += 997) positions.push_back(p);
    for (const std::uint64_t p : {4095ULL, 4096ULL, 4097ULL, 8191ULL, 8192ULL, 8193ULL, 8279ULL, 15360ULL, 32652ULL, 32768ULL, 65536ULL, 131072ULL})
        positions.push_back(p);
    for (const auto& s : schedules) {
        const auto cap = *std::max_element(s.begin(), s.end());
        for (const auto p : positions) {
            if (cap < 16 && p > 600) continue;  // tiny schedules: bounded plan length
            for (const bool tail : {false, true})
                for (const auto f : finals) {
                    const auto want = run_blocks(true, p, s, tail, f);
                    const auto got = run_blocks(false, p, s, tail, f);
                    check(want.has_value() == got.has_value(), "plan_blocks: same refusal", p, cap, f);
                    if (!want || !got) continue;
                    check(*want == *got, "plan_blocks: same plan as the previous body", p, cap, f);
                    check(covers(*got, p, cap), "plan_blocks: every row once, pieces in [1, cap]", p, cap, f);
                }
        }
    }
    check(!run_blocks(false, 100, {}, false, 0), "plan_blocks: empty schedule refused");
    check(!run_blocks(false, 100, {0}, false, 0), "plan_blocks: zero rows refused");
    check(!run_blocks(false, 100, {65521}, false, 0), "plan_blocks: grid limit refused");
    check(!run_blocks(false, 100, {64}, false, 65), "plan_blocks: final rows above capacity refused");
}

void sweep_chunks() {
    const V caps{0, 1, 3, 128, 1000, 3968, 4096, 6016, 8192};
    const V ramps{0, 1, 64, 127, 128, 512, 1000, 1024, 2048, 3968, 4096, 9000};
    const V aligns{0, 1, 3, 128, 256};
    std::vector<std::uint32_t> ns;
    for (std::uint32_t n = 0; n <= 700; ++n) ns.push_back(n);
    for (std::uint32_t n = 701; n <= 270000; n += 1499) ns.push_back(n);
    for (const std::uint32_t n : {3968u, 8192u, 8279u, 32652u, 65536u, 131072u, 262144u}) ns.push_back(n);
    for (const auto C : caps)
        for (const auto n : ns) {
            if (C < 128 && n > 700) continue;  // tiny chunks: bounded plan length
            for (const auto r : ramps)
                for (const auto a : aligns) {
                    const auto want = ref_plan_chunks(n, C, r, a);
                    const auto got = strix::cuda_core::plan_chunks(n, C, r, a);
                    check(same_spans(want, got), "plan_chunks: same spans as the previous body", n, C, r);
                }
        }
}

void shared_traits() {
    using namespace strix::prefill;
    check(greedy_pieces(0, {8}).empty(), "greedy: no rows, no pieces");
    check(greedy_pieces(20, {4, 8}) == V{4, 8, 8}, "greedy: last entry repeats");
    check(doubling_ramp(4096, 1024, 128) == V{1024, 2048, 4096}, "ramp: doubles up to the cap");
    check(doubling_ramp(4096, 0, 128) == V{4096}, "ramp: off");
    check(doubling_ramp(4096, 4096, 128) == V{4096}, "ramp: >= cap is off");
    check(doubling_ramp(4096, 64, 128) == V{128, 256, 512, 1024, 2048, 4096}, "ramp: below align uses align");
    check(doubling_ramp(0xFFFFFFFFu, 0x80000000u, 1) == V{0x80000000u, 0xFFFFFFFFu}, "ramp: no 32-bit wrap near the top");
    V t{8192, 87};
    tail_first(t);
    check(t == V{87, 8192}, "tail_first: remainder first");
    V r{8192, 87};
    reserve_final(r, 8279, 1000, 8192);
    check(r == V{7279, 1000}, "reserve_final: rows moved into the last piece");
    const auto s = piece_spans({3, 5, 2});
    check(s.size() == 3 && s[1].pos == 3 && s[2].pos == 8 && s[2].rows == 2, "spans: positions are prefix sums");
}
}  // namespace

int main() {
    shared_traits();
    sweep_blocks();
    sweep_chunks();
    if (fails) {
        std::fprintf(stderr, "prefill_piece_plan_test: %d of %ld checks failed\n", fails, checks);
        return EXIT_FAILURE;
    }
    std::printf("prefill_piece_plan_test: PASS (%ld checks)\n", checks);
    return EXIT_SUCCESS;
}
