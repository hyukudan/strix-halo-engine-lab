#include "strix/session/ring_plan.hpp"

#include "core/short_convolution.hpp"
#include "strix/session/kv_row_append.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace session = strix::session;
namespace cache_rows = strix::attn::cache_rows;

constexpr unsigned kWidth = 2;
constexpr unsigned kMaxWindow = 8;
using KvRow = std::array<std::uint16_t, kWidth>;
using WindowRow = std::array<KvRow, 2>;
using ConvRow = std::array<float, kWidth>;
using RingReference = session::PositionRingReference<WindowRow, ConvRow>;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <class Exception, class Action>
void expect_refusal(const char* message, Action action) {
    try {
        action();
    } catch (const Exception& error) {
        check(std::string_view(error.what()) == message, error.what());
        return;
    }
    throw std::runtime_error(message);
}

struct Block {
    std::array<WindowRow, kMaxWindow> window{};
    std::array<ConvRow, kMaxWindow> conv{};
};

Block make_block(std::mt19937& random) {
    Block block;
    for (auto& row : block.window)
        for (auto& plane : row)
            for (auto& value : plane) value = static_cast<std::uint16_t>(random());
    for (auto& row : block.conv) {
        const auto bits = (static_cast<std::uint32_t>(random()) & 0x807fffffU) | 0x3f000000U;
        row = {std::bit_cast<float>(bits), -0.0F};
    }
    return block;
}

struct HostCopies {
    unsigned kv_copies = 0;

    void copy(std::span<const float> from, std::span<float> to) { std::ranges::copy(from, to.begin()); }
    void copy(std::span<const std::uint16_t> from, std::span<std::uint16_t> to) {
        ++kv_copies;
        std::ranges::copy(from, to.begin());
    }
    void finish() {}
};

template <unsigned Window, unsigned Taps>
struct CacheTraits {
    using Visit = unsigned;
    static constexpr auto schedule() { return std::array{0U}; }
    static constexpr unsigned index(Visit visit) { return visit; }
    static constexpr bool valid(Visit visit) { return visit == 0; }
    static constexpr auto layout(Visit, unsigned context, unsigned max_chunk) {
        return session::make_visit_cache_layout(context, max_chunk, kWidth, Window, std::array{kWidth}, Taps - 1);
    }
};

// Exercise the incumbent append and replay path with actual host convolution
// history updates. Its history is oldest first and holds Taps - 1 past inputs.
template <unsigned Window, unsigned Taps>
struct BackupReplay {
    using Plan = session::VisitCachePlan<CacheTraits<Window, Taps>>;

    explicit BackupReplay(unsigned max_chunk)
        : plan(16384U, max_chunk), keys(plan.kv_elements_per_plane()), values(keys.size()),
          kv_checkpoint(plan.kv_checkpoint_elements()), state(plan, {keys, values}, history) {}

    void append(const Block& block, unsigned rows, bool stage) {
        const auto first = state.committed();
        state.append(backend, {first, rows}, history_checkpoint, kv_checkpoint, [&] {
            const auto visit = state.visit(0);
            const cache_rows::Rows layout{kWidth, visit.kv.slots};
            std::array<float, kMaxWindow * kWidth> input{}, output{};
            for (unsigned row = 0; row < rows; ++row) {
                const auto offset = cache_rows::row_offset(layout, first + row);
                std::ranges::copy(block.window[row][0], visit.kv.keys + offset);
                std::ranges::copy(block.window[row][1], visit.kv.values + offset);
                std::ranges::copy(block.conv[row], input.data() + row * kWidth);
            }
            constexpr std::array<float, Taps * kWidth> weights{};
            strix::core::causal_short_convolution({kWidth, Taps}, std::span(input).first(rows * kWidth),
                weights, {}, visit.history[0], std::span(output).first(rows * kWidth));
        }, [] {}, nullptr, stage);
    }

    void commit(const Block& block, unsigned keep) {
        state.commit_staged(backend, keep, [&](unsigned prefix) { append(block, prefix, false); });
    }

    void compare(const RingReference& ring) const {
        const auto end = state.committed();
        check(ring.committed() == end, "committed frontier differs");
        const auto visit = state.visit(0);
        const cache_rows::Rows layout{kWidth, visit.kv.slots};
        for (auto pos = end - std::min(end, Window); pos < end; ++pos) {
            const auto offset = cache_rows::row_offset(layout, pos);
            const auto& live = ring.window_at(pos);
            check(std::memcmp(live[0].data(), visit.kv.keys + offset, sizeof(KvRow)) == 0, "window keys differ");
            check(std::memcmp(live[1].data(), visit.kv.values + offset, sizeof(KvRow)) == 0, "window values differ");
        }
        constexpr ConvRow zero{};
        const auto visible = std::min(end, Taps - 1);
        const auto padding = (Taps - 1) - visible;
        for (unsigned i = 0; i < Taps - 1; ++i) {
            const auto& live = i < padding ? zero : ring.convolution_at(end - visible + i - padding);
            check(std::memcmp(live.data(), visit.history[0].data() + i * kWidth, sizeof(ConvRow)) == 0,
                  "convolution history differs");
        }
    }

    const Plan plan;
    std::vector<std::uint16_t> keys, values, kv_checkpoint;
    std::array<float, (Taps - 1) * kWidth> history{}, history_checkpoint{};
    session::KvStore::RowAppendState<Plan> state;
    HostCopies backend;
};

template <unsigned Window, unsigned Taps>
void random_acceptance() {
    const session::RingPlan plan(Window, Taps, kMaxWindow);
    std::vector<WindowRow> window(plan.window_cells());
    std::vector<ConvRow> conv(plan.convolution_cells());
    RingReference ring(plan, window, conv);
    BackupReplay<Window, Taps> incumbent(kMaxWindow);
    std::mt19937 random(0x7293afU + Window + Taps);

    const auto step = [&](unsigned rows, unsigned keep) {
        const auto block = make_block(random);
        ring.stage(std::span(block.window).first(rows), std::span(block.conv).first(rows));
        check(ring.staged_rows() == rows, "staged row count differs");
        incumbent.compare(ring);
        incumbent.append(block, rows, true);
        incumbent.commit(block, keep);
        ring.commit_staged(keep);
        check(ring.staged_rows() == 0, "commit left rows staged");
        incumbent.compare(ring);
    };

    incumbent.compare(ring);
    for (unsigned rows = 1; rows <= kMaxWindow; ++rows)
        for (unsigned keep = 0; keep <= rows; ++keep) step(rows, keep);
    for (unsigned i = 0; i < 2000; ++i) {
        const auto rows = 1U + static_cast<unsigned>(random() % kMaxWindow);
        const auto keep = static_cast<unsigned>(random() % (rows + 1));
        step(rows, keep);
    }
    check(ring.committed() > 2 * plan.window_cells(), "random acceptance missed repeated wraparound");
}

void wraparound() {
    constexpr session::RingPlan plan(4, 4, 5);
    static_assert(plan.window_cells() == 9 && plan.convolution_cells() == 9);
    static_assert(plan.window_cell(9) == 0 && plan.convolution_cell(10) == 1);
    std::array<WindowRow, plan.window_cells()> window{};
    std::array<ConvRow, plan.convolution_cells()> conv{};
    RingReference ring(plan, window, conv);
    BackupReplay<4, 4> incumbent(5);
    std::mt19937 random(0xffbd17U);

    for (const unsigned rows : {5U, 3U}) {
        const auto block = make_block(random);
        ring.stage(std::span(block.window).first(rows), std::span(block.conv).first(rows));
        ring.commit_staged(rows);
        incumbent.append(block, rows, false);
    }
    incumbent.compare(ring);
    for (const unsigned keep : {0U, 1U}) {
        const auto block = make_block(random);
        ring.stage(std::span(block.window).first(5), std::span(block.conv).first(5));
        check(window[8] == block.window[0] && window[0] == block.window[1], "window stores missed wraparound");
        check(std::memcmp(conv[8].data(), block.conv[0].data(), sizeof(ConvRow)) == 0 &&
              std::memcmp(conv[0].data(), block.conv[1].data(), sizeof(ConvRow)) == 0, "convolution stores missed wraparound");
        incumbent.compare(ring);
        const auto before_window = window;
        const auto before_conv = conv;
        incumbent.backend.kv_copies = 0;
        incumbent.append(block, 5, true);
        check(incumbent.backend.kv_copies == 2, "wraparound failed to exercise backup of both KV planes");
        incumbent.commit(block, keep);
        check(incumbent.backend.kv_copies == 4, "wraparound failed to exercise restore of both KV planes");
        if (keep == 0) ring.rollback_staged();
        else ring.commit_staged(keep);
        check(std::memcmp(window.data(), before_window.data(), window.size() * sizeof(WindowRow)) == 0 &&
              std::memcmp(conv.data(), before_conv.data(), conv.size() * sizeof(ConvRow)) == 0,
              "rollback or partial commit wrote ring cells");
        incumbent.compare(ring);
    }
    check(ring.committed() == 9, "partial commit after rollback used the wrong frontier");
    expect_refusal<std::out_of_range>("RingPlan: window position outside committed range",
                                      [&] { (void)ring.window_at(4); });
    expect_refusal<std::out_of_range>("RingPlan: window position outside committed range",
                                      [&] { (void)ring.window_at(9); });
    expect_refusal<std::out_of_range>("RingPlan: convolution position outside committed range",
                                      [&] { (void)ring.convolution_at(4); });
    expect_refusal<std::out_of_range>("RingPlan: convolution position outside committed range",
                                      [&] { (void)ring.convolution_at(9); });
    check(&ring.convolution_at(5) == &conv[5], "convolution ring lost the fourth retained row");
}

void refusals() {
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    constexpr session::RingPlan plan(4, 3, 2);
    static_assert(plan.staged_end(max - 2, 2) == max);
    static_assert(session::parse_rollback_mode("position_rings") == session::RollbackMode::position_rings);
    expect_refusal<std::invalid_argument>("rollback must be \"position_rings\"",
        [] { (void)session::parse_rollback_mode("Position_Rings"); });
    expect_refusal<std::invalid_argument>("RingPlan: zero window", [] { (void)session::RingPlan(0, 3, 2); });
    expect_refusal<std::invalid_argument>("RingPlan: max_window is zero", [] { (void)session::RingPlan(4, 3, 0); });
    expect_refusal<std::overflow_error>("RingPlan: window cell count overflow", [] { (void)session::RingPlan(max, 3, 2); });
    expect_refusal<std::overflow_error>("RingPlan: convolution cell count overflow", [] { (void)session::RingPlan(4, max, 2); });
    expect_refusal<std::overflow_error>("RingPlan: staged frontier overflow", [&] { (void)plan.staged_end(max - 1, 2); });

    std::array<WindowRow, plan.window_cells()> window{};
    std::array<ConvRow, plan.convolution_cells()> conv{};
    expect_refusal<std::invalid_argument>("RingPlan: storage size does not match ring cells",
        [&] { RingReference wrong_size(plan, std::span(window).first(5), conv); });
    RingReference ring(plan, window, conv);
    const Block block;
    expect_refusal<std::logic_error>("RingPlan: no staged append", [&] { ring.commit_staged(0); });
    expect_refusal<std::invalid_argument>("RingPlan: rows must be in [1, max_window]", [&] { ring.stage({}, {}); });
    expect_refusal<std::invalid_argument>("RingPlan: rows must be in [1, max_window]",
        [&] { ring.stage(block.window, block.conv); });
    expect_refusal<std::invalid_argument>("RingPlan: staged row counts differ",
        [&] { ring.stage(std::span(block.window).first(1), std::span(block.conv).first(2)); });
    check(ring.committed() == 0 && ring.staged_rows() == 0, "invalid stage changed the frontiers");
    ring.stage(std::span(block.window).first(2), std::span(block.conv).first(2));
    expect_refusal<std::logic_error>("RingPlan: append already staged", [&] { ring.stage({}, {}); });
    expect_refusal<std::invalid_argument>("RingPlan: keep exceeds staged rows", [&] { ring.commit_staged(3); });
    check(ring.committed() == 0 && ring.staged_rows() == 2, "invalid commit changed the frontiers");
    ring.commit_staged(2);
    check(ring.committed() == 2 && ring.staged_rows() == 0, "valid retry did not commit staged rows");
}

}  // namespace

int main() try {
    random_acceptance<4, 3>();
    random_acceptance<4, 4>();
    random_acceptance<128, 3>();
    random_acceptance<128, 4>();
    random_acceptance<1024, 3>();
    random_acceptance<1024, 4>();
    wraparound();
    refusals();
    std::puts("PASS session RingPlan position rings == backup+replay");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL session RingPlan: %s\n", error.what());
    return 1;
}
