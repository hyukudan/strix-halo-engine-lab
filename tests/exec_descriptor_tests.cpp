#include "strix/device/runtime.hpp"  // StepGraph's include path must coexist with the X3 ABI.
#include "strix/exec/descriptor.hpp"
#include "strix/session/kv_row_append.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {

namespace exec = strix::exec;
namespace session = strix::session;

constexpr std::uint32_t kSteps = 1000;
constexpr exec::StepStopTable kStops{{4, 5, 6, 7}};
constexpr auto kAddress = std::uint64_t{0x1234567887654320ULL};

constexpr bool layout_words() {
    const exec::StepRow row{1, 2, 3, 4, 5, 6, 7, 8};
    exec::StepHeader header{16, 15, 14, 13, 12, 4, 0, 0};
    exec::set_step_stop_address(header, kAddress);
    const auto input = exec::step_input(row);
    return std::bit_cast<std::array<std::uint32_t, 8>>(row) ==
               std::array<std::uint32_t, 8>{1, 2, 3, 4, 5, 6, 7, 8} &&
           std::bit_cast<std::array<std::uint32_t, 8>>(header) ==
               std::array<std::uint32_t, 8>{16, 15, 14, 13, 12, 4, 0x87654320U, 0x12345678U} &&
           std::bit_cast<std::array<std::uint32_t, 2>>(input) == std::array<std::uint32_t, 2>{2, 1} &&
           exec::step_stop_address(header) == kAddress && exec::step_stops_on(header, kStops, 7) &&
           !exec::step_stops_on(header, kStops, 8);
}

static_assert(layout_words());
static_assert(std::string_view{exec::kStepLayoutId} == "strix.exec.step.v1");
static_assert(std::bit_cast<std::array<std::uint32_t, 4>>(exec::RowOut{9, 1.0F, -0.0F, exec::RowStatus::stop}) ==
              std::array<std::uint32_t, 4>{9, 0x3f800000U, 0x80000000U, 2});

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
bool same_bytes(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

constexpr std::uint32_t bit(std::uint32_t i) { return std::uint32_t{1} << i; }

exec::StepBlock make_block(std::uint32_t width) {
    exec::StepBlock block{};
    block.h.rows = width;
    block.h.stop_count = exec::kStepMaxStops;
    block.h.sampling_policy = 17;
    exec::set_step_stop_address(block.h, kAddress);
    return block;
}

// Use Inkling's shared append transaction with one narrow cache visit.
struct CacheTraits {
    using Visit = unsigned;
    static constexpr auto schedule() { return std::array{0U}; }
    static constexpr unsigned index(Visit visit) { return visit; }
    static constexpr bool valid(Visit visit) { return visit == 0; }
    static constexpr session::VisitCacheLayout<1> layout(Visit, unsigned context, unsigned max_chunk) {
        return session::make_visit_cache_layout(context, max_chunk, 1U, 4U, std::array{1U}, 1U);
    }
};

using CachePlan = session::VisitCachePlan<CacheTraits>;
constexpr CachePlan kCache(kSteps, 1U);

struct IncumbentLane {
    std::array<std::uint16_t, kCache.kv_elements_per_plane()> keys{}, values{};
    std::array<float, kCache.convolution_elements()> history{}, history_backup{};
    std::array<std::uint16_t, kCache.kv_checkpoint_elements()> kv_backup{};
    session::KvStore::RowAppendState<CachePlan> state{kCache, {keys, values}, history};
    session::StepDescriptor uploaded{};
    session::StepReplayPolicy policy;
    std::uint32_t token = 0;
    bool stopped = false;

    template <class Work>
    bool try_step_graph(session::KvAppendChunk chunk, Work& work) {
        const auto action = policy.select(true, chunk.rows, true, true);
        check(action != session::StepReplayAction::eager, "incumbent graph refused a decode row");
        if (action == session::StepReplayAction::capture) policy.captured();
        uploaded = {chunk.first_position, token};
        work();
        return true;
    }

    template <class T>
    void copy(std::span<const T> from, std::span<T> to) { std::ranges::copy(from, to.begin()); }

    void finish() {}

    void append() {
        state.append(*this, {state.committed(), 1}, history_backup, kv_backup, [&] {
            const auto cache = state.visit(0).kv;
            const auto cell = cache.wrapped ? uploaded.position % cache.slots : uploaded.position;
            cache.keys[cell] = static_cast<std::uint16_t>(uploaded.token);
        }, [] {});
    }
};

void incumbent_equality() {
    std::mt19937 random(0x7ec8b420U);
    for (std::uint32_t rows = 1; rows <= 8; ++rows) {
        std::array<IncumbentLane, 8> incumbent{};
        std::array<std::uint32_t, 8> stops{};
        auto current = make_block(std::bit_ceil(rows));
        for (std::uint32_t i = 0; i < rows; ++i) {
            incumbent[i].token = i + 100U;
            stops[i] = static_cast<std::uint32_t>(random() % kSteps);
            current.row[i] = {incumbent[i].token, 0, i * 11U + 40U, exec::kStepNoParent, 0, exec::step_valid, i, i + 1U};
            current.h.valid_mask |= bit(i);
        }
        current.h.feed_mask = current.h.valid_mask;
        for (std::uint32_t step = 0; step < kSteps; ++step) {
            std::array<exec::RowOut, 8> picks{};
            for (std::uint32_t i = 0; i < current.h.rows; ++i) {
                const auto id = step == stops[i] ? kStops.ids[i % exec::kStepMaxStops]
                                                : 8U + static_cast<std::uint32_t>(random() % 30000U);
                picks[i] = {id, static_cast<float>(step), -0.0F, exec::RowStatus::picked};
            }
            const auto before = current;
            const auto advanced = exec::advance_step(current, std::span{picks}.first(current.h.rows), kStops);
            check(same_bytes(current, before), "advance mutated the submitted block");
            check(advanced.next.h.step == step + 1 && advanced.next.h.rows == current.h.rows &&
                      advanced.next.h.sampling_policy == current.h.sampling_policy &&
                      exec::step_stop_address(advanced.next.h) == kAddress,
                  "header changed across advance");
            for (std::uint32_t i = 0; i < current.h.rows; ++i) {
                auto& lane = incumbent[i];
                const auto& row = exec::step_row(current, i);
                const auto& next = exec::step_row(advanced.next, i);
                const auto& readback = advanced.readback[i];
                const auto commits = (advanced.commit_mask & bit(i)) != 0;
                const auto expected_active = i < rows && !lane.stopped;
                check(exec::step_row_valid(current, i) == expected_active && commits == expected_active,
                      "incumbent activation and commit mask differ");
                check(next.slot == row.slot && next.slot == (i < rows ? i * 11U + 40U : 0U),
                      "session lane changed across advance");
                if (!expected_active) {
                    check(same_bytes(row, next) && readback.status == exec::RowStatus::masked && readback.id == 0,
                          "padding or stopped lane published state or a token");
                    if (i < rows) check(next.position == lane.state.committed(), "overshoot moved the frontier");
                    continue;
                }
                lane.append();
                check(same_bytes(exec::step_input(row), lane.uploaded), "incumbent uploaded different bytes");
                check(next.position == lane.state.committed(), "incumbent committed a different frontier");
                const auto cell = strix::attn::cache_rows::slot(kCache.visit(0).kv, next.position - 1U);
                check(lane.keys[cell] == row.token, "incumbent wrote to a different cache cell");
                lane.stopped = stops[i] == step;
                auto expected_readback = picks[i];
                if (lane.stopped) expected_readback.status = exec::RowStatus::stop;
                check(same_bytes(readback, expected_readback), "row returned different pick bytes");
                if (!lane.stopped) lane.token = expected_readback.id;
                check(next.token == lane.token && exec::step_row_valid(advanced.next, i) == !lane.stopped,
                      "stop or feedback disagrees with incumbent input");
            }
            current = advanced.next;
        }
        for (std::uint32_t i = 0; i < rows; ++i)
            check(incumbent[i].stopped && incumbent[i].state.committed() == stops[i] + 1U,
                  "incumbent stop frontier differs from the number of consumed inputs");
    }
}

void row_invariance() {
    constexpr std::uint32_t stop_step = 3;
    constexpr std::uint32_t chain_steps = 8;
    auto solo = make_block(1);
    solo.row[0] = {121, 50, 93, exec::kStepNoParent, 0, exec::step_valid | exec::step_sample, 5, 87};
    solo.h.valid_mask = solo.h.feed_mask = 1;
    auto target = solo.row[0];
    for (std::uint32_t step = 0; step < stop_step + chain_steps; ++step) {
        const auto width = bit(step % 4U + 2U);
        const auto index = width - 1U;
        auto cohort = make_block(width);
        cohort.h.step = step;
        cohort.row[index] = target;
        if (exec::step_row_valid(solo, 0)) cohort.h.valid_mask = bit(index);
        for (std::uint32_t i = 0; i < step % 4U; ++i) {
            cohort.row[i] = {11U + i, 77U - i, 80U + i, exec::kStepNoParent, 0, exec::step_valid, 6, i};
            cohort.h.valid_mask |= bit(i);
        }
        cohort.h.feed_mask = cohort.h.valid_mask;
        std::array<exec::RowOut, exec::kStepMaxRows> picks{};
        // Leave every masked input with a non-masked result to model already queued work.
        picks.fill({500, 10.0F, 8.0F, exec::RowStatus::picked});
        picks[index] = {600U + step, -0.0F, 1.0F, step == stop_step ? exec::RowStatus::stop : exec::RowStatus::picked};

        const auto solo_next = exec::advance_step(solo, std::span{picks}.subspan(index, 1), kStops);
        const auto cohort_next = exec::advance_step(cohort, std::span{picks}.first(width), kStops);
        target = exec::step_row(cohort_next.next, index);
        check(same_bytes(target, exec::step_row(solo_next.next, 0)), "companions changed target descriptor bytes");
        check(same_bytes(cohort_next.readback[index], solo_next.readback[0]), "companions changed target readback bytes");
        check(((cohort_next.commit_mask & bit(index)) != 0) == (solo_next.commit_mask != 0),
              "companions changed target frontier publication");
        check(exec::step_row_valid(cohort_next.next, index) == exec::step_row_valid(solo_next.next, 0),
              "companions changed target validity");
        check(cohort_next.readback[width - 2U].status == exec::RowStatus::masked,
              "padded row emitted a token");
        if (step > stop_step)
            check(solo_next.readback[0].status == exec::RowStatus::masked && solo_next.commit_mask == 0,
                  "queued work after the stop published a token or state");
        solo = solo_next.next;
    }
    check(target.position == 54 && target.slot == 93 && target.token == 602 && (target.flags & exec::step_valid) == 0,
          "seven queued steps after stop changed the committed target");
}

void terminal_row() {
    auto single = make_block(1);
    single.row[0] = {82, 51, 43, exec::kStepNoParent, 0, exec::step_valid, 0, 0};
    single.h.valid_mask = 1;
    single.h.stop_count = 0;
    exec::set_step_stop_address(single.h, 0);
    const std::array<exec::RowOut, 1> picks{{{kStops.ids[0], 6.0F, 3.0F, exec::RowStatus::picked}}};
    const auto next = exec::advance_step(single, picks, kStops);
    check(next.next.h.valid_mask == 0 && next.next.h.feed_mask == 0 && next.commit_mask == 1 &&
              next.next.row[0].position == 52 && next.next.row[0].token == 82 &&
              next.readback[0].status == exec::RowStatus::picked,
          "non-feeding row failed to commit or zero-count stop table matched");
}

template <class Error>
void rejects_advance(const exec::StepBlock& block, const exec::RowOut& pick, const char* message) {
    try {
        (void)exec::advance_step(block, std::span{&pick, 1}, kStops);
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

void invalid_advance() {
    auto single = make_block(1);
    single.h.valid_mask = single.h.feed_mask = 1;
    single.row[0] = {82, 51, 43, exec::kStepNoParent, 0, exec::step_valid, 0, 0};
    const exec::RowOut pick{83, 3.0F, 2.0F, exec::RowStatus::picked};
    auto invalid = single;
    invalid.h.rows = 33;
    rejects_advance<std::invalid_argument>(invalid, pick, "unsupported width advanced");
    invalid = single;
    invalid.h.stop_count = exec::kStepMaxStops + 1U;
    rejects_advance<std::invalid_argument>(invalid, pick, "oversized stop table advanced");
    invalid = single;
    invalid.row[0].flags |= exec::step_pending_state;
    rejects_advance<std::invalid_argument>(invalid, pick, "pending state was published as a decode append");
    invalid = single;
    invalid.row[0].position = std::numeric_limits<std::uint32_t>::max();
    rejects_advance<std::overflow_error>(invalid, pick, "frontier advanced past the position limit");
    rejects_advance<std::invalid_argument>(single, {0, 0.0F, 0.0F, exec::RowStatus::masked},
                                           "active row committed with a masked readback");
}

}  // namespace

int main() {
    try {
        incumbent_equality();
        std::cout << "incumbent_equality PASS steps=1000 rows=1..8\n";
        row_invariance();
        terminal_row();
        invalid_advance();
        std::cout << "row_invariance PASS layout=strix.exec.step.v1 readback_bytes_per_row=16\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
