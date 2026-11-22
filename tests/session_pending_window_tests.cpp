// Pending verify windows on the shared resident append state (include/strix/session/kv_row_append.hpp):
// a window whose work writes only caller-owned pending storage (K/V rows and one convolution history
// image per row) commits a kept prefix by publication. Invariant: after any sequence of windows and
// kept prefixes, the resident KV planes, histories and frontier are byte-equal to a serial append of
// the kept tokens, one row at a time (ring and dense visits, wrapped ring segments, keep 0..R).
// A window of R rows runs on the smallest replay width >= R (padded_step_replay_width).
#include "strix/session/kv_row_append.hpp"
#include "strix/session/step_replay.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace session = strix::session;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

constexpr unsigned kContext = 96U;
constexpr unsigned kMaxChunk = 8U;
constexpr unsigned kHistoryRows = 3U;

// Visit 0: windowed (ring KV), visit 1: dense KV; one convolution stream of width 2 each.
struct Traits {
    using Visit = unsigned;
    static constexpr auto schedule() { return std::array{0U, 1U}; }
    static constexpr unsigned index(Visit visit) { return visit; }
    static constexpr bool valid(Visit visit) { return visit < 2U; }
    static constexpr session::VisitCacheLayout<1> layout(Visit visit, unsigned context, unsigned max_chunk) {
        return session::make_visit_cache_layout(context, max_chunk, 2U, visit == 0U ? 4U : 0U, std::array{2U},
                                                kHistoryRows);
    }
};

using Plan = session::VisitCachePlan<Traits>;
constexpr Plan kPlan(kContext, kMaxChunk);
using State = session::KvStore::RowAppendState<Plan>;

std::uint16_t key_of(unsigned token, unsigned visit, unsigned lane) {
    return static_cast<std::uint16_t>(token * 7U + visit * 3U + lane + 1U);
}
std::uint16_t value_of(unsigned token, unsigned visit, unsigned lane) {
    return static_cast<std::uint16_t>(token * 5U + visit * 11U + lane + 2U);
}
float input_of(unsigned token, unsigned visit, unsigned lane) {
    return static_cast<float>(token) + 0.25f * static_cast<float>(visit) + 0.125f * static_cast<float>(lane);
}

std::size_t cell(const session::VisitKvRows& kv, unsigned position) {
    return kv.wrapped ? position % kv.slots : position;
}

// History of one stream after appending one input row: drop the oldest row, append the new one.
void advance(std::span<float> history, unsigned token, unsigned visit) {
    const std::size_t width = history.size() / kHistoryRows;
    std::copy(history.begin() + static_cast<std::ptrdiff_t>(width), history.end(), history.begin());
    for (unsigned lane = 0; lane < width; ++lane)
        history[(kHistoryRows - 1U) * width + lane] = input_of(token, visit, lane);
}

struct Lane {
    std::vector<std::uint16_t> keys = std::vector<std::uint16_t>(kPlan.kv_elements_per_plane());
    std::vector<std::uint16_t> values = std::vector<std::uint16_t>(kPlan.kv_elements_per_plane());
    std::vector<float> history = std::vector<float>(kPlan.convolution_elements());
    std::vector<float> history_backup = std::vector<float>(kPlan.convolution_elements());
    std::vector<std::uint16_t> kv_backup = std::vector<std::uint16_t>(kPlan.kv_checkpoint_elements());
    State state{kPlan, {keys, values}, history};
    unsigned finishes = 0;

    template <class T>
    void copy(std::span<const T> from, std::span<T> to) { std::ranges::copy(from, to.begin()); }
    void finish() { ++finishes; }

    // Reference: one serial row.
    void append_serial(unsigned token) {
        const auto position = state.committed();
        state.append(*this, {position, 1U}, history_backup, kv_backup, [&] {
            for (const auto visit : Plan::visits()) {
                const auto rows = state.visit(visit);
                const auto width = kPlan.visit(visit).kv.width;
                const auto at = cell(rows.kv, position) * width;
                for (unsigned lane = 0; lane < width; ++lane) {
                    rows.kv.keys[at + lane] = key_of(token, visit, lane);
                    rows.kv.values[at + lane] = value_of(token, visit, lane);
                }
                advance(rows.history[0], token, visit);
            }
        }, [] {});
    }
};

// Caller-owned pending storage of one window: K/V rows [visit][row][width] and one history image per row.
struct Pending {
    std::vector<unsigned> tokens;
    std::array<std::vector<std::uint16_t>, 2> keys, values;
    std::array<std::vector<float>, 2> images;  // [row][history elements]
};

Pending stage_window(Lane& lane, std::span<const unsigned> tokens) {
    Pending pending{{tokens.begin(), tokens.end()}, {}, {}, {}};
    lane.state.append_pending(lane, {lane.state.committed(), static_cast<unsigned>(tokens.size())}, [&] {
        for (const auto visit : Plan::visits()) {
            const auto width = kPlan.visit(visit).kv.width;
            const auto resident = lane.state.visit(visit).history[0];
            std::vector<float> image(resident.begin(), resident.end());
            for (const auto token : tokens) {
                for (unsigned l = 0; l < width; ++l) {
                    pending.keys[visit].push_back(key_of(token, visit, l));
                    pending.values[visit].push_back(value_of(token, visit, l));
                }
                advance(image, token, visit);
                pending.images[visit].insert(pending.images[visit].end(), image.begin(), image.end());
            }
        }
    }, [] {});
    return pending;
}

// The publication an engine performs: kept KV rows in at most two ring segments, history image keep - 1.
void publish(Lane& lane, const Pending& pending, unsigned keep) {
    const auto first = lane.state.committed();
    for (const auto visit : Plan::visits()) {
        const auto rows = lane.state.visit(visit);
        const auto width = kPlan.visit(visit).kv.width;
        const auto start = static_cast<unsigned>(cell(rows.kv, first));
        const auto contiguous = rows.kv.wrapped ? std::min(keep, rows.kv.slots - start) : keep;
        std::copy_n(pending.keys[visit].data(), contiguous * width, rows.kv.keys + std::size_t{start} * width);
        std::copy_n(pending.values[visit].data(), contiguous * width, rows.kv.values + std::size_t{start} * width);
        std::copy_n(pending.keys[visit].data() + contiguous * width, (keep - contiguous) * width, rows.kv.keys);
        std::copy_n(pending.values[visit].data() + contiguous * width, (keep - contiguous) * width, rows.kv.values);
        const auto image = rows.history[0];
        std::copy_n(pending.images[visit].data() + std::size_t{keep - 1U} * image.size(), image.size(), image.data());
    }
}

void equality() {
    std::mt19937 random(0x5eedf00dU);
    unsigned windows = 0, partial = 0, wrapped = 0;
    for (unsigned trial = 0; trial < 40; ++trial) {
        Lane serial, spec;
        unsigned next_token = 3U + trial;
        while (true) {
            const unsigned rows = 1U + static_cast<unsigned>(random() % kMaxChunk);
            if (spec.state.committed() + rows > kContext) break;
            std::vector<unsigned> tokens(rows);
            for (auto& token : tokens) token = next_token++;
            const unsigned keep = static_cast<unsigned>(random() % (rows + 1U));
            const auto first = spec.state.committed();
            const auto ring = spec.state.visit(0).kv;
            const auto pending = stage_window(spec, tokens);
            check(spec.state.staged() && spec.state.committed() == first, "pending window keeps the frontier");
            spec.state.commit_pending(keep, [&](unsigned kept) { publish(spec, pending, kept); });
            for (unsigned r = 0; r < keep; ++r) serial.append_serial(tokens[r]);
            ++windows;
            partial += keep < rows ? 1U : 0U;
            wrapped += ring.wrapped && keep != 0U && cell(ring, first) + keep > ring.slots ? 1U : 0U;
            check(!spec.state.staged() && spec.state.committed() == first + keep, "commit moves the frontier by keep");
            check(spec.keys == serial.keys && spec.values == serial.values, "kept KV rows equal serial append");
            check(spec.history == serial.history, "history equals serial append");
            if (failures) return;
        }
    }
    check(partial > 50U && wrapped > 5U, "coverage: partial keeps and wrapped ring segments");
    std::printf("pending window equality: %u windows, %u partial, %u wrapped ring publications\n", windows, partial,
                wrapped);
}

void contract() {
    Lane lane;
    const std::array<unsigned, 4> tokens{10U, 11U, 12U, 13U};
    const auto keys = lane.keys;
    const auto history = lane.history;
    // keep = 0 discards: nothing written, frontier unchanged, a new append is admitted.
    stage_window(lane, tokens);
    bool called = false;
    lane.state.commit_pending(0U, [&](unsigned) { called = true; });
    check(!called && lane.state.committed() == 0U && lane.keys == keys && lane.history == history,
          "keep 0 discards without publication");
    // A staged pending window refuses appends, the replay commit and an oversize keep.
    stage_window(lane, tokens);
    bool refused = false;
    try { lane.append_serial(1U); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "append refused while a window is pending");
    refused = false;
    try { lane.state.commit_staged(lane, 4U, [](unsigned) {}); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "commit_staged refuses a pending window");
    refused = false;
    try { lane.state.commit_pending(5U, [](unsigned) {}); } catch (const std::invalid_argument&) { refused = true; }
    check(refused && lane.state.staged(), "oversize keep refused, window kept");
    lane.state.commit_pending(4U, [](unsigned) {});
    check(lane.state.committed() == 4U && !lane.state.poisoned(), "full keep publishes");
    // A failed work leaves the frontier and does not poison; the backend is drained.
    const auto drained = lane.finishes;
    refused = false;
    try {
        lane.state.append_pending(lane, {4U, 2U}, [] { throw std::runtime_error("work"); }, [] {});
    } catch (const std::runtime_error&) { refused = true; }
    check(refused && lane.finishes > drained && !lane.state.poisoned() && !lane.state.staged() &&
          lane.state.committed() == 4U, "failed work drains, keeps the frontier, no poison");
    // A failed publication poisons (part of the prefix may be written).
    stage_window(lane, std::span(tokens).first(2));
    refused = false;
    try { lane.state.commit_pending(1U, [](unsigned) { throw std::runtime_error("publish"); }); }
    catch (const std::runtime_error&) { refused = true; }
    check(refused && lane.state.poisoned() && lane.state.committed() == 4U, "failed publication poisons");
    // A staged (checkpointed) append is not a pending window.
    Lane staged;
    staged.state.append(staged, {0U, 2U}, staged.history_backup, staged.kv_backup, [] {}, [] {}, nullptr, true);
    refused = false;
    try { staged.state.commit_pending(1U, [](unsigned) {}); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "commit_pending refuses a checkpointed staged append");
}

void padded_width() {
    check(session::padded_step_replay_width(0U) == 0U, "width of no rows");
    check(session::padded_step_replay_width(session::step_replay_max_rows + 1U) == 0U, "width above the maximum");
    check(session::padded_step_replay_width(3U) == 4U, "3-row window uses the 4-row replay graph");
    check(session::padded_step_replay_width(17U) == 32U, "17-row window uses the 32-row replay graph");
    for (std::uint32_t rows = 1; rows <= session::step_replay_max_rows; ++rows) {
        const auto width = session::padded_step_replay_width(rows);
        check(width != 0U && (width & (width - 1U)) == 0U && width <= session::step_replay_max_rows &&
                  width >= rows && width < 2U * rows,
              "smallest power-of-two replay width covering the window");
    }
}

}  // namespace

int main() {
    equality();
    contract();
    padded_width();
    if (failures) {
        std::fprintf(stderr, "session pending window tests: %d failures\n", failures);
        return 1;
    }
    std::printf("session pending window tests: PASS\n");
    return 0;
}
