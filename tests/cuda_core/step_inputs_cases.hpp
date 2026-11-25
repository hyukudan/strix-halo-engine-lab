#pragma once

#include "core/step_inputs_plan.hpp"
#include "strix/head_argmax.hpp"

#include <bit>
#include <cstdint>
#include <cstring>

namespace e2i {

namespace scatter = strix::cuda::core::step_inputs;
namespace exec = strix::exec;
inline constexpr auto kRows = scatter::kRows;
inline constexpr auto kHost = strix::spec::DraftFeed::kHost;
inline constexpr auto kNone = strix::head_argmax::kNone;
inline constexpr std::uint32_t kPoison = 0xa7b8c9daU;

template <std::uint32_t N>
struct Guarded {
    std::uint32_t words[N + 2U];
    Guarded() { for (auto& word : words) word = kPoison; }
    std::uint32_t* data() { return words + 1; }
    const std::uint32_t* data() const { return words + 1; }
};

template <class T>
bool equal_bytes(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

enum class Mode { linear, tree, fed_row, fed_rows, slots };
struct ScatterShape { Mode mode; std::uint32_t rows; std::uint32_t first = 65535U; };
inline constexpr ScatterShape kScatterCases[]{
    {Mode::linear, 1}, {Mode::linear, 5}, {Mode::linear, 32},
    {Mode::tree, 1}, {Mode::tree, 5}, {Mode::tree, 32},
    {Mode::fed_row, 1, 1024U}, {Mode::fed_row, 1, 65535U},
    {Mode::fed_rows, 1}, {Mode::fed_rows, 5}, {Mode::fed_rows, 32},
    {Mode::slots, 1}, {Mode::slots, 5}, {Mode::slots, 32},
};

struct ScatterCase {
    std::uint32_t ids[kRows], positions[kRows], slots[kRows];
    std::uint32_t feeds[kRows], indices[kRows];
    scatter::TreeMetadata tree;
    scatter::SlotGroups groups;
};

inline ScatterCase make_scatter(ScatterShape shape) {
    ScatterCase c{};
    for (std::uint32_t i = 0; i < kRows; ++i) {
        c.indices[i] = kHost;
        c.feeds[i] = 160000U - 53U * i;
    }
    for (std::uint32_t i = 0; i < shape.rows; ++i) {
        c.ids[i] = i * 17U;
        c.positions[i] = shape.first + i;
        if (shape.mode == Mode::tree) {
            const auto depth = (i + 1U) / 2U;
            const bool leaf = i % 2U != 0U;
            if (leaf) c.tree.leaf |= 1U << i;
            c.positions[i] = shape.first + depth;
            c.tree.self_kv[i] = shape.first + (leaf ? (shape.rows + 1U) / 2U + i / 2U : depth);
        }
        if (shape.mode == Mode::fed_rows && (shape.rows == kRows || i % 2U != 0U)) c.indices[i] = kRows - 1U - i;
    }
    if (shape.mode == Mode::fed_row) c.indices[0] = shape.first == 1024U ? 0U : kRows - 1U;
    if (shape.mode == Mode::slots) {
        for (std::uint32_t base = 0, group = 0; base < shape.rows; ++group) {
            const auto count = shape.rows == 5U && group != 1U ? 2U : 1U;
            c.groups.row0[group] = base;
            c.groups.rows[group] = count;
            c.groups.slot[group] = 100U - 2U * group;
            for (std::uint32_t j = 0; j < count; ++j, ++base) {
                c.slots[base] = c.groups.slot[group];
                c.positions[base] = shape.first + c.slots[base] + j;
            }
        }
    }
    return c;
}

inline exec::StepBlock descriptor(ScatterShape shape, const ScatterCase& c) {
    exec::StepBlock block{};
    block.h.rows = std::bit_ceil(shape.rows);
    for (std::uint32_t i = 0; i < kRows; ++i) {
        auto& row = block.row[i];
        if (i >= shape.rows) {
            row.token = row.position = row.slot = kPoison;
            continue;
        }
        block.h.valid_mask |= 1U << i;
        row.flags = exec::step_valid;
        row.token = c.ids[i];
        row.position = c.positions[i];
        row.slot = c.slots[i];
        const auto depth = c.positions[i] - shape.first;
        row.parent = exec::kStepNoParent;
        row.ancestors = 1U << i;
        if (shape.mode == Mode::tree) {
            if (i != 0U) row.parent = 2U * (depth - 1U);
            for (std::uint32_t j = 0; j < i; j += 2U) {
                if (c.positions[j] < c.positions[i]) row.ancestors |= 1U << j;
            }
        }
    }
    return block;
}

struct ScatterOutput {
    Guarded<1> anchor;
    Guarded<kRows> ids, positions, slots, group_row0, group_rows, group_slot;
    Guarded<scatter::kTreeWords> tree;
};
static_assert(sizeof(ScatterOutput) == (3U + 6U * (kRows + 2U) + scatter::kTreeWords + 2U) * sizeof(std::uint32_t));

template <class Launch>
void with_scatter_view(Mode mode, const ScatterCase* c, ScatterOutput* out, Launch launch) {
    const scatter::LinearOutput linear{out->anchor.words + 1, out->ids.words + 1};
    switch (mode) {
        case Mode::linear: launch(scatter::Linear{linear}); break;
        case Mode::tree: launch(scatter::Tree{linear, &c->tree, out->tree.words + 1}); break;
        case Mode::fed_row: launch(scatter::FedRow{linear, {c->feeds, c->indices}}); break;
        case Mode::fed_rows: launch(scatter::FedRows{linear, {c->feeds, c->indices}}); break;
        case Mode::slots:
            launch(scatter::Slots{&c->groups, out->ids.words + 1, out->positions.words + 1, out->slots.words + 1,
                out->group_row0.words + 1, out->group_rows.words + 1, out->group_slot.words + 1});
            break;
    }
}

// Independent oracle over the host input arrays, including every legacy scratch word and guard.
inline ScatterOutput scatter_oracle(Mode mode, const ScatterCase& c) {
    ScatterOutput out;
    if (mode != Mode::slots) out.anchor.data()[0] = c.positions[0];
    if (mode == Mode::fed_row) {
        out.ids.data()[0] = c.feeds[c.indices[0]];
        return out;
    }
    if (mode == Mode::tree) out.tree.data()[0] = c.tree.leaf;
    for (std::uint32_t i = 0; i < kRows; ++i) {
        out.ids.data()[i] = mode == Mode::fed_rows && c.indices[i] != kHost ? c.feeds[c.indices[i]] : c.ids[i];
        if (mode == Mode::tree) {
            out.tree.data()[1U + i] = c.positions[i];
            out.tree.data()[1U + kRows + i] = c.tree.self_kv[i];
        }
        if (mode == Mode::slots) {
            out.positions.data()[i] = c.positions[i];
            out.slots.data()[i] = c.slots[i];
            out.group_row0.data()[i] = c.groups.row0[i];
            out.group_rows.data()[i] = c.groups.rows[i];
            out.group_slot.data()[i] = c.groups.slot[i];
        }
    }
    return out;
}

enum DraftOption : std::uint32_t {
    mapped = 1U, two = 2U, feed_top = 4U, second = 8U, feed_second = 16U,
    gap_input = 32U, gap_output = 64U, sentinel_last = 128U,
};
inline constexpr auto kTwoFeed = two | feed_top | second | feed_second | gap_input | gap_output;

struct DraftCase {
    std::uint32_t rows, group, options, last_gap;
    bool has(DraftOption option) const { return (options & option) != 0U; }
};

inline constexpr DraftCase kDraftCases[]{
    {0, 16, mapped | kTwoFeed, 0},
    {1, 16, mapped, 0},
    {5, 16, mapped | feed_top, 0},
    {32, 16, feed_top, 0},
    {32, 16, mapped | two | second, 0},
    {1, 16, mapped | kTwoFeed | sentinel_last, 0x7fc01234U},
    {5, 3, mapped | kTwoFeed, 0x80000000U},
    {32, 16, kTwoFeed, 0x7f800000U},
    {5, 16, two | feed_second | gap_input | gap_output, 0x3e800000U},
    {5, 16, mapped | two | second | feed_second | gap_output, 0},
    {5, 16, two | gap_input, 0},
    {5, 16, feed_top | second | feed_second | gap_input | gap_output, 0x80000000U},
};

struct DraftInput {
    std::uint32_t map[8] = {13, 1, 42, 7, 0, 18, 4, 30};
    float gap[kRows]{};
};

struct DraftOutput {
    Guarded<kRows> top, top2;
    Guarded<1> feed, feed2, gap;
};
static_assert(sizeof(DraftOutput) == (2U * (kRows + 2U) + 9U) * sizeof(std::uint32_t));

inline DraftInput make_draft_input(DraftCase c) {
    DraftInput in;
    if (c.rows) in.gap[c.rows - 1U] = std::bit_cast<float>(c.last_gap);
    return in;
}

inline DraftOutput make_draft_output(DraftCase c) {
    DraftOutput out;
    for (std::uint32_t i = 0; i < kRows; ++i) {
        out.top.data()[i] = (i * 7U + c.group - 1U) % (8U * c.group);
        out.top2.data()[i] = i % 6U == 0U ? kNone : (i * 11U + c.group) % (8U * c.group);
    }
    if (c.has(sentinel_last)) out.top2.data()[c.rows - 1U] = kNone;
    return out;
}

inline DraftOutput draft_oracle(DraftCase c, const DraftInput& in, DraftOutput out) {
    if (!c.rows) return out;
    const auto translate = [&](std::uint32_t compact) {
        const auto bucket = compact / c.group;
        const auto offset = compact - bucket * c.group;
        return in.map[bucket] * c.group + offset;
    };
    if (c.has(mapped)) {
        for (std::uint32_t i = 0; i < c.rows; ++i) {
            out.top.data()[i] = translate(out.top.data()[i]);
            if (c.has(two) && c.has(second) && out.top2.data()[i] != kNone)
                out.top2.data()[i] = translate(out.top2.data()[i]);
        }
    }
    const auto last = c.rows - 1U;
    if (c.has(feed_top)) out.feed.data()[0] = out.top.data()[last];
    if (c.has(two) && c.has(second) && c.has(feed_second)) out.feed2.data()[0] = out.top2.data()[last];
    if (c.has(two) && c.has(gap_input) && c.has(gap_output)) out.gap.data()[0] = std::bit_cast<std::uint32_t>(in.gap[last]);
    return out;
}

}  // namespace e2i
