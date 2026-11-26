#include "step_inputs_cases.hpp"
#include "strix/session/linear_verify_step.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
using namespace e2i;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

ScatterOutput project(Mode mode, const ScatterCase& c, const exec::StepBlock& block) {
    ScatterOutput out;
    with_scatter_view(mode, &c, &out, [&](auto view) {
        for (std::uint32_t i = 0; i < kRows; ++i) scatter::detail::scatter_row(block, view, i);
    });
    return out;
}

void check_masks() {
    constexpr ScatterShape shape{Mode::fed_rows, 5};
    auto c = make_scatter(shape);
    auto block = descriptor(shape, c);
    // One row is missing its flag, one is missing its mask bit, and one is beyond the width.
    block.row[1].flags = 0;
    block.h.valid_mask &= ~(1U << 2U);
    block.h.valid_mask |= 1U << 8U;
    block.row[8].flags = exec::step_valid;
    c.indices[1] = c.indices[2] = c.indices[8] = kPoison;
    block.h.feed_mask = block.h.valid_mask;
    const auto out = project(shape.mode, c, block);
    require(out.ids.data()[1] == 0U && out.ids.data()[2] == 0U && out.ids.data()[8] == 0U,
            "invalid descriptor row read a token or feed index");
    require(out.ids.data()[3] == c.feeds[c.indices[3]] && out.ids.data()[4] == c.ids[4],
            "run-ahead mask changed the feed selector");

    constexpr ScatterShape single{Mode::fed_row, 1};
    auto masked = descriptor(single, c);
    masked.h.valid_mask = 0;
    c.indices[0] = kPoison;
    require(equal_bytes(project(single.mode, c, masked), ScatterOutput{}), "masked single feed wrote scratch");
}

// A verify window program block (a captured K-row verify window): every row of a K-row window is valid for every
// step_row_valid consumer (id scatter, K/V store, flash, picks) only when the header mask is set. The shared builder
// sets it; a block with rows and per-row flags but no header mask scatters id 0 for every row.
void check_linear_verify_window() {
    for (std::uint32_t width = 2; width <= 8U; ++width) {
        std::array<std::uint32_t, 8> ids{};
        for (std::uint32_t i = 0; i < width; ++i) ids[i] = 1000U + 37U * i;
        const auto block = strix::session::linear_verify_step(
            {width, 658U + width, 3U, 0U, strix::session::LinearVerifyRows::inclusive_valid},
            std::span<const std::uint32_t>(ids.data(), width));
        auto unmasked = block;
        unmasked.h.valid_mask = 0U;
        std::array<std::uint32_t, kRows> got{}, got_unmasked{};
        std::uint32_t anchor = 0, anchor_unmasked = 0;
        for (std::uint32_t i = 0; i < kRows; ++i) {
            scatter::detail::scatter_row(block, scatter::Linear{{&anchor, got.data()}}, i);
            scatter::detail::scatter_row(unmasked, scatter::Linear{{&anchor_unmasked, got_unmasked.data()}}, i);
        }
        for (std::uint32_t i = 0; i < width; ++i) {
            require(exec::step_row_valid(block, i), "verify window row is not valid");
            require(block.row[i].position == 658U + width + i && block.row[i].slot == 3U,
                    "verify window row position or slot");
            require(got[i] == ids[i], "verify window row did not scatter its id");
            require(got_unmasked[i] == 0U, "a block without the header mask must read id 0 (the unmasked block class)");
        }
        for (std::uint32_t i = width; i < kRows; ++i)
            require(!exec::step_row_valid(block, i) && got[i] == 0U, "row past the window");
    }
}

void check_draft_oracle() {
    for (const auto c : kDraftCases) {
        const auto in = make_draft_input(c);
        const auto seed = make_draft_output(c);
        const auto out = draft_oracle(c, in, seed);
        if (c.rows == 0U) {
            require(equal_bytes(out, seed), "empty draft modified output");
            continue;
        }
        const auto last = c.rows - 1U;
        require(out.top.data()[0] == (c.has(mapped) ? 14U * c.group - 1U : c.group - 1U), "first compact group");
        require(out.top2.data()[0] == kNone, "runner-up sentinel rewritten");
        require(out.feed.data()[0] == (c.has(feed_top) ? out.top.data()[last] : kPoison), "last top feed");
        if (c.has(sentinel_last)) require(out.feed2.data()[0] == kNone, "last sentinel feed");
        if (!c.has(two) || !c.has(second) || !c.has(feed_second))
            require(out.feed2.data()[0] == kPoison, "disabled runner-up feed");
        if (c.has(two) && c.has(gap_input) && c.has(gap_output))
            require(out.gap.data()[0] == c.last_gap, "last gap payload");
        else require(out.gap.data()[0] == kPoison, "disabled gap feed");
    }
}

}  // namespace

int main() {
    try {
        for (const auto shape : kScatterCases) {
            const auto c = make_scatter(shape);
            require(equal_bytes(project(shape.mode, c, descriptor(shape, c)), scatter_oracle(shape.mode, c)),
                    "step input scratch bytes differ");
        }
        check_masks();
        check_linear_verify_window();
        check_draft_oracle();
        std::puts("PASS E2i CPU scatter, descriptor masks and draft-remap oracle");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL E2i CPU: %s\n", e.what());
        return 1;
    }
}
