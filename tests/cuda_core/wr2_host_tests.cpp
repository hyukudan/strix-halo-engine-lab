#include "exec/cuda_decode_wr2.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace {

using namespace strix;
namespace rows = core::decode_state;

void check(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

void check_topology() {
    using namespace exec;
    StepBlock block{};
    block.h.rows = 8U;
    block.h.valid_mask = 0x1fU;
    block.row[0] = {0U, 63U, 1U, kStepNoParent, 0x01U, step_valid, 0U, 0U};
    block.row[1] = {0U, 63U, 0U, kStepNoParent, 0x02U, step_valid, 0U, 0U};
    block.row[2] = {0U, 64U, 1U, 0U, 0x05U, step_valid, 0U, 0U};
    block.row[3] = {0U, 64U, 1U, 0U, 0x09U, step_valid, 0U, 0U};
    block.row[4] = {0U, 65U, 1U, 3U, 0x3bU, step_valid, 0U, 0U};
    block.row[5] = {0U, 65U, 1U, 3U, 0x29U, step_valid, 0U, 0U};
    check(rows::pending_row(block, 4U, 65U) == 4U, "inclusive self or padded key");
    check(rows::pending_row(block, 4U, 64U) == 3U, "sibling leaked into pending keys");
    check(rows::pending_row(block, 4U, 63U) == 0U, "another slot leaked into pending keys");
    check(rows::pending_row(block, 4U, 62U) == kStepNoParent, "committed key resolved to pending");
    check(rows::parent_row(block, 4U) == 3U && rows::parent_row(block, 1U) == kStepNoParent,
          "parent or root differs");
    block.row[4].parent = 1U;
    check(rows::parent_row(block, 4U) == kStepNoParent, "cross-slot or nonconsecutive parent admitted");
    block.row[4].parent = 3U;
    block.row[3].flags = 0U;
    check(rows::parent_row(block, 4U) == kStepNoParent && rows::pending_row(block, 4U, 64U) == kStepNoParent,
          "disabled parent admitted");
}

void check_directory() {
    std::array<std::uint64_t, 2> pages{100'000U, 200'000U};
    std::array<state::SlotEntry, 4> entries{{{16U, 4U, 0U, 80U, 0U}, {32U, 7U, 0U, 80U, 0U},
        {reinterpret_cast<std::uint64_t>(pages.data()), 63U, 0U, 100U, 0U}, {300'000U, 64U, 0U, 100U, 0U}}};
    const rows::SlotView table{{entries.data(), 2U, 2U}, 1U};
    const rows::PositionView paged{table, 32U, 0U, true};
    const rows::PositionView ring{table, 32U, 16U, false};
    const rows::PositionView contiguous{table, 32U, 0U, false};
    check(table.valid() && table.entry(0U) == &entries[2] && table.entry(1U) == &entries[3] && !table.entry(2U),
          "state-major slot lookup");
    check(paged.valid(32U) && !paged.valid(64U) && paged.address(entries[2], 63U) == 102'016U &&
          paged.address(entries[2], 65U) == 200'032U, "page-directory address");
    check(ring.address(entries[3], 65U) == 300'032U && contiguous.address(entries[3], 65U) == 302'080U,
          "ring or dense address");
    check(rows::accepts(table.entry(0U), 63U) && !rows::accepts(table.entry(0U), 62U) &&
          !rows::accepts(table.entry(0U), 100U) && !rows::accepts(nullptr, 0U), "capacity or frontier admission");
}

void fake_launch(std::span<const std::byte>, const exec::Grid&, void*) {}

void check_widths() {
    using namespace exec;
    using namespace exec::cuda_decode;
    for (const unsigned width : {1U, 2U, 4U, 8U, 16U}) {
        check(supports_wr2_width(width), "required width refused");
        CudaProgramRecipe recipe;
        recipe.stage("wr2_test");
        KernelBinding binding;
        binding.fn = fake_launch;
        binding.identity = "wr2-host-test-r" + std::to_string(width);
        binding.predicted.launches = 1U;
        BoundStepLeaf{std::move(binding), width}.append_to(recipe);
        ProgramKey key{};
        key.rows = width;
        const auto program = recipe.compile(key, CudaReadback::executor);
        check(program.program.launches.size() == 1U && program.program.launches[0].rows == width,
              "recipe did not carry the width");
        key.rows = width == 1U ? 2U : 1U;
        try {
            (void)recipe.compile(key, CudaReadback::executor);
        } catch (const std::invalid_argument& error) {
            check(std::string_view(error.what()).starts_with("CUDA recipe width unsupported"), "wrong refusal");
            continue;
        }
        throw std::runtime_error("recipe accepted a different width");
    }
    check(!supports_wr2_width(0U) && !supports_wr2_width(3U) && !supports_wr2_width(32U), "unsupported WR2 width");
}

}  // namespace

int main() {
    try {
        check_topology();
        check_directory();
        check_widths();
        std::puts("wr2_host PASS widths=1,2,4,8,16 topology=tree+slots addressing=dense+ring+paged");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "wr2_host FAIL %s\n", error.what());
        return 1;
    }
}
