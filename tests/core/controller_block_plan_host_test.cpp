// Host admission and row policy of the fused controller (no device access).
#include "core/controller_block_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace cb = strix::core::controller_block;

static_assert(cb::row_regs(2304U) == 9U);
static_assert(cb::row_regs(2048U) == 0U);
static_assert(cb::row_regs(0U) == 0U);
// one latent per wave (8 waves x 8 blocks = rank), gate blocks of 128 outputs tile the width
static_assert(cb::kDownBlocks * (cb::kThreads / 32U) == cb::kRank);
static_assert(2304U % cb::kGateOutputs == 0U && 2304U / cb::kGateOutputs == 18U);
static_assert(cb::scratch_elements_for(1U, 2304U) == 2U * 2304U + 64U + 1U);
static_assert(cb::scratch_elements_for(16U, 2304U) == 16U * (2U * 2304U + 64U + 1U));
// its rows fit the controller scratch past its message rows (residual_mix LowRankController: 4 width + rank)
static_assert(cb::scratch_elements_for(16U, 2304U) <= 16U * (4U * 2304U + 64U) - 16U * 2304U);

namespace {

int failures = 0;

void expect(const char* name, bool got, bool want) {
    if (got != want) {
        std::printf("controller-block-plan: FAIL %s admitted %d expected %d\n", name, got ? 1 : 0, want ? 1 : 0);
        ++failures;
    }
}

}  // namespace

int main() {
    constexpr unsigned width = 2304U, rank = 64U, rows = 8U;
    // 64-byte aligned arena: rows [rows][width] regions, the weights and the scale, all disjoint.
    alignas(64) static unsigned char arena[16U * 1024U * 1024U];
    constexpr std::size_t row_bytes = std::size_t{rows} * width * 4U;
    const auto at = [](std::size_t offset) { return static_cast<void*>(arena + offset); };
    cb::Args base;
    base.raw = static_cast<const float*>(at(0U));
    base.bank0 = static_cast<const float*>(at(1U * 1048576U));
    base.bank1 = static_cast<const float*>(at(2U * 1048576U));
    base.query = static_cast<const float*>(at(3U * 1048576U));
    base.down = static_cast<const std::uint16_t*>(at(4U * 1048576U));
    base.forget = static_cast<const std::uint16_t*>(at(5U * 1048576U));
    base.gain = static_cast<const std::uint16_t*>(at(6U * 1048576U));
    base.scale = static_cast<const std::uint16_t*>(at(7U * 1048576U));
    base.output = static_cast<float*>(at(8U * 1048576U));
    base.rows = rows;
    base.width = width;
    base.rank = rank;
    base.operand_epsilon = 1.0e-6F;
    base.norm_epsilon = 1.0e-6F;
    base.round_operands = true;
    base.round_latent = true;
    base.scratch = static_cast<float*>(at(10U * 1048576U));
    base.scratch_elements = cb::scratch_elements_for(rows, width);
    static_assert(row_bytes < 1048576U);

    expect("banks", cb::admits(base), true);
    // decode route policy: every row count of the band, nothing outside it
    expect("preferred 0", cb::preferred(0U), false);
    expect("preferred 1", cb::preferred(1U), true);
    expect("preferred 8", cb::preferred(8U), true);
    expect("preferred 9", cb::preferred(9U), true);
    expect("preferred 16", cb::preferred(16U), true);
    expect("preferred 17", cb::preferred(17U), false);
    auto a = base;
    a.bank0 = a.bank1 = nullptr;
    a.message = static_cast<const float*>(at(9U * 1048576U));
    expect("given message", cb::admits(a), true);
    a.bank0 = base.bank0;
    expect("message and one bank", cb::admits(a), false);
    a = base;
    a.bank1 = nullptr;
    expect("one bank", cb::admits(a), false);
    a = base;
    a.bank0 = a.bank1 = nullptr;
    expect("no readout", cb::admits(a), false);
    a = base;
    a.query = nullptr;
    expect("banks without query", cb::admits(a), false);
    a = base;
    a.output = const_cast<float*>(base.raw);
    expect("in place", cb::admits(a), true);
    a = base;
    a.output = const_cast<float*>(base.raw) + width;
    expect("raw shifted by a row", cb::admits(a), false);
    a = base;
    a.output = const_cast<float*>(base.bank1);
    expect("output is a bank", cb::admits(a), false);
    a = base;
    a.scale = reinterpret_cast<const std::uint16_t*>(base.output) + 8;
    expect("scale inside output", cb::admits(a), false);
    a = base;
    a.rows = 16U;
    expect("16 rows on an 8-row scratch", cb::admits(a), false);
    a.scratch_elements = cb::scratch_elements_for(16U, width);
    expect("16 rows", cb::admits(a), true);
    a.rows = 17U;
    expect("17 rows", cb::admits(a), false);
    a.rows = 0U;
    expect("0 rows", cb::admits(a), false);
    a = base;
    a.width = 2048U;
    expect("width not instantiated", cb::admits(a), false);
    a = base;
    a.rank = 32U;
    expect("rank 32", cb::admits(a), false);
    a = base;
    a.round_latent = false;
    expect("unrounded latent", cb::admits(a), false);
    a = base;
    a.round_operands = false;
    expect("unrounded operands", cb::admits(a), false);
    a = base;
    a.norm_epsilon = 0.0F;
    expect("norm epsilon 0", cb::admits(a), false);
    a = base;
    a.operand_epsilon = -1.0F;
    expect("operand epsilon negative", cb::admits(a), false);
    a = base;
    a.down = reinterpret_cast<const std::uint16_t*>(reinterpret_cast<const unsigned char*>(base.down) + 2);
    expect("unaligned down", cb::admits(a), false);
    a = base;
    a.gain = nullptr;
    expect("null gain", cb::admits(a), false);

    // scratch rows apart from everything the launches read or write
    a = base;
    a.scratch = nullptr;
    expect("no scratch", cb::admits(a), false);
    a = base;
    a.scratch_elements -= 1U;
    expect("scratch one short", cb::admits(a), false);
    a = base;
    a.scratch = base.scratch + 1;
    expect("scratch unaligned", cb::admits(a), false);
    a = base;
    a.scratch = base.output + width;
    expect("scratch inside output", cb::admits(a), false);
    a = base;
    a.scratch = const_cast<float*>(base.bank1);
    expect("scratch is a bank", cb::admits(a), false);
    a = base;
    a.bank0 = a.bank1 = nullptr;
    a.message = base.scratch + 4U;
    expect("scratch over the message", cb::admits(a), false);
    a = base;
    a.scratch = reinterpret_cast<float*>(const_cast<std::uint16_t*>(base.gain));
    expect("scratch over gain", cb::admits(a), false);
    a = base;
    a.scratch = const_cast<float*>(base.query) - 8;
    expect("scratch over query", cb::admits(a), false);

    if (failures == 0) std::printf("controller-block-plan: PASS\n");
    return failures == 0 ? 0 : 1;
}
