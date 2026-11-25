#include "core/dense_bf16_narrow_plan.hpp"
#include "core/mix_latency_plan.hpp"
#include "cuda/core/mix_cuda_latency_plan.hpp"
#include "strix/exec/catalog.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

using namespace strix;
using namespace strix::exec;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void geometry() {
    float storage[3]{};
    core::residual_mix::MixArgs mix{.banks = {storage}, .queries = storage + 1, .output = {storage + 2},
                                   .rows = 1, .width = 2304, .bank_count = 1, .query_count = 1, .epsilon = 1e-6F};
    const auto plan = [&](unsigned tile = 512U, unsigned maximum = 4096U) {
        return core::residual_mix::latency_grid(mix, tile, maximum);
    };
    auto grid = plan();
    require(grid.tiles == 5 && grid.queries == 1 && grid.banks == 1, "width 2304 has a partial final tile");
    mix.width = 513;
    grid = plan();
    require(grid.tiles == 2, "width 513 has a one-element tail");
    require(plan(256U, 1024U).tiles == 3 && plan(256U, 512U).tiles == 0 && plan(0U).tiles == 0,
            "caller tile and width limit are checked");
    mix.width = 4096;
    require(plan().tiles == 8, "maximum width fits");
    ++mix.width;
    require(plan().tiles == 0, "oversized width declined");
    mix.width = 2304;
    mix.rows = 2;
    require(plan().tiles == 0, "multi-row declined");
    mix.rows = 0;
    require(plan().tiles == 0, "empty readout declined");

    using core::bf16_narrow_blocks;
    static_assert(bf16_narrow_blocks(4608, 64, 32, 16) == 8);
    static_assert(bf16_narrow_blocks(32, 16, 32, 16) == 2);
    static_assert(bf16_narrow_blocks(32, 8, 32, 16) == 0);
    static_assert(bf16_narrow_blocks(31, 16, 32, 16) == 0);
    static_assert(bf16_narrow_blocks(0, 16, 32, 16) == 0);
    static_assert(bf16_narrow_blocks(32, 0, 32, 16) == 0);
    static_assert(bf16_narrow_blocks(16, 32, 16, 32) == 2);
    static_assert(bf16_narrow_blocks(32, 16, 0, 16) == 0);
    static_assert(bf16_narrow_blocks(32, 16, 32, 0) == 0);
    static_assert(bf16_narrow_blocks(32, 15, 32, 15) == 0);
}

// The batch contract: jobs numbered set by set, each job = one single-query readout, disjoint outputs required.
void batch() {
    namespace residual = core::residual_mix;
    constexpr unsigned width = 2304;
    std::vector<float> storage(std::size_t{width} * 104U);
    auto* bank = storage.data();                      // 2 rows
    auto* queries = bank + 2U * width;                // 48 rows
    auto* outputs = queries + 48U * width;            // 48 rows
    residual::BankBatchArgs args{.banks = {bank, bank + width}, .width = width, .bank_count = 2,
                                 .set_count = 2, .epsilon = 1e-6F, .normalized = true};
    args.sets[0] = {queries, outputs, 24};
    args.sets[1] = {queries + 24U * width, outputs + 24U * width, 24};
    require(residual::fits(args) && residual::job_count(args) == 48, "pass batch fits");
    const auto where = residual::locate(args, 30);
    require(where.set == 1 && where.index == 6, "job numbering is set by set");
    require(residual::locate(args, 48).set == 2, "out-of-range job is marked");
    const auto single = residual::job_args(args, 30);
    require(single.queries == queries + 30U * width && single.output[0] == outputs + 30U * width &&
            single.query_count == 1 && single.rows == 1 && single.bank_count == 2 && single.normalized &&
            single.banks[1] == bank + width && residual::fits(single), "job is the single-query readout");
    auto grid = cuda::mix::latency_batch_grid(args);
    require(grid.tiles == 5 && grid.queries == 48 && grid.banks == 2, "batch grid");
    bool thrown = false;
    try { (void)residual::job_args(args, 48); } catch (const std::out_of_range&) { thrown = true; }
    require(thrown, "job_args rejects an out-of-range job");

    auto bad = args;
    bad.sets[1].output = outputs + 23U * width;       // overlaps the last row of set 0
    require(!residual::fits(bad), "overlapping set outputs declined");
    bad = args;
    bad.sets[0].output = queries + 47U * width;       // overlaps a query image
    require(!residual::fits(bad), "output over a query image declined");
    bad = args;
    bad.sets[0].output = bank + 1U;                   // overlaps the bank
    require(!residual::fits(bad), "output over the bank declined");
    bad = args;
    bad.sets[1].count = 0;
    require(!residual::fits(bad), "empty set declined");
    bad = args;
    bad.set_count = residual::kMaxQuerySets + 1U;
    require(!residual::fits(bad), "too many sets declined");
    bad = args;
    bad.normalized = false;
    bad.epsilon = 0.0F;
    require(!residual::fits(bad), "raw banks need an epsilon");
    bad = args;
    bad.width = 4097;
    require(cuda::mix::latency_batch_grid(bad).tiles == 0, "oversized width declined");
}

void check_entry(KernelKey key, std::string_view identity, std::string_view mode) {
    const auto& catalog = Catalog::builtin(Backend::cuda);
    const auto rows = catalog.find(key, device::ArchMask::sm120);
    require(rows.size() == 1U && rows.front()->identity == identity, "variant key or identity");
    const auto& entry = *rows.front();
    require(entry.launcher->mode == mode && entry.launcher->max_rows == 1 && entry.capturable &&
            entry.reference == ReferenceClass::base && entry.exact == lever::Exactness::bitwise, "variant metadata");
    WorkspaceNeed need{128U, 16U};
    LaunchSite site{.scratch = need, .rows = 1};
    entry.workspace(site, need);
    require(need.bytes == 0 && need.alignment == 1, "unexpected workspace");
    const auto cost = entry.cost(site, {});
    require(cost.launches == 1 && std::isinf(cost.us), "unexpected measured cost");
    key.rows = RowsClass::r2_8;
    require(catalog.find(key, device::ArchMask::sm120).empty(), "unexpected multi-row variant");
}

} // namespace

int main() {
    try {
        geometry();
        batch();
        check_entry({graph::Op::mix, "residual.bank.batch", {}, OperandClass::f32_fma, RowsClass::r1},
                    "cuda-sm120-residual-bank-batch-tiled-r1-v1", "bank_softmax.f32.native.tiled.batch");
        check_entry({graph::Op::mix, "residual.bank.tiled", {}, OperandClass::f32_fma, RowsClass::r1},
                    "cuda-sm120-residual-bank-softmax-tiled-r1-v1", "bank_softmax.f32.native.tiled");
        check_entry({graph::Op::linear, "r1.bf16_image.output_split_round", {"plain.bf16", 1},
                     OperandClass::bf16_mma, RowsClass::r1},
                    "cuda-sm120-bf16-r1-mma3-output-split-round-v1", "mma3_round.output_split");
        std::puts("PASS MIXLAT host shape guards and D4 catalogue");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL MIXLAT %s\n", error.what());
        return 1;
    }
}
