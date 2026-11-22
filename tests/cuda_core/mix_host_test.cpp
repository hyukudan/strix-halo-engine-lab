#include "cuda/core/mix_catalog.hpp"
#include "mix_oracle.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace {

namespace test = strix::tests::mix;
namespace exec = strix::exec;
using strix::device::ArchMask;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void oracle_bits() {
    constexpr float zero[4]{};
    constexpr float alternating[] = {1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F};
    constexpr float zero_rows[8]{};
    test::MixArgs args{.banks = {zero_rows, alternating}, .queries = zero,
                       .rows = 2U, .width = 4U, .bank_count = 2U, .query_count = 1U, .epsilon = 1.0F};
    const auto raw = test::readout_oracle(args);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const auto expected = alternating[i] > 0.0F ? 0x3eb504f3U : 0xbeb504f3U;
        require(test::bits(raw[i]) == expected, "raw RMS and equal-score readout bits");
    }

    const float log_two = std::log(2.0F);
    const float queries[] = {log_two, -log_two};
    args = {.banks = {zero, alternating}, .queries = queries,
            .rows = 1U, .width = 1U, .bank_count = 2U, .query_count = 2U, .normalized = true};
    const auto unequal = test::readout_oracle(args);
    require(test::bits(unequal[0]) == 0x3f2aaaabU && test::bits(unequal[1]) == 0x3eaaaaabU,
            "normalized bank and unequal-score readout bits");

    std::array<float, 513> chain{}, zeros{}, ones{};
    chain[0] = 0x1p24F;
    chain[256] = 1.0F;
    chain[512] = -0x1p24F;
    ones.fill(1.0F);
    args = {.banks = {chain.data(), zeros.data()}, .queries = ones.data(),
            .rows = 1U, .width = 513U, .bank_count = 2U, .query_count = 1U, .normalized = true};
    const auto strided = test::readout_oracle(args);
    require(test::bits(strided[256]) == 0x3f000000U, "strided FMA chain order");

    chain = {};
    chain[0] = 0x1p24F;
    chain[64] = 1.0F;
    chain[128] = -0x1p24F;
    args.width = 256U;
    const auto halving = test::readout_oracle(args);
    const float expected = 1.0F / (1.0F + std::exp(-0.0625F));
    require(test::bits(halving[64]) == test::bits(expected), "halving tree lane order");
}

int enqueue_stub(void*, void*) { return 0; }

exec::EntryLaunch resolve(void* context, const exec::LauncherRef& launcher, const exec::LaunchSite& site) {
    require(launcher.symbol == "strix::cuda::mix::enqueue" && launcher.mode == "bank_softmax.f32.native",
            "launcher identity");
    require(site.backend_args == context, "backend arguments forwarded");
    return {enqueue_stub, context};
}

void catalog() {
    exec::Catalog table;
    table.add(exec::Catalog::builtin_entries(exec::Backend::cuda));
    const auto fragment = strix::cuda::mix::catalog_entries();
    require(fragment.size() == 4U, "one entry per row class");
    const std::array bounds{1U, 2U, 9U, 33U};
    const std::array ends{1U, 8U, 32U, 65536U};
    for (std::size_t i = 0; i < fragment.size(); ++i) {
        const auto& entry = fragment[i];
        const exec::KernelKey key{strix::graph::Op::mix, "residual.bank", {}, exec::OperandClass::f32_fma,
                                  static_cast<exec::RowsClass>(i)};
        const auto found = table.find(key, ArchMask::sm120);
        require(found.size() == 1U && found.front()->identity == entry.identity, "CUDA catalogue lookup (generated builtin row)");
        require(table.find(key, ArchMask::gfx1151).empty(), "CUDA entry selected for HIP");
        require(entry.identity == "cuda-sm120-residual-bank-softmax-v1" &&
                entry.exact == strix::lever::Exactness::bitwise && entry.reference == exec::ReferenceClass::base &&
                entry.capturable, "BASE identity metadata");
        exec::EntryBinder binder{resolve, &table};
        exec::LaunchSite site{.weight_bytes = 16U, .state_bytes = 32U, .macs = 64U,
                              .scratch = {128U, 16U}, .rows = bounds[i], .backend_args = &table};
        exec::WorkspaceNeed workspace = site.scratch;
        entry.workspace(site, workspace);
        require(workspace.bytes == 0U && workspace.alignment == 1U, "no global workspace");
        const auto cost = entry.cost(site, {});
        require(cost.launches == 1U && cost.weight_bytes == 16U && cost.state_bytes == 32U &&
                cost.macs == 64U && std::isinf(cost.us), "one launch with unmeasured cost");
        require(entry.bind(site, binder).binding == &table, "row class lower bound");
        site.rows = ends[i];
        require(entry.bind(site, binder).binding == &table, "row class upper bound");
        site.rows = bounds[i] - 1U;
        require(!entry.bind(site, binder), "rows below the class bound admitted");
        if (i + 1U < fragment.size()) {
            site.rows = ends[i] + 1U;
            require(!entry.bind(site, binder), "rows above the class bound admitted");
        }
        site.rows = bounds[i];
        binder.resolve = nullptr;
        require(!entry.bind(site, binder), "missing resolver admitted");
    }
}

} // namespace

int main() {
    try {
        oracle_bits();
        std::puts("PASS K8c CPU bank readout oracle");
        catalog();
        std::puts("PASS K8c D4 CUDA bank readout fragment");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL K8c %s\n", error.what());
        return 1;
    }
}
