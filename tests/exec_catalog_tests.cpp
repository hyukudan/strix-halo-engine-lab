#include "strix/exec/catalog.hpp"
// X1 and D4 headers coexist in one translation unit (no shared type names).
#include "strix/exec/compiler.hpp"
#include "strix/exec/program.hpp"

#include "strix/device/runtime.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

using namespace strix::exec;
namespace device = strix::device;
namespace graph = strix::graph;
namespace weight = strix::weight;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct BindingFixture {
    const LauncherRef* expected = nullptr;
    unsigned calls = 0;
};

int enqueue(void* binding, void* stream) { return binding == stream ? 0 : 1; }

EntryLaunch resolve(void* context, const LauncherRef& launcher, const LaunchSite& site) {
    auto& fixture = *static_cast<BindingFixture*>(context);
    if (&launcher != fixture.expected || site.backend_args != context) return {};
    ++fixture.calls;
    return {enqueue, context};
}

void check_entries(Backend backend, device::ArchMask arch, device::ArchMask other_arch, std::size_t expected_count) {
    const auto& catalog = Catalog::builtin(backend);
    const auto entries = Catalog::builtin_entries(backend);
    require(entries.size() == expected_count && catalog.size() == entries.size(), "unexpected builtin key count");
    BindingFixture fixture;
    EntryBinder arena{resolve, &fixture};
    device::Profile profile{};
    profile.memory_gbytes_ps = 1;
    profile.peaks.fp32 = 1;
    profile.peaks.fp16 = 2;
    profile.peaks.bf16 = 4;

    const auto& weights = weight::builtin_registry();
    for (const auto& entry : entries) {
        const auto found = catalog.find(entry.key, arch);
        require(found.size() == 1 && found.front() == &entry, "key must resolve uniquely to its original entry");
        require(catalog.find(entry.key, other_arch).empty(), "wrong architecture must miss");
        require(catalog.find(entry.key, device::ArchMask::none).empty(), "unknown architecture must miss");
        require(entry.reference == ReferenceClass::base && !entry.identity.empty(), "missing BASE identity");
        require(entry.key.weight.name.empty() || weight::find_format(weights, entry.key.weight),
                "weight format absent from the registry");

        LaunchSite site;
        site.rows = std::array{1U, 2U, 9U, 33U}.at(static_cast<std::size_t>(entry.key.rows));
        site.backend_args = &fixture;
        site.weight_bytes = entry.key.weight.name.empty() ? 0U : 1000U;
        site.state_bytes = 1000;
        site.macs = 500000;
        site.launches = 2;
        site.scratch = {4096, 128};
        const EntryCost cost = entry.cost(site, profile);
        require(cost.us == (site.weight_bytes == 0 ? 1.0 : 2.0) && cost.weight_bytes == site.weight_bytes &&
                cost.state_bytes == site.state_bytes && cost.macs == site.macs && cost.launches == site.launches,
                "cost lost site measurements or profile units");
        WorkspaceNeed reported{};
        entry.workspace(site, reported);
        require(reported.bytes == site.scratch.bytes && reported.alignment == site.scratch.alignment,
                "scratch requirements changed");

        fixture.expected = entry.launcher;
        const auto bound = entry.bind(site, arena);
        require(bound && bound.enqueue(bound.binding, &fixture) == 0, "host launcher binding did not resolve");
        site.rows = 0;
        require(!entry.bind(site, arena), "zero rows reached the resolver");
    }
    require(fixture.calls == entries.size(), "each valid builtin site must bind exactly once");
}

void check_named_keys() {
    using enum RowsClass;
    using enum OperandClass;
    const auto& cuda = Catalog::builtin(Backend::cuda);
    const auto& hip = Catalog::builtin(Backend::hip);
    const device::Capabilities sm120{12, 0};
    const KernelKey bf16{graph::Op::linear, "rows", weight::kBf16WeightStorage, f32_fma, r9_32};
    const auto direct = cuda.find(bf16, sm120);
    require(direct.size() == 1 && direct.front()->launcher->symbol == "strix::cuda_core::dense_rows" &&
            direct.front()->launcher->mode == "fp32", "CUDA FP32 dense launcher");
    const KernelKey mma{graph::Op::linear, "rows.multi", weight::kQ8WeightStorage, bf16_mma, r2_8};
    const auto grouped = cuda.find(mma, sm120);
    require(grouped.size() == 1 && grouped.front()->launcher->symbol == "strix::cuda_core::dense_mma_launch" &&
            grouped.front()->launcher->mode == "mma3", "CUDA MMA3 dense launcher");
    const KernelKey nvfp4{graph::Op::linear, "rows.multi", {"gguf.nvfp4_tile.strix", 1}, f16_mma, r1};
    const auto w4a16 = cuda.find(nvfp4, sm120);
    require(w4a16.size() == 1 && w4a16.front()->launcher->symbol == "strix::cuda_core::nvfp4_dense_rows_multi" &&
            !w4a16.front()->capturable, "CUDA NVFP4 W4A16 multi launcher and scratch guard");
    require(cuda.find(bf16, device::Capabilities{8, 9}).empty(), "unlisted CUDA arch");

    LaunchSite compute_site{};
    compute_site.macs = 1000000;
    device::Profile profile{};
    require(std::isinf(grouped.front()->cost(compute_site, profile).us), "missing compute peak must be unpriced");
    profile.peaks.bf16 = 2;
    profile.peaks.fp32 = 1;
    require(grouped.front()->cost(compute_site, profile).us == 1 && direct.front()->cost(compute_site, profile).us == 2,
            "compute cost must select the matching operand peak");

    const auto hip_arch = device::arch_mask("gfx1151");
    const auto gemv = hip.find(bf16, hip_arch);
    require(gemv.size() == 1 && gemv.front()->launcher->symbol == "strix::core::dense_linear::enqueue_bf16_tight_rows",
            "HIP direct BF16 GEMV launcher");
    BindingFixture fixture{gemv.front()->launcher, 0};
    EntryBinder arena{resolve, &fixture};
    LaunchSite site{};
    site.backend_args = &fixture;
    site.rows = 16;
    require(static_cast<bool>(gemv.front()->bind(site, arena)), "HIP direct GEMV row 16");
    site.rows = 17;
    require(!gemv.front()->bind(site, arena), "HIP direct GEMV row 17 is outside this partial rows class");
    site.rows = 8;
    require(!gemv.front()->bind(site, arena) && fixture.calls == 1, "wrong row class must decline before binding");
    require(device::arch_mask("sm_120a") == device::ArchMask::sm120 &&
            device::arch_mask("gfx0000") == device::ArchMask::none, "architecture mapping");
}

void check_add() {
    const auto entries = Catalog::builtin_entries(Backend::cuda);
    const auto& entry = entries.front();
    Catalog custom;
    custom.add(entries.first(1));
    const auto prior = custom.find(entry.key, entry.arch);
    auto other_math = entry;
    other_math.key.math = Math::ieee_cr;
    auto other_revision = entry;
    ++other_revision.key.weight.revision;
    const std::array duplicate{other_math, entry};
    try {
        custom.add(duplicate);
        throw std::runtime_error("duplicate addition accepted");
    } catch (const std::invalid_argument&) {
        require(custom.find(other_math.key, entry.arch).empty() && custom.size() == 1, "partial duplicate addition");
    }
    custom.add(std::span{&other_math, 1U});
    custom.add(std::span{&other_revision, 1U});
    require(entry.key != other_math.key && custom.find(other_math.key, entry.arch).front() == &other_math &&
            custom.find(other_revision.key, entry.arch).front() == &other_revision && prior.front() == &entry,
            "math and format revision must participate in the key while earlier lookups stay valid");
    auto unresolved = entry;
    unresolved.key.prologue = 1;
    unresolved.bind = nullptr;
    try {
        custom.add(std::span{&unresolved, 1U});
        throw std::runtime_error("incomplete entry accepted");
    } catch (const std::invalid_argument&) {
        require(custom.find(unresolved.key, entry.arch).empty(), "incomplete entry was published");
    }
}

} // namespace

int main() {
    try {
        check_entries(Backend::cuda, device::ArchMask::sm120, device::ArchMask::gfx1151, 54);
        check_entries(Backend::hip, device::ArchMask::gfx1151, device::ArchMask::sm120, 25);
        check_named_keys();
        check_add();
        require(Catalog::builtin(Backend::cpu).size() == 0 && Catalog::builtin(Backend::sycl).size() == 0,
                "unexpected CPU or SYCL builtin");
        std::cout << "PASS exec_catalog: 54 CUDA keys, 25 HIP keys, host bindings and cost fields\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL exec_catalog: " << error.what() << '\n';
        return 1;
    }
}
