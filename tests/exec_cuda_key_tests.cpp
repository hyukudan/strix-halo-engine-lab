#include "executor_cuda.hpp"
#include "strix/device/decode_geometry.hpp"
#include "executor_cuda_binding.hpp"

#include <array>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace {

using namespace strix::exec;
using cuda_detail::GraphAction;
using cuda_detail::GraphBinding;
using cuda_detail::capture_plan;

void require(bool condition, const char* detail) {
    if (!condition) throw std::runtime_error(detail);
}

void test_keys() {
    const ProgramKey base{};
    std::map<ProgramKey, GraphBinding> cache;
    cache[base];
    auto distinct = [&](auto member, auto value) {
        auto key = base;
        key.*member = value;
        require(cache.try_emplace(key).second, "ProgramKey variant reused an executable slot");
    };
    distinct(&ProgramKey::phase, Phase::verify);
    distinct(&ProgramKey::rows, 8U);
    distinct(&ProgramKey::pos, static_cast<strix::graph::Pos>(1));
    distinct(&ProgramKey::tree, true);
    distinct(&ProgramKey::sampled, true);
    distinct(&ProgramKey::tap, true);
    distinct(&ProgramKey::graph_id, std::uint8_t{1});
    distinct(&ProgramKey::tier, static_cast<strix::profile::Name>(1));
    distinct(&ProgramKey::ctx_bucket, std::uint8_t{1});
    require(!cache.try_emplace(base).second, "same key did not reuse its executable slot");
}

void test_invalidation() {
    Program base{};
    base.identity = 1;
    base.launches.resize(1);
    base.args = {std::byte{1}};
    GraphBinding binding;
    require(binding.select(base, 0) == GraphAction::capture, "first use must capture");
    binding.remember(base, 0, true);
    require(binding.select(base, 0) == GraphAction::replay, "same program must replay");
    require(binding.select(base, 1) == GraphAction::replay, "stable bindings survive layout change");
    binding.remember(base, 1, true);

    auto moved = base;
    moved.identity = 2;  // X1 identity includes the workspace layout.
    moved.args.front() = std::byte{2};
    require(binding.select(moved, 2) == GraphAction::rebind, "same kernel recipe must rebind moved addresses");
    binding.remember(moved, 2, true);
    require(binding.select(moved, 2) == GraphAction::replay, "updated executable must replay");

    moved.args.front() = std::byte{3};
    require(binding.select(moved, 3) == GraphAction::rebind, "explicit invalidation must refresh changed arguments");
    binding.remember(moved, 3, true);
    const auto equivalent = moved;
    require(binding.select(equivalent, 3) == GraphAction::replay, "refreshed binding must match its copied bytes");

    auto changed = moved;
    changed.launches.front().grid.blocks[0] = 2;
    require(binding.select(changed, 4) == GraphAction::capture, "grid changes must recapture");
    binding.remember(changed, 4, false);
    require(binding.select(changed, 4) == GraphAction::eager, "failed capture must fall back and stay eager");
    require(binding.select(changed, 5) == GraphAction::capture, "layout invalidation permits another attempt");
    changed.key.tap = true;
    require(binding.select(changed, 5) == GraphAction::eager, "a tap must stay eager");
}

void test_launch_shapes() {
    namespace geo = strix::device::decode_geometry;
    require(geo::pointwise(1).blocks[0] == 1 && geo::pointwise(257, 8).blocks == std::array{2U, 8U, 1U} &&
            geo::pointwise(65535U * 256U + 1U).blocks[0] == 65535, "pointwise launch boundaries");
    require(geo::head_clip(24, 8, 32).threads[0] == 32 && geo::head_clip(24, 8, 33).threads[0] == 128 &&
            geo::head_clip(24, 8, 129).threads[0] == 256, "head clip instantiation boundaries");
    const auto flash = geo::prepared_flash128(24, 8);
    require(flash.blocks == std::array{24U, 8U, 1U} && flash.shared_bytes == 35840U, "padded flash layout");
    require(geo::dense_fma(9).blocks[0] == 2 && geo::dense_mma(7, 2).blocks == std::array{7U, 2U, 1U},
            "dense projection tile grid");
}

void test_strict_capture() {
    Program program{};
    program.launches.resize(1);
    for (auto policy : {CudaReadback::executor, CudaReadback::bound, CudaReadback::resident}) {
        CudaProgramBinding binding{};
        binding.readback = policy;
        for (auto phase : {Phase::decode, Phase::verify, Phase::prefill, Phase::draft}) {
            program.key.phase = phase;
            require(cuda_detail::strict_capture(program, binding, true), "debug capture must check every readback");
            require(cuda_detail::strict_capture(program, binding, false) == (policy == CudaReadback::resident),
                    "release capture policy differs");
        }
    }
    program.launches.front().chain_identity = detail::cuda_capture_recipe;
    require(cuda_detail::strict_capture(program, {}, false), "tagged recipes must check their captured nodes");
    auto& launch = program.launches.front();
    launch.identity = "screened pass 1";
    launch.grid = {{1128, 1, 1}, {32, 1, 1}, 9216};
    const auto observed = launch.grid;
    cuda_detail::validate_capture_grid(observed, launch);
    const auto rejects = [&](const Grid& stale) {
        launch.grid = stale;
        try {
            cuda_detail::validate_capture_grid(observed, launch);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    auto stale = observed;
    stale.blocks[0] = 1504;
    require(rejects(stale), "old screened grid accepted");
    for (unsigned axis = 0; axis < 3; ++axis) {
        stale = observed;
        ++stale.blocks[axis];
        require(rejects(stale), "incorrect grid axis accepted");
        stale = observed;
        ++stale.threads[axis];
        require(rejects(stale), "incorrect block axis accepted");
    }
    stale = observed;
    ++stale.shared_bytes;
    require(rejects(stale), "incorrect dynamic shared memory accepted");
}

void test_node_plan() {
    using enum CudaNodeKind;
    const auto pdl = CudaEdgeKind::programmatic;
    Program program{};
    program.launches.resize(6);
    program.launches[3].after = 2;  // Explicit root supersedes the legacy edge.
    const CudaProgramBinding mixed{{{memset, {}}, {kernel, {}}, {copy_d2d, {{1}}}, {kernel, {}},
                                    {kernel, {{0}, {2}, {3, pdl}}}, {copy_d2h_pinned, {{4}}}}, CudaReadback::bound};
    const auto plan = capture_plan(program, mixed, true);
    const std::array kinds{memset, kernel, copy_d2d, kernel, kernel, copy_d2h_pinned};
    struct Edge {
        std::size_t to;
        CudaDependency from;
        bool operator==(const Edge&) const = default;
    };
    const std::vector<Edge> expected{{2, {1}}, {4, {0}}, {4, {2}}, {4, {3, pdl}}, {5, {4}}};
    std::vector<Edge> actual;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        require(plan[i].kind == kinds[i], "capture node kind differs");
        for (const auto& edge : plan[i].after) actual.push_back({i, edge});
    }
    require(actual == expected, "explicit captured edges differ");
    auto completion = mixed;
    completion.nodes[4].after.back().kind = CudaEdgeKind::completion;
    require(capture_plan(program, mixed, false) == completion.nodes, "capability fallback changed edge endpoints");
    auto launch_completion = mixed;
    launch_completion.nodes[4].after.back().kind = CudaEdgeKind::launch_completion;
    require(capture_plan(program, launch_completion, true) == launch_completion.nodes &&
            capture_plan(program, launch_completion, false) == completion.nodes, "launch-completion port changed");

    const auto legacy = capture_plan(program, {}, true);
    require(legacy[3].after == std::vector<CudaDependency>{{2, pdl}} &&
            legacy[1].after == std::vector<CudaDependency>{{0}}, "legacy serial/PDL edges changed");
    auto nonadjacent = program;
    nonadjacent.launches[3].after = 1;
    require(capture_plan(nonadjacent, {}, true)[3].after == std::vector<CudaDependency>{{2}, {1, pdl}},
            "legacy nonadjacent edge lost the serial predecessor");

    GraphBinding cached;
    cached.remember(program, 0, true, mixed);
    const auto equivalent = mixed;
    require(cached.select(program, 0, equivalent) == GraphAction::replay, "equivalent node plan did not replay");
    require(cached.select(program, 0, completion) == GraphAction::capture, "edge type change reused a graph");
    require(cached.select(program, 0, launch_completion) == GraphAction::capture, "edge port change reused a graph");
    auto changed = mixed;
    changed.nodes[4].after.push_back({1});
    require(capture_plan(program, changed, true)[4].after.size() == 4 &&
            cached.select(program, 0, changed) == GraphAction::capture, "added predecessor reused a graph");
    changed = mixed;
    changed.nodes[2].kind = memset;
    require(cached.select(program, 0, changed) == GraphAction::capture, "node kind change reused a graph");

    const auto reject = [&](const CudaProgramBinding& invalid) {
        try {
            (void)capture_plan(program, invalid, true);
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error("invalid CUDA capture plan accepted");
    };
    changed = mixed;
    changed.nodes[4].after.push_back({0});
    reject(changed);
    changed = mixed;
    changed.nodes[1].after = {{2}};
    reject(changed);
    changed = mixed;
    changed.nodes[4].after.front().kind = pdl;
    reject(changed);
    changed = mixed;
    changed.nodes.back().kind = copy_d2d;
    reject(changed);
}

void test_descriptor_projection() {
    constexpr StepRow row{19, 4700, 3, 1, 5, step_valid | step_sample, 27, 45};
    constexpr auto prior = legacy_step_row(row);
    static_assert(prior.input.position == 4700 && prior.input.token == 19 && prior.slot == 3 && prior.parent == 1 &&
                  prior.ancestors == 5 && prior.flags == (step_valid | step_sample) && prior.rng_lo == 27 && prior.rng_hi == 45);
    require(offsetof(StepRow, token) != offsetof(strix::session::StepDescriptor, token), "descriptor ABIs collapsed");
}

}  // namespace

int main() {
    try {
        test_keys();
        test_invalidation();
        test_node_plan();
        test_launch_shapes();
        test_strict_capture();
        test_descriptor_projection();
        std::puts("exec_cuda_keys PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "exec_cuda_keys FAIL: %s\n", error.what());
        return 1;
    }
}
