#include "executor_cuda.hpp"
#include "executor_cuda_binding.hpp"

#include <cstdio>
#include <map>
#include <stdexcept>

namespace {

using namespace strix::exec;
using cuda_detail::GraphAction;
using cuda_detail::GraphBinding;

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

}  // namespace

int main() {
    try {
        test_keys();
        test_invalidation();
        std::puts("exec_cuda_keys PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "exec_cuda_keys FAIL: %s\n", error.what());
        return 1;
    }
}
