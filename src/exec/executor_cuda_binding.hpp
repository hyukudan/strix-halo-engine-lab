#pragma once

#include "executor_cuda.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix::exec::cuda_detail {

enum class GraphAction { eager, capture, rebind, replay };

inline const CudaProgramBinding empty_binding{};

inline void validate_step_width(const ProgramKey& key) {
    if ((key.phase == Phase::decode || key.phase == Phase::verify) && !session::is_step_replay_width(key.rows))
        throw std::invalid_argument("unsupported CUDA step width R=" + std::to_string(key.rows));
}

inline void validate_step_feed(std::uint32_t valid, std::uint32_t feed, std::uint32_t steps) {
    if ((feed & ~valid) != 0 || (steps > 1 && feed == 0))
        throw std::invalid_argument("invalid CUDA step feed mask");
}

// Validate the block that the executor uploads to its stable device address.
// The width includes masked rows, even when only a subset of slots is active.
inline void validate_x3_input(const Program& program, const StepBlock& input, std::uint32_t steps) {
    if (input.h.rows != program.key.rows || !session::is_step_replay_width(input.h.rows) ||
        input.h.stop_count > kStepMaxStops || (input.h.stop_count != 0 && step_stop_address(input.h) == 0))
        throw std::invalid_argument("invalid CUDA X3 step header");
    std::uint32_t valid = 0;
    for (std::uint32_t i = 0; i < input.h.rows; ++i) {
        if ((step_row(input, i).flags & step_valid) != 0) valid |= std::uint32_t{1} << i;
    }
    if (valid != input.h.valid_mask) throw std::invalid_argument("invalid CUDA X3 step valid mask");
    validate_step_feed(valid, input.h.feed_mask, steps);
}

[[nodiscard]] inline CudaNodeKind node_kind(const CudaProgramBinding& binding, std::size_t index) {
    return binding.nodes.empty() ? CudaNodeKind::kernel : binding.nodes[index].kind;
}

inline void validate_plan(const Program& program, const CudaProgramBinding& binding) {
    const auto size = program.launches.size();
    const auto& nodes = binding.nodes;
    if (size == 0 || (!nodes.empty() && nodes.size() != size)) throw std::invalid_argument("invalid CUDA node count");
    if (binding.readback == CudaReadback::bound && std::ranges::none_of(nodes, [](const auto& node) {
            return node.kind == CudaNodeKind::copy_d2h_pinned;
        })) throw std::invalid_argument("bound CUDA readback needs a pinned D2H node");
    if (binding.readback == CudaReadback::resident && std::ranges::any_of(nodes, [](const auto& node) {
            return node.kind != CudaNodeKind::kernel;
        })) throw std::invalid_argument("resident CUDA graph requires kernel nodes");
    if (nodes.empty()) {
        for (std::size_t i = 0; i < size; ++i) {
            const auto after = program.launches[i].after;
            if (after != no_launch && after >= i) throw std::invalid_argument("invalid CUDA predecessor");
        }
        return;
    }
    for (std::size_t i = 0; i < size; ++i) {
        const auto& after = nodes[i].after;
        for (auto it = after.begin(); it != after.end(); ++it) {
            if (it->predecessor >= i || std::find_if(after.begin(), it, [&](const CudaDependency& edge) {
                    return edge.predecessor == it->predecessor;
                }) != it)
                throw std::invalid_argument("invalid CUDA predecessor list");
            if (it->kind != CudaEdgeKind::completion &&
                (nodes[i].kind != CudaNodeKind::kernel || nodes[it->predecessor].kind != CudaNodeKind::kernel))
                throw std::invalid_argument("programmatic CUDA edges require kernels");
        }
    }
}

// Launch order is topological and is also the eager order. Explicit lists
// specify all incoming edges. Without PDL the same endpoints wait for completion.
[[nodiscard]] inline std::vector<CudaNodeBinding> capture_plan(const Program& program,
                                                              const CudaProgramBinding& binding, bool pdl) {
    validate_plan(program, binding);
    auto nodes = binding.nodes;
    if (nodes.empty()) {
        nodes.resize(program.launches.size());
        for (std::size_t i = 1; i < nodes.size(); ++i) {
            const auto after = program.launches[i].after;
            if (!pdl || after != i - 1) nodes[i].after.push_back({i - 1});
            if (pdl && after != no_launch) nodes[i].after.push_back({after, CudaEdgeKind::programmatic});
        }
    } else if (!pdl) {
        for (auto& node : nodes) {
            for (auto& dependency : node.after) dependency.kind = CudaEdgeKind::completion;
        }
    }
    return nodes;
}

// A serial stream already captures these exact completion dependencies.
[[nodiscard]] inline bool is_completion_chain(std::span<const CudaNodeBinding> nodes) {
    if (nodes.empty() || !nodes.front().after.empty()) return false;
    for (std::size_t i = 1; i < nodes.size(); ++i) {
        const auto& after = nodes[i].after;
        if (after.size() != 1 || after.front() != CudaDependency{i - 1}) return false;
    }
    return true;
}

// One reusable executable per map key. Cache the exact capture inputs so an
// arena epoch change with stable indirection can keep the executable as-is.
class GraphBinding {
public:
    [[nodiscard]] GraphAction select(const Program& program, std::uint64_t epoch,
                                     const CudaProgramBinding& binding = empty_binding) const {
        if (!program.capturable || program.key.tap) return GraphAction::eager;
        if (!source_ || program.key != key_) return GraphAction::capture;
        if (matches(program, epoch, binding)) return ready_ ? GraphAction::replay : GraphAction::eager;
        if (binding != binding_ || !same_plan(program, epoch)) return GraphAction::capture;
        if (ready_) return program.args == args_ ? GraphAction::replay : GraphAction::rebind;
        return epoch != epoch_ || program.args != args_ ? GraphAction::capture : GraphAction::eager;
    }

    void remember(const Program& program, std::uint64_t epoch, bool ready,
                  const CudaProgramBinding& binding = empty_binding) {
        if (!matches(program, epoch, binding)) {
            auto args = program.args;
            auto launches = program.launches;
            auto snapshot = binding;
            args_ = std::move(args);
            launches_ = std::move(launches);
            binding_ = std::move(snapshot);
        }
        source_ = &program;
        binding_source_ = &binding;
        key_ = program.key;
        identity_ = program.identity;
        epoch_ = epoch;
        ready_ = ready;
    }

    [[nodiscard]] bool matches(const Program& program, std::uint64_t epoch,
                               const CudaProgramBinding& binding = empty_binding) const noexcept {
        return &program == source_ && &binding == binding_source_ && program.identity == identity_ && epoch == epoch_;
    }

private:
    [[nodiscard]] bool same_plan(const Program& program, std::uint64_t epoch) const {
        // X1's identity includes the arena layout. Across an explicit layout
        // change, a matching kernel recipe can update the existing graph.
        if ((program.identity != identity_ && epoch == epoch_) || program.launches.size() != launches_.size())
            return false;
        for (std::size_t i = 0; i < launches_.size(); ++i) {
            const BoundLaunch& a = launches_[i];
            const BoundLaunch& b = program.launches[i];
            if (a.kernel != b.kernel || a.fn != b.fn || a.entry != b.entry || a.identity != b.identity ||
                a.chain_identity != b.chain_identity || a.after != b.after || a.stream != b.stream ||
                a.grid.blocks != b.grid.blocks || a.grid.threads != b.grid.threads ||
                a.grid.shared_bytes != b.grid.shared_bytes || a.args_offset != b.args_offset ||
                a.args_bytes != b.args_bytes || a.capturable != b.capturable) return false;
        }
        return true;
    }

    const Program* source_ = nullptr;
    const CudaProgramBinding* binding_source_ = nullptr;
    ProgramKey key_{};
    std::uint64_t identity_ = 0;
    std::uint64_t epoch_ = 0;
    bool ready_ = false;
    std::vector<std::byte> args_;
    std::vector<BoundLaunch> launches_;
    CudaProgramBinding binding_;
};

}  // namespace strix::exec::cuda_detail
