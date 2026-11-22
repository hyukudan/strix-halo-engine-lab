#pragma once

#include "executor_cuda_binding.hpp"
#include "native_launch_contract.hpp"
#include "strix/exec/compiler.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::exec {

struct CudaRecipeProgram {
    Program program;
    CudaProgramBinding binding;
    std::vector<BoundEntryCost> entry_costs;  // Diagnostic facts in program launch order.
};

// Lower the ordered leaves emitted by an existing single-stream schedule.
// Several leaves may share a semantic stage or visit. A leaf always emits
// one CUDA node, so the compiler can report its own grid and dependency.
class CudaProgramRecipe {
public:
    explicit CudaProgramRecipe(bool tapped = false) : tapped_(tapped) {}

    void visit(std::int32_t logical) {
        if (logical < -1) throw std::invalid_argument("invalid CUDA recipe visit");
        visit_ = logical;
    }

    void stage(std::string_view semantic) { stage_ = semantic; }

    void kernel(KernelBinding binding, bool pdl = false) {
        if (!binding.fn || binding.identity.empty() || binding.predicted.launches > 1U)
            throw std::invalid_argument("CUDA recipe kernel requires one launch thunk");
        append(std::move(binding), CudaNodeKind::kernel, graph::Op::elementwise, pdl);
    }

    // X2c's eager thunk slot can also dispatch a synchronous host action.
    // Such an action is a recipe leaf but issues no kernel and cannot capture.
    void host_action(KernelBinding binding) {
        if (!tapped_ || !binding.fn || binding.identity.empty() || binding.predicted.launches != 0U)
            throw std::invalid_argument("invalid CUDA recipe host action");
        binding.capturable = false;
        append(std::move(binding), CudaNodeKind::kernel, graph::Op::elementwise, false, true);
    }

    void copy(CudaNodeKind kind, CudaCopyArgs args) {
        if ((kind != CudaNodeKind::copy_d2d && kind != CudaNodeKind::copy_d2h_pinned) ||
            !args.destination || !args.source || args.bytes == 0)
            throw std::invalid_argument("invalid CUDA recipe copy binding");
        if (kind == CudaNodeKind::copy_d2d)
            append(native("cuda-native-copy-d2d-v1", args), kind, graph::Op::copy, false);
        else
            append(native("cuda-native-copy-d2h-v1", args), kind, graph::Op::download, false);
    }

    void clear(CudaMemsetArgs args) {
        if (!args.destination || args.bytes == 0)
            throw std::invalid_argument("invalid CUDA recipe memset binding");
        append(native("cuda-native-memset-v1", args), CudaNodeKind::memset, graph::Op::clear, false);
    }

    [[nodiscard]] CudaRecipeProgram compile(ProgramKey key, CudaReadback readback = CudaReadback::bound) const {
        if (nodes_.empty() || key.tap != tapped_) throw std::invalid_argument("invalid CUDA recipe or tap key");
        std::vector<graph::StageDecl> global;
        std::vector<graph::StageDecl> trunk;
        global.reserve(nodes_.size());
        trunk.reserve(nodes_.size());
        graph::LaunchPlan plan;
        plan.instances.reserve(nodes_.size());
        plan.launches.reserve(nodes_.size());
        CudaProgramBinding binding;
        binding.readback = readback;
        binding.nodes.reserve(nodes_.size());
        std::int32_t max_visit = -1;
        for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            auto& table = node.visit < 0 ? global : trunk;
            max_visit = std::max(max_visit, node.visit);
            const auto form = node.kind == CudaNodeKind::kernel ? std::string_view{} : detail::native_transfer_form;
            table.push_back({.id = node.id, .op = node.op, .form = form, .entry = node.kernel.identity,
                             .out = node.id});
            plan.instances.push_back({&table.back(), node.visit, graph::Format::none});
            const auto previous = i == 0 ? no_launch : i - 1U;
            plan.launches.push_back({.entry = node.kernel.identity, .stages = {i}, .layer = node.visit,
                                     .rows = key.rows, .after = previous, .dependent = node.pdl_edge});
            CudaNodeBinding bound{node.kind, {}};
            if (previous != no_launch)
                bound.after.push_back({previous, node.pdl_edge ? CudaEdgeKind::programmatic : CudaEdgeKind::completion});
            binding.nodes.push_back(std::move(bound));
        }

        graph::validate_plan(plan);
        const auto visits = max_visit < 0 ? 0U : static_cast<std::uint32_t>(max_visit) + 1U;
        const graph::StageGraph declaration{.before = global, .layer = trunk, .layers = std::vector<graph::Traits>(visits)};
        const Compiler compiler(declaration,
            [](const ProgramKey&, const graph::Instance&) { return BufferLayout{BufferStorage::external}; },
            [this](const LaunchBindingRequest& request) { return nodes_[request.launch_index].kernel; });
        auto program = compiler.compile(key, plan);
        std::vector<BoundEntryCost> entry_costs;
        entry_costs.reserve(nodes_.size());
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            entry_costs.push_back(nodes_[i].kernel.entry_cost);
            if (nodes_[i].host_action) {
                program.launches[i].kernels = 0;
                --program.predicted.launches;
            }
        }
        cuda_detail::validate_plan(program, binding);
        return {std::move(program), std::move(binding), std::move(entry_costs)};
    }

private:
    struct Node {
        std::string id;
        KernelBinding kernel;
        CudaNodeKind kind;
        graph::Op op;
        std::int32_t visit;
        bool pdl_edge;
        bool host_action;
    };

    template <class Args>
    static KernelBinding native(std::string_view identity, const Args& args) {
        KernelBinding result;
        result.identity = result.exact_identity = identity;
        const auto bytes = std::as_bytes(std::span(&args, 1U));
        result.args.assign(bytes.begin(), bytes.end());
        return result;
    }

    void append(KernelBinding kernel, CudaNodeKind kind, graph::Op op, bool pdl, bool host_action = false) {
        if (nodes_.size() >= no_launch - 1U) throw std::length_error("CUDA recipe node limit exceeded");
        // The receiving kernel opts in. A predecessor without a trigger
        // releases the edge on termination of all its blocks.
        const bool pdl_edge = !tapped_ && pdl && !nodes_.empty() && nodes_.back().kind == CudaNodeKind::kernel;
        std::string tagged_chain(tapped_ ? "cuda-eager-recipe/v1/" : detail::cuda_capture_recipe);
        tagged_chain += pdl_edge ? "programmatic/" : "completion/";
        tagged_chain += kernel.chain_identity;
        kernel.chain_identity = std::move(tagged_chain);
        auto id = stage_.empty() ? kernel.identity : stage_;
        id += '/';
        id += std::to_string(nodes_.size());
        nodes_.push_back({std::move(id), std::move(kernel), kind, op, visit_, pdl_edge, host_action});
    }

    std::vector<Node> nodes_;
    std::string stage_;
    std::int32_t visit_ = -1;
    bool tapped_;
};

}  // namespace strix::exec
