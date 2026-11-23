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
    // The labels the next leaf receives (a deferring adapter restores them after emitting a held leaf).
    [[nodiscard]] std::string_view stage_label() const noexcept { return stage_; }
    [[nodiscard]] std::int32_t visit_label() const noexcept { return visit_; }

    // The emitter supplies geometry and arguments for the declared width.
    // Existing scalar emitters leave the width at its default.
    void kernel(KernelBinding binding, std::uint32_t rows = 1) {
        if (!binding.fn || binding.identity.empty() || binding.predicted.launches > 1U)
            throw std::invalid_argument("CUDA recipe kernel requires one launch thunk");
        const bool pdl = use_pdl(binding.pdl);
        append(std::move(binding), CudaNodeKind::kernel, graph::Op::elementwise, rows, pdl);
    }

    // Compatibility for adapters which declare their schedule at the call
    // site. An explicit true was their assertion of a wait in the thunk.
    // A native binding with a declared contract still has to agree.
    void kernel(KernelBinding binding, bool legacy_pdl, std::uint32_t rows = 1) {
        if (legacy_pdl) {
            if (binding.pdl.wait == PdlWait::undeclared) binding.pdl.wait = PdlWait::before_step_io;
            if (binding.pdl.wait != PdlWait::before_step_io || binding.pdl.opt_out != PdlOptOut::none)
                throw std::invalid_argument("explicit CUDA PDL edge conflicts with the kernel wait contract");
        } else if (binding.pdl.opt_out == PdlOptOut::none) {
            binding.pdl.opt_out = PdlOptOut::legacy_schedule;
        }
        kernel(std::move(binding), rows);
    }

    // X2c's eager thunk slot can also dispatch a synchronous host action.
    // Such an action is a recipe leaf but issues no kernel and cannot capture.
    void host_action(KernelBinding binding, std::uint32_t rows = 1) {
        if (!tapped_ || !binding.fn || binding.identity.empty() || binding.predicted.launches != 0U)
            throw std::invalid_argument("invalid CUDA recipe host action");
        binding.capturable = false;
        append(std::move(binding), CudaNodeKind::kernel, graph::Op::elementwise, rows, false, true);
    }

    void copy(CudaNodeKind kind, CudaCopyArgs args, std::uint32_t rows = 1) {
        if ((kind != CudaNodeKind::copy_d2d && kind != CudaNodeKind::copy_d2h_pinned) ||
            !args.destination || !args.source || args.bytes == 0)
            throw std::invalid_argument("invalid CUDA recipe copy binding");
        if (kind == CudaNodeKind::copy_d2d)
            append(native("cuda-native-copy-d2d-v1", args), kind, graph::Op::copy, rows, false);
        else
            append(native("cuda-native-copy-d2h-v1", args), kind, graph::Op::download, rows, false);
    }

    void clear(CudaMemsetArgs args, std::uint32_t rows = 1) {
        if (!args.destination || args.bytes == 0)
            throw std::invalid_argument("invalid CUDA recipe memset binding");
        append(native("cuda-native-memset-v1", args), CudaNodeKind::memset, graph::Op::clear, rows, false);
    }

    [[nodiscard]] CudaRecipeProgram compile(ProgramKey key, CudaReadback readback = CudaReadback::bound) const {
        if (nodes_.empty() || key.tap != tapped_) throw std::invalid_argument("invalid CUDA recipe or tap key");
        cuda_detail::validate_step_width(key);
        for (const auto& node : nodes_) {
            if (node.rows != key.rows)
                throw std::invalid_argument("CUDA recipe width unsupported at " + node.id +
                    ": requested R=" + std::to_string(key.rows) + ", bound R=" + std::to_string(node.rows));
        }
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
    static bool use_pdl(PdlLeaf policy) {
        switch (policy.wait) {
        case PdlWait::undeclared:
        case PdlWait::before_step_io:
        case PdlWait::absent: break;
        default: throw std::invalid_argument("invalid CUDA PDL wait declaration");
        }
        switch (policy.opt_out) {
        case PdlOptOut::none:
            if (policy.wait != PdlWait::before_step_io)
                throw std::invalid_argument("CUDA PDL leaf requires a grid dependency wait before step I/O");
            return true;
        case PdlOptOut::kernel_without_wait:
            if (policy.wait != PdlWait::absent)
                throw std::invalid_argument("CUDA PDL opt-out requires a body without a grid dependency wait");
            return false;
        case PdlOptOut::legacy_schedule:
            return false;
        }
        throw std::invalid_argument("invalid CUDA PDL opt-out reason");
    }

    struct Node {
        std::string id;
        KernelBinding kernel;
        CudaNodeKind kind;
        graph::Op op;
        std::int32_t visit;
        std::uint32_t rows;
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

    void append(KernelBinding kernel, CudaNodeKind kind, graph::Op op, std::uint32_t rows, bool pdl,
                bool host_action = false) {
        if (nodes_.size() >= no_launch - 1U) throw std::length_error("CUDA recipe node limit exceeded");
        // A predecessor without a trigger releases the edge on termination
        // of all its blocks. Transfers and tapped schedules keep completion.
        const bool pdl_edge = !tapped_ && pdl && !nodes_.empty() && nodes_.back().kind == CudaNodeKind::kernel;
        std::string tagged_chain(tapped_ ? "cuda-eager-recipe/v1/" : detail::cuda_capture_recipe);
        tagged_chain += pdl_edge ? "programmatic/" : "completion/";
        tagged_chain += kernel.chain_identity;
        kernel.chain_identity = std::move(tagged_chain);
        auto id = stage_.empty() ? kernel.identity : stage_;
        id += '/';
        id += std::to_string(nodes_.size());
        nodes_.push_back({std::move(id), std::move(kernel), kind, op, visit_, rows, pdl_edge, host_action});
    }

    std::vector<Node> nodes_;
    std::string stage_;
    std::int32_t visit_ = -1;
    bool tapped_;
};

}  // namespace strix::exec
