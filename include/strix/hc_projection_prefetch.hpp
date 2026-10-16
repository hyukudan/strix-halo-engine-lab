#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/model_runtime_binding.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::hc_projection_prefetch {
inline constexpr std::string_view marker = ".prefetch-next-tile-v1";
inline constexpr std::uint64_t normalized_bytes = 10240U * 4U;
inline constexpr std::uint64_t low_bytes = 320U * 4U;
inline constexpr std::uint64_t down_bytes = 320U * 320U * 34U;
inline constexpr std::uint64_t inject_bytes = 10240U * 4U * 4U;

struct View {
    std::uintptr_t address;
    std::uint64_t bytes;
};

inline void validate_view(View view, std::uint64_t required, std::uint64_t alignment = 16U) {
    if (!view.address || view.address % alignment || view.bytes < required ||
        view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
        throw std::invalid_argument("HC next-tile view alignment or extent invalid");
}

inline bool overlaps(View a, View b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}

// All checks precede normalization, which writes the normalized slice. The
// projection then reads that slice and writes low and optional injection.
inline void validate_projection_views(View normalized, View low, View down,
                                     std::optional<View> inject, std::optional<View> injection) {
    if (inject.has_value() != injection.has_value())
        throw std::invalid_argument("HC next-tile injection views must be paired");
    validate_view(normalized, normalized_bytes);
    validate_view(low, low_bytes);
    validate_view(down, down_bytes);
    if (inject) { validate_view(*inject, inject_bytes); validate_view(*injection, 16U); }
    if (overlaps(normalized, down) || (inject && overlaps(normalized, *inject)) ||
        overlaps(low, normalized) || overlaps(low, down) || (inject && overlaps(low, *inject)) ||
        (injection && (overlaps(*injection, normalized) || overlaps(*injection, down) ||
                       overlaps(*injection, *inject) || overlaps(*injection, low))))
        throw std::invalid_argument("HC next-tile mutable views overlap");
}

struct Weight {
    std::uint32_t index = 0;
    std::uint64_t bytes = 0;
    friend bool operator==(const Weight&, const Weight&) = default;
};
struct Mixer {
    std::uint32_t step_id = 0;
    std::array<Weight, 4> weights{};
    bool final = false;
    friend bool operator==(const Mixer&, const Mixer&) = default;
};
struct Plan {
    bool active = false;
    std::uint64_t execution_identity = 0;
    std::uint64_t manifest_identity = 0;
    std::uint64_t access_identity = 0;
    std::vector<Mixer> mixers;
    friend bool operator==(const Plan&, const Plan&) = default;
};

inline bool mixer_op(SerialOp op) {
    return op == SerialOp::attn_hyper_mix || op == SerialOp::mlp_hyper_mix || op == SerialOp::final_hyper_mix;
}

inline std::uint64_t linear_extent(std::uint32_t type, std::uint64_t width, std::uint64_t rows) {
    if (!width || !rows || width > 10240U || rows > 10240U)
        throw std::invalid_argument("HC next-tile linear geometry exceeds canonical bounds");
    switch (type) {
    case 0U: return width * rows * 4U;
    case 1U: case 30U: return width * rows * 2U;
    case 8U:
        if (width % 32U == 0U) return width / 32U * 34U * rows;
        break;
    case 23U:
        if (width % 256U == 0U) return width / 256U * 136U * rows;
        break;
    default: break;
    }
    throw std::invalid_argument("HC next-tile weight format or block width invalid");
}

// Phase A: validate every selected mixer before residency or state creation.
inline Plan build_plan(const ModelContract& model, const ModelManifest& manifest,
                       const CompiledSerialExecution& compiled,
                       const ExecutionTensorAccessPlan& access, bool enabled) {
    if (!enabled) return {};
    if (!model.validate().empty() || model.hidden_size != 2560U || model.hc_count != 4U ||
        model.hc_lowrank != 320U || model.hidden_layers != 48U || model.bytes_bf16 != 2U ||
        model.bytes_fp32 != 4U || !compiled.execution_identity || compiled.logical_rows != 1U ||
        compiled.request_concurrency != 1U || compiled.device_graph_inputs ||
        !manifest.layout_identity || access.manifest_layout_identity != manifest.layout_identity ||
        !access.plan_identity || manifest.tensors.size() > std::numeric_limits<std::uint32_t>::max() ||
        compiled.io.steps.size() != compiled.execution.steps.size() ||
        compiled.bindings.bindings.size() != compiled.execution.steps.size())
        throw std::invalid_argument("HC next-tile requires canonical eager C1 model bindings");
    Plan plan{true, compiled.execution_identity, manifest.layout_identity, access.plan_identity, {}};
    std::array<unsigned, 97> seen{};
    for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
        const auto& step = compiled.execution.steps[i];
        const auto& binding = compiled.bindings.bindings[i];
        if (!mixer_op(step.op)) {
            if (binding.kernel_id && binding.kernel_id->find(marker) != std::string::npos)
                throw std::invalid_argument("HC next-tile marker on another operator");
            continue;
        }
        const bool final = step.op == SerialOp::final_hyper_mix;
        if ((final && step.layer) || (!final && (!step.layer || *step.layer >= 48U)))
            throw std::invalid_argument("HC next-tile mixer layer invalid");
        const auto key = final ? 96U : 2U * *step.layer + (step.op == SerialOp::mlp_hyper_mix ? 1U : 0U);
        if (++seen[key] != 1U || step.id != i || binding.step_id != step.id || binding.op != step.op ||
            binding.host_or_runtime_step || !binding.kernel_id ||
            binding.kernel_id->find(marker) == std::string::npos ||
            binding.kernel_id->find(marker) != binding.kernel_id->rfind(marker))
            throw std::invalid_argument("HC next-tile selected mixer coverage or identity invalid");
        const auto& io = compiled.io.steps[i];
        if (io.step_id != step.id || io.op != step.op || io.reads.size() != 1U ||
            io.writes.size() != (final ? 1U : 2U) || io.temporaries.size() != 1U ||
            io.temporaries.front().buffer.buffer_class != ExecutionBufferClass::hyper_workspace)
            throw std::invalid_argument("HC next-tile mixer IO topology invalid");
        Mixer mixer{step.id, {}, final};
        const std::string prefix = final ? "output_hc_" :
            step.op == SerialOp::attn_hyper_mix ? "hc_attn_" : "hc_ffn_";
        constexpr std::array<std::string_view, 4> suffixes{"norm", "down", "up", "inject"};
        for (std::uint32_t slot = 0; slot < (final ? 3U : 4U); ++slot) {
            const auto* selected = find_execution_tensor_access(access, step.id, slot);
            if (!selected || selected->op != step.op || selected->tensor_indices.size() != 1U ||
                std::count_if(access.accesses.begin(), access.accesses.end(), [&](const auto& item) {
                    return item.step_id == step.id && item.binding_slot == slot;
                }) != 1 || selected->tensor_indices.front() >= manifest.tensors.size())
                throw std::invalid_argument("HC next-tile selected tensor access invalid");
            const auto index = selected->tensor_indices.front();
            const auto& tensor = manifest.tensors[index];
            const auto role = slot == 0U ? TensorRole::normalization : TensorRole::dense_projection;
            if (tensor.role != role || tensor.layer != step.layer ||
                tensor.name.find(prefix + std::string(suffixes[slot])) == std::string::npos)
                throw std::invalid_argument("HC next-tile tensor family differs from selected mixer");
            std::uint64_t required = 0;
            if (slot == 0U) {
                if (tensor.dimensions != std::vector<std::uint64_t>{10240U} ||
                    (tensor.ggml_type != 0U && tensor.ggml_type != 1U && tensor.ggml_type != 30U))
                    throw std::invalid_argument("HC next-tile normalization tensor invalid");
                required = 10240U * (tensor.ggml_type == 0U ? 4U : 2U);
            } else {
                const auto width = slot == 2U ? 320U : 10240U;
                const auto rows = slot == 1U ? 320U : slot == 2U ? 10240U : 4U;
                if (tensor.dimensions != std::vector<std::uint64_t>{width, rows} ||
                    (slot == 1U && tensor.ggml_type != 8U) || (slot == 3U && tensor.ggml_type != 0U))
                    throw std::invalid_argument("HC next-tile projection tensor invalid");
                required = linear_extent(tensor.ggml_type, width, rows);
                if ((slot == 1U || slot == 3U) && tensor.byte_size != required)
                    throw std::invalid_argument("HC next-tile producer tensor byte size invalid");
            }
            if (tensor.absolute_offset % 16U || tensor.byte_size < required ||
                tensor.byte_size > std::numeric_limits<std::uint64_t>::max() - tensor.absolute_offset)
                throw std::invalid_argument("HC next-tile file alignment or extent invalid");
            mixer.weights[slot] = {index, tensor.byte_size};
        }
        plan.mixers.push_back(mixer);
    }
    if (std::any_of(seen.begin(), seen.end(), [](auto count) { return count != 1U; }))
        throw std::invalid_argument("HC next-tile requires every canonical mixer");
    return plan;
}

// Phase B: recheck the snapshot and actual device-visible owners before fresh
// persistent state is initialized. This is setup work, never a per-token scan.
inline void validate_runtime(const Plan& plan, const ModelContract& model, const ModelManifest& manifest,
                             const CompiledSerialExecution& compiled, const ExecutionTensorAccessPlan& access,
                             const ExecutionMemoryBinding& memory, const ModelRuntimeBinding& binding) {
    if (!plan.active) {
        if (!plan.mixers.empty()) throw std::invalid_argument("inactive HC next-tile plan contains mixers");
        return;
    }
    if (plan != build_plan(model, manifest, compiled, access, true) ||
        binding.manifest_layout_identity != plan.manifest_identity)
        throw std::invalid_argument("HC next-tile prepared bindings changed");
    validate_execution_memory_binding(compiled.execution_identity, compiled.arena, memory);
    const auto resolve = [&](ExecutionBufferUse use) {
        const auto view = resolve_execution_buffer(compiled.execution_identity, compiled.arena, memory, use);
        if (view.address < memory.arena_base || view.address - memory.arena_base > memory.arena_bytes ||
            view.bytes > memory.arena_bytes - (view.address - memory.arena_base))
            throw std::invalid_argument("HC next-tile arena view exceeds actual allocation");
        return View{view.address, view.bytes};
    };
    for (const auto& mixer : plan.mixers) {
        std::array<View, 4> weights{};
        for (std::size_t slot = 0; slot < (mixer.final ? 3U : 4U); ++slot) {
            const auto& required = mixer.weights[slot];
            const auto* weight = find_runtime_tensor_binding(binding, required.index);
            if (!weight || !runtime_tensor_gpu_visible(weight->storage) || !weight->allocation_generation)
                throw std::invalid_argument("HC next-tile requires retained GPU-visible weights");
            weights[slot] = {weight->address, weight->bytes};
            validate_view(weights[slot], required.bytes);
        }
        const auto& io = compiled.io.steps[mixer.step_id];
        validate_view(resolve(io.reads.front()), 10240U * 2U, 2U);
        validate_view(resolve(io.writes.front()), 2560U * 2U, 2U);
        const auto temporary = resolve(io.temporaries.front());
        validate_view(temporary, normalized_bytes + low_bytes);
        const View normalized{temporary.address, normalized_bytes};
        const View low{temporary.address + normalized_bytes, low_bytes};
        std::optional<View> inject, injection;
        if (!mixer.final) { inject = weights[3]; injection = resolve(io.writes[1]); }
        validate_projection_views(normalized, low, weights[1], inject, injection);
        if (overlaps(normalized, weights[0]) || overlaps(normalized, weights[2]) ||
            overlaps(low, weights[0]) || overlaps(low, weights[2]) ||
            (injection && (overlaps(*injection, weights[0]) || overlaps(*injection, weights[2]))))
            throw std::invalid_argument("HC next-tile workspace aliases retained weights");
    }
}
} // namespace strix::hc_projection_prefetch
