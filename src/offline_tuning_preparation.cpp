#include "strix/offline_tuning_preparation.hpp"
#include "strix/model_content_hash.hpp"
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace strix {
LoadedOfflineTuningCache load_offline_tuning_cache_file(const std::filesystem::path& path,
    kernels::OfflineTuningCacheLimits limits) {
    const auto size = std::filesystem::file_size(path);
    if (!size || size > limits.max_bytes || size > std::numeric_limits<std::size_t>::max() ||
        size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::invalid_argument("offline tuning file exceeds its bounded byte capacity");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("offline tuning file cannot be opened");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (input.gcount() != static_cast<std::streamsize>(size) || input.bad())
        throw std::invalid_argument("offline tuning file changed extent or was not read completely");
    const auto tail = input.peek();
    if (tail != std::char_traits<char>::eof() || input.bad())
        throw std::invalid_argument("offline tuning file changed extent or final read failed");
    LoadedOfflineTuningCache result;
    result.payload_sha256 = model_content_identity_hex(sha256_bytes(bytes));
    result.payload_bytes = static_cast<std::uint64_t>(size);
    result.records = kernels::decode_offline_tuning_cache(bytes, limits);
    return result;
}
namespace {
void append(std::vector<std::byte>& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64U; shift += 8U) bytes.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
}
void append(std::vector<std::byte>& bytes, const std::string& text) {
    append(bytes, text.size());
    for (const auto value : text) bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(value)));
}
std::string formats(std::uint32_t step, const ModelManifest& manifest, const ExecutionTensorAccessPlan& access) {
    std::vector<std::byte> bytes;
    for (const auto& item : access.accesses) {
        if (item.step_id != step) continue;
        append(bytes,item.binding_slot); append(bytes,item.tensor_indices.size());
        for (const auto index : item.tensor_indices) {
            if (index >= manifest.tensors.size()) throw std::invalid_argument("tuning access tensor index exceeds manifest");
            const auto& tensor = manifest.tensors[index];
            append(bytes,index); append(bytes,tensor.name); append(bytes,tensor.ggml_type);
            append(bytes,tensor.byte_size); append(bytes,tensor.dimensions.size());
            for (const auto dimension : tensor.dimensions) append(bytes,dimension);
        }
    }
    return "actual-selected-tensor-schema-sha256:" + model_content_identity_hex(sha256_bytes(bytes));
}
} // namespace
PreparedOfflineTuning::PreparedOfflineTuning(LoadedOfflineTuningCache cache,
    const OfflineTuningLiveContext& live, const ModelContract& model,
    const CompiledSerialExecution& baseline, const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& access, const kernels::KernelRegistry& registry,
    const ExecutionBindingRequest& request)
    : records_(std::move(cache.records)), payload_sha256_(std::move(cache.payload_sha256)), payload_bytes_(cache.payload_bytes) {
    if (request.offline_tuning || access.manifest_layout_identity != manifest.layout_identity ||
        baseline.bindings.bindings.size() != baseline.execution.steps.size())
        throw std::invalid_argument("offline tuning preparation differs from untuned source admission");
    keys_.resize(baseline.execution.steps.size());
    std::size_t expected_step = 0;
    for (const auto& step : baseline.execution.steps) {
        if (step.id != expected_step++ || step.id >= keys_.size()) throw std::invalid_argument("offline tuning source step IDs are not dense");
        const auto& binding = baseline.bindings.bindings[step.id];
        if (binding.step_id != step.id || binding.op != step.op)
            throw std::invalid_argument("offline tuning source binding differs from execution step");
        if (binding.host_or_runtime_step || !binding.kernel_id || !binding.family) continue;
        const auto* variant = registry.find(*binding.kernel_id);
        if (!variant) throw std::invalid_argument("offline tuning baseline kernel is not registered");
        auto& key = keys_[step.id];
        key.gfx_target=live.gfx_target; key.device_sku=live.device_sku; key.hip_identity=live.hip_identity;
        key.binary_sha256=live.binary_sha256; key.model_content_sha256=live.model_content_sha256;
        key.compiler_flags_sha256=live.compiler_flags_sha256;
        key.compute_units=live.runtime_reported_processor_count; // legacy field name; reported MP domain is explicit in SKU/receipt
        key.compiler_identity=variant->compiler_identity;
        key.input_precision=variant->numerics.input_precision;
        key.accumulator_precision=variant->numerics.accumulator_precision;
        key.output_precision=variant->numerics.output_precision;
        key.input_layout=variant->input_layout;key.output_layout=variant->output_layout;
        key.weight_format=formats(step.id,manifest,access);
        key.state_layout=live.state_layout_identity ? "layout-id:"+std::to_string(live.state_layout_identity) : std::string{};
        key.context_bucket=live.context_tokens && !live.resolved_configuration.empty() ?
            "exact-context:"+std::to_string(live.context_tokens)+";resolved:"+live.resolved_configuration : std::string{};
        key.memory_regime=live.memory_regime;
        key.family=*binding.family; key.phase=kernels::ExecutionPhase::decode;
        key.shape=execution_shape_for(model,step.op,request); key.wave=variant->wave;
        // This is the complete operator ShapeClass convention, not an invented
        // single-GEMM geometry for composite operators. Zero dimensions reject.
        key.m=key.shape.logical_rows;key.n=key.shape.hidden_size;key.k=key.shape.auxiliary_size;
        key.numerical_contract_identity=variant->numerical_contract_identity;
    }
}
} // namespace strix
