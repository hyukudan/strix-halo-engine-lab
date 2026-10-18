#pragma once

#include "strix/execution_runtime.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <cstddef>
#include <string_view>

namespace strix {

[[nodiscard]] ExecutionBufferAddress resolve_dispatch_read(
    const ExecutionDispatchContext& context,
    std::size_t index);

[[nodiscard]] ExecutionBufferAddress resolve_dispatch_write(
    const ExecutionDispatchContext& context,
    std::size_t index);

[[nodiscard]] ExecutionBufferAddress resolve_dispatch_temporary(
    const ExecutionDispatchContext& context,
    std::size_t index);

// Arena-hinted forms (see resolve_execution_buffer with a hint): for rows admitted like a
// head row of the same step over row views of one known-window/prefill arena.
[[nodiscard]] ExecutionBufferAddress resolve_dispatch_read_hinted(
    const ExecutionDispatchContext& context, std::size_t index, std::size_t& hint);
[[nodiscard]] ExecutionBufferAddress resolve_dispatch_write_hinted(
    const ExecutionDispatchContext& context, std::size_t index, std::size_t& hint);
[[nodiscard]] ExecutionBufferAddress resolve_dispatch_temporary_hinted(
    const ExecutionDispatchContext& context, std::size_t index, std::size_t& hint);

[[nodiscard]] SessionStateAddress resolve_dispatch_state(
    const ExecutionDispatchContext& context,
    SessionStateComponent component);

[[nodiscard]] SessionStateAddress resolve_dispatch_gdn_layer_state(
    const ExecutionDispatchContext& context,
    const ModelContract& model,
    SessionStateComponent component);

[[nodiscard]] const ExecutionTensorAccess& resolve_dispatch_tensor_access(
    const ExecutionDispatchContext& context,
    std::uint32_t binding_slot);

[[nodiscard]] const RuntimeTensorBinding& resolve_dispatch_tensor(
    const ExecutionDispatchContext& context,
    std::uint32_t tensor_index);

[[nodiscard]] const RuntimeTensorBinding& resolve_dispatch_tensor(
    const ExecutionDispatchContext& context,
    std::string_view tensor_name);

// Validates one manifest/binding pair on construction. Until destruction, resolve_dispatch_tensor
// and validate_model_runtime_binding_unless_scoped on this thread do not re-validate that same pair
// (other pairs still are). The owner must keep both objects alive and unmodified for the scope's
// lifetime (one admitted prefill block's or known window's launch).
class ValidatedModelBindingScope {
public:
    ValidatedModelBindingScope(const ModelManifest& manifest, const ModelRuntimeBinding& binding);
    ~ValidatedModelBindingScope();
    ValidatedModelBindingScope(const ValidatedModelBindingScope&) = delete;
    ValidatedModelBindingScope& operator=(const ValidatedModelBindingScope&) = delete;

private:
    const ModelManifest* previous_manifest_;
    const ModelRuntimeBinding* previous_binding_;
};

// validate_model_runtime_binding, except for the pair of this thread's innermost
// ValidatedModelBindingScope, which was validated when the scope opened.
void validate_model_runtime_binding_unless_scoped(
    const ModelManifest& manifest, const ModelRuntimeBinding& binding);

}  // namespace strix
