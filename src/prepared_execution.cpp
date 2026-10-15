#include "strix/prepared_execution.hpp"
#include "strix/prepared_tensor_views.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {
constexpr std::array state_flags{
    PersistentState::attention_kv, PersistentState::recurrent_matrix,
    PersistentState::convolution_history, PersistentState::sparse_index,
    PersistentState::ngram_history, PersistentState::position_history};

bool same_layout(const SerialSessionStateLayout& left, const SerialSessionStateLayout& right) {
    if (left.context_tokens != right.context_tokens || left.device_alignment != right.device_alignment ||
        left.host_alignment != right.host_alignment || left.device_total_bytes != right.device_total_bytes ||
        left.host_total_bytes != right.host_total_bytes || left.layout_identity != right.layout_identity ||
        left.allocations.size() != right.allocations.size()) return false;
    for (std::size_t index = 0; index < left.allocations.size(); ++index) {
        const auto& a = left.allocations[index]; const auto& b = right.allocations[index];
        if (a.component != b.component || a.state_flag != b.state_flag || a.storage != b.storage ||
            a.offset != b.offset || a.bytes != b.bytes || a.provisional != b.provisional) return false;
    }
    return true;
}
} // namespace

struct PreparedEagerExecution::Impl {
    const CompiledSerialExecution compiled;
    const std::shared_ptr<const void> actual_owner;
    std::unique_ptr<const PreparedTensorViews> tensor_views;
    RuntimeExecutionIdentities identities;
    SerialSessionStateLayout state_layout;
    std::array<std::uint64_t, 6> write_counts{};
    PersistentState required_state = PersistentState::none;
    std::vector<std::uint32_t> order;
    std::optional<std::uint32_t> ple_step;

    Impl(CompiledSerialExecution source, const ExecutionRuntimeResources& resources,
         const EagerExecutionRequest& prototype, std::shared_ptr<const void> owner)
        : compiled(std::move(source)), actual_owner(std::move(owner)) {
        if (!actual_owner) throw std::invalid_argument("prepared eager execution requires retained actual owners");
        if (compiled.execution.steps.size() > std::numeric_limits<std::uint32_t>::max() ||
            !compiled.vocab_size || !compiled.max_position_embeddings)
            throw std::invalid_argument("prepared eager compiled bounds are invalid");
        // This existing identity builder calls the complete runtime-resource
        // validator once (at position zero). The request position is separately
        // checked below, so no second full resource-validation pass is needed.
        identities = build_runtime_execution_identities(compiled.execution_identity, compiled.arena, resources);
        validate_eager_execution_preflight(compiled.execution, compiled.bindings, compiled.io, compiled.arena,
            prototype, resources.session_state_layout, resources.session_state_memory, &resources);
        tensor_views = std::make_unique<const PreparedTensorViews>(compiled.execution_identity,
            compiled.execution, *resources.model_manifest, *resources.model_weights,
            *resources.tensor_access, actual_owner);
        state_layout = *resources.session_state_layout;
        order.reserve(compiled.execution.steps.size());
        for (const auto& step : compiled.execution.steps) {
            if (!ple_step && step.op == SerialOp::ple) ple_step = step.id;
            required_state = required_state | step.reads_state | step.writes_state;
            order.push_back(step.id);
            for (std::size_t index = 0; index < state_flags.size(); ++index)
                if (has_state(step.writes_state, state_flags[index])) ++write_counts[index];
        }
        validate_request(resources, prototype);
    }

    void validate_request(const ExecutionRuntimeResources& resources, const EagerExecutionRequest& request) const {
        if (request.logical_rows != compiled.logical_rows || request.request_concurrency != compiled.request_concurrency ||
            !request.logical_rows || !request.request_concurrency)
            throw std::invalid_argument("prepared eager request shape differs from the immutable plan");
        if ((!request.input_tokens.empty() && request.input_tokens.size() != request.logical_rows) ||
            (request.logical_rows > 1U && request.input_tokens.empty()))
            throw std::invalid_argument("prepared eager input tokens do not match logical rows");
        if (request.input_tokens.empty()) {
            if (static_cast<std::uint64_t>(request.input_token_id) >= compiled.vocab_size)
                throw std::invalid_argument("prepared eager input token is outside vocabulary");
        } else {
            for (const auto token : request.input_tokens)
                if (static_cast<std::uint64_t>(token) >= compiled.vocab_size)
                    throw std::invalid_argument("prepared eager row token is outside vocabulary");
        }
        if (request.position >= compiled.max_position_embeddings ||
            request.position >= resources.session_state_layout->context_tokens)
            throw std::out_of_range("prepared eager position exceeds model or allocated context capacity");
        if (!request.state.contains(required_state))
            throw std::invalid_argument("prepared eager required persistent state is unavailable");
        for (std::size_t index = 0; index < write_counts.size(); ++index)
            if (write_counts[index] > std::numeric_limits<std::uint64_t>::max() - request.state.generations[index])
                throw std::overflow_error("prepared eager persistent-state generation would overflow");
    }

    void validate_dynamic(const ExecutionRuntimeResources& resources, const EagerExecutionRequest& request) const {
        // The identity builder retains dynamic model/scratch/state/PLE checks,
        // including every weight allocation generation. Relocation needs a new
        // prepared object even if logical compatibility happens to be unchanged.
        const auto current = build_runtime_execution_identities(compiled.execution_identity, compiled.arena, resources);
        if (current.compatibility_identity != identities.compatibility_identity || current.binding_identity != identities.binding_identity ||
            !same_layout(state_layout, *resources.session_state_layout))
            throw std::invalid_argument("prepared eager runtime bindings or state layout changed");
        tensor_views->validate_source_metadata(*resources.model_manifest, *resources.model_weights,
            *resources.tensor_access);
        validate_request(resources, request);
    }

    PreparedEagerExecutionResult run(const ExecutionRuntimeResources& resources,
        const EagerExecutionRequest& request, EagerExecutionDispatcher& dispatcher) const {
        validate_dynamic(resources, request);
        PreparedEagerExecutionResult result;
        result.state = request.state;
        result.completion_order = order;
        for (const auto& step : compiled.execution.steps) {
            const auto& binding = compiled.bindings.bindings[step.id];
            const auto& io = compiled.io.steps[step.id];
            ExecutionDispatchContext context;
            context.execution_identity = compiled.execution_identity;
            context.step_id = step.id;
            context.op = step.op;
            context.layer = step.layer;
            context.reads = io.reads;
            context.writes = io.writes;
            context.temporaries = io.temporaries;
            context.arena = &compiled.arena;
            context.memory = resources.scratch;
            context.session_state_layout = resources.session_state_layout;
            context.session_state_memory = resources.session_state_memory;
            context.resources = &resources;
            context.prepared_tensor_views = tensor_views.get();
            context.reads_state = step.reads_state;
            context.writes_state = step.writes_state;
            context.logical_rows = request.logical_rows;
            context.request_concurrency = request.request_concurrency;
            context.input_token_id = request.input_token_id;
            context.input_tokens = request.input_tokens;
            context.position = request.position;
            if (binding.host_or_runtime_step) {
                dispatcher.dispatch_runtime_step(context);
                ++result.runtime_steps;
            } else {
                context.kernel_id = std::string_view(*binding.kernel_id);
                dispatcher.dispatch_kernel(context);
                ++result.kernel_steps;
            }
            // Local metadata only, exactly after successful dispatch. The caller
            // publishes session state after its completion/error boundary.
            result.state.publish_write(step.writes_state);
            ++result.completed_steps;
        }
        return result;
    }
};

PreparedEagerExecution::PreparedEagerExecution(CompiledSerialExecution compiled,
    const ExecutionRuntimeResources& resources, const EagerExecutionRequest& prototype,
    std::shared_ptr<const void> actual_owner)
    : impl_(std::make_unique<Impl>(std::move(compiled), resources, prototype, std::move(actual_owner))) {}
PreparedEagerExecution::~PreparedEagerExecution() = default;
const CompiledSerialExecution& PreparedEagerExecution::compiled() const noexcept { return impl_->compiled; }
std::uint64_t PreparedEagerExecution::compatibility_identity() const noexcept {
    return impl_->identities.compatibility_identity;
}
std::optional<std::uint32_t> PreparedEagerExecution::ple_step_id() const noexcept { return impl_->ple_step; }
void PreparedEagerExecution::validate_dynamic(const ExecutionRuntimeResources& resources,
    const EagerExecutionRequest& request) const { impl_->validate_dynamic(resources, request); }
PreparedEagerExecutionResult PreparedEagerExecution::run(const ExecutionRuntimeResources& resources,
    const EagerExecutionRequest& request, EagerExecutionDispatcher& dispatcher) const {
    return impl_->run(resources, request, dispatcher);
}
} // namespace strix
