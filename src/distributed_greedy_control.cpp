#include "strix/distributed_greedy_control.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
void require_greedy(bool value, const char* reason) { if (!value) throw std::invalid_argument(reason); }
void hash_word(std::uint64_t& h, std::uint64_t value) noexcept {
    for (unsigned bit = 0; bit < 64U; bit += 8U) { h ^= (value >> bit) & 255U; h *= 1099511628211ULL; }
}
std::uint64_t required_inputs(const DistributedGreedyRequestIdentity& r) noexcept {
    return r.prompt_tokens + (r.max_new_tokens
        ? std::min(r.max_new_tokens - 1U, r.context_capacity - r.prompt_tokens) : 0U);
}
bool no_selected(const DistributedGreedyMessage& m) noexcept {
    return !m.selected_present && !m.selected_token && !m.selected_logit_bits;
}
bool no_pending(const DistributedGreedyMessage& m) noexcept { return !m.pending_present && !m.pending_token; }
bool no_finish(const DistributedGreedyMessage& m) noexcept {
    return !m.emitted_tokens && no_pending(m) && m.finish_reason == DistributedGreedyFinishReason::none;
}
bool same_finish(const DistributedGreedyMessage& a, const DistributedGreedyMessage& b) noexcept {
    return a.generation == b.generation && a.binding_identity == b.binding_identity &&
        a.decision_identity == b.decision_identity && a.position == b.position &&
        a.committed_tokens == b.committed_tokens && a.history_identity == b.history_identity &&
        a.emitted_tokens == b.emitted_tokens && a.pending_present == b.pending_present &&
        a.pending_token == b.pending_token && a.finish_reason == b.finish_reason;
}
}
void validate_distributed_greedy_request_identity(const DistributedGreedyRequestIdentity& r) {
    require_greedy(r.contract_version == 1U && r.model_content_identity.valid() && r.prompt_sha256.valid() &&
        r.serial_execution_identity && r.kernel_plan_identity && r.placement_plan_identity && r.request_id &&
        r.connection_epoch && r.initial_binding_identity && r.initial_generation && r.boundary_dtype_id &&
        r.vocab_size && r.vocab_size <= UINT32_MAX && r.eos_token_id < r.vocab_size &&
        r.prompt_tokens && r.context_capacity && r.prompt_tokens <= r.context_capacity &&
        r.max_target_inputs && r.max_target_inputs <= r.context_capacity,
        "distributed greedy request lacks closed immutable identities or bounded target geometry");
    // Same worst-case consumed-input bound as the existing serial token cursor.
    // Zero emissions still consumes the complete known prefix. The final
    // finite selection is checked then discarded by the existing token cursor.
    const auto required = required_inputs(r);
    require_greedy(required <= r.max_target_inputs && r.max_target_inputs - 1U <= UINT64_MAX - r.initial_generation,
        "distributed greedy target/generation budget cannot cover the serial request");
}
std::uint64_t distributed_greedy_request_identity(const DistributedGreedyRequestIdentity& r) noexcept {
    std::uint64_t h = 14695981039346656037ULL; hash_word(h, 0x4447524551563031ULL); hash_word(h, r.contract_version);
    for (auto byte : r.model_content_identity.bytes) hash_word(h, byte);
    for (auto byte : r.prompt_sha256.bytes) hash_word(h, byte);
    for (const auto word : {r.serial_execution_identity, r.kernel_plan_identity, r.placement_plan_identity,
        r.request_id, r.connection_epoch, r.initial_binding_identity, r.initial_generation, r.initial_history_identity,
        r.prompt_tokens, r.vocab_size, r.context_capacity, r.max_new_tokens, r.max_target_inputs}) hash_word(h, word);
    hash_word(h, r.eos_token_id); hash_word(h, r.boundary_dtype_id); return h;
}
DistributedGreedyFinishReason distributed_greedy_finish_reason(SerialRequestStopReason reason) {
    switch (reason) {
    case SerialRequestStopReason::eos: return DistributedGreedyFinishReason::eos;
    case SerialRequestStopReason::max_new_tokens: return DistributedGreedyFinishReason::max_new_tokens;
    case SerialRequestStopReason::context_capacity: return DistributedGreedyFinishReason::context_capacity;
    }
    throw std::invalid_argument("unknown serial request stop reason");
}
void validate_distributed_greedy_message(const DistributedGreedyMessage& m) {
    validate_distributed_greedy_request_identity(m.request);
    require_greedy(m.payload_version == 1U && m.request_identity &&
        m.request_identity == distributed_greedy_request_identity(m.request) &&
        m.sequence && m.sequence != UINT64_MAX && m.sender_node <= 1U && m.receiver_node == 1U - m.sender_node,
        "distributed greedy message header, identity or sequence is invalid");
    if (m.kind == DistributedGreedyMessageKind::failure) {
        require_greedy(m.failure >= DistributedGreedyFailure::protocol && m.failure <= DistributedGreedyFailure::unavailable &&
            !m.generation && !m.binding_identity && !m.decision_identity && !m.position && !m.committed_tokens &&
            !m.history_identity && !m.previous_binding_identity && !m.previous_decision_identity && !m.input_token &&
            m.output_intent == DistributedGreedyOutputIntent::none && no_selected(m) && no_finish(m),
            "distributed greedy failure must contain no successful target or presentation payload");
        return;
    }
    require_greedy(m.failure == DistributedGreedyFailure::none && m.generation && m.binding_identity &&
        m.position == m.committed_tokens && m.position <= m.request.context_capacity,
        "distributed greedy target/frontier is incomplete or inconsistent");
    switch (m.kind) {
    case DistributedGreedyMessageKind::input_command:
        require_greedy(m.sender_node == 0U && !m.decision_identity &&
            m.position < m.request.context_capacity && m.input_token < m.request.vocab_size &&
            (m.output_intent == DistributedGreedyOutputIntent::state_only ||
             m.output_intent == DistributedGreedyOutputIntent::finite_firstmax) && no_selected(m) && no_finish(m),
            "distributed greedy input command contains invalid intent or result fields");
        break;
    case DistributedGreedyMessageKind::target_result:
        require_greedy(m.sender_node == 1U && m.decision_identity && m.position && m.history_identity &&
            !m.previous_binding_identity && !m.previous_decision_identity && !m.input_token && no_finish(m) &&
            ((m.output_intent == DistributedGreedyOutputIntent::state_only && no_selected(m)) ||
             (m.output_intent == DistributedGreedyOutputIntent::finite_firstmax && m.selected_present &&
              m.selected_token < m.request.vocab_size && (m.selected_logit_bits & 0x7f800000U) != 0x7f800000U)),
            "distributed greedy target result is not a bound finite scalar or state-only completion");
        break;
    case DistributedGreedyMessageKind::finish:
    case DistributedGreedyMessageKind::finished:
        require_greedy(m.sender_node == (m.kind == DistributedGreedyMessageKind::finish ? 0U : 1U) &&
            !m.previous_binding_identity && !m.previous_decision_identity && !m.input_token &&
            m.output_intent == DistributedGreedyOutputIntent::none && no_selected(m) &&
            m.finish_reason >= DistributedGreedyFinishReason::eos && m.finish_reason <= DistributedGreedyFinishReason::cancelled &&
            m.emitted_tokens <= m.request.max_new_tokens && (!m.pending_present || m.emitted_tokens != 0U) &&
            (m.finish_reason == DistributedGreedyFinishReason::cancelled || m.pending_present == (m.emitted_tokens != 0U)) &&
            (m.pending_present ? m.pending_token < m.request.vocab_size : m.pending_token == 0U),
            "distributed greedy finish contains invalid presentation metadata");
        if (m.finish_reason == DistributedGreedyFinishReason::eos)
            require_greedy(m.pending_present && m.pending_token == m.request.eos_token_id, "EOS finish has no emitted EOS frontier");
        if (m.finish_reason == DistributedGreedyFinishReason::context_capacity)
            require_greedy(m.position == m.request.context_capacity, "context finish is not at the consumed capacity frontier");
        if (m.finish_reason == DistributedGreedyFinishReason::max_new_tokens)
            require_greedy(m.emitted_tokens == m.request.max_new_tokens, "budget finish does not match emitted token count");
        break;
    case DistributedGreedyMessageKind::failure: break; // handled above
    default: throw std::invalid_argument("distributed greedy message kind is outside payloadv1");
    }
}
void validate_distributed_greedy_target_commit(const DistributedGreedyMessage& m,
    const DistributedTokenBinding& b, const DistributedTokenCommit& c) {
    validate_distributed_greedy_message(m);
    require_greedy(m.kind == DistributedGreedyMessageKind::target_result &&
        m.request.model_content_identity == b.model_content_identity &&
        m.request.serial_execution_identity == b.serial_execution_identity && m.request.kernel_plan_identity == b.kernel_plan_identity &&
        m.request.placement_plan_identity == b.placement_plan_identity && m.request.request_id == b.request_id &&
        m.request.connection_epoch == b.connection_epoch && m.request.boundary_dtype_id == b.boundary_dtype_id &&
        m.generation == b.generation && m.binding_identity == distributed_token_binding_identity(b) &&
        m.binding_identity == c.binding_identity && m.decision_identity == c.decision_identity && c.decision_identity &&
        b.position < UINT64_MAX && b.committed_tokens < UINT64_MAX && m.position == b.position + 1U &&
        m.committed_tokens == b.committed_tokens + 1U && m.position == c.next_position &&
        m.committed_tokens == c.committed_tokens && m.history_identity == c.history_identity,
        "selected reply differs from actual binding and two-ACK committed frontier");
}
DistributedGreedyControlState::DistributedGreedyControlState(DistributedGreedyRequestIdentity request)
    : request_(std::move(request)) { validate_distributed_greedy_request_identity(request_); }
DistributedGreedyMessage DistributedGreedyControlState::make_message(DistributedGreedyMessageKind kind, std::uint32_t sender) const {
    require_greedy(sender <= 1U && !failed_ && !finished_, "greedy transcript cannot build another message");
    DistributedGreedyMessage m; m.kind = kind; m.request = request_;
    m.request_identity = distributed_greedy_request_identity(request_); m.sequence = next_[sender];
    m.sender_node = sender; m.receiver_node = 1U - sender; return m;
}
void DistributedGreedyControlState::validate(const DistributedGreedyMessage& m) const {
    require_greedy(!failed_ && !finished_, "distributed greedy transcript is failed or finished");
    validate_distributed_greedy_message(m);
    require_greedy(m.request == request_ && m.sequence == next_[m.sender_node],
        "distributed greedy request or directional sequence is foreign/stale");
    if (m.kind == DistributedGreedyMessageKind::failure) return;
    if (m.kind == DistributedGreedyMessageKind::input_command) {
        require_greedy(!pending_ && !finish_ && target_inputs_ < request_.max_target_inputs &&
            target_inputs_ < required_inputs(request_),
            "distributed greedy input exceeds target budget or overlaps pending work");
        const auto expected_intent = m.position < request_.prompt_tokens - 1U
            ? DistributedGreedyOutputIntent::state_only : DistributedGreedyOutputIntent::finite_firstmax;
        require_greedy(m.output_intent == expected_intent,
            "greedy input intent differs from its known-prefix/selection frontier");
        if (!last_) {
            require_greedy(m.generation == request_.initial_generation && m.binding_identity == request_.initial_binding_identity &&
                !m.position && !m.committed_tokens && m.history_identity == request_.initial_history_identity &&
                !m.previous_binding_identity && !m.previous_decision_identity,
                "first greedy input differs from its negotiated initial frontier");
        } else {
            require_greedy(last_->generation < UINT64_MAX && m.generation == last_->generation + 1U &&
                m.position == last_->position && m.committed_tokens == last_->committed_tokens &&
                m.history_identity == last_->history_identity && m.previous_binding_identity == last_->binding_identity &&
                m.previous_decision_identity == last_->decision_identity,
                "next greedy input is not chained to the accepted committed result");
            if (m.position >= request_.prompt_tokens)
                require_greedy(last_->selected_present && last_->selected_token != request_.eos_token_id &&
                    m.input_token == last_->selected_token,
                    "generation input does not consume the previously selected pending token");
        }
    } else if (m.kind == DistributedGreedyMessageKind::target_result) {
        require_greedy(pending_ && !finish_ && m.generation == pending_->generation && m.binding_identity == pending_->binding_identity &&
            pending_->position < UINT64_MAX && m.position == pending_->position + 1U &&
            pending_->committed_tokens < UINT64_MAX && m.committed_tokens == pending_->committed_tokens + 1U &&
            m.output_intent == pending_->output_intent,
            "greedy result has no exact outstanding target input");
    } else if (m.kind == DistributedGreedyMessageKind::finish) {
        require_greedy(!pending_ && !finish_, "greedy finish overlaps an uncompleted target or another finish");
        if (last_) {
            require_greedy(m.generation == last_->generation && m.binding_identity == last_->binding_identity &&
                m.decision_identity == last_->decision_identity && m.position == last_->position &&
                m.committed_tokens == last_->committed_tokens && m.history_identity == last_->history_identity &&
                (!m.pending_present || (last_->selected_present && m.pending_token == last_->selected_token)),
                "greedy finish differs from the last confirmed consumed/pending frontier");
            const auto selections = m.position >= request_.prompt_tokens
                ? m.position - request_.prompt_tokens + 1U : 0U;
            if (m.finish_reason == DistributedGreedyFinishReason::cancelled) {
                // Cancellation after target completion may suppress that one
                // selection. Earlier emissions were already consumed as inputs.
                require_greedy(m.emitted_tokens == selections ||
                    (selections && m.emitted_tokens == selections - 1U),
                    "cancelled finish invents or loses earlier emitted tokens");
                require_greedy(m.pending_present == (selections && m.emitted_tokens == selections),
                    "cancelled pending frontier differs from completed presentation");
            } else {
                require_greedy(last_->selected_present && m.position >= request_.prompt_tokens &&
                    m.emitted_tokens == std::min(request_.max_new_tokens, selections),
                    "greedy finish emitted count differs from completed selected targets");
            }
        } else {
            require_greedy(m.generation == request_.initial_generation && m.binding_identity == request_.initial_binding_identity &&
                !m.decision_identity && !m.position && !m.committed_tokens && m.history_identity == request_.initial_history_identity &&
                !m.emitted_tokens && no_pending(m) &&
                m.finish_reason == DistributedGreedyFinishReason::cancelled,
                "zero-work greedy finish manufactured target consumption or an emitted bonus");
        }
    } else if (m.kind == DistributedGreedyMessageKind::finished) {
        require_greedy(finish_ && !pending_ && same_finish(m, *finish_), "greedy finish confirmation differs from the requested frontier");
    }
}
void DistributedGreedyControlState::observe(const DistributedGreedyMessage& m) {
    try {
        validate(m); ++next_[m.sender_node];
        switch (m.kind) {
        case DistributedGreedyMessageKind::input_command: pending_ = m; ++target_inputs_; break;
        case DistributedGreedyMessageKind::target_result: last_ = m; pending_.reset(); break;
        case DistributedGreedyMessageKind::finish: finish_ = m; break;
        case DistributedGreedyMessageKind::finished: finished_ = true; break;
        case DistributedGreedyMessageKind::failure: failed_ = true; break;
        }
    } catch (...) { failed_ = true; throw; }
}
FramedDistributedGreedyWire::FramedDistributedGreedyWire(DistributedControlByteStream& stream,
    DistributedGreedyRequestIdentity request, std::uint32_t node, DistributedControlWireLimits limits)
    : stream_(&stream), state_(std::move(request)), node_(node), limits_(limits) {
    require_greedy(node <= 1U && limits.protocol_version == 2U && limits.maximum_frame_bytes >= 16U &&
        limits.maximum_frame_bytes <= 64U * 1024U && limits.maximum_slices,
        "greedy control requires explicit node and bounded v2 channel");
}
DistributedGreedyMessage FramedDistributedGreedyWire::make_message(DistributedGreedyMessageKind kind) const {
    return state_.make_message(kind, node_);
}
void FramedDistributedGreedyWire::send(const DistributedGreedyMessage& m) {
    try {
        require_greedy(m.sender_node == node_ && m.receiver_node == 1U - node_, "greedy send claims a foreign endpoint");
        state_.validate(m); const auto bytes = encode_distributed_greedy_message(m, limits_);
        stream_->send(bytes); state_.observe(m);
    } catch (...) { state_.communication_failed(); throw; }
}
DistributedGreedyMessage FramedDistributedGreedyWire::receive() {
    try {
        const auto bytes = stream_->receive(); const auto m = decode_distributed_greedy_message(bytes, limits_);
        require_greedy(m.receiver_node == node_ && m.sender_node == 1U - node_, "greedy receive claims a foreign endpoint");
        state_.observe(m);
        if (m.kind == DistributedGreedyMessageKind::failure) throw std::runtime_error("peer terminated distributed greedy control");
        return m;
    } catch (...) { state_.communication_failed(); throw; }
}
} // namespace strix
