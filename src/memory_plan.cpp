#include "strix/memory_plan.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("memory-plan addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("memory-plan multiplication overflow");
    }
    return left * right;
}

std::uint64_t product(std::initializer_list<std::uint64_t> values) {
    std::uint64_t result = 1;
    for (const auto value : values) {
        result = checked_mul(result, value);
    }
    return result;
}

void add_to_total(MemoryBreakdown& plan, std::uint64_t value) {
    plan.total = checked_add(plan.total, value);
}

}  // namespace

std::uint64_t base_kv_bytes_per_token(const ModelContract& model) {
    return product({model.full_attention_layers, model.kv_heads, model.head_dim,
                    2, model.bytes_bf16});
}

std::uint64_t mtp_kv_bytes_per_token(const ModelContract& model) {
    return product({model.mtp_full_attention_layers, model.kv_heads,
                    model.head_dim, 2, model.bytes_bf16});
}

std::uint64_t provisional_indexer_bytes_per_token(const ModelContract& model) {
    const auto uncompressed = product({model.full_attention_layers,
                                       model.indexer_kv_heads,
                                       model.indexer_head_dim,
                                       model.bytes_bf16});
    if (uncompressed % model.indexer_compress_ratio != 0) {
        throw std::invalid_argument("indexer estimate does not divide exactly");
    }
    return uncompressed / model.indexer_compress_ratio;
}

std::uint64_t gdn_matrix_state_bytes_per_session(
    const ModelContract& model) {
    return product({model.gdn_layers, model.gdn_value_heads,
                    model.gdn_key_head_dim, model.gdn_value_head_dim,
                    model.bytes_fp32});
}

std::uint64_t gdn_conv_state_bytes_per_session(
    const ModelContract& model) {
    const auto key_width =
        product({model.gdn_key_heads, model.gdn_key_head_dim});
    const auto value_width =
        product({model.gdn_value_heads, model.gdn_value_head_dim});
    const auto conv_width =
        checked_add(checked_mul(key_width, 2), value_width);
    return product({model.gdn_layers, conv_width,
                    model.gdn_conv_kernel, model.bytes_bf16});
}

std::uint64_t gdn_state_bytes_per_session(
    const ModelContract& model) {
    return checked_add(
        gdn_matrix_state_bytes_per_session(model),
        gdn_conv_state_bytes_per_session(model));
}

MemoryBreakdown build_memory_plan(const ModelContract& model,
                                  const MemoryPlanRequest& request) {
    if (const auto error = model.validate(); !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.sessions == 0) {
        throw std::invalid_argument("sessions must be non-zero");
    }

    MemoryBreakdown plan;
    const auto positions = checked_mul(request.context_tokens, request.sessions);
    plan.base_kv = checked_mul(base_kv_bytes_per_token(model), positions);
    plan.mtp_kv = request.include_mtp_kv
                      ? checked_mul(mtp_kv_bytes_per_token(model), positions)
                      : 0;
    plan.provisional_indexer = request.include_provisional_indexer
                                   ? checked_mul(
                                         provisional_indexer_bytes_per_token(model),
                                         positions)
                                   : 0;
    plan.gdn_matrix_state = checked_mul(
        gdn_matrix_state_bytes_per_session(model), request.sessions);
    plan.gdn_conv_state = checked_mul(
        gdn_conv_state_bytes_per_session(model), request.sessions);
    plan.gdn_state = checked_add(
        plan.gdn_matrix_state, plan.gdn_conv_state);
    plan.rollback = request.rollback == RollbackPolicy::full_gdn_snapshots
                        ? product({gdn_state_bytes_per_session(model),
                                   request.sessions, request.draft_tokens})
                        : 0;
    plan.resident_model = request.resident_model_bytes;
    plan.bounded_cache = request.bounded_cache_bytes;

    add_to_total(plan, plan.base_kv);
    add_to_total(plan, plan.mtp_kv);
    add_to_total(plan, plan.provisional_indexer);
    add_to_total(plan, plan.gdn_state);
    add_to_total(plan, plan.rollback);
    add_to_total(plan, plan.resident_model);
    add_to_total(plan, plan.bounded_cache);
    return plan;
}

AdmissionDecision decide_admission(const HostBudget& host,
                                   const MemoryBreakdown& plan) {
    AdmissionDecision decision;
    decision.required = plan.total;

    const auto protected_bytes = checked_add(host.baseline_without_engine_bytes,
                                             host.recovery_reserve_bytes);
    if (protected_bytes >= host.mem_total_bytes) {
        decision.reason = "host baseline plus reserve leaves no engine budget";
        return decision;
    }
    decision.engine_limit = host.mem_total_bytes - protected_bytes;
    if (plan.total > decision.engine_limit) {
        decision.reason = "plan exceeds the engine admission limit";
        return decision;
    }
    decision.admitted = true;
    decision.headroom = decision.engine_limit - plan.total;
    decision.reason = "plan fits the explicit engine budget";
    return decision;
}

}  // namespace strix
