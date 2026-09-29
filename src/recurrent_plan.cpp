#include "strix/recurrent_plan.hpp"

#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "recurrent-plan addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "recurrent-plan multiplication overflow");
    }
    return left * right;
}

std::uint64_t product(
    std::initializer_list<std::uint64_t> values) {
    std::uint64_t result = 1;
    for (const auto value : values) {
        result = checked_mul(result, value);
    }
    return result;
}

}  // namespace

RecurrentWindowPlan build_recurrent_window_plan(
    const ModelContract& model,
    const RecurrentWindowRequest& request) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.logical_tokens == 0 ||
        request.sessions == 0) {
        throw std::invalid_argument(
            "recurrent window tokens and sessions must be non-zero");
    }

    RecurrentWindowPlan plan;
    plan.logical_tokens = request.logical_tokens;
    plan.sessions = request.sessions;
    plan.key_repeat_factor =
        static_cast<std::uint32_t>(
            model.gdn_value_heads /
            model.gdn_key_heads);

    plan.matrix_state_bytes_per_layer =
        product({
            model.gdn_value_heads,
            model.gdn_key_head_dim,
            model.gdn_value_head_dim,
            model.bytes_fp32,
        });

    const auto key_width =
        product({
            model.gdn_key_heads,
            model.gdn_key_head_dim,
        });
    const auto value_width =
        product({
            model.gdn_value_heads,
            model.gdn_value_head_dim,
        });
    const auto conv_width =
        checked_add(
            checked_mul(key_width, 2),
            value_width);

    plan.conv_state_bytes_per_layer =
        request.include_conv_state
            ? product({
                  conv_width,
                  model.gdn_conv_kernel,
                  model.bytes_bf16,
              })
            : 0;

    plan.persistent_state_bytes_per_layer =
        checked_add(
            plan.matrix_state_bytes_per_layer,
            plan.conv_state_bytes_per_layer);

    plan.persistent_state_bytes_all_layers =
        product({
            plan.persistent_state_bytes_per_layer,
            model.gdn_layers,
            request.sessions,
        });

    plan.serial_state_read_bytes =
        checked_mul(
            plan.persistent_state_bytes_all_layers,
            request.logical_tokens);
    plan.serial_state_write_bytes =
        plan.serial_state_read_bytes;

    plan.retained_state_read_bytes =
        plan.persistent_state_bytes_all_layers;
    plan.retained_state_write_bytes =
        plan.persistent_state_bytes_all_layers;

    const auto serial_roundtrip =
        checked_add(
            plan.serial_state_read_bytes,
            plan.serial_state_write_bytes);
    const auto retained_roundtrip =
        checked_add(
            plan.retained_state_read_bytes,
            plan.retained_state_write_bytes);

    plan.state_roundtrip_bytes_saved =
        serial_roundtrip >= retained_roundtrip
            ? serial_roundtrip - retained_roundtrip
            : 0;

    plan.full_candidate_snapshot_bytes =
        checked_mul(
            plan.persistent_state_bytes_all_layers,
            request.logical_tokens);
    plan.pre_window_snapshot_bytes =
        plan.persistent_state_bytes_all_layers;
    plan.snapshot_bytes_saved =
        plan.full_candidate_snapshot_bytes >=
                plan.pre_window_snapshot_bytes
            ? plan.full_candidate_snapshot_bytes -
                  plan.pre_window_snapshot_bytes
            : 0;

    return plan;
}

}  // namespace strix


namespace strix {

RecurrentRollbackPlan build_recurrent_rollback_plan(
    const ModelContract& model,
    const RecurrentRollbackRequest& request) {
    if (const auto error = model.validate(); !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.logical_tokens == 0 || request.sessions == 0) {
        throw std::invalid_argument(
            "recurrent rollback tokens and sessions must be non-zero");
    }

    RecurrentWindowRequest window;
    window.logical_tokens = request.logical_tokens;
    window.sessions = request.sessions;
    window.include_conv_state = request.include_conv_state;
    const auto base = build_recurrent_window_plan(model, window);

    RecurrentRollbackPlan plan;
    plan.strategy = request.strategy;
    plan.logical_tokens = request.logical_tokens;
    plan.sessions = request.sessions;
    plan.persistent_state_bytes = base.persistent_state_bytes_all_layers;

    switch (request.strategy) {
    case RecurrentRollbackStrategy::full_candidate_snapshots:
        plan.checkpoint_interval = 1;
        plan.checkpoint_count = request.logical_tokens;
        plan.maximum_replay_tokens = 0;
        plan.expected_replay_tokens_uniform_rejection = 0.0;
        plan.retained_checkpoint_bytes =
            checked_mul(plan.persistent_state_bytes, request.logical_tokens);
        plan.worst_case_replay_state_io_bytes = 0;
        break;

    case RecurrentRollbackStrategy::pre_window_replay:
        plan.checkpoint_interval = request.logical_tokens;
        plan.checkpoint_count = 1;
        plan.maximum_replay_tokens =
            request.logical_tokens > 0 ? request.logical_tokens - 1U : 0U;
        plan.expected_replay_tokens_uniform_rejection =
            static_cast<double>(plan.maximum_replay_tokens) / 2.0;
        plan.retained_checkpoint_bytes = plan.persistent_state_bytes;
        plan.worst_case_replay_state_io_bytes =
            checked_mul(
                checked_mul(plan.persistent_state_bytes,
                            plan.maximum_replay_tokens),
                2ULL);
        break;

    case RecurrentRollbackStrategy::sparse_checkpoints: {
        if (request.checkpoint_interval == 0 ||
            request.checkpoint_interval > request.logical_tokens) {
            throw std::invalid_argument(
                "sparse checkpoint interval must be within the logical window");
        }
        plan.checkpoint_interval = request.checkpoint_interval;
        plan.checkpoint_count =
            static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(request.logical_tokens) +
                 request.checkpoint_interval - 1ULL) /
                request.checkpoint_interval);
        plan.maximum_replay_tokens =
            request.checkpoint_interval > 0
                ? request.checkpoint_interval - 1U
                : 0U;
        plan.expected_replay_tokens_uniform_rejection =
            static_cast<double>(plan.maximum_replay_tokens) / 2.0;
        plan.retained_checkpoint_bytes =
            checked_mul(plan.persistent_state_bytes, plan.checkpoint_count);
        plan.worst_case_replay_state_io_bytes =
            checked_mul(
                checked_mul(plan.persistent_state_bytes,
                            plan.maximum_replay_tokens),
                2ULL);
        break;
    }
    }

    return plan;
}

}  // namespace strix
