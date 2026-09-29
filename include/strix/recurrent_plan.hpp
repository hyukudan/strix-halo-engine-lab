#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

enum class RecurrentRollbackStrategy {
    full_candidate_snapshots,
    pre_window_replay,
    sparse_checkpoints,
};

struct RecurrentRollbackRequest {
    std::uint32_t logical_tokens = 1;
    std::uint32_t sessions = 1;
    RecurrentRollbackStrategy strategy =
        RecurrentRollbackStrategy::pre_window_replay;
    std::uint32_t checkpoint_interval = 0;
    bool include_conv_state = true;
};

struct RecurrentRollbackPlan {
    RecurrentRollbackStrategy strategy =
        RecurrentRollbackStrategy::pre_window_replay;
    std::uint32_t logical_tokens = 0;
    std::uint32_t sessions = 0;
    std::uint32_t checkpoint_interval = 0;
    std::uint32_t checkpoint_count = 0;
    std::uint32_t maximum_replay_tokens = 0;
    double expected_replay_tokens_uniform_rejection = 0.0;
    std::uint64_t persistent_state_bytes = 0;
    std::uint64_t retained_checkpoint_bytes = 0;
    std::uint64_t worst_case_replay_state_io_bytes = 0;
};

struct RecurrentWindowRequest {
    std::uint32_t logical_tokens = 1;
    std::uint32_t sessions = 1;
    bool include_conv_state = true;
};

struct RecurrentWindowPlan {
    std::uint32_t logical_tokens = 0;
    std::uint32_t sessions = 0;
    std::uint32_t key_repeat_factor = 0;

    std::uint64_t matrix_state_bytes_per_layer = 0;
    std::uint64_t conv_state_bytes_per_layer = 0;
    std::uint64_t persistent_state_bytes_per_layer = 0;
    std::uint64_t persistent_state_bytes_all_layers = 0;

    std::uint64_t serial_state_read_bytes = 0;
    std::uint64_t serial_state_write_bytes = 0;
    std::uint64_t retained_state_read_bytes = 0;
    std::uint64_t retained_state_write_bytes = 0;
    std::uint64_t state_roundtrip_bytes_saved = 0;

    std::uint64_t full_candidate_snapshot_bytes = 0;
    std::uint64_t pre_window_snapshot_bytes = 0;
    std::uint64_t snapshot_bytes_saved = 0;
};

[[nodiscard]] RecurrentWindowPlan build_recurrent_window_plan(
    const ModelContract& model,
    const RecurrentWindowRequest& request);

[[nodiscard]] RecurrentRollbackPlan build_recurrent_rollback_plan(
    const ModelContract& model,
    const RecurrentRollbackRequest& request);

}  // namespace strix
