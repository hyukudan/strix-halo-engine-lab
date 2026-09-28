#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>
#include <string>

namespace strix {

enum class RollbackPolicy {
    recompute,
    full_gdn_snapshots,
};

struct MemoryPlanRequest {
    std::uint64_t context_tokens = 32'768;
    std::uint64_t sessions = 1;
    std::uint64_t draft_tokens = 3;
    std::uint64_t resident_model_bytes = 0;
    std::uint64_t bounded_cache_bytes = 0;
    bool include_mtp_kv = true;
    bool include_provisional_indexer = true;
    RollbackPolicy rollback = RollbackPolicy::recompute;
};

struct MemoryBreakdown {
    std::uint64_t base_kv = 0;
    std::uint64_t mtp_kv = 0;
    std::uint64_t provisional_indexer = 0;
    std::uint64_t gdn_state = 0;
    std::uint64_t rollback = 0;
    std::uint64_t resident_model = 0;
    std::uint64_t bounded_cache = 0;
    std::uint64_t total = 0;
};

struct HostBudget {
    std::uint64_t mem_total_bytes = 0;
    std::uint64_t baseline_without_engine_bytes = 0;
    std::uint64_t recovery_reserve_bytes = 0;
};

struct AdmissionDecision {
    bool admitted = false;
    std::uint64_t engine_limit = 0;
    std::uint64_t required = 0;
    std::uint64_t headroom = 0;
    std::string reason;
};

[[nodiscard]] std::uint64_t base_kv_bytes_per_token(const ModelContract& model);
[[nodiscard]] std::uint64_t mtp_kv_bytes_per_token(const ModelContract& model);
[[nodiscard]] std::uint64_t provisional_indexer_bytes_per_token(const ModelContract& model);
[[nodiscard]] std::uint64_t gdn_state_bytes_per_session(const ModelContract& model);
[[nodiscard]] MemoryBreakdown build_memory_plan(
    const ModelContract& model,
    const MemoryPlanRequest& request);
[[nodiscard]] AdmissionDecision decide_admission(
    const HostBudget& host,
    const MemoryBreakdown& plan);

[[nodiscard]] constexpr std::uint64_t gib(std::uint64_t value) {
    return value * 1024ULL * 1024ULL * 1024ULL;
}

}  // namespace strix
