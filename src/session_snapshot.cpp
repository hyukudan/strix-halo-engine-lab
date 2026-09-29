#include "strix/session_snapshot.hpp"

#include "strix/memory_plan.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "session snapshot byte overflow");
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
            "session snapshot multiplication overflow");
    }
    return left * right;
}

std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "snapshot alignment must be non-zero");
    }
    const auto remainder = value % alignment;
    return remainder == 0
        ? value
        : checked_add(
              value,
              alignment - remainder);
}

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t bytes_for_component(
    const ModelContract& model,
    const SerialSessionStateLayout& layout,
    const SessionStateAllocation& allocation,
    std::uint64_t next_position) {
    switch (allocation.component) {
    case SessionStateComponent::attention_kv:
        return checked_mul(
            base_kv_bytes_per_token(model),
            next_position);

    case SessionStateComponent::sparse_index:
        return checked_mul(
            provisional_indexer_bytes_per_token(model),
            next_position);

    case SessionStateComponent::recurrent_matrix:
    case SessionStateComponent::gdn_convolution:
    case SessionStateComponent::ple_convolution:
    case SessionStateComponent::ngram_history:
    case SessionStateComponent::position_history:
        return allocation.bytes;
    }

    throw std::logic_error(
        "unknown session-state component");
}

bool flag_valid(
    const PersistentStateGenerations& state,
    PersistentState flag) noexcept {
    return state.contains(flag);
}

std::uint64_t build_identity(
    const SerialSessionSnapshotPlan& plan) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(
        hash,
        plan.runtime_compatibility_identity);
    hash_u64(
        hash,
        plan.state_layout_identity);
    hash_u64(hash, plan.next_position);
    hash_u64(hash, plan.committed_tokens);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            plan.state.valid));
    for (const auto generation :
         plan.state.generations) {
        hash_u64(hash, generation);
    }
    hash_u64(hash, plan.device_bytes);
    hash_u64(hash, plan.host_bytes);

    for (const auto& range : plan.ranges) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                range.component));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                range.storage));
        hash_u64(hash, range.source_offset);
        hash_u64(
            hash,
            range.destination_offset);
        hash_u64(hash, range.bytes);
        hash_u64(
            hash,
            range.provisional ? 1ULL : 0ULL);
    }
    return hash;
}

}  // namespace

SerialSessionSnapshotPlan
build_serial_session_snapshot_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& layout,
    const SerialSessionSnapshotRequest& request) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (layout.layout_identity == 0) {
        throw std::invalid_argument(
            "snapshot requires a valid session-state layout");
    }
    if (request.runtime_compatibility_identity == 0) {
        throw std::invalid_argument(
            "snapshot requires runtime compatibility identity");
    }
    if (request.next_position >
        layout.context_tokens) {
        throw std::out_of_range(
            "snapshot position exceeds allocated session capacity");
    }
    if (request.alignment == 0) {
        throw std::invalid_argument(
            "snapshot alignment must be non-zero");
    }

    SerialSessionSnapshotPlan plan;
    plan.runtime_compatibility_identity =
        request.runtime_compatibility_identity;
    plan.state_layout_identity =
        layout.layout_identity;
    plan.next_position =
        request.next_position;
    plan.committed_tokens =
        request.committed_tokens;
    plan.state = request.state;

    for (const auto& allocation :
         layout.allocations) {
        if (!flag_valid(
                request.state,
                allocation.state_flag)) {
            continue;
        }

        const auto bytes =
            bytes_for_component(
                model,
                layout,
                allocation,
                request.next_position);
        if (bytes == 0) {
            continue;
        }
        if (bytes > allocation.bytes) {
            throw std::invalid_argument(
                "snapshot valid-state bytes exceed session allocation");
        }

        auto& total =
            allocation.storage ==
                    SessionStateStorage::device
                ? plan.device_bytes
                : plan.host_bytes;
        const auto destination =
            align_up(
                total,
                request.alignment);
        total =
            checked_add(
                destination,
                bytes);

        plan.ranges.push_back(
            SessionSnapshotRange{
                allocation.component,
                allocation.storage,
                allocation.offset,
                destination,
                bytes,
                allocation.provisional,
            });
    }

    // Every valid state flag used by serial decode must be represented by at
    // least one concrete component in the layout.
    const auto require_component =
        [&](PersistentState flag,
            SessionStateComponent component,
            const char* message) {
            if (!flag_valid(request.state, flag)) {
                return;
            }
            if (find_session_state_allocation(
                    layout,
                    component) == nullptr) {
                throw std::invalid_argument(message);
            }
        };

    require_component(
        PersistentState::attention_kv,
        SessionStateComponent::attention_kv,
        "snapshot is missing attention KV component");
    require_component(
        PersistentState::recurrent_matrix,
        SessionStateComponent::recurrent_matrix,
        "snapshot is missing recurrent matrix component");
    if (flag_valid(
            request.state,
            PersistentState::convolution_history)) {
        if (find_session_state_allocation(
                layout,
                SessionStateComponent::gdn_convolution) == nullptr ||
            find_session_state_allocation(
                layout,
                SessionStateComponent::ple_convolution) == nullptr) {
            throw std::invalid_argument(
                "snapshot is missing convolution-history component");
        }
    }
    require_component(
        PersistentState::sparse_index,
        SessionStateComponent::sparse_index,
        "snapshot is missing sparse-index component");
    require_component(
        PersistentState::ngram_history,
        SessionStateComponent::ngram_history,
        "snapshot is missing n-gram history component");
    require_component(
        PersistentState::position_history,
        SessionStateComponent::position_history,
        "snapshot is missing position-history component");

    plan.device_bytes =
        align_up(
            plan.device_bytes,
            request.alignment);
    plan.host_bytes =
        align_up(
            plan.host_bytes,
            request.alignment);

    plan.plan_identity =
        build_identity(plan);
    if (plan.plan_identity == 0) {
        throw std::logic_error(
            "snapshot plan identity must be non-zero");
    }
    return plan;
}

}  // namespace strix
