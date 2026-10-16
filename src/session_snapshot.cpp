#include "strix/session_snapshot.hpp"
#include "strix/indexed_attention_state.hpp"

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

bool flag_valid(
    const PersistentStateGenerations& state,
    PersistentState flag) noexcept {
    return state.contains(flag);
}

std::uint64_t build_identity(
    const SerialSessionSnapshotPlan& plan) noexcept {
    std::uint64_t hash = fnv_offset;
    // Versioned whole-component format: distinguish it from the former
    // contiguous-prefix format and from any future compact per-layer layout.
    hash_u64(hash, 0x53534e415046554cULL);
    hash_u64(hash, 2);
    if (plan.policy != SessionSnapshotPolicy::full_components) {
        hash_u64(hash, 0x4f43434b565631ULL);
        hash_u64(hash, static_cast<std::uint32_t>(plan.policy));
    }
    hash_u64(
        hash,
        plan.compiled_execution_identity);
    hash_u64(
        hash,
        plan.runtime_compatibility_identity);
    hash_u64(
        hash,
        plan.state_layout_identity);
    hash_u64(hash, plan.next_position);
    hash_u64(hash, plan.committed_tokens);
    hash_u64(hash, plan.alignment);
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
    if (plan.recurrent_storage != GdnRecurrentStorage::key_value) {
        hash_u64(hash, 0x47444e53544f5231ULL);
        hash_u64(hash, static_cast<std::uint32_t>(plan.recurrent_storage));
    }
    return hash;
}

}  // namespace

void validate_session_snapshot_policy(SessionSnapshotPolicy policy) {
    if (policy != SessionSnapshotPolicy::full_components &&
        policy != SessionSnapshotPolicy::occupied_kv_v1)
        throw std::invalid_argument("unknown session snapshot policy");
}
const char* session_snapshot_policy_name(SessionSnapshotPolicy policy) {
    validate_session_snapshot_policy(policy);
    return policy == SessionSnapshotPolicy::full_components ? "full-components" : "occupied-kv-v1";
}

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
    validate_serial_session_state_layout_identity(model, layout);
    if (request.compiled_execution_identity == 0) {
        throw std::invalid_argument(
            "snapshot requires compiled execution identity");
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
    validate_session_snapshot_policy(request.policy);
    // Compact interpretation is admitted only for the canonical serial layout.
    // A caller-authored metadata hash alone does not prove per-layer strides.
    if (request.policy == SessionSnapshotPolicy::occupied_kv_v1) {
        const auto canonical = build_serial_session_state_layout(model,
            {layout.context_tokens, layout.device_alignment, layout.host_alignment,
             find_session_state_allocation(layout, SessionStateComponent::sparse_index) != nullptr,
             layout.recurrent_storage});
        if (model.bytes_bf16 != 2U || canonical.layout_identity != layout.layout_identity)
            throw std::invalid_argument("occupied KV snapshot requires the canonical BF16 session layout");
    }

    SerialSessionSnapshotPlan plan;
    plan.policy = request.policy;
    plan.recurrent_storage = layout.recurrent_storage;
    plan.compiled_execution_identity =
        request.compiled_execution_identity;
    plan.runtime_compatibility_identity =
        request.runtime_compatibility_identity;
    plan.state_layout_identity =
        layout.layout_identity;
    plan.next_position =
        request.next_position;
    plan.committed_tokens =
        request.committed_tokens;
    plan.alignment =
        request.alignment;
    plan.state = request.state;

    const auto append = [&](const SessionStateAllocation& allocation,
                            std::uint64_t source, std::uint64_t bytes) {
        if (!bytes) return;
        if (source < allocation.offset || source - allocation.offset > allocation.bytes ||
            bytes > allocation.bytes - (source - allocation.offset))
            throw std::invalid_argument("snapshot range exceeds its state allocation");
        auto& total = allocation.storage == SessionStateStorage::device ? plan.device_bytes : plan.host_bytes;
        const auto destination = align_up(total, request.alignment);
        total = checked_add(destination, bytes);
        plan.ranges.push_back({allocation.component, allocation.storage, source,
            destination, bytes, allocation.provisional});
    };

    for (const auto& allocation :
         layout.allocations) {
        if (!flag_valid(
                request.state,
                allocation.state_flag)) {
            continue;
        }

        if (request.policy == SessionSnapshotPolicy::occupied_kv_v1 &&
            allocation.component == SessionStateComponent::attention_kv) {
            const auto row = build_indexed_attention_state_layout(model).kv_row_bytes;
            if (row && layout.context_tokens > std::numeric_limits<std::uint64_t>::max() / row)
                throw std::overflow_error("occupied KV layer stride overflow");
            const auto stride = layout.context_tokens * row;
            if (stride && model.full_attention_layers > std::numeric_limits<std::uint64_t>::max() / stride)
                throw std::overflow_error("occupied KV component extent overflow");
            if (allocation.bytes != stride * model.full_attention_layers)
                throw std::invalid_argument("occupied KV component extent is not canonical");
            for (std::uint64_t layer = 0; layer < model.full_attention_layers; ++layer)
                append(allocation, checked_add(allocation.offset, layer * stride), request.next_position * row);
            continue;
        }

        // KV and sparse state use per-layer context-capacity strides. A
        // component-wide prefix length would omit later layers. Preserve the
        // whole allocation, including inactive capacity, in the unchanged
        // full format. The explicit per-layer candidate is handled above.
        const auto bytes = allocation.bytes;
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

void validate_serial_session_snapshot_plan(const ModelContract& model,
    const SerialSessionStateLayout& layout, const SerialSessionSnapshotPlan& snapshot) {
    SerialSessionSnapshotRequest request;
    request.compiled_execution_identity = snapshot.compiled_execution_identity;
    request.runtime_compatibility_identity = snapshot.runtime_compatibility_identity;
    request.next_position = snapshot.next_position;
    request.committed_tokens = snapshot.committed_tokens;
    request.state = snapshot.state;
    request.alignment = snapshot.alignment;
    request.policy = snapshot.policy;
    const auto expected = build_serial_session_snapshot_plan(model, layout, request);
    if (snapshot.state_layout_identity != expected.state_layout_identity ||
        snapshot.recurrent_storage != expected.recurrent_storage ||
        snapshot.plan_identity != expected.plan_identity || snapshot.device_bytes != expected.device_bytes ||
        snapshot.host_bytes != expected.host_bytes || snapshot.ranges.size() != expected.ranges.size())
        throw std::invalid_argument("snapshot structure or identity is invalid");
    for (std::size_t i = 0; i < expected.ranges.size(); ++i) {
        const auto& a = snapshot.ranges[i]; const auto& b = expected.ranges[i];
        if (a.component != b.component || a.storage != b.storage || a.source_offset != b.source_offset ||
            a.destination_offset != b.destination_offset || a.bytes != b.bytes || a.provisional != b.provisional)
            throw std::invalid_argument("snapshot range is not canonical");
    }
}

void validate_serial_session_snapshot_identity(const SerialSessionSnapshotPlan& snapshot) {
    validate_session_snapshot_policy(snapshot.policy);
    if (!snapshot.plan_identity || build_identity(snapshot) != snapshot.plan_identity)
        throw std::invalid_argument("snapshot metadata identity mismatch");
}

}  // namespace strix
