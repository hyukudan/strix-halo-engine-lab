#include "strix/distributed_placement.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "distributed placement byte count overflow");
    }
    return left + right;
}

bool same_use(
    const ExecutionBufferUse& left,
    const ExecutionBufferUse& right) noexcept {
    return left == right;
}

std::uint64_t boundary_identity(
    std::uint64_t plan_seed,
    std::uint32_t source_step,
    std::uint32_t destination_step,
    std::uint32_t source_node,
    std::uint32_t destination_node,
    std::span<const ExecutionBufferUse> payloads,
    std::uint64_t bytes) noexcept {
    std::uint64_t hash = plan_seed;
    hash_u64(hash, source_step);
    hash_u64(hash, destination_step);
    hash_u64(hash, source_node);
    hash_u64(hash, destination_node);
    hash_u64(hash, bytes);
    hash_u64(hash, payloads.size());
    for (const auto& payload : payloads) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                payload.buffer.buffer_class));
        hash_u64(hash, payload.buffer.slot);
        hash_u64(hash, payload.generation);
    }
    return hash;
}

}  // namespace

DistributedPlacementPlan build_distributed_placement_plan(
    const CompiledSerialExecution& compiled,
    const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& tensor_access,
    std::uint32_t local_node) {
    if (compiled.execution_identity == 0 ||
        manifest.layout_identity == 0 ||
        tensor_access.plan_identity == 0 ||
        tensor_access.manifest_layout_identity !=
            manifest.layout_identity ||
        compiled.execution.steps.size() !=
            compiled.io.steps.size()) {
        throw std::invalid_argument(
            "distributed placement requires compatible compiled execution and tensor plan");
    }

    DistributedPlacementPlan plan;
    plan.local_node = local_node;
    plan.compiled_execution_identity =
        compiled.execution_identity;
    plan.manifest_layout_identity =
        manifest.layout_identity;
    plan.tensor_access_identity =
        tensor_access.plan_identity;
    plan.steps.resize(
        compiled.execution.steps.size());

    std::unordered_set<std::uint32_t>
        remote_nodes;

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        const auto& step =
            compiled.execution.steps[index];
        auto& placement =
            plan.steps[index];
        placement.step_id = step.id;
        placement.op = step.op;
        placement.node = local_node;

        std::optional<std::uint32_t>
            declared_node;
        bool conflict = false;

        for (const auto& access :
             tensor_access.accesses) {
            if (access.step_id != step.id) {
                continue;
            }
            for (const auto tensor_index :
                 access.tensor_indices) {
                if (tensor_index >=
                    manifest.tensors.size()) {
                    throw std::out_of_range(
                        "distributed placement tensor index exceeds manifest");
                }
                const auto& tensor =
                    manifest.tensors[
                        tensor_index];
                const auto node =
                    tensor.placement.kind ==
                            PlacementKind::remote
                        ? tensor.placement.node
                        : local_node;
                if (!declared_node.has_value()) {
                    declared_node = node;
                } else if (*declared_node !=
                           node) {
                    conflict = true;
                }
            }
        }

        if (conflict) {
            placement.placement_ambiguous =
                true;
            continue;
        }

        if (declared_node.has_value()) {
            placement.node =
                *declared_node;
        } else if (!step.dependencies.empty()) {
            const auto first =
                step.dependencies.front();
            if (first >= index) {
                throw std::invalid_argument(
                    "distributed placement expects topological step ids");
            }
            const auto inherited =
                plan.steps[first].node;
            bool same = true;
            for (const auto dependency :
                 step.dependencies) {
                if (dependency >= index ||
                    plan.steps[dependency].node !=
                        inherited ||
                    plan.steps[dependency].
                        placement_ambiguous) {
                    same = false;
                    break;
                }
            }
            if (same) {
                placement.node =
                    inherited;
            }
        }

        placement.remote =
            placement.node !=
            local_node;
        if (placement.remote) {
            remote_nodes.insert(
                placement.node);
        }
    }

    plan.remote_nodes.assign(
        remote_nodes.begin(),
        remote_nodes.end());
    std::sort(
        plan.remote_nodes.begin(),
        plan.remote_nodes.end());
    plan.has_remote_work =
        !plan.remote_nodes.empty();

    for (std::size_t destination = 0;
         destination <
            compiled.execution.steps.size();
         ++destination) {
        const auto& step =
            compiled.execution.steps[
                destination];
        const auto& dst_placement =
            plan.steps[destination];

        if (dst_placement.
            placement_ambiguous) {
            continue;
        }

        for (const auto source :
             step.dependencies) {
            if (source >=
                destination) {
                throw std::invalid_argument(
                    "distributed placement dependency is not topological");
            }
            const auto& src_placement =
                plan.steps[source];
            if (src_placement.
                    placement_ambiguous ||
                src_placement.node ==
                    dst_placement.node) {
                continue;
            }

            DistributedExecutionBoundary
                boundary;
            boundary.source_step =
                source;
            boundary.destination_step =
                static_cast<std::uint32_t>(
                    destination);
            boundary.source_node =
                src_placement.node;
            boundary.destination_node =
                dst_placement.node;

            const auto& source_io =
                compiled.io.steps[source];
            const auto& destination_io =
                compiled.io.steps[destination];

            for (const auto& write :
                 source_io.writes) {
                const auto found =
                    std::find_if(
                        destination_io.reads.begin(),
                        destination_io.reads.end(),
                        [&](const auto& read) {
                            return same_use(
                                write,
                                read);
                        });
                if (found ==
                    destination_io.reads.end()) {
                    continue;
                }

                const auto* allocation =
                    find_arena_allocation(
                        compiled.arena,
                        write);
                if (allocation == nullptr) {
                    throw std::logic_error(
                        "distributed boundary payload has no arena allocation");
                }
                boundary.payloads.push_back(
                    write);
                boundary.payload_bytes =
                    checked_add(
                        boundary.payload_bytes,
                        allocation->bytes);
            }

            std::uint64_t seed =
                fnv_offset;
            hash_u64(
                seed,
                compiled.execution_identity);
            hash_u64(
                seed,
                manifest.layout_identity);
            hash_u64(
                seed,
                tensor_access.plan_identity);
            boundary.boundary_id =
                boundary_identity(
                    seed,
                    boundary.source_step,
                    boundary.destination_step,
                    boundary.source_node,
                    boundary.destination_node,
                    boundary.payloads,
                    boundary.payload_bytes);
            if (boundary.boundary_id == 0) {
                throw std::logic_error(
                    "distributed boundary identity must be non-zero");
            }

            plan.boundary_payload_bytes_per_token =
                checked_add(
                    plan.boundary_payload_bytes_per_token,
                    boundary.payload_bytes);
            plan.boundaries.push_back(
                std::move(boundary));
        }
    }

    bool ambiguous = false;
    for (const auto& step : plan.steps) {
        ambiguous =
            ambiguous ||
            step.placement_ambiguous;
    }
    plan.executable =
        !ambiguous;

    std::uint64_t hash =
        fnv_offset;
    hash_u64(
        hash,
        plan.compiled_execution_identity);
    hash_u64(
        hash,
        plan.manifest_layout_identity);
    hash_u64(
        hash,
        plan.tensor_access_identity);
    hash_u64(hash, plan.local_node);
    hash_u64(hash, plan.steps.size());
    for (const auto& step : plan.steps) {
        hash_u64(hash, step.step_id);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                step.op));
        hash_u64(hash, step.node);
        hash_u64(
            hash,
            step.placement_ambiguous
                ? 1ULL
                : 0ULL);
    }
    hash_u64(
        hash,
        plan.boundaries.size());
    for (const auto& boundary :
         plan.boundaries) {
        hash_u64(
            hash,
            boundary.boundary_id);
    }
    plan.plan_identity =
        hash;
    if (plan.plan_identity == 0) {
        throw std::logic_error(
            "distributed placement plan identity must be non-zero");
    }
    return plan;
}

DistributedExecutionDecision
select_distributed_execution_mode(
    const DistributedPlacementPlan& plan,
    std::span<const std::uint32_t>
        available_remote_nodes,
    DistributedFallbackPolicy fallback) {
    if (plan.plan_identity == 0 ||
        !plan.executable) {
        return {
            DistributedExecutionMode::unavailable,
            {},
        };
    }

    if (!plan.has_remote_work) {
        return {
            DistributedExecutionMode::local_only,
            {},
        };
    }

    DistributedExecutionDecision
        decision;
    for (const auto node :
         plan.remote_nodes) {
        if (std::find(
                available_remote_nodes.begin(),
                available_remote_nodes.end(),
                node) ==
            available_remote_nodes.end()) {
            decision.missing_remote_nodes.push_back(
                node);
        }
    }

    if (decision.missing_remote_nodes.empty()) {
        decision.mode =
            DistributedExecutionMode::
                distributed;
    } else if (fallback.
                   local_model_copy_available) {
        decision.mode =
            DistributedExecutionMode::
                local_fallback;
    } else {
        decision.mode =
            DistributedExecutionMode::
                unavailable;
    }
    return decision;
}

transport::TransferDescriptor
make_distributed_boundary_descriptor(
    const DistributedPlacementPlan& plan,
    const DistributedExecutionBoundary& boundary,
    std::uint64_t connection_epoch,
    std::uint64_t request_id,
    std::uint64_t generation,
    std::uint64_t sequence,
    std::uint32_t dtype_id) {
    if (plan.plan_identity == 0 ||
        boundary.boundary_id == 0 ||
        connection_epoch == 0 ||
        request_id == 0 ||
        sequence == 0 ||
        dtype_id == 0 ||
        boundary.payload_bytes == 0) {
        throw std::invalid_argument(
            "distributed boundary descriptor requires initialized identities and payload");
    }

    transport::TransferDescriptor
        descriptor;
    descriptor.connection_epoch =
        connection_epoch;
    descriptor.model_plan_id =
        plan.plan_identity;
    descriptor.request_id =
        request_id;
    descriptor.generation =
        generation;
    descriptor.boundary_id =
        boundary.boundary_id;
    descriptor.sequence =
        sequence;
    descriptor.shape_fingerprint =
        boundary.boundary_id ^
        boundary.payload_bytes;
    descriptor.dtype_id =
        dtype_id;
    descriptor.byte_length =
        boundary.payload_bytes;
    return descriptor;
}

}  // namespace strix
