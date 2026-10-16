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

static DistributedPlacementPlan build_distributed_placement_plan_impl(
    const CompiledSerialExecution& compiled,
    const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& tensor_access,
    std::uint32_t local_node, std::span<const std::uint32_t> compute_nodes,
    bool explicit_compute) {
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

    if(explicit_compute) {
        if(local_node>1U || compute_nodes.size()!=compiled.execution.steps.size() || compute_nodes.empty() ||
            std::any_of(compute_nodes.begin(),compute_nodes.end(),[](auto node){return node>1U;}))
            throw std::invalid_argument("explicit compute placement requires one node0/1 per serial step");
        validate_serial_execution_io_plan(compiled.execution,compiled.io);
        if(!compiled.arena.alignment || !compiled.arena.total_bytes)
            throw std::invalid_argument("explicit compute placement requires a bounded serial arena");
        for(std::size_t i=0;i<compiled.execution.steps.size();++i) {
            if(compiled.execution.steps[i].id!=i)
                throw std::invalid_argument("explicit compute placement requires dense serial IDs");
            for(const auto* uses:{&compiled.io.steps[i].reads,&compiled.io.steps[i].writes,&compiled.io.steps[i].temporaries})
                for(const auto use:*uses) {
                    const auto* a=find_arena_allocation(compiled.arena,use);
                    if(!a || !a->bytes || a->offset%compiled.arena.alignment ||
                        checked_add(a->offset,a->bytes)>compiled.arena.total_bytes)
                        throw std::invalid_argument("explicit compute placement IO has no bounded arena payload");
                }
        }
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

        if(explicit_compute) placement.node=compute_nodes[index];
        else {
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
    if(explicit_compute) {
        hash_u64(hash,0x535445504e4f4431ULL); // Explicit compute placement v1.
        hash_u64(hash,compiled.kernel_plan_identity);hash_u64(hash,compiled.logical_rows);
        hash_u64(hash,compiled.request_concurrency);hash_u64(hash,compiled.vocab_size);
        hash_u64(hash,compiled.max_position_embeddings);hash_u64(hash,compiled.device_graph_inputs);
        for(const auto& step:compiled.execution.steps) {
            hash_u64(hash,step.id);hash_u64(hash,static_cast<std::uint64_t>(step.op));
            hash_u64(hash,step.layer.has_value());if(step.layer) hash_u64(hash,*step.layer);
            hash_u64(hash,static_cast<std::uint64_t>(step.reads_state));
            hash_u64(hash,static_cast<std::uint64_t>(step.writes_state));
            hash_u64(hash,step.graph_safe_candidate);hash_u64(hash,step.dynamic_shape);
            hash_u64(hash,step.dependencies.size());for(const auto dependency:step.dependencies) hash_u64(hash,dependency);
        }
        hash_u64(hash,compiled.bindings.bindings.size());
        for(const auto& binding:compiled.bindings.bindings) {
            hash_u64(hash,binding.step_id);hash_u64(hash,static_cast<std::uint64_t>(binding.op));
            hash_u64(hash,binding.host_or_runtime_step);hash_u64(hash,binding.family.has_value());
            if(binding.family) hash_u64(hash,static_cast<std::uint64_t>(*binding.family));
            hash_u64(hash,binding.kernel_id.has_value());
            if(binding.kernel_id) {
                hash_u64(hash,binding.kernel_id->size());
                for(const char value:*binding.kernel_id) hash_u64(hash,static_cast<unsigned char>(value));
            }
        }
        hash_u64(hash,compiled.arena.alignment);hash_u64(hash,compiled.arena.total_bytes);
        hash_u64(hash,compiled.arena.allocations.size());
        const auto use_hash=[&](ExecutionBufferUse use) {
            hash_u64(hash,static_cast<std::uint64_t>(use.buffer.buffer_class));
            hash_u64(hash,use.buffer.slot);hash_u64(hash,use.generation);
        };
        for(const auto& a:compiled.arena.allocations) {
            use_hash(a.use);hash_u64(hash,a.offset);hash_u64(hash,a.bytes);
            hash_u64(hash,a.first_step);hash_u64(hash,a.last_step);
        }
        for(const auto& io:compiled.io.steps) {
            hash_u64(hash,io.step_id);hash_u64(hash,static_cast<std::uint64_t>(io.op));
            for(const auto* uses:{&io.reads,&io.writes,&io.temporaries}) {
                hash_u64(hash,uses->size());for(const auto use:*uses) use_hash(use);
            }
        }
    }
    plan.plan_identity =
        hash;
    if (plan.plan_identity == 0) {
        throw std::logic_error(
            "distributed placement plan identity must be non-zero");
    }
    return plan;
}

DistributedPlacementPlan build_distributed_placement_plan(
    const CompiledSerialExecution& compiled,const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& access,std::uint32_t local_node) {
    return build_distributed_placement_plan_impl(compiled,manifest,access,local_node,{},false);
}
DistributedPlacementPlan build_distributed_placement_plan(
    const CompiledSerialExecution& compiled,const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& access,std::span<const std::uint32_t> nodes,std::uint32_t local_node) {
    return build_distributed_placement_plan_impl(compiled,manifest,access,local_node,nodes,true);
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
