#pragma once
// Block-boundary prefix state cache on the tiled known-prefix prefill: session-state capture and
// restore over strix::PrefixStateCache (docs/engine/serving/prefix-state-cache.md).
#include "strix/arithmetic_lineage.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/prefix_state_cache.hpp"
#include "strix/rtx_state_import.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::decode_cli {

// Everything that changes the state the tiled prefill produces for a token prefix. Fields that
// cannot change it under one profile (the decode identities when T divides the prefix) are kept
// anyway: a key must never be shared by two configurations whose outputs could differ.
struct PrefixStateCacheIdentityInputs {
    ModelContentIdentity model_content;
    ModelContentIdentity engine_build;  // digest of the running executable (zero: not bound)
    std::uint64_t state_layout_identity = 0, runtime_compatibility_identity = 0;
    std::uint64_t decode_execution_identity = 0, decode_kernel_plan_identity = 0;
    std::uint64_t prefill_execution_identity = 0, prefill_kernel_plan_identity = 0;
    std::uint64_t tiled_execution_identity = 0, tiled_kernel_plan_identity = 0;
    std::uint64_t prefill_profile_identity = 0, decode_profile_identity = 0, profile_pair_identity = 0;
    std::uint32_t block_rows = 0;
    std::vector<std::uint32_t> tail_rows;
    std::uint64_t context_tokens = 0;
};

inline ModelContentIdentity prefix_state_cache_identity(const PrefixStateCacheIdentityInputs& in) {
    if (!in.model_content.valid() || !in.block_rows || !in.state_layout_identity || !in.runtime_compatibility_identity ||
        !in.decode_execution_identity || !in.prefill_execution_identity || !in.tiled_execution_identity)
        throw std::invalid_argument("prefix state cache identity is incomplete");
    std::vector<std::byte> frame;
    const auto u64 = [&](std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) frame.push_back(static_cast<std::byte>((value >> shift) & 255U));
    };
    const auto digest = [&](const ModelContentIdentity& value) {
        for (const auto byte : value.bytes) frame.push_back(static_cast<std::byte>(byte));
    };
    u64(0x5354524958504331ULL);  // format: prefix state cache v1
    digest(in.model_content);
    digest(in.engine_build);
    for (const auto value : {in.state_layout_identity, in.runtime_compatibility_identity, in.decode_execution_identity,
             in.decode_kernel_plan_identity, in.prefill_execution_identity, in.prefill_kernel_plan_identity,
             in.tiled_execution_identity, in.tiled_kernel_plan_identity, in.prefill_profile_identity,
             in.decode_profile_identity, in.profile_pair_identity, static_cast<std::uint64_t>(in.block_rows),
             static_cast<std::uint64_t>(in.tail_rows.size()), in.context_tokens})
        u64(value);
    for (const auto rows : in.tail_rows) u64(rows);
    return sha256_bytes(frame);
}

inline PrefixCacheBuffer hip_pinned_prefix_buffer(std::uint64_t bytes) {
    void* memory = nullptr;
    if (bytes && hipHostMalloc(&memory, static_cast<std::size_t>(bytes), hipHostMallocDefault) != hipSuccess)
        throw std::runtime_error("prefix state cache pinned host allocation failed");
    return {std::shared_ptr<std::byte>(static_cast<std::byte*>(memory), [](std::byte* p) { if (p) (void)hipHostFree(p); }), bytes};
}

struct PrefixStateCacheRequestStats {
    std::uint64_t prompt_tokens = 0, restored_position = 0, captured_checkpoints = 0, captured_segment_bytes = 0;
    std::uint64_t captured_fixed_bytes = 0, restored_bytes = 0;
    double lookup_ms = 0.0, restore_ms = 0.0, capture_ms = 0.0;
    // --rtx-state-import: this request started from an imported state (docs/rtx-state-import.md).
    bool rtx_import = false, rtx_sha_ok = false;
    std::uint64_t rtx_position = 0, rtx_bytes = 0, rtx_chunks = 0, rtx_local_rows = 0, rtx_local_bytes = 0;
    double rtx_restore_ms = 0.0, rtx_verify_ms = 0.0;
    std::string rtx_id, rtx_manifest_sha256;
};

// STRIX_RTX_EXPORT_STATE=<file>: write the state of the tiled MTP prompt at position STRIX_RTX_EXPORT_POSITION
// (default: the end of the prompt) in the strix-qwen-rtx-state-v1 file form (a producer for testing the import
// without an RTX service). Opt-in; unset, nothing changes.
struct RtxStateExportRequest {
    std::string path;
    std::uint64_t position = 0;  // 0: the end of every prompt
};
inline const std::optional<RtxStateExportRequest>& rtx_state_export_request() {
    static const std::optional<RtxStateExportRequest> request = []() -> std::optional<RtxStateExportRequest> {
        const char* path = std::getenv("STRIX_RTX_EXPORT_STATE");
        if (!path || !path[0]) return std::nullopt;
        RtxStateExportRequest r;
        r.path = path;
        if (const char* p = std::getenv("STRIX_RTX_EXPORT_POSITION"); p && p[0]) r.position = rtx_parse_u64(p);
        return r;
    }();
    return request;
}

// STRIX_PROMPT_STATE_DIGEST=1 (diagnostic, gates): every tiled MTP prompt gets the state component registry even without a
// prefix cache (no capture, no lookup), so its receipt carries the end-of-prompt state digest (target state, predictor
// rows and held row) on every route; STRIX_PREFIX_CACHE_DIGEST=1 prints it.
inline bool prompt_state_digest_request() {
    static const bool on = [] { const char* v = std::getenv("STRIX_PROMPT_STATE_DIGEST"); return v && v[0] == '1'; }();
    return on;
}

// What the consumer knows about the state an import must reproduce (computed from the compiled serial plan).
struct RtxStateImportFacts {
    std::array<std::uint64_t, 6> generations_per_row{};  // steps of the serial plan writing flag f (bit f)
    PersistentState valid_union = PersistentState::none;  // union of every step's writes
    ModelContentIdentity model_content;
};

// One live session's view of the store. Capture copies the state after a completed tiled block;
// restore copies a checkpoint into a fresh session and republishes its recorded metadata.
class HipPrefixStateCacheSession final {
public:
    HipPrefixStateCacheSession(std::shared_ptr<PrefixStateCache> store, ModelContentIdentity identity,
        std::uint32_t block_rows, std::uint64_t every, const ModelContract& model,
        const SerialSessionStateLayout& layout, const SessionStateMemoryBinding& memory)
        : store_(std::move(store)), identity_(identity), block_rows_(block_rows), every_(every),
          layout_(layout), memory_(memory) {
        // A null store: the component registry only (--rtx-state-import / STRIX_RTX_EXPORT_STATE without a prefix
        // cache): no capture, no lookup, no disk tier.
        if (!identity_.valid() || !block_rows_ || !every_ || every_ % block_rows_)
            throw std::invalid_argument("prefix state cache session needs an identity and an interval multiple of T");
        validate_session_state_memory_binding(layout_, memory_);
        if (memory_.stage_mapping) throw std::invalid_argument("prefix state cache does not support staged state");
        // Row components: every attention layer's KV rows, then every layer's raw index rows.
        for (const auto component : {SessionStateComponent::attention_kv, SessionStateComponent::sparse_index}) {
            if (!find_session_state_allocation(layout_, component)) continue;
            for (std::uint64_t ordinal = 0; ordinal < model.full_attention_layers; ++ordinal) {
                const auto layer = static_cast<std::uint32_t>((ordinal + 1U) * model.full_attention_interval - 1U);
                const auto address = resolve_indexed_attention_layer_state(layout_, memory_, model, layer, component);
                if (address.storage != SessionStateStorage::device || !layout_.context_tokens ||
                    address.bytes % layout_.context_tokens)
                    throw std::invalid_argument("prefix state cache row component is not a per-position device range");
                rows_.push_back({address.address, address.bytes / layout_.context_tokens, 0U,
                    std::string(component == SessionStateComponent::attention_kv ? "kv.L" : "index.L") + std::to_string(layer)});
                row_bytes_ += address.bytes / layout_.context_tokens;
            }
        }
        // Fixed components: whole allocations, overwritten in place.
        for (const auto component : {SessionStateComponent::recurrent_matrix, SessionStateComponent::gdn_convolution,
                 SessionStateComponent::ple_convolution, SessionStateComponent::ngram_history,
                 SessionStateComponent::position_history}) {
            if (!find_session_state_allocation(layout_, component)) continue;
            const auto address = resolve_session_state(layout_, memory_, component);
            const char* name = component == SessionStateComponent::recurrent_matrix ? "recurrent_matrix"
                : component == SessionStateComponent::gdn_convolution ? "gdn_convolution"
                : component == SessionStateComponent::ple_convolution ? "ple_convolution"
                : component == SessionStateComponent::ngram_history ? "ngram_history" : "position_history";
            fixed_.push_back({address.address, address.bytes, fixed_bytes_, address.storage == SessionStateStorage::device, name});
            fixed_bytes_ += (address.bytes + 255U) & ~std::uint64_t{255};
        }
        for (const auto& allocation : layout_.allocations) {
            const bool known = allocation.component == SessionStateComponent::attention_kv ||
                allocation.component == SessionStateComponent::sparse_index ||
                std::any_of(fixed_.begin(), fixed_.end(), [&](const Fixed& f) {
                    return f.address == resolve_session_state(layout_, memory_, allocation.component).address; });
            if (!known) throw std::invalid_argument("prefix state cache does not cover every session state component");
        }
        if (hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess)
            throw std::runtime_error("prefix state cache stream creation failed");
    }
    ~HipPrefixStateCacheSession() { if (stream_) (void)hipStreamDestroy(stream_); }
    HipPrefixStateCacheSession(const HipPrefixStateCacheSession&) = delete;
    HipPrefixStateCacheSession& operator=(const HipPrefixStateCacheSession&) = delete;

    // Extensions, before the first request (MTP predictor state on the MTP route):
    // per-position rows stored `shift` positions late (row j is kept with the block that holds
    // position j + shift: the MTP cache row of pair (j, j+1) needs the token at j + 1) ...
    void add_rows(std::uintptr_t address, std::uint64_t row_bytes, std::uint32_t shift, std::string id = {}) {
        require_unused();
        if (!address || !row_bytes || shift > 1U) throw std::invalid_argument("prefix state cache extra rows are invalid");
        if (id.empty()) id = "extra.rows." + std::to_string(rows_.size());
        rows_.push_back({address, row_bytes, shift, std::move(id)});
        row_bytes_ += row_bytes;
    }
    // ... whole device ranges saved with the fixed state ...
    void add_fixed_device(std::uintptr_t address, std::uint64_t bytes, std::string id = {}) {
        require_unused();
        if (!address || !bytes) throw std::invalid_argument("prefix state cache extra fixed range is invalid");
        if (id.empty()) id = "extra.fixed." + std::to_string(fixed_.size());
        fixed_.push_back({address, bytes, fixed_bytes_, true, std::move(id)});
        fixed_bytes_ += (bytes + 255U) & ~std::uint64_t{255};
    }
    // ... more identity (the consumers of the extra state), and a hook run after a restore.
    void extend_identity(const ModelContentIdentity& extra) {
        require_unused();
        std::vector<std::byte> frame(64);
        std::memcpy(frame.data(), identity_.bytes.data(), 32);
        std::memcpy(frame.data() + 32, extra.bytes.data(), 32);
        identity_ = sha256_bytes(frame);
        identity_extension_ = extra;
    }
    void set_restore_hook(std::function<void(std::uint64_t)> hook) { require_unused(); restore_hook_ = std::move(hook); }
    [[nodiscard]] const ModelContentIdentity& identity() const noexcept { return identity_; }
    // Bytes of one segment (block k): every row component's rows of that block.
    [[nodiscard]] std::uint64_t segment_bytes(std::uint64_t k) const {
        std::uint64_t total = 0;
        for (const auto& row : rows_) total += row.bytes * row_count(row, k);
        return total;
    }
    // SHA-256 of the state a session holds after `position` prompt rows: every row component's rows
    // [0, position - shift) and every fixed range (diagnostic, STRIX_PREFIX_CACHE_DIGEST=1).
    [[nodiscard]] ModelContentIdentity state_digest(std::uint64_t position) {
        check(hipDeviceSynchronize(), "prefix digest fence");
        std::vector<std::byte> frame;
        std::vector<std::byte> chunk;
        const auto append = [&](const void* source, std::uint64_t bytes, bool device) {
            chunk.resize(static_cast<std::size_t>(bytes));
            if (device) check(hipMemcpy(chunk.data(), source, static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost), "prefix digest copy");
            else std::memcpy(chunk.data(), source, static_cast<std::size_t>(bytes));
            const auto digest = sha256_bytes(chunk);
            frame.insert(frame.end(), reinterpret_cast<const std::byte*>(digest.bytes.data()),
                reinterpret_cast<const std::byte*>(digest.bytes.data()) + 32);
        };
        for (const auto& row : rows_) {
            const auto count = position > row.shift ? position - row.shift : 0U;
            if (count) append(reinterpret_cast<const void*>(row.address), count * row.bytes, true);
        }
        for (const auto& fixed : fixed_) append(reinterpret_cast<const void*>(fixed.address), fixed.bytes, fixed.device);
        return sha256_bytes(frame);
    }
    // STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1 (with STRIX_PREFIX_CACHE_DIGEST=1): state_digest's per-component hashes as a JSON object (component id -> SHA-256 of its
    // committed rows below `position` or of the whole fixed allocation), to locate a digest difference.
    [[nodiscard]] std::string component_digests(std::uint64_t position) {
        check(hipDeviceSynchronize(), "prefix digest fence");
        std::vector<std::byte> chunk;
        std::string out = "{";
        const auto append = [&](const std::string& id, const void* source, std::uint64_t bytes, bool device) {
            chunk.resize(static_cast<std::size_t>(bytes));
            if (device) check(hipMemcpy(chunk.data(), source, static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost), "prefix digest copy");
            else std::memcpy(chunk.data(), source, static_cast<std::size_t>(bytes));
            out += (out.size() > 1U ? ",\"" : "\"") + id + "\":\"" + model_content_identity_hex(sha256_bytes(chunk)) + '"';
        };
        for (const auto& row : rows_) {
            const auto count = position > row.shift ? position - row.shift : 0U;
            if (count) append(row.id, reinterpret_cast<const void*>(row.address), count * row.bytes, true);
        }
        for (const auto& fixed : fixed_) append(fixed.id, reinterpret_cast<const void*>(fixed.address), fixed.bytes, fixed.device);
        return out + '}';
    }
    [[nodiscard]] std::uint64_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept { return fixed_bytes_; }
    [[nodiscard]] const PrefixStateCacheRequestStats& request() const noexcept { return request_; }
    [[nodiscard]] PrefixStateCache& store() noexcept { return *store_; }
    // False: the component registry of --rtx-state-import / STRIX_RTX_EXPORT_STATE alone (no capture, no lookup).
    [[nodiscard]] bool has_store() const noexcept { return store_ != nullptr; }

    // ---- externally produced state (strix-qwen-rtx-state-v1, docs/rtx-state-import.md) ----
    // The consumer's facts, before the first request. Needed by the import, reported by the export.
    void set_rtx_facts(const RtxStateImportFacts& facts) { require_unused(); rtx_facts_ = facts; }
    // The import source: begin_request restores a parked state of `registry` matching the prompt, before the
    // checkpoint lookup. Publishes this session's geometry to the registry (GET /admin/rtx-import/layout).
    void enable_rtx_import(std::shared_ptr<RtxStateImportRegistry> registry, const SerialSessionContext& fresh) {
        require_unused();
        if (!registry || !rtx_facts_) throw std::invalid_argument("rtx state import needs a registry and the consumer facts");
        rtx_registry_ = std::move(registry);
        rtx_fresh_execution_ = fresh.execution_identity;
        rtx_fresh_compatibility_ = fresh.runtime_compatibility_identity;
        rtx_registry_->set_layout(rtx_layout(fresh.state));
    }
    // The geometry of this session's state (row and fixed components in capture order).
    [[nodiscard]] RtxStateLayout rtx_layout(const PersistentStateGenerations& base) const {
        RtxStateLayout l;
        for (const auto& row : rows_) l.rows.push_back({row.id, row.bytes, row.shift});
        for (const auto& f : fixed_) l.fixed.push_back({f.id, f.bytes, f.offset});
        l.row_bytes = row_bytes_;
        l.fixed_bytes = fixed_bytes_;
        l.layout_identity = layout_.layout_identity;
        l.context_tokens = layout_.context_tokens;
        if (rtx_facts_) {
            l.generations_per_row = rtx_facts_->generations_per_row;
            l.valid_union = static_cast<std::uint32_t>(rtx_facts_->valid_union);
            if (rtx_facts_->model_content.valid()) l.model_content_hex = model_content_identity_hex(rtx_facts_->model_content);
        }
        l.generations_base = base.generations;
        l.valid_base = static_cast<std::uint32_t>(base.valid);
        if (identity_extension_) l.identity_extension_hex = model_content_identity_hex(*identity_extension_);
        l.cache_identity_hex = model_content_identity_hex(identity_);
        l.execution_identity = rtx_fresh_execution_;
        l.runtime_compatibility_identity = rtx_fresh_compatibility_;
        return l;
    }
    // After a unit of the tiled prompt (the session at `position`): writes the STRIX_RTX_EXPORT_STATE file when
    // this is the requested position (default: the end of the prompt). Rows [0, position - shift) of every row
    // component and the fixed state: exactly what an import restores.
    void after_unit(const SerialSessionContext& session, std::span<const std::uint32_t> prompt, std::uint64_t position) {
        const auto& request = rtx_state_export_request();
        if (!request || !position || position > prompt.size() || session.next_position != position ||
            (request->position ? request->position != position : position != prompt.size()))
            return;
        if (session.health != SerialSessionHealth::ready || session.arithmetic_lineage.tainted ||
            session.arithmetic_lineage.intervals.empty() || !arithmetic_lineage_valid(session.arithmetic_lineage, position))
            throw std::runtime_error("STRIX_RTX_EXPORT_STATE: the session cannot be exported at this position");
        const auto start = std::chrono::steady_clock::now();
        auto layout = rtx_layout(session.state);
        if (!layout.execution_identity) {
            layout.execution_identity = session.execution_identity;
            layout.runtime_compatibility_identity = session.runtime_compatibility_identity;
        }
        std::vector<std::byte> payload(static_cast<std::size_t>(layout.payload_bytes(position)));
        check(hipDeviceSynchronize(), "rtx export fence");
        std::uint64_t at = 0;
        for (const auto& row : rows_) {
            const auto count = position > row.shift ? position - row.shift : 0U;
            if (count)
                check(hipMemcpy(payload.data() + at, reinterpret_cast<const void*>(row.address), static_cast<std::size_t>(count * row.bytes),
                    hipMemcpyDeviceToHost), "rtx export rows");
            at += count * row.bytes;
        }
        for (const auto& fixed : fixed_) {
            if (fixed.device)
                check(hipMemcpy(payload.data() + at + fixed.offset, reinterpret_cast<const void*>(fixed.address),
                    static_cast<std::size_t>(fixed.bytes), hipMemcpyDeviceToHost), "rtx export fixed state");
            else
                std::memcpy(payload.data() + at + fixed.offset, reinterpret_cast<const void*>(fixed.address), static_cast<std::size_t>(fixed.bytes));
        }
        RtxStateFileSession facts;
        facts.next_position = session.next_position;
        facts.committed_tokens = session.committed_tokens;
        facts.valid = static_cast<std::uint32_t>(session.state.valid);
        facts.generations = session.state.generations;
        for (const auto& i : session.arithmetic_lineage.intervals)
            facts.lineage.push_back({i.begin, i.end, static_cast<std::uint64_t>(i.producer), i.numerical_plan_identity,
                i.reduction_plan_identity, i.state_update_identity});
        // The producer of an AMD export: the first interval's identities (they name this engine's arithmetic).
        const auto& first = session.arithmetic_lineage.intervals.front();
        const RtxStateProducer producer{first.numerical_plan_identity, first.reduction_plan_identity, first.state_update_identity};
        const auto sha = write_rtx_state_file(request->path, layout, prompt.first(static_cast<std::size_t>(position)), position, producer,
            facts, payload);
        std::cerr << "strix rtx export: position " << position << " payload " << payload.size() << " B sha256 "
                  << model_content_identity_hex(sha) << " -> " << request->path << " ("
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << " ms)" << std::endl;
    }

    // Restores the longest cached prefix of `prompt` that leaves at least one prompt token to run
    // into the fresh `session`. Returns the restored position (0: miss, nothing changed).
    std::uint64_t begin_request(SerialSessionContext& session, std::span<const std::uint32_t> prompt) {
        used_ = true;
        request_ = {};
        request_.prompt_tokens = prompt.size();
        segments_.clear();
        if (session.next_position || session.committed_tokens || session.health != SerialSessionHealth::ready ||
            !session.arithmetic_lineage.intervals.empty() || session.state_layout_identity != layout_.layout_identity)
            throw std::invalid_argument("prefix state cache restores only into a fresh ready session");
        imported_ = false;
        if (rtx_registry_ && prompt.size() >= 2U)
            if (auto state = rtx_registry_->take(prompt)) return restore_rtx_state(session, *state);
        if (!store_) return 0;
        // STRIX_PREFIX_CACHE_LOOKUP=0: capture only (diagnostic; every request prefills from position 0).
        static const bool lookup = [] { const char* v = std::getenv("STRIX_PREFIX_CACHE_LOOKUP"); return !v || v[0] != '0'; }();
        if (prompt.size() < 2U || !lookup) return 0;
        const auto start = std::chrono::steady_clock::now();
        const auto checkpoint = store_->find_longest(identity_, prompt, block_rows_, prompt.size() - 1U);
        request_.lookup_ms = ms(start);
        if (!checkpoint) return 0;
        const auto restore_start = std::chrono::steady_clock::now();
        const auto meta = restore_checkpoint(session, *checkpoint);
        session.next_position = meta.position;
        session.committed_tokens = meta.committed_tokens;
        session.state = meta.state;
        session.arithmetic_lineage = meta.lineage;
        if (!arithmetic_lineage_valid(session.arithmetic_lineage, session.next_position)) {
            session.health = SerialSessionHealth::poisoned;
            throw std::runtime_error("prefix state cache restored an invalid arithmetic lineage");
        }
        segments_ = checkpoint->segments;
        if (restore_hook_) restore_hook_(meta.position);
        request_.restored_position = meta.position;
        request_.restore_ms = ms(restore_start);
        return meta.position;
    }

    // After a completed tiled block: the session is at a block boundary of `prompt` (absolute
    // positions). Captures it when the boundary is selected and not cached yet.
    // `covered` is how far complete T-row blocks can run for this prompt (the prompt length, or one
    // less when the last prompt row must run serially): the last boundary below it is always captured.
    void after_block(const SerialSessionContext& session, std::span<const std::uint32_t> prompt,
        std::uint64_t covered = UINT64_MAX) {
        // An imported state was produced elsewhere: nothing derived from it enters the store under this identity.
        if (!store_ || imported_) return;
        const auto position = session.next_position;
        covered = std::min<std::uint64_t>(covered, prompt.size());
        if (!position || position % block_rows_ || position > prompt.size() || session.health != SerialSessionHealth::ready ||
            session.arithmetic_lineage.tainted || session.committed_tokens != position)
            return;
        const bool last = position + block_rows_ > covered;
        if (position % every_ && !last) return;
        const auto blocks = position / block_rows_;
        const auto digests = prefix_cache_block_digests(identity_, prompt, block_rows_, blocks);
        if (digests.size() != blocks || store_->contains(digests.back())) return;
        const auto start = std::chrono::steady_clock::now();
        check(hipDeviceSynchronize(), "prefix capture fence");
        // Segments: retained from the restore or an earlier capture of this request, else from the
        // store, else read now (rows never change once written).
        for (auto k = static_cast<std::uint64_t>(segments_.size()); k < blocks; ++k) {
            if (auto retained = store_->find_segment(digests[k])) {
                segments_.push_back(std::move(retained));
                continue;
            }
            auto segment = std::make_shared<PrefixCacheSegment>();
            segment->identity = identity_;
            segment->digest = digests[k];
            segment->parent = k ? digests[k - 1U] : identity_;
            segment->block_index = k;
            segment->block_rows = block_rows_;
            segment->tokens.assign(prompt.begin() + static_cast<std::ptrdiff_t>(k * block_rows_),
                prompt.begin() + static_cast<std::ptrdiff_t>((k + 1U) * block_rows_));
            segment->payload = store_->allocate(segment_bytes(k));
            std::uint64_t offset = 0;
            for (const auto& row : rows_) {
                const auto bytes = row.bytes * row_count(row, k);
                if (bytes)
                    check(hipMemcpyAsync(segment->payload.data.get() + offset,
                        reinterpret_cast<const void*>(row.address + row_first(row, k) * row.bytes), static_cast<std::size_t>(bytes),
                        hipMemcpyDeviceToHost, stream_), "prefix capture rows");
                offset += bytes;
            }
            request_.captured_segment_bytes += segment->payload.bytes;
            segments_.push_back(std::move(segment));
        }
        auto checkpoint = std::make_shared<PrefixCacheCheckpoint>();
        checkpoint->identity = identity_;
        checkpoint->digest = digests.back();
        checkpoint->position = position;
        checkpoint->block_rows = block_rows_;
        checkpoint->segments.assign(segments_.begin(), segments_.begin() + static_cast<std::ptrdiff_t>(blocks));
        checkpoint->fixed = store_->allocate(fixed_bytes_);
        for (const auto& fixed : fixed_) {
            if (fixed.device)
                check(hipMemcpyAsync(checkpoint->fixed.data.get() + fixed.offset, reinterpret_cast<const void*>(fixed.address),
                    static_cast<std::size_t>(fixed.bytes), hipMemcpyDeviceToHost, stream_), "prefix capture fixed state");
            else
                std::memcpy(checkpoint->fixed.data.get() + fixed.offset, reinterpret_cast<const void*>(fixed.address),
                    static_cast<std::size_t>(fixed.bytes));
        }
        check(hipStreamSynchronize(stream_), "prefix capture completion");
        checkpoint->metadata = encode_metadata(session);
        request_.captured_fixed_bytes += fixed_bytes_;
        used_ = true;
        if (store_->insert(std::move(checkpoint))) ++request_.captured_checkpoints;
        request_.capture_ms += ms(start);
    }

private:
    // Restores a committed import at p = state.position() exactly like a checkpoint: rows [0, p - shift) of every
    // row component, the fixed blob, then the counters a run over p rows publishes (generation f advances once per
    // row and per serial-plan step writing f) and one honest prefill_chunk lineage interval [0, p) carrying the
    // producer's identities (cooperative local_rows q: the checkpoint of [0, q), then an interval [q, p)).
    std::uint64_t restore_rtx_state(SerialSessionContext& session, const RtxStateStaging& state) {
        const auto start = std::chrono::steady_clock::now();
        const auto& l = state.layout();
        const auto p = state.position();
        if (l.layout_identity != layout_.layout_identity || l.row_bytes != row_bytes_ || l.fixed_bytes != fixed_bytes_ ||
            l.rows.size() != rows_.size() || l.fixed.size() != fixed_.size() || !rtx_facts_ || p >= layout_.context_tokens ||
            session.execution_identity != rtx_fresh_execution_ || session.runtime_compatibility_identity != rtx_fresh_compatibility_)
            throw std::runtime_error("rtx state import does not match the live session geometry");
        for (std::size_t c = 0; c < rows_.size(); ++c)
            if (l.rows[c].bytes_per_row != rows_[c].bytes || l.rows[c].shift != rows_[c].shift)
                throw std::runtime_error("rtx state import row component differs");
        for (std::size_t f = 0; f < fixed_.size(); ++f)
            if (l.fixed[f].bytes != fixed_[f].bytes || l.fixed[f].offset != fixed_[f].offset)
                throw std::runtime_error("rtx state import fixed component differs");
        PersistentStateGenerations generations = session.state;
        generations.valid = generations.valid | rtx_facts_->valid_union;
        for (std::size_t f = 0; f < generations.generations.size(); ++f) {
            const auto per_row = rtx_facts_->generations_per_row[f];
            if (per_row && p > (UINT64_MAX - generations.generations[f]) / per_row)
                throw std::overflow_error("rtx state import generation overflows");
            generations.generations[f] += p * per_row;
        }
        const auto& producer = state.request().producer;
        // Cooperative (local_rows q > 0): rows [0, q) and the facts of prefix [0, q) come from this node's own
        // checkpoint of prompt[0, q); the import adds the rows [q, p) and the fixed state of position p.
        const auto q = state.request().local_rows;
        std::uint64_t local_bytes = 0;
        if (q) {
            if (!store_) throw std::runtime_error("rtx state import with local_rows needs the prefix state cache (--prefix-cache-host-bytes)");
            const auto checkpoint = store_->find_longest(identity_, state.request().tokens, block_rows_, q);
            if (!checkpoint || checkpoint->position != q)
                throw std::runtime_error("rtx state import local_rows " + std::to_string(q) + ": this node holds no checkpoint of prompt[0, " +
                    std::to_string(q) + ")");
            const auto meta = restore_checkpoint(session, *checkpoint);
            if (meta.position != q || meta.committed_tokens != q) {
                session.health = SerialSessionHealth::poisoned;
                throw std::runtime_error("rtx state import local checkpoint facts differ");
            }
            for (const auto& segment : checkpoint->segments) local_bytes += segment->payload.bytes;
            generations = meta.state;
            generations.valid = generations.valid | rtx_facts_->valid_union;
            for (std::size_t f = 0; f < generations.generations.size(); ++f) {
                const auto per_row = rtx_facts_->generations_per_row[f];
                if (per_row && p - q > (UINT64_MAX - generations.generations[f]) / per_row)
                    throw std::overflow_error("rtx state import generation overflows");
                generations.generations[f] += (p - q) * per_row;
            }
            session.arithmetic_lineage = meta.lineage;
        }
        auto lineage = prepare_prefill_chunk_lineage(session.arithmetic_lineage, q, p, producer.numerical, producer.reduction,
            producer.state_update);
        try {
            for (std::size_t c = 0; c < rows_.size(); ++c) {
                const auto rows = state.component(c);
                const auto skip = static_cast<std::size_t>(rows_[c].bytes * l.component_rows(c, q));
                if (rows.size() > skip)
                    check(hipMemcpyAsync(reinterpret_cast<void*>(rows_[c].address + skip), rows.data() + skip, rows.size() - skip,
                        hipMemcpyHostToDevice, stream_), "rtx import rows");
            }
            const auto blob = state.fixed_blob();
            for (const auto& fixed : fixed_) {
                if (fixed.device)
                    check(hipMemcpyAsync(reinterpret_cast<void*>(fixed.address), blob.data() + fixed.offset, static_cast<std::size_t>(fixed.bytes),
                        hipMemcpyHostToDevice, stream_), "rtx import fixed state");
                else
                    std::memcpy(reinterpret_cast<void*>(fixed.address), blob.data() + fixed.offset, static_cast<std::size_t>(fixed.bytes));
            }
            check(hipStreamSynchronize(stream_), "rtx import completion");
        } catch (...) {
            // The device state may be partially overwritten: the session cannot be used.
            session.health = SerialSessionHealth::poisoned;
            session.arithmetic_lineage.tainted = true;
            throw;
        }
        session.next_position = p;
        session.committed_tokens = p;
        session.state = generations;
        session.arithmetic_lineage = std::move(lineage);
        if (!arithmetic_lineage_valid(session.arithmetic_lineage, session.next_position)) {
            session.health = SerialSessionHealth::poisoned;
            throw std::runtime_error("rtx state import produced an invalid arithmetic lineage");
        }
        if (restore_hook_) restore_hook_(p);
        imported_ = true;
        request_.restored_position = p;
        request_.restored_bytes = state.payload_bytes();
        request_.restore_ms = ms(start);
        request_.rtx_import = true;
        request_.rtx_sha_ok = state.committed();  // committed = coverage + per-chunk digests + manifest verified
        request_.rtx_position = p;
        request_.rtx_bytes = state.payload_bytes();
        request_.rtx_local_rows = q;
        request_.rtx_local_bytes = local_bytes;
        request_.rtx_chunks = state.chunks();
        request_.rtx_restore_ms = request_.restore_ms;
        request_.rtx_verify_ms = state.commit_ms();
        request_.rtx_id = state.id();
        request_.rtx_manifest_sha256 = model_content_identity_hex(state.manifest_sha256());
        return p;
    }

    struct Row { std::uintptr_t address = 0; std::uint64_t bytes = 0; std::uint32_t shift = 0; std::string id; };
    // Row j of a component with shift s belongs to block floor((j + s) / T).
    std::uint64_t row_first(const Row& row, std::uint64_t k) const {
        const auto first = k * block_rows_;
        return first > row.shift ? first - row.shift : 0U;
    }
    std::uint64_t row_count(const Row& row, std::uint64_t k) const {
        return (k + 1U) * block_rows_ - row.shift - row_first(row, k);
    }
    void require_unused() const {
        if (used_) throw std::logic_error("prefix state cache extensions must precede the first request");
    }
    struct Fixed { std::uintptr_t address = 0; std::uint64_t bytes = 0, offset = 0; bool device = true; std::string id; };
    struct Metadata {
        std::uint64_t position = 0, committed_tokens = 0, layout_identity = 0, row_bytes = 0, fixed_bytes = 0;
        std::uint64_t execution_identity = 0, runtime_compatibility_identity = 0;
        PersistentStateGenerations state;
        ArithmeticLineage lineage;
    };
    static constexpr std::uint64_t metadata_magic = 0x31544d5043585054ULL;

    static void check(hipError_t error, const char* what) {
        if (error != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
    }
    static double ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    std::vector<std::byte> encode_metadata(const SerialSessionContext& session) const {
        std::vector<std::byte> out;
        const auto u64 = [&](std::uint64_t value) {
            for (unsigned shift = 0; shift < 64; shift += 8) out.push_back(static_cast<std::byte>((value >> shift) & 255U));
        };
        u64(metadata_magic);
        u64(session.next_position); u64(session.committed_tokens); u64(layout_.layout_identity);
        u64(row_bytes_); u64(fixed_bytes_); u64(session.execution_identity); u64(session.runtime_compatibility_identity);
        u64(static_cast<std::uint64_t>(session.state.valid));
        for (const auto generation : session.state.generations) u64(generation);
        const auto& lineage = session.arithmetic_lineage;
        u64(lineage.history_identity); u64(lineage.tainted ? 1U : 0U); u64(lineage.intervals.size());
        for (const auto& interval : lineage.intervals) {
            u64(interval.begin); u64(interval.end); u64(static_cast<std::uint64_t>(interval.producer));
            u64(interval.numerical_plan_identity); u64(interval.reduction_plan_identity); u64(interval.state_update_identity);
        }
        return out;
    }
    static Metadata decode_metadata(std::span<const std::byte> in) {
        std::size_t at = 0;
        const auto u64 = [&]() {
            if (at + 8U > in.size()) throw std::runtime_error("prefix state cache metadata is truncated");
            std::uint64_t value = 0;
            for (unsigned shift = 0; shift < 64; shift += 8)
                value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(in[at++])) << shift;
            return value;
        };
        if (u64() != metadata_magic) throw std::runtime_error("prefix state cache metadata has another format");
        Metadata m;
        m.position = u64(); m.committed_tokens = u64(); m.layout_identity = u64(); m.row_bytes = u64(); m.fixed_bytes = u64();
        m.execution_identity = u64(); m.runtime_compatibility_identity = u64();
        m.state.valid = static_cast<PersistentState>(u64());
        for (auto& generation : m.state.generations) generation = u64();
        m.lineage.history_identity = u64();
        m.lineage.tainted = u64() != 0U;
        const auto count = u64();
        if (count > in.size() / 48U) throw std::runtime_error("prefix state cache lineage is malformed");
        for (std::uint64_t i = 0; i < count; ++i) {
            ArithmeticRowInterval interval;
            interval.begin = u64(); interval.end = u64(); interval.producer = static_cast<ArithmeticProducer>(u64());
            interval.numerical_plan_identity = u64(); interval.reduction_plan_identity = u64(); interval.state_update_identity = u64();
            m.lineage.intervals.push_back(interval);
        }
        if (at != in.size()) throw std::runtime_error("prefix state cache metadata has trailing bytes");
        return m;
    }

    // Copies a store checkpoint's rows and fixed state into the session's device buffers (synchronized) after
    // checking it against the live geometry; the caller sets the session facts. Throws (poisoning the session
    // once a copy started) on a mismatch or a failed copy.
    Metadata restore_checkpoint(SerialSessionContext& session, const PrefixCacheCheckpoint& checkpoint_ref) {
        const auto* checkpoint = &checkpoint_ref;
        const auto meta = decode_metadata(checkpoint->metadata);
        if (meta.position != checkpoint->position || meta.layout_identity != layout_.layout_identity ||
            meta.row_bytes != row_bytes_ || meta.fixed_bytes != fixed_bytes_ || checkpoint->fixed.bytes != fixed_bytes_ ||
            meta.execution_identity != session.execution_identity ||
            meta.runtime_compatibility_identity != session.runtime_compatibility_identity)
            throw std::runtime_error("prefix state cache checkpoint does not match the live session geometry");
        for (const auto& segment : checkpoint->segments)
            if (segment->payload.bytes != segment_bytes(segment->block_index))
                throw std::runtime_error("prefix state cache segment size differs");
        try {
            for (const auto& segment : checkpoint->segments) {
                std::uint64_t offset = 0;
                for (const auto& row : rows_) {
                    const auto bytes = row.bytes * row_count(row, segment->block_index);
                    if (bytes)
                        check(hipMemcpyAsync(reinterpret_cast<void*>(row.address + row_first(row, segment->block_index) * row.bytes),
                            segment->payload.data.get() + offset, static_cast<std::size_t>(bytes), hipMemcpyHostToDevice, stream_),
                            "prefix restore rows");
                    offset += bytes;
                }
                request_.restored_bytes += segment->payload.bytes;
            }
            for (const auto& fixed : fixed_) {
                if (fixed.device)
                    check(hipMemcpyAsync(reinterpret_cast<void*>(fixed.address), checkpoint->fixed.data.get() + fixed.offset,
                        static_cast<std::size_t>(fixed.bytes), hipMemcpyHostToDevice, stream_), "prefix restore fixed state");
                else
                    std::memcpy(reinterpret_cast<void*>(fixed.address), checkpoint->fixed.data.get() + fixed.offset,
                        static_cast<std::size_t>(fixed.bytes));
            }
            request_.restored_bytes += checkpoint->fixed.bytes;
            check(hipStreamSynchronize(stream_), "prefix restore completion");
        } catch (...) {
            // The device state may be partially overwritten: the session cannot be used.
            session.health = SerialSessionHealth::poisoned;
            session.arithmetic_lineage.tainted = true;
            throw;
        }
        return meta;
    }

    std::shared_ptr<PrefixStateCache> store_;
    ModelContentIdentity identity_;
    std::uint32_t block_rows_ = 0;
    std::uint64_t every_ = 0;
    const SerialSessionStateLayout& layout_;
    const SessionStateMemoryBinding& memory_;
    std::vector<Row> rows_;
    std::vector<Fixed> fixed_;
    std::uint64_t row_bytes_ = 0, fixed_bytes_ = 0;
    std::vector<std::shared_ptr<const PrefixCacheSegment>> segments_;
    std::function<void(std::uint64_t)> restore_hook_;
    bool used_ = false, imported_ = false;
    std::optional<ModelContentIdentity> identity_extension_;
    std::optional<RtxStateImportFacts> rtx_facts_;
    std::shared_ptr<RtxStateImportRegistry> rtx_registry_;
    std::uint64_t rtx_fresh_execution_ = 0, rtx_fresh_compatibility_ = 0;
    PrefixStateCacheRequestStats request_;
    hipStream_t stream_ = nullptr;
};

// Digest of the running executable, so a disk tier never serves a state another build produced.
inline ModelContentIdentity prefix_state_cache_engine_build() {
    std::ifstream file("/proc/self/exe", std::ios::binary);
    if (!file) return {};
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return sha256_bytes(std::as_bytes(std::span<const char>(bytes)));
}

// Token IDs separated by commas or whitespace.
inline std::vector<std::uint32_t> read_prefix_cache_prompt(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("prefix cache prompt file cannot be read: " + path);
    std::stringstream text;
    text << file.rdbuf();
    auto content = text.str();
    std::replace(content.begin(), content.end(), ',', ' ');
    std::istringstream in(content);
    std::vector<std::uint32_t> tokens;
    unsigned long long value = 0;
    while (in >> value) {
        if (value > UINT32_MAX) throw std::invalid_argument("prefix cache prompt token exceeds uint32");
        tokens.push_back(static_cast<std::uint32_t>(value));
    }
    if (!in.eof()) throw std::invalid_argument("prefix cache prompt file has a non-numeric token: " + path);
    if (tokens.empty()) throw std::invalid_argument("prefix cache prompt file is empty: " + path);
    return tokens;
}

}  // namespace strix::decode_cli
