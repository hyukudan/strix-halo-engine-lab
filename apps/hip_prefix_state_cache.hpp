#pragma once
// Block-boundary prefix state cache on the tiled known-prefix prefill: session-state capture and
// restore over strix::PrefixStateCache (docs/prefix-state-cache.md).
#include "strix/arithmetic_lineage.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/prefix_state_cache.hpp"
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
#include <memory>
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
    std::uint32_t block_rows = 0, tail_rows = 0;
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
             static_cast<std::uint64_t>(in.tail_rows), in.context_tokens})
        u64(value);
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
        if (!store_ || !identity_.valid() || !block_rows_ || !every_ || every_ % block_rows_)
            throw std::invalid_argument("prefix state cache session needs a store, identity and an interval multiple of T");
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
                rows_.push_back({address.address, address.bytes / layout_.context_tokens, 0U});
                row_bytes_ += address.bytes / layout_.context_tokens;
            }
        }
        // Fixed components: whole allocations, overwritten in place.
        for (const auto component : {SessionStateComponent::recurrent_matrix, SessionStateComponent::gdn_convolution,
                 SessionStateComponent::ple_convolution, SessionStateComponent::ngram_history,
                 SessionStateComponent::position_history}) {
            if (!find_session_state_allocation(layout_, component)) continue;
            const auto address = resolve_session_state(layout_, memory_, component);
            fixed_.push_back({address.address, address.bytes, fixed_bytes_, address.storage == SessionStateStorage::device});
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
    void add_rows(std::uintptr_t address, std::uint64_t row_bytes, std::uint32_t shift) {
        require_unused();
        if (!address || !row_bytes || shift > 1U) throw std::invalid_argument("prefix state cache extra rows are invalid");
        rows_.push_back({address, row_bytes, shift});
        row_bytes_ += row_bytes;
    }
    // ... whole device ranges saved with the fixed state ...
    void add_fixed_device(std::uintptr_t address, std::uint64_t bytes) {
        require_unused();
        if (!address || !bytes) throw std::invalid_argument("prefix state cache extra fixed range is invalid");
        fixed_.push_back({address, bytes, fixed_bytes_, true});
        fixed_bytes_ += (bytes + 255U) & ~std::uint64_t{255};
    }
    // ... more identity (the consumers of the extra state), and a hook run after a restore.
    void extend_identity(const ModelContentIdentity& extra) {
        require_unused();
        std::vector<std::byte> frame(64);
        std::memcpy(frame.data(), identity_.bytes.data(), 32);
        std::memcpy(frame.data() + 32, extra.bytes.data(), 32);
        identity_ = sha256_bytes(frame);
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
    [[nodiscard]] std::uint64_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept { return fixed_bytes_; }
    [[nodiscard]] const PrefixStateCacheRequestStats& request() const noexcept { return request_; }
    [[nodiscard]] PrefixStateCache& store() noexcept { return *store_; }

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
        // STRIX_PREFIX_CACHE_LOOKUP=0: capture only (diagnostic; every request prefills from position 0).
        static const bool lookup = [] { const char* v = std::getenv("STRIX_PREFIX_CACHE_LOOKUP"); return !v || v[0] != '0'; }();
        if (prompt.size() < 2U || !lookup) return 0;
        const auto start = std::chrono::steady_clock::now();
        const auto checkpoint = store_->find_longest(identity_, prompt, block_rows_, prompt.size() - 1U);
        request_.lookup_ms = ms(start);
        if (!checkpoint) return 0;
        const auto restore_start = std::chrono::steady_clock::now();
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
    struct Row { std::uintptr_t address = 0; std::uint64_t bytes = 0; std::uint32_t shift = 0; };
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
    struct Fixed { std::uintptr_t address = 0; std::uint64_t bytes = 0, offset = 0; bool device = true; };
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
    bool used_ = false;
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
