#include "strix/session_state_layout.hpp"

#include "strix/memory_plan.hpp"

#include <algorithm>
#include <cstring>
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
            "session-state addition overflow");
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
            "session-state multiplication overflow");
    }
    return left * right;
}

std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "session-state alignment must be non-zero");
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
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void append_allocation(
    SerialSessionStateLayout& layout,
    SessionStateComponent component,
    PersistentState flag,
    SessionStateStorage storage,
    std::uint64_t bytes,
    bool provisional) {
    if (bytes == 0) {
        throw std::invalid_argument(
            "session-state allocation must be non-zero");
    }

    auto& total =
        storage == SessionStateStorage::device
            ? layout.device_total_bytes
            : layout.host_total_bytes;
    const auto alignment =
        storage == SessionStateStorage::device
            ? layout.device_alignment
            : layout.host_alignment;

    const auto offset =
        align_up(total, alignment);
    total =
        checked_add(offset, bytes);

    layout.allocations.push_back(
        SessionStateAllocation{
            component,
            flag,
            storage,
            offset,
            bytes,
            provisional,
        });
}

std::uint64_t build_identity(
    const ModelContract& model,
    const SerialSessionStateLayout& layout) noexcept {
    std::uint64_t hash = fnv_offset;

    hash_u64(hash, model.vocab_size);
    hash_u64(hash, model.hidden_size);
    hash_u64(hash, model.max_position_embeddings);
    hash_u64(hash, model.hidden_layers);
    hash_u64(hash, model.full_attention_layers);
    hash_u64(hash, model.gdn_layers);
    hash_u64(hash, model.kv_heads);
    hash_u64(hash, model.head_dim);
    hash_u64(hash, model.gdn_key_heads);
    hash_u64(hash, model.gdn_key_head_dim);
    hash_u64(hash, model.gdn_value_heads);
    hash_u64(hash, model.gdn_value_head_dim);
    hash_u64(hash, model.gdn_conv_kernel);
    hash_u64(hash, model.hc_count);
    hash_u64(hash, model.ple_conv_kernel_size);
    hash_u64(hash, model.ngram_size);
    hash_u64(hash, model.indexer_n_heads);
    hash_u64(hash, model.indexer_kv_heads);
    hash_u64(hash, model.indexer_head_dim);
    hash_u64(hash, model.indexer_budget);
    hash_u64(hash, model.indexer_compress_ratio);

    std::uint32_t partial_rotary_bits = 0;
    static_assert(
        sizeof(partial_rotary_bits) ==
        sizeof(model.partial_rotary_factor));
    std::memcpy(
        &partial_rotary_bits,
        &model.partial_rotary_factor,
        sizeof(partial_rotary_bits));
    hash_u64(hash, partial_rotary_bits);

    hash_u64(hash, layout.context_tokens);
    hash_u64(hash, layout.device_alignment);
    hash_u64(hash, layout.host_alignment);
    hash_u64(hash, layout.device_total_bytes);
    hash_u64(hash, layout.host_total_bytes);

    for (const auto& allocation :
         layout.allocations) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                allocation.component));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                allocation.state_flag));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                allocation.storage));
        hash_u64(hash, allocation.offset);
        hash_u64(hash, allocation.bytes);
        hash_u64(
            hash,
            allocation.provisional
                ? 1ULL
                : 0ULL);
    }

    return hash;
}

}  // namespace

SerialSessionStateLayout
build_serial_session_state_layout(
    const ModelContract& model,
    const SerialSessionStateLayoutRequest& request) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.context_tokens == 0 ||
        request.context_tokens >
            model.max_position_embeddings) {
        throw std::invalid_argument(
            "serial session context is outside the model limit");
    }
    if (request.device_alignment == 0 ||
        request.host_alignment == 0) {
        throw std::invalid_argument(
            "serial session state alignments must be non-zero");
    }

    SerialSessionStateLayout layout;
    layout.context_tokens =
        request.context_tokens;
    layout.device_alignment =
        request.device_alignment;
    layout.host_alignment =
        request.host_alignment;

    append_allocation(
        layout,
        SessionStateComponent::attention_kv,
        PersistentState::attention_kv,
        SessionStateStorage::device,
        checked_mul(
            base_kv_bytes_per_token(model),
            request.context_tokens),
        false);

    append_allocation(
        layout,
        SessionStateComponent::recurrent_matrix,
        PersistentState::recurrent_matrix,
        SessionStateStorage::device,
        gdn_matrix_state_bytes_per_session(model),
        false);

    append_allocation(
        layout,
        SessionStateComponent::gdn_convolution,
        PersistentState::convolution_history,
        SessionStateStorage::device,
        gdn_conv_state_bytes_per_session(model),
        false);

    append_allocation(
        layout,
        SessionStateComponent::ple_convolution,
        PersistentState::convolution_history,
        SessionStateStorage::device,
        ple_conv_state_bytes_per_session(model),
        false);

    if (request.include_sparse_index_candidate) {
        append_allocation(
            layout,
            SessionStateComponent::sparse_index,
            PersistentState::sparse_index,
            SessionStateStorage::device,
            checked_mul(
                provisional_indexer_bytes_per_token(model),
                request.context_tokens),
            false);
    }

    append_allocation(
        layout,
        SessionStateComponent::ngram_history,
        PersistentState::ngram_history,
        SessionStateStorage::host,
        ngram_history_bytes_per_session(model),
        false);

    append_allocation(
        layout,
        SessionStateComponent::position_history,
        PersistentState::position_history,
        SessionStateStorage::host,
        position_history_bytes_per_session(model),
        false);

    layout.device_total_bytes =
        align_up(
            layout.device_total_bytes,
            layout.device_alignment);
    layout.host_total_bytes =
        align_up(
            layout.host_total_bytes,
            layout.host_alignment);

    layout.layout_identity =
        build_identity(model, layout);
    if (layout.layout_identity == 0) {
        throw std::logic_error(
            "serial session state layout identity must be non-zero");
    }

    return layout;
}

const SessionStateAllocation*
find_session_state_allocation(
    const SerialSessionStateLayout& layout,
    SessionStateComponent component) noexcept {
    const auto found =
        std::find_if(
            layout.allocations.begin(),
            layout.allocations.end(),
            [&](const SessionStateAllocation& allocation) {
                return allocation.component ==
                       component;
            });
    return found == layout.allocations.end()
        ? nullptr
        : &*found;
}

}  // namespace strix
