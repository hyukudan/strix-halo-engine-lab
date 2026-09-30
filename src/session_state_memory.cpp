#include "strix/session_state_memory.hpp"

#include "strix/memory_plan.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace strix {
namespace {

void validate_pool(
    std::uintptr_t base,
    std::uint64_t bytes,
    std::uint64_t required,
    std::uint64_t generation,
    const char* name) {
    if (required == 0) {
        return;
    }
    if (base == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " session-state pool has a null base");
    }
    if (bytes < required) {
        throw std::invalid_argument(
            std::string(name) +
            " session-state pool is undersized");
    }
    if (generation == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " session-state allocation generation must be non-zero");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (required >
        static_cast<std::uint64_t>(
            maximum - base)) {
        throw std::overflow_error(
            std::string(name) +
            " session-state address range overflows uintptr_t");
    }
}

}  // namespace

void validate_session_state_memory_binding(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory) {
    if (layout.layout_identity == 0 ||
        memory.layout_identity !=
            layout.layout_identity) {
        throw std::invalid_argument(
            "session-state memory identity does not match layout");
    }

    validate_pool(
        memory.device_base,
        memory.device_bytes,
        layout.device_total_bytes,
        memory.device_allocation_generation,
        "device");

    validate_pool(
        memory.host_base,
        memory.host_bytes,
        layout.host_total_bytes,
        memory.host_allocation_generation,
        "host");
}

SessionStateAddress resolve_session_state(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory,
    SessionStateComponent component) {
    validate_session_state_memory_binding(
        layout, memory);

    const auto* allocation =
        find_session_state_allocation(
            layout, component);
    if (allocation == nullptr) {
        throw std::invalid_argument(
            "session-state component is not present in layout");
    }

    const bool device =
        allocation->storage ==
        SessionStateStorage::device;
    const auto base =
        device
            ? memory.device_base
            : memory.host_base;
    const auto pool_bytes =
        device
            ? memory.device_bytes
            : memory.host_bytes;
    const auto generation =
        device
            ? memory.device_allocation_generation
            : memory.host_allocation_generation;

    if (allocation->offset > pool_bytes ||
        allocation->bytes >
            pool_bytes - allocation->offset) {
        throw std::invalid_argument(
            "session-state component exceeds bound pool");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (allocation->offset >
        static_cast<std::uint64_t>(
            maximum - base)) {
        throw std::overflow_error(
            "session-state component address overflows uintptr_t");
    }

    return SessionStateAddress{
        allocation->component,
        allocation->state_flag,
        allocation->storage,
        base +
            static_cast<std::uintptr_t>(
                allocation->offset),
        allocation->bytes,
        generation,
        allocation->provisional,
    };
}

}  // namespace strix


namespace strix {


std::uint32_t indexed_attention_layer_ordinal(
    const ModelContract& model,
    std::uint32_t zero_based_layer) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (zero_based_layer >=
        model.hidden_layers) {
        throw std::out_of_range(
            "indexed-attention layer index exceeds model layer count");
    }

    const auto one_based =
        static_cast<std::uint64_t>(
            zero_based_layer) +
        1ULL;
    if (one_based %
            model.full_attention_interval !=
        0) {
        throw std::invalid_argument(
            "linear-attention layer has no indexed-attention state");
    }

    const auto ordinal =
        one_based /
            model.full_attention_interval -
        1ULL;
    if (ordinal >=
        model.full_attention_layers) {
        throw std::logic_error(
            "derived indexed-attention ordinal exceeds configured layers");
    }
    return static_cast<std::uint32_t>(
        ordinal);
}

SessionStateAddress resolve_indexed_attention_layer_state(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory,
    const ModelContract& model,
    std::uint32_t zero_based_layer,
    SessionStateComponent component) {
    if (component !=
            SessionStateComponent::attention_kv &&
        component !=
            SessionStateComponent::sparse_index) {
        throw std::invalid_argument(
            "indexed-attention layer resolver supports only KV and sparse-index state");
    }

    auto result =
        resolve_session_state(
            layout,
            memory,
            component);
    const auto ordinal =
        indexed_attention_layer_ordinal(
            model,
            zero_based_layer);

    const auto per_token =
        component ==
                SessionStateComponent::attention_kv
            ? base_kv_bytes_per_token(model) /
                  model.full_attention_layers
            : provisional_indexer_bytes_per_token(model) /
                  model.full_attention_layers;
    const auto per_layer =
        per_token *
        layout.context_tokens;
    const auto offset =
        static_cast<std::uint64_t>(
            ordinal) *
        per_layer;

    if (offset > result.bytes ||
        per_layer >
            result.bytes - offset) {
        throw std::invalid_argument(
            "indexed-attention layer state slice exceeds component allocation");
    }
    if (offset >
        static_cast<std::uint64_t>(
            std::numeric_limits<
                std::uintptr_t>::max() -
            result.address)) {
        throw std::overflow_error(
            "indexed-attention layer state slice address overflows uintptr_t");
    }

    result.address +=
        static_cast<std::uintptr_t>(
            offset);
    result.bytes =
        per_layer;
    return result;
}

std::uint32_t gdn_layer_ordinal(
    const ModelContract& model,
    std::uint32_t zero_based_layer) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (zero_based_layer >=
        model.hidden_layers) {
        throw std::out_of_range(
            "GDN layer index exceeds model layer count");
    }

    const auto one_based =
        static_cast<std::uint64_t>(
            zero_based_layer) +
        1ULL;
    if (one_based %
            model.full_attention_interval ==
        0) {
        throw std::invalid_argument(
            "indexed-attention layer has no GDN state");
    }

    const auto indexed_before_or_at =
        one_based /
        model.full_attention_interval;
    const auto ordinal =
        static_cast<std::uint64_t>(
            zero_based_layer) -
        indexed_before_or_at;

    if (ordinal >=
        model.gdn_layers) {
        throw std::logic_error(
            "derived GDN ordinal exceeds configured GDN layers");
    }
    return static_cast<std::uint32_t>(
        ordinal);
}

SessionStateAddress resolve_gdn_layer_state(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory,
    const ModelContract& model,
    std::uint32_t zero_based_layer,
    SessionStateComponent component) {
    if (component !=
            SessionStateComponent::recurrent_matrix &&
        component !=
            SessionStateComponent::gdn_convolution) {
        throw std::invalid_argument(
            "GDN layer state resolver supports only recurrent matrix and GDN convolution");
    }

    auto result =
        resolve_session_state(
            layout,
            memory,
            component);
    const auto ordinal =
        gdn_layer_ordinal(
            model,
            zero_based_layer);
    const auto per_layer =
        component ==
                SessionStateComponent::recurrent_matrix
            ? gdn_matrix_state_bytes_per_layer(
                  model)
            : gdn_conv_state_bytes_per_layer(
                  model);

    const auto offset =
        static_cast<std::uint64_t>(
            ordinal) *
        per_layer;
    if (offset > result.bytes ||
        per_layer >
            result.bytes - offset) {
        throw std::invalid_argument(
            "GDN layer state slice exceeds component allocation");
    }
    if (offset >
        static_cast<std::uint64_t>(
            std::numeric_limits<
                std::uintptr_t>::max() -
            result.address)) {
        throw std::overflow_error(
            "GDN layer state slice address overflows uintptr_t");
    }

    result.address +=
        static_cast<std::uintptr_t>(
            offset);
    result.bytes =
        per_layer;
    return result;
}

}  // namespace strix
