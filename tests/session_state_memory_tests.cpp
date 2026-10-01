#include "strix/session_state_memory.hpp"

#include "strix/model_contract.hpp"
#include "strix/memory_plan.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest request;
    request.context_tokens = 32'768;
    const auto layout =
        strix::build_serial_session_state_layout(
            model, request);

    strix::SessionStateMemoryBinding memory;
    memory.layout_identity =
        layout.layout_identity;
    memory.device_base =
        static_cast<std::uintptr_t>(
            0x40000000ULL);
    memory.device_bytes =
        layout.device_total_bytes;
    memory.device_allocation_generation = 3;
    memory.host_base =
        static_cast<std::uintptr_t>(
            0x50000000ULL);
    memory.host_bytes =
        layout.host_total_bytes;
    memory.host_allocation_generation = 5;

    strix::validate_session_state_memory_binding(
        layout, memory);

    const auto kv =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::attention_kv);
    expect(kv.storage ==
               strix::SessionStateStorage::device &&
               kv.address >= memory.device_base &&
               kv.allocation_generation == 3,
           "KV resolves into device state pool");

    const auto position =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::position_history);
    expect(position.storage ==
               strix::SessionStateStorage::host &&
               position.address >= memory.host_base &&
               position.allocation_generation == 5,
           "position metadata resolves into host state pool");

    const auto sparse =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::sparse_index);
    expect(!sparse.provisional,
           "resolved sparse state reflects exact persistent raw-index geometry");

    auto wrong = memory;
    ++wrong.layout_identity;
    bool identity_rejected = false;
    try {
        strix::validate_session_state_memory_binding(
            layout, wrong);
    } catch (const std::invalid_argument&) {
        identity_rejected = true;
    }
    expect(identity_rejected,
           "state memory rejects layout identity mismatch");

    auto undersized = memory;
    --undersized.device_bytes;
    bool size_rejected = false;
    try {
        strix::validate_session_state_memory_binding(
            layout, undersized);
    } catch (const std::invalid_argument&) {
        size_rejected = true;
    }
    expect(size_rejected,
           "state memory rejects undersized device pool");

    auto overflow = memory;
    overflow.device_base =
        std::numeric_limits<std::uintptr_t>::max() -
        static_cast<std::uintptr_t>(
            layout.device_total_bytes / 2U);
    bool overflow_rejected = false;
    try {
        strix::validate_session_state_memory_binding(
            layout, overflow);
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    expect(overflow_rejected,
           "state memory rejects device address overflow");


    expect(
        strix::gdn_layer_ordinal(model, 0) == 0 &&
            strix::gdn_layer_ordinal(model, 1) == 1 &&
            strix::gdn_layer_ordinal(model, 2) == 2 &&
            strix::gdn_layer_ordinal(model, 4) == 3,
        "GDN ordinal skips indexed-attention layers");

    bool indexed_rejected = false;
    try {
        static_cast<void>(
            strix::gdn_layer_ordinal(
                model,
                3));
    } catch (const std::invalid_argument&) {
        indexed_rejected = true;
    }
    expect(
        indexed_rejected,
        "indexed-attention layer is rejected by GDN state resolver");

    const auto layer0_matrix =
        strix::resolve_gdn_layer_state(
            layout,
            memory,
            model,
            0,
            strix::SessionStateComponent::recurrent_matrix);
    const auto layer4_matrix =
        strix::resolve_gdn_layer_state(
            layout,
            memory,
            model,
            4,
            strix::SessionStateComponent::recurrent_matrix);
    expect(
        layer0_matrix.bytes ==
            strix::gdn_matrix_state_bytes_per_layer(model) &&
            layer4_matrix.address ==
                layer0_matrix.address +
                3ULL *
                    strix::gdn_matrix_state_bytes_per_layer(model),
        "GDN matrix layer slice uses compact GDN ordinal");

    const auto layer2_conv =
        strix::resolve_gdn_layer_state(
            layout,
            memory,
            model,
            2,
            strix::SessionStateComponent::gdn_convolution);
    expect(
        layer2_conv.bytes ==
            strix::gdn_conv_state_bytes_per_layer(model),
        "GDN convolution slice exposes one layer only");


    expect(
        strix::indexed_attention_layer_ordinal(
            model, 3) == 0 &&
        strix::indexed_attention_layer_ordinal(
            model, 7) == 1 &&
        strix::indexed_attention_layer_ordinal(
            model, 47) == 11,
        "indexed-attention ordinal follows every fourth layer");

    bool linear_rejected = false;
    try {
        static_cast<void>(
            strix::indexed_attention_layer_ordinal(
                model, 2));
    } catch (const std::invalid_argument&) {
        linear_rejected = true;
    }
    expect(
        linear_rejected,
        "linear-attention layer has no indexed state ordinal");

    const auto indexed_kv =
        strix::resolve_indexed_attention_layer_state(
            layout,
            memory,
            model,
            7,
            strix::SessionStateComponent::attention_kv);
    const auto indexed_raw =
        strix::resolve_indexed_attention_layer_state(
            layout,
            memory,
            model,
            7,
            strix::SessionStateComponent::sparse_index);
    const auto kv_per_layer =
        strix::base_kv_bytes_per_token(model) /
        model.full_attention_layers *
        layout.context_tokens;
    const auto raw_per_layer =
        strix::indexer_bytes_per_token(model) /
        model.full_attention_layers *
        layout.context_tokens;
    expect(
        indexed_kv.bytes == kv_per_layer &&
        indexed_raw.bytes == raw_per_layer,
        "indexed layer resolver returns exact per-layer KV and raw-index sizes");

    const auto kv_base =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::attention_kv);
    const auto raw_base =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::sparse_index);
    expect(
        indexed_kv.address ==
            kv_base.address +
                kv_per_layer &&
        indexed_raw.address ==
            raw_base.address +
                raw_per_layer,
        "second indexed layer resolves one per-layer stride into both pools");

    if (failures == 0) {
        std::cout << "session state memory tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
