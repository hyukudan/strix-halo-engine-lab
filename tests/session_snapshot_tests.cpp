#include "strix/session_snapshot.hpp"

#include "strix/memory_plan.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest layout_request;
    layout_request.context_tokens = 32'768;
    const auto layout =
        strix::build_serial_session_state_layout(
            model,
            layout_request);

    strix::SerialSessionSnapshotRequest request;
    request.compiled_execution_identity =
        0x1234ULL;
    request.runtime_compatibility_identity =
        0x12345678ULL;
    request.next_position = 128;
    request.committed_tokens = 128;
    request.state.valid = all_state();
    for (auto& generation :
         request.state.generations) {
        generation = 7;
    }

    const auto snapshot =
        strix::build_serial_session_snapshot_plan(
            model,
            layout,
            request);

    expect(snapshot.plan_identity != 0,
           "snapshot plan has non-zero identity");
    expect(snapshot.alignment == request.alignment,
           "snapshot preserves requested compact-payload alignment");
    expect(snapshot.ranges.size() == 7,
           "snapshot contains every valid serial state component");

    const auto* kv =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::attention_kv);
    expect(kv != nullptr,
           "KV fixture allocation exists");

    std::uint64_t snapshotted_kv = 0;
    std::uint64_t snapshotted_sparse = 0;
    for (const auto& range :
         snapshot.ranges) {
        if (range.component ==
            strix::SessionStateComponent::attention_kv) {
            snapshotted_kv = range.bytes;
        }
        if (range.component ==
            strix::SessionStateComponent::sparse_index) {
            snapshotted_sparse = range.bytes;
        }
    }

    expect(kv != nullptr && snapshotted_kv == kv->bytes,
           "snapshot preserves all context-strided KV layers and inactive capacity");
    const auto* sparse = strix::find_session_state_allocation(
        layout, strix::SessionStateComponent::sparse_index);
    expect(sparse != nullptr && snapshotted_sparse == sparse->bytes,
           "snapshot preserves all context-strided sparse layers and inactive capacity");
    auto shorter = request;
    shorter.next_position = 1;
    shorter.committed_tokens = 1;
    const auto short_snapshot = strix::build_serial_session_snapshot_plan(model, layout, shorter);
    expect(short_snapshot.device_bytes == snapshot.device_bytes &&
               short_snapshot.plan_identity != snapshot.plan_identity,
           "short prefixes preserve full capacity but retain distinct session identities");

    auto no_sparse_layout_request =
        layout_request;
    no_sparse_layout_request
        .include_sparse_index_candidate = false;
    const auto no_sparse_layout =
        strix::build_serial_session_state_layout(
            model,
            no_sparse_layout_request);

    bool missing_sparse_rejected = false;
    try {
        static_cast<void>(
            strix::build_serial_session_snapshot_plan(
                model,
                no_sparse_layout,
                request));
    } catch (const std::invalid_argument&) {
        missing_sparse_rejected = true;
    }
    expect(missing_sparse_rejected,
           "valid sparse flag cannot be snapshotted without sparse allocation");

    auto empty_request = request;
    empty_request.next_position = 0;
    empty_request.committed_tokens = 0;
    empty_request.state.valid =
        strix::PersistentState::none;
    empty_request.state.generations.fill(0);
    const auto empty =
        strix::build_serial_session_snapshot_plan(
            model,
            layout,
            empty_request);
    expect(empty.ranges.empty() &&
               empty.device_bytes == 0 &&
               empty.host_bytes == 0,
           "empty fresh session snapshot stores no state payload");

    auto too_far = request;
    too_far.next_position =
        layout.context_tokens + 1;
    bool capacity_rejected = false;
    try {
        static_cast<void>(
            strix::build_serial_session_snapshot_plan(
                model,
                layout,
                too_far));
    } catch (const std::out_of_range&) {
        capacity_rejected = true;
    }
    expect(capacity_rejected,
           "snapshot rejects prefix beyond allocated context");

    auto vk_request = layout_request; vk_request.recurrent_storage = strix::GdnRecurrentStorage::value_key;
    const auto vk_layout = strix::build_serial_session_state_layout(model, vk_request);
    const auto vk_snapshot = strix::build_serial_session_snapshot_plan(model, vk_layout, request);
    expect(vk_snapshot.recurrent_storage == strix::GdnRecurrentStorage::value_key &&
        vk_snapshot.plan_identity != snapshot.plan_identity && vk_snapshot.device_bytes == snapshot.device_bytes &&
        vk_snapshot.host_bytes == snapshot.host_bytes,
        "snapshot versions physical matrix storage while retaining exact payload extents");
    auto forged_layout = layout; forged_layout.recurrent_storage = strix::GdnRecurrentStorage::value_key;
    bool storage_rejected = false;
    try { static_cast<void>(strix::build_serial_session_snapshot_plan(model, forged_layout, request)); }
    catch (const std::invalid_argument&) { storage_rejected = true; }
    expect(storage_rejected, "snapshot rejects forged storage enum with an incumbent layout identity");

    for (const auto occupied : {0ULL, 1ULL, 3ULL, 4ULL, 5ULL, 32768ULL}) {
        auto compact_request = request;
        compact_request.next_position = occupied;
        compact_request.committed_tokens = occupied;
        compact_request.policy = strix::SessionSnapshotPolicy::occupied_kv_v1;
        const auto compact = strix::build_serial_session_snapshot_plan(model, layout, compact_request);
        strix::validate_serial_session_snapshot_plan(model, layout, compact);
        const auto row = strix::base_kv_bytes_per_token(model) / model.full_attention_layers;
        std::uint64_t layers = 0;
        for (const auto& range : compact.ranges) {
            const auto* allocation = strix::find_session_state_allocation(layout, range.component);
            if (range.component == strix::SessionStateComponent::attention_kv) {
                expect(range.source_offset == allocation->offset + layers * layout.context_tokens * row &&
                    range.bytes == occupied * row, "occupied KV preserves each capacity-strided layer prefix");
                ++layers;
            } else expect(range.source_offset == allocation->offset && range.bytes == allocation->bytes,
                "occupied KV leaves sparse, recurrent, convolution and host histories complete");
        }
        expect(layers == (occupied ? model.full_attention_layers : 0ULL) &&
            compact.plan_identity != snapshot.plan_identity, "compact format has closed range census and distinct identity");
        if (occupied) {
            for (unsigned fault = 0; fault < 6U; ++fault) {
                auto bad = compact;
                if (fault == 0) bad.ranges.erase(bad.ranges.begin() + model.full_attention_layers - 1U);
                if (fault == 1) ++bad.ranges[1].source_offset;
                if (fault == 2) bad.ranges[1] = bad.ranges[0];
                if (fault == 3) ++bad.ranges.back().bytes;
                if (fault == 4) bad.policy = static_cast<strix::SessionSnapshotPolicy>(99);
                if (fault == 5) ++bad.next_position;
                bool rejected = false;
                try { strix::validate_serial_session_snapshot_plan(model, layout, bad); }
                catch (const std::exception&) { rejected = true; }
                expect(rejected, "compact snapshot rejects missing layer, forged stride/duplicate/payload/policy/frontier");
            }
        }
    }

    if (failures == 0) {
        std::cout << "session snapshot tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
