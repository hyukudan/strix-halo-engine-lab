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

    expect(snapshotted_kv ==
               strix::base_kv_bytes_per_token(model) *
                   request.next_position,
           "snapshot truncates KV to valid prefix");
    expect(kv != nullptr &&
               snapshotted_kv < kv->bytes,
           "snapshot does not copy unused KV capacity");
    expect(snapshotted_sparse ==
               strix::provisional_indexer_bytes_per_token(model) *
                   request.next_position,
           "snapshot truncates provisional sparse state to valid prefix");
    expect(snapshot.device_bytes <
               layout.device_total_bytes,
           "compact snapshot is smaller than full allocated device state");

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

    if (failures == 0) {
        std::cout << "session snapshot tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
