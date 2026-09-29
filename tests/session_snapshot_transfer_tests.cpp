#include "strix/session_snapshot_transfer.hpp"

#include "strix/model_contract.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class MemcpyBackend final
    : public strix::SessionSnapshotTransferBackend {
public:
    int fail_after = -1;
    int calls = 0;

    void capture_device(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        copy(source, destination, bytes);
    }

    void capture_host(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        copy(source, destination, bytes);
    }

    void restore_device(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        copy(source, destination, bytes);
    }

    void restore_host(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        copy(source, destination, bytes);
    }

private:
    void copy(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) {
        if (fail_after >= 0 &&
            calls == fail_after) {
            ++calls;
            throw std::runtime_error(
                "injected snapshot copy failure");
        }
        ++calls;
        std::memcpy(
            reinterpret_cast<void*>(
                destination),
            reinterpret_cast<const void*>(
                source),
            static_cast<std::size_t>(
                bytes));
    }
};

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
    layout_request.context_tokens = 4;
    layout_request.device_alignment = 64;
    layout_request.host_alignment = 64;
    const auto layout =
        strix::build_serial_session_state_layout(
            model,
            layout_request);

    std::vector<std::uint8_t> device_state(
        static_cast<std::size_t>(
            layout.device_total_bytes));
    std::vector<std::uint8_t> host_state(
        static_cast<std::size_t>(
            layout.host_total_bytes));

    for (std::size_t index = 0;
         index < device_state.size();
         ++index) {
        device_state[index] =
            static_cast<std::uint8_t>(
                (index * 17U + 3U) &
                0xffU);
    }
    for (std::size_t index = 0;
         index < host_state.size();
         ++index) {
        host_state[index] =
            static_cast<std::uint8_t>(
                (index * 7U + 1U) &
                0xffU);
    }

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        layout.layout_identity;
    state_memory.device_base =
        reinterpret_cast<std::uintptr_t>(
            device_state.data());
    state_memory.device_bytes =
        device_state.size();
    state_memory.device_allocation_generation = 1;
    state_memory.host_base =
        reinterpret_cast<std::uintptr_t>(
            host_state.data());
    state_memory.host_bytes =
        host_state.size();
    state_memory.host_allocation_generation = 2;

    strix::SerialSessionSnapshotRequest request;
    request.runtime_compatibility_identity =
        0xabcdefULL;
    request.next_position = 2;
    request.committed_tokens = 2;
    request.state.valid = all_state();
    for (auto& generation :
         request.state.generations) {
        generation = 5;
    }

    const auto snapshot =
        strix::build_serial_session_snapshot_plan(
            model,
            layout,
            request);
    const auto restore =
        strix::build_serial_session_restore_plan(
            model,
            layout,
            request.runtime_compatibility_identity,
            snapshot);

    std::vector<std::uint8_t> snapshot_device(
        static_cast<std::size_t>(
            snapshot.device_bytes));
    std::vector<std::uint8_t> snapshot_host(
        static_cast<std::size_t>(
            snapshot.host_bytes));

    strix::SessionSnapshotMemoryBinding snapshot_memory;
    snapshot_memory.snapshot_plan_identity =
        snapshot.plan_identity;
    snapshot_memory.device_section_base =
        reinterpret_cast<std::uintptr_t>(
            snapshot_device.data());
    snapshot_memory.device_section_bytes =
        snapshot_device.size();
    snapshot_memory.host_section_base =
        reinterpret_cast<std::uintptr_t>(
            snapshot_host.data());
    snapshot_memory.host_section_bytes =
        snapshot_host.size();
    snapshot_memory.allocation_generation = 3;

    MemcpyBackend backend;
    strix::capture_serial_session_snapshot(
        snapshot,
        layout,
        state_memory,
        snapshot_memory,
        backend);

    std::fill(
        device_state.begin(),
        device_state.end(),
        0);
    std::fill(
        host_state.begin(),
        host_state.end(),
        0);

    strix::SerialSessionContext session;
    session.execution_identity =
        0x1234ULL;
    session.runtime_compatibility_identity =
        request.runtime_compatibility_identity;
    session.state_layout_identity =
        layout.layout_identity;
    session.health =
        strix::SerialSessionHealth::poisoned;

    strix::restore_serial_session_snapshot(
        session,
        restore,
        layout,
        state_memory,
        snapshot_memory,
        backend);

    expect(session.health ==
               strix::SerialSessionHealth::ready &&
               session.next_position == 2 &&
               session.committed_tokens == 2 &&
               session.state.generations ==
                   request.state.generations,
           "successful restore publishes snapshot metadata and recovers session");

    bool prefix_restored = true;
    for (const auto& range :
         restore.ranges) {
        const auto base =
            range.storage ==
                    strix::SessionStateStorage::device
                ? device_state.data()
                : host_state.data();
        const auto snapshot_base =
            range.storage ==
                    strix::SessionStateStorage::device
                ? snapshot_device.data()
                : snapshot_host.data();

        if (std::memcmp(
                base + range.destination_offset,
                snapshot_base + range.snapshot_offset,
                static_cast<std::size_t>(
                    range.bytes)) != 0) {
            prefix_restored = false;
            break;
        }
    }
    expect(prefix_restored,
           "restore copies every compact snapshot range to state layout");

    session.health =
        strix::SerialSessionHealth::poisoned;
    MemcpyBackend failing;
    failing.fail_after = 1;
    bool failure_seen = false;
    try {
        strix::restore_serial_session_snapshot(
            session,
            restore,
            layout,
            state_memory,
            snapshot_memory,
            failing);
    } catch (const std::runtime_error&) {
        failure_seen = true;
    }
    expect(failure_seen &&
               session.health ==
                   strix::SerialSessionHealth::poisoned,
           "partial restore failure leaves session poisoned");

    if (failures == 0) {
        std::cout << "session snapshot transfer tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
