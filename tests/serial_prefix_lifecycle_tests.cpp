#include "strix/serial_prefix_lifecycle.hpp"

#include <algorithm>
#include <cstdint>
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

strix::PersistentState all_state() {
    using strix::PersistentState;
    return
        PersistentState::attention_kv |
        PersistentState::recurrent_matrix |
        PersistentState::convolution_history |
        PersistentState::sparse_index |
        PersistentState::ngram_history |
        PersistentState::position_history;
}

class MemcpyTransferBackend final
    : public strix::SessionSnapshotTransferBackend {
public:
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
    static void copy(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) {
        std::memcpy(
            reinterpret_cast<void*>(destination),
            reinterpret_cast<const void*>(source),
            static_cast<std::size_t>(bytes));
    }
};

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest
        layout_request;
    layout_request.context_tokens = 128;
    const auto layout =
        strix::build_serial_session_state_layout(
            model,
            layout_request);

    const std::vector<std::uint32_t> tokens{
        11, 22, 33,
    };

    strix::SerialSessionSnapshotRequest
        request;
    request.compiled_execution_identity =
        0x1111ULL;
    request.runtime_compatibility_identity =
        0x2222ULL;
    request.next_position =
        tokens.size();
    request.committed_tokens =
        tokens.size();
    request.state.valid =
        all_state();
    for (auto& generation :
         request.state.generations) {
        generation = 7;
    }

    const auto snapshot =
        strix::build_serial_session_snapshot_plan(
            model,
            layout,
            request);
    const auto prefix =
        strix::make_serial_prefix_snapshot(
            tokens,
            snapshot);

    expect(
        prefix.prefix_identity != 0 &&
            prefix.prefix_tokens == tokens,
        "prefix lifecycle captures exact token sequence and stable identity");

    expect(
        strix::serial_prefix_snapshot_matches(
            prefix,
            tokens,
            request.compiled_execution_identity,
            request.runtime_compatibility_identity,
            layout.layout_identity),
        "exact prefix/runtime identities match cached prefix");

    auto changed_tokens =
        tokens;
    changed_tokens.back() = 34;
    expect(
        !strix::serial_prefix_snapshot_matches(
            prefix,
            changed_tokens,
            request.compiled_execution_identity,
            request.runtime_compatibility_identity,
            layout.layout_identity),
        "different prefix token sequence cannot reuse snapshot");

    expect(
        !strix::serial_prefix_snapshot_matches(
            prefix,
            tokens,
            request.compiled_execution_identity + 1ULL,
            request.runtime_compatibility_identity,
            layout.layout_identity),
        "different compiled execution cannot reuse prefix snapshot");

    expect(
        !strix::serial_prefix_snapshot_matches(
            prefix,
            tokens,
            request.compiled_execution_identity,
            request.runtime_compatibility_identity + 1ULL,
            layout.layout_identity),
        "different runtime compatibility cannot reuse prefix snapshot");

    const auto restore =
        strix::build_serial_prefix_restore_plan(
            model,
            layout,
            request.compiled_execution_identity,
            request.runtime_compatibility_identity,
            tokens,
            prefix);
    expect(
        restore.snapshot_plan_identity ==
            snapshot.plan_identity &&
            restore.next_position ==
                tokens.size() &&
            restore.committed_tokens ==
                tokens.size(),
        "matching prefix produces restore plan at exact committed position");

    bool wrong_prefix_rejected = false;
    try {
        static_cast<void>(
            strix::build_serial_prefix_restore_plan(
                model,
                layout,
                request.compiled_execution_identity,
                request.runtime_compatibility_identity,
                changed_tokens,
                prefix));
    } catch (const std::invalid_argument&) {
        wrong_prefix_rejected = true;
    }
    expect(
        wrong_prefix_rejected,
        "restore planning rejects non-matching prefix");

    auto malformed =
        snapshot;
    malformed.next_position = 2;
    bool malformed_rejected = false;
    try {
        static_cast<void>(
            strix::make_serial_prefix_snapshot(
                tokens,
                malformed));
    } catch (const std::invalid_argument&) {
        malformed_rejected = true;
    }
    expect(
        malformed_rejected,
        "prefix lifecycle rejects snapshot metadata that does not match prefix length");

    {
        strix::SerialSessionStateLayoutRequest
            physical_layout_request;
        physical_layout_request.context_tokens = 4;
        physical_layout_request.device_alignment = 64;
        physical_layout_request.host_alignment = 64;
        const auto physical_layout =
            strix::build_serial_session_state_layout(
                model,
                physical_layout_request);

        std::vector<std::uint8_t> device_state(
            static_cast<std::size_t>(
                physical_layout.device_total_bytes));
        std::vector<std::uint8_t> host_state(
            static_cast<std::size_t>(
                physical_layout.host_total_bytes));
        for (std::size_t index = 0;
             index < device_state.size();
             ++index) {
            device_state[index] =
                static_cast<std::uint8_t>(
                    (index * 13U + 5U) & 0xffU);
        }
        for (std::size_t index = 0;
             index < host_state.size();
             ++index) {
            host_state[index] =
                static_cast<std::uint8_t>(
                    (index * 3U + 7U) & 0xffU);
        }

        strix::SessionStateMemoryBinding
            state_memory;
        state_memory.layout_identity =
            physical_layout.layout_identity;
        state_memory.device_base =
            reinterpret_cast<std::uintptr_t>(
                device_state.data());
        state_memory.device_bytes =
            device_state.size();
        state_memory.device_allocation_generation = 3;
        state_memory.host_base =
            reinterpret_cast<std::uintptr_t>(
                host_state.data());
        state_memory.host_bytes =
            host_state.size();
        state_memory.host_allocation_generation = 4;

        const std::vector<std::uint32_t>
            physical_tokens{41, 42};

        strix::SerialSessionSnapshotRequest
            physical_request;
        physical_request.compiled_execution_identity =
            0x9001ULL;
        physical_request.runtime_compatibility_identity =
            0x9002ULL;
        physical_request.next_position =
            physical_tokens.size();
        physical_request.committed_tokens =
            physical_tokens.size();
        physical_request.state.valid =
            all_state();
        for (auto& generation :
             physical_request.state.generations) {
            generation = 9;
        }

        const auto physical_plan =
            strix::build_serial_session_snapshot_plan(
                model,
                physical_layout,
                physical_request);
        const auto physical_prefix =
            strix::make_serial_prefix_snapshot(
                physical_tokens,
                physical_plan);

        std::vector<std::uint8_t> snapshot_device(
            static_cast<std::size_t>(
                physical_plan.device_bytes));
        std::vector<std::uint8_t> snapshot_host(
            static_cast<std::size_t>(
                physical_plan.host_bytes));

        strix::SessionSnapshotMemoryBinding
            snapshot_memory;
        snapshot_memory.snapshot_plan_identity =
            physical_plan.plan_identity;
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
        snapshot_memory.allocation_generation = 5;

        strix::SerialSessionContext
            physical_session;
        physical_session.execution_identity =
            physical_request.compiled_execution_identity;
        physical_session.runtime_compatibility_identity =
            physical_request.runtime_compatibility_identity;
        physical_session.state_layout_identity =
            physical_layout.layout_identity;
        physical_session.next_position =
            physical_tokens.size();
        physical_session.committed_tokens =
            physical_tokens.size();
        physical_session.state =
            physical_request.state;

        MemcpyTransferBackend transfer;
        strix::capture_serial_prefix_snapshot(
            physical_prefix,
            physical_session,
            physical_layout,
            state_memory,
            snapshot_memory,
            transfer);

        std::fill(
            device_state.begin(),
            device_state.end(),
            0);
        std::fill(
            host_state.begin(),
            host_state.end(),
            0);
        physical_session.next_position = 0;
        physical_session.committed_tokens = 0;
        physical_session.state = {};
        physical_session.health =
            strix::SerialSessionHealth::poisoned;

        strix::restore_serial_prefix_snapshot(
            physical_session,
            model,
            physical_layout,
            state_memory,
            snapshot_memory,
            transfer,
            physical_tokens,
            physical_prefix);

        expect(
            physical_session.health ==
                strix::SerialSessionHealth::ready &&
            physical_session.next_position ==
                physical_tokens.size() &&
            physical_session.committed_tokens ==
                physical_tokens.size() &&
            physical_session.state.generations ==
                physical_request.state.generations,
            "physical prefix restore republishes exact cached session metadata");

        bool restored_bytes_match = true;
        for (const auto& range :
             physical_plan.ranges) {
            const auto* state_base =
                range.storage ==
                        strix::SessionStateStorage::device
                    ? device_state.data()
                    : host_state.data();
            const auto* snapshot_base =
                range.storage ==
                        strix::SessionStateStorage::device
                    ? snapshot_device.data()
                    : snapshot_host.data();
            if (std::memcmp(
                    state_base + range.source_offset,
                    snapshot_base +
                        range.destination_offset,
                    static_cast<std::size_t>(
                        range.bytes)) != 0) {
                restored_bytes_match = false;
                break;
            }
        }
        expect(
            restored_bytes_match,
            "physical prefix restore copies every cached snapshot range");

        auto wrong_session =
            physical_session;
        ++wrong_session.runtime_compatibility_identity;
        bool wrong_runtime_capture_rejected = false;
        try {
            strix::capture_serial_prefix_snapshot(
                physical_prefix,
                wrong_session,
                physical_layout,
                state_memory,
                snapshot_memory,
                transfer);
        } catch (const std::invalid_argument&) {
            wrong_runtime_capture_rejected = true;
        }
        expect(
            wrong_runtime_capture_rejected,
            "physical prefix capture rejects a foreign runtime identity");
    }

    if (failures == 0) {
        std::cout
            << "serial prefix lifecycle tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
