#include "strix/session_snapshot_transfer.hpp"

#include "strix/model_contract.hpp"
#include "strix/indexed_attention_state.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
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

class MemcpyBackend
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

// A batch backend observes completion before the caller is allowed to
// publish restore metadata; a completion error is as destructive as a copy error.
class BatchBackend final : public MemcpyBackend {
public:
    strix::SerialSessionContext* session = nullptr;
    std::uint64_t expected_position = 0;
    bool completion_failure = false;
    unsigned captures = 0, restores = 0;
    std::size_t ranges = 0;
    void capture_batch(std::span<const strix::SessionSnapshotTransfer> transfers) override {
        ++captures; ranges=transfers.size();
        strix::SessionSnapshotTransferBackend::capture_batch(transfers);
        if(completion_failure) throw std::runtime_error("injected capture completion failure");
    }
    void restore_batch(std::span<const strix::SessionSnapshotTransfer> transfers) override {
        ++restores; ranges=transfers.size();
        expect(session && session->next_position==expected_position,
            "restore metadata remains at original frontier before batch completion");
        strix::SessionSnapshotTransferBackend::restore_batch(transfers);
        expect(session && session->next_position==expected_position,
            "restore metadata is not published after individual copies");
        if(completion_failure) throw std::runtime_error("injected restore completion failure");
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

void occupied_kv_roundtrip() {
    using namespace strix;
    auto model = ModelContract::qwen38_flash_next();
    model.gdn_key_heads = model.gdn_value_heads = 1;
    model.gdn_key_head_dim = model.gdn_value_head_dim = 2;
    const auto layout = build_serial_session_state_layout(model, {8, 64, 64, true});
    std::vector<std::uint8_t> device(static_cast<std::size_t>(layout.device_total_bytes), 0x35U);
    std::vector<std::uint8_t> host(static_cast<std::size_t>(layout.host_total_bytes), 0x47U);
    const auto* kv = find_session_state_allocation(layout, SessionStateComponent::attention_kv);
    const auto row = build_indexed_attention_state_layout(model).kv_row_bytes;
    for (std::uint64_t layer = 0; layer < model.full_attention_layers; ++layer)
        for (std::uint64_t token = 0; token < layout.context_tokens; ++token)
            for (std::uint64_t byte = 0; byte < row; ++byte)
                device[kv->offset + (layer * layout.context_tokens + token) * row + byte] =
                    static_cast<std::uint8_t>(layer * 23U + token * 7U + byte);
    const auto original = device;
    const auto original_host = host;
    const SessionStateMemoryBinding memory{layout.layout_identity,
        reinterpret_cast<std::uintptr_t>(device.data()), device.size(), 17,
        reinterpret_cast<std::uintptr_t>(host.data()), host.size(), 18};
    for (auto occupied : {0ULL, 1ULL, 3ULL, 4ULL, 5ULL, 8ULL}) {
        std::copy(original.begin(), original.end(), device.begin());
        std::copy(original_host.begin(), original_host.end(), host.begin());
        SerialSessionSnapshotRequest request;
        request.compiled_execution_identity = 0x1234; request.runtime_compatibility_identity = 0x5678;
        request.next_position = request.committed_tokens = occupied;
        request.state.valid = all_state(); request.state.generations.fill(9);
        request.policy = SessionSnapshotPolicy::occupied_kv_v1;
        const auto plan = build_serial_session_snapshot_plan(model, layout, request);
        const auto restore = build_serial_session_restore_plan(model, layout, 0x5678, plan);
        std::vector<std::uint8_t> saved_device(static_cast<std::size_t>(plan.device_bytes));
        std::vector<std::uint8_t> saved_host(static_cast<std::size_t>(plan.host_bytes));
        const SessionSnapshotMemoryBinding saved{plan.plan_identity,
            reinterpret_cast<std::uintptr_t>(saved_device.data()), saved_device.size(),
            reinterpret_cast<std::uintptr_t>(saved_host.data()), saved_host.size(), 19};
        BatchBackend backend;
        SerialSessionContext session;
        session.execution_identity = 0x1234; session.runtime_compatibility_identity = 0x5678;
        session.state_layout_identity = layout.layout_identity;
        session.next_position = session.committed_tokens = 8;
        backend.session = &session; backend.expected_position = 8;
        capture_serial_session_snapshot(plan, layout, memory, saved, backend, &model);
        expect(backend.captures == 1, "occupied KV submits exactly one capture batch");
        // Each copied KV source range stops before its own inactive tail.
        for (const auto& range : plan.ranges) if (range.component == SessionStateComponent::attention_kv)
            expect(range.bytes == occupied * row && (range.source_offset - kv->offset) %
                (layout.context_tokens * row) == 0, "no inactive KV/PAD byte enters the snapshot payload");
        std::fill(device.begin(), device.end(), 0xd3U);
        std::fill(host.begin(), host.end(), 0xd3U);
        restore_serial_session_snapshot(session, restore, layout, memory, saved, backend, &model);
        expect(session.next_position == occupied && session.state.generations == request.state.generations &&
            host == original_host && backend.restores == 1, "compact restore publishes saved consumed frontier only after completion");
        for (const auto& allocation : layout.allocations) if (allocation.storage == SessionStateStorage::device) {
            for (std::uint64_t i = 0; i < allocation.bytes; ++i) {
                const bool active = allocation.component != SessionStateComponent::attention_kv ||
                    (i % (layout.context_tokens * row)) < occupied * row;
                expect(device[allocation.offset + i] == (active ? original[allocation.offset + i] : 0xd3U),
                    "all occupied layers and non-KV state restore; inactive KV remains explicitly unspecified");
            }
        }
        if (occupied < layout.context_tokens) {
            auto reference = original;
            // Logical continuation writes the complete new row before any read.
            for (std::uint64_t layer = 0; layer < model.full_attention_layers; ++layer) {
                const auto first = kv->offset + (layer * layout.context_tokens + occupied) * row;
                std::fill_n(device.begin() + first, row, 0x29U);
                std::fill_n(reference.begin() + first, row, 0x29U);
                const auto base = kv->offset + layer * layout.context_tokens * row;
                expect(std::equal(device.begin() + base, device.begin() + base + (occupied + 1U) * row,
                    reference.begin() + base), "continuation reads only restored prefix plus fully overwritten current KV row");
            }
        }
        for (unsigned fault = 0; fault < 12U; ++fault) {
            auto bad = plan;
            if (fault == 0) ++bad.ranges.back().source_offset;
            if (fault == 1) bad.ranges.pop_back();
            if (fault == 2) bad.ranges.back().destination_offset = UINT64_MAX;
            if (fault == 3) bad.ranges.back().bytes = UINT64_MAX;
            if (fault == 4) bad.policy = static_cast<SessionSnapshotPolicy>(88);
            if (fault == 5) ++bad.state.generations[0];
            if (fault == 6) bad.ranges.back().storage = static_cast<SessionStateStorage>(88);
            if (fault == 10) bad.policy = SessionSnapshotPolicy::full_components;
            if (fault == 11 && occupied) ++bad.ranges[model.full_attention_layers - 1U].source_offset;
            if (fault == 11 && !occupied) ++bad.ranges[0].source_offset;
            auto truncated = saved;
            if (fault == 7) --truncated.device_section_bytes;
            if (fault == 9) truncated.allocation_generation = 0;
            MemcpyBackend untouched;
            bool rejected = false;
            try { capture_serial_session_snapshot(bad, layout, memory, truncated, untouched, fault == 8U ? nullptr : &model); }
            catch (const std::exception&) { rejected = true; }
            expect(rejected && untouched.calls == 0, "compact malformed late range/header/truncation rejects before any copy");
        }
        auto forged = restore; ++forged.ranges.back().destination_offset;
        MemcpyBackend untouched;
        bool rejected = false;
        try { restore_serial_session_snapshot(session, forged, layout, memory, saved, untouched, &model); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && untouched.calls == 0, "compact restore validates interior destinations before any copy");
        forged = restore; forged.policy = SessionSnapshotPolicy::full_components;
        rejected = false;
        try { restore_serial_session_snapshot(session, forged, layout, memory, saved, untouched); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && untouched.calls == 0, "compact-to-full forged restore policy cannot bypass canonical admission");
        session.next_position = session.committed_tokens = 8;
        session.health = SerialSessionHealth::ready;
        backend.completion_failure = true;
        rejected = false;
        try { restore_serial_session_snapshot(session, restore, layout, memory, saved, backend, &model); }
        catch (const std::runtime_error&) { rejected = true; }
        expect(rejected && session.health == SerialSessionHealth::poisoned && session.arithmetic_lineage.tainted &&
            session.next_position == 8, "compact failed batch completion poisons without publishing rollback metadata");
    }
}

}  // namespace

int main() {
    occupied_kv_roundtrip();
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

    const auto original_device_state = device_state;
    const auto original_host_state = host_state;

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
    request.compiled_execution_identity =
        0x1234ULL;
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

    // The first range is valid in every fixture. Reject a malformed later
    // range before even the first backend call, including asynchronous backends.
    expect(snapshot.ranges.size() > 1 && restore.ranges.size() > 1,
           "preflight fixture has a valid first range and malformed later range");
    const auto captured_device = snapshot_device;
    const auto captured_host = snapshot_host;
    for (unsigned mutation = 0; mutation != 5; ++mutation) {
        auto malformed = snapshot;
        auto& later = malformed.ranges.back();
        if (mutation == 0) ++later.source_offset;
        if (mutation == 1) later.bytes = std::numeric_limits<std::uint64_t>::max();
        if (mutation == 2) later.destination_offset = std::numeric_limits<std::uint64_t>::max();
        if (mutation == 3) later.storage = static_cast<strix::SessionStateStorage>(99);
        if (mutation == 4) later.component = static_cast<strix::SessionStateComponent>(99);
        MemcpyBackend preflight_backend;
        bool rejected = false;
        try {
            strix::capture_serial_session_snapshot(malformed, layout, state_memory,
                                                   snapshot_memory, preflight_backend);
        } catch (const std::exception&) { rejected = true; }
        expect(rejected && preflight_backend.calls == 0,
               "malformed later capture range is rejected before any copy submission");
        expect(snapshot_device == captured_device && snapshot_host == captured_host &&
                   device_state == original_device_state && host_state == original_host_state,
               "capture preflight rejection changes neither payload nor source allocations");
    }

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

    session.next_position = 3;
    session.committed_tokens = 3;
    session.state = request.state;
    session.arithmetic_lineage.history_identity = 42;
    session.arithmetic_lineage.tainted = false;
    const auto prior_session = session;
    for (unsigned mutation = 0; mutation != 5; ++mutation) {
        auto malformed = restore;
        auto& later = malformed.ranges.back();
        if (mutation == 0) ++later.destination_offset;
        if (mutation == 1) later.bytes = std::numeric_limits<std::uint64_t>::max();
        if (mutation == 2) later.snapshot_offset = std::numeric_limits<std::uint64_t>::max();
        if (mutation == 3) later.storage = static_cast<strix::SessionStateStorage>(99);
        if (mutation == 4) later.component = static_cast<strix::SessionStateComponent>(99);
        MemcpyBackend preflight_backend;
        bool rejected = false;
        try {
            strix::restore_serial_session_snapshot(session, malformed, layout, state_memory,
                                                   snapshot_memory, preflight_backend);
        } catch (const std::exception&) { rejected = true; }
        expect(rejected && preflight_backend.calls == 0,
               "malformed later restore range is rejected before any copy submission");
        expect(std::all_of(device_state.begin(), device_state.end(), [](auto byte) { return byte == 0; }) &&
                   std::all_of(host_state.begin(), host_state.end(), [](auto byte) { return byte == 0; }) &&
                   snapshot_device == captured_device && snapshot_host == captured_host,
               "restore preflight rejection changes neither destination nor captured payload");
        expect(session.health == prior_session.health && session.next_position == prior_session.next_position &&
                   session.committed_tokens == prior_session.committed_tokens && session.state.generations == prior_session.state.generations &&
                   session.state.valid == prior_session.state.valid &&
                   session.arithmetic_lineage.history_identity == prior_session.arithmetic_lineage.history_identity &&
                   session.arithmetic_lineage.tainted == prior_session.arithmetic_lineage.tainted,
               "restore preflight rejection preserves existing session health, metadata and lineage");
    }

    auto wrong_snapshot_memory =
        snapshot_memory;
    ++wrong_snapshot_memory.snapshot_plan_identity;
    bool wrong_snapshot_identity_rejected = false;
    try {
        strix::restore_serial_session_snapshot(
            session,
            restore,
            layout,
            state_memory,
            wrong_snapshot_memory,
            backend);
    } catch (const std::invalid_argument&) {
        wrong_snapshot_identity_rejected = true;
    }
    expect(wrong_snapshot_identity_rejected,
           "wrong snapshot payload identity is rejected");

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

    // Independent component comparison catches the former contiguous-prefix
    // bug: next_position=2 with context=4 must still restore later layers and
    // all previously captured inactive capacity, not just layer zero's prefix.
    for (const auto& allocation : layout.allocations) {
        const auto* actual = allocation.storage == strix::SessionStateStorage::device
            ? device_state.data() : host_state.data();
        const auto* original = allocation.storage == strix::SessionStateStorage::device
            ? original_device_state.data() : original_host_state.data();
        expect(std::memcmp(actual + allocation.offset, original + allocation.offset,
                   static_cast<std::size_t>(allocation.bytes)) == 0,
               "partial-prefix snapshot restores every byte of every state component");
    }

    session.health =
        strix::SerialSessionHealth::ready;
    {
        const auto before_device = device_state, before_host = host_state;
        const auto before_snapshot_device = snapshot_device, before_snapshot_host = snapshot_host;
        auto forged_snapshot = snapshot;
        forged_snapshot.recurrent_storage = strix::GdnRecurrentStorage::value_key;
        MemcpyBackend untouched;
        bool rejected = false;
        try { strix::capture_serial_session_snapshot(forged_snapshot, layout, state_memory, snapshot_memory, untouched); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && untouched.calls == 0 && device_state == before_device && host_state == before_host &&
            snapshot_device == before_snapshot_device && snapshot_host == before_snapshot_host,
            "capture rejects same-ID forged physical storage before copying any owner bytes");
        auto forged_restore = restore;
        forged_restore.recurrent_storage = strix::GdnRecurrentStorage::value_key;
        rejected = false;
        try { strix::restore_serial_session_snapshot(session, forged_restore, layout, state_memory, snapshot_memory, untouched); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && untouched.calls == 0 && device_state == before_device && host_state == before_host &&
            session.health == strix::SerialSessionHealth::ready && session.next_position == restore.next_position,
            "restore rejects forged physical storage before payload or session publication");
    }
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
                   strix::SerialSessionHealth::poisoned && session.arithmetic_lineage.tainted &&
                   failing.calls == 2,
           "partial restore failure leaves session poisoned");

    {
        BatchBackend batched;
        batched.session=&session; batched.expected_position=3;
        strix::capture_serial_session_snapshot(snapshot,layout,state_memory,snapshot_memory,batched);
        expect(batched.captures==1 && batched.ranges==snapshot.ranges.size(),
            "snapshot invokes one complete admitted capture batch");
        session.next_position=3; session.committed_tokens=3;
        session.health=strix::SerialSessionHealth::ready;
        strix::restore_serial_session_snapshot(session,restore,layout,state_memory,snapshot_memory,batched);
        expect(batched.restores==1 && batched.ranges==restore.ranges.size() &&
            session.next_position==restore.next_position && session.committed_tokens==restore.committed_tokens &&
            session.health==strix::SerialSessionHealth::ready,
            "all ranges complete in one batch before restore frontier publication");
        // The backend may fail its single final fence after ALL copies submitted.
        session.next_position=3; session.committed_tokens=3;
        batched.completion_failure=true;
        bool rejected=false;
        try {strix::restore_serial_session_snapshot(session,restore,layout,state_memory,snapshot_memory,batched);}
        catch(const std::runtime_error&) {rejected=true;}
        expect(rejected && session.health==strix::SerialSessionHealth::poisoned &&
            session.arithmetic_lineage.tainted && session.next_position==3 && session.committed_tokens==3,
            "batch completion failure poisons without publishing saved metadata");
        auto malformed=restore; ++malformed.ranges.back().destination_offset;
        const auto submitted=batched.restores;
        rejected=false;
        try {strix::restore_serial_session_snapshot(session,malformed,layout,state_memory,snapshot_memory,batched);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && batched.restores==submitted,
            "invalid last range is rejected before entering a batched backend");
    }

    if (failures == 0) {
        std::cout << "session snapshot transfer tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
