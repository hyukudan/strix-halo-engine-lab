#include "strix/serial_prefix_lifecycle.hpp"

#include <cstdint>
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

    if (failures == 0) {
        std::cout
            << "serial prefix lifecycle tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
