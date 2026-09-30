#include "strix/runtime_launchers.hpp"

#include <cstdint>
#include <iostream>
#include <span>
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
    std::uint64_t position_storage = 0;

    strix::SerialSessionStateLayout layout;
    layout.context_tokens = 16;
    layout.host_alignment = alignof(std::uint64_t);
    layout.host_total_bytes =
        sizeof(position_storage);
    layout.layout_identity =
        0x1111ULL;
    layout.allocations.push_back(
        {
            strix::SessionStateComponent::position_history,
            strix::PersistentState::position_history,
            strix::SessionStateStorage::host,
            0,
            sizeof(position_storage),
            false,
        });

    strix::SessionStateMemoryBinding memory;
    memory.layout_identity =
        layout.layout_identity;
    memory.host_base =
        reinterpret_cast<std::uintptr_t>(
            &position_storage);
    memory.host_bytes =
        sizeof(position_storage);
    memory.host_allocation_generation = 1;

    strix::ExecutionRuntimeResources resources;
    resources.session_state_layout =
        &layout;
    resources.session_state_memory =
        &memory;

    strix::ExecutionDispatchContext context;
    context.step_id = 7;
    context.op =
        strix::SerialOp::position_prepare;
    context.position = 12345;
    context.session_state_layout =
        &layout;
    context.session_state_memory =
        &memory;
    context.resources = &resources;

    const auto launcher =
        strix::make_position_prepare_launcher();
    launcher.function(
        context,
        launcher.user_data);

    expect(position_storage == 12345,
           "position launcher writes absolute position to host state");

    auto wrong = context;
    wrong.op =
        strix::SerialOp::token_embedding;
    bool wrong_op_rejected = false;
    try {
        launcher.function(
            wrong,
            launcher.user_data);
    } catch (const std::invalid_argument&) {
        wrong_op_rejected = true;
    }
    expect(wrong_op_rejected,
           "position launcher rejects wrong operator binding");


    strix::ExecutionLauncherRegistry registry;
    strix::register_default_runtime_launchers(
        registry);
    const auto* registered =
        registry.find_runtime_launcher(
            strix::SerialOp::position_prepare);
    expect(registered != nullptr &&
               registered->implementation_identity ==
                   strix::position_prepare_launcher_identity &&
               registered->function != nullptr,
           "default runtime launcher registration exposes position prepare");

    bool duplicate_default_rejected = false;
    try {
        strix::register_default_runtime_launchers(
            registry);
    } catch (const std::invalid_argument&) {
        duplicate_default_rejected = true;
    }
    expect(duplicate_default_rejected,
           "default runtime launcher registration rejects duplicate installation");


    {
        std::uint64_t verify_position_storage = 0;

        auto verify_layout = layout;
        auto verify_memory = memory;
        verify_memory.host_base =
            reinterpret_cast<std::uintptr_t>(
                &verify_position_storage);

        strix::ExecutionRuntimeResources verify_resources;
        verify_resources.session_state_layout =
            &verify_layout;
        verify_resources.session_state_memory =
            &verify_memory;

        const std::uint32_t tokens[] = {
            11, 12, 13, 14,
        };

        strix::ExecutionDispatchContext verify_context;
        verify_context.step_id = 7;
        verify_context.op =
            strix::SerialOp::position_prepare;
        verify_context.logical_rows = 4;
        verify_context.input_tokens =
            std::span<const std::uint32_t>(
                tokens, 4);
        verify_context.position = 100;
        verify_context.session_state_layout =
            &verify_layout;
        verify_context.session_state_memory =
            &verify_memory;
        verify_context.resources =
            &verify_resources;

        const auto verify_launcher =
            strix::make_position_verify_window_launcher();
        verify_launcher.function(
            verify_context,
            verify_launcher.user_data);

        expect(verify_position_storage == 103,
               "verify position launcher writes final processed candidate position");

        strix::ExecutionLauncherRegistry verify_registry;
        strix::register_verify_runtime_launchers(
            verify_registry);
        const auto* registered_verify =
            verify_registry.find_runtime_launcher(
                strix::SerialOp::position_prepare);
        expect(registered_verify != nullptr &&
                   registered_verify->implementation_identity ==
                       strix::position_verify_window_launcher_identity,
               "verify runtime registry installs verify-specific position launcher");

        auto bad_verify = verify_context;
        bad_verify.input_tokens =
            std::span<const std::uint32_t>(
                tokens, 3);
        bool bad_verify_rejected = false;
        try {
            verify_launcher.function(
                bad_verify,
                verify_launcher.user_data);
        } catch (const std::invalid_argument&) {
            bad_verify_rejected = true;
        }
        expect(bad_verify_rejected,
               "verify position launcher rejects token-span mismatch");
    }

    if (failures == 0) {
        std::cout << "runtime launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
