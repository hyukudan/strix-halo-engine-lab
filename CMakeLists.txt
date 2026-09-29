cmake_minimum_required(VERSION 3.24)

project(strix_engine_lab VERSION 0.1.0 LANGUAGES CXX)

option(STRIX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(STRIX_ENABLE_HIP_RUNTIME "Build optional HIP allocation/runtime support" OFF)

function(strix_target_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(STRIX_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(STRIX_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(strix_core STATIC
    src/compiled_execution.cpp
    src/dispatch_resources.cpp
    src/execution_arena.cpp
    src/execution_binding.cpp
    src/execution_buffers.cpp
    src/execution_io.cpp
    src/execution_memory.cpp
    src/execution_plan.cpp
    src/execution_runtime.cpp
    src/execution_tensor_binding.cpp
    src/execution_tensor_access.cpp
    src/execution_resources.cpp
    src/gguf_reader.cpp
    src/kernel_registry.cpp
    src/mapped_model_file.cpp
    src/iq4_xs.cpp
    src/launch_dispatch.cpp
    src/memory_plan.cpp
    src/model_contract.cpp
    src/model_content_hash.cpp
    src/model_manifest.cpp
    src/model_runtime_binding.cpp
    src/model_tensor_index.cpp
    src/moe_plan.cpp
    src/oracles.cpp
    src/packed_projection.cpp
    src/recurrent_plan.cpp
    src/runtime_allocator.cpp
    src/runtime_launchers.cpp
    src/serial_session.cpp
    src/session_state_layout.cpp
    src/session_state_memory.cpp
    src/session_snapshot.cpp
    src/session_snapshot_transfer.cpp
    src/session_restore.cpp
    src/sparse_lookup.cpp
    src/speculation.cpp
    src/speculation_controller.cpp
    src/platform_probe.cpp
    src/transport.cpp
)
target_include_directories(strix_core PUBLIC include)
target_compile_features(strix_core PUBLIC cxx_std_20)
strix_target_warnings(strix_core)

add_executable(strix-memory-plan apps/memory_plan_main.cpp)
target_link_libraries(strix-memory-plan PRIVATE strix_core)
strix_target_warnings(strix-memory-plan)

add_executable(strix-inspect-gguf apps/inspect_gguf_main.cpp)
target_link_libraries(strix-inspect-gguf PRIVATE strix_core)
strix_target_warnings(strix-inspect-gguf)

add_executable(strix-model-manifest apps/model_manifest_main.cpp)
target_link_libraries(strix-model-manifest PRIVATE strix_core)
strix_target_warnings(strix-model-manifest)

add_executable(strix-platform-probe apps/platform_probe_main.cpp)
target_link_libraries(strix-platform-probe PRIVATE strix_core)
strix_target_warnings(strix-platform-probe)

add_executable(strix-audit-execution-tensors apps/execution_tensor_audit_main.cpp)
target_link_libraries(strix-audit-execution-tensors PRIVATE strix_core)
strix_target_warnings(strix-audit-execution-tensors)

include(CTest)
if(BUILD_TESTING)
    add_executable(strix-core-tests tests/core_tests.cpp)
    target_link_libraries(strix-core-tests PRIVATE strix_core)
    strix_target_warnings(strix-core-tests)
    add_test(NAME strix-core-tests COMMAND strix-core-tests)

    add_executable(strix-manifest-platform-tests tests/manifest_platform_tests.cpp)
    target_link_libraries(strix-manifest-platform-tests PRIVATE strix_core)
    strix_target_warnings(strix-manifest-platform-tests)
    add_test(NAME strix-manifest-platform-tests COMMAND strix-manifest-platform-tests)

    add_executable(strix-model-runtime-binding-tests tests/model_runtime_binding_tests.cpp)
    target_link_libraries(strix-model-runtime-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-runtime-binding-tests)
    add_test(NAME strix-model-runtime-binding-tests COMMAND strix-model-runtime-binding-tests)

    add_executable(strix-model-content-hash-tests tests/model_content_hash_tests.cpp)
    target_link_libraries(strix-model-content-hash-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-content-hash-tests)
    add_test(NAME strix-model-content-hash-tests COMMAND strix-model-content-hash-tests)

    add_executable(strix-mapped-model-file-tests tests/mapped_model_file_tests.cpp)
    target_link_libraries(strix-mapped-model-file-tests PRIVATE strix_core)
    strix_target_warnings(strix-mapped-model-file-tests)
    add_test(NAME strix-mapped-model-file-tests COMMAND strix-mapped-model-file-tests)

    add_executable(strix-model-tensor-index-tests tests/model_tensor_index_tests.cpp)
    target_link_libraries(strix-model-tensor-index-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-tensor-index-tests)
    add_test(NAME strix-model-tensor-index-tests COMMAND strix-model-tensor-index-tests)

    add_executable(strix-transport-tests tests/transport_tests.cpp)
    target_link_libraries(strix-transport-tests PRIVATE strix_core)
    strix_target_warnings(strix-transport-tests)
    add_test(NAME strix-transport-tests COMMAND strix-transport-tests)

    add_executable(strix-kernel-registry-tests tests/kernel_registry_tests.cpp)
    target_link_libraries(strix-kernel-registry-tests PRIVATE strix_core)
    strix_target_warnings(strix-kernel-registry-tests)
    add_test(NAME strix-kernel-registry-tests COMMAND strix-kernel-registry-tests)

    add_executable(strix-packed-projection-tests tests/packed_projection_tests.cpp)
    target_link_libraries(strix-packed-projection-tests PRIVATE strix_core)
    strix_target_warnings(strix-packed-projection-tests)
    add_test(NAME strix-packed-projection-tests COMMAND strix-packed-projection-tests)

    add_executable(strix-iq4-xs-tests tests/iq4_xs_tests.cpp)
    target_link_libraries(strix-iq4-xs-tests PRIVATE strix_core)
    strix_target_warnings(strix-iq4-xs-tests)
    add_test(NAME strix-iq4-xs-tests COMMAND strix-iq4-xs-tests)

    add_executable(strix-recurrent-plan-tests tests/recurrent_plan_tests.cpp)
    target_link_libraries(strix-recurrent-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-recurrent-plan-tests)
    add_test(NAME strix-recurrent-plan-tests COMMAND strix-recurrent-plan-tests)

    add_executable(strix-moe-plan-tests tests/moe_plan_tests.cpp)
    target_link_libraries(strix-moe-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-plan-tests)
    add_test(NAME strix-moe-plan-tests COMMAND strix-moe-plan-tests)

    add_executable(strix-sparse-lookup-tests tests/sparse_lookup_tests.cpp)
    target_link_libraries(strix-sparse-lookup-tests PRIVATE strix_core)
    strix_target_warnings(strix-sparse-lookup-tests)
    add_test(NAME strix-sparse-lookup-tests COMMAND strix-sparse-lookup-tests)

    add_executable(strix-execution-plan-tests tests/execution_plan_tests.cpp)
    target_link_libraries(strix-execution-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-plan-tests)
    add_test(NAME strix-execution-plan-tests COMMAND strix-execution-plan-tests)

    add_executable(strix-execution-buffer-tests tests/execution_buffers_tests.cpp)
    target_link_libraries(strix-execution-buffer-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-buffer-tests)
    add_test(NAME strix-execution-buffer-tests COMMAND strix-execution-buffer-tests)

    add_executable(strix-execution-binding-tests tests/execution_binding_tests.cpp)
    target_link_libraries(strix-execution-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-binding-tests)
    add_test(NAME strix-execution-binding-tests COMMAND strix-execution-binding-tests)

    add_executable(strix-execution-runtime-tests tests/execution_runtime_tests.cpp)
    target_link_libraries(strix-execution-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-runtime-tests)
    add_test(NAME strix-execution-runtime-tests COMMAND strix-execution-runtime-tests)

    add_executable(strix-execution-tensor-binding-tests tests/execution_tensor_binding_tests.cpp)
    target_link_libraries(strix-execution-tensor-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-tensor-binding-tests)
    add_test(NAME strix-execution-tensor-binding-tests COMMAND strix-execution-tensor-binding-tests)

    add_executable(strix-execution-io-tests tests/execution_io_tests.cpp)
    target_link_libraries(strix-execution-io-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-io-tests)
    add_test(NAME strix-execution-io-tests COMMAND strix-execution-io-tests)

    add_executable(strix-execution-arena-tests tests/execution_arena_tests.cpp)
    target_link_libraries(strix-execution-arena-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-arena-tests)
    add_test(NAME strix-execution-arena-tests COMMAND strix-execution-arena-tests)

    add_executable(strix-execution-memory-tests tests/execution_memory_tests.cpp)
    target_link_libraries(strix-execution-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-memory-tests)
    add_test(NAME strix-execution-memory-tests COMMAND strix-execution-memory-tests)

    add_executable(strix-execution-resources-tests tests/execution_resources_tests.cpp)
    target_link_libraries(strix-execution-resources-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-resources-tests)
    add_test(NAME strix-execution-resources-tests COMMAND strix-execution-resources-tests)

    add_executable(strix-execution-tensor-access-tests tests/execution_tensor_access_tests.cpp)
    target_link_libraries(strix-execution-tensor-access-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-tensor-access-tests)
    add_test(NAME strix-execution-tensor-access-tests COMMAND strix-execution-tensor-access-tests)

    add_executable(strix-dispatch-resources-tests tests/dispatch_resources_tests.cpp)
    target_link_libraries(strix-dispatch-resources-tests PRIVATE strix_core)
    strix_target_warnings(strix-dispatch-resources-tests)
    add_test(NAME strix-dispatch-resources-tests COMMAND strix-dispatch-resources-tests)

    add_executable(strix-launch-dispatch-tests tests/launch_dispatch_tests.cpp)
    target_link_libraries(strix-launch-dispatch-tests PRIVATE strix_core)
    strix_target_warnings(strix-launch-dispatch-tests)
    add_test(NAME strix-launch-dispatch-tests COMMAND strix-launch-dispatch-tests)

    add_executable(strix-runtime-launchers-tests tests/runtime_launchers_tests.cpp)
    target_link_libraries(strix-runtime-launchers-tests PRIVATE strix_core)
    strix_target_warnings(strix-runtime-launchers-tests)
    add_test(NAME strix-runtime-launchers-tests COMMAND strix-runtime-launchers-tests)

    add_executable(strix-compiled-execution-tests tests/compiled_execution_tests.cpp)
    target_link_libraries(strix-compiled-execution-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-execution-tests)
    add_test(NAME strix-compiled-execution-tests COMMAND strix-compiled-execution-tests)

    add_executable(strix-serial-session-tests tests/serial_session_tests.cpp)
    target_link_libraries(strix-serial-session-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-session-tests)
    add_test(NAME strix-serial-session-tests COMMAND strix-serial-session-tests)

    add_executable(strix-session-state-layout-tests tests/session_state_layout_tests.cpp)
    target_link_libraries(strix-session-state-layout-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-state-layout-tests)
    add_test(NAME strix-session-state-layout-tests COMMAND strix-session-state-layout-tests)

    add_executable(strix-session-state-memory-tests tests/session_state_memory_tests.cpp)
    target_link_libraries(strix-session-state-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-state-memory-tests)
    add_test(NAME strix-session-state-memory-tests COMMAND strix-session-state-memory-tests)

    add_executable(strix-session-snapshot-tests tests/session_snapshot_tests.cpp)
    target_link_libraries(strix-session-snapshot-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-snapshot-tests)
    add_test(NAME strix-session-snapshot-tests COMMAND strix-session-snapshot-tests)

    add_executable(strix-session-restore-tests tests/session_restore_tests.cpp)
    target_link_libraries(strix-session-restore-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-restore-tests)
    add_test(NAME strix-session-restore-tests COMMAND strix-session-restore-tests)

    add_executable(strix-session-snapshot-transfer-tests tests/session_snapshot_transfer_tests.cpp)
    target_link_libraries(strix-session-snapshot-transfer-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-snapshot-transfer-tests)
    add_test(NAME strix-session-snapshot-transfer-tests COMMAND strix-session-snapshot-transfer-tests)

    add_executable(strix-runtime-allocator-tests tests/runtime_allocator_tests.cpp)
    target_link_libraries(strix-runtime-allocator-tests PRIVATE strix_core)
    strix_target_warnings(strix-runtime-allocator-tests)
    add_test(NAME strix-runtime-allocator-tests COMMAND strix-runtime-allocator-tests)

    add_executable(strix-speculation-tests tests/speculation_tests.cpp)
    target_link_libraries(strix-speculation-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-tests)
    add_test(NAME strix-speculation-tests COMMAND strix-speculation-tests)

    add_executable(strix-speculation-controller-tests tests/speculation_controller_tests.cpp)
    target_link_libraries(strix-speculation-controller-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-controller-tests)
    add_test(NAME strix-speculation-controller-tests COMMAND strix-speculation-controller-tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME strix-python-oracle-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/oracles/test_qwen4_exp_oracle.py
        )
    endif()
endif()


if(STRIX_ENABLE_HIP_RUNTIME)
    enable_language(HIP)
    find_package(hip REQUIRED CONFIG)

    add_library(strix_hip_runtime STATIC
        src/hip_runtime_allocator.cpp
        src/hip_launcher_catalog.cpp
        src/hip_runtime_launchers.hip
        src/hip_moe_route_launcher.hip
        src/hip_embedding_launcher.hip
        src/hip_hyper_mix_launcher.hip
        src/hip_lm_head_launcher.hip
        src/hip_shared_expert_launcher.hip
    )
    target_include_directories(strix_hip_runtime PUBLIC include)
    target_compile_features(strix_hip_runtime PUBLIC cxx_std_20)
    set_target_properties(
        strix_hip_runtime
        PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
        HIP_ARCHITECTURES gfx1151
    )
    target_link_libraries(strix_hip_runtime PUBLIC strix_core hip::host)
    strix_target_warnings(strix_hip_runtime)

    add_executable(
        strix-hip-runtime-allocator
        apps/hip_runtime_allocator_main.cpp
    )
    target_link_libraries(
        strix-hip-runtime-allocator
        PRIVATE strix_hip_runtime
    )
    strix_target_warnings(strix-hip-runtime-allocator)

    if(BUILD_TESTING)
        add_executable(
            strix-hip-runtime-launchers-tests
            tests/hip_runtime_launchers_tests.hip
        )
        set_target_properties(
            strix-hip-runtime-launchers-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-runtime-launchers-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-runtime-launchers-tests
        )
        add_test(
            NAME strix-hip-runtime-launchers-tests
            COMMAND strix-hip-runtime-launchers-tests
        )

        add_executable(
            strix-hip-moe-route-launcher-tests
            tests/hip_moe_route_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-moe-route-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-moe-route-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-moe-route-launcher-tests
        )
        add_test(
            NAME strix-hip-moe-route-launcher-tests
            COMMAND strix-hip-moe-route-launcher-tests
        )

        add_executable(
            strix-hip-embedding-launcher-tests
            tests/hip_embedding_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-embedding-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-embedding-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-embedding-launcher-tests
        )
        add_test(
            NAME strix-hip-embedding-launcher-tests
            COMMAND strix-hip-embedding-launcher-tests
        )

        add_executable(
            strix-hip-hyper-mix-launcher-tests
            tests/hip_hyper_mix_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-hyper-mix-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-hyper-mix-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-hyper-mix-launcher-tests
        )
        add_test(
            NAME strix-hip-hyper-mix-launcher-tests
            COMMAND strix-hip-hyper-mix-launcher-tests
        )

        add_executable(
            strix-hip-launcher-catalog-tests
            tests/hip_launcher_catalog_tests.cpp
        )
        target_link_libraries(
            strix-hip-launcher-catalog-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-launcher-catalog-tests
        )
        add_test(
            NAME strix-hip-launcher-catalog-tests
            COMMAND strix-hip-launcher-catalog-tests
        )

        add_executable(
            strix-hip-lm-head-launcher-tests
            tests/hip_lm_head_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-lm-head-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-lm-head-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-lm-head-launcher-tests
        )
        add_test(
            NAME strix-hip-lm-head-launcher-tests
            COMMAND strix-hip-lm-head-launcher-tests
        )
    endif()
endif()
