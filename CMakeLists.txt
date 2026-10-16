cmake_minimum_required(VERSION 3.24)

project(strix_engine_lab VERSION 0.1.0 LANGUAGES CXX)

# Optional evaluated-input export only. No manifest is self-generated or
# admitted by configuration; explicit valid Ninja dependency export is required.
option(STRIX_EXPORT_OFFLINE_TUNING_PROVENANCE "Export Ninja POSIX FileAPI inputs for offline compile provenance" OFF)
if(STRIX_EXPORT_OFFLINE_TUNING_PROVENANCE)
    if(NOT UNIX OR NOT CMAKE_GENERATOR STREQUAL "Ninja")
        message(FATAL_ERROR "Offline compile provenance export supports single-config Ninja POSIX only")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/.cmake/api/v1/query")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/codemodel-v2" "")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/toolchains-v1" "")
endif()

option(STRIX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(STRIX_ENABLE_HIP_RUNTIME "Build optional HIP allocation/runtime support" OFF)
option(STRIX_USE_OPENSSL_SHA256 "Use OpenSSL for full-file SHA-256 verification" OFF)
option(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE "Build standalone experimental Q4K A8G32 HIP probe" OFF)
option(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME "Build opt-in native Q4K A8G32 gate/up integration" OFF)
set(STRIX_HC_REPACKED_ADMITTED_SCHEDULES "" CACHE STRING
    "Externally admitted HC research schedules (canonical ascending comma-separated subset of 31,32,33)")
# This only validates syntax, never compiler/ISA or numerical admission. The
# caller must bind the exact nonempty subset to an external qualification receipt.
set(_strix_hc_repacked_canonical_lists "31" "32" "33" "31,32" "31,33" "32,33" "31,32,33")
if(NOT "${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}" STREQUAL "")
    list(FIND _strix_hc_repacked_canonical_lists "${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}" _strix_hc_repacked_list_index)
    if(_strix_hc_repacked_list_index EQUAL -1)
        message(FATAL_ERROR "STRIX_HC_REPACKED_ADMITTED_SCHEDULES must be an ascending comma-separated subset of 31,32,33")
    endif()
endif()
unset(_strix_hc_repacked_canonical_lists)
unset(_strix_hc_repacked_list_index)
if((STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME) AND NOT STRIX_ENABLE_HIP_RUNTIME)
    message(FATAL_ERROR "Experimental Q4K A8 probe requires HIP runtime support")
endif()

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

find_package(Threads REQUIRED)

add_library(strix_core STATIC
    src/final_generation_state_export.cpp
    src/greedy_serving_scheduler.cpp
    src/serving_cohort.cpp
    src/generation_output.cpp
    src/known_prompt_chunks.cpp
    src/known_prefix_prefill.cpp
    src/known_prefix_prefill_runtime.cpp
    src/offline_tuning_cache.cpp
    src/offline_tuning_preparation.cpp
    src/offline_compile_provenance.cpp
    src/uma_owner_inventory.cpp
    src/linux_uma_observation.cpp
    src/moe_fork_join_arena.cpp
    src/moe_token_cost_backend.cpp
    src/prepared_execution.cpp
    src/prepared_tensor_views.cpp
    src/prepared_arena_views.cpp
    src/prepared_serial_session.cpp
    src/execution_telemetry.cpp
    src/qsa_input_prefetch.cpp
    src/model_copy_residency_layout.cpp
    src/lane_chain_hyper_output.cpp
    src/shared_expert_prefetch.cpp
    src/q51_grouped_down_contract.cpp
    src/q51_slot_parallel_contract.cpp
    src/moe_q4k_a8_runtime.cpp
    src/gdn_output_admission.cpp
    src/arithmetic_lineage.cpp
    src/graph_eligibility.cpp
    src/graph_inputs.cpp
    src/graph_attention_state.cpp
    src/graph_ple.cpp
    src/graph_operator_control.cpp
    src/integer_projection.cpp
    src/integer_projection_execution.cpp
    src/integer_projection_runtime.cpp
    src/host_upload.cpp
    src/compiled_execution.cpp
    src/copied_stream_transport.cpp
    src/direct_transport_gpu_export.cpp
    src/direct_gpu_boundary_staging.cpp
    src/linux_direct_transport_driver.cpp
    src/direct_gpu_stream_commissioning.cpp
    src/raw_baseline_recording.cpp
    src/compiled_mtp_draft_plan.cpp
    src/compiled_speculative_verify.cpp
    src/compiled_speculation_backend.cpp
    src/compiled_sampled_speculation_backend.cpp
    src/dispatch_resources.cpp
    src/distributed_flow.cpp
    src/distributed_placement.cpp
    src/distributed_data_plane.cpp
    src/distributed_native_backend.cpp
    src/distributed_token_transaction.cpp
    src/distributed_stage_resources.cpp
    src/distributed_stage_state.cpp
    src/distributed_greedy_control.cpp
    src/distributed_remote_control.cpp
    src/distributed_control_wire.cpp
    src/distributed_control_server.cpp
    src/distributed_control_receiver.cpp
    src/distributed_tcp_control_stream.cpp
    src/distributed_runtime.cpp
    src/distributed_session.cpp
    src/direct_transport_completion.cpp
    src/direct_transport_driver.cpp
    src/direct_stream_control.cpp
    src/direct_transport_native_backend.cpp
    src/direct_transport_native_session.cpp
    src/direct_transport_frames.cpp
    src/direct_transport_lease.cpp
    src/direct_transport_pool.cpp
    src/direct_transport_pool_memory.cpp
    src/direct_transport_rearm.cpp
    src/dual_link_runtime.cpp
    src/dual_link_telemetry.cpp
    src/draft_proposal_state.cpp
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
    src/indexed_attention_workspace.cpp
    src/indexed_attention_state.cpp
    src/iq4_nl.cpp
    src/launch_dispatch.cpp
    src/low_precision.cpp
    src/memory_plan.cpp
    src/model_contract.cpp
    src/model_content_hash.cpp
    src/mtp_draft_binding.cpp
    src/mtp_draft_contract.cpp
    src/mtp_draft_qualification.cpp
    src/mtp_draft_tensor_audit.cpp
    src/mtp_draft_state_contract.cpp
    src/mtp_draft_state_memory.cpp
    src/mtp_draft_readiness.cpp
    src/mtp_draft_activation.cpp
    src/mtp_draft_cycle_input.cpp
    src/mtp_forward_replay.cpp
    src/mtp_forward_contract.cpp
    src/mtp_forward_golden_file.cpp
    src/mtp_target_window_policy.cpp
    src/mtp_sidecar_binding.cpp
    src/mtp_forward_semantics.cpp
    src/mtp_forward_runtime.cpp
    src/mtp_input_fusion.cpp
    src/mtp_output_contract.cpp
    src/mtp_attention_contract.cpp
    src/qsa_selection_plan.cpp
    src/qualified_mtp_draft_backend.cpp
    src/qualified_mtp_forward_runtime.cpp
    src/model_manifest.cpp
    src/model_runtime_binding.cpp
    src/model_tensor_index.cpp
    src/moe_plan.cpp
    src/moe_layer_schedule.cpp
    src/moe_layer_runtime.cpp
    src/partition_search.cpp
    src/oracles.cpp
    src/packed_projection.cpp
    src/recurrent_plan.cpp
    src/runtime_allocator.cpp
    src/runtime_launchers.cpp
    src/serial_decode_reference.cpp
    src/serial_decode_replay.cpp
    src/serial_prefix_cache.cpp
    src/serial_prefix_lifecycle.cpp
    src/serial_request.cpp
    src/serial_session.cpp
    src/session_state_layout.cpp
    src/session_state_memory.cpp
    src/session_snapshot.cpp
    src/session_snapshot_memory.cpp
    src/session_snapshot_transfer.cpp
    src/session_restore.cpp
    src/sparse_lookup.cpp
    src/sparse_lookup_executor.cpp
    src/speculation.cpp
    src/speculation_controller.cpp
    src/speculation_measurement.cpp
    src/speculation_rng.cpp
    src/speculation_sampling.cpp
    src/speculation_sampling_cycle.cpp
    src/speculation_sampling_executor.cpp
    src/speculation_cycle.cpp
    src/speculation_executor.cpp
    src/speculative_verify_plan.cpp
    src/speculative_verify_runtime.cpp
    src/speculative_verify_binding.cpp
    src/platform_probe.cpp
    src/prompt_lookup_draft.cpp
    src/projection_reduction.cpp
    src/ple_lookup.cpp
    src/ple_workspace.cpp
    src/transport.cpp
)
target_include_directories(strix_core PUBLIC include)
target_compile_features(strix_core PUBLIC cxx_std_20)
target_link_libraries(strix_core PUBLIC Threads::Threads)
if(STRIX_USE_OPENSSL_SHA256)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    target_link_libraries(strix_core PRIVATE OpenSSL::Crypto)
    target_compile_definitions(strix_core PRIVATE STRIX_USE_OPENSSL_SHA256=1)
endif()
strix_target_warnings(strix_core)
if(NOT MSVC)
    set_source_files_properties(src/low_precision.cpp src/projection_reduction.cpp src/integer_projection.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
else()
    set_source_files_properties(src/low_precision.cpp src/projection_reduction.cpp src/integer_projection.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
endif()

add_executable(strix-low-precision-reference lab/low_precision/reference_main.cpp)
target_link_libraries(strix-low-precision-reference PRIVATE strix_core)
strix_target_warnings(strix-low-precision-reference)

add_executable(strix-memory-plan apps/memory_plan_main.cpp)
target_link_libraries(strix-memory-plan PRIVATE strix_core)
strix_target_warnings(strix-memory-plan)

add_executable(strix-inspect-gguf apps/inspect_gguf_main.cpp)
target_link_libraries(strix-inspect-gguf PRIVATE strix_core)
strix_target_warnings(strix-inspect-gguf)

add_executable(strix-mtp-manifest-audit apps/mtp_manifest_audit_main.cpp)
target_link_libraries(strix-mtp-manifest-audit PRIVATE strix_core)
strix_target_warnings(strix-mtp-manifest-audit)

add_executable(strix-model-manifest apps/model_manifest_main.cpp)
target_link_libraries(strix-model-manifest PRIVATE strix_core)
strix_target_warnings(strix-model-manifest)

add_executable(strix-platform-probe apps/platform_probe_main.cpp)
target_link_libraries(strix-platform-probe PRIVATE strix_core)
strix_target_warnings(strix-platform-probe)

add_executable(strix-validate-serial-reference apps/serial_decode_reference_main.cpp)
target_link_libraries(strix-validate-serial-reference PRIVATE strix_core)
strix_target_warnings(strix-validate-serial-reference)

add_executable(strix-audit-execution-tensors apps/execution_tensor_audit_main.cpp)
target_link_libraries(strix-audit-execution-tensors PRIVATE strix_core)
strix_target_warnings(strix-audit-execution-tensors)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(strix-direct-gpu-observe-driver
        lab/direct_gpu_stream/observe_main.cpp lab/direct_gpu_stream/observe_driver.cpp)
    target_link_libraries(strix-direct-gpu-observe-driver PRIVATE strix_core)
    strix_target_warnings(strix-direct-gpu-observe-driver)
endif()

include(CTest)
if(BUILD_TESTING OR STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
    add_library(strix_q4k_a8_contract STATIC src/moe_q4k_a8_contract.cpp)
    target_include_directories(strix_q4k_a8_contract PUBLIC include)
    target_compile_features(strix_q4k_a8_contract PUBLIC cxx_std_20)
    strix_target_warnings(strix_q4k_a8_contract)
    if(MSVC)
        target_compile_options(strix_q4k_a8_contract PRIVATE /fp:strict)
    else()
        target_compile_options(strix_q4k_a8_contract PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
endif()
if(BUILD_TESTING)
    add_executable(strix-final-generation-state-export-tests tests/final_generation_state_export_tests.cpp)
    target_link_libraries(strix-final-generation-state-export-tests PRIVATE strix_core)
    strix_target_warnings(strix-final-generation-state-export-tests)
    add_test(NAME strix-final-generation-state-export-tests COMMAND strix-final-generation-state-export-tests)
    add_executable(strix-uma-owner-inventory-tests tests/uma_owner_inventory_tests.cpp)
    target_link_libraries(strix-uma-owner-inventory-tests PRIVATE strix_core)
    strix_target_warnings(strix-uma-owner-inventory-tests)
    add_test(NAME strix-uma-owner-inventory-tests COMMAND strix-uma-owner-inventory-tests)

    add_executable(strix-shared-expert-rows-diagnostic-contract-tests tests/shared_expert_rows_diagnostic_contract_tests.cpp)
    target_link_libraries(strix-shared-expert-rows-diagnostic-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-shared-expert-rows-diagnostic-contract-tests)
    add_test(NAME strix-shared-expert-rows-diagnostic-contract-tests COMMAND strix-shared-expert-rows-diagnostic-contract-tests)

    add_executable(strix-moe-fork-join-arena-tests tests/moe_fork_join_arena_tests.cpp)
    target_link_libraries(strix-moe-fork-join-arena-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-fork-join-arena-tests)
    add_test(NAME strix-moe-fork-join-arena-tests COMMAND strix-moe-fork-join-arena-tests)

    add_executable(strix-prepared-tensor-views-tests tests/prepared_tensor_views_tests.cpp)
    target_link_libraries(strix-prepared-tensor-views-tests PRIVATE strix_core)
    strix_target_warnings(strix-prepared-tensor-views-tests)
    add_test(NAME strix-prepared-tensor-views-tests COMMAND strix-prepared-tensor-views-tests)

    add_executable(strix-prepared-execution-tests tests/prepared_execution_tests.cpp)
    target_link_libraries(strix-prepared-execution-tests PRIVATE strix_core)
    strix_target_warnings(strix-prepared-execution-tests)
    add_test(NAME strix-prepared-execution-tests COMMAND strix-prepared-execution-tests)
    add_executable(strix-generation-output-dispatch-tests tests/generation_output_dispatch_tests.cpp)
    target_link_libraries(strix-generation-output-dispatch-tests PRIVATE strix_core)
    strix_target_warnings(strix-generation-output-dispatch-tests)
    add_test(NAME strix-generation-output-dispatch-tests COMMAND strix-generation-output-dispatch-tests)
    add_executable(strix-greedy-token-request-tests tests/greedy_token_request_tests.cpp)
    target_link_libraries(strix-greedy-token-request-tests PRIVATE strix_core)
    strix_target_warnings(strix-greedy-token-request-tests)
    add_test(NAME strix-greedy-token-request-tests COMMAND strix-greedy-token-request-tests)
    add_executable(strix-execution-inventory-receipt-tests tests/execution_inventory_receipt_tests.cpp)
    target_link_libraries(strix-execution-inventory-receipt-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-inventory-receipt-tests)
    add_test(NAME strix-execution-inventory-receipt-tests COMMAND strix-execution-inventory-receipt-tests)
    add_executable(strix-moe-token-cost-backend-tests tests/moe_token_cost_backend_tests.cpp)
    target_link_libraries(strix-moe-token-cost-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-token-cost-backend-tests)
    add_test(NAME strix-moe-token-cost-backend-tests COMMAND strix-moe-token-cost-backend-tests)

    add_executable(strix-serving-cohort-tests tests/serving_cohort_tests.cpp)
    target_link_libraries(strix-serving-cohort-tests PRIVATE strix_core)
    strix_target_warnings(strix-serving-cohort-tests)
    add_test(NAME strix-serving-cohort-tests COMMAND strix-serving-cohort-tests)

    add_executable(strix-known-prompt-chunks-tests tests/known_prompt_chunks_tests.cpp)
    target_link_libraries(strix-known-prompt-chunks-tests PRIVATE strix_core)
    strix_target_warnings(strix-known-prompt-chunks-tests)
    add_test(NAME strix-known-prompt-chunks-tests COMMAND strix-known-prompt-chunks-tests)
    add_executable(strix-offline-compile-provenance-tests tests/offline_compile_provenance_tests.cpp)
    target_link_libraries(strix-offline-compile-provenance-tests PRIVATE strix_core)
    strix_target_warnings(strix-offline-compile-provenance-tests)
    add_test(NAME offline-compile-provenance COMMAND strix-offline-compile-provenance-tests)
    add_executable(strix-offline-tuning-preparation-tests tests/offline_tuning_preparation_tests.cpp)
    target_link_libraries(strix-offline-tuning-preparation-tests PRIVATE strix_core)
    strix_target_warnings(strix-offline-tuning-preparation-tests)
    add_test(NAME strix-offline-tuning-preparation-tests COMMAND strix-offline-tuning-preparation-tests)
    add_executable(strix-offline-tuning-cache-tests tests/offline_tuning_cache_tests.cpp)
    target_link_libraries(strix-offline-tuning-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-offline-tuning-cache-tests)
    add_test(NAME strix-offline-tuning-cache-tests COMMAND strix-offline-tuning-cache-tests)
    add_executable(strix-generation-output-tests tests/generation_output_tests.cpp)
    target_link_libraries(strix-generation-output-tests PRIVATE strix_core)
    strix_target_warnings(strix-generation-output-tests)
    add_test(NAME strix-generation-output-tests COMMAND strix-generation-output-tests)
    add_executable(strix-greedy-serving-scheduler-tests tests/greedy_serving_scheduler_tests.cpp)
    target_link_libraries(strix-greedy-serving-scheduler-tests PRIVATE strix_core)
    strix_target_warnings(strix-greedy-serving-scheduler-tests)
    add_test(NAME strix-greedy-serving-scheduler-tests COMMAND strix-greedy-serving-scheduler-tests)
    add_executable(strix-execution-telemetry-tests tests/execution_telemetry_tests.cpp)
    target_link_libraries(strix-execution-telemetry-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-telemetry-tests)
    add_test(NAME strix-execution-telemetry-tests COMMAND strix-execution-telemetry-tests)
    # Diagnostic-only byte permutation. It is never linked into the decoder.
    add_library(strix_hc_repacked_down_pack STATIC src/hc_repacked_down_pack.cpp)
    target_include_directories(strix_hc_repacked_down_pack PUBLIC include)
    target_compile_features(strix_hc_repacked_down_pack PUBLIC cxx_std_20)
    target_link_libraries(strix_hc_repacked_down_pack PUBLIC strix_core)
    strix_target_warnings(strix_hc_repacked_down_pack)
    add_executable(strix-hc-repacked-down-pack-tests tests/hc_repacked_down_pack_tests.cpp)
    target_link_libraries(strix-hc-repacked-down-pack-tests PRIVATE strix_hc_repacked_down_pack)
    strix_target_warnings(strix-hc-repacked-down-pack-tests)
    add_test(NAME strix-hc-repacked-down-pack-tests COMMAND strix-hc-repacked-down-pack-tests)

    add_executable(strix-q8-projection-organization-tests tests/q8_projection_organization_tests.cpp)
    target_link_libraries(strix-q8-projection-organization-tests PRIVATE strix_core)
    strix_target_warnings(strix-q8-projection-organization-tests)
    add_test(NAME strix-q8-projection-organization-tests COMMAND strix-q8-projection-organization-tests)
    add_executable(strix-hc-projection-prefetch-tests tests/hc_projection_prefetch_tests.cpp)
    target_link_libraries(strix-hc-projection-prefetch-tests PRIVATE strix_core)
    strix_target_warnings(strix-hc-projection-prefetch-tests)
    add_test(NAME strix-hc-projection-prefetch-tests COMMAND strix-hc-projection-prefetch-tests)
    add_executable(strix-q8-projection-options-tests tests/q8_projection_options_tests.cpp)
    target_link_libraries(strix-q8-projection-options-tests PRIVATE strix_core)
    strix_target_warnings(strix-q8-projection-options-tests)
    add_test(NAME strix-q8-projection-options-tests COMMAND strix-q8-projection-options-tests)
    add_executable(strix-gdn-lm-wave8-diagnostic-contract-tests tests/gdn_lm_wave8_diagnostic_contract_tests.cpp)
    target_link_libraries(strix-gdn-lm-wave8-diagnostic-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-gdn-lm-wave8-diagnostic-contract-tests)
    add_test(NAME strix-gdn-lm-wave8-diagnostic-contract-tests COMMAND strix-gdn-lm-wave8-diagnostic-contract-tests)
    add_executable(strix-decode-optimization-profile-tests tests/decode_optimization_profile_tests.cpp)
    target_link_libraries(strix-decode-optimization-profile-tests PRIVATE strix_core)
    strix_target_warnings(strix-decode-optimization-profile-tests)
    add_test(NAME strix-decode-optimization-profile-tests COMMAND strix-decode-optimization-profile-tests)
    add_executable(strix-decode-optimization-profile-a8-cli-tests tests/decode_optimization_profile_tests.cpp)
    target_compile_definitions(strix-decode-optimization-profile-a8-cli-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    target_link_libraries(strix-decode-optimization-profile-a8-cli-tests PRIVATE strix_core)
    strix_target_warnings(strix-decode-optimization-profile-a8-cli-tests)
    add_test(NAME strix-decode-optimization-profile-a8-cli-tests COMMAND strix-decode-optimization-profile-a8-cli-tests)
    add_executable(strix-spec-v1-integration-tests tests/spec_v1_integration_tests.cpp)
    target_link_libraries(strix-spec-v1-integration-tests PRIVATE strix_core)
    strix_target_warnings(strix-spec-v1-integration-tests)
    add_test(NAME strix-spec-v1-integration-tests COMMAND strix-spec-v1-integration-tests)
    add_executable(strix-qsa-input-prefetch-tests tests/qsa_input_prefetch_tests.cpp)
    target_link_libraries(strix-qsa-input-prefetch-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-input-prefetch-tests)
    add_test(NAME strix-qsa-input-prefetch-tests COMMAND strix-qsa-input-prefetch-tests)
    add_executable(strix-shared-expert-prefetch-tests tests/shared_expert_prefetch_tests.cpp)
    target_link_libraries(strix-shared-expert-prefetch-tests PRIVATE strix_core)
    strix_target_warnings(strix-shared-expert-prefetch-tests)
    add_test(NAME strix-shared-expert-prefetch-tests COMMAND strix-shared-expert-prefetch-tests)
    add_executable(strix-moe-q4k-a8-contract-tests tests/moe_q4k_a8_contract_tests.cpp)
    target_link_libraries(strix-moe-q4k-a8-contract-tests PRIVATE strix_q4k_a8_contract strix_core)
    strix_target_warnings(strix-moe-q4k-a8-contract-tests)
    if(MSVC)
        target_compile_options(strix-moe-q4k-a8-contract-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-q4k-a8-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-moe-q4k-a8-contract-tests COMMAND strix-moe-q4k-a8-contract-tests)
    add_executable(strix-hc-parallel-projection-oracle-tests tests/hc_parallel_projection_oracle_tests.cpp)
    target_compile_features(strix-hc-parallel-projection-oracle-tests PRIVATE cxx_std_20)
    target_include_directories(strix-hc-parallel-projection-oracle-tests PRIVATE include)
    strix_target_warnings(strix-hc-parallel-projection-oracle-tests)
    add_test(NAME strix-hc-parallel-projection-oracle-tests COMMAND strix-hc-parallel-projection-oracle-tests)

    add_executable(strix-lane-chain-hyper-output-tests tests/lane_chain_hyper_output_tests.cpp)
    target_link_libraries(strix-lane-chain-hyper-output-tests PRIVATE strix_core)
    strix_target_warnings(strix-lane-chain-hyper-output-tests)
    add_test(NAME strix-lane-chain-hyper-output-tests COMMAND strix-lane-chain-hyper-output-tests)
    add_executable(strix-gdn-output-admission-tests tests/gdn_output_admission_tests.cpp)
    target_link_libraries(strix-gdn-output-admission-tests PRIVATE strix_core)
    strix_target_warnings(strix-gdn-output-admission-tests)
    add_test(NAME strix-gdn-output-admission-tests COMMAND strix-gdn-output-admission-tests)
    add_executable(strix-virtual-q8-gdn-oracle-tests tests/virtual_q8_gdn_oracle_tests.cpp)
    target_compile_features(strix-virtual-q8-gdn-oracle-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-virtual-q8-gdn-oracle-tests)
    add_test(NAME strix-virtual-q8-gdn-oracle-tests COMMAND strix-virtual-q8-gdn-oracle-tests)
    add_executable(strix-router-ranked-topk-tests tests/router_ranked_topk_tests.cpp)
    strix_target_warnings(strix-router-ranked-topk-tests)
    target_compile_features(strix-router-ranked-topk-tests PRIVATE cxx_std_20)
    add_test(NAME strix-router-ranked-topk-tests COMMAND strix-router-ranked-topk-tests)
    add_executable(strix-router-wave-topk-tests tests/router_wave_topk_tests.cpp)
    target_link_libraries(strix-router-wave-topk-tests PRIVATE strix_core)
    target_include_directories(strix-router-wave-topk-tests PRIVATE include)
    target_compile_features(strix-router-wave-topk-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-router-wave-topk-tests)
    add_test(NAME strix-router-wave-topk-tests COMMAND strix-router-wave-topk-tests)
    add_executable(strix-deferred-moe-status-tests tests/deferred_moe_status_tests.cpp)
    target_link_libraries(strix-deferred-moe-status-tests PRIVATE strix_core)
    target_compile_features(strix-deferred-moe-status-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-deferred-moe-status-tests)
    add_test(NAME strix-deferred-moe-status-tests COMMAND strix-deferred-moe-status-tests)
    add_executable(strix-virtual-q8-oracle-tests tests/virtual_q8_oracle_tests.cpp)
    target_compile_features(strix-virtual-q8-oracle-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-virtual-q8-oracle-tests)
    add_test(NAME strix-virtual-q8-oracle-tests COMMAND strix-virtual-q8-oracle-tests)
    add_executable(strix-direct-gpu-stream-commissioning-tests
        tests/direct_gpu_stream_commissioning_tests.cpp
        lab/direct_gpu_stream/bounded_tcp_control.cpp)
    target_link_libraries(strix-direct-gpu-stream-commissioning-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-gpu-stream-commissioning-tests)
    add_test(NAME strix-direct-gpu-stream-commissioning-tests COMMAND strix-direct-gpu-stream-commissioning-tests)
    add_executable(strix-raw-baseline-recording-tests tests/raw_baseline_recording_tests.cpp)
    target_link_libraries(strix-raw-baseline-recording-tests PRIVATE strix_core)
    strix_target_warnings(strix-raw-baseline-recording-tests)
    add_test(NAME strix-raw-baseline-recording-tests COMMAND strix-raw-baseline-recording-tests)
    add_executable(strix-direct-transport-gpu-export-tests tests/direct_transport_gpu_export_tests.cpp)
    target_link_libraries(strix-direct-transport-gpu-export-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-gpu-export-tests)
    add_test(NAME strix-direct-transport-gpu-export-tests COMMAND strix-direct-transport-gpu-export-tests)
    add_executable(strix-graph-ple-tests tests/graph_ple_tests.cpp)
    add_executable(strix-direct-gpu-boundary-staging-tests tests/direct_gpu_boundary_staging_tests.cpp)
    target_link_libraries(strix-direct-gpu-boundary-staging-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-gpu-boundary-staging-tests)
    add_test(NAME strix-direct-gpu-boundary-staging-tests COMMAND strix-direct-gpu-boundary-staging-tests)
    target_link_libraries(strix-graph-ple-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-ple-tests)
    add_test(NAME strix-graph-ple-tests COMMAND strix-graph-ple-tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(strix-direct-gpu-stream-driver-observation-tests
            tests/direct_gpu_stream_driver_observation_tests.cpp lab/direct_gpu_stream/observe_driver.cpp)
        target_link_libraries(strix-direct-gpu-stream-driver-observation-tests PRIVATE strix_core)
        strix_target_warnings(strix-direct-gpu-stream-driver-observation-tests)
        add_test(NAME strix-direct-gpu-stream-driver-observation-tests COMMAND strix-direct-gpu-stream-driver-observation-tests)
        enable_language(C)
        add_executable(strix-linux-direct-transport-driver-tests tests/linux_direct_transport_driver_tests.cpp)
        target_link_libraries(strix-linux-direct-transport-driver-tests PRIVATE strix_core)
        strix_target_warnings(strix-linux-direct-transport-driver-tests)
        add_test(NAME strix-linux-direct-transport-driver-tests COMMAND strix-linux-direct-transport-driver-tests)
        add_executable(strix-linux-direct-gpu-stream-logic-tests lab/direct_gpu_stream/kernel/logic_tests.c)
        target_include_directories(strix-linux-direct-gpu-stream-logic-tests PRIVATE include lab/direct_gpu_stream/kernel)
        target_compile_features(strix-linux-direct-gpu-stream-logic-tests PRIVATE c_std_11)
        strix_target_warnings(strix-linux-direct-gpu-stream-logic-tests)
        add_test(NAME strix-linux-direct-gpu-stream-logic-tests COMMAND strix-linux-direct-gpu-stream-logic-tests)
        add_executable(strix-linux-direct-gpu-stream-uapi-tests tests/linux_direct_gpu_stream_uapi_tests.cpp)
        target_include_directories(strix-linux-direct-gpu-stream-uapi-tests PRIVATE include)
        target_compile_features(strix-linux-direct-gpu-stream-uapi-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-linux-direct-gpu-stream-uapi-tests)
        add_test(NAME strix-linux-direct-gpu-stream-uapi-tests COMMAND strix-linux-direct-gpu-stream-uapi-tests)
    endif()
    add_executable(strix-integer-projection-runtime-tests tests/integer_projection_runtime_tests.cpp)
    target_link_libraries(strix-integer-projection-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-integer-projection-runtime-tests)
    add_test(NAME strix-integer-projection-runtime-tests COMMAND strix-integer-projection-runtime-tests)
    add_executable(strix-graph-operator-control-tests tests/graph_operator_control_tests.cpp)
    target_link_libraries(strix-graph-operator-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-operator-control-tests)
    add_test(NAME strix-graph-operator-control-tests COMMAND strix-graph-operator-control-tests)
    add_executable(strix-copied-stream-transport-tests tests/copied_stream_transport_tests.cpp)
    target_link_libraries(strix-copied-stream-transport-tests PRIVATE strix_core)
    strix_target_warnings(strix-copied-stream-transport-tests)
    add_test(NAME strix-copied-stream-transport-tests COMMAND strix-copied-stream-transport-tests)
    add_executable(strix-integer-projection-execution-tests tests/integer_projection_execution_tests.cpp)
    target_link_libraries(strix-integer-projection-execution-tests PRIVATE strix_core)
    strix_target_warnings(strix-integer-projection-execution-tests)
    add_test(NAME strix-integer-projection-execution-tests COMMAND strix-integer-projection-execution-tests)
    add_executable(strix-graph-attention-state-tests tests/graph_attention_state_tests.cpp)
    target_link_libraries(strix-graph-attention-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-attention-state-tests)
    add_test(NAME strix-graph-attention-state-tests COMMAND strix-graph-attention-state-tests)
    add_executable(strix-direct-stream-control-tests tests/direct_stream_control_tests.cpp)
    target_link_libraries(strix-direct-stream-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-stream-control-tests)
    add_test(NAME strix-direct-stream-control-tests COMMAND strix-direct-stream-control-tests)
    add_executable(strix-direct-transport-driver-tests tests/direct_transport_driver_tests.cpp)
    target_link_libraries(strix-direct-transport-driver-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-driver-tests)
    add_test(NAME strix-direct-transport-driver-tests COMMAND strix-direct-transport-driver-tests)
    add_executable(strix-integer-projection-tests tests/integer_projection_tests.cpp)
    target_link_libraries(strix-integer-projection-tests PRIVATE strix_core)
    strix_target_warnings(strix-integer-projection-tests)
    if(NOT MSVC)
        target_compile_options(strix-integer-projection-tests PRIVATE -ffp-contract=off)
    else()
        target_compile_options(strix-integer-projection-tests PRIVATE /fp:strict)
    endif()
    add_test(NAME strix-integer-projection-tests COMMAND strix-integer-projection-tests)
    add_executable(strix-decode-execution-options-tests tests/decode_execution_options_tests.cpp)
    target_link_libraries(strix-decode-execution-options-tests PRIVATE strix_core)
    strix_target_warnings(strix-decode-execution-options-tests)
    add_test(NAME strix-decode-execution-options-tests COMMAND strix-decode-execution-options-tests)
    add_executable(strix-native-setup-quiescence-tests tests/native_setup_quiescence_tests.cpp)
    target_link_libraries(strix-native-setup-quiescence-tests PRIVATE strix_core)
    strix_target_warnings(strix-native-setup-quiescence-tests)
    add_test(NAME strix-native-setup-quiescence-tests COMMAND strix-native-setup-quiescence-tests)
    add_executable(strix-graph-inputs-tests tests/graph_inputs_tests.cpp)
    target_link_libraries(strix-graph-inputs-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-inputs-tests)
    add_test(NAME strix-graph-inputs-tests COMMAND strix-graph-inputs-tests)
    add_executable(strix-host-upload-tests tests/host_upload_tests.cpp)
    target_link_libraries(strix-host-upload-tests PRIVATE strix_core)
    strix_target_warnings(strix-host-upload-tests)
    add_test(NAME strix-host-upload-tests COMMAND strix-host-upload-tests)
    add_executable(strix-graph-eligibility-tests tests/graph_eligibility_tests.cpp)
    target_link_libraries(strix-graph-eligibility-tests PRIVATE strix_core)
    strix_target_warnings(strix-graph-eligibility-tests)
    add_test(NAME strix-graph-eligibility-tests COMMAND strix-graph-eligibility-tests)
    add_executable(strix-projection-reduction-tests tests/projection_reduction_tests.cpp)
    target_link_libraries(strix-projection-reduction-tests PRIVATE strix_core)
    strix_target_warnings(strix-projection-reduction-tests)
    if(NOT MSVC)
        target_compile_options(strix-projection-reduction-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-projection-reduction-tests COMMAND strix-projection-reduction-tests)
    add_executable(strix-low-precision-tests tests/low_precision_tests.cpp)
    target_link_libraries(strix-low-precision-tests PRIVATE strix_core)
    strix_target_warnings(strix-low-precision-tests)
    if(NOT MSVC)
        target_compile_options(strix-low-precision-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-low-precision-tests COMMAND strix-low-precision-tests)
    add_test(NAME strix-low-precision-vectors COMMAND strix-low-precision-reference vectors)
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

    add_executable(strix-direct-transport-frame-tests tests/direct_transport_frames_tests.cpp)
    target_link_libraries(strix-direct-transport-frame-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-frame-tests)
    add_test(NAME strix-direct-transport-frame-tests COMMAND strix-direct-transport-frame-tests)

    add_executable(strix-direct-transport-pool-tests tests/direct_transport_pool_tests.cpp)
    target_link_libraries(strix-direct-transport-pool-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-pool-tests)
    add_test(NAME strix-direct-transport-pool-tests COMMAND strix-direct-transport-pool-tests)

    add_executable(strix-direct-transport-pool-memory-tests tests/direct_transport_pool_memory_tests.cpp)
    target_link_libraries(strix-direct-transport-pool-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-pool-memory-tests)
    add_test(NAME strix-direct-transport-pool-memory-tests COMMAND strix-direct-transport-pool-memory-tests)

    add_executable(strix-direct-transport-lease-tests tests/direct_transport_lease_tests.cpp)
    target_link_libraries(strix-direct-transport-lease-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-lease-tests)
    add_test(NAME strix-direct-transport-lease-tests COMMAND strix-direct-transport-lease-tests)

    add_executable(strix-direct-transport-completion-tests tests/direct_transport_completion_tests.cpp)
    target_link_libraries(strix-direct-transport-completion-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-completion-tests)
    add_test(NAME strix-direct-transport-completion-tests COMMAND strix-direct-transport-completion-tests)

    add_executable(strix-direct-transport-native-backend-tests tests/direct_transport_native_backend_tests.cpp)
    target_link_libraries(strix-direct-transport-native-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-native-backend-tests)
    add_test(NAME strix-direct-transport-native-backend-tests COMMAND strix-direct-transport-native-backend-tests)

    add_executable(strix-direct-transport-native-session-tests tests/direct_transport_native_session_tests.cpp)
    target_link_libraries(strix-direct-transport-native-session-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-native-session-tests)
    add_test(NAME strix-direct-transport-native-session-tests COMMAND strix-direct-transport-native-session-tests)

    add_executable(strix-direct-transport-rearm-tests tests/direct_transport_rearm_tests.cpp)
    target_link_libraries(strix-direct-transport-rearm-tests PRIVATE strix_core)
    strix_target_warnings(strix-direct-transport-rearm-tests)
    add_test(NAME strix-direct-transport-rearm-tests COMMAND strix-direct-transport-rearm-tests)

    add_executable(strix-dual-link-runtime-tests tests/dual_link_runtime_tests.cpp)
    target_link_libraries(strix-dual-link-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-dual-link-runtime-tests)
    add_test(NAME strix-dual-link-runtime-tests COMMAND strix-dual-link-runtime-tests)

    add_executable(strix-dual-link-telemetry-tests tests/dual_link_telemetry_tests.cpp)
    target_link_libraries(strix-dual-link-telemetry-tests PRIVATE strix_core)
    strix_target_warnings(strix-dual-link-telemetry-tests)
    add_test(NAME strix-dual-link-telemetry-tests COMMAND strix-dual-link-telemetry-tests)

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

    add_executable(strix-iq4-nl-tests tests/iq4_nl_tests.cpp)
    target_link_libraries(strix-iq4-nl-tests PRIVATE strix_core)
    strix_target_warnings(strix-iq4-nl-tests)
    add_test(NAME strix-iq4-nl-tests COMMAND strix-iq4-nl-tests)

    add_executable(
        strix-indexed-attention-workspace-tests
        tests/indexed_attention_workspace_tests.cpp
    )
    target_link_libraries(
        strix-indexed-attention-workspace-tests
        PRIVATE strix_core
    )
    strix_target_warnings(strix-indexed-attention-workspace-tests)
    add_test(
        NAME strix-indexed-attention-workspace-tests
        COMMAND strix-indexed-attention-workspace-tests
    )

    add_executable(strix-recurrent-plan-tests tests/recurrent_plan_tests.cpp)
    target_link_libraries(strix-recurrent-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-recurrent-plan-tests)
    add_test(NAME strix-recurrent-plan-tests COMMAND strix-recurrent-plan-tests)

    add_executable(strix-moe-plan-tests tests/moe_plan_tests.cpp)
    target_link_libraries(strix-moe-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-plan-tests)
    add_test(NAME strix-moe-plan-tests COMMAND strix-moe-plan-tests)

    add_executable(strix-moe-layer-schedule-tests tests/moe_layer_schedule_tests.cpp)
    target_link_libraries(strix-moe-layer-schedule-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-layer-schedule-tests)
    add_test(NAME strix-moe-layer-schedule-tests COMMAND strix-moe-layer-schedule-tests)

    add_executable(strix-moe-layer-runtime-tests tests/moe_layer_runtime_tests.cpp)
    target_link_libraries(strix-moe-layer-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-layer-runtime-tests)
    add_test(NAME strix-moe-layer-runtime-tests COMMAND strix-moe-layer-runtime-tests)

    add_executable(strix-partition-search-tests tests/partition_search_tests.cpp)
    target_link_libraries(strix-partition-search-tests PRIVATE strix_core)
    strix_target_warnings(strix-partition-search-tests)
    add_test(NAME strix-partition-search-tests COMMAND strix-partition-search-tests)

    add_executable(strix-sparse-lookup-tests tests/sparse_lookup_tests.cpp)
    target_link_libraries(strix-sparse-lookup-tests PRIVATE strix_core)
    strix_target_warnings(strix-sparse-lookup-tests)
    add_test(NAME strix-sparse-lookup-tests COMMAND strix-sparse-lookup-tests)

    add_executable(strix-sparse-lookup-executor-tests tests/sparse_lookup_executor_tests.cpp)
    target_link_libraries(strix-sparse-lookup-executor-tests PRIVATE strix_core)
    strix_target_warnings(strix-sparse-lookup-executor-tests)
    add_test(NAME strix-sparse-lookup-executor-tests COMMAND strix-sparse-lookup-executor-tests)

    add_executable(strix-ple-lookup-tests tests/ple_lookup_tests.cpp)
    target_link_libraries(strix-ple-lookup-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-lookup-tests)
    add_test(NAME strix-ple-lookup-tests COMMAND strix-ple-lookup-tests)

    add_executable(strix-ple-iq4-nl-tests tests/ple_iq4_nl_tests.cpp)
    target_link_libraries(strix-ple-iq4-nl-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-iq4-nl-tests)
    add_test(NAME strix-ple-iq4-nl-tests COMMAND strix-ple-iq4-nl-tests)

    add_executable(strix-ple-workspace-tests tests/ple_workspace_tests.cpp)
    target_link_libraries(strix-ple-workspace-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-workspace-tests)
    add_test(NAME strix-ple-workspace-tests COMMAND strix-ple-workspace-tests)

    add_executable(strix-execution-plan-tests tests/execution_plan_tests.cpp)
    target_link_libraries(strix-execution-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-plan-tests)
    add_test(NAME strix-execution-plan-tests COMMAND strix-execution-plan-tests)

    add_executable(strix-q51-grouped-down-contract-tests tests/q51_grouped_down_contract_tests.cpp)
    target_link_libraries(strix-q51-grouped-down-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-q51-grouped-down-contract-tests)
    if(NOT MSVC)
        target_compile_options(strix-q51-grouped-down-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-q51-grouped-down-contract-tests COMMAND strix-q51-grouped-down-contract-tests)

    add_executable(strix-q51-slot-parallel-contract-tests tests/q51_slot_parallel_contract_tests.cpp)
    target_link_libraries(strix-q51-slot-parallel-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-q51-slot-parallel-contract-tests)
    add_test(NAME strix-q51-slot-parallel-contract-tests COMMAND strix-q51-slot-parallel-contract-tests)

    add_executable(strix-moe-q4k-a8-runtime-tests tests/moe_q4k_a8_runtime_tests.cpp)
    target_link_libraries(strix-moe-q4k-a8-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-q4k-a8-runtime-tests)
    add_test(NAME strix-moe-q4k-a8-runtime-tests COMMAND strix-moe-q4k-a8-runtime-tests)

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

    add_executable(strix-memory-plan-tests tests/memory_plan_tests.cpp)
    target_link_libraries(strix-memory-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-memory-plan-tests)
    add_test(NAME strix-memory-plan-tests COMMAND strix-memory-plan-tests)

    add_executable(strix-execution-memory-tests tests/execution_memory_tests.cpp)
    target_link_libraries(strix-execution-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-memory-tests)
    add_test(NAME strix-execution-memory-tests COMMAND strix-execution-memory-tests)

    add_executable(strix-execution-resources-tests tests/execution_resources_tests.cpp)
    target_link_libraries(strix-execution-resources-tests PRIVATE strix_core)
    strix_target_warnings(strix-execution-resources-tests)
    add_test(NAME strix-execution-resources-tests COMMAND strix-execution-resources-tests)

    add_executable(strix-runtime-owner-inventory-receipt-tests tests/runtime_owner_inventory_receipt_tests.cpp)
    target_link_libraries(strix-runtime-owner-inventory-receipt-tests PRIVATE strix_core)
    strix_target_warnings(strix-runtime-owner-inventory-receipt-tests)
    add_test(NAME strix-runtime-owner-inventory-receipt-tests COMMAND strix-runtime-owner-inventory-receipt-tests)

    add_executable(strix-linux-uma-observation-tests tests/linux_uma_observation_tests.cpp)
    target_link_libraries(strix-linux-uma-observation-tests PRIVATE strix_core)
    strix_target_warnings(strix-linux-uma-observation-tests)
    add_test(NAME strix-linux-uma-observation-tests COMMAND strix-linux-uma-observation-tests)

    add_executable(strix-distributed-stage-state-tests tests/distributed_stage_state_tests.cpp)
    target_link_libraries(strix-distributed-stage-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-stage-state-tests)
    add_test(NAME strix-distributed-stage-state-tests COMMAND strix-distributed-stage-state-tests)

    add_executable(strix-distributed-greedy-control-tests tests/distributed_greedy_control_tests.cpp)
    target_link_libraries(strix-distributed-greedy-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-greedy-control-tests)
    add_test(NAME strix-distributed-greedy-control-tests COMMAND strix-distributed-greedy-control-tests)

    add_executable(strix-distributed-token-transaction-tests tests/distributed_token_transaction_tests.cpp)
    target_link_libraries(strix-distributed-token-transaction-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-token-transaction-tests)
    add_test(NAME strix-distributed-token-transaction-tests COMMAND strix-distributed-token-transaction-tests)

    add_executable(strix-distributed-placement-tests tests/distributed_placement_tests.cpp)
    target_link_libraries(strix-distributed-placement-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-placement-tests)
    add_test(NAME strix-distributed-placement-tests COMMAND strix-distributed-placement-tests)

    add_executable(strix-distributed-flow-tests tests/distributed_flow_tests.cpp)
    target_link_libraries(strix-distributed-flow-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-flow-tests)
    add_test(NAME strix-distributed-flow-tests COMMAND strix-distributed-flow-tests)

    add_executable(strix-distributed-data-plane-tests tests/distributed_data_plane_tests.cpp)
    target_link_libraries(strix-distributed-data-plane-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-data-plane-tests)
    add_test(NAME strix-distributed-data-plane-tests COMMAND strix-distributed-data-plane-tests)

    add_executable(strix-distributed-runtime-tests tests/distributed_runtime_tests.cpp)
    target_link_libraries(strix-distributed-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-runtime-tests)
    add_test(NAME strix-distributed-runtime-tests COMMAND strix-distributed-runtime-tests)

    add_executable(strix-distributed-native-backend-tests tests/distributed_native_backend_tests.cpp)
    target_link_libraries(strix-distributed-native-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-native-backend-tests)
    add_test(NAME strix-distributed-native-backend-tests COMMAND strix-distributed-native-backend-tests)

    add_executable(strix-distributed-remote-control-tests tests/distributed_remote_control_tests.cpp)
    target_link_libraries(strix-distributed-remote-control-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-remote-control-tests)
    add_test(NAME strix-distributed-remote-control-tests COMMAND strix-distributed-remote-control-tests)

    add_executable(strix-distributed-control-wire-tests tests/distributed_control_wire_tests.cpp)
    target_link_libraries(strix-distributed-control-wire-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-control-wire-tests)
    add_test(NAME strix-distributed-control-wire-tests COMMAND strix-distributed-control-wire-tests)

    add_executable(strix-distributed-control-server-tests tests/distributed_control_server_tests.cpp)
    target_link_libraries(strix-distributed-control-server-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-control-server-tests)
    add_test(NAME strix-distributed-control-server-tests COMMAND strix-distributed-control-server-tests)

    add_executable(strix-distributed-control-end-to-end-tests tests/distributed_control_end_to_end_tests.cpp)
    target_link_libraries(strix-distributed-control-end-to-end-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-control-end-to-end-tests)
    add_test(NAME strix-distributed-control-end-to-end-tests COMMAND strix-distributed-control-end-to-end-tests)

    add_executable(strix-distributed-control-receiver-tests tests/distributed_control_receiver_tests.cpp)
    target_link_libraries(strix-distributed-control-receiver-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-control-receiver-tests)
    add_test(NAME strix-distributed-control-receiver-tests COMMAND strix-distributed-control-receiver-tests)

    add_executable(strix-distributed-tcp-control-stream-tests tests/distributed_tcp_control_stream_tests.cpp)
    target_link_libraries(strix-distributed-tcp-control-stream-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-tcp-control-stream-tests)
    add_test(NAME strix-distributed-tcp-control-stream-tests COMMAND strix-distributed-tcp-control-stream-tests)

    add_executable(strix-distributed-session-tests tests/distributed_session_tests.cpp)
    target_link_libraries(strix-distributed-session-tests PRIVATE strix_core)
    strix_target_warnings(strix-distributed-session-tests)
    add_test(NAME strix-distributed-session-tests COMMAND strix-distributed-session-tests)

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

    add_executable(strix-speculative-verify-launch-dispatch-tests tests/speculative_verify_launch_dispatch_tests.cpp)
    target_link_libraries(strix-speculative-verify-launch-dispatch-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-launch-dispatch-tests)
    add_test(NAME strix-speculative-verify-launch-dispatch-tests COMMAND strix-speculative-verify-launch-dispatch-tests)

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

    add_executable(strix-serial-request-tests tests/serial_request_tests.cpp)
    target_link_libraries(strix-serial-request-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-request-tests)
    add_test(NAME strix-serial-request-tests COMMAND strix-serial-request-tests)

    add_executable(strix-serial-prefix-lifecycle-tests tests/serial_prefix_lifecycle_tests.cpp)
    target_link_libraries(strix-serial-prefix-lifecycle-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-prefix-lifecycle-tests)
    add_test(NAME strix-serial-prefix-lifecycle-tests COMMAND strix-serial-prefix-lifecycle-tests)

    add_executable(strix-serial-prefix-cache-tests tests/serial_prefix_cache_tests.cpp)
    target_link_libraries(strix-serial-prefix-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-prefix-cache-tests)
    add_test(NAME strix-serial-prefix-cache-tests COMMAND strix-serial-prefix-cache-tests)

    add_executable(strix-serial-decode-replay-tests tests/serial_decode_replay_tests.cpp)
    target_link_libraries(strix-serial-decode-replay-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-decode-replay-tests)
    add_test(NAME strix-serial-decode-replay-tests COMMAND strix-serial-decode-replay-tests)

    add_executable(strix-serial-decode-reference-tests tests/serial_decode_reference_tests.cpp)
    target_link_libraries(strix-serial-decode-reference-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-decode-reference-tests)
    add_test(NAME strix-serial-decode-reference-tests COMMAND strix-serial-decode-reference-tests)

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

    add_executable(strix-session-snapshot-memory-tests tests/session_snapshot_memory_tests.cpp)
    target_link_libraries(strix-session-snapshot-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-snapshot-memory-tests)
    add_test(NAME strix-session-snapshot-memory-tests COMMAND strix-session-snapshot-memory-tests)

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

    add_executable(strix-speculation-sampling-tests tests/speculation_sampling_tests.cpp)
    target_link_libraries(strix-speculation-sampling-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-tests)
    add_test(NAME strix-speculation-sampling-tests COMMAND strix-speculation-sampling-tests)

    add_executable(strix-speculation-sampling-distribution-tests tests/speculation_sampling_distribution_tests.cpp)
    target_link_libraries(strix-speculation-sampling-distribution-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-distribution-tests)
    add_test(NAME strix-speculation-sampling-distribution-tests COMMAND strix-speculation-sampling-distribution-tests)

    add_executable(strix-speculation-sampling-cycle-tests tests/speculation_sampling_cycle_tests.cpp)
    target_link_libraries(strix-speculation-sampling-cycle-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-cycle-tests)
    add_test(NAME strix-speculation-sampling-cycle-tests COMMAND strix-speculation-sampling-cycle-tests)

    add_executable(strix-speculation-sampling-executor-tests tests/speculation_sampling_executor_tests.cpp)
    target_link_libraries(strix-speculation-sampling-executor-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-sampling-executor-tests)
    add_test(NAME strix-speculation-sampling-executor-tests COMMAND strix-speculation-sampling-executor-tests)

    add_executable(strix-speculation-rng-tests tests/speculation_rng_tests.cpp)
    target_link_libraries(strix-speculation-rng-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-rng-tests)
    add_test(NAME strix-speculation-rng-tests COMMAND strix-speculation-rng-tests)

    add_executable(strix-speculation-controller-tests tests/speculation_controller_tests.cpp)
    target_link_libraries(strix-speculation-controller-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-controller-tests)
    add_test(NAME strix-speculation-controller-tests COMMAND strix-speculation-controller-tests)

    add_executable(strix-speculation-measurement-tests tests/speculation_measurement_tests.cpp)
    target_link_libraries(strix-speculation-measurement-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-measurement-tests)
    add_test(NAME strix-speculation-measurement-tests COMMAND strix-speculation-measurement-tests)

    add_executable(strix-speculation-cycle-tests tests/speculation_cycle_tests.cpp)
    target_link_libraries(strix-speculation-cycle-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-cycle-tests)
    add_test(NAME strix-speculation-cycle-tests COMMAND strix-speculation-cycle-tests)

    add_executable(strix-prompt-lookup-draft-tests tests/prompt_lookup_draft_tests.cpp)
    target_link_libraries(strix-prompt-lookup-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-prompt-lookup-draft-tests)
    add_test(NAME strix-prompt-lookup-draft-tests COMMAND strix-prompt-lookup-draft-tests)

    add_executable(strix-qsa-selection-plan-tests tests/qsa_selection_plan_tests.cpp)
    target_link_libraries(strix-qsa-selection-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-selection-plan-tests)
    add_test(NAME strix-qsa-selection-plan-tests COMMAND strix-qsa-selection-plan-tests)

    add_executable(strix-compiled-mtp-draft-plan-tests tests/compiled_mtp_draft_plan_tests.cpp)
    target_link_libraries(strix-compiled-mtp-draft-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-mtp-draft-plan-tests)
    add_test(NAME strix-compiled-mtp-draft-plan-tests COMMAND strix-compiled-mtp-draft-plan-tests)

    add_executable(strix-mtp-draft-cycle-input-tests tests/mtp_draft_cycle_input_tests.cpp)
    target_link_libraries(strix-mtp-draft-cycle-input-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-cycle-input-tests)
    add_test(NAME strix-mtp-draft-cycle-input-tests COMMAND strix-mtp-draft-cycle-input-tests)

    add_executable(strix-mtp-draft-activation-tests tests/mtp_draft_activation_tests.cpp)
    target_link_libraries(strix-mtp-draft-activation-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-activation-tests)
    add_test(NAME strix-mtp-draft-activation-tests COMMAND strix-mtp-draft-activation-tests)

    add_executable(strix-mtp-forward-replay-tests tests/mtp_forward_replay_tests.cpp)
    target_link_libraries(strix-mtp-forward-replay-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-replay-tests)
    add_test(NAME strix-mtp-forward-replay-tests COMMAND strix-mtp-forward-replay-tests)

    add_executable(strix-mtp-forward-golden-file-tests tests/mtp_forward_golden_file_tests.cpp)
    target_link_libraries(strix-mtp-forward-golden-file-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-golden-file-tests)
    add_test(NAME strix-mtp-forward-golden-file-tests COMMAND strix-mtp-forward-golden-file-tests)

    add_executable(strix-mtp-target-window-policy-tests tests/mtp_target_window_policy_tests.cpp)
    target_link_libraries(strix-mtp-target-window-policy-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-target-window-policy-tests)
    add_test(NAME strix-mtp-target-window-policy-tests COMMAND strix-mtp-target-window-policy-tests)

    add_executable(strix-mtp-sidecar-binding-tests tests/mtp_sidecar_binding_tests.cpp)
    target_link_libraries(strix-mtp-sidecar-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-sidecar-binding-tests)
    add_test(NAME strix-mtp-sidecar-binding-tests COMMAND strix-mtp-sidecar-binding-tests)

    add_executable(strix-mtp-forward-semantics-tests tests/mtp_forward_semantics_tests.cpp)
    target_link_libraries(strix-mtp-forward-semantics-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-semantics-tests)
    add_test(NAME strix-mtp-forward-semantics-tests COMMAND strix-mtp-forward-semantics-tests)

    add_executable(strix-mtp-forward-contract-tests tests/mtp_forward_contract_tests.cpp)
    target_link_libraries(strix-mtp-forward-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-contract-tests)
    add_test(NAME strix-mtp-forward-contract-tests COMMAND strix-mtp-forward-contract-tests)

    add_executable(strix-mtp-forward-runtime-tests tests/mtp_forward_runtime_tests.cpp)
    target_link_libraries(strix-mtp-forward-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-runtime-tests)
    add_test(NAME strix-mtp-forward-runtime-tests COMMAND strix-mtp-forward-runtime-tests)

    add_executable(strix-mtp-draft-readiness-tests tests/mtp_draft_readiness_tests.cpp)
    target_link_libraries(strix-mtp-draft-readiness-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-readiness-tests)
    add_test(NAME strix-mtp-draft-readiness-tests COMMAND strix-mtp-draft-readiness-tests)

    add_executable(strix-mtp-draft-qualification-tests tests/mtp_draft_qualification_tests.cpp)
    target_link_libraries(strix-mtp-draft-qualification-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-qualification-tests)
    add_test(NAME strix-mtp-draft-qualification-tests COMMAND strix-mtp-draft-qualification-tests)

    add_executable(strix-qualified-mtp-draft-backend-tests tests/qualified_mtp_draft_backend_tests.cpp)
    target_link_libraries(strix-qualified-mtp-draft-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-qualified-mtp-draft-backend-tests)
    add_test(NAME strix-qualified-mtp-draft-backend-tests COMMAND strix-qualified-mtp-draft-backend-tests)

    add_executable(strix-qualified-mtp-forward-runtime-tests tests/qualified_mtp_forward_runtime_tests.cpp)
    target_link_libraries(strix-qualified-mtp-forward-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-qualified-mtp-forward-runtime-tests)
    add_test(NAME strix-qualified-mtp-forward-runtime-tests COMMAND strix-qualified-mtp-forward-runtime-tests)

    add_executable(strix-mtp-attention-contract-tests tests/mtp_attention_contract_tests.cpp)
    target_link_libraries(strix-mtp-attention-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-attention-contract-tests)
    add_test(NAME strix-mtp-attention-contract-tests COMMAND strix-mtp-attention-contract-tests)

    add_executable(strix-mtp-memory-state-tests tests/mtp_memory_state_tests.cpp)
    target_link_libraries(strix-mtp-memory-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-memory-state-tests)
    add_test(NAME strix-mtp-memory-state-tests COMMAND strix-mtp-memory-state-tests)

    add_executable(strix-mtp-draft-state-contract-tests tests/mtp_draft_state_contract_tests.cpp)
    target_link_libraries(strix-mtp-draft-state-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-state-contract-tests)
    add_test(NAME strix-mtp-draft-state-contract-tests COMMAND strix-mtp-draft-state-contract-tests)

    add_executable(strix-mtp-draft-state-memory-tests tests/mtp_draft_state_memory_tests.cpp)
    target_link_libraries(strix-mtp-draft-state-memory-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-state-memory-tests)
    add_test(NAME strix-mtp-draft-state-memory-tests COMMAND strix-mtp-draft-state-memory-tests)

    add_executable(strix-mtp-output-contract-tests tests/mtp_output_contract_tests.cpp)
    target_link_libraries(strix-mtp-output-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-output-contract-tests)
    add_test(NAME strix-mtp-output-contract-tests COMMAND strix-mtp-output-contract-tests)

    add_executable(strix-mtp-input-fusion-tests tests/mtp_input_fusion_tests.cpp)
    target_link_libraries(strix-mtp-input-fusion-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-input-fusion-tests)
    add_test(NAME strix-mtp-input-fusion-tests COMMAND strix-mtp-input-fusion-tests)

    add_executable(strix-mtp-draft-tensor-audit-tests tests/mtp_draft_tensor_audit_tests.cpp)
    target_link_libraries(strix-mtp-draft-tensor-audit-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-tensor-audit-tests)
    add_test(NAME strix-mtp-draft-tensor-audit-tests COMMAND strix-mtp-draft-tensor-audit-tests)

    add_executable(strix-mtp-draft-contract-tests tests/mtp_draft_contract_tests.cpp)
    target_link_libraries(strix-mtp-draft-contract-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-contract-tests)
    add_test(NAME strix-mtp-draft-contract-tests COMMAND strix-mtp-draft-contract-tests)

    add_executable(strix-mtp-draft-binding-tests tests/mtp_draft_binding_tests.cpp)
    target_link_libraries(strix-mtp-draft-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-binding-tests)
    add_test(NAME strix-mtp-draft-binding-tests COMMAND strix-mtp-draft-binding-tests)

    add_executable(strix-draft-proposal-state-tests tests/draft_proposal_state_tests.cpp)
    target_link_libraries(strix-draft-proposal-state-tests PRIVATE strix_core)
    strix_target_warnings(strix-draft-proposal-state-tests)
    add_test(NAME strix-draft-proposal-state-tests COMMAND strix-draft-proposal-state-tests)

    add_executable(strix-speculation-executor-tests tests/speculation_executor_tests.cpp)
    target_link_libraries(strix-speculation-executor-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-executor-tests)
    add_test(NAME strix-speculation-executor-tests COMMAND strix-speculation-executor-tests)

    add_executable(strix-compiled-speculation-backend-tests tests/compiled_speculation_backend_tests.cpp)
    target_link_libraries(strix-compiled-speculation-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-speculation-backend-tests)
    add_test(NAME strix-compiled-speculation-backend-tests COMMAND strix-compiled-speculation-backend-tests)

    add_executable(strix-compiled-sampled-speculation-backend-tests tests/compiled_sampled_speculation_backend_tests.cpp)
    target_link_libraries(strix-compiled-sampled-speculation-backend-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-sampled-speculation-backend-tests)
    add_test(NAME strix-compiled-sampled-speculation-backend-tests COMMAND strix-compiled-sampled-speculation-backend-tests)

    add_executable(strix-speculative-verify-plan-tests tests/speculative_verify_plan_tests.cpp)
    target_link_libraries(strix-speculative-verify-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-plan-tests)
    add_test(NAME strix-speculative-verify-plan-tests COMMAND strix-speculative-verify-plan-tests)

    add_executable(strix-speculative-verify-binding-tests tests/speculative_verify_binding_tests.cpp)
    target_link_libraries(strix-speculative-verify-binding-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-binding-tests)
    add_test(NAME strix-speculative-verify-binding-tests COMMAND strix-speculative-verify-binding-tests)

    add_executable(strix-compiled-speculative-verify-tests tests/compiled_speculative_verify_tests.cpp)
    target_link_libraries(strix-compiled-speculative-verify-tests PRIVATE strix_core)
    strix_target_warnings(strix-compiled-speculative-verify-tests)
    add_test(NAME strix-compiled-speculative-verify-tests COMMAND strix-compiled-speculative-verify-tests)

    add_executable(strix-speculative-verify-runtime-tests tests/speculative_verify_runtime_tests.cpp)
    target_link_libraries(strix-speculative-verify-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-runtime-tests)
    add_test(NAME strix-speculative-verify-runtime-tests COMMAND strix-speculative-verify-runtime-tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-ctest-evidence-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest_evidence_tests.py)
        add_test(NAME strix-queue-trace-evidence-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/queue_trace_evidence_tests.py)
        add_test(NAME strix-runtime-inventory-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/runtime_inventory_tests.py)
    endif()
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-dual-transport-runner-tests
            COMMAND ${Python3_EXECUTABLE} -B -W error::ResourceWarning ${CMAKE_SOURCE_DIR}/lab/transport/test_dual_tbstream.py)
        add_test(NAME strix-private-kernel-preparation-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/lab/direct_gpu_stream/kernel/test_build_private_kernel.py)
        add_test(NAME strix-native-numerical-campaign-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/lab/native_campaign/numerical/test_campaign.py)
        add_test(NAME strix-native-raw-import-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/lab/native_campaign/numerical/test_raw_import.py)
        add_test(
            NAME strix-python-oracle-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/oracles/test_qwen4_exp_oracle.py
        )
        add_test(
            NAME strix-ple-qualification-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/lab/sparse_lookup/test_qualify_serial_ple.py
        )
        add_test(
            NAME strix-ple-policy-sweep-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/lab/sparse_lookup/test_sweep_serial_ple.py
        )
    endif()
endif()


if(STRIX_ENABLE_HIP_RUNTIME)
    enable_language(HIP)

    # CMake's HIP language drives device compilation directly. ROCm 7.x
    # development images do not necessarily ship the legacy hipConfig.cmake /
    # hip::host package target, so locate the runtime header/library from the
    # toolkit instead of requiring that package surface.
    find_path(
        STRIX_HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        HINTS "$ENV{ROCM_PATH}" /opt/rocm
        PATH_SUFFIXES include
        REQUIRED
    )
    find_library(
        STRIX_HIP_RUNTIME_LIBRARY
        NAMES amdhip64
        HINTS "$ENV{ROCM_PATH}" /opt/rocm
        PATH_SUFFIXES lib lib64
        REQUIRED
    )

    if(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
        # Only these isolated arithmetic TUs receive the strict experimental flags.
        add_library(strix_q4k_a8_hip_probe STATIC src/hip_moe_q4k_a8_probe.hip src/hip_moe_q4k_a8_known_rows.hip)
        target_link_libraries(strix_q4k_a8_hip_probe PUBLIC strix_q4k_a8_contract "${STRIX_HIP_RUNTIME_LIBRARY}")
        foreach(target strix_q4k_a8_hip_probe)
            target_include_directories(${target} PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_compile_options(${target} PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(${target})
        endforeach()
        if(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE)
            add_library(strix_q4k_a8_dot4_compile_probe OBJECT lab/q4k_a8/dot4_compile_probe.hip)
            target_include_directories(strix_q4k_a8_dot4_compile_probe PRIVATE include "${STRIX_HIP_INCLUDE_DIR}")
            set_target_properties(strix_q4k_a8_dot4_compile_probe PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_compile_options(strix_q4k_a8_dot4_compile_probe PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix_q4k_a8_dot4_compile_probe)
        endif()
        if(BUILD_TESTING AND STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE)
            add_executable(strix-hip-q4k-a8-probe-tests tests/hip_moe_q4k_a8_probe_tests.hip $<TARGET_OBJECTS:strix_q4k_a8_dot4_compile_probe>)
            target_link_libraries(strix-hip-q4k-a8-probe-tests PRIVATE strix_q4k_a8_hip_probe strix_core)
            set_target_properties(strix-hip-q4k-a8-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_compile_options(strix-hip-q4k-a8-probe-tests PRIVATE -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q4k-a8-probe-tests)
            add_test(NAME strix-hip-q4k-a8-probe-tests COMMAND strix-hip-q4k-a8-probe-tests)
        endif()
    endif()

    add_library(strix_hip_runtime STATIC
        src/hip_host_upload_backend.cpp
        src/hip_runtime_allocator.cpp
        src/hip_sampled_target_logits_reader.cpp
        src/hip_session_snapshot_transfer.cpp
        src/hip_model_residency.cpp
        src/hip_launcher_catalog.cpp
        src/hip_runtime_launchers.hip
        src/hip_moe_route_launcher.hip
        src/hip_moe_experts_launcher.hip
        src/hip_ple_launcher.hip
        src/hip_mtp_resources.hip
        src/hip_mtp_runtime.hip
        src/hip_mtp_target_window.hip
        src/hip_mtp_target_checkpoint.hip
        src/hip_mtp_window_verifier.hip
        src/hip_mtp_target_hidden.hip
        src/hip_mtp_greedy_runtime.hip
        src/hip_mtp_asset_loader.hip
        src/hip_mtp_input_fusion.hip
        src/hip_mtp_forward_replay.hip
        src/hip_embedding_launcher.hip
        src/hip_gdn_launcher.hip
        src/hip_known_prefix_window.hip
        src/hip_known_prompt_backend.cpp
        src/hip_prompt_lookup_runtime.hip
        src/hip_serving_greedy_backend.hip
        src/hip_serving_cohort.hip
        src/hip_serving_prefix_cache.hip
        src/hip_event_queue_trace.hip
        src/hip_distributed_stage.hip
        src/hip_distributed_transaction.hip
        src/hip_distributed_sequence.hip
        src/hip_distributed_stage_resources.hip
        src/hip_distributed_greedy.hip
        src/hip_hyper_mix_launcher.hip
        src/hip_indexed_attention_launcher.hip
        src/hip_qsa_selection_diagnostic.hip
        src/hip_serial_graph.hip
        src/hip_direct_transport_gpu_export.cpp
        src/hip_direct_gpu_pool_allocator.cpp
        src/hip_direct_gpu_stream_gpu.cpp
        src/hip_integer_projection.hip
        src/hip_integer_projection_runtime.hip
        src/hip_lm_head_launcher.hip
        src/hip_generation_output.hip
        src/hip_moe_overlap_dispatcher.hip
        src/hip_shared_expert_launcher.hip
    )
    target_include_directories(
        strix_hip_runtime
        PUBLIC
        include
        "${STRIX_HIP_INCLUDE_DIR}"
    )
    target_compile_features(strix_hip_runtime PUBLIC cxx_std_20)
    # Host-side redirects only: SDK declarations are included before macros;
    # kernels are never counted as submits by this observer.
    target_compile_options(strix_hip_runtime PRIVATE
        "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
    target_compile_definitions(
        strix_hip_runtime
        PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__=1>
    )
    set_target_properties(
        strix_hip_runtime
        PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
        HIP_ARCHITECTURES gfx1151
    )
    target_link_libraries(
        strix_hip_runtime
        PUBLIC
        strix_core
        "${STRIX_HIP_RUNTIME_LIBRARY}"
    )
    if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
        target_link_libraries(strix_hip_runtime PUBLIC strix_q4k_a8_hip_probe)
        target_compile_definitions(strix_hip_runtime PUBLIC STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    endif()
    set_source_files_properties(src/hip_qsa_selection_diagnostic.hip
        PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64;-fno-fast-math")
    set_source_files_properties(src/hip_mtp_input_fusion.hip
        PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off")
    strix_target_warnings(strix_hip_runtime)
    if(BUILD_TESTING)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GDN_RECURRENT_LAYOUT_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_SHARED_EXPERT_ROWS_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_QSA_GQA_REUSE_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC=1)
        foreach(mtp_fixture input-fusion forward-replay target-checkpoint)
            string(REPLACE "-" "_" mtp_fixture_file "${mtp_fixture}")
            add_executable(strix-hip-mtp-${mtp_fixture}-tests tests/hip_mtp_${mtp_fixture_file}_tests.hip)
            target_link_libraries(strix-hip-mtp-${mtp_fixture}-tests PRIVATE strix_hip_runtime)
            set_target_properties(strix-hip-mtp-${mtp_fixture}-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_compile_options(strix-hip-mtp-${mtp_fixture}-tests PRIVATE
                -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off)
            strix_target_warnings(strix-hip-mtp-${mtp_fixture}-tests)
            add_test(NAME strix-hip-mtp-${mtp_fixture}-tests COMMAND strix-hip-mtp-${mtp_fixture}-tests)
            set_tests_properties(strix-hip-mtp-${mtp_fixture}-tests PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
        # Callable fixture requires actual independently supplied component goldens.
        # Building this source library does not execute or qualify the components.
        add_library(strix_mtp_projection_component_fixtures STATIC tests/hip_mtp_projection_components.hip tests/hip_mtp_shared_expert_components.hip tests/hip_mtp_target_window_components.hip tests/hip_mtp_greedy_cycle_components.hip)
        target_link_libraries(strix_mtp_projection_component_fixtures PRIVATE strix_hip_runtime)
        set_target_properties(strix_mtp_projection_component_fixtures PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        strix_target_warnings(strix_mtp_projection_component_fixtures)
        add_library(strix_qsa_selection_diagnostic INTERFACE)
        target_link_libraries(strix_qsa_selection_diagnostic INTERFACE strix_hip_runtime)
        add_executable(strix-hip-qsa-selection-diagnostic-tests tests/hip_qsa_selection_diagnostic_tests.hip)
        target_link_libraries(strix-hip-qsa-selection-diagnostic-tests PRIVATE strix_qsa_selection_diagnostic)
        set_target_properties(strix-hip-qsa-selection-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        strix_target_warnings(strix-hip-qsa-selection-diagnostic-tests)
        add_test(NAME strix-hip-qsa-selection-diagnostic-tests COMMAND strix-hip-qsa-selection-diagnostic-tests)
    endif()
    set_source_files_properties(src/hip_integer_projection.hip src/hip_integer_projection_runtime.hip
        PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64;-ffp-contract=off")

    add_executable(
        strix-hip-runtime-allocator
        apps/hip_runtime_allocator_main.cpp
    )

    add_executable(
        strix-hip-model-residency
        apps/hip_model_residency_main.cpp
    )

    add_executable(
        strix-hip-decode-token
        apps/hip_decode_token_main.cpp
    )
    # This embeds DECLARED configuration only. Heterogeneous per-source HIP
    # argv is not represented completely, so live flags SHA stays unknown.
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" STRIX_DECLARED_CMAKE_SOURCE_SHA)
    string(HEX "cmake=${STRIX_DECLARED_CMAKE_SOURCE_SHA};cxx=${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION}/${CMAKE_CXX_COMPILER};hip=${CMAKE_HIP_COMPILER_ID}/${CMAKE_HIP_COMPILER_VERSION}/${CMAKE_HIP_COMPILER};build=${CMAKE_BUILD_TYPE};cxxflags=${CMAKE_CXX_FLAGS};hipflags=${CMAKE_HIP_FLAGS};debug=${CMAKE_HIP_FLAGS_DEBUG};release=${CMAKE_HIP_FLAGS_RELEASE};reldeb=${CMAKE_HIP_FLAGS_RELWITHDEBINFO};minsize=${CMAKE_HIP_FLAGS_MINSIZEREL};per-source=unresolved" STRIX_DECLARED_BUILD_CONTEXT_HEX)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(apps/offline_tuning_build_context.hpp.in
        "${CMAKE_CURRENT_BINARY_DIR}/generated/offline_tuning_build_context.hpp" @ONLY)
    target_include_directories(strix-hip-decode-token PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(
        strix-hip-decode-token
        PRIVATE strix_hip_runtime
    )
    strix_target_warnings(
        strix-hip-decode-token
    )
    target_compile_options(strix-hip-decode-token PRIVATE
        "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
    if(BUILD_TESTING)
        add_test(NAME strix-virtual-q8-gdn-output-cli-tests
            COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:strix-hip-decode-token>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_virtual_q8_gdn_output_cli.cmake)
        add_test(NAME strix-virtual-q8-cli-duplicate-tests
            COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:strix-hip-decode-token>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_virtual_q8_cli.cmake)
    endif()
    target_link_libraries(
        strix-hip-model-residency
        PRIVATE strix_hip_runtime
    )
    strix_target_warnings(
        strix-hip-model-residency
    )
    target_link_libraries(
        strix-hip-runtime-allocator
        PRIVATE strix_hip_runtime
    )
    strix_target_warnings(strix-hip-runtime-allocator)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(strix-direct-gpu-commission
            lab/direct_gpu_stream/commission_main.hip
            lab/direct_gpu_stream/bounded_tcp_control.cpp)
        set_target_properties(strix-direct-gpu-commission PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-direct-gpu-commission PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-direct-gpu-commission)
    endif()

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
        add_executable(strix-hip-graph-inputs-tests tests/hip_graph_inputs_tests.hip)
        add_executable(strix-hip-integer-projection-runtime-tests tests/hip_integer_projection_runtime_tests.hip)
        add_executable(strix-hip-serial-graph-tests tests/hip_serial_graph_tests.hip)
        set_target_properties(strix-hip-serial-graph-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-serial-graph-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-serial-graph-tests)
        add_test(NAME strix-hip-serial-graph-tests COMMAND strix-hip-serial-graph-tests)
        set_tests_properties(strix-hip-serial-graph-tests PROPERTIES SKIP_RETURN_CODE 77)
        set_target_properties(strix-hip-integer-projection-runtime-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-integer-projection-runtime-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-integer-projection-runtime-tests PRIVATE -ffp-contract=off)
        strix_target_warnings(strix-hip-integer-projection-runtime-tests)
        add_test(NAME strix-hip-integer-projection-runtime-tests COMMAND strix-hip-integer-projection-runtime-tests)
        set_tests_properties(strix-hip-integer-projection-runtime-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-direct-transport-gpu-export-tests tests/hip_direct_transport_gpu_export_tests.cpp)
        add_executable(strix-hip-direct-gpu-pool-allocator-tests tests/hip_direct_gpu_pool_allocator_tests.cpp)
        add_executable(strix-hip-direct-gpu-stream-gpu-tests tests/hip_direct_gpu_stream_gpu_tests.cpp)
        target_link_libraries(strix-hip-direct-gpu-stream-gpu-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-direct-gpu-stream-gpu-tests)
        add_test(NAME strix-hip-direct-gpu-stream-gpu-tests COMMAND strix-hip-direct-gpu-stream-gpu-tests)
        target_link_libraries(strix-hip-direct-gpu-pool-allocator-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-direct-gpu-pool-allocator-tests)
        add_test(NAME strix-hip-direct-gpu-pool-allocator-tests COMMAND strix-hip-direct-gpu-pool-allocator-tests)
        add_executable(strix-hip-launch-context-tests tests/hip_launch_context_tests.cpp)
        target_link_libraries(strix-hip-launch-context-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-launch-context-tests)
        add_test(NAME strix-hip-launch-context-tests COMMAND strix-hip-launch-context-tests)
        target_link_libraries(strix-hip-direct-transport-gpu-export-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-direct-transport-gpu-export-tests)
        add_test(NAME strix-hip-direct-transport-gpu-export-tests COMMAND strix-hip-direct-transport-gpu-export-tests)
        add_executable(strix-hip-graph-attention-state-tests tests/hip_graph_attention_state_tests.hip)
        set_target_properties(strix-hip-graph-attention-state-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-graph-attention-state-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-graph-attention-state-tests)
        add_test(NAME strix-hip-graph-attention-state-tests COMMAND strix-hip-graph-attention-state-tests)
        set_target_properties(strix-hip-graph-inputs-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-graph-inputs-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-graph-inputs-tests)
        add_test(NAME strix-hip-graph-inputs-tests COMMAND strix-hip-graph-inputs-tests)
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

        add_test(NAME strix-lane-chain-cli-tests COMMAND ${CMAKE_COMMAND} -DAPP=$<TARGET_FILE:strix-hip-decode-token> -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_lane_chain_cli.cmake)
        add_test(NAME strix-hip-lane-chain-hyper-integration-tests COMMAND strix-hip-hyper-mix-launcher-tests lane-chain)
        add_test(NAME strix-hip-hc-next-tile-integration-tests COMMAND strix-hip-hyper-mix-launcher-tests next-tile)
        add_executable(strix-hip-hc-projection-next-tile-tests tests/hip_hc_projection_next_tile_tests.hip)
        set_target_properties(strix-hip-hc-projection-next-tile-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-hc-projection-next-tile-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-hc-projection-next-tile-tests)
        add_test(NAME strix-hip-hc-projection-next-tile-tests COMMAND strix-hip-hc-projection-next-tile-tests)
        add_executable(strix-hip-lane-chain-hyper-output-tests tests/hip_lane_chain_hyper_output_tests.hip)
        set_target_properties(strix-hip-lane-chain-hyper-output-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-lane-chain-hyper-output-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-lane-chain-hyper-output-tests)
        add_test(NAME strix-hip-lane-chain-hyper-output-tests COMMAND strix-hip-lane-chain-hyper-output-tests)
        add_executable(strix-hip-parallel-hyper-projection-tests tests/hip_parallel_hyper_projection_tests.hip)
        set_target_properties(strix-hip-parallel-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-parallel-hyper-projection-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-parallel-hyper-projection-tests)
        add_test(NAME strix-hip-parallel-hyper-projection-tests COMMAND strix-hip-parallel-hyper-projection-tests)
        add_executable(strix-hip-splitk-hyper-integration-tests tests/hip_splitk_hyper_integration_tests.cpp)
        target_link_libraries(strix-hip-splitk-hyper-integration-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-splitk-hyper-integration-tests)
        add_test(NAME strix-hip-splitk-hyper-integration-tests COMMAND strix-hip-splitk-hyper-integration-tests)

        add_executable(strix-hip-lane-chain-hyper-projection-tests tests/hip_lane_chain_hyper_projection_tests.hip)
        set_target_properties(strix-hip-lane-chain-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-lane-chain-hyper-projection-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-lane-chain-hyper-projection-tests)
        add_test(NAME strix-hip-lane-chain-hyper-projection-tests COMMAND strix-hip-lane-chain-hyper-projection-tests)
        add_library(strix_hc_repacked_diagnostic STATIC src/hip_hc_repacked_diagnostic.hip)
        target_include_directories(strix_hc_repacked_diagnostic PUBLIC include)
        set_target_properties(strix_hc_repacked_diagnostic PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix_hc_repacked_diagnostic PUBLIC strix_core "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix_hc_repacked_diagnostic)
        add_executable(strix-hip-repacked-hyper-projection-tests tests/hip_repacked_hyper_projection_tests.hip)
        set_target_properties(strix-hip-repacked-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-repacked-hyper-projection-tests PRIVATE strix_hip_runtime strix_hc_repacked_down_pack strix_hc_repacked_diagnostic)
        strix_target_warnings(strix-hip-repacked-hyper-projection-tests)
        # Always build the diagnostic binary; only an externally admitted subset
        # registers its research test. CTest receives the cache value explicitly,
        # independent of the parent environment stripped by native workers.
        if(NOT "${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}" STREQUAL "")
            add_test(NAME strix-hip-repacked-hyper-projection-tests COMMAND strix-hip-repacked-hyper-projection-tests)
            set_tests_properties(strix-hip-repacked-hyper-projection-tests PROPERTIES
                ENVIRONMENT "STRIX_HC_REPACKED_ADMITTED_SCHEDULES=${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}"
                LABELS "research;externally-admitted-hc")
            message(STATUS "HC repacked research CTest registered for externally admitted schedules: ${STRIX_HC_REPACKED_ADMITTED_SCHEDULES}")
        else()
            message(STATUS "HC repacked research CTest NOT registered: no schedules externally admitted; binary and portable pack contracts remain built")
        endif()
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
        add_test(NAME strix-hip-hyper-row-prefetch-tests
                 COMMAND strix-hip-hyper-mix-launcher-tests prefetch)
        add_test(NAME strix-hip-hyper-output-prefetch-tests
                 COMMAND strix-hip-hyper-mix-launcher-tests output-prefetch)

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

        add_executable(
            strix-hip-shared-expert-launcher-tests
            tests/hip_shared_expert_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-shared-expert-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-shared-expert-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-shared-expert-launcher-tests
        )
        add_test(
            NAME strix-hip-shared-expert-launcher-tests
            COMMAND strix-hip-shared-expert-launcher-tests
        )

        add_executable(strix-hip-shared-known-window-rows-tests tests/hip_shared_known_window_rows_tests.hip)
        set_target_properties(strix-hip-shared-known-window-rows-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-shared-known-window-rows-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-shared-known-window-rows-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-shared-known-window-rows-tests)
        foreach(rows 4 8)
            foreach(prefetch 0 1)
                add_test(NAME strix-hip-shared-known-window-rows-${rows}-prefetch-${prefetch}
                    COMMAND strix-hip-shared-known-window-rows-tests ${rows} ${prefetch})
            endforeach()
        endforeach()

        add_executable(strix-hip-gdn-retained-window-tests tests/hip_gdn_retained_window_tests.hip)
        set_target_properties(strix-hip-gdn-retained-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-gdn-retained-window-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-gdn-retained-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-retained-window-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-retained-window-tests)
        add_test(NAME strix-hip-gdn-retained-window-tests COMMAND strix-hip-gdn-retained-window-tests)

        add_executable(strix-hip-gdn-wide-projection-tests tests/hip_gdn_wide_projection_tests.hip)
        set_target_properties(strix-hip-gdn-wide-projection-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-gdn-wide-projection-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-gdn-wide-projection-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-wide-projection-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-wide-projection-tests)
        add_test(NAME strix-hip-gdn-wide-projection-tests COMMAND strix-hip-gdn-wide-projection-tests)

        if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
            add_executable(strix-hip-moe-q4k-a8-known-rows-tests tests/hip_moe_q4k_a8_known_rows_tests.hip)
            set_target_properties(strix-hip-moe-q4k-a8-known-rows-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_link_libraries(strix-hip-moe-q4k-a8-known-rows-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-moe-q4k-a8-known-rows-tests PRIVATE
                -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-q4k-a8-known-rows-tests)
            foreach(rows 2 4 8)
                add_test(NAME strix-hip-moe-q4k-a8-known-rows-${rows}
                    COMMAND strix-hip-moe-q4k-a8-known-rows-tests ${rows})
            endforeach()
            add_executable(strix-hip-moe-known-rows-tests tests/hip_moe_known_rows_tests.hip)
            set_target_properties(strix-hip-moe-known-rows-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_link_libraries(strix-hip-moe-known-rows-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-moe-known-rows-tests PRIVATE
                -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-known-rows-tests)
            foreach(rows 2 4 8)
                foreach(batched 0 1)
                    add_test(NAME strix-hip-moe-known-rows-${rows}-batched-${batched}
                        COMMAND strix-hip-moe-known-rows-tests ${rows} ${batched})
                endforeach()
            endforeach()
        endif()

        add_executable(strix-hip-event-queue-trace-tests tests/hip_event_queue_trace_tests.hip)
        set_target_properties(strix-hip-event-queue-trace-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-event-queue-trace-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-event-queue-trace-tests PRIVATE -mno-wavefrontsize64)
        strix_target_warnings(strix-hip-event-queue-trace-tests)
        add_test(NAME hip-event-queue-trace COMMAND strix-hip-event-queue-trace-tests)
        add_executable(strix-hip-known-window-tests tests/hip_known_window_tests.hip)
        set_target_properties(strix-hip-known-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-known-window-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-known-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-known-window-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-known-window-tests)
        add_test(NAME strix-hip-known-window-tests COMMAND strix-hip-known-window-tests)

        # Full-model qualification is explicit and requires a model and teacher;
        # it is intentionally absent from the automatic device regression.
        add_executable(strix-hip-known-window-model-qualification tests/hip_known_window_model_qualification.hip)
        set_target_properties(strix-hip-known-window-model-qualification PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-known-window-model-qualification PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-known-window-model-qualification)

        # Explicit known-prompt/cache/continuation qualification; requires the
        # real model and ROOT-sealed inputs, so it is not an automatic CTest.
        # Explicit model-dependent serving qualification; never automatic CTest.
        # Explicit local native stage component with a real model. This does
        # not run transport or qualify a distributed model; never automatic CTest.
        add_executable(strix-hip-distributed-stage-model-qualification tests/hip_distributed_stage_tests.hip)
        set_target_properties(strix-hip-distributed-stage-model-qualification PROPERTIES
            HIP_ARCHITECTURES gfx1151 HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_options(strix-hip-distributed-stage-model-qualification PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-distributed-stage-model-qualification PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-distributed-stage-model-qualification)

        # Callable model fixture requires injected actual peer/link ownership.
        # No standalone bootstrap or automatic model/transport execution.
        add_library(strix-hip-distributed-sequence-model-fixture STATIC
            tests/hip_distributed_sequence_model_qualification.hip)
        set_target_properties(strix-hip-distributed-sequence-model-fixture PROPERTIES
            HIP_ARCHITECTURES gfx1151 HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_options(strix-hip-distributed-sequence-model-fixture PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-distributed-sequence-model-fixture PUBLIC strix_hip_runtime)
        target_include_directories(strix-hip-distributed-sequence-model-fixture PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        strix_target_warnings(strix-hip-distributed-sequence-model-fixture)

        # Callable actual native generation fixture; no synthetic bootstrap,
        # main function or automatic model/transport execution.
        add_library(strix-hip-distributed-greedy-model-fixture STATIC
            tests/hip_distributed_greedy_model_qualification.hip)
        set_target_properties(strix-hip-distributed-greedy-model-fixture PROPERTIES
            HIP_ARCHITECTURES gfx1151 HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_options(strix-hip-distributed-greedy-model-fixture PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_include_directories(strix-hip-distributed-greedy-model-fixture PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries(strix-hip-distributed-greedy-model-fixture PUBLIC
            strix_hip_runtime strix-hip-distributed-sequence-model-fixture)
        strix_target_warnings(strix-hip-distributed-greedy-model-fixture)

        add_executable(strix-hip-serving-greedy-model-qualification tests/hip_serving_greedy_backend_tests.hip)
        set_target_properties(strix-hip-serving-greedy-model-qualification PROPERTIES
            HIP_ARCHITECTURES gfx1151 HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_definitions(strix-hip-serving-greedy-model-qualification PRIVATE STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC=1)
        target_compile_options(strix-hip-serving-greedy-model-qualification PRIVATE -fno-fast-math -ffp-contract=off)
        target_link_libraries(strix-hip-serving-greedy-model-qualification PRIVATE strix_hip_runtime Threads::Threads)
        strix_target_warnings(strix-hip-serving-greedy-model-qualification)

        add_executable(strix-hip-known-prompt-model-qualification tests/hip_known_prompt_model_qualification.hip)
        set_target_properties(strix-hip-known-prompt-model-qualification PROPERTIES
            HIP_ARCHITECTURES gfx1151 HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_link_libraries(strix-hip-known-prompt-model-qualification PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-known-prompt-model-qualification)

        add_executable(
            strix-hip-moe-experts-launcher-tests
            tests/hip_moe_experts_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-moe-experts-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-moe-experts-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-moe-experts-launcher-tests
        )
        add_test(
            NAME strix-hip-moe-experts-launcher-tests
            COMMAND strix-hip-moe-experts-launcher-tests
        )

        if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
            add_executable(strix-hip-q51-grouped-down-probe-tests tests/hip_q51_grouped_down_probe_tests.hip)
            set_target_properties(strix-hip-q51-grouped-down-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_link_libraries(strix-hip-q51-grouped-down-probe-tests PRIVATE strix_hip_runtime)
            strix_target_warnings(strix-hip-q51-grouped-down-probe-tests)
            add_test(NAME strix-hip-q51-grouped-down-probe-tests COMMAND strix-hip-q51-grouped-down-probe-tests)
            add_executable(strix-hip-q51-slot-parallel-probe-tests tests/hip_q51_slot_parallel_probe_tests.hip)
            set_target_properties(strix-hip-q51-slot-parallel-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_link_libraries(strix-hip-q51-slot-parallel-probe-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-q51-slot-parallel-probe-tests PRIVATE -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q51-slot-parallel-probe-tests)
            add_test(NAME strix-hip-q51-slot-parallel-probe-tests COMMAND strix-hip-q51-slot-parallel-probe-tests)
            foreach(mode 0 1 4 5)
                add_test(NAME strix-hip-moe-q51-slot-parallel-native-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-slot-parallel)
            endforeach()
            foreach(mode 0 4)
                add_test(NAME strix-hip-moe-q51-slot-parallel-native-duplicate-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-slot-parallel-duplicate)
            endforeach()
            foreach(mode 0 1 4 5)
                add_test(NAME strix-hip-moe-q51-grouped-native-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-grouped)
            endforeach()
            foreach(mode 0 4)
                add_test(NAME strix-hip-moe-q51-grouped-native-duplicate-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-q51-grouped-duplicate)
            endforeach()
            foreach(mode 0 4)
                add_test(NAME strix-hip-moe-q4k-a8-native-duplicate-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8-duplicate)
            endforeach()
            foreach(mode 0 1 4 5)
                add_test(NAME strix-hip-moe-q4k-a8-native-${mode}-tests
                    COMMAND strix-hip-moe-kquant-tests ${mode} q4k-a8)
            endforeach()
        endif()
        add_test(NAME strix-q4k-a8-native-cli-tests
            COMMAND "${CMAKE_COMMAND}" "-DBINARY=$<TARGET_FILE:strix-hip-decode-token>"
            "-DCAPABILITY=${STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_q4k_a8_runtime_cli.cmake")
        add_executable(strix-hip-moe-kquant-tests tests/hip_moe_kquant_tests.hip)
        set_target_properties(strix-hip-moe-kquant-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-moe-kquant-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-kquant-tests)
        foreach(mode RANGE 0 7)
            add_test(NAME strix-hip-moe-kquant-${mode}-tests
                     COMMAND strix-hip-moe-kquant-tests ${mode})
        endforeach()

        add_executable(strix-hip-gdn-output-projection-tests tests/hip_gdn_output_projection_tests.hip)
        target_link_libraries(strix-hip-gdn-output-projection-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-gdn-output-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        strix_target_warnings(strix-hip-gdn-output-projection-tests)
        add_test(NAME strix-hip-gdn-output-projection-tests COMMAND strix-hip-gdn-output-projection-tests)

        add_executable(strix-hip-qsa-output-projection-tests tests/hip_qsa_output_projection_tests.hip)
        target_compile_definitions(strix-hip-qsa-output-projection-tests PRIVATE STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC=1)
        target_link_libraries(strix-hip-qsa-output-projection-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-qsa-output-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        strix_target_warnings(strix-hip-qsa-output-projection-tests)
        add_test(NAME strix-hip-qsa-output-projection-tests COMMAND strix-hip-qsa-output-projection-tests)

        add_executable(
            strix-hip-gdn-launcher-tests
            tests/hip_gdn_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-gdn-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-gdn-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-gdn-launcher-tests
        )
        add_test(
            NAME strix-hip-gdn-launcher-tests
            COMMAND strix-hip-gdn-launcher-tests
        )

        add_test(NAME strix-hip-gdn-virtual-q8-f32-fallback-tests
            COMMAND strix-hip-gdn-launcher-tests 128 fallback)
        add_test(NAME strix-hip-gdn-prefetch-q8-f32-fallback-tests
            COMMAND strix-hip-gdn-launcher-tests 128 prefetch-fallback)
        foreach(prefetch_width 128 288 480 640 2560)
            add_test(NAME strix-hip-gdn-prefetch-q8-${prefetch_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${prefetch_width} prefetch)
        endforeach()
        foreach(q8_width 128 288 640 2560)
            add_test(NAME strix-hip-gdn-virtual-q8-${q8_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${q8_width})
        endforeach()

        add_executable(
            strix-hip-ple-launcher-tests
            tests/hip_ple_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-ple-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-ple-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-ple-launcher-tests
        )
        add_test(
            NAME strix-hip-ple-launcher-tests
            COMMAND strix-hip-ple-launcher-tests
        )

        add_executable(strix-hip-qsa-input-prefetch-tests tests/hip_qsa_input_prefetch_tests.hip)
        set_target_properties(strix-hip-qsa-input-prefetch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-qsa-input-prefetch-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-input-prefetch-tests)
        add_test(NAME strix-hip-qsa-input-prefetch-tests COMMAND strix-hip-qsa-input-prefetch-tests)
        add_executable(strix-hip-shared-expert-prefetch-tests tests/hip_shared_expert_prefetch_tests.hip)
        set_target_properties(strix-hip-shared-expert-prefetch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-shared-expert-prefetch-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-shared-expert-prefetch-tests)
        add_test(NAME strix-hip-shared-expert-prefetch-tests COMMAND strix-hip-shared-expert-prefetch-tests)

        add_executable(strix-hip-q8-projection-organization-tests tests/hip_q8_projection_organization_tests.hip)
        set_target_properties(strix-hip-q8-projection-organization-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-q8-projection-organization-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-q8-projection-organization-tests)
        add_test(NAME strix-hip-q8-projection-organization-tests COMMAND strix-hip-q8-projection-organization-tests)
        foreach(wave8_width IN ITEMS 128 288 480 640 2560)
            add_test(NAME strix-hip-gdn-wave8-q8-${wave8_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${wave8_width} wave8)
        endforeach()
        foreach(tail_heads IN ITEMS 4 12)
            add_test(NAME strix-hip-gdn-wave8-q8-tail-${tail_heads}-tests
                COMMAND strix-hip-gdn-launcher-tests 2560 wave8 ${tail_heads})
        endforeach()
        foreach(wave8_output_width IN ITEMS 128 288 480 640 2560)
            add_test(NAME strix-hip-gdn-wave8-q8-output-${wave8_output_width}-tests
                COMMAND strix-hip-gdn-launcher-tests ${wave8_output_width} wave8-output)
        endforeach()
        foreach(output_heads IN ITEMS 4 12)
            add_test(NAME strix-hip-gdn-wave8-q8-output-heads-${output_heads}-tests
                COMMAND strix-hip-gdn-launcher-tests 640 wave8-output ${output_heads})
        endforeach()
        add_executable(strix-hip-gdn-lm-wave8-diagnostic-tests tests/hip_gdn_lm_wave8_diagnostic_tests.hip)
        add_executable(strix-hip-shared-expert-rows-diagnostic-tests tests/hip_shared_expert_rows_diagnostic_tests.hip)
        set_target_properties(strix-hip-shared-expert-rows-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-shared-expert-rows-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-shared-expert-rows-diagnostic-tests)
        add_test(NAME strix-hip-shared-expert-rows-diagnostic-tests COMMAND strix-hip-shared-expert-rows-diagnostic-tests)
        add_executable(strix-hip-qsa-gqa-reuse-diagnostic-tests tests/hip_qsa_gqa_reuse_diagnostic_tests.hip)
        set_target_properties(strix-hip-qsa-gqa-reuse-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-qsa-gqa-reuse-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-gqa-reuse-diagnostic-tests)
        add_test(NAME strix-hip-qsa-gqa-reuse-diagnostic-tests COMMAND strix-hip-qsa-gqa-reuse-diagnostic-tests)
        add_executable(strix-hip-moe-batched-activation-diagnostic-tests tests/hip_moe_batched_activation_diagnostic_tests.hip)
        set_target_properties(strix-hip-moe-batched-activation-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-moe-batched-activation-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-batched-activation-diagnostic-tests)
        add_test(NAME strix-hip-moe-batched-activation-diagnostic-tests COMMAND strix-hip-moe-batched-activation-diagnostic-tests)
        add_executable(strix-hip-moe-overlap-dispatcher-tests tests/hip_moe_overlap_dispatcher_tests.hip)
        set_target_properties(strix-hip-moe-overlap-dispatcher-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-moe-overlap-dispatcher-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-overlap-dispatcher-tests)
        add_test(NAME strix-hip-moe-overlap-dispatcher-tests COMMAND strix-hip-moe-overlap-dispatcher-tests)
        add_executable(strix-hip-generation-output-tests tests/hip_generation_output_tests.hip)
        add_executable(strix-hip-gdn-recurrent-layout-diagnostic-tests tests/hip_gdn_recurrent_layout_diagnostic_tests.hip)
        set_target_properties(strix-hip-gdn-recurrent-layout-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-gdn-recurrent-layout-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-gdn-recurrent-layout-diagnostic-tests)
        add_test(NAME strix-hip-gdn-recurrent-layout-diagnostic-tests COMMAND strix-hip-gdn-recurrent-layout-diagnostic-tests)
        set_target_properties(strix-hip-generation-output-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-generation-output-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-generation-output-tests)
        add_test(NAME strix-hip-generation-output-tests COMMAND strix-hip-generation-output-tests)
        set_target_properties(strix-hip-gdn-lm-wave8-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_link_libraries(strix-hip-gdn-lm-wave8-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-gdn-lm-wave8-diagnostic-tests)
        add_test(NAME strix-hip-gdn-lm-wave8-diagnostic-tests COMMAND strix-hip-gdn-lm-wave8-diagnostic-tests)

        add_executable(
            strix-hip-indexed-attention-launcher-tests
            tests/hip_indexed_attention_launcher_tests.hip
        )
        set_target_properties(
            strix-hip-indexed-attention-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES gfx1151
        )
        target_link_libraries(
            strix-hip-indexed-attention-launcher-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-indexed-attention-launcher-tests
        )
        add_test(
            NAME strix-hip-indexed-attention-launcher-tests
            COMMAND strix-hip-indexed-attention-launcher-tests
        )

        add_executable(
            strix-hip-model-residency-tests
            tests/hip_model_residency_tests.cpp
        )
        target_link_libraries(
            strix-hip-model-residency-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-model-residency-tests
        )
        add_test(
            NAME strix-hip-model-residency-tests
            COMMAND strix-hip-model-residency-tests
        )

        add_executable(
            strix-hip-session-snapshot-transfer-tests
            tests/hip_session_snapshot_transfer_tests.cpp
        )
        target_link_libraries(
            strix-hip-session-snapshot-transfer-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-session-snapshot-transfer-tests
        )
        add_test(
            NAME strix-hip-session-snapshot-transfer-tests
            COMMAND strix-hip-session-snapshot-transfer-tests
        )

        add_executable(
            strix-hip-runtime-memory-init-tests
            tests/hip_runtime_memory_init_tests.cpp
        )
        target_link_libraries(
            strix-hip-runtime-memory-init-tests
            PRIVATE strix_hip_runtime
        )
        strix_target_warnings(
            strix-hip-runtime-memory-init-tests
        )
        add_test(
            NAME strix-hip-runtime-memory-init-tests
            COMMAND strix-hip-runtime-memory-init-tests
        )
    endif()
endif()
