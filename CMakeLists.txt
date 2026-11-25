cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/StrixHipArchs.cmake")

# Single-config builds default to optimized code; explicit configurations and
# multi-config generators keep their existing selection. Select before project():
# the MSVC platform otherwise supplies its own implicit Debug default.
get_property(strix_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT strix_multi_config AND NOT DEFINED CMAKE_BUILD_TYPE AND NOT "$ENV{CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "$ENV{CMAKE_BUILD_TYPE}" CACHE STRING "Build type")
endif()
if(NOT strix_multi_config AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug, Release, RelWithDebInfo or MinSizeRel" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set: using Release")
endif()

project(strix_engine_lab VERSION 0.1.0 LANGUAGES CXX)
include(cmake/StrixWeight.cmake)
option(STRIX_WFMT_TESTS "Build shared weight-format contract tests" OFF)
if(STRIX_WFMT_TESTS)
    enable_testing()
    add_subdirectory(tests/wfmt)
endif()
if(MSVC)
    # Portable C library calls (getenv, fopen) are used on purpose; unknown loop pragmas (unroll), padding
    # from alignas and code after Linux-only stubs that throw on Windows (C4702) are expected. Without this,
    # /WX turns them into C2220 errors in every MSVC target. Conversion, sign-compare and shadow checks
    # (C4244 C4267 C4389 C4459) stay enforced by the -Wconversion -Wshadow -Werror GCC build; MSVC also
    # reports them inside standard-library template instantiations, which GCC does not.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4068> $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4702> $<$<COMPILE_LANGUAGE:CXX>:/wd4244> $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4389> $<$<COMPILE_LANGUAGE:CXX>:/wd4459>)
endif()

option(STRIX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
function(strix_apply_target_warnings target)
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

# Isolated shared CUDA-core qualification, with no model/runtime dependency.
option(STRIX_CUDA_NVFP4_ONLY "Build only shared NVFP4 host/native checks" OFF)
if(STRIX_CUDA_NVFP4_ONLY)
    enable_testing()
    add_subdirectory(tests/nvfp4)
    return()
endif()

include(cmake/StrixCli.cmake)

# Install rules and CPack packages (docs/install.md). Deferred to the end of this file so every target exists,
# also in the CUDA-only modes below that return() early.
cmake_language(DEFER CALL include "${CMAKE_CURRENT_SOURCE_DIR}/packaging/StrixPackaging.cmake")

option(STRIX_LEGACY_MAINS "Build legacy Gemma4 and MiniMax serve executables" ON)

# Sampling policy and reference are also used by the standalone CUDA backends.
find_package(Threads REQUIRED)
include(cmake/StrixTokenSampling.cmake)

# Standalone CUDA Qwen prefill and host-only scheduling/codec checks.
option(STRIX_CUDA_CORE "Build the shared CUDA core and strix-cuda-qwen35-decode" OFF)
option(STRIX_CUDA_CORE_ONLY "Build only the shared CUDA core (CUDA language, no HIP tree)" OFF)
if(STRIX_CUDA_CORE_ONLY)
    include(cmake/CudaCore.cmake)
    include(cmake/StrixBuildInfo.cmake)
    return()
endif()
option(STRIX_CUDA_QWEN_ONLY "Build only the CUDA Qwen prefill service and gates" OFF)
if(STRIX_CUDA_QWEN_ONLY)
    include(CTest)
    include(cmake/StrixSession.cmake)
    include(cmake/QwenCuda.cmake)
    include(cmake/StrixDenseModels.cmake)
    include(cmake/StrixLevers.cmake)
    return()
endif()

if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Evaluate after all optional language/toolchain and STRIX option configuration.
cmake_language(DEFER CALL include "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StrixBuildInfo.cmake")

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

option(STRIX_ENABLE_HIP_RUNTIME "Build optional HIP allocation/runtime support" OFF)
option(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT "Use non-temporal weight loads in routed Q2_K decode down projections" OFF)
if(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
    if(NOT STRIX_ENABLE_HIP_RUNTIME)
        message(FATAL_ERROR "MALL Q2_K down pilot requires HIP runtime support")
    endif()
    set_property(SOURCE src/hip_ds41_formats.hip APPEND PROPERTY COMPILE_DEFINITIONS STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT=1)
endif()
option(STRIX_FAMILY_DEEPSEEK41 "Build the deepseek41 model family module" ON)
option(STRIX_FAMILY_QWEN35 "Build the qwen35 (Qwen3.8-27B dense) model family module" ON)
option(STRIX_FAMILY_QWEN35_HIP "Build the qwen35 dense (Qwen3.8-27B) HIP decode app" OFF)
option(STRIX_FAMILY_MINIMAX "Build the experimental minimax-m3 bring-up modules (MSA host reference)" OFF)
option(STRIX_ENABLE_CUDA_DS41 "Build the deepseek41 CUDA backend (sm_120 / sm_89 / sm_86) and its tests" OFF)
option(STRIX_ENABLE_CUDA_TRELLIS "Build experimental compressed-weight CUDA linear kernels and probes" OFF)
option(STRIX_DS41_TAP_CAPTURE "Build the host-forward capture tool of the deepseek41 CUDA checks (host only)" OFF)
# Minimal configure for the deepseek41 CUDA backend alone (used for the native Windows build): the host
# libraries it links (a trimmed strix_core, the deepseek41 family, ds41 formats/layer/attention, the host
# forward runtime and the prefill plan), the CUDA libraries, their tests and apps. Everything else (HIP,
# serving, networking, pipeline wire, DSpark) is skipped.
option(STRIX_CUDA_DS41_ONLY "Configure only the deepseek41 CUDA backend and the host libraries it needs" OFF)
if(STRIX_CUDA_DS41_ONLY)
    if(STRIX_ENABLE_HIP_RUNTIME OR NOT STRIX_FAMILY_DEEPSEEK41)
        message(FATAL_ERROR "STRIX_CUDA_DS41_ONLY needs STRIX_FAMILY_DEEPSEEK41=ON and STRIX_ENABLE_HIP_RUNTIME=OFF")
    endif()
    set(STRIX_ENABLE_CUDA_DS41 ON CACHE BOOL "Build the deepseek41 CUDA backend (sm_120 / sm_89 / sm_86) and its tests" FORCE)
    if(MSVC)
        # Host arithmetic as on Linux (-ffp-contract=off): /fp:precise without /fp:contract never contracts
        # into FMA (and x64 SSE2 code has no FMA instruction); float evaluates in float.
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:precise> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor> $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    endif()
endif()
if(STRIX_ENABLE_CUDA_TRELLIS AND (NOT STRIX_ENABLE_CUDA_DS41 OR NOT STRIX_FAMILY_DEEPSEEK41))
    message(FATAL_ERROR "STRIX_ENABLE_CUDA_TRELLIS requires STRIX_ENABLE_CUDA_DS41 and STRIX_FAMILY_DEEPSEEK41")
endif()
option(STRIX_EXPERIMENTAL_MALL_SHARED_DOWN_NT "Use non-temporal Q8_0 weight loads in shared-expert dense-A8 decode down projections" OFF)
if(STRIX_EXPERIMENTAL_MALL_SHARED_DOWN_NT)
    if(NOT STRIX_ENABLE_HIP_RUNTIME)
        message(FATAL_ERROR "MALL shared down pilot requires HIP runtime support")
    endif()
    set_property(SOURCE src/hip_shared_expert_launcher.hip APPEND PROPERTY COMPILE_DEFINITIONS STRIX_EXPERIMENTAL_MALL_SHARED_DOWN_NT=1)
endif()
option(STRIX_BACKEND_CPU "Build the CPU backend scaffold (src/cpu) and its host tests" OFF)
option(STRIX_USE_OPENSSL_SHA256 "Use OpenSSL for full-file SHA-256 verification" OFF)
option(STRIX_CHECK_KERNEL_SCRATCH "Fail the build when a kernel's private scratch/spills exceed tools/kernel_scratch_allowance.txt" OFF)
# Post-build check of every amdgcn code object linked into target (tools/check_kernel_scratch.py).
function(strix_check_kernel_scratch target)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/check_kernel_scratch.py --quiet
            --allowance ${CMAKE_SOURCE_DIR}/tools/kernel_scratch_allowance.txt $<TARGET_FILE:${target}>
        COMMENT "Kernel scratch/spill check: ${target}" VERBATIM)
endfunction()
option(STRIX_ENABLE_EXPERIMENTAL_DENSE_TRELLIS "Build the explicit experimental dense trellis pack route" OFF)
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
    strix_apply_target_warnings(${target})
endfunction()

# Generated model constants (include/strix/model_data/<ident>.hpp, from models/*.toml) and their binding helpers
# (strix/model/bind.hpp). Header only and backend neutral: every target whose translation units include an engine
# header bound to the TOML values (for example src/glm/rtxpf/import/glm_dims.hpp) links this one target.
add_library(strix_model_data INTERFACE)
target_include_directories(strix_model_data INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(strix_model_data INTERFACE cxx_std_20)

# Named host-table arithmetic: no reassociation or implicit multiply-add.
if(MSVC)
    set_source_files_properties(src/rotary_descriptor.cpp src/qwen_yarn_table.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
    set_source_files_properties(src/rotary_descriptor.cpp src/qwen_yarn_table.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-frounding-math")
endif()

include(cmake/StrixStateCache.cmake)

# Prompt-lookup (n-gram) index shared by the engines' drafting (it only proposes draft tokens: verification decides).
add_library(strix_lookup STATIC src/lookup/index.cpp)
target_include_directories(strix_lookup PUBLIC include)
target_compile_features(strix_lookup PUBLIC cxx_std_20)
strix_target_warnings(strix_lookup)
# Shared drafting layer: context proposer (copy cursor, long proposals), lookup/drafter/serial arbiter, stats line.
include(cmake/StrixTokenSampling.cmake)
add_library(strix_spec STATIC src/spec/drafting.cpp src/spec/block_drafter.cpp src/spec/draft_rows.cpp
    src/spec/block_parallel.cpp src/spec/row_rule.cpp)
target_link_libraries(strix_spec PUBLIC strix_lookup strix_token_sampling)
strix_target_warnings(strix_spec)

# Host mathematical oracle and candidate adapter; production device paths stay unbound.
add_library(strix_block_candidate STATIC src/core/block_draft_reference.cpp src/spec/block_candidate.cpp)
target_link_libraries(strix_block_candidate PUBLIC strix_spec strix-weight)
strix_target_warnings(strix_block_candidate)
if(MSVC)
    set_source_files_properties(src/weight/block_draft.cpp PROPERTIES COMPILE_OPTIONS "/W4;/WX")
    target_compile_options(strix_block_candidate PRIVATE /fp:strict)
else()
    set_source_files_properties(src/weight/block_draft.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow;-Werror")
    target_compile_options(strix_block_candidate PRIVATE -ffp-contract=off)
endif()

if(STRIX_CUDA_DS41_ONLY)
# Only what the deepseek41 family and the ds41 libraries link: GGUF reader, manifest, binding, content
# hash and the family registry.
add_library(strix_core STATIC
    src/gguf_reader.cpp
    src/model_manifest.cpp
    src/model_family_registry.cpp
    src/cost_table_cache.cpp
    src/logprob_partials.cpp
    src/token_sampling.cpp  # shared sampler: sampled replica requests
    src/speculation_rng.cpp
)
else()
add_library(strix_core STATIC
    src/qwen_yarn_table.cpp
    src/final_generation_state_export.cpp
    src/greedy_serving_scheduler.cpp
    src/serving_cohort.cpp
    src/generation_output.cpp
    src/logprob_partials.cpp
    src/known_prompt_chunks.cpp
    src/known_prefix_prefill.cpp
    src/known_prefix_prefill_runtime.cpp
    src/offline_tuning_cache.cpp
    src/offline_tuning_preparation.cpp
    src/offline_compile_provenance.cpp
    src/uma_owner_inventory.cpp
    src/linux_uma_observation.cpp
    src/preflight.cpp
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
    src/session_executor_handoff.cpp
    src/graph_eligibility.cpp
    src/graph_inputs.cpp
    src/graph_attention_state.cpp
    src/graph_ple.cpp
    src/graph_operator_control.cpp
    src/integer_projection.cpp
    src/integer_projection_execution.cpp
    src/integer_projection_runtime.cpp
    src/host_upload.cpp
    src/qwen_moe_prepared.cpp
    src/compiled_execution.cpp
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
    src/model_family_registry.cpp
    src/kernel_registry.cpp
    src/mapped_model_file.cpp
    src/iq4_xs.cpp
    src/indexed_attention_workspace.cpp
    src/indexed_attention_state.cpp
    src/iq4_nl.cpp
    src/ggml_dequant.cpp
    src/q6p_registry.cpp
    src/launch_dispatch.cpp
    src/low_precision.cpp
    src/memory_plan.cpp
    src/model_contract.cpp
    src/mtp_draft_binding.cpp
    src/mtp_draft_contract.cpp
    src/mtp_draft_qualification.cpp
    src/mtp_ngram_draft.cpp
    src/mtp_frame_draft.cpp
    src/mtp_context_draft.cpp
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
    src/mtp_selection_transaction.cpp
    src/mtp_window_controller.cpp
    src/cost_table_cache.cpp
    src/mtp_cost_table_session.cpp
    src/mtp_depth_window_controller.cpp
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
    src/mtp_target_checkpoint_plan.cpp
    src/mtp_draft_vocabulary.cpp
    src/session_restore.cpp
    src/sparse_lookup.cpp
    src/sparse_lookup_executor.cpp
    src/speculation.cpp
    src/speculation_controller.cpp
    src/speculation_measurement.cpp
    src/speculation_multi_draft.cpp
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
    src/ple_reclaim.cpp
    src/ple_rolling.cpp
    src/ple_workspace.cpp
    src/rtx_state_import.cpp
    src/bp_codec.cpp
    src/transport.cpp
)
target_link_libraries(strix_core PRIVATE strix_session)
endif()
target_include_directories(strix_core PUBLIC include src)
target_compile_features(strix_core PUBLIC cxx_std_20)
target_link_libraries(strix_core PUBLIC Threads::Threads strix_state_cache strix_token_sampling)
if(NOT STRIX_CUDA_DS41_ONLY)
    target_link_libraries(strix_core PRIVATE strix_spec)
endif()
if(STRIX_USE_OPENSSL_SHA256)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    target_link_libraries(strix_state_cache PRIVATE OpenSSL::Crypto)
    target_compile_definitions(strix_state_cache PRIVATE STRIX_USE_OPENSSL_SHA256=1)
endif()
strix_target_warnings(strix_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(strix-ieee-half-tests tests/ieee_half_tests.cpp)
    target_include_directories(strix-ieee-half-tests PRIVATE include)
    target_compile_features(strix-ieee-half-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-ieee-half-tests)
    add_test(NAME strix-ieee-half-tests COMMAND strix-ieee-half-tests)
endif()
include(cmake/StrixSession.cmake)
include(cmake/StrixResourceContracts.cmake)
include(cmake/StrixVisionPreprocess.cmake)
if(BUILD_TESTING)
    add_executable(weight_resident_slices_tests tests/weight_resident_slices_tests.cpp)
    target_link_libraries(weight_resident_slices_tests PRIVATE strix-weight)
    strix_target_warnings(weight_resident_slices_tests)
    add_test(NAME weight-resident-slices COMMAND weight_resident_slices_tests)
    add_executable(weight_lowbit_tests tests/weight_lowbit_tests.cpp)
    target_link_libraries(weight_lowbit_tests PRIVATE strix-weight)
    target_include_directories(weight_lowbit_tests PRIVATE src include)
    strix_target_warnings(weight_lowbit_tests)
    add_test(NAME weight-lowbit COMMAND weight_lowbit_tests)
    add_executable(weight_microscaled_encode_test tests/weight/microscaled_encode_test.cpp)
    target_link_libraries(weight_microscaled_encode_test PRIVATE strix-weight)
    target_include_directories(weight_microscaled_encode_test PRIVATE include)
    target_compile_features(weight_microscaled_encode_test PRIVATE cxx_std_20)
    strix_target_warnings(weight_microscaled_encode_test)
    add_test(NAME weight-microscaled-encode COMMAND weight_microscaled_encode_test)
    add_executable(weight_row_pitch_tests tests/weight_row_pitch_tests.cpp)
    target_link_libraries(weight_row_pitch_tests PRIVATE strix-weight)
    strix_target_warnings(weight_row_pitch_tests)
    add_test(NAME weight-row-pitch COMMAND weight_row_pitch_tests)
    add_executable(strix-weight-decision-scope-tests tests/weight_decision_scope_tests.cpp)
    target_link_libraries(strix-weight-decision-scope-tests PRIVATE strix-weight)
    strix_target_warnings(strix-weight-decision-scope-tests)
    add_test(NAME weight-decision-scope COMMAND strix-weight-decision-scope-tests)
endif()
if(BUILD_TESTING)
    add_executable(strix-inkling2-mini-nvfp4-table-test tests/inkling2/mini_nvfp4_table_test.cpp)
    target_include_directories(strix-inkling2-mini-nvfp4-table-test PRIVATE src include)
    target_compile_features(strix-inkling2-mini-nvfp4-table-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-inkling2-mini-nvfp4-table-test)
    if(MSVC)
        target_compile_options(strix-inkling2-mini-nvfp4-table-test PRIVATE /fp:strict)
    else()
        target_compile_options(strix-inkling2-mini-nvfp4-table-test PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-inkling2-mini-nvfp4-table-test COMMAND strix-inkling2-mini-nvfp4-table-test)
    set_tests_properties(strix-inkling2-mini-nvfp4-table-test PROPERTIES LABELS "cpu;inkling2")
    add_executable(strix-nvfp4-tile-words-test tests/core/nvfp4_tile_words_test.cpp)
    target_include_directories(strix-nvfp4-tile-words-test PRIVATE src include)
    target_compile_features(strix-nvfp4-tile-words-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-nvfp4-tile-words-test)
    if(MSVC)
        target_compile_options(strix-nvfp4-tile-words-test PRIVATE /fp:strict)
    else()
        target_compile_options(strix-nvfp4-tile-words-test PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-nvfp4-tile-words-test COMMAND strix-nvfp4-tile-words-test)
    set_tests_properties(strix-nvfp4-tile-words-test PROPERTIES LABELS "cpu;exactness")
endif()
if(NOT STRIX_CUDA_DS41_ONLY)
    add_library(strix_inkling2_mini_checkpoint STATIC
        src/inkling2/inkling2_mini_checkpoint.cpp
        src/inkling2/inkling2_mini_resident_weights.cpp)
    target_include_directories(strix_inkling2_mini_checkpoint PUBLIC src)
    target_link_libraries(strix_inkling2_mini_checkpoint PUBLIC strix-weight)
    strix_target_warnings(strix_inkling2_mini_checkpoint)
    add_executable(strix-inkling2-mini-inspect apps/inkling2_mini_inspect.cpp)
    target_link_libraries(strix-inkling2-mini-inspect PRIVATE strix_inkling2_mini_checkpoint)
    strix_target_warnings(strix-inkling2-mini-inspect)
    add_executable(strix-inkling2-mini-cpu
        apps/inkling2_mini_cpu.cpp
        src/inkling2/inkling2_mini_host.cpp
        src/inkling2/inkling2_mini_dspark_reference.cpp)
    target_link_libraries(strix-inkling2-mini-cpu PRIVATE strix_inkling2_mini_checkpoint strix_serve)
    strix_target_warnings(strix-inkling2-mini-cpu)
    if(MSVC)
        target_compile_options(strix-inkling2-mini-cpu PRIVATE /fp:strict)
    else()
        target_compile_options(strix-inkling2-mini-cpu PRIVATE -ffp-contract=off)
    endif()
    if(BUILD_TESTING)
        add_subdirectory(tests/inkling2)
    endif()
endif()
add_library(strix_ds41_http_session_owner STATIC src/deepseek41/http_session_owner.cpp)
target_include_directories(strix_ds41_http_session_owner PUBLIC src)
target_link_libraries(strix_ds41_http_session_owner PUBLIC strix_session_owner)
strix_target_warnings(strix_ds41_http_session_owner)
if(BUILD_TESTING)
    # CPU-only checks against the real HIP find_library command (toolkit priority and legacy fallback).
    add_test(NAME strix-hip-toolkit-lookup-tests
        COMMAND "${CMAKE_COMMAND}" "-DSTRIX_REPO_ROOT=${CMAKE_SOURCE_DIR}"
                -P "${CMAKE_SOURCE_DIR}/tests/cmake/hip_toolkit_paths_tests.cmake")
    set_tests_properties(strix-hip-toolkit-lookup-tests PROPERTIES LABELS "cpu;build")
    add_executable(strix-exact-reduce-contract-tests tests/exact_reduce_contract_tests.cpp)
    target_compile_features(strix-exact-reduce-contract-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-exact-reduce-contract-tests)
    add_test(NAME strix-exact-reduce-contract-tests COMMAND strix-exact-reduce-contract-tests)

    add_executable(strix-sparse-select-policy-tests tests/sparse_select_policy_tests.cpp)
    target_include_directories(strix-sparse-select-policy-tests PRIVATE include)
    target_compile_features(strix-sparse-select-policy-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-sparse-select-policy-tests)
    add_test(NAME strix-sparse-select-policy-tests COMMAND strix-sparse-select-policy-tests)

    add_executable(strix-head-argmax-tests tests/head_argmax_tests.cpp)
    target_include_directories(strix-head-argmax-tests PRIVATE include)
    target_compile_features(strix-head-argmax-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-head-argmax-tests)
    add_test(NAME strix-head-argmax-tests COMMAND strix-head-argmax-tests)
    set_tests_properties(strix-head-argmax-tests PROPERTIES LABELS "cpu")
    # Shared multi-row logits pitch (strix/logit_rows.hpp) and the strided greedy pick admission.
    add_executable(strix-logit-rows-tests tests/logit_rows_tests.cpp)
    target_include_directories(strix-logit-rows-tests PRIVATE include)
    target_compile_features(strix-logit-rows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-logit-rows-tests)
    add_test(NAME strix-logit-rows-tests COMMAND strix-logit-rows-tests)
    set_tests_properties(strix-logit-rows-tests PROPERTIES LABELS "cpu")

    add_executable(strix-qwen-gdn-prepass-tests tests/qwen_gdn_prepass_tests.cpp)
    target_include_directories(strix-qwen-gdn-prepass-tests PRIVATE include)
    target_compile_features(strix-qwen-gdn-prepass-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen-gdn-prepass-tests)
    add_test(NAME strix-qwen-gdn-prepass-tests COMMAND strix-qwen-gdn-prepass-tests)
    set_tests_properties(strix-qwen-gdn-prepass-tests PROPERTIES LABELS "cpu")

    add_executable(strix-gdn-prefill-overlap-tests tests/gdn_prefill_overlap_tests.cpp)
    target_include_directories(strix-gdn-prefill-overlap-tests PRIVATE include)
    target_compile_features(strix-gdn-prefill-overlap-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-gdn-prefill-overlap-tests)
    add_test(NAME strix-gdn-prefill-overlap-tests COMMAND strix-gdn-prefill-overlap-tests)
    set_tests_properties(strix-gdn-prefill-overlap-tests PROPERTIES LABELS "cpu")
    add_executable(strix-qsa-index-rb8-host-tests tests/qsa_index_rb8_host_tests.cpp)
    target_compile_features(strix-qsa-index-rb8-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qsa-index-rb8-host-tests)
    add_test(NAME strix-qsa-index-rb8-host-tests COMMAND strix-qsa-index-rb8-host-tests)

    add_executable(strix-dist-transport-spec-tests tests/dist_transport_spec_tests.cpp)
    target_include_directories(strix-dist-transport-spec-tests PRIVATE include)
    target_compile_features(strix-dist-transport-spec-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-dist-transport-spec-tests)
    add_test(NAME strix-dist-transport-spec-tests COMMAND strix-dist-transport-spec-tests)
    add_executable(strix-dist-link-cost-tests tests/dist_link_cost_tests.cpp)
    target_include_directories(strix-dist-link-cost-tests PRIVATE include)
    target_compile_features(strix-dist-link-cost-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-dist-link-cost-tests)
    add_test(NAME strix-dist-link-cost-tests COMMAND strix-dist-link-cost-tests)

    add_executable(strix-device-profile-host-tests tests/device_profile_host_tests.cpp src/device_profile.cpp)
    target_include_directories(strix-device-profile-host-tests PRIVATE include)
    target_compile_features(strix-device-profile-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-device-profile-host-tests)
    add_test(NAME strix-device-profile-host-tests COMMAND strix-device-profile-host-tests)
    add_executable(strix-hip-arch-layer-host-tests tests/hip_arch_layer_host_tests.cpp)
    target_include_directories(strix-hip-arch-layer-host-tests PRIVATE tests/stubs/hip_archguard include)
    target_compile_features(strix-hip-arch-layer-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hip-arch-layer-host-tests)
    add_test(NAME strix-hip-arch-layer-host-tests COMMAND strix-hip-arch-layer-host-tests)
    add_test(NAME strix-hip-arch-dense-a8-target-host-tests COMMAND "${CMAKE_COMMAND}"
        -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/device_profile_host/dense_a8_target"
        -B "${CMAKE_CURRENT_BINARY_DIR}/dense-a8-target-host-test" -G "${CMAKE_GENERATOR}")
    add_executable(strix-split-k-policy-host-tests tests/split_k_policy_host_tests.cpp)
    target_include_directories(strix-split-k-policy-host-tests PRIVATE include)
    target_compile_features(strix-split-k-policy-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-split-k-policy-host-tests)
    add_test(NAME strix-split-k-policy-host-tests COMMAND strix-split-k-policy-host-tests)
    add_executable(strix-qwen35-dense-profile-host-tests tests/qwen35_dense_profile_host_tests.cpp)
    target_include_directories(strix-qwen35-dense-profile-host-tests PRIVATE include)
    target_compile_features(strix-qwen35-dense-profile-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen35-dense-profile-host-tests)
    add_test(NAME strix-qwen35-dense-profile-host-tests COMMAND strix-qwen35-dense-profile-host-tests)
    add_executable(strix-qwen-dense-hip-profile-tests tests/qwen_dense_hip_profile_tests.cpp)
    target_include_directories(strix-qwen-dense-hip-profile-tests PRIVATE include)
    target_compile_features(strix-qwen-dense-hip-profile-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen-dense-hip-profile-tests)
    add_test(NAME strix-qwen-dense-hip-profile-tests COMMAND strix-qwen-dense-hip-profile-tests)
    set_tests_properties(strix-qwen-dense-hip-profile-tests PROPERTIES LABELS "cpu;profile")
    add_executable(strix-qwen35-prefill-windows-tests tests/qwen35_prefill_windows_tests.cpp)
    target_include_directories(strix-qwen35-prefill-windows-tests PRIVATE include)
    target_compile_features(strix-qwen35-prefill-windows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen35-prefill-windows-tests)
    add_test(NAME strix-qwen35-prefill-windows-tests COMMAND strix-qwen35-prefill-windows-tests)
    add_executable(strix-wave-fit-host-tests tests/wave_fit_host_tests.cpp)
    target_include_directories(strix-wave-fit-host-tests PRIVATE include)
    target_compile_features(strix-wave-fit-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-wave-fit-host-tests)
    add_test(NAME strix-wave-fit-host-tests COMMAND strix-wave-fit-host-tests)
    add_executable(strix-relative-bias-bound-tests tests/relative_bias_bound_tests.cpp)
    target_include_directories(strix-relative-bias-bound-tests PRIVATE include src)
    target_compile_features(strix-relative-bias-bound-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-relative-bias-bound-tests)
    add_test(NAME strix-relative-bias-bound-tests COMMAND strix-relative-bias-bound-tests)
    add_executable(strix-attn-row-share-tests tests/attn_row_share_tests.cpp)
    target_include_directories(strix-attn-row-share-tests PRIVATE include)
    target_compile_features(strix-attn-row-share-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-attn-row-share-tests)
    add_test(NAME strix-attn-row-share-tests COMMAND strix-attn-row-share-tests)
    add_executable(strix-attn-chunk-decode-tests tests/attn_chunk_decode_tests.cpp)
    target_include_directories(strix-attn-chunk-decode-tests PRIVATE src)
    target_compile_features(strix-attn-chunk-decode-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-attn-chunk-decode-tests)
    if(MSVC)
        target_compile_options(strix-attn-chunk-decode-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-attn-chunk-decode-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-attn-chunk-decode-tests COMMAND strix-attn-chunk-decode-tests)
    # The lane arithmetic must not depend on the binder's contraction flags: this TU contracts and enables FMA where it can.
    add_executable(strix-attn-chunk-decode-lane-tests tests/attn_chunk_decode_lane_tests.cpp)
    target_include_directories(strix-attn-chunk-decode-lane-tests PRIVATE src)
    target_compile_features(strix-attn-chunk-decode-lane-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-attn-chunk-decode-lane-tests)
    if(NOT MSVC)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(strix-attn-chunk-decode-lane-tests PRIVATE -ffp-contract=fast-honor-pragmas)
        else()
            target_compile_options(strix-attn-chunk-decode-lane-tests PRIVATE -ffp-contract=fast)
        endif()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            target_compile_options(strix-attn-chunk-decode-lane-tests PRIVATE -mfma)
        endif()
    endif()
    add_test(NAME strix-attn-chunk-decode-lane-tests COMMAND strix-attn-chunk-decode-lane-tests)
    add_executable(strix-fma-rows-policy-tests tests/fma_rows_policy_tests.cpp)
    target_include_directories(strix-fma-rows-policy-tests PRIVATE include)
    target_compile_features(strix-fma-rows-policy-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-fma-rows-policy-tests)
    add_test(NAME strix-fma-rows-policy-tests COMMAND strix-fma-rows-policy-tests)
    add_executable(strix-fma-rows-blocked16-tests tests/fma_rows_blocked16_tests.cpp)
    target_include_directories(strix-fma-rows-blocked16-tests PRIVATE src)
    target_compile_features(strix-fma-rows-blocked16-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-fma-rows-blocked16-tests)
    add_test(NAME strix-fma-rows-blocked16-tests COMMAND strix-fma-rows-blocked16-tests)
    add_executable(strix-dense-row-plan-tests tests/dense_row_plan_tests.cpp)
    target_include_directories(strix-dense-row-plan-tests PRIVATE src)
    target_compile_features(strix-dense-row-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-dense-row-plan-tests)
    add_test(NAME strix-dense-row-plan-tests COMMAND strix-dense-row-plan-tests)

    add_executable(strix-core-resident-q8-tile-tests tests/core_resident_q8_tile_tests.cpp)
    target_include_directories(strix-core-resident-q8-tile-tests PRIVATE include src)
    target_compile_features(strix-core-resident-q8-tile-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-core-resident-q8-tile-tests PRIVATE strix-weight)
    strix_target_warnings(strix-core-resident-q8-tile-tests)
    add_test(NAME strix-core-resident-q8-tile-tests COMMAND strix-core-resident-q8-tile-tests)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        # Source form of the shared EXACT dense decode attention (fp contraction pinned, one arithmetic copy).
        add_test(NAME strix-dense-decode-exact-form-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/dense_decode_exact_form_tests.py)
        # Kernel contract checker (tools/check_kernel_scratch.py): allowance / launch-budget parser, budget check and
        # machine-code identity normalisation against planted controls; no GPU, no binary.
        add_test(NAME strix-kernel-contract-self-test
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/check_kernel_scratch.py --self-test)
    endif()
    add_executable(strix-a4-tiles-host-tests tests/a4_tiles_host_tests.cpp)
    target_include_directories(strix-a4-tiles-host-tests PRIVATE src)
    target_compile_features(strix-a4-tiles-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-a4-tiles-host-tests)
    add_test(NAME strix-a4-tiles-host-tests COMMAND strix-a4-tiles-host-tests)
    add_executable(strix-attention-split-plan-host-tests tests/attention_split_plan_host_tests.cpp)
    target_include_directories(strix-attention-split-plan-host-tests PRIVATE include)
    target_compile_features(strix-attention-split-plan-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-attention-split-plan-host-tests)
    add_test(NAME strix-attention-split-plan-host-tests COMMAND strix-attention-split-plan-host-tests)

    add_executable(strix-attention-row-walk-host-tests tests/attention_row_walk_host_tests.cpp)
    target_include_directories(strix-attention-row-walk-host-tests PRIVATE include)
    target_compile_features(strix-attention-row-walk-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-attention-row-walk-host-tests)
    add_test(NAME strix-attention-row-walk-host-tests COMMAND strix-attention-row-walk-host-tests)
    # Transposed butterfly halving of the CUDA decode attention scores: bits == the contract's per-dot xor butterfly.
    add_executable(strix-attention-score-reduce-host-tests tests/attention_score_reduce_host_tests.cpp)
    target_include_directories(strix-attention-score-reduce-host-tests PRIVATE include)
    target_compile_features(strix-attention-score-reduce-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-attention-score-reduce-host-tests)
    target_link_libraries(strix-attention-score-reduce-host-tests PRIVATE Threads::Threads)
    add_test(NAME strix-attention-score-reduce-host-tests COMMAND strix-attention-score-reduce-host-tests)
    # K/V cache storage classes (strix/kv_storage.hpp, the model contract's AttentionKvStorage): bytes, role / tier policy,
    # contract binding, quantiser layouts, draft-cache ids == serial.
    add_executable(strix-kv-storage-tests tests/kv_storage_tests.cpp)
    target_include_directories(strix-kv-storage-tests PRIVATE include)
    target_compile_features(strix-kv-storage-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-kv-storage-tests)
    add_test(NAME strix-kv-storage-tests COMMAND strix-kv-storage-tests)
    # Shared KV precision toggle (strix/session/kv_precision.hpp): --kv / serve-config kv, per-tier resolution, per-engine
    # refusals, receipts, the CUDA Qwen dense attn8 adapter.
    add_executable(strix-kv-precision-tests tests/kv_precision_tests.cpp)
    target_include_directories(strix-kv-precision-tests PRIVATE include)
    target_compile_features(strix-kv-precision-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-kv-precision-tests PRIVATE strix_serve_config)
    strix_target_warnings(strix-kv-precision-tests)
    add_test(NAME strix-kv-precision-tests COMMAND strix-kv-precision-tests)

    # Streamed-expert run epochs (both CUDA DS41 expert providers): run counters frozen at end_run, idle preload apart.
    add_executable(strix-expert-stream-epoch-tests tests/expert_stream_epoch_tests.cpp)
    target_include_directories(strix-expert-stream-epoch-tests PRIVATE include)
    target_compile_features(strix-expert-stream-epoch-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-expert-stream-epoch-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-expert-stream-epoch-tests)
    add_test(NAME strix-expert-stream-epoch-tests COMMAND strix-expert-stream-epoch-tests)
    add_executable(strix-iu4a4-rot-host-tests tests/iu4a4_rot_host_tests.cpp)
    target_include_directories(strix-iu4a4-rot-host-tests PRIVATE src)
    target_compile_features(strix-iu4a4-rot-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-iu4a4-rot-host-tests)
    add_test(NAME strix-iu4a4-rot-host-tests COMMAND strix-iu4a4-rot-host-tests)
    add_executable(strix-grouped-iu4a4-host-tests tests/grouped_iu4a4_host_tests.cpp)
    target_include_directories(strix-grouped-iu4a4-host-tests PRIVATE src include)
    target_compile_features(strix-grouped-iu4a4-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-grouped-iu4a4-host-tests)
    add_test(NAME strix-grouped-iu4a4-host-tests COMMAND strix-grouped-iu4a4-host-tests)
    add_executable(strix-dense-pair-map-host-tests tests/dense_pair_map_host_tests.cpp)
    target_include_directories(strix-dense-pair-map-host-tests PRIVATE src)
    target_compile_features(strix-dense-pair-map-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-dense-pair-map-host-tests)
    add_test(NAME strix-dense-pair-map-host-tests COMMAND strix-dense-pair-map-host-tests)
    add_executable(strix-f16-hilo-host-tests tests/f16_hilo_host_tests.cpp)
    target_include_directories(strix-f16-hilo-host-tests PRIVATE src)
    target_compile_features(strix-f16-hilo-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-f16-hilo-host-tests)
    if(MSVC)
        target_compile_options(strix-f16-hilo-host-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-f16-hilo-host-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-f16-hilo-host-tests COMMAND strix-f16-hilo-host-tests)
    add_executable(strix-gdn-recurrent-lds-index-tests tests/gdn_recurrent_lds_index_tests.cpp)
    target_include_directories(strix-gdn-recurrent-lds-index-tests PRIVATE include)
    target_compile_features(strix-gdn-recurrent-lds-index-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-gdn-recurrent-lds-index-tests)
    add_test(NAME strix-gdn-recurrent-lds-index-tests COMMAND strix-gdn-recurrent-lds-index-tests)
    add_executable(strix-gdn-recurrence-rows-tests tests/gdn_recurrence_rows_tests.cpp)
    target_include_directories(strix-gdn-recurrence-rows-tests PRIVATE include)
    target_compile_features(strix-gdn-recurrence-rows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-gdn-recurrence-rows-tests)
    add_test(NAME strix-gdn-recurrence-rows-tests COMMAND strix-gdn-recurrence-rows-tests)
    add_executable(strix-launch-overlap-tests tests/launch_overlap_tests.cpp)
    target_include_directories(strix-launch-overlap-tests PRIVATE include)
    target_compile_features(strix-launch-overlap-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-launch-overlap-tests)
    add_test(NAME strix-launch-overlap-tests COMMAND strix-launch-overlap-tests)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-hip-arch-target-host-tests COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip_arch_target_host_tests.py" --compiler "${CMAKE_CXX_COMPILER}")
        set(controller_rms_host_inc "${CMAKE_CURRENT_BINARY_DIR}/controller_rms_launch_host.inc")
        add_custom_command(OUTPUT "${controller_rms_host_inc}"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/extract_controller_rms_launch.py"
                "${controller_rms_host_inc}"
            DEPENDS tests/core/extract_controller_rms_launch.py src/core/controller_rms_rows.hpp src/core/fma_bf16_rows.hpp
            VERBATIM)
        add_executable(strix-hip-controller-rms-launch-host-test tests/core/controller_rms_launch_host_test.cpp
            "${controller_rms_host_inc}")
        target_include_directories(strix-hip-controller-rms-launch-host-test PRIVATE
            tests/stubs/hip_archguard include "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_features(strix-hip-controller-rms-launch-host-test PRIVATE cxx_std_20)
        strix_target_warnings(strix-hip-controller-rms-launch-host-test)
        add_test(NAME strix-hip-controller-rms-launch-host-test COMMAND strix-hip-controller-rms-launch-host-test)
        add_subdirectory(tests/qwen_cuda_graph_host)
        add_test(NAME strix-hip-arch-caps-host-tests COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip_arch_caps_host_tests.py" --compiler "${CMAKE_CXX_COMPILER}")
        # Bench kit receipts: request / prefill / decode windows and the shared rejection gates.
        add_test(NAME strix-bench-receipt-window-tests COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bench/tests" -p test_request_window.py)
    endif()
endif()
add_library(strix_xport INTERFACE)
target_include_directories(strix_xport INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(strix_xport INTERFACE cxx_std_20)
target_sources(strix_xport INTERFACE
    src/xport/borrowed_transport.hpp include/strix/dist/transport.hpp
    src/xport/channel.hpp src/xport/row_fold.hpp src/xport/channel_communicator.hpp src/xport/channel_device.hpp src/xport/wire.hpp src/xport/small_progress.hpp
    src/xport/transport_backend.hpp src/xport/staging.hpp src/xport/stats.hpp src/xport/backend.hpp src/xport/communicator_backend.hpp
    src/xport/ds41_adapter.hpp src/xport/kvshard_channel.hpp src/xport/kvshard_channel_hip.hpp
    include/strix/dist/host_wait.hpp include/strix/dist/transport_caps.hpp)
if(BUILD_TESTING)
    add_executable(strix-ds41-prefill-structure-tests tests/ds41_prefill_structure_tests.cpp)
    target_link_libraries(strix-ds41-prefill-structure-tests PRIVATE strix_xport)
    strix_target_warnings(strix-ds41-prefill-structure-tests)
    if(MSVC)
        target_compile_options(strix-ds41-prefill-structure-tests PRIVATE /WX /fp:strict)
    else()
        target_compile_options(strix-ds41-prefill-structure-tests PRIVATE -Werror -Wshadow -Wsign-conversion -Wunused-function)
    endif()
    add_test(NAME strix-ds41-prefill-structure-tests COMMAND strix-ds41-prefill-structure-tests)
    set_tests_properties(strix-ds41-prefill-structure-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-borrowed-transport-tests tests/xport_borrowed_transport_tests.cpp)
    target_link_libraries(strix-xport-borrowed-transport-tests PRIVATE strix_xport)
    target_include_directories(strix-xport-borrowed-transport-tests PRIVATE include)
    strix_target_warnings(strix-xport-borrowed-transport-tests)
    if(MSVC)
        target_compile_options(strix-xport-borrowed-transport-tests PRIVATE /WX)
    else()
        target_compile_options(strix-xport-borrowed-transport-tests PRIVATE -Werror -Wshadow -Wsign-conversion -Wunused-function)
    endif()
    add_test(NAME strix-xport-borrowed-transport-tests COMMAND strix-xport-borrowed-transport-tests)
    set_tests_properties(strix-xport-borrowed-transport-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-small-progress-tests tests/xport_small_progress_tests.cpp)
    target_link_libraries(strix-xport-small-progress-tests PRIVATE strix_xport Threads::Threads)
    strix_target_warnings(strix-xport-small-progress-tests)
    if(MSVC)
        target_compile_options(strix-xport-small-progress-tests PRIVATE /WX /fp:strict)
    else()
        target_compile_options(strix-xport-small-progress-tests PRIVATE -Werror -Wshadow -Wsign-conversion -Wunused-function)
    endif()
    add_test(NAME strix-xport-small-progress-tests COMMAND strix-xport-small-progress-tests)
    set_tests_properties(strix-xport-small-progress-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-kvshard-channel-tests tests/xport_kvshard_channel_tests.cpp)
    target_link_libraries(strix-xport-kvshard-channel-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-kvshard-channel-tests)
    if(MSVC)
        target_compile_options(strix-xport-kvshard-channel-tests PRIVATE /WX /fp:strict)
    endif()
    add_test(NAME strix-xport-kvshard-channel-tests COMMAND strix-xport-kvshard-channel-tests)
    set_tests_properties(strix-xport-kvshard-channel-tests PROPERTIES LABELS "cpu;xport;kvshard")
    add_executable(strix-xport-kvshard-oracle-tests tests/xport_kvshard_oracle_tests.cpp)
    add_executable(strix-glm-kvshard-layout-tests tests/glm_kvshard_layout_tests.cpp)
    target_link_libraries(strix-glm-kvshard-layout-tests PRIVATE strix_xport)
    target_compile_features(strix-glm-kvshard-layout-tests PRIVATE cxx_std_17)
    strix_target_warnings(strix-glm-kvshard-layout-tests)
    if(MSVC)
        target_compile_options(strix-glm-kvshard-layout-tests PRIVATE /WX /fp:strict)
    else()
        target_compile_options(strix-glm-kvshard-layout-tests PRIVATE -Werror -Wshadow -Wsign-conversion -Wunused-function -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-glm-kvshard-layout-tests COMMAND strix-glm-kvshard-layout-tests)
    set_tests_properties(strix-glm-kvshard-layout-tests PROPERTIES LABELS "cpu;glm;kvshard")
    add_executable(strix-glm-kvshard-anomaly-tests tests/glm_kvshard_anomaly_tests.cpp)
    target_link_libraries(strix-glm-kvshard-anomaly-tests PRIVATE strix_xport)
    strix_target_warnings(strix-glm-kvshard-anomaly-tests)
    if(MSVC)
        target_compile_options(strix-glm-kvshard-anomaly-tests PRIVATE /WX /fp:strict)
    else()
        target_compile_options(strix-glm-kvshard-anomaly-tests PRIVATE -Werror -Wshadow -Wsign-conversion -Wunused-function -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-glm-kvshard-anomaly-tests COMMAND strix-glm-kvshard-anomaly-tests)
    set_tests_properties(strix-glm-kvshard-anomaly-tests PROPERTIES LABELS "cpu;glm;kvshard;oracle")
    # Shared compressed group indexer contract (gated-ape policy; GLM pool bit identity on the host).
    add_executable(strix-group-indexer-tests tests/group_indexer_tests.cpp src/attn/group_indexer_ref.cpp)
    target_include_directories(strix-group-indexer-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-group-indexer-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-group-indexer-tests)
    if(MSVC)
        target_compile_options(strix-group-indexer-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-group-indexer-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-group-indexer-tests COMMAND strix-group-indexer-tests)
    set_tests_properties(strix-group-indexer-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared group pool statements (HIP block core src/attn_core/group_pool.hpp): gated-ape lanes == oracle, GLM sites ==
    # the pre-lift statements, tie keys.
    add_executable(strix-group-pool-tests tests/group_pool_tests.cpp src/attn/group_indexer_ref.cpp)
    target_include_directories(strix-group-pool-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-group-pool-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-group-pool-tests)
    if(MSVC)
        target_compile_options(strix-group-pool-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-group-pool-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-group-pool-tests COMMAND strix-group-pool-tests)
    set_tests_properties(strix-group-pool-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared Threshold select schedule (HIP body src/attn_core/select_threshold.hpp): both threshold schedules, the
    # two-level slice / merge form and the candidate-block pass == the top-k rule of select.hpp; frozen order key.
    add_executable(strix-select-threshold-tests tests/select_threshold_tests.cpp)
    target_include_directories(strix-select-threshold-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-select-threshold-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-select-threshold-tests)
    add_test(NAME strix-select-threshold-tests COMMAND strix-select-threshold-tests)
    set_tests_properties(strix-select-threshold-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared eager selected softmax (HIP body src/attn_core/eager_softmax.hpp): strided lane partials, LDS halving trees,
    # failure checks and rounded probabilities (host rule eager_softmax_rule.hpp).
    add_executable(strix-eager-softmax-tests tests/eager_softmax_tests.cpp)
    target_include_directories(strix-eager-softmax-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-eager-softmax-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-eager-softmax-tests)
    add_test(NAME strix-eager-softmax-tests COMMAND strix-eager-softmax-tests)
    set_tests_properties(strix-eager-softmax-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared eager selected score / value forms (HIP bodies src/attn_core/eager_selected.hpp): one product per lane +
    # LDS halving tree, ascending fused value chain, gated rounded epilogue (host rule eager_selected_rule.hpp).
    add_executable(strix-eager-selected-tests tests/eager_selected_tests.cpp)
    target_include_directories(strix-eager-selected-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-eager-selected-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-eager-selected-tests)
    if(MSVC)
        target_compile_options(strix-eager-selected-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-eager-selected-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-eager-selected-tests COMMAND strix-eager-selected-tests)
    set_tests_properties(strix-eager-selected-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared Radix3 select schedule (HIP body src/attn_core/select_radix3.hpp): three-pass radix, ordered compaction, zero fill
    # and the candidate-group form == the serial first-k-blocks rule; frozen zero-folding order key.
    add_executable(strix-select-radix3-tests tests/select_radix3_tests.cpp)
    target_include_directories(strix-select-radix3-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-select-radix3-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-select-radix3-tests)
    if(MSVC)
        target_compile_options(strix-select-radix3-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-select-radix3-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-select-radix3-tests COMMAND strix-select-radix3-tests)
    set_tests_properties(strix-select-radix3-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared MLA attention contract (DSA / SWA layers, GLM-5 NoPE): fp64 host oracle, absorbed == reference form.
    add_executable(strix-mla-contract-tests tests/mla_contract_tests.cpp src/attn/mla_ref.cpp)
    target_include_directories(strix-mla-contract-tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(strix-mla-contract-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-mla-contract-tests)
    add_test(NAME strix-mla-contract-tests COMMAND strix-mla-contract-tests)
    set_tests_properties(strix-mla-contract-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared EXACT materialised rows-selected latent attention (HIP body src/attn_core/selected_exact_mla.hpp): the lanes
    # o-bar counter tree == the 256-lane halving tree bit for bit; the lane-major probability layout is a bijection.
    add_executable(strix-selected-exact-mla-tests tests/selected_exact_mla_tests.cpp)
    target_include_directories(strix-selected-exact-mla-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-selected-exact-mla-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-selected-exact-mla-tests)
    add_test(NAME strix-selected-exact-mla-tests COMMAND strix-selected-exact-mla-tests)
    set_tests_properties(strix-selected-exact-mla-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared EXACT chunk-partial combine (HIP body src/attn_core/chunk_exact_combine.hpp): host rule == lane form ==
    # scalar form, bit for bit.
    add_executable(strix-chunk-exact-combine-tests tests/chunk_exact_combine_tests.cpp)
    target_include_directories(strix-chunk-exact-combine-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-chunk-exact-combine-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-chunk-exact-combine-tests)
    if(MSVC)
        target_compile_options(strix-chunk-exact-combine-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-chunk-exact-combine-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-chunk-exact-combine-tests COMMAND strix-chunk-exact-combine-tests)
    set_tests_properties(strix-chunk-exact-combine-tests PROPERTIES LABELS "cpu;attn;oracle")
    # Shared attention cache-row contract (MLA latent rows, SWA ring, indexer token keys; GLM cache stores bind it).
    # Dense site geometry and selection thresholds derived from the model constants (host, equal to the previous literals).
    add_executable(strix-dense-site-policy-tests tests/dense_site_policy_tests.cpp)
    target_include_directories(strix-dense-site-policy-tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(strix-dense-site-policy-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-dense-site-policy-tests)
    add_test(NAME strix-dense-site-policy-tests COMMAND strix-dense-site-policy-tests)
    set_tests_properties(strix-dense-site-policy-tests PROPERTIES LABELS "cpu;core")
    add_executable(strix-cache-rows-tests tests/cache_rows_tests.cpp)
    target_include_directories(strix-cache-rows-tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(strix-cache-rows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-cache-rows-tests)
    add_test(NAME strix-cache-rows-tests COMMAND strix-cache-rows-tests)
    set_tests_properties(strix-cache-rows-tests PROPERTIES LABELS "cpu;attn")
    # Backend-neutral decode lane contract (launch blocks and their packed rows, distinct lane caches).
    add_executable(strix-prepared-lanes-tests tests/prepared_lanes_tests.cpp)
    target_include_directories(strix-prepared-lanes-tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(strix-prepared-lanes-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-prepared-lanes-tests)
    add_test(NAME strix-prepared-lanes-tests COMMAND strix-prepared-lanes-tests)
    set_tests_properties(strix-prepared-lanes-tests PROPERTIES LABELS "cpu;attn")
    # Multi-request decode programs per width class: width rule, X3 cohort step block, lane resolution, policy.
    add_executable(strix-step-lanes-tests tests/step_lanes_tests.cpp)
    target_include_directories(strix-step-lanes-tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(strix-step-lanes-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-step-lanes-tests)
    add_test(NAME strix-step-lanes-tests COMMAND strix-step-lanes-tests)
    set_tests_properties(strix-step-lanes-tests PROPERTIES LABELS "cpu;serve")
    # Shared KV shadow contract (strix/attn/cache_shadow.hpp: F16 / int8-g16 shadow of an FP32 master, tag check) vs the
    # scalar form of the engine mirror writers and reader conversions it replaces.
    add_executable(strix-cache-shadow-tests tests/cache_shadow_tests.cpp)
    target_include_directories(strix-cache-shadow-tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(strix-cache-shadow-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-cache-shadow-tests)
    add_test(NAME strix-cache-shadow-tests COMMAND strix-cache-shadow-tests)
    set_tests_properties(strix-cache-shadow-tests PROPERTIES LABELS "cpu;attn")
    # Decode lanes of the prepared BF16 flash: per-lane split, workspace region and image rows (batch invariance).
    add_executable(strix-prepared-bf16-lanes-plan-tests tests/prepared_bf16_lanes_plan_tests.cpp)
    target_include_directories(strix-prepared-bf16-lanes-plan-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(strix-prepared-bf16-lanes-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-prepared-bf16-lanes-plan-tests)
    add_test(NAME strix-prepared-bf16-lanes-plan-tests COMMAND strix-prepared-bf16-lanes-plan-tests)
    set_tests_properties(strix-prepared-bf16-lanes-plan-tests PROPERTIES LABELS "cpu;attn")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-glm-dsa-policy-identity-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/glm_dsa_policy_identity_tests.py)
        set_tests_properties(strix-glm-dsa-policy-identity-tests PROPERTIES LABELS "cpu;glm;kvshard;identity")
        add_test(NAME strix-glm-trellis-identity-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/glm_trellis_identity_tests.py)
    endif()
    target_link_libraries(strix-xport-kvshard-oracle-tests PRIVATE strix_xport)
    target_compile_features(strix-xport-kvshard-oracle-tests PRIVATE cxx_std_17)
    strix_target_warnings(strix-xport-kvshard-oracle-tests)
    if(MSVC)
        target_compile_options(strix-xport-kvshard-oracle-tests PRIVATE /WX /fp:strict)
    else()
        target_compile_options(strix-xport-kvshard-oracle-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-xport-kvshard-oracle-tests COMMAND strix-xport-kvshard-oracle-tests)
    set_tests_properties(strix-xport-kvshard-oracle-tests PROPERTIES LABELS "cpu;xport;oracle")
    add_executable(strix-xport-minimax-pfx-tests tests/xport_minimax_pfx_tests.cpp)
    target_link_libraries(strix-xport-minimax-pfx-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-minimax-pfx-tests)
    if(MSVC)
        target_compile_options(strix-xport-minimax-pfx-tests PRIVATE /WX)
    endif()
    add_test(NAME strix-xport-minimax-pfx-tests COMMAND strix-xport-minimax-pfx-tests)
    set_tests_properties(strix-xport-minimax-pfx-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-ds41-raw-slots-tests tests/xport_ds41_raw_slots_tests.cpp)
    target_link_libraries(strix-xport-ds41-raw-slots-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-ds41-raw-slots-tests)
    if(MSVC)
        target_compile_options(strix-xport-ds41-raw-slots-tests PRIVATE /WX)
    endif()
    add_test(NAME strix-xport-ds41-raw-slots-tests COMMAND strix-xport-ds41-raw-slots-tests)
    set_tests_properties(strix-xport-ds41-raw-slots-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-ds41-half-sum-tests tests/xport_ds41_half_sum_tests.cpp)
    target_link_libraries(strix-xport-ds41-half-sum-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-ds41-half-sum-tests)
    add_test(NAME strix-xport-ds41-half-sum-tests COMMAND strix-xport-ds41-half-sum-tests)
    set_tests_properties(strix-xport-ds41-half-sum-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-ds41-adapter-tests tests/xport_ds41_adapter_tests.cpp)
    target_link_libraries(strix-xport-ds41-adapter-tests PRIVATE strix_xport)
    target_include_directories(strix-xport-ds41-adapter-tests PRIVATE include)
    strix_target_warnings(strix-xport-ds41-adapter-tests)
    add_test(NAME strix-xport-ds41-adapter-tests COMMAND strix-xport-ds41-adapter-tests)
    set_tests_properties(strix-xport-ds41-adapter-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-transport-backend-tests tests/xport_transport_backend_tests.cpp)
    target_link_libraries(strix-xport-transport-backend-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-transport-backend-tests)
    add_test(NAME strix-xport-transport-backend-tests COMMAND strix-xport-transport-backend-tests)
    set_tests_properties(strix-xport-transport-backend-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-staging-tests tests/xport_staging_tests.cpp)
    target_link_libraries(strix-xport-staging-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-staging-tests)
    add_test(NAME strix-xport-staging-tests COMMAND strix-xport-staging-tests)
    set_tests_properties(strix-xport-staging-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-glm-wire-tests tests/xport_glm_wire_tests.cpp)
    target_link_libraries(strix-xport-glm-wire-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-glm-wire-tests)
    add_test(NAME strix-xport-glm-wire-tests COMMAND strix-xport-glm-wire-tests)
    set_tests_properties(strix-xport-glm-wire-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-row-fold-tests tests/xport_row_fold_tests.cpp)
    target_link_libraries(strix-xport-row-fold-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-row-fold-tests)
    add_test(NAME strix-xport-row-fold-tests COMMAND strix-xport-row-fold-tests)
    set_tests_properties(strix-xport-row-fold-tests PROPERTIES LABELS "cpu;xport;exactness")
    add_executable(strix-xport-minimax-wire-tests tests/xport_minimax_wire_tests.cpp)
    target_link_libraries(strix-xport-minimax-wire-tests PRIVATE strix_xport)
    strix_target_warnings(strix-xport-minimax-wire-tests)
    add_test(NAME strix-xport-minimax-wire-tests COMMAND strix-xport-minimax-wire-tests)
    set_tests_properties(strix-xport-minimax-wire-tests PROPERTIES LABELS "cpu;xport;exactness")
    include(cmake/StrixMoePairGather.cmake)
    add_executable(strix-moe-grouped-prefill-host-tests tests/moe_grouped_prefill_host_tests.cpp)
    target_include_directories(strix-moe-grouped-prefill-host-tests PRIVATE src)
    target_compile_features(strix-moe-grouped-prefill-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-moe-grouped-prefill-host-tests)
    add_test(NAME strix-moe-grouped-prefill-host-tests COMMAND strix-moe-grouped-prefill-host-tests)
    set_tests_properties(strix-moe-grouped-prefill-host-tests PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-core-exl3-rows-plan-tests tests/core/exl3_rows_plan_test.cpp)
    target_include_directories(strix-core-exl3-rows-plan-tests PRIVATE src)
    target_compile_features(strix-core-exl3-rows-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-core-exl3-rows-plan-tests)
    add_test(NAME strix-core-exl3-rows-plan-tests COMMAND strix-core-exl3-rows-plan-tests)
    set_tests_properties(strix-core-exl3-rows-plan-tests PROPERTIES LABELS "cpu;core;exactness")
    add_executable(strix-numeric-partition-host-tests tests/moe_core/numeric_partition_host_tests.cpp)
    target_include_directories(strix-numeric-partition-host-tests PRIVATE src)
    target_compile_features(strix-numeric-partition-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-numeric-partition-host-tests)
    add_test(NAME strix-numeric-partition-host-tests COMMAND strix-numeric-partition-host-tests)
    set_tests_properties(strix-numeric-partition-host-tests PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-microscaled-tile-host-tests tests/moe_core/microscaled_tile_host_tests.cpp)
    target_include_directories(strix-microscaled-tile-host-tests PRIVATE src)
    target_link_libraries(strix-microscaled-tile-host-tests PRIVATE strix_core)
    strix_target_warnings(strix-microscaled-tile-host-tests)
    add_test(NAME strix-microscaled-tile-host-tests COMMAND strix-microscaled-tile-host-tests)
    set_tests_properties(strix-microscaled-tile-host-tests PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-grouped-tall-plan-test tests/moe_core/grouped_tall_plan_test.cpp)
    target_include_directories(strix-grouped-tall-plan-test PRIVATE src)
    target_link_libraries(strix-grouped-tall-plan-test PRIVATE strix_core)
    strix_target_warnings(strix-grouped-tall-plan-test)
    add_test(NAME strix-grouped-tall-plan-test COMMAND strix-grouped-tall-plan-test)
    set_tests_properties(strix-grouped-tall-plan-test PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-microscaled-shift-decode-test tests/weight/microscaled_shift_decode_test.cpp)
    target_link_libraries(strix-microscaled-shift-decode-test PRIVATE strix_core)
    strix_target_warnings(strix-microscaled-shift-decode-test)
    add_test(NAME strix-microscaled-shift-decode-test COMMAND strix-microscaled-shift-decode-test)
    set_tests_properties(strix-microscaled-shift-decode-test PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-grouped-w4a4-host-test tests/moe_core/grouped_w4a4_host_test.cpp)
    target_include_directories(strix-grouped-w4a4-host-test PRIVATE src)
    target_link_libraries(strix-grouped-w4a4-host-test PRIVATE strix_core)
    strix_target_warnings(strix-grouped-w4a4-host-test)
    add_test(NAME strix-grouped-w4a4-host-test COMMAND strix-grouped-w4a4-host-test)
    set_tests_properties(strix-grouped-w4a4-host-test PROPERTIES LABELS "cpu;moe")
    add_executable(strix-nvfp4-act-rule-test tests/moe_core/nvfp4_act_rule_test.cpp)
    target_include_directories(strix-nvfp4-act-rule-test PRIVATE src)
    target_link_libraries(strix-nvfp4-act-rule-test PRIVATE strix_core)
    strix_target_warnings(strix-nvfp4-act-rule-test)
    add_test(NAME strix-nvfp4-act-rule-test COMMAND strix-nvfp4-act-rule-test)
    set_tests_properties(strix-nvfp4-act-rule-test PROPERTIES LABELS "cpu;moe")
    add_executable(strix-microscaled-r1-columns-test tests/moe_core/microscaled_r1_columns_test.cpp)
    target_include_directories(strix-microscaled-r1-columns-test PRIVATE src)
    target_link_libraries(strix-microscaled-r1-columns-test PRIVATE strix_core)
    strix_target_warnings(strix-microscaled-r1-columns-test)
    add_test(NAME strix-microscaled-r1-columns-test COMMAND strix-microscaled-r1-columns-test)
    set_tests_properties(strix-microscaled-r1-columns-test PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-moe-class-tree-host-test tests/moe_core/class_tree_host_test.cpp)
    target_include_directories(strix-moe-class-tree-host-test PRIVATE src)
    target_compile_features(strix-moe-class-tree-host-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-moe-class-tree-host-test)
    if(MSVC)
        target_compile_options(strix-moe-class-tree-host-test PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-class-tree-host-test PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-moe-class-tree-host-test COMMAND strix-moe-class-tree-host-test)
    set_tests_properties(strix-moe-class-tree-host-test PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-moe-decode-down-frame-test tests/moe_core/decode_down_frame_test.cpp)
    target_include_directories(strix-moe-decode-down-frame-test PRIVATE src)
    target_compile_features(strix-moe-decode-down-frame-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-moe-decode-down-frame-test)
    if(MSVC)
        target_compile_options(strix-moe-decode-down-frame-test PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-decode-down-frame-test PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-moe-decode-down-frame-test COMMAND strix-moe-decode-down-frame-test)
    set_tests_properties(strix-moe-decode-down-frame-test PROPERTIES LABELS "cpu;moe;exactness")
    add_executable(strix-microscaled-r1-rows-chunk-test tests/moe_core/microscaled_r1_rows_chunk_test.cpp)
    target_include_directories(strix-microscaled-r1-rows-chunk-test PRIVATE src)
    target_link_libraries(strix-microscaled-r1-rows-chunk-test PRIVATE strix_core strix_device)
    strix_target_warnings(strix-microscaled-r1-rows-chunk-test)
    add_test(NAME strix-microscaled-r1-rows-chunk-test COMMAND strix-microscaled-r1-rows-chunk-test)
    set_tests_properties(strix-microscaled-r1-rows-chunk-test PROPERTIES LABELS "cpu;moe;exactness")
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-microscaled-r1-bench-status-test
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/moe_core/microscaled_r1_bench_status_test.py)
        set_tests_properties(strix-microscaled-r1-bench-status-test PROPERTIES LABELS "cpu;moe")
    endif()
    add_executable(strix-dense-linear-group-test tests/core/dense_linear_group_test.cpp)
    target_include_directories(strix-dense-linear-group-test PRIVATE src)
    strix_target_warnings(strix-dense-linear-group-test)
    add_test(NAME strix-dense-linear-group-test COMMAND strix-dense-linear-group-test)
    set_tests_properties(strix-dense-linear-group-test PROPERTIES LABELS "cpu;exactness")
    # Byte-range overlap of launch operands (reads vs writes of one launch).
    add_executable(strix-byte-range-test tests/core/byte_range_test.cpp)
    target_include_directories(strix-byte-range-test PRIVATE src)
    target_compile_features(strix-byte-range-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-byte-range-test)
    add_test(NAME strix-byte-range-test COMMAND strix-byte-range-test)
    set_tests_properties(strix-byte-range-test PROPERTIES LABELS "cpu")
    # EXACT screened LM head: plane builder and the error bound (candidate set contains the true argmax and every tie).
    add_executable(strix-screened-head-tests tests/screened_head_tests.cpp)
    target_include_directories(strix-screened-head-tests PRIVATE src include)
    target_compile_features(strix-screened-head-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-screened-head-tests)
    add_test(NAME strix-screened-head-tests COMMAND strix-screened-head-tests)
    set_tests_properties(strix-screened-head-tests PROPERTIES LABELS "cpu;exactness" TIMEOUT 600)
    add_executable(strix-minimax-fast-i4-schedule-tests tests/minimax_fast_i4_schedule_tests.cpp)
    target_include_directories(strix-minimax-fast-i4-schedule-tests PRIVATE src)
    target_compile_features(strix-minimax-fast-i4-schedule-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-minimax-fast-i4-schedule-tests)
    add_test(NAME strix-minimax-fast-i4-schedule-tests COMMAND strix-minimax-fast-i4-schedule-tests)
    set_tests_properties(strix-minimax-fast-i4-schedule-tests PROPERTIES LABELS "cpu;minimax;fast-i4")
    add_executable(strix-env-flags-tests tests/env_flags_tests.cpp)
    target_include_directories(strix-env-flags-tests PRIVATE include)
    target_compile_features(strix-env-flags-tests PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_definitions(strix-env-flags-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-env-flags-tests)
    add_test(NAME strix-env-flags-tests COMMAND strix-env-flags-tests)
    set_tests_properties(strix-env-flags-tests PROPERTIES LABELS "cpu;env")
    add_executable(strix-runtime-host-memory-tests tests/runtime_host_memory_tests.cpp)
    target_include_directories(strix-runtime-host-memory-tests PRIVATE include)
    target_compile_features(strix-runtime-host-memory-tests PRIVATE cxx_std_20)
    target_compile_definitions(strix-runtime-host-memory-tests PRIVATE
        STRIX_RUNTIME_HOST_MEMORY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    if(MSVC)
        target_compile_definitions(strix-runtime-host-memory-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-runtime-host-memory-tests)
    add_test(NAME strix-runtime-host-memory-tests COMMAND strix-runtime-host-memory-tests)
    set_tests_properties(strix-runtime-host-memory-tests PROPERTIES LABELS "cpu;host-memory")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    find_program(STRIX_PFAB_TEST_BASH NAMES bash)
    if(Python3_Interpreter_FOUND AND STRIX_PFAB_TEST_BASH)
        foreach(case summary pair)
            add_test(NAME strix-pfab-kit-tests-${case} COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bench/pfab/tests/test_pfab_kit.py"
                ${case} "${STRIX_PFAB_TEST_BASH}")
            set_tests_properties(strix-pfab-kit-tests-${case} PROPERTIES
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;PFAB_TEST_TMP=${CMAKE_CURRENT_BINARY_DIR}")
        endforeach()
        set_tests_properties(strix-pfab-kit-tests-pair PROPERTIES TIMEOUT 90)
    endif()
    add_executable(strix-chunk-contract-tests tests/chunk_contract_tests.cpp)
    target_include_directories(strix-chunk-contract-tests PRIVATE src)
    target_compile_features(strix-chunk-contract-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-chunk-contract-tests)
    add_test(NAME strix-chunk-contract-tests COMMAND strix-chunk-contract-tests)
    set_tests_properties(strix-chunk-contract-tests PROPERTIES LABELS "cpu;chunkinv")
    add_executable(strix-gemv-rows-lookahead-tests tests/gemv_rows_lookahead_tests.cpp)
    target_include_directories(strix-gemv-rows-lookahead-tests PRIVATE src)
    target_compile_features(strix-gemv-rows-lookahead-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-gemv-rows-lookahead-tests)
    # Bitwise schedule check: the host must not contract acc + w * x into FMA in some forms only (-mfma hosts).
    if(NOT MSVC)
        target_compile_options(strix-gemv-rows-lookahead-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-gemv-rows-lookahead-tests COMMAND strix-gemv-rows-lookahead-tests)
    set_tests_properties(strix-gemv-rows-lookahead-tests PROPERTIES LABELS "cpu")
    add_executable(strix-lds-rows-tests tests/lds_rows_tests.cpp)
    target_include_directories(strix-lds-rows-tests PRIVATE src)
    target_compile_features(strix-lds-rows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-lds-rows-tests)
    add_test(NAME strix-lds-rows-tests COMMAND strix-lds-rows-tests)
    set_tests_properties(strix-lds-rows-tests PROPERTIES LABELS "cpu")
    add_executable(strix-vgpr-budget-tests tests/vgpr_budget_tests.cpp)
    target_include_directories(strix-vgpr-budget-tests PRIVATE src)
    target_compile_features(strix-vgpr-budget-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-vgpr-budget-tests)
    add_test(NAME strix-vgpr-budget-tests COMMAND strix-vgpr-budget-tests)
    set_tests_properties(strix-vgpr-budget-tests PROPERTIES LABELS "cpu")
    find_program(STRIX_CHUNKINV_BASH NAMES bash)
    if(STRIX_CHUNKINV_BASH)
        add_test(NAME strix-chunkinv-fixtures-tests COMMAND "${STRIX_CHUNKINV_BASH}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/chunkinv_fixtures_tests.sh"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bench/chunkinv/fixtures.sh"
            "${CMAKE_CURRENT_BINARY_DIR}/chunkinv-fixtures-host")
        set_tests_properties(strix-chunkinv-fixtures-tests PROPERTIES LABELS "cpu;chunkinv")
    endif()
    add_executable(strix-engine-profile-common-tests tests/engine_profile_common_tests.cpp)
    target_include_directories(strix-engine-profile-common-tests PRIVATE include apps)
    target_compile_features(strix-engine-profile-common-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-engine-profile-common-tests)
    add_test(NAME strix-engine-profile-common-tests COMMAND strix-engine-profile-common-tests)
    set_tests_properties(strix-engine-profile-common-tests PROPERTIES LABELS "cpu;profile")
    add_executable(strix-profile-identity-variant-tests tests/profile_identity_variant_tests.cpp)
    target_include_directories(strix-profile-identity-variant-tests PRIVATE include)
    target_compile_features(strix-profile-identity-variant-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-profile-identity-variant-tests)
    add_test(NAME strix-profile-identity-variant-tests COMMAND strix-profile-identity-variant-tests)
    set_tests_properties(strix-profile-identity-variant-tests PROPERTIES LABELS "cpu;profile")
    add_executable(strix-glm5-seqpar-cpu-tests tests/glm_seqpar_cpu_tests.cpp)
    target_include_directories(strix-glm5-seqpar-cpu-tests PRIVATE src)
    target_link_libraries(strix-glm5-seqpar-cpu-tests PRIVATE strix_model_data)
    target_compile_features(strix-glm5-seqpar-cpu-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-glm5-seqpar-cpu-tests)
    add_test(NAME strix-glm5-seqpar-cpu-tests COMMAND strix-glm5-seqpar-cpu-tests)
    set_tests_properties(strix-glm5-seqpar-cpu-tests PROPERTIES LABELS "cpu;glm;exactness")
    add_executable(strix-glm5-pf-workspace-cpu-tests tests/glm_pf_workspace_cpu_tests.cpp)
    target_include_directories(strix-glm5-pf-workspace-cpu-tests PRIVATE src apps include)
    target_compile_features(strix-glm5-pf-workspace-cpu-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_definitions(strix-glm5-pf-workspace-cpu-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(strix-glm5-pf-workspace-cpu-tests PRIVATE /utf-8)
    endif()
    strix_target_warnings(strix-glm5-pf-workspace-cpu-tests)
    add_test(NAME strix-glm5-pf-workspace-cpu-tests COMMAND strix-glm5-pf-workspace-cpu-tests)
    set_tests_properties(strix-glm5-pf-workspace-cpu-tests PROPERTIES LABELS "cpu;glm;exactness")
    add_executable(strix-glm5-kvshard-prefill-cpu-tests tests/glm_kvshard_prefill_cpu_tests.cpp)
    target_include_directories(strix-glm5-kvshard-prefill-cpu-tests PRIVATE src)
    target_link_libraries(strix-glm5-kvshard-prefill-cpu-tests PRIVATE strix_model_data)
    target_compile_features(strix-glm5-kvshard-prefill-cpu-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-glm5-kvshard-prefill-cpu-tests)
    if(MSVC)
        target_compile_options(strix-glm5-kvshard-prefill-cpu-tests PRIVATE /WX /fp:strict)
    else()
        target_compile_options(strix-glm5-kvshard-prefill-cpu-tests PRIVATE
            -Werror -Wshadow -Wsign-conversion -Wunused-function -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-glm5-kvshard-prefill-cpu-tests COMMAND strix-glm5-kvshard-prefill-cpu-tests)
    set_tests_properties(strix-glm5-kvshard-prefill-cpu-tests PROPERTIES LABELS "cpu;glm;exactness")
    add_executable(strix-glm5-kda-fuse-cpu-tests tests/glm_kda_fuse_cpu_tests.cpp)
    target_include_directories(strix-glm5-kda-fuse-cpu-tests PRIVATE src)
    target_link_libraries(strix-glm5-kda-fuse-cpu-tests PRIVATE strix_model_data)
    target_compile_features(strix-glm5-kda-fuse-cpu-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-glm5-kda-fuse-cpu-tests)
    if(MSVC)
        target_compile_options(strix-glm5-kda-fuse-cpu-tests PRIVATE /WX /fp:strict)
    else()
        target_compile_options(strix-glm5-kda-fuse-cpu-tests PRIVATE
            -Werror -Wshadow -Wsign-conversion -Wunused-function -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-glm5-kda-fuse-cpu-tests COMMAND strix-glm5-kda-fuse-cpu-tests)
    set_tests_properties(strix-glm5-kda-fuse-cpu-tests PROPERTIES LABELS "cpu;glm;exactness")
    add_executable(strix-glm5-rank-defaults-common-cpu-tests tests/glm_rank_defaults_common_cpu_tests.cpp)
    target_include_directories(strix-glm5-rank-defaults-common-cpu-tests PRIVATE include apps src)
    target_compile_features(strix-glm5-rank-defaults-common-cpu-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_definitions(strix-glm5-rank-defaults-common-cpu-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(strix-glm5-rank-defaults-common-cpu-tests PRIVATE /utf-8)
    endif()
    strix_target_warnings(strix-glm5-rank-defaults-common-cpu-tests)
    add_test(NAME strix-glm5-rank-defaults-common-cpu-tests COMMAND strix-glm5-rank-defaults-common-cpu-tests)
    set_tests_properties(strix-glm5-rank-defaults-common-cpu-tests PROPERTIES LABELS "cpu;profile")
endif()
set(STRIX_TIERSTREAM_DIR "" CACHE PATH "tierstream source tree (empty: no multi-disk reads)")
if(STRIX_TIERSTREAM_DIR)
    if(STRIX_ENABLE_CUDA_DS41)
        set(strix_tierstream_build "build-cuda")
    else()
        set(strix_tierstream_build "build-cpu")
    endif()
    set(STRIX_TIERSTREAM_LIBRARY "${STRIX_TIERSTREAM_DIR}/${strix_tierstream_build}/tierstream${CMAKE_STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "tierstream archive matching this build's backend (may be an out-of-tree build)")
    add_library(strix_tierstream STATIC IMPORTED)
    set_target_properties(strix_tierstream PROPERTIES IMPORTED_LOCATION "${STRIX_TIERSTREAM_LIBRARY}"
                  INTERFACE_INCLUDE_DIRECTORIES "${STRIX_TIERSTREAM_DIR}/include")
    target_link_libraries(strix_tierstream INTERFACE Threads::Threads "$<$<PLATFORM_ID:Windows>:avrt;winmm;bcrypt>")
    file(STRINGS "${STRIX_TIERSTREAM_DIR}/include/strix_stream/ss.h" strix_mfile_define REGEX "^#define SS_MFILE +1")
    if(strix_mfile_define)
        target_compile_definitions(strix_tierstream INTERFACE STRIX_HAVE_TIERSTREAM_MFILE=1)
    endif()
    # Device profiles (ss_devprof_*): the library loads the HIP / CUDA / NVML runtimes at run time.
    file(STRINGS "${STRIX_TIERSTREAM_DIR}/include/strix_stream/ss.h" strix_devprof_define REGEX "^#define SS_DEVPROF +1")
    if(strix_devprof_define)
        target_compile_definitions(strix_tierstream INTERFACE STRIX_HAVE_TIERSTREAM_DEVPROF=1)
        target_link_libraries(strix_tierstream INTERFACE ${CMAKE_DL_LIBS})
    endif()
endif()

if(BUILD_TESTING)
    add_executable(strix-ds41-file-reader-tests tests/ds41_file_reader_multi_tests.cpp src/ds41_file_reader.cpp)
    target_include_directories(strix-ds41-file-reader-tests PRIVATE include)
    target_compile_features(strix-ds41-file-reader-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-ds41-file-reader-tests PRIVATE Threads::Threads)
    if(STRIX_TIERSTREAM_DIR AND strix_mfile_define)
        target_link_libraries(strix-ds41-file-reader-tests PRIVATE strix_tierstream)
        target_include_directories(strix-ds41-file-reader-tests PRIVATE "${STRIX_TIERSTREAM_DIR}/src")
    endif()
    add_test(NAME strix-ds41-file-reader-tests COMMAND strix-ds41-file-reader-tests)
    set_tests_properties(strix-ds41-file-reader-tests PROPERTIES LABELS "cpu;io" TIMEOUT 60)
endif()
if(STRIX_TIERSTREAM_DIR AND strix_mfile_define)
    add_executable(strix-ds41-read-bench apps/ds41_read_bench.cpp src/ds41_file_reader.cpp)
    target_include_directories(strix-ds41-read-bench PRIVATE include)
    target_compile_features(strix-ds41-read-bench PRIVATE cxx_std_20)
    target_link_libraries(strix-ds41-read-bench PRIVATE strix_tierstream Threads::Threads)
    if(BUILD_TESTING)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME strix-ds41-read-tools-tests COMMAND ${Python3_EXECUTABLE} -B
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/ds41_read_tools_tests.py --bench $<TARGET_FILE:strix-ds41-read-bench>)
            set_tests_properties(strix-ds41-read-tools-tests PROPERTIES LABELS "cpu;io" TIMEOUT 60)
        endif()
    endif()
endif()
# Model readers and placement tools need JSON without the POSIX HTTP server.
add_library(strix_serve_json STATIC src/serve/json.cpp)
target_include_directories(strix_serve_json PUBLIC include)
target_compile_features(strix_serve_json PUBLIC cxx_std_20)
strix_target_warnings(strix_serve_json)
target_link_libraries(strix_core PUBLIC strix_serve_json)

if(NOT STRIX_CUDA_DS41_ONLY)
if(NOT MSVC)
    set_source_files_properties(src/low_precision.cpp src/projection_reduction.cpp src/integer_projection.cpp src/ggml_dequant.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
else()
    set_source_files_properties(src/low_precision.cpp src/projection_reduction.cpp src/integer_projection.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
endif()

add_executable(strix-low-precision-reference lab/low_precision/reference_main.cpp)
target_link_libraries(strix-low-precision-reference PRIVATE strix_core)
strix_target_warnings(strix-low-precision-reference)

add_executable(strix-memory-plan apps/memory_plan_main.cpp)
target_link_libraries(strix-memory-plan PRIVATE strix_core)
strix_target_warnings(strix-memory-plan)

# Syntax-only model description reader (models/*.toml); model contracts live in explicit per-engine adapters.
# Backend-neutral descriptive graph; no runtime/device dependencies or global src include.
add_library(strix_model_graph STATIC src/model_graph/model_graph.cpp src/model_graph/plan.cpp)
target_compile_features(strix_model_graph PUBLIC cxx_std_20)
strix_target_warnings(strix_model_graph)
# Stage graph: the one stage declaration a model writes, the launch planner (check mode / fused) and the rows planner.
# Host only, backend neutral; strix_model_graph_stage converts a description into it.
add_library(strix_stage_graph STATIC src/graph/stage_graph.cpp src/graph/launch_plan.cpp src/graph/rows_plan.cpp)
target_include_directories(strix_stage_graph PUBLIC include)
target_compile_features(strix_stage_graph PUBLIC cxx_std_20)
strix_target_warnings(strix_stage_graph)
add_library(strix_model_graph_stage STATIC src/model_graph/stage_bridge.cpp)
target_include_directories(strix_model_graph_stage PUBLIC src/model_graph)
target_link_libraries(strix_model_graph_stage PUBLIC strix_model_graph strix_stage_graph)
strix_target_warnings(strix_model_graph_stage)
if(BUILD_TESTING)
    add_executable(strix-stage-graph-tests tests/stage_graph_tests.cpp)
    target_link_libraries(strix-stage-graph-tests PRIVATE strix_model_graph_stage)
    strix_target_warnings(strix-stage-graph-tests)
    add_test(NAME strix-stage-graph-tests COMMAND strix-stage-graph-tests)
    set_tests_properties(strix-stage-graph-tests PROPERTIES LABELS "cpu;plan" TIMEOUT 30)
endif()
include(cmake/StrixDenseModels.cmake)

if(NOT TARGET strix_model_description)
    add_library(strix_model_description STATIC src/model_description.cpp)
endif()
target_include_directories(strix_model_description PUBLIC include)
target_compile_features(strix_model_description PUBLIC cxx_std_20)
strix_target_warnings(strix_model_description)
# Shared --serve-config TOML expander (every serve CLI binds it with its own option table); reads TOML with the
# model-description reader above.
add_library(strix_serve_config STATIC src/serve/serve_config.cpp)
target_include_directories(strix_serve_config PUBLIC include)
target_compile_features(strix_serve_config PUBLIC cxx_std_20)
target_link_libraries(strix_serve_config PUBLIC strix_model_description)
strix_target_warnings(strix_serve_config)
target_link_libraries(strix_core PUBLIC strix_serve_config)

# Deployment planning is host-only. Its TOML reader is the existing model reader.
add_library(strix_deploy STATIC src/deploy/device.cpp src/deploy/spec.cpp src/deploy/plan.cpp src/deploy/link.cpp
    src/deploy/route.cpp)
target_link_libraries(strix_deploy PUBLIC strix_model_description strix_device)
target_compile_features(strix_deploy PUBLIC cxx_std_20)
if(MSVC)
    target_compile_definitions(strix_deploy PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
strix_target_warnings(strix_deploy)
add_library(strix_engine_deploy STATIC src/minimax/engine/deployment.cpp src/gemma4/engine/deployment.cpp src/gguf_reader.cpp)
target_include_directories(strix_engine_deploy PUBLIC src)
target_link_libraries(strix_engine_deploy PUBLIC strix_deploy strix_serve_json)
strix_target_warnings(strix_engine_deploy)
if(BUILD_TESTING)
    add_executable(strix-engine-deploy-tests tests/engine_deploy_tests.cpp)
    target_link_libraries(strix-engine-deploy-tests PRIVATE strix_engine_deploy)
    strix_target_warnings(strix-engine-deploy-tests)
    add_test(NAME strix-engine-deploy-tests COMMAND strix-engine-deploy-tests)
endif()
# Device declaration (strix-devices-v1) and the start gate every GPU binary calls before its first allocation.
include(cmake/StrixDevice.cmake)
# Intel GPU backend (SYCL / oneAPI, Level Zero): opt-in, host-only unless the C++ compiler is icpx (cmake/StrixIntel.cmake).
option(STRIX_BACKEND_INTEL "Build the Intel GPU backend scaffold (SYCL when the compiler is oneAPI DPC++)" OFF)
if(STRIX_BACKEND_INTEL)
    include(cmake/StrixIntel.cmake)
endif()
# Typed lever declaration and the census of environment switches (host tests).
include(cmake/StrixLevers.cmake)
include(cmake/StrixPairState.cmake)
# Every HIP app calls the start gate first; the core library carries it to all of them.
target_link_libraries(strix_core PUBLIC strix_device_hip)
# strix_core (src/serial_request.cpp) and strix_serve (src/serve/api.cpp) call the shared power sampler themselves
# and link it by name (cmake/StrixPower.cmake), not through the gate.
target_link_libraries(strix_core PUBLIC strix_power)
# Effective-config headers are consumed by the core CLI and its host tests.
target_link_libraries(strix_core PUBLIC strix_deploy)
if(BUILD_TESTING)
    add_executable(strix-deploy-tests tests/deploy_tests.cpp)
    target_link_libraries(strix-deploy-tests PRIVATE strix_deploy strix_serve_json)
    strix_target_warnings(strix-deploy-tests)
    add_test(NAME strix-deploy-tests COMMAND strix-deploy-tests)
    add_executable(strix-route-planner-tests tests/route_planner_tests.cpp)
    target_link_libraries(strix-route-planner-tests PRIVATE strix_deploy)
    strix_target_warnings(strix-route-planner-tests)
    add_test(NAME strix-route-planner-tests COMMAND strix-route-planner-tests)
    add_executable(strix-power-sampler-tests tests/power_sampler_tests.cpp)
    target_link_libraries(strix-power-sampler-tests PRIVATE strix_power strix_serve_json)
    strix_target_warnings(strix-power-sampler-tests)
    add_test(NAME strix-power-sampler-tests COMMAND strix-power-sampler-tests)
    # The serving graph of the standalone CUDA Qwen configuration, linked on the host without the gate or the core.
    include(cmake/StrixServeStandalone.cmake)
    strix_add_standalone_serve(strix_serve_standalone_check strix_serve_standalone_handoff_check)
    add_executable(strix-serve-standalone-link-tests tests/serve_standalone_link_tests.cpp)
    target_link_libraries(strix-serve-standalone-link-tests PRIVATE strix_serve_standalone_check)
    add_test(NAME strix-serve-standalone-link-tests COMMAND strix-serve-standalone-link-tests)
    add_executable(strix-route-receipt-tests tests/route_receipt_tests.cpp)
    target_link_libraries(strix-route-receipt-tests PRIVATE strix_serve_json)
    if(MSVC)
        target_compile_definitions(strix-route-receipt-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-route-receipt-tests)
    add_test(NAME strix-route-receipt-tests COMMAND strix-route-receipt-tests)
    # Resident runner (one load, many runs of the same binary): plan, runner, per-run lever caches, DS41 prefill adapter.
    foreach(_resident_test resident_plan resident_session lever_cache deepseek41_prefill_resident)
        string(REPLACE "_" "-" _resident_name "strix-${_resident_test}-cpu-tests")
        add_executable(${_resident_name} tests/${_resident_test}_cpu_tests.cpp)
        target_link_libraries(${_resident_name} PRIVATE strix_serve_json)
        target_include_directories(${_resident_name} PRIVATE apps include)
        target_compile_definitions(${_resident_name} PRIVATE STRIX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        if(MSVC)
            target_compile_definitions(${_resident_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
            target_compile_options(${_resident_name} PRIVATE /utf-8)
        endif()
        strix_target_warnings(${_resident_name})
        add_test(NAME ${_resident_name} COMMAND ${_resident_name})
        set_tests_properties(${_resident_name} PROPERTIES LABELS "cpu;serve" TIMEOUT 60)
    endforeach()
    add_executable(strix-deepseek41-deploy-host-tests tests/deepseek41_deploy_host_tests.cpp)
    target_link_libraries(strix-deepseek41-deploy-host-tests PRIVATE strix_deploy)
    if(MSVC)
        target_compile_definitions(strix-deepseek41-deploy-host-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-deepseek41-deploy-host-tests)
    add_test(NAME strix-deepseek41-deploy-host-tests COMMAND strix-deepseek41-deploy-host-tests)
    foreach(engine glm qwen)
        add_executable(strix-${engine}-deploy-host-tests tests/${engine}_deploy_host_tests.cpp)
        target_include_directories(strix-${engine}-deploy-host-tests PRIVATE src)
        target_link_libraries(strix-${engine}-deploy-host-tests PRIVATE strix_deploy)
        strix_target_warnings(strix-${engine}-deploy-host-tests)
        add_test(NAME strix-${engine}-deploy-host-tests COMMAND strix-${engine}-deploy-host-tests)
    endforeach()
endif()

add_library(strix_model_registry STATIC src/model_registry.cpp)
target_link_libraries(strix_model_registry PUBLIC strix_model_description)
strix_target_warnings(strix_model_registry)
# Model constants derived from models/*.toml (strix/model/constants.hpp). The headers under include/strix/model_data
# are generated by strix-model-gen and committed; strix-model-constants-tests fails on any drift from the TOML.
add_library(strix_model_constants STATIC src/model/constants.cpp)
target_link_libraries(strix_model_constants PUBLIC strix_model_description)
strix_target_warnings(strix_model_constants)
add_executable(strix-model-gen tools/model_gen_main.cpp)
target_link_libraries(strix-model-gen PRIVATE strix_model_constants)
strix_target_warnings(strix-model-gen)
if(BUILD_TESTING)
    add_executable(strix-model-constants-tests tests/model_constants_tests.cpp)
    target_link_libraries(strix-model-constants-tests PRIVATE strix_model_constants strix_model_registry)
    target_compile_definitions(strix-model-constants-tests PRIVATE STRIX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    strix_target_warnings(strix-model-constants-tests)
    add_test(NAME strix-model-constants-tests COMMAND strix-model-constants-tests)
    set_tests_properties(strix-model-constants-tests PROPERTIES LABELS "cpu" TIMEOUT 30)
endif()
include(cmake/StrixSupport.cmake)
include(cmake/StrixVulkan.cmake)
include(cmake/StrixMetal.cmake)

add_library(strix_qwen_model_description STATIC
    src/qwen_model_description.cpp src/model_contract.cpp src/gguf_reader.cpp
    src/model_graph/qwen_adapter.cpp src/execution_plan.cpp)
target_link_libraries(strix_qwen_model_description PUBLIC strix_model_description strix_model_graph)
strix_target_warnings(strix_qwen_model_description)
add_executable(strix-model-describe apps/strix-model-describe.cpp)
target_link_libraries(strix-model-describe PRIVATE strix_qwen_model_description)
strix_target_warnings(strix-model-describe)

add_executable(strix-inspect-gguf apps/inspect_gguf_main.cpp)
target_link_libraries(strix-inspect-gguf PRIVATE strix_core)
strix_target_warnings(strix-inspect-gguf)
# Split-pack header check against the shared tensor naming map (inspect a partial download / verify a complete one).
if(NOT TARGET strix-gguf-pack-check)
    add_executable(strix-gguf-pack-check apps/gguf_pack_check_main.cpp)
    target_link_libraries(strix-gguf-pack-check PRIVATE strix-weight)
    strix_target_warnings(strix-gguf-pack-check)
endif()

# Hy4-flash host reference (CPU oracle): the declared arithmetic of src/hy4/ref/hy4_host_ref.hpp over the pack's tensors.
# It calls the shared host references (hyper-connection, group indexer, RoPE, SwiGLU, route) where they compute its bits.
add_library(strix_hy4_ref STATIC src/hy4/ref/hy4_host_ref.cpp src/attn/group_indexer_ref.cpp)
target_include_directories(strix_hy4_ref PUBLIC include src)
target_compile_features(strix_hy4_ref PUBLIC cxx_std_20)
target_link_libraries(strix_hy4_ref PUBLIC strix_weight_archive strix_hc strix_hy4_engram Threads::Threads)
if(NOT MSVC)
    target_compile_options(strix_hy4_ref PRIVATE -fno-fast-math -ffp-contract=off)
endif()
strix_target_warnings(strix_hy4_ref)
# The oracle on the real pack: prompt ids + greedy ids, top-K and logits digests per position (device identity receipts).
add_executable(strix-hy4-ref apps/hy4_ref_main.cpp)
target_link_libraries(strix-hy4-ref PRIVATE strix_hy4_ref)
include(cmake/StrixIntelHy4.cmake)
strix_target_warnings(strix-hy4-ref)
# DFly2 drafter tools on the oracle: target-state taps of a greedy run (the drafter's KV-share / hidden inputs).
add_library(strix_hy4_dfly_ref STATIC src/hy4/ref/hy4_dfly_ref.cpp)
target_link_libraries(strix_hy4_dfly_ref PUBLIC strix_hy4_ref)
strix_target_warnings(strix_hy4_dfly_ref)
add_executable(strix-hy4-dfly apps/hy4_dfly_main.cpp)
target_link_libraries(strix-hy4-dfly PRIVATE strix_hy4_dfly_ref)
strix_target_warnings(strix-hy4-dfly)

# Host-only serving front end (no GPU): JSON, tokenizer, chat template, HTTP and the API mapping.
add_library(strix_serve STATIC
    src/serve/json_constraint.cpp
    src/serve/constraint_selection.cpp
    src/serve/bpe_tokenizer.cpp
    src/serve/prompt_stream.cpp
    src/serve/token_memo.cpp
    src/serve/chat_template.cpp
    src/serve/deepseek_chat.cpp
    src/serve/glm_chat.cpp
    src/serve/chat_template_kit.cpp
    src/serve/hy4_chat.cpp
    src/serve/inkling2_chat.cpp
    src/serve/serial_generation_cli.cpp
    src/serve/http_server.cpp
    src/serve/engine.cpp
    src/serve/api.cpp
    src/serve/ollama.cpp
    src/serve/deployment.cpp
    src/serve/request_fields.cpp
    src/serve/health.cpp
    src/serve/media.cpp
)
target_include_directories(strix_serve PUBLIC include PRIVATE src/serve src)
target_compile_features(strix_serve PUBLIC cxx_std_20)
target_link_libraries(strix_serve PUBLIC Threads::Threads strix_core strix_serve_json strix_power)
if(WIN32)
    target_link_libraries(strix_serve PUBLIC ws2_32)
endif()
strix_target_warnings(strix_serve)

if(BUILD_TESTING AND TARGET strix_inkling2_mini_checkpoint)
    add_executable(inkling2-mini-tokenizer-test tests/inkling2/mini_tokenizer_test.cpp)
    target_include_directories(inkling2-mini-tokenizer-test PRIVATE src)
    target_link_libraries(inkling2-mini-tokenizer-test PRIVATE strix_serve)
    strix_target_warnings(inkling2-mini-tokenizer-test)
    if(MSVC)
        target_compile_options(inkling2-mini-tokenizer-test PRIVATE /utf-8)
    endif()
    set(STRIX_INKLING2_TOKENIZER_JSON "" CACHE FILEPATH "Optional official Inkling 2 Mini tokenizer.json for fixture checks")
    add_test(NAME inkling2-mini-tokenizer COMMAND inkling2-mini-tokenizer-test ${STRIX_INKLING2_TOKENIZER_JSON})
    add_executable(inkling2-mini-chat-test tests/inkling2/mini_chat_test.cpp)
    target_link_libraries(inkling2-mini-chat-test PRIVATE strix_serve)
    strix_target_warnings(inkling2-mini-chat-test)
    add_test(NAME inkling2-mini-chat COMMAND inkling2-mini-chat-test)
    set_tests_properties(inkling2-mini-tokenizer inkling2-mini-chat PROPERTIES LABELS "cpu;inkling2")
endif()
strix_session_add_runtime()
include(cmake/StrixQwenServe.cmake)

# Example family for docs/adding-an-engine.md: served by the shared shell with the CPU reference backend.
# GLM request ownership on the shared session runtime (host-only code: one closure for the server and its fixtures)
add_library(strix_glm_session_owner STATIC src/glm/serve/glm_session_owner.cpp)
target_include_directories(strix_glm_session_owner PUBLIC src)
target_link_libraries(strix_glm_session_owner PUBLIC strix_session_owner)
strix_target_warnings(strix_glm_session_owner)
# strix-skeleton-serve is defined with strix-serve in cmake/StrixSession.cmake (every build mode).
if(BUILD_TESTING)
    add_executable(skeleton_engine_tests tests/skeleton_engine_tests.cpp)
    target_link_libraries(skeleton_engine_tests PRIVATE strix_session_shell)
    strix_target_warnings(skeleton_engine_tests)
    add_test(NAME skeleton_engine COMMAND skeleton_engine_tests "${CMAKE_CURRENT_SOURCE_DIR}/examples/skeleton_engine")
    set_tests_properties(skeleton_engine PROPERTIES LABELS "cpu;session" TIMEOUT 30)
endif()

add_executable(strix-tokenize apps/tokenize_main.cpp)
target_link_libraries(strix-tokenize PRIVATE strix_serve)
strix_target_warnings(strix-tokenize)

add_executable(strix-mtp-manifest-audit apps/mtp_manifest_audit_main.cpp)
target_link_libraries(strix-mtp-manifest-audit PRIVATE strix_core)
strix_target_warnings(strix-mtp-manifest-audit)

add_library(strix_assets STATIC src/asset_manifest.cpp src/serve/json.cpp)
target_link_libraries(strix_assets PUBLIC strix_core)
strix_target_warnings(strix_assets)
add_executable(strix-model-manifest apps/model_manifest_main.cpp)
add_executable(strix-gsq-content-identity lab/gsq/content_identity.cpp)
target_link_libraries(strix-gsq-content-identity PRIVATE strix_core)
target_link_libraries(strix-model-manifest PRIVATE strix_assets)
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

if(BUILD_TESTING OR STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
    add_library(strix_q4k_a8_contract STATIC src/moe_q4k_a8_contract.cpp src/moe_a8_down_contract.cpp)
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
    add_executable(strix-model-description-tests tests/model_description_tests.cpp)
    add_executable(strix-decode-head-rows-tests tests/decode_head_rows_tests.cpp)
    target_compile_features(strix-decode-head-rows-tests PRIVATE cxx_std_20)
    target_include_directories(strix-decode-head-rows-tests PRIVATE include)
    strix_target_warnings(strix-decode-head-rows-tests)
    add_test(NAME strix-decode-head-rows-tests COMMAND strix-decode-head-rows-tests)
    add_executable(strix-norm-reference-tests tests/norm_reference_tests.cpp)
    target_compile_features(strix-norm-reference-tests PRIVATE cxx_std_20)
    target_include_directories(strix-norm-reference-tests PRIVATE include)
    strix_target_warnings(strix-norm-reference-tests)
    if(MSVC)
        target_compile_options(strix-norm-reference-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-norm-reference-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-norm-reference-tests COMMAND strix-norm-reference-tests)
    # Shared norm numeric traits (src/ops/norm/norm_numeric_traits.hpp): wave-chunk order, square form, correctly rounded rsqrt.
    add_executable(strix-norm-numeric-traits-tests tests/norm_numeric_traits_tests.cpp)
    target_compile_features(strix-norm-numeric-traits-tests PRIVATE cxx_std_20)
    target_include_directories(strix-norm-numeric-traits-tests PRIVATE include)
    strix_target_warnings(strix-norm-numeric-traits-tests)
    if(MSVC)
        target_compile_options(strix-norm-numeric-traits-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-norm-numeric-traits-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-norm-numeric-traits-tests COMMAND strix-norm-numeric-traits-tests)
    # Shared RoPE / gated elementwise ops (src/ops/rope, src/ops/glu): host contract vs the lifted DS41 forms and the BF16-table form.
    add_executable(strix-ops-rope-glu-host-tests tests/ops_rope_glu_host_tests.cpp)
    target_compile_features(strix-ops-rope-glu-host-tests PRIVATE cxx_std_20)
    target_include_directories(strix-ops-rope-glu-host-tests PRIVATE include src)
    strix_target_warnings(strix-ops-rope-glu-host-tests)
    if(MSVC)
        target_compile_options(strix-ops-rope-glu-host-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-ops-rope-glu-host-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-ops-rope-glu-host-tests COMMAND strix-ops-rope-glu-host-tests)
    # Shared F9 recurrence (src/ops/recur): gated delta step policies, KDA per-key-channel gate vs its transcription.
    add_executable(strix-recur-gated-delta-tests tests/recur_gated_delta_tests.cpp)
    target_compile_features(strix-recur-gated-delta-tests PRIVATE cxx_std_20)
    target_include_directories(strix-recur-gated-delta-tests PRIVATE include src)
    strix_target_warnings(strix-recur-gated-delta-tests)
    if(MSVC)
        target_compile_options(strix-recur-gated-delta-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-recur-gated-delta-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    if(MSVC)
        target_compile_options(strix-recur-gated-delta-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-recur-gated-delta-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-recur-gated-delta-tests COMMAND strix-recur-gated-delta-tests)
    # Shared F9 gate prologue and F7 coded-history short conv decode form vs the former GDN prologue text.
    add_executable(strix-recur-gdn-prologue-tests tests/recur_gdn_prologue_tests.cpp)
    target_compile_features(strix-recur-gdn-prologue-tests PRIVATE cxx_std_20)
    target_include_directories(strix-recur-gdn-prologue-tests PRIVATE include src)
    strix_target_warnings(strix-recur-gdn-prologue-tests)
    if(MSVC)
        target_compile_options(strix-recur-gdn-prologue-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-recur-gdn-prologue-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    if(MSVC)
        target_compile_options(strix-recur-gdn-prologue-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-recur-gdn-prologue-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-recur-gdn-prologue-tests COMMAND strix-recur-gdn-prologue-tests)
    # Shared F9 gated delta window form: host emulation of its trees and deferred commit vs the serial reference.
    add_executable(strix-recur-gdn-window-tests tests/recur_gdn_window_tests.cpp)
    target_compile_features(strix-recur-gdn-window-tests PRIVATE cxx_std_20)
    target_include_directories(strix-recur-gdn-window-tests PRIVATE include src)
    strix_target_warnings(strix-recur-gdn-window-tests)
    if(MSVC)
        target_compile_options(strix-recur-gdn-window-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-recur-gdn-window-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-recur-gdn-window-tests COMMAND strix-recur-gdn-window-tests)
    add_executable(strix-qwen-model-graph-tests tests/qwen_model_graph_tests.cpp)
    target_link_libraries(strix-qwen-model-graph-tests PRIVATE strix_qwen_model_description)
    strix_target_warnings(strix-qwen-model-graph-tests)
    add_test(NAME strix-qwen-model-graph-tests COMMAND strix-qwen-model-graph-tests
        "${CMAKE_CURRENT_SOURCE_DIR}/models/qwen3.8-flash-next.toml")
    set_tests_properties(strix-qwen-model-graph-tests PROPERTIES LABELS "cpu;model" TIMEOUT 30)
    target_link_libraries(strix-model-description-tests PRIVATE strix_qwen_model_description)
    if(MSVC)
        target_compile_definitions(strix-model-description-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-model-description-tests)
    add_test(NAME strix-model-description-tests COMMAND strix-model-description-tests
        "${CMAKE_CURRENT_SOURCE_DIR}/models/qwen3.8-flash-next.toml")
    set_tests_properties(strix-model-description-tests PROPERTIES LABELS "cpu;model" TIMEOUT 30)

    add_executable(strix-model-registry-tests tests/model_registry_tests.cpp)
    target_link_libraries(strix-model-registry-tests PRIVATE strix_model_registry)
    strix_target_warnings(strix-model-registry-tests)
    add_test(NAME strix-model-registry-tests COMMAND strix-model-registry-tests "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(strix-model-registry-tests PROPERTIES LABELS "cpu;model" TIMEOUT 30)

    add_executable(strix-spec-drafting-tests tests/spec_drafting_tests.cpp)
    target_link_libraries(strix-spec-drafting-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-drafting-tests)
    add_test(NAME strix-spec-drafting-tests COMMAND strix-spec-drafting-tests)
    add_executable(strix-spec-priced-exploration-test tests/spec_priced_exploration_test.cpp)
    target_link_libraries(strix-spec-priced-exploration-test PRIVATE strix_spec)
    strix_target_warnings(strix-spec-priced-exploration-test)
    add_test(NAME strix-spec-priced-exploration-test COMMAND strix-spec-priced-exploration-test)
    add_executable(strix-spec-frozen-bar-test tests/spec_frozen_bar_test.cpp)
    target_link_libraries(strix-spec-frozen-bar-test PRIVATE strix_spec)
    strix_target_warnings(strix-spec-frozen-bar-test)
    add_test(NAME strix-spec-frozen-bar-test COMMAND strix-spec-frozen-bar-test)
    add_executable(strix-spec-block-drafter-tests tests/spec_block_drafter_tests.cpp)
    target_link_libraries(strix-spec-block-drafter-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-block-drafter-tests)
    add_test(NAME strix-spec-block-drafter-tests COMMAND strix-spec-block-drafter-tests)
    add_executable(strix-spec-frozen-costs-test tests/spec_frozen_costs_test.cpp)
    target_link_libraries(strix-spec-frozen-costs-test PRIVATE strix_core strix_spec)
    strix_target_warnings(strix-spec-frozen-costs-test)
    add_test(NAME strix-spec-frozen-costs-test COMMAND strix-spec-frozen-costs-test)

    add_executable(strix-spec-sampled-rows-tests tests/spec_sampled_rows_tests.cpp)
    target_link_libraries(strix-spec-sampled-rows-tests PRIVATE strix_core strix_spec)
    strix_target_warnings(strix-spec-sampled-rows-tests)
    add_test(NAME strix-spec-sampled-rows-tests COMMAND strix-spec-sampled-rows-tests)
    add_executable(strix-spec-round-sink-tests tests/spec_round_sink_tests.cpp)
    target_link_libraries(strix-spec-round-sink-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-round-sink-tests)
    add_test(NAME strix-spec-round-sink-tests COMMAND strix-spec-round-sink-tests)
    add_executable(strix-spec-commit-tail-tests tests/spec_commit_tail_tests.cpp)
    target_link_libraries(strix-spec-commit-tail-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-commit-tail-tests)
    add_test(NAME strix-spec-commit-tail-tests COMMAND strix-spec-commit-tail-tests)
    add_executable(strix-spec-tree-window-tests tests/spec_tree_window_tests.cpp)
    target_link_libraries(strix-spec-tree-window-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-tree-window-tests)
    # Host graph oracle: FMA contraction can differ between the serial call and the tree loop on x86,
    # changing one rounded bit before greedy_of hashes it. The production GPU path is tested separately.
    if(MSVC)
        target_compile_options(strix-spec-tree-window-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-spec-tree-window-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-spec-tree-window-tests COMMAND strix-spec-tree-window-tests)
    add_executable(strix-spec-draft-rows-tests tests/spec_draft_rows_tests.cpp)
    target_link_libraries(strix-spec-draft-rows-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-draft-rows-tests)
    add_test(NAME strix-spec-draft-rows-tests COMMAND strix-spec-draft-rows-tests)
    add_executable(strix-qwen35-iu4-draft-head-tests tests/qwen35_iu4_draft_head_tests.cpp)
    target_link_libraries(strix-qwen35-iu4-draft-head-tests PRIVATE strix_spec)
    strix_target_warnings(strix-qwen35-iu4-draft-head-tests)
    add_test(NAME strix-qwen35-iu4-draft-head-tests COMMAND strix-qwen35-iu4-draft-head-tests)
    add_executable(strix-hip-dense-spec-costs-tests tests/hip_dense_spec_costs_tests.cpp)
    target_link_libraries(strix-hip-dense-spec-costs-tests PRIVATE strix_spec)
    strix_target_warnings(strix-hip-dense-spec-costs-tests)
    add_test(NAME strix-hip-dense-spec-costs-tests COMMAND strix-hip-dense-spec-costs-tests)
    add_executable(strix-hip-dense-spec-resolver-tests tests/hip_dense_spec_resolver_tests.cpp)
    target_link_libraries(strix-hip-dense-spec-resolver-tests PRIVATE strix_spec)
    target_compile_definitions(strix-hip-dense-spec-resolver-tests PRIVATE
        STRIX_SPEC_RESOLVER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    strix_target_warnings(strix-hip-dense-spec-resolver-tests)
    add_test(NAME strix-hip-dense-spec-resolver-tests COMMAND strix-hip-dense-spec-resolver-tests)
    add_executable(strix-core-dense-row-sweeps-tests tests/core_dense_row_sweeps_tests.cpp)
    target_include_directories(strix-core-dense-row-sweeps-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(strix-core-dense-row-sweeps-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-core-dense-row-sweeps-tests)
    add_test(NAME strix-core-dense-row-sweeps-tests COMMAND strix-core-dense-row-sweeps-tests)
    add_executable(strix-core-window-state-buffers-tests tests/core_window_state_buffers_tests.cpp)
    target_include_directories(strix-core-window-state-buffers-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(strix-core-window-state-buffers-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-core-window-state-buffers-tests)
    add_test(NAME strix-core-window-state-buffers-tests COMMAND strix-core-window-state-buffers-tests)
    add_executable(strix-core-null-perturb-tests tests/core_null_perturb_tests.cpp)
    target_include_directories(strix-core-null-perturb-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(strix-core-null-perturb-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-core-null-perturb-tests)
    add_test(NAME strix-core-null-perturb-tests COMMAND strix-core-null-perturb-tests)
    add_executable(strix-core-iu4-x-pairs-tests tests/core_iu4_x_pairs_tests.cpp)
    target_include_directories(strix-core-iu4-x-pairs-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(strix-core-iu4-x-pairs-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-core-iu4-x-pairs-tests)
    add_test(NAME strix-core-iu4-x-pairs-tests COMMAND strix-core-iu4-x-pairs-tests)
    add_executable(strix-decode-projection-group-tests tests/decode_projection_group_tests.cpp)
    target_include_directories(strix-decode-projection-group-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-decode-projection-group-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-decode-projection-group-tests)
    add_test(NAME strix-decode-projection-group-tests COMMAND strix-decode-projection-group-tests)
    add_executable(strix-moe-slot-merge-tests tests/moe_slot_merge_tests.cpp)
    target_include_directories(strix-moe-slot-merge-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-moe-slot-merge-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-moe-slot-merge-tests)
    add_test(NAME strix-moe-slot-merge-tests COMMAND strix-moe-slot-merge-tests)
    add_executable(strix-decode-weight-sweep-tests tests/decode_weight_sweep_tests.cpp)
    target_include_directories(strix-decode-weight-sweep-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-decode-weight-sweep-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-decode-weight-sweep-tests)
    add_test(NAME strix-decode-weight-sweep-tests COMMAND strix-decode-weight-sweep-tests)
    add_executable(strix-core-dense-iu4-rows-tests tests/core_dense_iu4_rows_tests.cpp)
    target_include_directories(strix-core-dense-iu4-rows-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(strix-core-dense-iu4-rows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-core-dense-iu4-rows-tests)
    add_test(NAME strix-core-dense-iu4-rows-tests COMMAND strix-core-dense-iu4-rows-tests)
    add_executable(strix-q8-0-rule-tests tests/q8_0_rule_tests.cpp)
    target_include_directories(strix-q8-0-rule-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-q8-0-rule-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-q8-0-rule-tests)
    add_test(NAME strix-q8-0-rule-tests COMMAND strix-q8-0-rule-tests)
    add_executable(strix-pair-bf16-wire-tests tests/pair_bf16_wire_tests.cpp)
    target_include_directories(strix-pair-bf16-wire-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-pair-bf16-wire-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-bf16-wire-tests)
    add_test(NAME strix-pair-bf16-wire-tests COMMAND strix-pair-bf16-wire-tests)
    add_executable(strix-spec-mtp-chain-tests tests/spec_mtp_chain_tests.cpp)
    target_link_libraries(strix-spec-mtp-chain-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-mtp-chain-tests)
    add_test(NAME strix-spec-mtp-chain-tests COMMAND strix-spec-mtp-chain-tests)
    add_executable(strix-owned-layer-ordinals-tests tests/owned_layer_ordinals_tests.cpp)
    target_include_directories(strix-owned-layer-ordinals-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-owned-layer-ordinals-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-owned-layer-ordinals-tests)
    add_test(NAME strix-owned-layer-ordinals-tests COMMAND strix-owned-layer-ordinals-tests)
    add_executable(strix-spec-pair-round-tests tests/spec_pair_round_tests.cpp)
    target_link_libraries(strix-spec-pair-round-tests PRIVATE strix_spec Threads::Threads)
    strix_target_warnings(strix-spec-pair-round-tests)
    add_test(NAME strix-spec-pair-round-tests COMMAND strix-spec-pair-round-tests)
    add_executable(strix-spec-block-parallel-tests tests/spec_block_parallel_tests.cpp)
    add_executable(strix-block-candidate-tests tests/block_candidate_tests.cpp)
    target_include_directories(strix-block-candidate-tests PRIVATE src)
    target_link_libraries(strix-block-candidate-tests PRIVATE strix_block_candidate)
    strix_target_warnings(strix-block-candidate-tests)
    add_test(NAME strix-block-candidate-tests COMMAND strix-block-candidate-tests)
    target_link_libraries(strix-spec-block-parallel-tests PRIVATE strix_spec)
    strix_target_warnings(strix-spec-block-parallel-tests)
    add_test(NAME strix-spec-block-parallel-tests COMMAND strix-spec-block-parallel-tests)
    # Host execution of the production plan/lifecycle blocks, with GPU boundaries stubbed.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_subdirectory(tests/dspark_host)
    endif()
    add_executable(strix-batch-nested-recipe-tests tests/batch_nested_recipe_tests.cpp)
    target_include_directories(strix-batch-nested-recipe-tests PRIVATE include)
    target_compile_features(strix-batch-nested-recipe-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-batch-nested-recipe-tests)
    add_test(NAME strix-batch-nested-recipe-tests COMMAND strix-batch-nested-recipe-tests)
    add_executable(strix-block-summary-state-tests tests/block_summary_state_tests.cpp)
    target_include_directories(strix-block-summary-state-tests PRIVATE include)
    target_compile_features(strix-block-summary-state-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-block-summary-state-tests)
    add_test(NAME strix-block-summary-state-tests COMMAND strix-block-summary-state-tests)
    add_executable(strix-batch-service-handoff-tests tests/batch_service_handoff_tests.cpp)
    target_include_directories(strix-batch-service-handoff-tests PRIVATE include)
    target_link_libraries(strix-batch-service-handoff-tests PRIVATE Threads::Threads)
    target_compile_features(strix-batch-service-handoff-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-batch-service-handoff-tests)
    add_test(NAME strix-batch-service-handoff-tests COMMAND strix-batch-service-handoff-tests)
    # Host lane and byte budget: the one host pool (critical / background) and the background byte governor.
    add_executable(strix-host-lane-tests tests/host_lane_tests.cpp)
    target_include_directories(strix-host-lane-tests PRIVATE src)
    target_link_libraries(strix-host-lane-tests PRIVATE Threads::Threads)
    target_compile_features(strix-host-lane-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-host-lane-tests)
    add_test(NAME strix-host-lane-tests COMMAND strix-host-lane-tests)
    set_tests_properties(strix-host-lane-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    # Free host memory before a device load: the populate plan (pure policy) and the no-op entry.
    add_executable(strix-host-free-memory-tests tests/host_free_memory_tests.cpp)
    target_include_directories(strix-host-free-memory-tests PRIVATE src include)
    target_link_libraries(strix-host-free-memory-tests PRIVATE Threads::Threads)
    target_compile_features(strix-host-free-memory-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-host-free-memory-tests)
    add_test(NAME strix-host-free-memory-tests COMMAND strix-host-free-memory-tests)
    set_tests_properties(strix-host-free-memory-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    # Shared host memory readings, floor admission, recovery wait and pressure governor (strix/host_memory.hpp).
    add_executable(strix-host-memory-tests tests/host_memory_tests.cpp)
    target_include_directories(strix-host-memory-tests PRIVATE include)
    target_compile_features(strix-host-memory-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-host-memory-tests)
    add_test(NAME strix-host-memory-tests COMMAND strix-host-memory-tests)
    set_tests_properties(strix-host-memory-tests PROPERTIES LABELS "cpu" TIMEOUT 60)
    # Shared weight-load queue: overlapped reads, per-ticket staging inside a resident budget, one publication wait.
    add_executable(strix-weight-load-queue-tests tests/weight_load_queue_tests.cpp)
    target_include_directories(strix-weight-load-queue-tests PRIVATE src include)
    target_link_libraries(strix-weight-load-queue-tests PRIVATE Threads::Threads strix-weight)
    target_compile_features(strix-weight-load-queue-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-weight-load-queue-tests)
    add_test(NAME strix-weight-load-queue-tests COMMAND strix-weight-load-queue-tests)
    set_tests_properties(strix-weight-load-queue-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    # Checkpoint publication and reads under the process-wide byte budget (the test sets its own small cap).
    add_executable(strix-governed-checkpoint-io-tests tests/governed_checkpoint_io_tests.cpp)
    target_include_directories(strix-governed-checkpoint-io-tests PRIVATE src)
    target_link_libraries(strix-governed-checkpoint-io-tests PRIVATE strix_state_cache Threads::Threads)
    target_compile_features(strix-governed-checkpoint-io-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-governed-checkpoint-io-tests)
    add_test(NAME strix-governed-checkpoint-io-tests COMMAND strix-governed-checkpoint-io-tests)
    set_tests_properties(strix-governed-checkpoint-io-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    add_executable(strix-rotary-descriptor-tests tests/rotary_descriptor_tests.cpp)
    target_link_libraries(strix-rotary-descriptor-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-rotary-descriptor-tests)
    add_test(NAME strix-rotary-descriptor-tests COMMAND strix-rotary-descriptor-tests)
    add_executable(strix-qwen-yarn-table-tests tests/qwen_yarn_table_tests.cpp)
    target_link_libraries(strix-qwen-yarn-table-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-yarn-table-tests)
    add_test(NAME strix-qwen-yarn-table-tests COMMAND strix-qwen-yarn-table-tests)
    add_executable(strix-qwen-yarn-memory-tests tests/qwen_yarn_memory_tests.cpp)
    target_compile_features(strix-qwen-yarn-memory-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-qwen-yarn-memory-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-qwen-yarn-memory-tests)
    add_test(NAME strix-qwen-yarn-memory-tests COMMAND strix-qwen-yarn-memory-tests)
    set_tests_properties(strix-qwen-yarn-memory-tests PROPERTIES LABELS "cpu" TIMEOUT 10)
endif()

if(BUILD_TESTING)
    add_executable(strix-serve-health-tests tests/serve_health_test.cpp)
    add_executable(strix-serve-deployment-tests tests/serve_deployment_tests.cpp)
    target_link_libraries(strix-serve-deployment-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-deployment-tests)
    add_test(NAME strix-serve-deployment-tests COMMAND strix-serve-deployment-tests)
    set_tests_properties(strix-serve-deployment-tests PROPERTIES TIMEOUT 60)
    target_link_libraries(strix-serve-health-tests PRIVATE strix_serve strix_assets)
    target_include_directories(strix-serve-health-tests PRIVATE apps)
    strix_target_warnings(strix-serve-health-tests)
    add_test(NAME strix-serve-health-tests COMMAND strix-serve-health-tests)
    set_tests_properties(strix-serve-health-tests PROPERTIES TIMEOUT 60)
    add_executable(strix-serve-frontend-tests tests/serve_frontend_tests.cpp)
    target_link_libraries(strix-serve-frontend-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-frontend-tests)
    add_test(NAME strix-serve-frontend-tests COMMAND strix-serve-frontend-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/chat_template_cases.json)
    # Incremental encoding: ids equal to the one-shot encoding at every slice end, for every pre-tokenizer, and the
    # prompt stream that runs it on the host lane.
    add_executable(strix-incremental-tokenizer-tests tests/incremental_tokenizer_tests.cpp)
    target_link_libraries(strix-incremental-tokenizer-tests PRIVATE strix_serve)
    target_include_directories(strix-incremental-tokenizer-tests PRIVATE src)
    strix_target_warnings(strix-incremental-tokenizer-tests)
    add_test(NAME strix-incremental-tokenizer-tests COMMAND strix-incremental-tokenizer-tests)
    set_tests_properties(strix-incremental-tokenizer-tests PROPERTIES LABELS "cpu" TIMEOUT 300)
    # Token memo: memoised prompt ids equal a full encode for random chats, edits, shared prefixes and split points.
    add_executable(strix-token-memo-tests tests/token_memo_tests.cpp)
    target_link_libraries(strix-token-memo-tests PRIVATE strix_serve)
    strix_target_warnings(strix-token-memo-tests)
    add_test(NAME strix-token-memo-tests COMMAND strix-token-memo-tests)
    set_tests_properties(strix-token-memo-tests PROPERTIES LABELS "cpu" TIMEOUT 300)
    # Prompt-state mark plan: previous CUDA Qwen capture points without a root, root mark properties, sibling replay.
    add_executable(strix-session-mark-plan-tests tests/session_mark_plan_tests.cpp)
    target_include_directories(strix-session-mark-plan-tests PRIVATE include)
    strix_target_warnings(strix-session-mark-plan-tests)
    add_test(NAME strix-session-mark-plan-tests COMMAND strix-session-mark-plan-tests)
    set_tests_properties(strix-session-mark-plan-tests PROPERTIES LABELS "cpu" TIMEOUT 300)
    # Pending verify windows: a kept prefix commits by publication (no checkpoint, no replay), equal to serial append.
    add_executable(strix-session-pending-window-tests tests/session_pending_window_tests.cpp)
    target_include_directories(strix-session-pending-window-tests PRIVATE include)
    target_compile_features(strix-session-pending-window-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-session-pending-window-tests)
    add_test(NAME strix-session-pending-window-tests COMMAND strix-session-pending-window-tests)
    set_tests_properties(strix-session-pending-window-tests PROPERTIES LABELS "cpu" TIMEOUT 300)
    add_executable(strix-qwen-prefix-root-tests tests/qwen_prefix_root_tests.cpp)
    target_include_directories(strix-qwen-prefix-root-tests PRIVATE include apps)
    target_compile_features(strix-qwen-prefix-root-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen-prefix-root-tests)
    add_test(NAME strix-qwen-prefix-root-tests COMMAND strix-qwen-prefix-root-tests)
    set_tests_properties(strix-qwen-prefix-root-tests PROPERTIES LABELS "cpu" TIMEOUT 300)
    add_executable(strix-glm-tokenizer-tests tests/glm_tokenizer_tests.cpp)
    target_link_libraries(strix-glm-tokenizer-tests PRIVATE strix_serve)
    strix_target_warnings(strix-glm-tokenizer-tests)
    set(STRIX_GLM_TOKENIZER_JSON "" CACHE FILEPATH "GLM-4/5 tokenizer.json (optional; enables the HF fixture checks)")
    add_test(NAME strix-glm-tokenizer-tests COMMAND strix-glm-tokenizer-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/glm_tokenizer_fixtures.json ${STRIX_GLM_TOKENIZER_JSON})
    add_executable(strix-pre-split-rule-tests tests/pre_split_rule_tests.cpp)
    target_link_libraries(strix-pre-split-rule-tests PRIVATE strix_serve)
    target_include_directories(strix-pre-split-rule-tests PRIVATE tests/golden)
    strix_target_warnings(strix-pre-split-rule-tests)
    add_test(NAME strix-pre-split-rule-tests COMMAND strix-pre-split-rule-tests)
    add_executable(strix-glm-chat-tests tests/glm_chat_tests.cpp)
    target_link_libraries(strix-glm-chat-tests PRIVATE strix_serve)
    strix_target_warnings(strix-glm-chat-tests)
    add_test(NAME strix-glm-chat-tests COMMAND strix-glm-chat-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/glm_chat_cases.json)
    # Hy4: the shared BPE with the hy_v4 pre name (the DeepSeek split rule); optional GGUF = vocabulary/id checks.
    add_executable(strix-hy4-tokenizer-tests tests/hy4_tokenizer_tests.cpp)
    target_link_libraries(strix-hy4-tokenizer-tests PRIVATE strix_serve)
    strix_target_warnings(strix-hy4-tokenizer-tests)
    set(STRIX_HY4_GGUF "" CACHE FILEPATH "Hy4 GGUF (first part; optional, only its header is read)")
    add_test(NAME strix-hy4-tokenizer-tests COMMAND strix-hy4-tokenizer-tests ${STRIX_HY4_GGUF})
    add_executable(strix-hy4-chat-tests tests/hy4_chat_tests.cpp)
    target_link_libraries(strix-hy4-chat-tests PRIVATE strix_serve)
    strix_target_warnings(strix-hy4-chat-tests)
    add_test(NAME strix-hy4-chat-tests COMMAND strix-hy4-chat-tests)
    # Hy4 host reference (CPU oracle): synthetic tiny model end to end + contract invariants; optional GGUF = metadata binding.
    add_executable(strix-hy4-host-ref-tests tests/hy4_host_ref_tests.cpp)
    target_link_libraries(strix-hy4-host-ref-tests PRIVATE strix_hy4_ref)
    strix_target_warnings(strix-hy4-host-ref-tests)
    add_test(NAME strix-hy4-host-ref-tests COMMAND strix-hy4-host-ref-tests ${STRIX_HY4_GGUF})
    # DFly2 drafter host reference: synthetic tiny drafter + target view (causal prefix, KV share, slot-0 hidden, switches).
    add_executable(strix-hy4-dfly-ref-tests tests/hy4_dfly_ref_tests.cpp)
    target_link_libraries(strix-hy4-dfly-ref-tests PRIVATE strix_hy4_dfly_ref)
    strix_target_warnings(strix-hy4-dfly-ref-tests)
    add_test(NAME strix-hy4-dfly-ref-tests COMMAND strix-hy4-dfly-ref-tests)
    # Hy4 HIP binding plan (stage map: every stage a shared entry; cache plan on the shared cache-row layouts).
    add_executable(strix-hy4-hip-plan-tests tests/hy4_hip_plan_tests.cpp)
    target_link_libraries(strix-hy4-hip-plan-tests PRIVATE strix_hy4_ref)
    strix_target_warnings(strix-hy4-hip-plan-tests)
    add_test(NAME strix-hy4-hip-plan-tests COMMAND strix-hy4-hip-plan-tests ${STRIX_HY4_GGUF})
    # Hy4 verify rows (host): ring rewind replay for R <= 8, position words, speculative decode ids == plain greedy.
    add_executable(strix-hy4-rows-tests tests/hy4_rows_tests.cpp src/attn/group_indexer_ref.cpp)
    target_link_libraries(strix-hy4-rows-tests PRIVATE strix_hy4_ref strix_spec)
    strix_target_warnings(strix-hy4-rows-tests)
    add_test(NAME strix-hy4-rows-tests COMMAND strix-hy4-rows-tests ${STRIX_HY4_GGUF})
    # Rows tile of the head-family dense kernel: tile map and tree map (bits of the single-row reduction).
    add_executable(strix-dense-rows-tile-tests tests/dense_rows_tile_tests.cpp)
    target_include_directories(strix-dense-rows-tile-tests PRIVATE include)
    target_compile_features(strix-dense-rows-tile-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-dense-rows-tile-tests)
    add_test(NAME strix-dense-rows-tile-tests COMMAND strix-dense-rows-tile-tests)
    # Wave form of the head-family dense kernel: leaf map and tree pairs (bits of the single-row reduction).
    add_executable(strix-dense-rows-wave-tests tests/dense_rows_wave_tests.cpp)
    target_include_directories(strix-dense-rows-wave-tests PRIVATE include)
    target_compile_features(strix-dense-rows-wave-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-dense-rows-wave-tests)
    add_test(NAME strix-dense-rows-wave-tests COMMAND strix-dense-rows-wave-tests)
    add_executable(strix-serve-deepseek-chat-tests tests/serve_deepseek_chat_tests.cpp)
    target_link_libraries(strix-serve-deepseek-chat-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-deepseek-chat-tests)
    add_test(NAME strix-serve-deepseek-chat-tests COMMAND strix-serve-deepseek-chat-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/serve/deepseek_v41_chat_cases.json)
    add_executable(strix-serve-tool-choice-none-tests tests/serve_tool_choice_none_tests.cpp)
    target_link_libraries(strix-serve-tool-choice-none-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-tool-choice-none-tests)
    add_test(NAME strix-serve-tool-choice-none-tests COMMAND strix-serve-tool-choice-none-tests)
    add_executable(strix-serve-api-tests tests/serve_api_tests.cpp)
    target_link_libraries(strix-serve-api-tests PRIVATE strix_serve strix_core)
    strix_target_warnings(strix-serve-api-tests)
    add_test(NAME strix-serve-api-tests COMMAND strix-serve-api-tests)
    add_test(NAME strix-serve-api-sampling-tests COMMAND strix-serve-api-tests --sampling-only)
    add_executable(strix-pair-lockstep-tests tests/pair_lockstep_tests.cpp)
    target_link_libraries(strix-pair-lockstep-tests PRIVATE strix_core)
    strix_target_warnings(strix-pair-lockstep-tests)
    add_test(NAME strix-pair-lockstep-tests COMMAND strix-pair-lockstep-tests)

    add_executable(strix-mtp-selection-transaction-tests tests/mtp_selection_transaction_tests.cpp)
    target_link_libraries(strix-mtp-selection-transaction-tests PRIVATE strix_serve)
    strix_target_warnings(strix-mtp-selection-transaction-tests)
    add_test(NAME strix-mtp-selection-transaction-tests COMMAND strix-mtp-selection-transaction-tests)
    # Its worker/selector gates are bounded; this bounds a wait inside the selection itself.
    set_tests_properties(strix-mtp-selection-transaction-tests PROPERTIES TIMEOUT 900)

    # Thinking budgets on device-greedy windows (argmax replies only): select_greedy_window + ArgmaxConstraintRequests.
    add_executable(strix-greedy-window-selection-tests tests/greedy_window_selection_tests.cpp)
    target_link_libraries(strix-greedy-window-selection-tests PRIVATE strix_serve)
    strix_target_warnings(strix-greedy-window-selection-tests)
    add_test(NAME strix-greedy-window-selection-tests COMMAND strix-greedy-window-selection-tests)

    add_executable(strix-serve-json-constraint-tests tests/serve_json_constraint_tests.cpp)
    target_link_libraries(strix-serve-json-constraint-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-json-constraint-tests)
    add_test(NAME strix-serve-json-constraint-tests COMMAND strix-serve-json-constraint-tests)

    add_executable(strix-serve-media-tests tests/serve_media_tests.cpp)
    target_link_libraries(strix-serve-media-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-media-tests)
    add_test(NAME strix-serve-media-tests COMMAND strix-serve-media-tests)
    add_executable(strix-serve-observe-tests tests/serve_observe_tests.cpp)
    target_link_libraries(strix-serve-observe-tests PRIVATE strix_serve)
    strix_target_warnings(strix-serve-observe-tests)
    add_test(NAME strix-serve-observe-tests COMMAND strix-serve-observe-tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")  # glibc mallinfo2 / malloc_trim for the heap check
        add_executable(strix-serve-observe-cost-tests tests/serve_observe_cost_tests.cpp)
        target_link_libraries(strix-serve-observe-cost-tests PRIVATE strix_serve)
        strix_target_warnings(strix-serve-observe-cost-tests)
        add_test(NAME strix-serve-observe-cost-tests COMMAND strix-serve-observe-cost-tests)
    endif()
    add_executable(strix-teacher-logit-spool-tests tests/teacher_logit_spool_tests.cpp)
    target_compile_features(strix-teacher-logit-spool-tests PRIVATE cxx_std_20)
    target_include_directories(strix-teacher-logit-spool-tests PRIVATE include)
    strix_target_warnings(strix-teacher-logit-spool-tests)
    add_test(NAME strix-teacher-logit-spool-tests COMMAND strix-teacher-logit-spool-tests)
    add_executable(strix-kquant-block-reads-tests tests/kquant_block_reads_tests.cpp)
    target_compile_features(strix-kquant-block-reads-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-kquant-block-reads-tests)
    add_test(NAME strix-kquant-block-reads-tests COMMAND strix-kquant-block-reads-tests)
    add_executable(strix-moe-expert-union-tests tests/moe_expert_union_tests.cpp)
    target_link_libraries(strix-moe-expert-union-tests PRIVATE strix_q4k_a8_contract strix_core)
    strix_target_warnings(strix-moe-expert-union-tests)
    add_test(NAME strix-moe-expert-union-tests COMMAND strix-moe-expert-union-tests)
    add_executable(strix-moe-a8-down-contract-tests tests/moe_a8_down_contract_tests.cpp)
    target_link_libraries(strix-moe-a8-down-contract-tests PRIVATE strix_q4k_a8_contract strix_core)
    strix_target_warnings(strix-moe-a8-down-contract-tests)
    if(MSVC)
        target_compile_options(strix-moe-a8-down-contract-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-a8-down-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-moe-a8-down-contract-tests COMMAND strix-moe-a8-down-contract-tests)
    add_executable(strix-a8-row-pitch-tests tests/a8_row_pitch_tests.cpp)
    target_compile_features(strix-a8-row-pitch-tests PRIVATE cxx_std_20)
    target_include_directories(strix-a8-row-pitch-tests PRIVATE include src/intel/core)
    strix_target_warnings(strix-a8-row-pitch-tests)
    if(MSVC)
        target_compile_options(strix-a8-row-pitch-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-a8-row-pitch-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-a8-row-pitch-tests COMMAND strix-a8-row-pitch-tests)
    add_executable(strix-core-a8g32-tests tests/core_a8g32_tests.cpp)
    target_compile_features(strix-core-a8g32-tests PRIVATE cxx_std_20)
    target_include_directories(strix-core-a8g32-tests PRIVATE include)
    strix_target_warnings(strix-core-a8g32-tests)
    if(MSVC)
        target_compile_options(strix-core-a8g32-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-core-a8g32-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-core-a8g32-tests COMMAND strix-core-a8g32-tests)
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

    add_executable(strix-cohort-step-plan-tests tests/cohort_step_plan_tests.cpp)
    target_link_libraries(strix-cohort-step-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-cohort-step-plan-tests)
    add_test(NAME strix-cohort-step-plan-tests COMMAND strix-cohort-step-plan-tests)

    add_executable(strix-cohort-small-plan-tests tests/cohort_small_plan_tests.cpp)
    target_link_libraries(strix-cohort-small-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-cohort-small-plan-tests)
    add_test(NAME strix-cohort-small-plan-tests COMMAND strix-cohort-small-plan-tests)

    add_executable(strix-http-session-store-tests tests/http_session_store_tests.cpp)
    target_link_libraries(strix-http-session-store-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-http-session-store-tests)
    add_test(NAME strix-http-session-store-tests COMMAND strix-http-session-store-tests)

    add_executable(strix-session-state-cache-tests tests/session_state_cache_tests.cpp)
    target_link_libraries(strix-session-state-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-state-cache-tests)
    add_test(NAME strix-session-state-cache-tests COMMAND strix-session-state-cache-tests)

    add_executable(strix-qwen-tiled-prompt-plan-tests tests/qwen_tiled_prompt_plan_tests.cpp)
    target_link_libraries(strix-qwen-tiled-prompt-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-tiled-prompt-plan-tests)
    add_test(NAME strix-qwen-tiled-prompt-plan-tests COMMAND strix-qwen-tiled-prompt-plan-tests)

    add_executable(strix-prefix-state-cache-tests tests/prefix_state_cache_tests.cpp tests/prefix_dma_tests.cpp)
    target_include_directories(strix-prefix-state-cache-tests PRIVATE tests/prefix_dma_shim)
    target_link_libraries(strix-prefix-state-cache-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-prefix-state-cache-tests)
    add_test(NAME strix-prefix-state-cache-tests COMMAND strix-prefix-state-cache-tests)
    # Prepared lookups (prepare_lookup + find_longest(ticket)) against recorded one-step lookups.
    add_executable(strix-prefix-lookup-ticket-tests tests/prefix_lookup_ticket_tests.cpp)
    target_link_libraries(strix-prefix-lookup-ticket-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-prefix-lookup-ticket-tests)
    add_test(NAME strix-prefix-lookup-ticket-tests COMMAND strix-prefix-lookup-ticket-tests)
    set_tests_properties(strix-prefix-lookup-ticket-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    # Queue protection of prefix states, the governed automatic budget rule and the pending cache rows.
    add_executable(strix-prefix-queue-protection-tests tests/prefix_queue_protection_tests.cpp)
    target_include_directories(strix-prefix-queue-protection-tests PRIVATE apps)
    target_link_libraries(strix-prefix-queue-protection-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-prefix-queue-protection-tests)
    add_test(NAME strix-prefix-queue-protection-tests COMMAND strix-prefix-queue-protection-tests)
    set_tests_properties(strix-prefix-queue-protection-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    # Server start: phase line and overlapped start work (strix/serve/startup_phases.hpp).
    add_executable(strix-serve-startup-phases-tests tests/serve_startup_phases_tests.cpp)
    target_link_libraries(strix-serve-startup-phases-tests PRIVATE strix_core Threads::Threads)
    strix_target_warnings(strix-serve-startup-phases-tests)
    add_test(NAME strix-serve-startup-phases-tests COMMAND strix-serve-startup-phases-tests)
    set_tests_properties(strix-serve-startup-phases-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    # Server start warm-up: plan, synthetic ids and run wrapper (strix/serve/startup_warmup.hpp).
    add_executable(strix-serve-startup-warmup-tests tests/serve_startup_warmup_tests.cpp)
    target_link_libraries(strix-serve-startup-warmup-tests PRIVATE strix_core)
    strix_target_warnings(strix-serve-startup-warmup-tests)
    add_test(NAME strix-serve-startup-warmup-tests COMMAND strix-serve-startup-warmup-tests)
    set_tests_properties(strix-serve-startup-warmup-tests PROPERTIES LABELS "cpu" TIMEOUT 120)
    add_test(NAME strix-prefill-option-cache-write COMMAND strix-prefix-state-cache-tests
        --write-option-checkpoint "${CMAKE_CURRENT_BINARY_DIR}/prefill-option-cache")
    add_test(NAME strix-prefill-option-cache-read COMMAND strix-prefix-state-cache-tests
        --read-option-checkpoint "${CMAKE_CURRENT_BINARY_DIR}/prefill-option-cache")
    set_tests_properties(strix-prefill-option-cache-write PROPERTIES FIXTURES_SETUP prefill_option_cache)
    set_tests_properties(strix-prefill-option-cache-read PROPERTIES FIXTURES_REQUIRED prefill_option_cache)
    add_test(NAME strix-prefill-option-cache-clean COMMAND ${CMAKE_COMMAND} -E rm -rf
        "${CMAKE_CURRENT_BINARY_DIR}/prefill-option-cache")
    set_tests_properties(strix-prefill-option-cache-clean PROPERTIES FIXTURES_CLEANUP prefill_option_cache)

    add_executable(strix-rtx-state-import-tests tests/rtx_state_import_tests.cpp)
    target_include_directories(strix-rtx-state-import-tests PRIVATE apps)
    target_link_libraries(strix-rtx-state-import-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-rtx-state-import-tests)
    add_test(NAME strix-rtx-state-import-tests COMMAND strix-rtx-state-import-tests)
    add_executable(strix-bp-codec-tests tests/bp_codec_tests.cpp)
    target_link_libraries(strix-bp-codec-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-bp-codec-tests)
    add_test(NAME strix-bp-codec-tests COMMAND strix-bp-codec-tests)
    # shared state hand-off transport + its policy resolver (loopback TCP sessions, cost model, effective-config lines)
    add_executable(strix-handoff-tests tests/handoff_tests.cpp src/handoff.cpp src/bp_codec.cpp)
    target_compile_definitions(strix-handoff-tests PRIVATE STRIX_HANDOFF_TESTING=1)
    target_include_directories(strix-handoff-tests PRIVATE include)
    target_link_libraries(strix-handoff-tests PRIVATE Threads::Threads)
    target_compile_features(strix-handoff-tests PRIVATE cxx_std_20)
    if(WIN32)
        target_link_libraries(strix-handoff-tests PRIVATE ws2_32)
    endif()
    strix_target_warnings(strix-handoff-tests)
    add_test(NAME strix-handoff-tests COMMAND strix-handoff-tests)

    add_executable(strix-known-prompt-chunks-tests tests/known_prompt_chunks_tests.cpp)
    target_link_libraries(strix-known-prompt-chunks-tests PRIVATE strix_core)
    strix_target_warnings(strix-known-prompt-chunks-tests)
    add_test(NAME strix-known-prompt-chunks-tests COMMAND strix-known-prompt-chunks-tests)
    add_executable(strix-prefill-row-liveness-tests tests/prefill_row_liveness_tests.cpp)
    target_include_directories(strix-prefill-row-liveness-tests PRIVATE include)
    target_compile_features(strix-prefill-row-liveness-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-prefill-row-liveness-tests)
    add_test(NAME strix-prefill-row-liveness-tests COMMAND strix-prefill-row-liveness-tests)
    add_executable(strix-serial-prompt-plan-tests tests/serial_prompt_plan_tests.cpp)
    target_link_libraries(strix-serial-prompt-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-serial-prompt-plan-tests)
    foreach(schedule IN ITEMS 2 4 16 20)
        add_test(NAME strix-serial-prompt-plan-sched-${schedule} COMMAND strix-serial-prompt-plan-tests ${schedule})
    endforeach()
    add_executable(strix-serve-config-tests tests/serve_config_tests.cpp)
    target_link_libraries(strix-serve-config-tests PRIVATE strix_core)
    strix_target_warnings(strix-serve-config-tests)
    add_test(NAME strix-serve-config-tests COMMAND strix-serve-config-tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(strix-serve-config-race-tests tests/serve_config_race_tests.cpp)
        target_link_libraries(strix-serve-config-race-tests PRIVATE strix_serve_config)
        target_link_options(strix-serve-config-race-tests PRIVATE -Wl,--wrap=stat -Wl,--wrap=fstat)
        strix_target_warnings(strix-serve-config-race-tests)
        add_test(NAME strix-serve-config-race-tests COMMAND strix-serve-config-race-tests)
        set_tests_properties(strix-serve-config-race-tests PROPERTIES TIMEOUT 10)
    endif()
    add_executable(strix-http-slot-capacity-tests tests/http_slot_capacity_tests.cpp)
    target_link_libraries(strix-http-slot-capacity-tests PRIVATE strix_core)
    strix_target_warnings(strix-http-slot-capacity-tests)
    add_test(NAME strix-http-slot-capacity-tests COMMAND strix-http-slot-capacity-tests)
    add_executable(strix-session-reservation-tests tests/session_reservation_tests.cpp)
    target_include_directories(strix-session-reservation-tests PRIVATE include)
    target_compile_features(strix-session-reservation-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-session-reservation-tests)
    add_test(NAME strix-session-reservation-tests COMMAND strix-session-reservation-tests)
    add_executable(strix-qwen-mtp-defer-tests tests/qwen_mtp_defer_tests.cpp)
    target_include_directories(strix-qwen-mtp-defer-tests PRIVATE include)
    target_compile_features(strix-qwen-mtp-defer-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen-mtp-defer-tests)
    add_test(NAME strix-qwen-mtp-defer-tests COMMAND strix-qwen-mtp-defer-tests)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-qwen-serve-bench-tests
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_SOURCE_DIR}/tests/qwen_serve_bench_tests.py")
    endif()
    add_executable(strix-qwen-yarn-schedule-tests tests/qwen_yarn_schedule_tests.cpp)
    target_compile_definitions(strix-qwen-yarn-schedule-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    target_link_libraries(strix-qwen-yarn-schedule-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-yarn-schedule-tests)
    foreach(schedule 0 8 13 16 12)
        add_test(NAME strix-qwen-yarn-schedule-${schedule} COMMAND strix-qwen-yarn-schedule-tests)
        set_tests_properties(strix-qwen-yarn-schedule-${schedule} PROPERTIES ENVIRONMENT "STRIX_QPF2K_SCHED=${schedule}")
    endforeach()

    add_executable(strix-hip-prefill-pool-host-tests tests/hip_prefill_pool_host_tests.cpp)
    target_include_directories(strix-hip-prefill-pool-host-tests PRIVATE tests/stubs/hip_prefill_pool src include)
    target_link_libraries(strix-hip-prefill-pool-host-tests PRIVATE Threads::Threads)
    target_compile_features(strix-hip-prefill-pool-host-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hip-prefill-pool-host-tests)
    add_test(NAME strix-hip-prefill-pool-host-tests COMMAND strix-hip-prefill-pool-host-tests)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # dist::Communicator exchange contract on the host (CPU HIP stub, in-process pair): exchange_host/exchange
        # interchangeable with deferred CRCs, one-way transfer() on every route.
        add_executable(strix-dist-exchange-contract-tests tests/dist_exchange_contract_tests.cpp src/dist_communicator.cpp
            src/serve/json.cpp)
        target_include_directories(strix-dist-exchange-contract-tests PRIVATE tests/stubs/dist_communicator include src)
        target_link_libraries(strix-dist-exchange-contract-tests PRIVATE Threads::Threads)
        target_compile_features(strix-dist-exchange-contract-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-dist-exchange-contract-tests)
        # Keep this host-only transport contract under ASan/UBSan in Linux CI: the stub pair must never
        # hide a bad buffer lifetime, data race-induced use-after-free, or out-of-bounds trailer.
        # These flags apply to this test executable only, never to the engine or kernel targets.
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(strix-dist-exchange-contract-tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(strix-dist-exchange-contract-tests PRIVATE -fsanitize=address,undefined)
        endif()
        add_test(NAME strix-dist-exchange-contract-tests COMMAND strix-dist-exchange-contract-tests)
        set_tests_properties(strix-dist-exchange-contract-tests PROPERTIES TIMEOUT 120)
    endif()
    add_executable(strix-hip-prefill-call-shapes-host-tests tests/hip_prefill_call_shapes_host_tests.cpp)
    target_include_directories(strix-hip-prefill-call-shapes-host-tests PRIVATE tests/stubs/hip_prefill_pool src apps)
    target_link_libraries(strix-hip-prefill-call-shapes-host-tests PRIVATE strix_core)
    strix_target_warnings(strix-hip-prefill-call-shapes-host-tests)
    foreach(schedule IN ITEMS 0 12 28)
        add_test(NAME strix-hip-prefill-call-shapes-${schedule}
            COMMAND strix-hip-prefill-call-shapes-host-tests ${schedule})
    endforeach()
    add_executable(strix-qwen-startup-policy-tests tests/qwen_startup_policy_tests.cpp)
    target_link_libraries(strix-qwen-startup-policy-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-qwen-startup-policy-tests)
    add_test(NAME strix-qwen-startup-policy-tests COMMAND strix-qwen-startup-policy-tests)
    add_executable(strix-qsa-index-score-host-tests tests/qsa_index_score_host_tests.cpp)
    target_include_directories(strix-qsa-index-score-host-tests PRIVATE include)
    target_compile_features(strix-qsa-index-score-host-tests PRIVATE cxx_std_20)
    if(NOT MSVC)
        target_compile_options(strix-qsa-index-score-host-tests PRIVATE -ffp-contract=off)
    endif()
    strix_target_warnings(strix-qsa-index-score-host-tests)
    add_test(NAME strix-qsa-index-score-host-tests COMMAND strix-qsa-index-score-host-tests)
    add_executable(strix-qwen-effective-config-tests tests/qwen_effective_config_tests.cpp)
    add_executable(strix-weight-pack-tests tests/weight_pack_tests.cpp src/gguf_reader.cpp)
    target_include_directories(strix-weight-pack-tests PRIVATE include)
    target_link_libraries(strix-weight-pack-tests PRIVATE strix_serve_json)
    target_compile_features(strix-weight-pack-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-weight-pack-tests)
    add_test(NAME strix-weight-pack-tests COMMAND strix-weight-pack-tests)
    set_tests_properties(strix-weight-pack-tests PROPERTIES LABELS "cpu")
    add_executable(strix-qwen-mtp-v2-shape-tests tests/qwen_mtp_v2_shape_tests.cpp)
    target_include_directories(strix-qwen-mtp-v2-shape-tests PRIVATE include)
    target_compile_features(strix-qwen-mtp-v2-shape-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen-mtp-v2-shape-tests)
    add_test(NAME strix-qwen-mtp-v2-shape-tests COMMAND strix-qwen-mtp-v2-shape-tests)
    set_tests_properties(strix-qwen-mtp-v2-shape-tests PROPERTIES LABELS "cpu")
    add_executable(strix-qwen-hc-dag-tests tests/qwen_hc_dag_tests.cpp)
    target_include_directories(strix-qwen-hc-dag-tests PRIVATE include)
    target_compile_features(strix-qwen-hc-dag-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen-hc-dag-tests)
    add_test(NAME strix-qwen-hc-dag-tests COMMAND strix-qwen-hc-dag-tests)
    set_tests_properties(strix-qwen-hc-dag-tests PROPERTIES LABELS "cpu")
    add_executable(strix-qwen-router-bf16-tests tests/qwen_router_bf16_tests.cpp)
    target_link_libraries(strix-qwen-router-bf16-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-qwen-router-bf16-tests)
    add_test(NAME strix-qwen-router-bf16-tests COMMAND strix-qwen-router-bf16-tests)
    set_tests_properties(strix-qwen-router-bf16-tests PROPERTIES LABELS "cpu")
    add_executable(strix-onfly-i4-tests tests/onfly_i4_tests.cpp)
    target_link_libraries(strix-onfly-i4-tests PRIVATE strix_core strix_serve)
    target_include_directories(strix-onfly-i4-tests PRIVATE src)
    strix_target_warnings(strix-onfly-i4-tests)
    add_test(NAME strix-onfly-i4-tests COMMAND strix-onfly-i4-tests)
    set_tests_properties(strix-onfly-i4-tests PROPERTIES LABELS "cpu")
    add_executable(strix-q8-0-transpose-tests tests/q8_0_transpose_tests.cpp)
    target_link_libraries(strix-q8-0-transpose-tests PRIVATE strix_core)
    target_include_directories(strix-q8-0-transpose-tests PRIVATE src)
    strix_target_warnings(strix-q8-0-transpose-tests)
    add_test(NAME strix-q8-0-transpose-tests COMMAND strix-q8-0-transpose-tests)
    set_tests_properties(strix-q8-0-transpose-tests PROPERTIES LABELS "cpu")
    add_executable(strix-onfly-i4-route-tests tests/onfly_i4_route_tests.cpp)
    target_link_libraries(strix-onfly-i4-route-tests PRIVATE strix_core strix_serve)
    target_include_directories(strix-onfly-i4-route-tests PRIVATE src)
    strix_target_warnings(strix-onfly-i4-route-tests)
    add_test(NAME strix-onfly-i4-route-tests COMMAND strix-onfly-i4-route-tests)
    set_tests_properties(strix-onfly-i4-route-tests PROPERTIES LABELS "cpu" TIMEOUT 30)
    add_executable(strix-qwen-mtp-bundle-tests tests/qwen_mtp_bundle_tests.cpp)
    target_link_libraries(strix-qwen-mtp-bundle-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-mtp-bundle-tests)
    add_test(NAME strix-qwen-mtp-bundle-tests COMMAND strix-qwen-mtp-bundle-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/recipes/qwen-distill-g-yarn.mtp-bundle.json)
    set_tests_properties(strix-qwen-mtp-bundle-tests PROPERTIES LABELS "cpu")
    target_compile_definitions(strix-qwen-effective-config-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1
        STRIX_QWEN_CACHE_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen_cache_public_main_reference.txt")
    target_link_libraries(strix-qwen-effective-config-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-qwen-effective-config-tests)
    if(MSVC)
        # The compiler/resolver fixture has a large main frame; Windows defaults to 1 MiB.
        target_link_options(strix-qwen-effective-config-tests PRIVATE /STACK:16777216)
    endif()
    add_test(NAME strix-qwen-effective-config-tests COMMAND strix-qwen-effective-config-tests)
    add_test(NAME strix-qwen-yarn-owner-inventory-tests COMMAND strix-qwen-effective-config-tests --yarn-owner-inventory)
    add_test(NAME strix-qwen-serve-prefill-admission-tests COMMAND strix-qwen-effective-config-tests --serve-prefill-admission)
    add_executable(strix-compact-stage-images-tests tests/compact_stage_images_tests.cpp)
    target_link_libraries(strix-compact-stage-images-tests PRIVATE strix_core)
    strix_target_warnings(strix-compact-stage-images-tests)
    add_test(NAME strix-compact-stage-images-tests COMMAND strix-compact-stage-images-tests)
    add_executable(strix-qpf2k-defaults-tests tests/qpf2k_defaults_tests.cpp)
    target_include_directories(strix-qpf2k-defaults-tests PRIVATE apps)
    target_link_libraries(strix-qpf2k-defaults-tests PRIVATE strix_core)
    strix_target_warnings(strix-qpf2k-defaults-tests)
    add_executable(strix-qwen-allon-tests tests/qwen_allon_tests.cpp)
    target_link_libraries(strix-qwen-allon-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-allon-tests)
    add_test(NAME strix-qwen-allon-tests COMMAND strix-qwen-allon-tests)
    add_test(NAME strix-qpf2k-defaults-tests COMMAND strix-qpf2k-defaults-tests)
    add_test(NAME strix-qpf2k-readers-gsq-first COMMAND strix-qpf2k-defaults-tests --gsq-first)
    add_test(NAME strix-qpf2k-readers-ordinary-first COMMAND strix-qpf2k-defaults-tests --ordinary-first)
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
    if(UNIX)
        add_executable(strix-startup-markers-tests tests/startup_markers_tests.cpp)
        target_link_libraries(strix-startup-markers-tests PRIVATE strix_core)
        strix_target_warnings(strix-startup-markers-tests)
        add_test(NAME strix-startup-markers-tests COMMAND strix-startup-markers-tests)
    endif()
    add_executable(strix-generation-output-tests tests/generation_output_tests.cpp)
    target_link_libraries(strix-generation-output-tests PRIVATE strix_core)
    strix_target_warnings(strix-generation-output-tests)
    add_test(NAME strix-generation-output-tests COMMAND strix-generation-output-tests)
    add_executable(strix-logprob-partials-tests tests/logprob_partials_tests.cpp)
    target_link_libraries(strix-logprob-partials-tests PRIVATE strix_core)
    strix_target_warnings(strix-logprob-partials-tests)
    add_test(NAME strix-logprob-partials-tests COMMAND strix-logprob-partials-tests)
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
    add_executable(strix-qwen-product-profile-tests tests/qwen_product_profile_tests.cpp)
    target_link_libraries(strix-qwen-product-profile-tests PRIVATE strix_core strix_serve)
    strix_target_warnings(strix-qwen-product-profile-tests)
    add_test(NAME strix-qwen-product-profile-tests COMMAND strix-qwen-product-profile-tests)
    add_executable(strix-qwen-profile-resolution-tests tests/qwen_profile_resolution_tests.cpp)
    target_link_libraries(strix-qwen-profile-resolution-tests PRIVATE strix_core strix_serve_json)
    strix_target_warnings(strix-qwen-profile-resolution-tests)
    add_test(NAME strix-qwen-profile-resolution-tests COMMAND strix-qwen-profile-resolution-tests)
    set_tests_properties(strix-qwen-profile-resolution-tests PROPERTIES LABELS "cpu;profile" TIMEOUT 60)
    # Backend binding sequence of the route tap (read -> tap -> write back only on change), used by the CUDA routers.
    add_executable(strix-moe-route-tap-exchange-tests tests/moe_route_tap_exchange_tests.cpp)
    target_link_libraries(strix-moe-route-tap-exchange-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-route-tap-exchange-tests)
    add_test(NAME strix-moe-route-tap-exchange-tests COMMAND strix-moe-route-tap-exchange-tests)
    set_tests_properties(strix-moe-route-tap-exchange-tests PROPERTIES LABELS "cpu;gate" TIMEOUT 60)
    add_executable(strix-turbo-profile-tests tests/turbo_profile_tests.cpp)
    target_link_libraries(strix-turbo-profile-tests PRIVATE strix_core strix_serve_json)
    strix_target_warnings(strix-turbo-profile-tests)
    add_test(NAME strix-turbo-profile-tests COMMAND strix-turbo-profile-tests)
    set_tests_properties(strix-turbo-profile-tests PROPERTIES LABELS "cpu;profile" TIMEOUT 60)
    add_executable(strix-turbo-bundle-tests tests/turbo_bundle_tests.cpp src/minimax/engine/mm_defaults.cpp)
    target_link_libraries(strix-turbo-bundle-tests PRIVATE strix_core strix_serve_json)
    target_include_directories(strix-turbo-bundle-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    strix_target_warnings(strix-turbo-bundle-tests)
    add_test(NAME strix-turbo-bundle-tests COMMAND strix-turbo-bundle-tests)
    set_tests_properties(strix-turbo-bundle-tests PROPERTIES LABELS "cpu;profile" TIMEOUT 60)
    # One shared check of the tier word over every engine table: a qualified (or provisional) FAST row is in the
    # resolved fast set, fast != exact when one exists, exact enables no bit-changing row.
    add_executable(strix-profile-fast-set-tests tests/profile_fast_set_tests.cpp src/minimax/engine/mm_defaults.cpp src/ds41_rtx_profile.cpp)
    target_link_libraries(strix-profile-fast-set-tests PRIVATE strix_core strix_serve_json)
    target_include_directories(strix-profile-fast-set-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    strix_target_warnings(strix-profile-fast-set-tests)
    add_test(NAME strix-profile-fast-set-tests COMMAND strix-profile-fast-set-tests)
    set_tests_properties(strix-profile-fast-set-tests PROPERTIES LABELS "cpu;profile" TIMEOUT 60)

    # FAST gate v2, engine side: shared MoE route record / replay / one-ulp null tap.
    add_executable(strix-moe-route-file-tests tests/moe_route_file_tests.cpp)
    target_link_libraries(strix-moe-route-file-tests PRIVATE strix_core)
    strix_target_warnings(strix-moe-route-file-tests)
    add_test(NAME strix-moe-route-file-tests COMMAND strix-moe-route-file-tests)
    set_tests_properties(strix-moe-route-file-tests PROPERTIES LABELS "cpu;gate" TIMEOUT 60)
    # Device state image codec (long teacher-forced references carried across processes).
    add_executable(strix-session-state-image-tests tests/session_state_image_tests.cpp)
    target_link_libraries(strix-session-state-image-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-state-image-tests)
    add_test(NAME strix-session-state-image-tests COMMAND strix-session-state-image-tests)
    set_tests_properties(strix-session-state-image-tests PROPERTIES LABELS "cpu;gate" TIMEOUT 60)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-turbo-kit-tests COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_SOURCE_DIR}/tests/turbo_kit_tests.py $<TARGET_FILE:strix-turbo-bundle-tests>)
        set_tests_properties(strix-turbo-kit-tests PROPERTIES LABELS "cpu;profile" TIMEOUT 120)
    endif()
    target_link_libraries(strix-decode-optimization-profile-tests PRIVATE strix_core)
    target_include_directories(strix-decode-optimization-profile-tests PRIVATE src)
    strix_target_warnings(strix-decode-optimization-profile-tests)
    add_test(NAME strix-decode-optimization-profile-tests COMMAND strix-decode-optimization-profile-tests)
    add_executable(strix-decode-optimization-profile-a8-cli-tests tests/decode_optimization_profile_tests.cpp)
    target_compile_definitions(strix-decode-optimization-profile-a8-cli-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    target_link_libraries(strix-decode-optimization-profile-a8-cli-tests PRIVATE strix_core)
    strix_target_warnings(strix-decode-optimization-profile-a8-cli-tests)
    add_test(NAME strix-decode-optimization-profile-a8-cli-tests COMMAND strix-decode-optimization-profile-a8-cli-tests)
    foreach(target IN ITEMS strix-turbo-audit730-host-tests strix-turbo-audit730-a8-host-tests)
        add_executable(${target} tests/audit730_host_tests.cpp)
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_include_directories(${target} PRIVATE include src)
        if(MSVC)
            target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        strix_target_warnings(${target})
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "cpu;profile" TIMEOUT 60)
    endforeach()
    target_compile_definitions(strix-turbo-audit730-a8-host-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
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
    add_executable(strix-moe-q5k-a8-contract-tests tests/moe_q5k_a8_contract_tests.cpp)
    target_link_libraries(strix-moe-q5k-a8-contract-tests PRIVATE strix_q4k_a8_contract strix_core)
    strix_target_warnings(strix-moe-q5k-a8-contract-tests)
    if(MSVC)
        target_compile_options(strix-moe-q5k-a8-contract-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-q5k-a8-contract-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-moe-q5k-a8-contract-tests COMMAND strix-moe-q5k-a8-contract-tests)
    add_executable(strix-hc-parallel-projection-oracle-tests tests/hc_parallel_projection_oracle_tests.cpp)
    target_compile_features(strix-hc-parallel-projection-oracle-tests PRIVATE cxx_std_20)
    target_include_directories(strix-hc-parallel-projection-oracle-tests PRIVATE include)
    strix_target_warnings(strix-hc-parallel-projection-oracle-tests)
    add_test(NAME strix-hc-parallel-projection-oracle-tests COMMAND strix-hc-parallel-projection-oracle-tests)
    add_executable(strix-hc-deferred-norm-oracle-tests tests/hc_deferred_norm_oracle_tests.cpp)
    target_link_libraries(strix-hc-deferred-norm-oracle-tests PRIVATE strix_q4k_a8_contract)
    strix_target_warnings(strix-hc-deferred-norm-oracle-tests)
    if(MSVC)
        target_compile_options(strix-hc-deferred-norm-oracle-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-hc-deferred-norm-oracle-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-hc-deferred-norm-oracle-tests COMMAND strix-hc-deferred-norm-oracle-tests)

    add_executable(strix-lane-chain-hyper-output-tests tests/lane_chain_hyper_output_tests.cpp)
    target_link_libraries(strix-lane-chain-hyper-output-tests PRIVATE strix_core)
    strix_target_warnings(strix-lane-chain-hyper-output-tests)
    add_test(NAME strix-lane-chain-hyper-output-tests COMMAND strix-lane-chain-hyper-output-tests)
    add_executable(strix-model-copy-source-tests tests/model_copy_source_tests.cpp)
    target_include_directories(strix-model-copy-source-tests PRIVATE include)
    strix_target_warnings(strix-model-copy-source-tests)
    add_test(NAME strix-model-copy-source-tests COMMAND strix-model-copy-source-tests)
    add_executable(strix-model-selective-copy-layout-tests tests/model_selective_copy_layout_tests.cpp)
    target_link_libraries(strix-model-selective-copy-layout-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-selective-copy-layout-tests)
    add_test(NAME strix-model-selective-copy-layout-tests COMMAND strix-model-selective-copy-layout-tests)
    add_executable(strix-gdn-output-admission-tests tests/gdn_output_admission_tests.cpp)
    target_link_libraries(strix-gdn-output-admission-tests PRIVATE strix_core)
    strix_target_warnings(strix-gdn-output-admission-tests)
    add_test(NAME strix-gdn-output-admission-tests COMMAND strix-gdn-output-admission-tests)
    add_executable(strix-flash-pf-policy-tests tests/flash_pf_policy_tests.cpp)
    target_include_directories(strix-flash-pf-policy-tests PRIVATE src)
    target_link_libraries(strix-flash-pf-policy-tests PRIVATE strix_model_data)
    strix_target_warnings(strix-flash-pf-policy-tests)
    add_test(NAME strix-flash-pf-policy-tests COMMAND strix-flash-pf-policy-tests)
    add_executable(strix-recurrent-state-codec-tests tests/recurrent_state_codec_tests.cpp)
    target_link_libraries(strix-recurrent-state-codec-tests PRIVATE strix_core)
    target_include_directories(strix-recurrent-state-codec-tests PRIVATE src)
    strix_target_warnings(strix-recurrent-state-codec-tests)
    add_test(NAME strix-recurrent-state-codec-tests COMMAND strix-recurrent-state-codec-tests)
    add_executable(strix-recurrent-state-admission-tests tests/recurrent_state_admission_tests.cpp)
    target_link_libraries(strix-recurrent-state-admission-tests PRIVATE strix_core)
    target_include_directories(strix-recurrent-state-admission-tests PRIVATE src)
    strix_target_warnings(strix-recurrent-state-admission-tests)
    foreach(state IN ITEMS f32 f16 bf16 empty invalid)
        add_test(NAME strix-recurrent-state-admission-${state}
            COMMAND strix-recurrent-state-admission-tests ${state})
    endforeach()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-recurrent-state-boundary-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/recurrent_state_boundary_tests.py)
    endif()
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

    add_executable(strix-qwen-moe-prepared-tests tests/qwen_moe_prepared_tests.cpp)
    target_link_libraries(strix-qwen-moe-prepared-tests PRIVATE strix_core)
    strix_target_warnings(strix-qwen-moe-prepared-tests)
    add_test(NAME strix-qwen-moe-prepared-tests COMMAND strix-qwen-moe-prepared-tests)
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
    add_executable(strix-model-family-registry-tests tests/model_family_registry_tests.cpp)
    target_link_libraries(strix-model-family-registry-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-family-registry-tests)
    add_test(NAME strix-model-family-registry-tests COMMAND strix-model-family-registry-tests)
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

    add_executable(strix-moe-i4-chain-host-tests tests/moe_i4_chain_host_tests.cpp)
    target_include_directories(strix-moe-i4-chain-host-tests PRIVATE src)
    target_compile_features(strix-moe-i4-chain-host-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(strix-moe-i4-chain-host-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-moe-i4-chain-host-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME strix-moe-i4-chain-host-tests COMMAND strix-moe-i4-chain-host-tests)
    set_tests_properties(strix-moe-i4-chain-host-tests PROPERTIES LABELS "cpu;moe;fast-i4")
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-moe-i4-source-bytes COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/moe_i4_source_bytes.py)
    endif()
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

    add_executable(strix-ple-direct-tests tests/ple_direct_tests.cpp)
    target_link_libraries(strix-ple-direct-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-direct-tests)
    add_test(NAME strix-ple-direct-tests COMMAND strix-ple-direct-tests)
    set_tests_properties(strix-ple-direct-tests PROPERTIES TIMEOUT 120)

    add_executable(strix-ple-rolling-tests tests/ple_rolling_tests.cpp)
    target_link_libraries(strix-ple-rolling-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-rolling-tests)
    add_test(NAME strix-ple-rolling-tests COMMAND strix-ple-rolling-tests)
    set_tests_properties(strix-ple-rolling-tests PROPERTIES TIMEOUT 120)

    add_executable(strix-ple-discard-safety-tests tests/ple_discard_safety_tests.cpp)
    target_include_directories(strix-ple-discard-safety-tests PRIVATE include)
    target_compile_features(strix-ple-discard-safety-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-ple-discard-safety-tests)
    add_test(NAME strix-ple-discard-safety-tests COMMAND strix-ple-discard-safety-tests)

    add_executable(strix-hip-model-residency-release-host-tests
        tests/hip_model_residency_release_host_tests.cpp src/hip_model_residency.cpp)
    target_include_directories(strix-hip-model-residency-release-host-tests PRIVATE tests/stubs/hip_model_residency)
    target_link_libraries(strix-hip-model-residency-release-host-tests PRIVATE strix_core)
    strix_target_warnings(strix-hip-model-residency-release-host-tests)
    add_test(NAME strix-hip-model-residency-release-host-tests COMMAND strix-hip-model-residency-release-host-tests)
    set_tests_properties(strix-hip-model-residency-release-host-tests PROPERTIES
        LABELS "cpu" SKIP_RETURN_CODE 77 TIMEOUT 30)

    add_executable(strix-ple-iq4-nl-tests tests/ple_iq4_nl_tests.cpp)
    target_link_libraries(strix-ple-iq4-nl-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-iq4-nl-tests)
    add_test(NAME strix-ple-iq4-nl-tests COMMAND strix-ple-iq4-nl-tests)

    add_executable(strix-ple-workspace-tests tests/ple_workspace_tests.cpp)
    target_link_libraries(strix-ple-workspace-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-workspace-tests)
    add_test(NAME strix-ple-workspace-tests COMMAND strix-ple-workspace-tests)

    add_executable(strix-ple-staged-gather-tests tests/ple_staged_gather_tests.cpp)
    target_link_libraries(strix-ple-staged-gather-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-staged-gather-tests)
    add_test(NAME strix-ple-staged-gather-tests COMMAND strix-ple-staged-gather-tests)

    add_executable(strix-ple-serve-policy-tests tests/ple_serve_policy_tests.cpp)
    target_link_libraries(strix-ple-serve-policy-tests PRIVATE strix_core)
    strix_target_warnings(strix-ple-serve-policy-tests)
    add_test(NAME strix-ple-serve-policy-tests COMMAND strix-ple-serve-policy-tests)

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

    add_executable(strix-row-layout-tests tests/row_layout_tests.cpp)
    target_link_libraries(strix-row-layout-tests PRIVATE strix_core)
    strix_target_warnings(strix-row-layout-tests)
    add_test(NAME strix-row-layout-tests COMMAND strix-row-layout-tests)

    add_executable(strix-qsa-score-scratch-tests tests/qsa_score_scratch_tests.cpp)
    target_link_libraries(strix-qsa-score-scratch-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-score-scratch-tests)
    add_test(NAME strix-qsa-score-scratch-tests COMMAND strix-qsa-score-scratch-tests)

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

    find_package(Python3 COMPONENTS Interpreter QUIET)
    option(STRIX_ASSET_CLI_TESTS "Require asset CLI qualification" ON)
    if(STRIX_ASSET_CLI_TESTS)
        if(Python3_Interpreter_FOUND)
            add_test(NAME strix-asset-manifest-cli-tests COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/asset_manifest_cli_test.py $<TARGET_FILE:strix-model-manifest>)
        else()
            add_test(NAME strix-asset-manifest-cli-tests COMMAND ${CMAKE_COMMAND}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/asset_cli_requires_python.cmake)
        endif()
    endif()
    add_executable(strix-asset-manifest-tests tests/asset_manifest_test.cpp src/asset_manifest.cpp src/serve/json.cpp)
    target_link_libraries(strix-asset-manifest-tests PRIVATE strix_core)
    target_compile_definitions(strix-asset-manifest-tests PRIVATE STRIX_ASSET_TEST_HOOKS=1)
    target_compile_definitions(strix-asset-manifest-tests PRIVATE STRIX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    strix_target_warnings(strix-asset-manifest-tests)
    add_test(NAME strix-asset-manifest-tests COMMAND strix-asset-manifest-tests)
    add_executable(strix-submit-window-tests tests/submit_window_test.cpp)
    target_include_directories(strix-submit-window-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-submit-window-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-submit-window-tests)
    add_test(NAME strix-submit-window-tests COMMAND strix-submit-window-tests)
    add_executable(strix-prefill-piece-plan-tests tests/prefill_piece_plan_test.cpp)
    target_include_directories(strix-prefill-piece-plan-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-prefill-piece-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-prefill-piece-plan-tests)
    add_test(NAME strix-prefill-piece-plan-tests COMMAND strix-prefill-piece-plan-tests)

    add_executable(strix-preflight-tests tests/preflight_test.cpp src/serve/json.cpp)
    target_link_libraries(strix-preflight-tests PRIVATE strix_core)
    strix_target_warnings(strix-preflight-tests)
    add_test(NAME strix-preflight-tests COMMAND strix-preflight-tests)

    add_executable(strix-build-info-tests tests/build_info_test.cpp src/serve/json.cpp)
    target_link_libraries(strix-build-info-tests PRIVATE strix_core)
    strix_target_warnings(strix-build-info-tests)
    add_test(NAME strix-build-info-tests COMMAND strix-build-info-tests)

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

    add_executable(strix-session-executor-handoff-tests tests/session_executor_handoff_tests.cpp)
    target_link_libraries(strix-session-executor-handoff-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-executor-handoff-tests)
    add_test(NAME strix-session-executor-handoff-tests COMMAND strix-session-executor-handoff-tests)

    add_executable(strix-bitcalc-measurement-tests tests/bitcalc_measurement_tests.cpp)
    target_link_libraries(strix-bitcalc-measurement-tests PRIVATE strix_core)
    strix_target_warnings(strix-bitcalc-measurement-tests)
    add_test(NAME strix-bitcalc-measurement-tests COMMAND strix-bitcalc-measurement-tests)

    add_executable(strix-serial-generation-cli-tests tests/serial_generation_cli_tests.cpp)
    target_link_libraries(strix-serial-generation-cli-tests PRIVATE strix_serve)
    target_include_directories(strix-serial-generation-cli-tests BEFORE PRIVATE tests/stubs/cuda_status)
    strix_target_warnings(strix-serial-generation-cli-tests)
    add_test(NAME strix-serial-generation-cli-tests COMMAND strix-serial-generation-cli-tests)

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

    add_executable(strix-session-conv-tape-tests tests/session_conv_tape_tests.cpp)
    target_link_libraries(strix-session-conv-tape-tests PRIVATE strix_core)
    strix_target_warnings(strix-session-conv-tape-tests)
    add_test(NAME strix-session-conv-tape-tests COMMAND strix-session-conv-tape-tests)

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

    add_executable(strix-speculation-multi-draft-tests tests/speculation_multi_draft_tests.cpp)
    target_link_libraries(strix-speculation-multi-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculation-multi-draft-tests)
    add_test(NAME strix-speculation-multi-draft-tests COMMAND strix-speculation-multi-draft-tests)

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

    add_executable(strix-token-sampling-tests tests/token_sampling_tests.cpp)
    target_link_libraries(strix-token-sampling-tests PRIVATE strix_core)
    strix_target_warnings(strix-token-sampling-tests)
    add_test(NAME strix-token-sampling-tests COMMAND strix-token-sampling-tests)
    add_executable(strix-token-sampling-device-plan-tests tests/token_sampling_device_plan_tests.cpp)
    target_link_libraries(strix-token-sampling-device-plan-tests PRIVATE strix_core)
    strix_target_warnings(strix-token-sampling-device-plan-tests)
    add_test(NAME strix-token-sampling-device-plan-tests COMMAND strix-token-sampling-device-plan-tests)
    add_executable(strix-job-sampling-tests tests/job_sampling_tests.cpp)
    target_link_libraries(strix-job-sampling-tests PRIVATE strix_core)
    strix_target_warnings(strix-job-sampling-tests)
    add_test(NAME strix-job-sampling-tests COMMAND strix-job-sampling-tests)
    set_tests_properties(strix-job-sampling-tests PROPERTIES ENVIRONMENT "STRIX_SAMPLING_REQUEST_IDENTITY=single")

    add_executable(strix-pair-sampling-host-tests tests/pair_sampling_host_tests.cpp)
    target_include_directories(strix-pair-sampling-host-tests PRIVATE src)
    target_link_libraries(strix-pair-sampling-host-tests PRIVATE strix_core)
    strix_target_warnings(strix-pair-sampling-host-tests)
    add_test(NAME strix-pair-sampling-host-tests COMMAND strix-pair-sampling-host-tests)

    add_executable(strix-cuda-serve-sampling-tests tests/cuda_serve_sampling_tests.cpp)
    target_link_libraries(strix-cuda-serve-sampling-tests PRIVATE strix_core)
    strix_target_warnings(strix-cuda-serve-sampling-tests)
    add_test(NAME strix-cuda-serve-sampling-tests COMMAND strix-cuda-serve-sampling-tests)
    set_tests_properties(strix-cuda-serve-sampling-tests PROPERTIES
        ENVIRONMENT "STRIX_SAMPLING_REQUEST_IDENTITY=single")

    add_executable(strix-cohort-sampling-tests tests/cohort_sampling_tests.cpp)
    target_link_libraries(strix-cohort-sampling-tests PRIVATE strix_serve)
    strix_target_warnings(strix-cohort-sampling-tests)
    add_test(NAME strix-cohort-sampling-tests COMMAND strix-cohort-sampling-tests)

    add_executable(strix-token-sampling-coupled-tests tests/token_sampling_coupled_tests.cpp)
    target_link_libraries(strix-token-sampling-coupled-tests PRIVATE strix_core)
    strix_target_warnings(strix-token-sampling-coupled-tests)
    add_test(NAME strix-token-sampling-coupled-tests COMMAND strix-token-sampling-coupled-tests)

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

    add_executable(strix-qsa-committed-pool-lease-tests tests/qsa_committed_pool_lease_tests.cpp)
    target_link_libraries(strix-qsa-committed-pool-lease-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-committed-pool-lease-tests)
    add_test(NAME strix-qsa-committed-pool-lease-tests COMMAND strix-qsa-committed-pool-lease-tests)

    add_executable(strix-mtp-step-logits-digest-tests tests/mtp_step_logits_digest_tests.cpp)
    target_link_libraries(strix-mtp-step-logits-digest-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-step-logits-digest-tests)
    add_test(NAME strix-mtp-step-logits-digest-tests COMMAND strix-mtp-step-logits-digest-tests)
    add_executable(strix-qsa-committed-pool-tests tests/qsa_committed_pool_tests.cpp)
    target_link_libraries(strix-qsa-committed-pool-tests PRIVATE strix_core)
    strix_target_warnings(strix-qsa-committed-pool-tests)
    add_test(NAME strix-qsa-committed-pool-tests COMMAND strix-qsa-committed-pool-tests)

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

    add_executable(strix-mtp-ngram-draft-tests tests/mtp_ngram_draft_tests.cpp)
    target_link_libraries(strix-mtp-ngram-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-ngram-draft-tests)
    add_test(NAME strix-mtp-ngram-draft-tests COMMAND strix-mtp-ngram-draft-tests)

    add_executable(strix-mtp-frame-draft-tests tests/mtp_frame_draft_tests.cpp)
    target_link_libraries(strix-mtp-frame-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-frame-draft-tests)
    add_test(NAME strix-mtp-frame-draft-tests COMMAND strix-mtp-frame-draft-tests)

    add_executable(strix-mtp-context-draft-tests tests/mtp_context_draft_tests.cpp)
    target_link_libraries(strix-mtp-context-draft-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-context-draft-tests)
    add_test(NAME strix-mtp-context-draft-tests COMMAND strix-mtp-context-draft-tests)

    add_executable(strix-mtp-draft-activation-tests tests/mtp_draft_activation_tests.cpp)
    target_link_libraries(strix-mtp-draft-activation-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-activation-tests)
    add_test(NAME strix-mtp-draft-activation-tests COMMAND strix-mtp-draft-activation-tests)

    add_executable(strix-mtp-forward-replay-tests tests/mtp_forward_replay_tests.cpp)
    target_link_libraries(strix-mtp-forward-replay-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-replay-tests)
    add_test(NAME strix-mtp-forward-replay-tests COMMAND strix-mtp-forward-replay-tests)

    add_executable(strix-hip-mtp-decode-options-tests tests/hip_mtp_decode_options_tests.cpp)
    target_link_libraries(strix-hip-mtp-decode-options-tests PRIVATE strix_core)
    strix_target_warnings(strix-hip-mtp-decode-options-tests)
    add_test(NAME strix-hip-mtp-decode-options-tests COMMAND strix-hip-mtp-decode-options-tests)
    add_executable(strix-pair-baseline-tests tests/pair_baseline_tests.cpp)
    target_link_libraries(strix-pair-baseline-tests PRIVATE strix_core strix_serve_config strix_serve_json)
    target_include_directories(strix-pair-baseline-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    strix_target_warnings(strix-pair-baseline-tests)
    foreach(mode cli env default)
        add_test(NAME strix-pair-baseline-${mode}-tests COMMAND strix-pair-baseline-tests ${mode})
    endforeach()

    add_executable(strix-mtp-forward-golden-file-tests tests/mtp_forward_golden_file_tests.cpp)
    target_link_libraries(strix-mtp-forward-golden-file-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-forward-golden-file-tests)
    add_test(NAME strix-mtp-forward-golden-file-tests COMMAND strix-mtp-forward-golden-file-tests)

    add_executable(strix-mtp-target-window-policy-tests tests/mtp_target_window_policy_tests.cpp)
    target_link_libraries(strix-mtp-target-window-policy-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-target-window-policy-tests)
    add_test(NAME strix-mtp-target-window-policy-tests COMMAND strix-mtp-target-window-policy-tests)

    add_executable(strix-mtp-window-controller-tests tests/mtp_window_controller_tests.cpp)
    target_link_libraries(strix-mtp-window-controller-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-window-controller-tests)
    add_test(NAME strix-mtp-window-controller-tests COMMAND strix-mtp-window-controller-tests)
    add_executable(strix-cost-table-cache-tests tests/cost_table_cache_tests.cpp)
    target_link_libraries(strix-cost-table-cache-tests PRIVATE strix_core)
    strix_target_warnings(strix-cost-table-cache-tests)
    add_test(NAME strix-cost-table-cache-tests COMMAND strix-cost-table-cache-tests)
    add_executable(strix-mtp-depth-window-controller-tests tests/mtp_depth_window_controller_tests.cpp)
    target_link_libraries(strix-mtp-depth-window-controller-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-depth-window-controller-tests)
    add_test(NAME strix-mtp-depth-window-controller-tests COMMAND strix-mtp-depth-window-controller-tests)
    add_executable(strix-mtp-draft-split-tests tests/mtp_draft_split_tests.cpp)
    target_link_libraries(strix-mtp-draft-split-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-draft-split-tests)
    add_test(NAME strix-mtp-draft-split-tests COMMAND strix-mtp-draft-split-tests)
    add_executable(strix-mtp-shared-execution-witness-tests tests/mtp_shared_execution_witness_tests.cpp)
    target_link_libraries(strix-mtp-shared-execution-witness-tests PRIVATE strix_core)
    strix_target_warnings(strix-mtp-shared-execution-witness-tests)
    add_test(NAME strix-mtp-shared-execution-witness-tests COMMAND strix-mtp-shared-execution-witness-tests)

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
    add_test(NAME strix-compiled-speculative-verify-compact-images-off-tests COMMAND strix-compiled-speculative-verify-tests)
    set_tests_properties(strix-compiled-speculative-verify-compact-images-off-tests PROPERTIES ENVIRONMENT "STRIX_QPF2K_COMPACT_IMAGES=0")

    add_executable(strix-hip-gdn-identity-tests tests/hip_gdn_identity_tests.cpp)
    target_link_libraries(strix-hip-gdn-identity-tests PRIVATE strix_core)
    strix_target_warnings(strix-hip-gdn-identity-tests)
    foreach(fuse IN ITEMS 0 1)
        add_test(NAME strix-hip-gdn-identity-fuse-${fuse}-tests COMMAND strix-hip-gdn-identity-tests ${fuse})
        add_test(NAME strix-compiled-speculative-verify-fuse-${fuse}-tests COMMAND strix-compiled-speculative-verify-tests)
        set_tests_properties(strix-hip-gdn-identity-fuse-${fuse}-tests strix-compiled-speculative-verify-fuse-${fuse}-tests
            PROPERTIES LABELS "cpu" ENVIRONMENT "STRIX_QWEN_DECODE_FUSE=${fuse};STRIX_BASELINE=0;STRIX_QPF2K_DEFAULTS=1")
    endforeach()

    add_executable(strix-speculative-verify-runtime-tests tests/speculative_verify_runtime_tests.cpp)
    target_link_libraries(strix-speculative-verify-runtime-tests PRIVATE strix_core)
    strix_target_warnings(strix-speculative-verify-runtime-tests)
    add_test(NAME strix-speculative-verify-runtime-tests COMMAND strix-speculative-verify-runtime-tests)

    add_executable(strix-known-window-rows-binding-tests tests/known_window_rows_binding_tests.cpp)
    target_include_directories(strix-known-window-rows-binding-tests PRIVATE include)
    target_compile_features(strix-known-window-rows-binding-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-known-window-rows-binding-tests)
    add_test(NAME strix-known-window-rows-binding-tests COMMAND strix-known-window-rows-binding-tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-qwen-yarn-python-check
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/oracles/qwen_yarn_table_check.py
                --emitter $<TARGET_FILE:strix-qwen-yarn-table-tests>)
        add_test(NAME strix-qsa-score-split-loads-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/qsa_score_split_loads_tests.py)
        add_test(NAME strix-session-gate-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/session_gate_tests.py)
        add_test(NAME strix-ds41-pair-setup-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/ds41_pair_setup_tests.py)
        add_test(NAME strix-sampled-coupled-gate-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/sampled_coupled_gate_tests.py)
        add_test(NAME strix-qwen-prefill-public-controls-gate-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/qwen_prefill_public_controls_gate_tests.py)
        add_test(NAME strix-qwen-decode-fuse-gate-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/qwen_decode_fuse_gate_tests.py)
        add_test(NAME strix-ctest-evidence-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest_evidence_tests.py)
        add_test(NAME strix-queue-trace-evidence-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/queue_trace_evidence_tests.py)
        add_test(NAME strix-serve-cycles-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/serve_cycles_tests.py)
        add_test(NAME strix-runtime-inventory-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/runtime_inventory_tests.py)
        # Option qualification kit (needs numpy). Package imports, so run from the source root.
        add_test(NAME strix-turbo-bar-python-tests
            COMMAND ${Python3_EXECUTABLE} -B -m unittest discover -s tools/qualification/turbo_bar/tests
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        add_test(NAME strix-turbo-bench-python-tests
            COMMAND ${Python3_EXECUTABLE} -B -m unittest discover -s scripts/bench/turbo
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        add_test(NAME strix-option-qual-core-tests
            COMMAND ${Python3_EXECUTABLE} -B -m unittest tools.qualification.option_qual.tests.test_core
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        set_tests_properties(strix-option-qual-core-tests PROPERTIES
            TIMEOUT 3600 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
        # The end-to-end module is the longest host test (~8.5 min serial on a 4-core CI runner). It runs as
        # deterministic shards (every test id in exactly one shard, tools/ci/unittest_shard.py) so a parallel
        # ctest spreads it; the high COST makes a parallel ctest start the shards first.
        set(strix_option_qual_e2e_shards 6)
        math(EXPR strix_option_qual_e2e_last "${strix_option_qual_e2e_shards} - 1")
        foreach(strix_shard RANGE ${strix_option_qual_e2e_last})
            add_test(NAME strix-option-qual-end-to-end-tests-${strix_shard}
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/ci/unittest_shard.py
                    tools.qualification.option_qual.tests.test_end_to_end ${strix_shard} ${strix_option_qual_e2e_shards}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
            set_tests_properties(strix-option-qual-end-to-end-tests-${strix_shard} PROPERTIES
                TIMEOUT 3600 COST 1000 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
        endforeach()
        # FAST gate v2 statistics (per-window null-relative KL, held-out dNLL + se, route replay bookkeeping).
        add_test(NAME strix-fast-gate-v2-tests
            COMMAND ${Python3_EXECUTABLE} -B -m unittest discover -s tools/strix-gate -p test_gate_v2.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
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


option(STRIX_DPP_REDUCE "Use exact DPP exchanges in guarded Qwen reductions (build-time lever, off unless requested)" OFF)

if(BUILD_TESTING)
    add_executable(strix-stored-i4-host-tests tests/stored_i4_host_tests.cpp)
    target_include_directories(strix-stored-i4-host-tests PRIVATE src)
    target_compile_features(strix-stored-i4-host-tests PRIVATE cxx_std_20)
    add_test(NAME strix-stored-i4-host-tests COMMAND strix-stored-i4-host-tests)
    add_executable(strix-moe-int4-layer-policy-host-tests tests/moe_int4_layer_policy_host_tests.cpp)
    target_include_directories(strix-moe-int4-layer-policy-host-tests PRIVATE src)
    target_compile_features(strix-moe-int4-layer-policy-host-tests PRIVATE cxx_std_20)
    add_test(NAME strix-moe-int4-layer-policy-host-tests COMMAND strix-moe-int4-layer-policy-host-tests)
endif()

if(STRIX_ENABLE_HIP_RUNTIME)
    enable_language(HIP)
    include_directories("$<$<COMPILE_LANGUAGE:HIP>:${CMAKE_CURRENT_SOURCE_DIR}/include>")

    # CMake's HIP language drives device compilation directly. ROCm 7.x
    # development images do not necessarily ship the legacy hipConfig.cmake /
    # hip::host package target, so locate the runtime header/library from the
    # toolkit instead of requiring that package surface. The compiler's own ROCm root comes before
    # /opt/rocm so a toolkit selected only by CMAKE_HIP_COMPILER gets its own headers and runtime.
    find_path(
        STRIX_HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        HINTS "$ENV{ROCM_PATH}" "${CMAKE_HIP_COMPILER_ROCM_ROOT}" /opt/rocm
        PATH_SUFFIXES include
        REQUIRED
    )
    find_library(
        STRIX_HIP_RUNTIME_LIBRARY
        NAMES amdhip64
        HINTS "$ENV{ROCM_PATH}" "${CMAKE_HIP_COMPILER_ROCM_ROOT}" /opt/rocm
        PATH_SUFFIXES lib lib64
        REQUIRED
    )
    add_library(strix_tiled_core STATIC src/core/wmma_tiled.hip)
    target_include_directories(strix_tiled_core PUBLIC src include PRIVATE "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_options(strix_tiled_core PRIVATE -O3 ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
        -fhip-fp32-correctly-rounded-divide-sqrt)
    set_target_properties(strix_tiled_core PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_link_libraries(strix_tiled_core PUBLIC "${STRIX_HIP_RUNTIME_LIBRARY}")

    add_executable(strix-inkling2-mini-hip apps/inkling2_mini_hip.hip)
    target_link_libraries(strix-inkling2-mini-hip PRIVATE strix_tiled_core strix_hip_runtime strix_inkling2_mini_checkpoint strix_core strix_serve)
    strix_target_warnings(strix-inkling2-mini-hip)
    target_compile_options(strix-inkling2-mini-hip PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off -fno-fast-math
        -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
    set_target_properties(strix-inkling2-mini-hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")

    if(BUILD_TESTING)
        add_executable(strix-hip-resident-q8-tile-tests tests/hip_resident_q8_tile_tests.hip)
        set_target_properties(strix-hip-resident-q8-tile-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-resident-q8-tile-tests PRIVATE include src "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_options(strix-hip-resident-q8-tile-tests PRIVATE
            -O3 ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-resident-q8-tile-tests PRIVATE strix-weight "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix-hip-resident-q8-tile-tests)
        add_test(NAME strix-hip-resident-q8-tile-tests COMMAND strix-hip-resident-q8-tile-tests)
        # No-model fixture of the resident dense rows storages (bf16 / q8_0 / q4_g64 / q6_g64): check + GB/s (GPU).
        add_executable(strix-hip-dense-lowbit-bench tests/hip_dense_lowbit_bench.hip)
        set_target_properties(strix-hip-dense-lowbit-bench PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-dense-lowbit-bench PRIVATE include src "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_options(strix-hip-dense-lowbit-bench PRIVATE
            -O3 ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-dense-lowbit-bench PRIVATE strix-weight "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix-hip-dense-lowbit-bench)
        # F9 recur: the rows launch against single-row step launches (padded strides, distinct gate rows); GPU run lanes only.
        add_executable(strix-hip-recur-gated-delta-rows-tests tests/recur_gated_delta_rows_hip_tests.hip)
        set_target_properties(strix-hip-recur-gated-delta-rows-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-recur-gated-delta-rows-tests PRIVATE include src "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_options(strix-hip-recur-gated-delta-rows-tests PRIVATE
            -O3 ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-recur-gated-delta-rows-tests PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix-hip-recur-gated-delta-rows-tests)
        add_test(NAME strix-hip-recur-gated-delta-rows-tests COMMAND strix-hip-recur-gated-delta-rows-tests)
        # F9 recur window form: consecutive deferred windows (alternating saved slots, cohort segments) against the serial form; GPU run lanes only.
        add_executable(strix-hip-recur-gdn-window-tests tests/recur_gdn_window_hip_tests.hip)
        set_target_properties(strix-hip-recur-gdn-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-recur-gdn-window-tests PRIVATE include src "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_options(strix-hip-recur-gdn-window-tests PRIVATE
            -O3 ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-recur-gdn-window-tests PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix-hip-recur-gdn-window-tests)
        add_test(NAME strix-hip-recur-gdn-window-tests COMMAND strix-hip-recur-gdn-window-tests)
        # Shared Radix3 select device entry point (src/attn_core/select_radix3.hpp) vs the host rule, eager + graph forms,
        # blocks <= k and k == 0 boundaries with guarded outputs. Needs a gfx1151 device (exit 77 = no device).
        add_executable(strix-hip-select-radix3-tests tests/hip_select_radix3_tests.hip)
        set_target_properties(strix-hip-select-radix3-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-select-radix3-tests PRIVATE include src "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_options(strix-hip-select-radix3-tests PRIVATE
            -O3 ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-select-radix3-tests PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix-hip-select-radix3-tests)
        add_test(NAME strix-hip-select-radix3-tests COMMAND strix-hip-select-radix3-tests)
        set_tests_properties(strix-hip-select-radix3-tests PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;attn;exactness")
        # Shared EXACT chunk-partial combine device entry point (src/attn_core/chunk_exact_combine.hpp) vs the host rule, output
        # bits, plus the empty-row canary. Needs a gfx1151 device (exit 77 = no device).
        add_executable(strix-hip-chunk-exact-combine-tests tests/hip_chunk_exact_combine_tests.hip)
        set_target_properties(strix-hip-chunk-exact-combine-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-chunk-exact-combine-tests PRIVATE include src "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_options(strix-hip-chunk-exact-combine-tests PRIVATE
            -O3 ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
        target_link_libraries(strix-hip-chunk-exact-combine-tests PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix-hip-chunk-exact-combine-tests)
        add_test(NAME strix-hip-chunk-exact-combine-tests COMMAND strix-hip-chunk-exact-combine-tests)
        set_tests_properties(strix-hip-chunk-exact-combine-tests PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;attn;exactness")

        # The verify fixture requires a separately supplied model checkpoint.
        add_executable(strix-hip-inkling2-mini-verify-rows-test tests/inkling2/mini_verify_rows_hip.hip)
        target_link_libraries(strix-hip-inkling2-mini-verify-rows-test PRIVATE
            strix_tiled_core strix_hip_runtime strix_inkling2_mini_checkpoint strix_core)
        strix_target_warnings(strix-hip-inkling2-mini-verify-rows-test)
        target_compile_options(strix-hip-inkling2-mini-verify-rows-test PRIVATE
            -O3 -mno-wavefrontsize64 -ffp-contract=off -fno-fast-math
            -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
        set_target_properties(strix-hip-inkling2-mini-verify-rows-test PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        # Sampled serving lanes read the scores of their own (screened) step; also needs the model checkpoint.
        add_executable(strix-hip-inkling2-mini-screened-scores-test tests/inkling2/mini_screened_scores_hip.hip)
        target_link_libraries(strix-hip-inkling2-mini-screened-scores-test PRIVATE
            strix_tiled_core strix_hip_runtime strix_inkling2_mini_checkpoint strix_core)
        strix_target_warnings(strix-hip-inkling2-mini-screened-scores-test)
        target_compile_options(strix-hip-inkling2-mini-screened-scores-test PRIVATE
            -O3 -mno-wavefrontsize64 -ffp-contract=off -fno-fast-math
            -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
        set_target_properties(strix-hip-inkling2-mini-screened-scores-test PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        add_executable(strix-hip-inkling2-mini-attention-prepare-test tests/inkling2/mini_attention_prepare_hip.hip)
        add_executable(strix-hip-inkling2-mini-attention-projected-bias-test tests/inkling2/mini_attention_projected_bias_hip.hip)
        add_executable(strix-hip-inkling2-mini-prepared-round-fold-test tests/inkling2/mini_prepared_round_fold_hip.hip)
        add_executable(strix-hip-inkling2-mini-dense-test tests/inkling2/mini_dense_hip.hip)
        add_executable(strix-hip-inkling2-mini-gemv-test tests/inkling2/mini_gemv_hip.hip)
        add_executable(strix-hip-inkling2-mini-residual-test tests/inkling2/mini_residual_device.hip)
        add_executable(strix-hip-inkling2-mini-attention-test tests/inkling2/mini_attention_policy_hip.hip)
        add_executable(strix-hip-inkling2-mini-attention-keysplit-test tests/inkling2/mini_attention_keysplit_hip.hip)
        add_executable(strix-hip-inkling2-mini-short-conv-test tests/inkling2/mini_short_conv_hip.hip)
        add_executable(strix-hip-inkling2-mini-grouped-test tests/inkling2/mini_grouped_microscaled_hip.hip)
        add_executable(strix-hip-inkling2-mini-grouped-rows-test tests/inkling2/mini_grouped_rows_hip.hip)
        add_executable(strix-hip-inkling2-mini-expert-activation-test tests/inkling2/mini_expert_activation_hip.hip)
        add_executable(strix-hip-inkling2-mini-moe-test tests/inkling2/mini_moe_hip.hip)
        add_executable(strix-hip-inkling2-mini-dspark-test tests/inkling2/mini_dspark_device.hip)
        target_link_libraries(strix-hip-inkling2-mini-dspark-test PRIVATE
            strix_hip_runtime strix_inkling2_mini_checkpoint strix_core strix_serve)
        add_executable(strix-hip-inkling2-mini-nvfp4-table-test tests/inkling2/mini_nvfp4_table_test.hip)
        add_executable(strix-hip-short-bf16-projection-test tests/core/short_bf16_projection_hip.hip)
        add_executable(strix-hip-dense-flash-projected-bias-test tests/attn_core/dense_flash_projected_bias_hip.hip)
        add_executable(strix-hip-bf16-controller-rows-test tests/core/bf16_controller_rows_hip.hip)
        add_executable(strix-hip-controller-image-test tests/core/controller_image_hip.hip)
        add_executable(strix-hip-projection-image-packed-test tests/core/projection_image_packed_hip.hip)
        add_executable(strix-hip-controller-rms-test tests/core/controller_rms_hip.hip)
        add_executable(strix-hip-mix-serial-registers-test tests/core/mix_serial_registers_hip.hip)
        add_executable(strix-hip-rms-wave-rows-test tests/core/rms_wave_rows_hip.hip)
        add_executable(strix-hip-dense-linear-tight-rows-test tests/core/dense_linear_tight_rows_hip.hip)
        add_executable(strix-hip-screened-head-test tests/core/screened_head_hip.hip)
        add_executable(strix-hip-round-fold-test tests/core/round_fold_hip.hip)
        add_executable(strix-hip-round-store-incumbent-test tests/core/round_store_incumbent_hip.hip)
        add_executable(strix-hip-combine-image-test tests/moe_core/combine_image_hip.hip)
        foreach(target strix-hip-inkling2-mini-attention-prepare-test strix-hip-inkling2-mini-attention-projected-bias-test
                       strix-hip-inkling2-mini-prepared-round-fold-test
                       strix-hip-inkling2-mini-dense-test strix-hip-inkling2-mini-residual-test
                       strix-hip-inkling2-mini-gemv-test
                       strix-hip-inkling2-mini-attention-test strix-hip-inkling2-mini-short-conv-test
                       strix-hip-inkling2-mini-attention-keysplit-test
                       strix-hip-inkling2-mini-grouped-test strix-hip-inkling2-mini-grouped-rows-test
                       strix-hip-inkling2-mini-expert-activation-test strix-hip-inkling2-mini-moe-test
                       strix-hip-inkling2-mini-dspark-test
                       strix-hip-inkling2-mini-nvfp4-table-test strix-hip-short-bf16-projection-test strix-hip-bf16-controller-rows-test
                       strix-hip-controller-image-test
                       strix-hip-projection-image-packed-test
                       strix-hip-controller-rms-test
                       strix-hip-mix-serial-registers-test
                       strix-hip-rms-wave-rows-test
                       strix-hip-dense-linear-tight-rows-test strix-hip-dense-flash-projected-bias-test
                       strix-hip-screened-head-test strix-hip-round-fold-test strix-hip-combine-image-test
                       strix-hip-round-store-incumbent-test)
            target_link_libraries(${target} PRIVATE strix_tiled_core)
            strix_target_warnings(${target})
            target_compile_options(${target} PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off -fno-fast-math)
            set_target_properties(${target} PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            add_test(NAME ${target} COMMAND ${target})
            set_tests_properties(${target} PROPERTIES LABELS "hip;inkling2")
        endforeach()
        # resident picker over a reply the screened head produced (enqueue_produced)
        target_link_libraries(strix-hip-screened-head-test PRIVATE strix_hip_runtime)
        # lanes form of the screened head: several activation rows against one head (verify rows, cohort lanes)
        add_executable(strix-hip-screened-head-lanes-test tests/core/screened_head_lanes_hip.hip)
        target_link_libraries(strix-hip-screened-head-lanes-test PRIVATE strix_tiled_core)
        strix_target_warnings(strix-hip-screened-head-lanes-test)
        target_compile_options(strix-hip-screened-head-lanes-test PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off
            -fno-fast-math)
        set_target_properties(strix-hip-screened-head-lanes-test PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        add_test(NAME strix-hip-screened-head-lanes-test COMMAND strix-hip-screened-head-lanes-test)
        set_tests_properties(strix-hip-screened-head-lanes-test PROPERTIES LABELS "hip;core")
        # rescore trait of the screened head: exact rows for the largest approximate scores (--bench: no-model timing)
        add_executable(strix-hip-screened-head-rescore-test tests/core/screened_head_rescore_hip.hip)
        target_link_libraries(strix-hip-screened-head-rescore-test PRIVATE strix_tiled_core)
        strix_target_warnings(strix-hip-screened-head-rescore-test)
        target_compile_options(strix-hip-screened-head-rescore-test PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off
            -fno-fast-math)
        set_target_properties(strix-hip-screened-head-rescore-test PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        add_test(NAME strix-hip-screened-head-rescore-test COMMAND strix-hip-screened-head-rescore-test)
        set_tests_properties(strix-hip-screened-head-rescore-test PROPERTIES LABELS "hip;core")
        target_compile_options(strix-hip-inkling2-mini-short-conv-test PRIVATE -fdenormal-fp-math=ieee)
        target_compile_options(strix-hip-inkling2-mini-nvfp4-table-test PRIVATE -fdenormal-fp-math=ieee)
        target_compile_options(strix-hip-short-bf16-projection-test PRIVATE -fdenormal-fp-math=ieee)
        target_compile_options(strix-hip-dense-flash-projected-bias-test PRIVATE -fdenormal-fp-math=ieee
            -fhip-fp32-correctly-rounded-divide-sqrt)
        target_compile_options(strix-hip-dense-linear-tight-rows-test PRIVATE -fdenormal-fp-math=ieee)
        # No-model bench of the packed tile schedules (grouped expert projection, Inkling shapes); not a ctest.
        add_executable(strix-hip-grouped-tile-schedules tests/moe_core/grouped_tile_schedules_hip.hip)
        target_link_libraries(strix-hip-grouped-tile-schedules PRIVATE strix_tiled_core)
        strix_target_warnings(strix-hip-grouped-tile-schedules)
        target_compile_options(strix-hip-grouped-tile-schedules PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off -fno-fast-math)
        set_target_properties(strix-hip-grouped-tile-schedules PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        # Decode-height schedules of the grouped expert tile: same bits for every schedule (ctest); --bench times them.
        add_executable(strix-hip-grouped-schedule-test tests/moe_core/grouped_schedule_hip.hip)
        target_link_libraries(strix-hip-grouped-schedule-test PRIVATE strix_tiled_core)
        strix_target_warnings(strix-hip-grouped-schedule-test)
        target_compile_options(strix-hip-grouped-schedule-test PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off
            -fno-fast-math -fdenormal-fp-math=ieee)
        set_target_properties(strix-hip-grouped-schedule-test PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        add_test(NAME strix-hip-grouped-schedule-test COMMAND strix-hip-grouped-schedule-test)
        set_tests_properties(strix-hip-grouped-schedule-test PROPERTIES LABELS "hip;moe")
        # Grouped rotated A4W4 expert class grouped-iu4a4-rot-v1 (moe_core/grouped_iu4a4.hpp): expert images and routed
        # outputs bit for bit against iu4a4_rot (ctest); --bench [tokens] times the pass at Inkling shapes.
        add_executable(strix-hip-grouped-iu4a4-tests tests/hip_grouped_iu4a4_tests.hip)
        target_link_libraries(strix-hip-grouped-iu4a4-tests PRIVATE strix_hip_runtime strix_core)
        target_include_directories(strix-hip-grouped-iu4a4-tests PRIVATE src include)
        set_target_properties(strix-hip-grouped-iu4a4-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        add_test(NAME strix-hip-grouped-iu4a4-tests COMMAND strix-hip-grouped-iu4a4-tests)
        set_tests_properties(strix-hip-grouped-iu4a4-tests PROPERTIES LABELS "hip;moe")
        # No-model check and bench of the R1 decode launch (Inkling shapes): rows-launch rows equal one-row launches,
        # per-case output digests for cross-build identity, --bench ms per launch; not a ctest (needs a GPU).
        add_executable(strix-hip-microscaled-r1-bench tests/moe_core/microscaled_r1_hip.hip)
        target_link_libraries(strix-hip-microscaled-r1-bench PRIVATE strix_core)
        strix_target_warnings(strix-hip-microscaled-r1-bench)
        target_compile_options(strix-hip-microscaled-r1-bench PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off
            -fno-fast-math -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
        set_target_properties(strix-hip-microscaled-r1-bench PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        foreach(target strix-hip-inkling2-mini-expert-activation-test strix-hip-inkling2-mini-moe-test
                       strix-hip-inkling2-mini-dense-test strix-hip-inkling2-mini-residual-test
                       strix-hip-inkling2-mini-gemv-test
                       strix-hip-inkling2-mini-attention-keysplit-test strix-hip-inkling2-mini-dspark-test
                       strix-hip-inkling2-mini-attention-prepare-test strix-hip-bf16-controller-rows-test
                       strix-hip-controller-image-test
                       strix-hip-projection-image-packed-test
                       strix-hip-controller-rms-test
                       strix-hip-mix-serial-registers-test
                       strix-hip-rms-wave-rows-test
                       strix-hip-inkling2-mini-attention-projected-bias-test
                       strix-hip-inkling2-mini-prepared-round-fold-test)
            target_compile_options(${target} PRIVATE -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
        endforeach()
        target_link_libraries(strix-hip-inkling2-mini-moe-test PRIVATE strix_core)
        add_executable(strix-hip-iu4-convert-tests tests/hip_iu4_convert_tests.hip)
        add_executable(strix-hip-stored-i4-decode-tests tests/hip_stored_i4_decode_tests.hip)
        add_executable(strix-hip-iu4-load-cohort-tests tests/hip_iu4_load_cohort_tests.hip)
        foreach(target strix-hip-iu4-convert-tests strix-hip-stored-i4-decode-tests strix-hip-iu4-load-cohort-tests)
            target_link_libraries(${target} PRIVATE strix_tiled_core)
            target_compile_options(${target} PRIVATE -O3 ${STRIX_HIP_WAVE32_FLAG} -ffp-contract=off -fno-fast-math)
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            add_test(NAME ${target} COMMAND ${target})
        endforeach()
    endif()

    add_library(strix_device_profile STATIC src/device_profile.cpp)
    target_include_directories(strix_device_profile PUBLIC include)
    target_include_directories(strix_device_profile SYSTEM PRIVATE "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_features(strix_device_profile PUBLIC cxx_std_20)
    target_compile_definitions(strix_device_profile PRIVATE STRIX_DEVICE_PROFILE_HIP=1 __HIP_PLATFORM_AMD__=1)
    target_link_libraries(strix_device_profile PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
    strix_target_warnings(strix_device_profile)
    # The HIP compiler treats the toolkit include directory as implicit and searches /usr/include first, so a
    # distribution HIP development package there (another HIP version) silently wins over the toolkit headers
    # the runtime library belongs to. Put the toolkit headers first for every HIP translation unit.
    add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:SHELL:-isystem ${STRIX_HIP_INCLUDE_DIR}>")

    if(BUILD_TESTING)
        add_executable(strix-moe-mul1-pair-decode-tests tests/moe_mul1_pair_decode_tests.hip)
        target_include_directories(strix-moe-mul1-pair-decode-tests PRIVATE src)
        set_target_properties(strix-moe-mul1-pair-decode-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_compile_options(strix-moe-mul1-pair-decode-tests PRIVATE
            -Wsign-conversion -Wshadow -Werror -Wrange-loop-construct -ffp-contract=off -fno-fast-math)
        add_test(NAME strix-moe-mul1-pair-decode-tests COMMAND strix-moe-mul1-pair-decode-tests)
        set_tests_properties(strix-moe-mul1-pair-decode-tests PROPERTIES LABELS "cpu;moe;exactness")
        # Shared trellis codec (src/core/trellis_codec.hpp): CPU reference/geometry gate and the device gate/bench.
        add_executable(strix-trellis-codec-tests tests/trellis_codec_tests.hip)
        target_include_directories(strix-trellis-codec-tests PRIVATE src)
        set_target_properties(strix-trellis-codec-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_compile_options(strix-trellis-codec-tests PRIVATE
            -Wsign-conversion -Wshadow -Werror -Wrange-loop-construct -ffp-contract=off -fno-fast-math)
        add_test(NAME strix-trellis-codec-tests COMMAND strix-trellis-codec-tests)
        set_tests_properties(strix-trellis-codec-tests PROPERTIES LABELS "cpu;core;exactness")
        add_executable(strix-hip-trellis-codec-tests tests/hip_trellis_codec_tests.hip)
        target_include_directories(strix-hip-trellis-codec-tests PRIVATE src)
        set_target_properties(strix-hip-trellis-codec-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_compile_options(strix-hip-trellis-codec-tests PRIVATE -O3 ${STRIX_HIP_WAVE32_FLAG} -ffp-contract=off -fno-fast-math)
        add_test(NAME strix-hip-trellis-codec-tests COMMAND strix-hip-trellis-codec-tests)
        set_tests_properties(strix-hip-trellis-codec-tests PROPERTIES LABELS "gpu;core;exactness")
        # Some HIP SDKs reject RDNA wave64 at preprocessing time. Test that
        # fallback only when the installed SDK supports building it.
        include(CMakePushCheckState)
        include(CheckSourceCompiles)
        cmake_push_check_state(RESET)
        set(CMAKE_REQUIRED_FLAGS "-mwavefrontsize64")
        check_source_compiles(HIP "#include <hip/hip_runtime.h>
int main() { return 0; }"
            STRIX_HIP_SUPPORTS_WAVE64)
        cmake_pop_check_state()
        set(qwen_dpp_test_waves 32)
        if(STRIX_HIP_SUPPORTS_WAVE64)
            list(APPEND qwen_dpp_test_waves 64)
        endif()
        foreach(wave IN LISTS qwen_dpp_test_waves)
            set(target strix-hip-qwen-dpp-reduce-wave${wave}-tests)
            add_executable(${target} tests/hip_qwen_dpp_reduce_tests.hip)
            target_include_directories(${target} PRIVATE include "${STRIX_HIP_INCLUDE_DIR}")
            target_link_libraries(${target} PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON
                HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_definitions(${target} PRIVATE STRIX_DPP_REDUCE=$<BOOL:${STRIX_DPP_REDUCE}>)
            target_compile_options(${target} PRIVATE -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
            if(wave EQUAL 32)
                target_compile_options(${target} PRIVATE ${STRIX_HIP_WAVE32_FLAG})
            else()
                target_compile_options(${target} PRIVATE -mwavefrontsize64)
            endif()
            strix_target_warnings(${target})
            add_test(NAME ${target} COMMAND ${target})
            set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;qwen;exactness")
        endforeach()
    endif()

    if(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE OR STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
        # Only these isolated arithmetic TUs receive the strict experimental flags.
        add_library(strix_q4k_a8_hip_probe STATIC src/hip_moe_q4k_a8_probe.hip src/hip_moe_q4k_a8_known_rows.hip)
        target_compile_definitions(strix_q4k_a8_hip_probe PUBLIC STRIX_DPP_REDUCE=$<BOOL:${STRIX_DPP_REDUCE}>)
        target_link_libraries(strix_q4k_a8_hip_probe PUBLIC strix_q4k_a8_contract "${STRIX_HIP_RUNTIME_LIBRARY}")
        foreach(target strix_q4k_a8_hip_probe)
            target_include_directories(${target} PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(${target} PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(${target})
        endforeach()
        if(STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE)
            add_library(strix_q4k_a8_dot4_compile_probe OBJECT lab/q4k_a8/dot4_compile_probe.hip)
            target_include_directories(strix_q4k_a8_dot4_compile_probe PRIVATE include "${STRIX_HIP_INCLUDE_DIR}")
            # This object is the native gfx1151 instruction probe.
            set_target_properties(strix_q4k_a8_dot4_compile_probe PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
            target_compile_options(strix_q4k_a8_dot4_compile_probe PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix_q4k_a8_dot4_compile_probe)
        endif()
        if(BUILD_TESTING AND STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE)
            add_executable(strix-hip-moe-a8-union-tests tests/hip_moe_a8_union_tests.hip)
            target_link_libraries(strix-hip-moe-a8-union-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core strix_hip_runtime)
            set_target_properties(strix-hip-moe-a8-union-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            strix_target_warnings(strix-hip-moe-a8-union-tests)
            add_test(NAME strix-hip-moe-a8-union-tests COMMAND strix-hip-moe-a8-union-tests)
            add_executable(strix-hip-moe-a8-down-tests tests/hip_moe_a8_down_tests.hip)
            target_link_libraries(strix-hip-moe-a8-down-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core strix_hip_runtime)
            set_target_properties(strix-hip-moe-a8-down-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-moe-a8-down-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-a8-down-tests)
            add_test(NAME strix-hip-moe-a8-down-tests COMMAND strix-hip-moe-a8-down-tests)
            add_executable(strix-hip-moe-wmma-prefill-tests tests/hip_moe_wmma_prefill_tests.hip)
            target_link_libraries(strix-hip-moe-wmma-prefill-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core strix_hip_runtime)
            set_target_properties(strix-hip-moe-wmma-prefill-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-moe-wmma-prefill-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-wmma-prefill-tests)
            add_executable(strix-hip-moe-f16x-kquant-prefill-tests tests/hip_moe_f16x_kquant_prefill_tests.hip)  # f16x K-quant envelope (Q4_K/Q6_K down, top-16, clamp)
            target_link_libraries(strix-hip-moe-f16x-kquant-prefill-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core strix_hip_runtime)
            set_target_properties(strix-hip-moe-f16x-kquant-prefill-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-moe-f16x-kquant-prefill-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-f16x-kquant-prefill-tests)
            add_test(NAME strix-hip-moe-f16x-kquant-prefill-tests COMMAND strix-hip-moe-f16x-kquant-prefill-tests)
            add_executable(strix-hip-route-prune-tests tests/hip_route_prune_tests.hip)  # shared RoutePrune policy + pruned CSR plan
            target_link_libraries(strix-hip-route-prune-tests PRIVATE strix_core)
            set_target_properties(strix-hip-route-prune-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-route-prune-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-route-prune-tests)
            add_test(NAME strix-hip-route-prune-tests COMMAND strix-hip-route-prune-tests)
            add_executable(strix-hip-exact-reduce-tests tests/hip_exact_reduce_tests.hip)  # core/ordered_reduce trees + CSR wave scan vs the LDS loops
            target_link_libraries(strix-hip-exact-reduce-tests PRIVATE strix_core)
            set_target_properties(strix-hip-exact-reduce-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-exact-reduce-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-exact-reduce-tests)
            add_test(NAME strix-hip-exact-reduce-tests COMMAND strix-hip-exact-reduce-tests)
            # Shared group-indexer device cores (pool, exact decode scan, AlwaysOwn select binding) vs the host oracle.
            add_executable(strix-hip-group-pool-tests tests/hip_group_pool_tests.hip src/attn/group_indexer_ref.cpp)
            set_target_properties(strix-hip-group-pool-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_include_directories(strix-hip-group-pool-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
            target_compile_options(strix-hip-group-pool-tests PRIVATE -fno-fast-math -ffp-contract=off
                "$<$<COMPILE_LANGUAGE:HIP>:${STRIX_HIP_WAVE32_FLAG};-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee>")
            strix_target_warnings(strix-hip-group-pool-tests)
            add_test(NAME strix-hip-group-pool-tests COMMAND strix-hip-group-pool-tests)
            # Shared Engram row kernel (DS41 decode, Hy4 head_rms_bf16) and MoE route / combine kernels (Hy4 preset / glm5) vs host oracles.
            add_executable(strix-hip-hy4-routed-tests tests/hip_hy4_routed_tests.hip)
            set_target_properties(strix-hip-hy4-routed-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_include_directories(strix-hip-hy4-routed-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
            target_compile_options(strix-hip-hy4-routed-tests PRIVATE -fno-fast-math -ffp-contract=off
                "$<$<COMPILE_LANGUAGE:HIP>:${STRIX_HIP_WAVE32_FLAG};-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee>")
            strix_target_warnings(strix-hip-hy4-routed-tests)
            add_test(NAME strix-hip-hy4-routed-tests COMMAND strix-hip-hy4-routed-tests)
            add_test(NAME strix-hip-moe-wmma-prefill-tests COMMAND strix-hip-moe-wmma-prefill-tests)
            add_test(NAME strix-hip-moe-route-bf16-tests COMMAND strix-hip-moe-wmma-prefill-tests route-bf16)
            add_test(NAME strix-hip-moe-prefill-pool16k-tests COMMAND strix-hip-moe-wmma-prefill-tests pool16k)
            add_test(NAME strix-hip-moe-prepared-layout-tests COMMAND strix-hip-moe-wmma-prefill-tests prepared-layout)
            add_test(NAME strix-hip-moe-direct-b-tests COMMAND strix-hip-moe-wmma-prefill-tests direct-b)
            add_test(NAME strix-hip-moe-prefill-schedule-tests COMMAND strix-hip-moe-wmma-prefill-tests schedule)
            add_test(NAME strix-hip-moe-prefill-gsq-tests COMMAND strix-hip-moe-wmma-prefill-tests gsq)
            add_test(NAME strix-hip-moe-prefill-tile64-tests COMMAND strix-hip-moe-wmma-prefill-tests tile64)
            add_executable(strix-hip-q4k-a8-probe-tests tests/hip_moe_q4k_a8_probe_tests.hip $<TARGET_OBJECTS:strix_q4k_a8_dot4_compile_probe>)
            target_link_libraries(strix-hip-q4k-a8-probe-tests PRIVATE strix_q4k_a8_hip_probe strix_core)
            set_target_properties(strix-hip-q4k-a8-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-q4k-a8-probe-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q4k-a8-probe-tests)
            add_test(NAME strix-hip-q4k-a8-probe-tests COMMAND strix-hip-q4k-a8-probe-tests)
            add_executable(strix-hip-q5k-a8-probe-tests tests/hip_moe_q5k_a8_probe_tests.hip)
            target_link_libraries(strix-hip-q5k-a8-probe-tests PRIVATE strix_q4k_a8_hip_probe strix_core)
            set_target_properties(strix-hip-q5k-a8-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-q5k-a8-probe-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q5k-a8-probe-tests)
            add_test(NAME strix-hip-q5k-a8-probe-tests COMMAND strix-hip-q5k-a8-probe-tests)
            add_executable(strix-hip-q20-a8-probe-tests tests/hip_moe_q20_a8_probe_tests.hip)
            target_link_libraries(strix-hip-q20-a8-probe-tests PRIVATE strix_q4k_a8_hip_probe strix_q4k_a8_contract strix_core strix_hip_runtime)
            set_target_properties(strix-hip-q20-a8-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-q20-a8-probe-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-q20-a8-probe-tests)
            add_test(NAME strix-hip-q20-a8-probe-tests COMMAND strix-hip-q20-a8-probe-tests)
        endif()
    endif()

    add_library(strix_hip_runtime STATIC
        src/hip_host_upload_backend.cpp
        src/hip_qwen_moe_prepared.cpp
        src/hip_q6p_pack.cpp
        src/hip_qwen_router_bf16.cpp
        src/hip_qwen_onfly_i4.cpp
        src/hip_dense_trellis_pack.cpp
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
        src/session/cycle_pack.hip
        src/hip_mtp_runtime.hip
        src/hip_mtp_draft_head.hip
        src/hip_mtp_fast_draft.hip
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
        src/hip_hyper_mix_launcher.hip
        src/hip_qwen_rotary_table.hip
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
        src/hip_dense_trellis.hip
    )
    # HIP backend core, model-agnostic: logprob partials of device rows (strix/logprob_partials.hpp); its own small
    # library so any engine binds it without the whole runtime (DS41 batch forward, the runtime's own users).
    add_library(strix_hip_logprob STATIC src/hip_logprob_partials.hip)
    target_include_directories(strix_hip_logprob PUBLIC include)
    target_include_directories(strix_hip_logprob SYSTEM PUBLIC "${STRIX_HIP_INCLUDE_DIR}")
    set_target_properties(strix_hip_logprob PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_link_libraries(strix_hip_logprob PUBLIC strix_core "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_compile_definitions(strix_hip_logprob PUBLIC $<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__=1>)
    add_library(strix_hip_token_sampling STATIC src/hip_token_sampling.hip src/hip_token_sampling_device.hip)
    target_include_directories(strix_hip_token_sampling PUBLIC include)
    target_include_directories(strix_hip_token_sampling SYSTEM PRIVATE "${STRIX_HIP_INCLUDE_DIR}")
    set_target_properties(strix_hip_token_sampling PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_compile_options(strix_hip_token_sampling PRIVATE "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
    target_link_libraries(strix_hip_token_sampling PUBLIC strix_core "${STRIX_HIP_RUNTIME_LIBRARY}")
    strix_target_warnings(strix_hip_token_sampling)
    target_include_directories(
        strix_hip_runtime
        PUBLIC
        include
    )
    target_include_directories(strix_hip_runtime SYSTEM PUBLIC "${STRIX_HIP_INCLUDE_DIR}")
    target_include_directories(strix_hip_runtime PRIVATE src)
    if(STRIX_ENABLE_EXPERIMENTAL_DENSE_TRELLIS)
        target_compile_definitions(strix_hip_runtime PUBLIC STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS=1)
    endif()
    target_compile_features(strix_hip_runtime PUBLIC cxx_std_20)
    target_compile_definitions(strix_hip_runtime PUBLIC STRIX_DPP_REDUCE=$<BOOL:${STRIX_DPP_REDUCE}>)
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
        HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
    )
    target_link_libraries(
        strix_hip_runtime
        PUBLIC
        strix_core
        strix_hip_logprob
        strix_hip_token_sampling
        "${STRIX_HIP_RUNTIME_LIBRARY}"
        strix_tiled_core
    )
    if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
        target_link_libraries(strix_hip_runtime PUBLIC strix_q4k_a8_hip_probe)
        target_compile_definitions(strix_hip_runtime PUBLIC STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
    endif()
    set_source_files_properties(src/hip_qsa_selection_diagnostic.hip
        PROPERTIES COMPILE_OPTIONS "${STRIX_HIP_WAVE32_FLAG};-fno-fast-math")
    set_source_files_properties(src/hip_dense_trellis.hip
        PROPERTIES COMPILE_OPTIONS "${STRIX_HIP_WAVE32_FLAG};-fno-fast-math;-ffp-contract=off;-fdenormal-fp-math=ieee")
    # logprob partials: the nonfinite test and the contract's FP32 lane order hold as written
    set_source_files_properties(src/hip_logprob_partials.hip tests/hip_logprob_partials_tests.hip
        PROPERTIES COMPILE_OPTIONS "${STRIX_HIP_WAVE32_FLAG};-fno-fast-math;-ffp-contract=off;-fdenormal-fp-math=ieee")
    set_source_files_properties(src/hip_mtp_input_fusion.hip
        PROPERTIES COMPILE_OPTIONS "${STRIX_HIP_WAVE32_FLAG};-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee")
    strix_target_warnings(strix_hip_runtime)
    if(BUILD_TESTING)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_GDN_RECURRENT_LAYOUT_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_SHARED_EXPERT_ROWS_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_QSA_GQA_REUSE_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_QWEN_ROTARY_DIAGNOSTIC=1)
        add_executable(strix-hip-qwen-yarn-rotary-tests tests/hip_qwen_yarn_rotary_tests.hip)
        target_link_libraries(strix-hip-qwen-yarn-rotary-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-qwen-yarn-rotary-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        strix_target_warnings(strix-hip-qwen-yarn-rotary-tests)
        add_test(NAME strix-hip-qwen-yarn-rotary-tests COMMAND strix-hip-qwen-yarn-rotary-tests)
        add_test(NAME strix-qwen-rotary-host-admission-tests COMMAND strix-hip-qwen-yarn-rotary-tests --host-admission-only)
        add_test(NAME strix-qwen-rotary-watchdog-tests COMMAND ${CMAKE_COMMAND}
            -DFIXTURE=$<TARGET_FILE:strix-hip-qwen-yarn-rotary-tests>
            -P ${CMAKE_SOURCE_DIR}/tests/qwen_rotary_watchdog_test.cmake)
        set_tests_properties(strix-hip-qwen-yarn-rotary-tests PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_QPERSIST_DIAGNOSTIC=1)
        target_compile_definitions(strix_hip_runtime PRIVATE STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC=1)
        foreach(mtp_fixture input-fusion forward-replay target-checkpoint draft-head fast-draft draft-batch)
            string(REPLACE "-" "_" mtp_fixture_file "${mtp_fixture}")
            add_executable(strix-hip-mtp-${mtp_fixture}-tests tests/hip_mtp_${mtp_fixture_file}_tests.hip)
            target_link_libraries(strix-hip-mtp-${mtp_fixture}-tests PRIVATE strix_hip_runtime)
            set_target_properties(strix-hip-mtp-${mtp_fixture}-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(strix-hip-mtp-${mtp_fixture}-tests PRIVATE
                ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off)
            strix_target_warnings(strix-hip-mtp-${mtp_fixture}-tests)
            add_test(NAME strix-hip-mtp-${mtp_fixture}-tests COMMAND strix-hip-mtp-${mtp_fixture}-tests)
            set_tests_properties(strix-hip-mtp-${mtp_fixture}-tests PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
        # Prompt priming with one output row per wave (STRIX_MTP_PRIME_PROJ_ROWS=1) against the same one-token reference.
        add_test(NAME strix-hip-mtp-draft-batch-one-row-prime-tests COMMAND strix-hip-mtp-draft-batch-tests)
        set_tests_properties(strix-hip-mtp-draft-batch-one-row-prime-tests PROPERTIES SKIP_RETURN_CODE 77
            ENVIRONMENT "STRIX_MTP_PRIME_PROJ_ROWS=1")
        # Callable fixture requires actual independently supplied component goldens.
        # Building this source library does not execute or qualify the components.
        add_library(strix_mtp_projection_component_fixtures STATIC tests/hip_mtp_projection_components.hip tests/hip_mtp_shared_expert_components.hip tests/hip_mtp_target_window_components.hip tests/hip_mtp_greedy_cycle_components.hip)
        target_sources(strix_mtp_projection_component_fixtures PRIVATE
            tests/hip_mtp_lm_head_components.hip
            tests/hip_mtp_target_checkpoint_components.hip
            tests/hip_mtp_qualification_reference.hip)
        target_link_libraries(strix_mtp_projection_component_fixtures PRIVATE strix_hip_runtime)
        set_target_properties(strix_mtp_projection_component_fixtures PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        strix_target_warnings(strix_mtp_projection_component_fixtures)
        add_library(strix_qsa_selection_diagnostic INTERFACE)
        target_link_libraries(strix_qsa_selection_diagnostic INTERFACE strix_hip_runtime)
        add_executable(strix-hip-qsa-selection-diagnostic-tests tests/hip_qsa_selection_diagnostic_tests.hip)
        target_link_libraries(strix-hip-qsa-selection-diagnostic-tests PRIVATE strix_qsa_selection_diagnostic)
        set_target_properties(strix-hip-qsa-selection-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        strix_target_warnings(strix-hip-qsa-selection-diagnostic-tests)
        add_test(NAME strix-hip-qsa-selection-diagnostic-tests COMMAND strix-hip-qsa-selection-diagnostic-tests)
    endif()
    set_source_files_properties(src/hip_integer_projection.hip src/hip_integer_projection_runtime.hip
        PROPERTIES COMPILE_OPTIONS "${STRIX_HIP_WAVE32_FLAG};-ffp-contract=off")

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
        apps/hip_decode_token_entry.cpp
    )
    # This embeds DECLARED configuration only. Heterogeneous per-source HIP
    # argv is not represented completely, so live flags SHA stays unknown.
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" STRIX_DECLARED_CMAKE_SOURCE_SHA)
    string(HEX "cmake=${STRIX_DECLARED_CMAKE_SOURCE_SHA};cxx=${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION}/${CMAKE_CXX_COMPILER};hip=${CMAKE_HIP_COMPILER_ID}/${CMAKE_HIP_COMPILER_VERSION}/${CMAKE_HIP_COMPILER};build=${CMAKE_BUILD_TYPE};cxxflags=${CMAKE_CXX_FLAGS};hipflags=${CMAKE_HIP_FLAGS};debug=${CMAKE_HIP_FLAGS_DEBUG};release=${CMAKE_HIP_FLAGS_RELEASE};reldeb=${CMAKE_HIP_FLAGS_RELWITHDEBINFO};minsize=${CMAKE_HIP_FLAGS_MINSIZEREL};per-source=unresolved" STRIX_DECLARED_BUILD_CONTEXT_HEX)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(apps/offline_tuning_build_context.hpp.in
        "${CMAKE_CURRENT_BINARY_DIR}/generated/offline_tuning_build_context.hpp" @ONLY)
    strix_target_warnings(
        strix-hip-decode-token
    )
    if(STRIX_CHECK_KERNEL_SCRATCH)
        strix_check_kernel_scratch(strix-hip-decode-token)
    endif()
    # Shared Channel instantiation of the TP slot gathers (src/xport/tp_gather*.hpp): the pair gather link's channel route,
    # bound by every family's exact TP decode (qwen35 dense, hy4).
    add_library(strix_tp_xport STATIC src/xport/tp_gather.hip)
    set_target_properties(strix_tp_xport PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_link_libraries(strix_tp_xport PUBLIC strix_dist)
    target_include_directories(strix_tp_xport PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    if(STRIX_FAMILY_QWEN35_HIP)
        # HipEngine (window engine: forward_rows + commit(k)) over the shared HIP cores; both apps bind it.
        add_library(strix_qwen35_hip_engine STATIC src/qwen35/hip/engine.hip src/qwen35/config.cpp src/qwen35/family.cpp)
        set_target_properties(strix_qwen35_hip_engine PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        # Floating-point options of the EXACT HIP translation units (no contraction). The engine and the frozen v0 of the
        # decode attention tests use this one variable, so a test compares against the bits the engine build computes.
        set(STRIX_HIP_EXACT_FP_OPTIONS "-ffp-contract=off")
        set_source_files_properties(src/qwen35/hip/engine.hip PROPERTIES COMPILE_OPTIONS "${STRIX_HIP_EXACT_FP_OPTIONS}")
        target_link_libraries(strix_qwen35_hip_engine PUBLIC strix_hip_runtime strix_core)
        target_include_directories(strix_qwen35_hip_engine PRIVATE src)  # attn_core headers include "attn_core/..."
        add_executable(strix-hip-qwen35-decode apps/hip_qwen35_decode_main.hip)
        set_target_properties(strix-hip-qwen35-decode PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qwen35-decode PRIVATE strix_qwen35_hip_engine strix_spec strix_dist strix_device strix_tp_xport)  # --tp-rank: pair link, topology rates
        # N-rank layer split over the topology declaration + shared planner + shared layer pipeline (one rank = the
        # decode app's path); --self-cut runs the one-process cut test of the engine's range hooks.
        add_executable(strix-hip-qwen35-pair apps/hip_qwen35_pair_main.hip)
        set_target_properties(strix-hip-qwen35-pair PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qwen35-pair PRIVATE strix_qwen35_hip_engine strix_device)
        target_include_directories(strix-hip-qwen35-pair PRIVATE src)
        add_executable(strix-hip-qwen35-spec apps/hip_qwen35_spec_main.hip)
        set_target_properties(strix-hip-qwen35-spec PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qwen35-spec PRIVATE strix_qwen35_hip_engine strix_spec)
        # Measurement probe: IU4 decode projection split by output columns (run_cols) - bits vs the full launch, us per share.
        add_executable(strix-hip-dense-iu4-cols-probe tools/hip_dense_iu4_cols_probe.hip)
        set_target_properties(strix-hip-dense-iu4-cols-probe PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-dense-iu4-cols-probe PRIVATE src include)
        target_link_libraries(strix-hip-dense-iu4-cols-probe PRIVATE strix_hip_runtime strix_core)
        if(BUILD_TESTING)
            # Shared FAST policy dense-iu4a16-f16-v1 (IU4 wide prefill windows): decode bits + GEMM error
            add_executable(strix-hip-f16-dense-iu4-tests tests/hip_f16_dense_iu4_tests.hip)
            set_target_properties(strix-hip-f16-dense-iu4-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_link_libraries(strix-hip-f16-dense-iu4-tests PRIVATE strix_hip_runtime strix_core)
            target_include_directories(strix-hip-f16-dense-iu4-tests PRIVATE src)
            add_test(NAME strix-hip-f16-dense-iu4-tests COMMAND strix-hip-f16-dense-iu4-tests)
            add_executable(strix-hip-iu4a4-dense-tests tests/hip_iu4a4_dense_tests.hip)
            set_target_properties(strix-hip-iu4a4-dense-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_link_libraries(strix-hip-iu4a4-dense-tests PRIVATE strix_hip_runtime strix_core)
            target_include_directories(strix-hip-iu4a4-dense-tests PRIVATE src)
            add_test(NAME strix-hip-iu4a4-dense-tests COMMAND strix-hip-iu4a4-dense-tests)
            # Shared FAST wide-window attention (flash over the dense_decode BF16 cache) vs the decode form
            add_executable(strix-hip-dense-decode-flash-tests tests/hip_dense_decode_flash_tests.hip)
            set_target_properties(strix-hip-dense-decode-flash-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_include_directories(strix-hip-dense-decode-flash-tests PRIVATE src include)
            target_compile_options(strix-hip-dense-decode-flash-tests PRIVATE ${STRIX_HIP_EXACT_FP_OPTIONS})
            add_test(NAME strix-hip-dense-decode-flash-tests COMMAND strix-hip-dense-decode-flash-tests)
            # Shared exact decode attention: every attend form == main's kernel as the engine builds it, bit for bit. The
            # frozen v0 TU always takes the engine's options; the forms are built with them and with the HIP default.
            set_source_files_properties(tests/hip_dense_decode_rows_v0.hip PROPERTIES COMPILE_OPTIONS "${STRIX_HIP_EXACT_FP_OPTIONS}")
            foreach(rows_test strix-hip-dense-decode-rows-tests strix-hip-dense-decode-rows-default-tests)
                add_executable(${rows_test} tests/hip_dense_decode_rows_tests.hip tests/hip_dense_decode_rows_v0.hip)
                set_target_properties(${rows_test} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                target_include_directories(${rows_test} PRIVATE src include)
                add_test(NAME ${rows_test} COMMAND ${rows_test})
            endforeach()
            target_compile_options(strix-hip-dense-decode-rows-tests PRIVATE ${STRIX_HIP_EXACT_FP_OPTIONS})
        endif()
        if(STRIX_CHECK_KERNEL_SCRATCH)
            strix_check_kernel_scratch(strix-hip-qwen35-decode)
            strix_check_kernel_scratch(strix-hip-qwen35-spec)
        endif()
    endif()
    if(BUILD_TESTING)
        # Same actual model/PLE/asset/replay setup as the decoder. External
        # references and the serial-oracle prefix diagnostic have separate
        # admission; neither establishes model fidelity or performance alone.
        add_executable(strix-hip-mtp-model-qualification
            apps/hip_decode_token_main.cpp tests/hip_mtp_model_qualification_main.cpp)
        target_compile_definitions(strix-hip-mtp-model-qualification PRIVATE
            STRIX_DECODE_ENTRY=strix_hip_decode_token_entry STRIX_MTP_MODEL_QUALIFICATION_RUNNER=1)
        target_include_directories(strix-hip-mtp-model-qualification PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
        # Same link closure as strix-hip-decode-token: the runner compiles the same main.
        target_link_libraries(strix-hip-mtp-model-qualification PRIVATE
            strix_hip_runtime strix_mtp_projection_component_fixtures strix_serve strix_serve_json strix_assets
            strix_qwen_model_description strix_device_profile strix_qwen_http_session_owner)
        # Large state snapshots are diagnostic-only. Prefer the system SHA-256
        # provider when available; the portable digest remains the fallback.
        find_package(OpenSSL QUIET COMPONENTS Crypto)
        if(TARGET OpenSSL::Crypto)
            target_link_libraries(strix-hip-mtp-model-qualification PRIVATE OpenSSL::Crypto)
            target_compile_definitions(strix-hip-mtp-model-qualification PRIVATE STRIX_PREFIX_GATE_OPENSSL_SHA256=1)
        endif()
        target_compile_options(strix-hip-mtp-model-qualification PRIVATE
            "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
        strix_target_warnings(strix-hip-mtp-model-qualification)
        add_executable(strix-hip-mtp-qualification-reference-tests tests/hip_mtp_qualification_reference_tests.hip)
        target_link_libraries(strix-hip-mtp-qualification-reference-tests PRIVATE
            strix_hip_runtime strix_mtp_projection_component_fixtures strix_serve strix_assets)
        set_target_properties(strix-hip-mtp-qualification-reference-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        strix_target_warnings(strix-hip-mtp-qualification-reference-tests)
        add_test(NAME strix-hip-mtp-qualification-reference-tests COMMAND strix-hip-mtp-qualification-reference-tests)
    endif()
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
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
        # Raw embedding row gather (host ids / device id, FP32 / BF16 out) vs the shared host dequantizer.
        add_executable(strix-hip-embedding-rows-tests tests/hip_embedding_rows_tests.hip)
        set_target_properties(strix-hip-embedding-rows-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-embedding-rows-tests PRIVATE strix_hip_runtime strix_weight_archive)
        strix_target_warnings(strix-hip-embedding-rows-tests)
        add_test(NAME strix-hip-embedding-rows-tests COMMAND strix-hip-embedding-rows-tests)
        set_tests_properties(strix-hip-embedding-rows-tests PROPERTIES SKIP_RETURN_CODE 77)
        # Dense rows (head family F32 / BF16 weights), shared RoPE and gated elementwise ops vs their host references.
        add_executable(strix-hip-ops-dense-rows-tests tests/hip_ops_dense_rows_tests.hip)
        set_target_properties(strix-hip-ops-dense-rows-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-ops-dense-rows-tests PRIVATE include src)
        target_compile_options(strix-hip-ops-dense-rows-tests PRIVATE -fno-fast-math -ffp-contract=off)
        target_link_libraries(strix-hip-ops-dense-rows-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-ops-dense-rows-tests)
        add_test(NAME strix-hip-ops-dense-rows-tests COMMAND strix-hip-ops-dense-rows-tests)
        set_tests_properties(strix-hip-ops-dense-rows-tests PROPERTIES SKIP_RETURN_CODE 77)
        # Wave form of launch_hip_dense_rows rows 1 == block form, batched rmsnorm_row == loop form (bitwise); timing printed.
        add_executable(strix-hip-dense-rows-wave-tests tests/hip_dense_rows_wave_tests.hip)
        set_target_properties(strix-hip-dense-rows-wave-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-dense-rows-wave-tests PRIVATE include src)
        target_link_libraries(strix-hip-dense-rows-wave-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-dense-rows-wave-tests)
        add_test(NAME strix-hip-dense-rows-wave-tests COMMAND strix-hip-dense-rows-wave-tests)
        set_tests_properties(strix-hip-dense-rows-wave-tests PROPERTIES SKIP_RETURN_CODE 77)
        # Rows tile and grouped rows of launch_hip_dense_rows == one single-row launch per row / group (bitwise); timing printed.
        add_executable(strix-hip-dense-rows-tile-tests tests/hip_dense_rows_tile_tests.hip)
        set_target_properties(strix-hip-dense-rows-tile-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-dense-rows-tile-tests PRIVATE include src)
        target_link_libraries(strix-hip-dense-rows-tile-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-dense-rows-tile-tests)
        add_test(NAME strix-hip-dense-rows-tile-tests COMMAND strix-hip-dense-rows-tile-tests)
        set_tests_properties(strix-hip-dense-rows-tile-tests PROPERTIES SKIP_RETURN_CODE 77)
        # Shared cache-row writer (attn_core/cache_rows.hpp) vs the host reference, test geometries.
        add_executable(strix-hip-cache-rows-tests tests/hip_cache_rows_tests.hip)
        set_target_properties(strix-hip-cache-rows-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-cache-rows-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
        target_compile_options(strix-hip-cache-rows-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off)
        strix_target_warnings(strix-hip-cache-rows-tests)
        add_test(NAME strix-hip-cache-rows-tests COMMAND strix-hip-cache-rows-tests)
        # Decode lanes of the prepared BF16 flash == one append per lane, bitwise (outputs, caches, failure word).
        add_executable(strix-hip-prepared-bf16-lanes-tests tests/hip_prepared_bf16_lanes_tests.hip)
        set_target_properties(strix-hip-prepared-bf16-lanes-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-prepared-bf16-lanes-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
        target_compile_options(strix-hip-prepared-bf16-lanes-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off)
        strix_target_warnings(strix-hip-prepared-bf16-lanes-tests)
        add_test(NAME strix-hip-prepared-bf16-lanes-tests COMMAND strix-hip-prepared-bf16-lanes-tests)
        set_tests_properties(strix-hip-prepared-bf16-lanes-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-serial-graph-tests tests/hip_serial_graph_tests.hip)
        set_target_properties(strix-hip-serial-graph-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-serial-graph-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-serial-graph-tests)
        add_test(NAME strix-hip-serial-graph-tests COMMAND strix-hip-serial-graph-tests)
        set_tests_properties(strix-hip-serial-graph-tests PROPERTIES SKIP_RETURN_CODE 77)
        set_target_properties(strix-hip-integer-projection-runtime-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
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
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-graph-attention-state-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-graph-attention-state-tests)
        add_test(NAME strix-hip-graph-attention-state-tests COMMAND strix-hip-graph-attention-state-tests)
        set_target_properties(strix-hip-graph-inputs-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-graph-inputs-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-graph-inputs-tests)
        add_test(NAME strix-hip-graph-inputs-tests COMMAND strix-hip-graph-inputs-tests)
        set_target_properties(
            strix-hip-embedding-launcher-tests
            PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
        set_target_properties(strix-hip-hc-projection-next-tile-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-hc-projection-next-tile-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-hc-projection-next-tile-tests)
        add_test(NAME strix-hip-hc-projection-next-tile-tests COMMAND strix-hip-hc-projection-next-tile-tests)
        add_executable(strix-hip-lane-chain-hyper-output-tests tests/hip_lane_chain_hyper_output_tests.hip)
        set_target_properties(strix-hip-lane-chain-hyper-output-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-lane-chain-hyper-output-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-lane-chain-hyper-output-tests)
        add_test(NAME strix-hip-lane-chain-hyper-output-tests COMMAND strix-hip-lane-chain-hyper-output-tests)
        add_executable(strix-hip-hc-wide-rows-tests tests/hip_hc_wide_rows_tests.hip)
        set_target_properties(strix-hip-hc-wide-rows-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-hc-wide-rows-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-hc-wide-rows-tests)
        add_test(NAME strix-hip-hc-wide-rows-tests COMMAND strix-hip-hc-wide-rows-tests)
        add_test(NAME strix-hip-hc-recompute-norm-tests COMMAND strix-hip-hc-wide-rows-tests --recompute-norm)
        set_tests_properties(strix-hip-hc-recompute-norm-tests PROPERTIES
            ENVIRONMENT "STRIX_QPF2K_HC_CHECK=0;STRIX_QPF2K_HC_UPCOPY=0;STRIX_HC_WIDE2_OUTPUT_ROWS=32")
        add_test(NAME strix-hip-hc-recompute-mirror-tests COMMAND strix-hip-hc-wide-rows-tests --recompute-norm)
        set_tests_properties(strix-hip-hc-recompute-mirror-tests PROPERTIES
            ENVIRONMENT "STRIX_QPF2K_HC_CHECK=0;STRIX_QPF2K_HC_UPCOPY=0;STRIX_HC_WIDE2_OUTPUT_ROWS=32;STRIX_HC_RECOMPUTE_STREAM_STRIDE=20480")
        add_test(NAME strix-hip-hc-out16-tests COMMAND strix-hip-hc-wide-rows-tests)
        set_tests_properties(strix-hip-hc-out16-tests PROPERTIES ENVIRONMENT "STRIX_PREFILL_EXACT_FP16=0;STRIX_PREFILL_HC_OUTPUT_WMMA=1;STRIX_HCW_OUTPUT=1;STRIX_HC_OUT16_FIXTURES_ONLY=1;STRIX_QPF2K_HC=1;STRIX_QPF2K_HC_LOWBAND=3")
        add_executable(strix-hip-parallel-hyper-projection-tests tests/hip_parallel_hyper_projection_tests.hip)
        set_target_properties(strix-hip-parallel-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-parallel-hyper-projection-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-parallel-hyper-projection-tests)
        add_test(NAME strix-hip-parallel-hyper-projection-tests COMMAND strix-hip-parallel-hyper-projection-tests)
        add_executable(strix-hip-splitk-hyper-integration-tests tests/hip_splitk_hyper_integration_tests.cpp)
        target_link_libraries(strix-hip-splitk-hyper-integration-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-splitk-hyper-integration-tests)
        add_test(NAME strix-hip-splitk-hyper-integration-tests COMMAND strix-hip-splitk-hyper-integration-tests)
        add_executable(strix-hip-hc-deferred-norm-tests tests/hip_hc_deferred_norm_tests.hip)
        target_link_libraries(strix-hip-hc-deferred-norm-tests PRIVATE strix_hip_runtime strix_q4k_a8_contract)
        set_target_properties(strix-hip-hc-deferred-norm-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-hip-hc-deferred-norm-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-hc-deferred-norm-tests)
        add_test(NAME strix-hip-hc-deferred-norm-tests COMMAND strix-hip-hc-deferred-norm-tests)

        add_executable(strix-hip-lane-chain-hyper-projection-tests tests/hip_lane_chain_hyper_projection_tests.hip)
        set_target_properties(strix-hip-lane-chain-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-lane-chain-hyper-projection-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-lane-chain-hyper-projection-tests)
        add_test(NAME strix-hip-lane-chain-hyper-projection-tests COMMAND strix-hip-lane-chain-hyper-projection-tests)
        add_library(strix_hc_repacked_diagnostic STATIC src/hip_hc_repacked_diagnostic.hip)
        target_include_directories(strix_hc_repacked_diagnostic PUBLIC include)
        set_target_properties(strix_hc_repacked_diagnostic PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix_hc_repacked_diagnostic PUBLIC strix_core "${STRIX_HIP_RUNTIME_LIBRARY}")
        strix_target_warnings(strix_hc_repacked_diagnostic)
        add_executable(strix-hip-repacked-hyper-projection-tests tests/hip_repacked_hyper_projection_tests.hip)
        set_target_properties(strix-hip-repacked-hyper-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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

        if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
            if(UNIX)
                add_executable(strix-http-batch-session-tests tests/hip_http_batch_session_tests.cpp)
                target_compile_definitions(strix-http-batch-session-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
                target_link_libraries(strix-http-batch-session-tests PRIVATE strix_hip_runtime strix_serve)
                strix_target_warnings(strix-http-batch-session-tests)
                add_test(NAME strix-http-batch-session-tests COMMAND strix-http-batch-session-tests)
                set_tests_properties(strix-http-batch-session-tests PROPERTIES LABELS "cpu" RUN_SERIAL TRUE TIMEOUT 30)
            endif()
            add_executable(strix-qwen-yarn-host-compile-tests tests/qwen_yarn_host_compile_tests.cpp)
            target_compile_definitions(strix-qwen-yarn-host-compile-tests PRIVATE STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME=1)
            target_link_libraries(strix-qwen-yarn-host-compile-tests PRIVATE strix_hip_runtime strix_serve)
            strix_target_warnings(strix-qwen-yarn-host-compile-tests)
            add_test(NAME strix-qwen-yarn-host-compile-tests COMMAND strix-qwen-yarn-host-compile-tests)
            set_tests_properties(strix-qwen-yarn-host-compile-tests PROPERTIES LABELS "cpu" RUN_SERIAL TRUE TIMEOUT 180)
        endif()

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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-shared-known-window-rows-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-shared-known-window-rows-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-shared-known-window-rows-tests)
        foreach(rows 4 8)
            foreach(prefetch 0 1)
                add_test(NAME strix-hip-shared-known-window-rows-${rows}-prefetch-${prefetch}
                    COMMAND strix-hip-shared-known-window-rows-tests ${rows} ${prefetch})
            endforeach()
        endforeach()

        add_test(NAME strix-hip-shared-pre-fixtures COMMAND strix-hip-shared-known-window-rows-tests)
        set_tests_properties(strix-hip-shared-pre-fixtures PROPERTIES ENVIRONMENT "STRIX_SHARED_PRE_FIXTURES_ONLY=1")

        add_executable(strix-hip-gdn-retained-window-tests tests/hip_gdn_retained_window_tests.hip)
        set_target_properties(strix-hip-gdn-retained-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-gdn-retained-window-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-gdn-retained-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-retained-window-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-retained-window-tests)
        add_test(NAME strix-hip-gdn-retained-window-tests COMMAND strix-hip-gdn-retained-window-tests)

        # Rollback replay of the qwen35 HIP engine: in-place replay of k rows (chunked cols core + rows tail) == the
        # out-of-place verify window of k rows (rows_kernel), state / outputs / conv history bitwise, k = 1..16.
        add_executable(strix-hip-gdn-dims-replay-tests tests/hip_gdn_dims_replay_tests.hip)
        set_target_properties(strix-hip-gdn-dims-replay-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-gdn-dims-replay-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-gdn-dims-replay-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-dims-replay-tests)
        add_test(NAME strix-hip-gdn-dims-replay-tests COMMAND strix-hip-gdn-dims-replay-tests)

        add_executable(strix-hip-gdn-deferred-window-tests tests/hip_gdn_deferred_window_tests.hip)
        set_target_properties(strix-hip-gdn-deferred-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-gdn-deferred-window-tests PRIVATE strix_hip_runtime)
        target_include_directories(strix-hip-gdn-deferred-window-tests PRIVATE src)
        target_compile_definitions(strix-hip-gdn-deferred-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-deferred-window-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-deferred-window-tests)
        add_test(NAME strix-hip-gdn-deferred-window-tests COMMAND strix-hip-gdn-deferred-window-tests)

        add_executable(strix-hip-gdn-wide-projection-tests tests/hip_gdn_wide_projection_tests.hip)
        set_target_properties(strix-hip-gdn-wide-projection-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-gdn-wide-projection-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-gdn-wide-projection-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gdn-wide-projection-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-wide-projection-tests)
        add_test(NAME strix-hip-gdn-wide-projection-tests COMMAND strix-hip-gdn-wide-projection-tests)

        add_executable(strix-hip-gdn-fp16-chunk-tests tests/hip_gdn_fp16_chunk_tests.hip)
        set_target_properties(strix-hip-gdn-fp16-chunk-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-hip-gdn-fp16-chunk-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-gdn-fp16-chunk-tests)
        add_test(NAME strix-hip-gdn-fp16-chunk-tests COMMAND strix-hip-gdn-fp16-chunk-tests)
        set_tests_properties(strix-hip-gdn-fp16-chunk-tests PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(strix-hip-q8-rows-projection-tests tests/hip_q8_rows_projection_tests.hip)
        set_target_properties(strix-hip-q8-rows-projection-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-q8-rows-projection-tests PRIVATE strix_hip_runtime)
        target_include_directories(strix-hip-q8-rows-projection-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_compile_definitions(strix-hip-q8-rows-projection-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-q8-rows-projection-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-q8-rows-projection-tests)
        add_test(NAME strix-hip-q8-rows-projection-tests COMMAND strix-hip-q8-rows-projection-tests)
        # The same byte gate with the stepped LDS windows forced for every window they can hold.
        add_test(NAME strix-hip-q8-rows-projection-stepped-tests COMMAND strix-hip-q8-rows-projection-tests)
        set_tests_properties(strix-hip-q8-rows-projection-stepped-tests PROPERTIES ENVIRONMENT "STRIX_Q8_ROWS_STEPPED=1")

        # Stock Q8_0 sites: LinearRowsF32<2..8> against LinearF32 per row, default kernels and project_rows_kernel only.
        add_executable(strix-hip-q8-rows-stock-tests tests/hip_q8_rows_stock_tests.hip)
        set_target_properties(strix-hip-q8-rows-stock-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-q8-rows-stock-tests PRIVATE strix_hip_runtime)
        target_include_directories(strix-hip-q8-rows-stock-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_compile_options(strix-hip-q8-rows-stock-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-q8-rows-stock-tests)
        add_test(NAME strix-hip-q8-rows-stock-tests COMMAND strix-hip-q8-rows-stock-tests)
        add_test(NAME strix-hip-q8-rows-stock-direct-tests COMMAND strix-hip-q8-rows-stock-tests)
        set_tests_properties(strix-hip-q8-rows-stock-direct-tests PROPERTIES ENVIRONMENT "STRIX_Q8_ROWS_VOPD=0")

        # External fixtures pin compressed bits and identify the materialized
        # reference separately. Never register a GPU invocation with CTest.
        add_executable(strix-hip-dense-trellis-tests tests/hip_dense_trellis_tests.hip)
        set_target_properties(strix-hip-dense-trellis-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-dense-trellis-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-dense-trellis-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-dense-trellis-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-dense-trellis-tests)

        include(cmake/StrixDenseA8WmmaTest.cmake)
        # q8_rows head rows: one launch_heads == H per-head launches (bitwise), MLA absorb geometries; --bench times them.
        add_executable(strix-hip-q8-rows-heads-tests tests/hip_q8_rows_heads_tests.hip)
        set_target_properties(strix-hip-q8-rows-heads-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-hip-q8-rows-heads-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-q8-rows-heads-tests PRIVATE strix_hip_runtime strix_core)
        strix_target_warnings(strix-hip-q8-rows-heads-tests)
        add_test(NAME strix-hip-q8-rows-heads-tests COMMAND strix-hip-q8-rows-heads-tests)
        # Decode FFN fusions: GLU projection group (q8_rows paired store) and combine with rounded expert outputs ==
        # the separate launches, bitwise, at the Hy4 shapes; --bench times them.
        add_executable(strix-hip-glu-pair-combine-tests tests/hip_glu_pair_combine_tests.hip)
        set_target_properties(strix-hip-glu-pair-combine-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_include_directories(strix-hip-glu-pair-combine-tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
        target_compile_options(strix-hip-glu-pair-combine-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-glu-pair-combine-tests PRIVATE strix_hip_runtime strix_core)
        strix_target_warnings(strix-hip-glu-pair-combine-tests)
        add_test(NAME strix-hip-glu-pair-combine-tests COMMAND strix-hip-glu-pair-combine-tests)
        add_executable(strix-hip-gsq-dense-a8-tests tests/hip_gsq_dense_a8_tests.hip)
        set_target_properties(strix-hip-gsq-dense-a8-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-hip-gsq-dense-a8-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-gsq-dense-a8-tests PRIVATE strix_hip_runtime strix_core)
        strix_target_warnings(strix-hip-gsq-dense-a8-tests)
        add_test(NAME strix-hip-gsq-dense-a8-tests COMMAND strix-hip-gsq-dense-a8-tests)
        # The engine's own GSQ typed site kernels (compiled in strix_hip_runtime with its flags) through the launchers'
        # diagnostic probes: every row alone == batched T = 1..8 == the host contract.
        add_executable(strix-hip-gsq-typed-sites-tests tests/hip_gsq_typed_sites_tests.hip)
        set_target_properties(strix-hip-gsq-typed-sites-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_definitions(strix-hip-gsq-typed-sites-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-gsq-typed-sites-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-gsq-typed-sites-tests PRIVATE strix_hip_runtime strix_core)
        strix_target_warnings(strix-hip-gsq-typed-sites-tests)
        add_test(NAME strix-hip-gsq-typed-sites-tests COMMAND strix-hip-gsq-typed-sites-tests)
        add_test(NAME strix-hip-prefill-dense-schedule-tests COMMAND ${CMAKE_COMMAND} -E env
            STRIX_INTPREFILL_GATES=1 $<TARGET_FILE:strix-hip-dense-a8-wmma-tests>)
        # Whole-T register-blocked dense FP16 projections (STRIX_QPF2K_DENSE=1) == launch_dense_f16, bitwise.
        add_executable(strix-hip-dense-f16-tiled-tests tests/hip_dense_f16_tiled_tests.hip)
        set_target_properties(strix-hip-dense-f16-tiled-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-hip-dense-f16-tiled-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        target_link_libraries(strix-hip-dense-f16-tiled-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-dense-f16-tiled-tests)
        add_test(NAME strix-hip-dense-f16-tiled-tests COMMAND strix-hip-dense-f16-tiled-tests)

        if(STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME)
            add_executable(strix-hip-moe-q4k-a8-known-rows-tests tests/hip_moe_q4k_a8_known_rows_tests.hip)
            set_target_properties(strix-hip-moe-q4k-a8-known-rows-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_link_libraries(strix-hip-moe-q4k-a8-known-rows-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-moe-q4k-a8-known-rows-tests PRIVATE
                ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-q4k-a8-known-rows-tests)
            foreach(rows 2 4 8)
                add_test(NAME strix-hip-moe-q4k-a8-known-rows-${rows}
                    COMMAND strix-hip-moe-q4k-a8-known-rows-tests ${rows})
            endforeach()
            add_executable(strix-hip-moe-known-rows-tests tests/hip_moe_known_rows_tests.hip)
            set_target_properties(strix-hip-moe-known-rows-tests PROPERTIES
                HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_link_libraries(strix-hip-moe-known-rows-tests PRIVATE strix_hip_runtime)
            target_compile_options(strix-hip-moe-known-rows-tests PRIVATE
                ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
            strix_target_warnings(strix-hip-moe-known-rows-tests)
            foreach(rows 2 4 8)
                foreach(batched 0 1)
                    add_test(NAME strix-hip-moe-known-rows-${rows}-batched-${batched}
                        COMMAND strix-hip-moe-known-rows-tests ${rows} ${batched})
                endforeach()
                # A8 profile at canonical geometry: Q5_1, Q8_0 (layer 4), Q5_K+Q8_0 (layer 2).
                foreach(profile a8q51 a8q80 a8q5k)
                    add_test(NAME strix-hip-moe-known-rows-${rows}-${profile}
                        COMMAND strix-hip-moe-known-rows-tests ${rows} 1 ${profile})
                endforeach()
            endforeach()
            # Cohort unions of several sessions' 8-row windows (A8 profile).
            foreach(rows 12 16 24 32)
                foreach(profile a8q51 a8q80 a8q5k)
                    add_test(NAME strix-hip-moe-known-rows-${rows}-${profile}
                        COMMAND strix-hip-moe-known-rows-tests ${rows} 1 ${profile})
                endforeach()
            endforeach()
        endif()

        add_executable(strix-hip-event-queue-trace-tests tests/hip_event_queue_trace_tests.hip)
        set_target_properties(strix-hip-event-queue-trace-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-event-queue-trace-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-event-queue-trace-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG})
        strix_target_warnings(strix-hip-event-queue-trace-tests)
        add_test(NAME hip-event-queue-trace COMMAND strix-hip-event-queue-trace-tests)
        add_executable(strix-hip-known-window-tests tests/hip_known_window_tests.hip)
        set_target_properties(strix-hip-known-window-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-known-window-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-known-window-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-known-window-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-known-window-tests)
        add_test(NAME strix-hip-known-window-tests COMMAND strix-hip-known-window-tests)
        add_test(NAME strix-hip-qwen-router-bf16-tests COMMAND strix-hip-known-window-tests --router-bf16)
        set_tests_properties(strix-hip-qwen-router-bf16-tests PROPERTIES LABELS "gpu" RUN_SERIAL TRUE
            ENVIRONMENT "STRIX_Q6P=1;STRIX_QWEN_ROUTER_BF16=1")
        add_executable(strix-hip-token-sampling-device-tests tests/hip_token_sampling_device_tests.hip)
        set_target_properties(strix-hip-token-sampling-device-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-token-sampling-device-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-token-sampling-device-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG})
        strix_target_warnings(strix-hip-token-sampling-device-tests)
        add_test(NAME strix-hip-token-sampling-device-tests COMMAND strix-hip-token-sampling-device-tests)
        add_test(NAME strix-hip-known-window-hc-persistent-tests COMMAND strix-hip-known-window-tests --hc-rows)
        set_tests_properties(strix-hip-known-window-hc-persistent-tests PROPERTIES ENVIRONMENT "STRIX_HC_PERSISTENT=1;STRIX_HC_PERSISTENT_INJECT=1")
        add_test(NAME strix-hip-known-window-fold-tests COMMAND strix-hip-known-window-tests --window-folds)
        # STRIX_DISPATCH_FUSE=1 alone: persistent HC at 2 and 4 rows, the inject + norm fold at 8 rows.
        add_test(NAME strix-hip-known-window-hc-fuse-tests COMMAND strix-hip-known-window-tests --hc-rows)
        set_tests_properties(strix-hip-known-window-hc-fuse-tests PROPERTIES ENVIRONMENT "STRIX_DISPATCH_FUSE=1")
        add_test(NAME strix-hip-known-window-fold-fuse-tests COMMAND strix-hip-known-window-tests --window-folds)
        set_tests_properties(strix-hip-known-window-fold-fuse-tests PROPERTIES ENVIRONMENT "STRIX_DISPATCH_FUSE=1")
        # FP32 streams: the folds and the persistent HC against the separate launches.
        add_test(NAME strix-hip-known-window-fp32-fold-tests COMMAND strix-hip-known-window-tests --folds-fp32)
        set_tests_properties(strix-hip-known-window-fp32-fold-tests PROPERTIES ENVIRONMENT "STRIX_HC_PERSISTENT=1;STRIX_HC_PERSISTENT_INJECT=1")
        add_executable(strix-hip-qsa-committed-pool-tests tests/hip_qsa_committed_pool_tests.hip)
        set_target_properties(strix-hip-qsa-committed-pool-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qsa-committed-pool-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-qsa-committed-pool-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-qsa-committed-pool-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off)
        strix_target_warnings(strix-hip-qsa-committed-pool-tests)
        add_test(NAME strix-hip-qsa-committed-pool-tests COMMAND strix-hip-qsa-committed-pool-tests)
        foreach(yarn_factor 2 4)
            add_test(NAME strix-hip-qsa-committed-pool-yarn${yarn_factor}-tests
                COMMAND strix-hip-qsa-committed-pool-tests --yarn-factor ${yarn_factor})
            set_tests_properties(strix-hip-qsa-committed-pool-yarn${yarn_factor}-tests
                PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
        endforeach()
        set_tests_properties(strix-hip-qsa-committed-pool-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-compact-stage-images-tests tests/hip_compact_stage_images_tests.hip)
        set_target_properties(strix-hip-compact-stage-images-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-compact-stage-images-tests PRIVATE strix_hip_runtime)
        target_compile_options(strix-hip-compact-stage-images-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-compact-stage-images-tests)
        add_test(NAME strix-hip-compact-stage-images-tests COMMAND strix-hip-compact-stage-images-tests)
        set_tests_properties(strix-hip-compact-stage-images-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-qsa-decode-pool-bf16-tests tests/qsa_decode_pool_bf16.hip)
        set_target_properties(strix-hip-qsa-decode-pool-bf16-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qsa-decode-pool-bf16-tests PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_compile_options(strix-hip-qsa-decode-pool-bf16-tests PRIVATE
            ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-qsa-decode-pool-bf16-tests)
        add_test(NAME strix-hip-qsa-decode-pool-bf16-tests COMMAND strix-hip-qsa-decode-pool-bf16-tests)
        set_tests_properties(strix-hip-qsa-decode-pool-bf16-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-qsa-block-select-tests tests/hip_qsa_block_select_tests.hip)
        set_target_properties(strix-hip-qsa-block-select-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qsa-block-select-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-qsa-block-select-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-qsa-block-select-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG})
        strix_target_warnings(strix-hip-qsa-block-select-tests)
        add_test(NAME strix-hip-qsa-block-select-tests COMMAND strix-hip-qsa-block-select-tests)
        set_tests_properties(strix-hip-qsa-block-select-tests PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(strix-hip-attention-kv-int8-tests tests/hip_attention_kv_int8_tests.hip)
        set_target_properties(strix-hip-attention-kv-int8-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-attention-kv-int8-tests PRIVATE strix_hip_runtime)
        target_compile_definitions(strix-hip-attention-kv-int8-tests PRIVATE STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC=1)
        target_compile_options(strix-hip-attention-kv-int8-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math)
        strix_target_warnings(strix-hip-attention-kv-int8-tests)
        add_test(NAME strix-hip-attention-kv-int8-tests COMMAND strix-hip-attention-kv-int8-tests)
        set_tests_properties(strix-hip-attention-kv-int8-tests PROPERTIES SKIP_RETURN_CODE 77)

        # Full-model qualification is explicit and requires a model and teacher;
        # it is intentionally absent from the automatic device regression.
        add_executable(strix-hip-known-window-model-qualification tests/hip_known_window_model_qualification.hip)
        set_target_properties(strix-hip-known-window-model-qualification PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-known-window-model-qualification PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-known-window-model-qualification)

        # Explicit known-prompt/cache/continuation qualification; requires the
        # real model and ROOT-sealed inputs, so it is not an automatic CTest.
        # Explicit model-dependent serving qualification; never automatic CTest.
        add_executable(strix-hip-serving-greedy-model-qualification tests/hip_serving_greedy_backend_tests.hip)
        set_target_properties(strix-hip-serving-greedy-model-qualification PROPERTIES
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
        target_compile_definitions(strix-hip-serving-greedy-model-qualification PRIVATE STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC=1)
        target_compile_options(strix-hip-serving-greedy-model-qualification PRIVATE -fno-fast-math -ffp-contract=off)
        target_link_libraries(strix-hip-serving-greedy-model-qualification PRIVATE strix_hip_runtime Threads::Threads)
        strix_target_warnings(strix-hip-serving-greedy-model-qualification)

        add_executable(strix-hip-known-prompt-model-qualification tests/hip_known_prompt_model_qualification.hip)
        set_target_properties(strix-hip-known-prompt-model-qualification PROPERTIES
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
            add_test(NAME strix-hip-moe-a8-down-runtime-admission-tests
                COMMAND strix-hip-moe-kquant-tests 4 a8-down-runtime-admission)
            add_test(NAME strix-hip-moe-r1-glue-tail-tests COMMAND strix-hip-moe-kquant-tests 0 r1-glue-tail)
            add_test(NAME strix-hip-moe-r1-slot-merge-tail-tests COMMAND strix-hip-moe-kquant-tests 0 r1-slot-merge-tail)
            set_target_properties(strix-hip-q51-grouped-down-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_link_libraries(strix-hip-q51-grouped-down-probe-tests PRIVATE strix_hip_runtime)
            strix_target_warnings(strix-hip-q51-grouped-down-probe-tests)
            add_test(NAME strix-hip-q51-grouped-down-probe-tests COMMAND strix-hip-q51-grouped-down-probe-tests)
            add_executable(strix-hip-q51-slot-parallel-probe-tests tests/hip_q51_slot_parallel_probe_tests.hip)
            set_target_properties(strix-hip-q51-slot-parallel-probe-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
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
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-moe-kquant-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-kquant-tests)
        foreach(mode RANGE 0 7)
            add_test(NAME strix-hip-moe-kquant-${mode}-tests
                     COMMAND strix-hip-moe-kquant-tests ${mode})
        endforeach()

        add_executable(strix-hip-ggml-weight-formats-tests tests/hip_ggml_weight_formats_tests.hip)
        set_target_properties(strix-hip-ggml-weight-formats-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-ggml-weight-formats-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-ggml-weight-formats-tests)
        add_test(NAME strix-hip-ggml-weight-formats-tests COMMAND strix-hip-ggml-weight-formats-tests)

        add_executable(strix-hip-gdn-output-projection-tests tests/hip_gdn_output_projection_tests.hip)
        target_link_libraries(strix-hip-gdn-output-projection-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-gdn-output-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        strix_target_warnings(strix-hip-gdn-output-projection-tests)
        add_test(NAME strix-hip-gdn-output-projection-tests COMMAND strix-hip-gdn-output-projection-tests)

        add_executable(strix-hip-qsa-index-batch-tests tests/hip_qsa_index_batch_tests.hip)
        target_link_libraries(strix-hip-qsa-index-batch-tests PRIVATE strix_hip_runtime)
        target_include_directories(strix-hip-qsa-index-batch-tests PRIVATE src)
        set_target_properties(strix-hip-qsa-index-batch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        strix_target_warnings(strix-hip-qsa-index-batch-tests)
        add_test(NAME strix-hip-qsa-index-batch-tests COMMAND strix-hip-qsa-index-batch-tests 7)
        set_tests_properties(strix-hip-qsa-index-batch-tests PROPERTIES ENVIRONMENT "STRIX_QSA_INDEX_ROWS=1;STRIX_QPF2K_INDEX_BATCH=1;STRIX_PREFILL_INDEX_PROJ_WMMA=1;STRIX_PREFILL_EXACT_FP16=0")
        add_executable(strix-hip-qsa-index-score-wmma-tests tests/qsa_index_score_wmma.hip)
        target_include_directories(strix-hip-qsa-index-score-wmma-tests PRIVATE include src)
        set_target_properties(strix-hip-qsa-index-score-wmma-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-hip-qsa-index-score-wmma-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-qsa-index-score-wmma-tests)
        add_test(NAME strix-hip-qsa-index-score-wmma-tests COMMAND strix-hip-qsa-index-score-wmma-tests)
        set_tests_properties(strix-hip-qsa-index-score-wmma-tests PROPERTIES ENVIRONMENT "STRIX_QPF2K_SCORE_ROWS=auto")

        add_executable(strix-hip-qsa-index-band-tests tests/qsa_index_band.hip)
        target_include_directories(strix-hip-qsa-index-band-tests PRIVATE include src)
        set_target_properties(strix-hip-qsa-index-band-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-hip-qsa-index-band-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(strix-hip-qsa-index-band-tests)
        add_test(NAME strix-hip-qsa-index-band-tests COMMAND strix-hip-qsa-index-band-tests)
        set_tests_properties(strix-hip-qsa-index-band-tests PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "STRIX_QSA_INDEX_BAND=1")

        add_executable(strix-hip-qsa-output-projection-tests tests/hip_qsa_output_projection_tests.hip)
        target_compile_definitions(strix-hip-qsa-output-projection-tests PRIVATE STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC=1)
        target_link_libraries(strix-hip-qsa-output-projection-tests PRIVATE strix_hip_runtime)
        target_include_directories(strix-hip-qsa-output-projection-tests PRIVATE src)
        set_target_properties(strix-hip-qsa-output-projection-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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

        add_executable(strix-hip-gdn-decode-fuse-tests tests/hip_gdn_decode_fuse_tests.hip)
        target_link_libraries(strix-hip-gdn-decode-fuse-tests PRIVATE strix_hip_runtime)
        set_target_properties(strix-hip-gdn-decode-fuse-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        strix_target_warnings(strix-hip-gdn-decode-fuse-tests)
        add_test(NAME strix-hip-gdn-decode-fuse-tests COMMAND strix-hip-gdn-decode-fuse-tests)
        set_tests_properties(strix-hip-gdn-decode-fuse-tests PROPERTIES LABELS "gpu;exactness")

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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
        # STRIX_PLE_WINDOW_WARM: verify-window PLE rows with and without decode cache warming, bitwise.
        add_executable(strix-hip-ple-window-warm-tests tests/hip_ple_window_warm_tests.hip)
        set_target_properties(strix-hip-ple-window-warm-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-ple-window-warm-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-ple-window-warm-tests)
        add_test(NAME strix-hip-ple-window-warm-tests COMMAND strix-hip-ple-window-warm-tests)

        add_executable(strix-hip-qsa-input-prefetch-tests tests/hip_qsa_input_prefetch_tests.hip)
        set_target_properties(strix-hip-qsa-input-prefetch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qsa-input-prefetch-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-input-prefetch-tests)
        add_test(NAME strix-hip-qsa-input-prefetch-tests COMMAND strix-hip-qsa-input-prefetch-tests)
        add_executable(strix-hip-shared-expert-prefetch-tests tests/hip_shared_expert_prefetch_tests.hip)
        set_target_properties(strix-hip-shared-expert-prefetch-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-shared-expert-prefetch-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-shared-expert-prefetch-tests)
        add_test(NAME strix-hip-shared-expert-prefetch-tests COMMAND strix-hip-shared-expert-prefetch-tests)

        add_executable(strix-hip-q8-projection-organization-tests tests/hip_q8_projection_organization_tests.hip)
        set_target_properties(strix-hip-q8-projection-organization-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
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
        set_target_properties(strix-hip-shared-expert-rows-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-shared-expert-rows-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-shared-expert-rows-diagnostic-tests)
        add_test(NAME strix-hip-shared-expert-rows-diagnostic-tests COMMAND strix-hip-shared-expert-rows-diagnostic-tests)
        add_executable(strix-hip-qpersist-attention-tests tests/hip_qpersist_attention_tests.hip)
        set_target_properties(strix-hip-qpersist-attention-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qpersist-attention-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qpersist-attention-tests)
        add_test(NAME strix-hip-qpersist-attention-tests COMMAND strix-hip-qpersist-attention-tests)
        add_executable(strix-hip-router-select-tests tests/hip_router_select_tests.hip)
        set_target_properties(strix-hip-router-select-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-router-select-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-router-select-tests)
        add_test(NAME strix-hip-router-select-tests COMMAND strix-hip-router-select-tests 5)
        add_executable(strix-hip-draft-attention-exact-tests tests/hip_draft_attention_exact_tests.hip)
        set_target_properties(strix-hip-draft-attention-exact-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-draft-attention-exact-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-draft-attention-exact-tests)
        add_test(NAME strix-hip-draft-attention-exact-tests COMMAND strix-hip-draft-attention-exact-tests)
        add_executable(strix-hip-qsa-window-attn-v2-tests tests/hip_qsa_window_attn_v2_tests.hip)
        set_target_properties(strix-hip-qsa-window-attn-v2-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qsa-window-attn-v2-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-window-attn-v2-tests)
        add_test(NAME strix-hip-qsa-window-attn-v2-tests COMMAND strix-hip-qsa-window-attn-v2-tests)
        add_executable(strix-hip-qsa-gqa-reuse-diagnostic-tests tests/hip_qsa_gqa_reuse_diagnostic_tests.hip)
        set_target_properties(strix-hip-qsa-gqa-reuse-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-qsa-gqa-reuse-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-qsa-gqa-reuse-diagnostic-tests)
        add_test(NAME strix-hip-qsa-gqa-reuse-diagnostic-tests COMMAND strix-hip-qsa-gqa-reuse-diagnostic-tests)
        add_executable(strix-hip-moe-batched-activation-diagnostic-tests tests/hip_moe_batched_activation_diagnostic_tests.hip)
        set_target_properties(strix-hip-moe-batched-activation-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-moe-batched-activation-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-batched-activation-diagnostic-tests)
        add_test(NAME strix-hip-moe-batched-activation-diagnostic-tests COMMAND strix-hip-moe-batched-activation-diagnostic-tests)
        add_executable(strix-hip-moe-overlap-dispatcher-tests tests/hip_moe_overlap_dispatcher_tests.hip)
        set_target_properties(strix-hip-moe-overlap-dispatcher-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-moe-overlap-dispatcher-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-moe-overlap-dispatcher-tests)
        add_test(NAME strix-hip-moe-overlap-dispatcher-tests COMMAND strix-hip-moe-overlap-dispatcher-tests)
        add_executable(strix-hip-generation-output-tests tests/hip_generation_output_tests.hip)
        add_executable(strix-hip-gdn-recurrent-layout-diagnostic-tests tests/hip_gdn_recurrent_layout_diagnostic_tests.hip)
        set_target_properties(strix-hip-gdn-recurrent-layout-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-gdn-recurrent-layout-diagnostic-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-gdn-recurrent-layout-diagnostic-tests)
        add_test(NAME strix-hip-gdn-recurrent-layout-diagnostic-tests COMMAND strix-hip-gdn-recurrent-layout-diagnostic-tests)
        set_target_properties(strix-hip-generation-output-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-generation-output-tests PRIVATE strix_hip_runtime)
        strix_target_warnings(strix-hip-generation-output-tests)
        add_test(NAME strix-hip-generation-output-tests COMMAND strix-hip-generation-output-tests)
        add_executable(strix-hip-logprob-partials-tests tests/hip_logprob_partials_tests.hip)
        set_target_properties(strix-hip-logprob-partials-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_link_libraries(strix-hip-logprob-partials-tests PRIVATE strix_hip_runtime strix_core)
        strix_target_warnings(strix-hip-logprob-partials-tests)
        add_test(NAME strix-hip-logprob-partials-tests COMMAND strix-hip-logprob-partials-tests)
        set_target_properties(strix-hip-gdn-lm-wave8-diagnostic-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
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
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
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
            src/hip_model_residency.cpp
        )
        target_compile_definitions(strix-hip-model-residency-tests PRIVATE STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC=1)
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

if(STRIX_ENABLE_HIP_RUNTIME)
    # Two-rank distributed runtime: transports and the device-buffer communicator.
    add_library(strix_dist STATIC src/dist_communicator.cpp src/dist_gpu_direct.cpp src/dist_gpu_direct_setup.cpp
        src/dist_gpu_direct_copy.hip src/dist_host_copy.hip src/dist_staged_ready.hip src/linux_gpu_queue.cpp
        src/xport/device_crc32c.hip)
    target_include_directories(strix_dist PUBLIC include)
    target_include_directories(strix_dist PRIVATE src)
    target_include_directories(strix_dist SYSTEM PUBLIC "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_features(strix_dist PUBLIC cxx_std_20)
    target_compile_definitions(strix_dist PUBLIC $<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__=1>)
    find_package(Threads REQUIRED)
    target_link_libraries(strix_dist PUBLIC "${STRIX_HIP_RUNTIME_LIBRARY}" Threads::Threads PRIVATE strix_serve)
    strix_target_warnings(strix_dist)
    target_link_libraries(strix_hip_runtime PUBLIC strix_dist)
    add_executable(strix-dist-tcp-probe apps/dist_tcp_probe_main.cpp)
    target_link_libraries(strix-dist-tcp-probe PRIVATE strix_dist)
    strix_target_warnings(strix-dist-tcp-probe)
    add_executable(strix-dist-host-exchange-bench apps/dist_host_exchange_bench_main.cpp)
    target_link_libraries(strix-dist-host-exchange-bench PRIVATE strix_dist strix_device_hip)
    strix_target_warnings(strix-dist-host-exchange-bench)
    add_executable(strix-dist-gpu-probe apps/dist_gpu_probe_main.cpp)
    target_link_libraries(strix-dist-gpu-probe PRIVATE strix_dist strix_xport strix_device_hip)
    strix_target_warnings(strix-dist-gpu-probe)
    if(BUILD_TESTING)
        add_executable(strix-dist-transport-tests tests/dist_transport_tests.cpp)
        add_executable(strix-dist-crc32c-tests tests/dist_crc32c_tests.cpp)
        target_compile_features(strix-dist-crc32c-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-dist-crc32c-tests)
        add_test(NAME strix-dist-crc32c-tests COMMAND strix-dist-crc32c-tests)
        # Split-producer chunk frames (xport::Channel::split_head): selftest (memory pair, bytewise) + usb4 pair bench
        add_executable(strix-hip-xport-split-frame-bench tests/hip_xport_split_frame_bench.hip)
        target_link_libraries(strix-hip-xport-split-frame-bench PRIVATE strix_dist strix_xport)
        target_include_directories(strix-hip-xport-split-frame-bench PRIVATE src include)
        # Bounded device wait of the channel consumer (wait_view budget + sticky abort word): one GPU, no transport
        add_executable(strix-hip-xport-bounded-wait-tests tests/hip_xport_bounded_wait_tests.hip)
        target_link_libraries(strix-hip-xport-bounded-wait-tests PRIVATE strix_dist strix_xport)
        target_include_directories(strix-hip-xport-bounded-wait-tests PRIVATE src include)
        add_executable(strix-hip-dist-hostpath-tests tests/hip_dist_hostpath_tests.cpp)
        target_link_libraries(strix-hip-dist-hostpath-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-hostpath-tests)
        target_link_libraries(strix-dist-transport-tests PRIVATE strix_dist)
        strix_target_warnings(strix-dist-transport-tests)
        add_test(NAME strix-dist-transport-tests COMMAND strix-dist-transport-tests)
        add_executable(strix-hip-dist-gpu-direct-tests tests/hip_dist_gpu_direct_tests.cpp)
        add_executable(strix-hip-dist-host-copy-tests tests/hip_dist_host_copy_tests.cpp)
        add_executable(strix-hip-staged-ready-tests tests/hip_staged_ready_tests.cpp)
        target_link_libraries(strix-hip-staged-ready-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-staged-ready-tests)
        add_executable(strix-hip-dist-crc-defer-tests tests/hip_dist_crc_defer_tests.cpp)
        target_link_libraries(strix-hip-dist-crc-defer-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-crc-defer-tests)
        target_link_libraries(strix-hip-dist-host-copy-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-host-copy-tests)
        target_link_libraries(strix-hip-dist-gpu-direct-tests PRIVATE strix_dist)
        strix_target_warnings(strix-hip-dist-gpu-direct-tests)
    endif()
endif()
endif()  # NOT STRIX_CUDA_DS41_ONLY

# ---- qwen35 model family module (optional; host contract only, backends consume include/strix/qwen35/config.hpp) ----
if(STRIX_FAMILY_QWEN35)
add_library(strix_family_qwen35 STATIC src/qwen35/config.cpp src/qwen35/family.cpp)
target_link_libraries(strix_family_qwen35 PUBLIC strix_core)
strix_target_warnings(strix_family_qwen35)
if(BUILD_TESTING)
    add_executable(strix-qwen35-config-tests tests/qwen35_config_tests.cpp)
    target_link_libraries(strix-qwen35-config-tests PRIVATE strix_family_qwen35)
    strix_target_warnings(strix-qwen35-config-tests)
    add_test(NAME strix-qwen35-config-tests COMMAND strix-qwen35-config-tests)
    set_tests_properties(strix-qwen35-config-tests PROPERTIES LABELS "cpu;model" TIMEOUT 30)
    add_subdirectory(tests/qwen35_host_guards)
endif()
endif()

# ---- deepseek41 model family module (optional) ----
if(STRIX_FAMILY_DEEPSEEK41)
add_library(strix_family_deepseek41 STATIC src/deepseek41_model.cpp src/deepseek41_plan.cpp src/deepseek41_family.cpp)
target_link_libraries(strix_family_deepseek41 PUBLIC strix_core)
strix_target_warnings(strix_family_deepseek41)
add_executable(strix-deepseek41-inspect apps/deepseek41_inspect_main.cpp)
target_link_libraries(strix-deepseek41-inspect PRIVATE strix_family_deepseek41)
strix_target_warnings(strix-deepseek41-inspect)
# Host contract/graph inspection only; execution never consumes description data.
add_library(strix_deepseek41_model_description STATIC src/deepseek41_model_description.cpp src/model_graph/deepseek41_adapter.cpp)
target_link_libraries(strix_deepseek41_model_description PUBLIC strix_family_deepseek41 strix_model_description strix_model_graph)
strix_target_warnings(strix_deepseek41_model_description)
add_executable(strix-deepseek41-model-describe apps/model_describe_main.cpp)
target_link_libraries(strix-deepseek41-model-describe PRIVATE strix_deepseek41_model_description)
strix_target_warnings(strix-deepseek41-model-describe)
if(BUILD_TESTING)
    add_executable(strix-deepseek41-model-description-tests tests/deepseek41_model_description_tests.cpp)
    target_link_libraries(strix-deepseek41-model-description-tests PRIVATE strix_deepseek41_model_description)
    strix_target_warnings(strix-deepseek41-model-description-tests)
    add_test(NAME strix-deepseek41-model-description-tests COMMAND strix-deepseek41-model-description-tests
        "${CMAKE_CURRENT_SOURCE_DIR}/models/deepseek-v4.1-flash.toml")
    set_tests_properties(strix-deepseek41-model-description-tests PROPERTIES LABELS "cpu;model-description")
    add_executable(strix-deepseek41-plan-tests tests/deepseek41_plan_tests.cpp)
    add_executable(strix-deepseek41-model-graph-tests tests/deepseek41_model_graph_tests.cpp)
    target_link_libraries(strix-deepseek41-model-graph-tests PRIVATE strix_deepseek41_model_description)
    strix_target_warnings(strix-deepseek41-model-graph-tests)
    add_test(NAME strix-deepseek41-model-graph-tests COMMAND strix-deepseek41-model-graph-tests)
    set_tests_properties(strix-deepseek41-model-graph-tests PROPERTIES LABELS "cpu;model-graph;plan")
    target_link_libraries(strix-deepseek41-plan-tests PRIVATE strix_family_deepseek41)
    strix_target_warnings(strix-deepseek41-plan-tests)
    add_test(NAME strix-deepseek41-plan-tests COMMAND strix-deepseek41-plan-tests)
    add_executable(strix-deepseek41-model-tests tests/deepseek41_model_tests.cpp)
    target_link_libraries(strix-deepseek41-model-tests PRIVATE strix_family_deepseek41)
    strix_target_warnings(strix-deepseek41-model-tests)
    add_test(NAME strix-deepseek41-model-tests COMMAND strix-deepseek41-model-tests)
endif()
# deepseek41 weight/cache formats: isolated host and device TUs with strict arithmetic flags.
add_library(strix_ds41_formats STATIC src/ds41_formats.cpp src/ds41_moe.cpp src/ds41_packed.cpp src/ds41_expert_tp.cpp)
target_include_directories(strix_ds41_formats PUBLIC include)
target_compile_features(strix_ds41_formats PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_formats)
if(MSVC)
    target_compile_definitions(strix_ds41_formats PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(NOT MSVC)
    target_compile_options(strix_ds41_formats PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-formats-tests tests/ds41_formats_tests.cpp)
    target_link_libraries(strix-ds41-formats-tests PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-ds41-formats-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-formats-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-formats-tests COMMAND strix-ds41-formats-tests)
    add_executable(strix-ds41-expert-tp-tests tests/ds41_expert_tp_tests.cpp)
    target_link_libraries(strix-ds41-expert-tp-tests PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-ds41-expert-tp-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-expert-tp-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-expert-tp-tests COMMAND strix-ds41-expert-tp-tests)
    add_executable(strix-ds41-packed-tests tests/ds41_packed_tests.cpp)
    target_link_libraries(strix-ds41-packed-tests PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-ds41-packed-tests)
    add_test(NAME strix-ds41-packed-tests COMMAND strix-ds41-packed-tests)
endif()
# deepseek41 native FP8 dense weights of the original checkpoint: safetensors reader and the exact product reference.
add_library(strix_ds41_native_fp8 STATIC src/ds41_native_fp8.cpp)
target_link_libraries(strix_ds41_native_fp8 PUBLIC strix_ds41_formats strix_serve_json)
strix_target_warnings(strix_ds41_native_fp8)
if(NOT MSVC)
    target_compile_options(strix_ds41_native_fp8 PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)  # strix_serve is not configured there
    add_executable(strix-ds41-native-fp8-tests tests/ds41_native_fp8_tests.cpp)
    target_link_libraries(strix-ds41-native-fp8-tests PRIVATE strix_ds41_native_fp8)
    strix_target_warnings(strix-ds41-native-fp8-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-native-fp8-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-native-fp8-tests COMMAND strix-ds41-native-fp8-tests)
endif()
# deepseek41 trellis dense sites (ds41_trellis.hpp, identity ds41-trellis-k5-v1): the pack reader and slicer (host).
add_library(strix_ds41_trellis STATIC src/ds41_trellis.cpp)
target_link_libraries(strix_ds41_trellis PUBLIC strix_ds41_native_fp8)
strix_target_warnings(strix_ds41_trellis)
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)  # strix_serve is not configured there
    # site bytes, step / configuration identities under single packs and mixes, the weights namespace (host only)
    add_executable(strix-ds41-trellis-identity-tests tests/ds41_trellis_identity_tests.cpp)
    target_link_libraries(strix-ds41-trellis-identity-tests PRIVATE strix_ds41_trellis)
    strix_target_warnings(strix-ds41-trellis-identity-tests)
    add_test(NAME strix-ds41-trellis-identity-tests COMMAND strix-ds41-trellis-identity-tests)
endif()
# deepseek41 hot-expert placement: reader of the versioned placement file and the local/remote split of a layer's
# routed rows under the canonical combine (host only).
add_library(strix_ds41_hot_placement STATIC src/ds41_hot_placement.cpp)
target_include_directories(strix_ds41_hot_placement PUBLIC include)
target_compile_features(strix_ds41_hot_placement PUBLIC cxx_std_20)
target_link_libraries(strix_ds41_hot_placement PUBLIC strix_serve_json)
strix_target_warnings(strix_ds41_hot_placement)
if(BUILD_TESTING)
    add_executable(strix-ds41-hot-placement-tests tests/ds41_hot_placement_tests.cpp)
    target_link_libraries(strix-ds41-hot-placement-tests PRIVATE strix_ds41_hot_placement)
    strix_target_warnings(strix-ds41-hot-placement-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-hot-placement-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-hot-placement-tests COMMAND strix-ds41-hot-placement-tests)
endif()
# Shared Engram component (hashed n-gram memory, any family): Spec, hash, row reader, row decode. Host, vendor-free.
find_package(Threads REQUIRED)
add_library(strix_engram STATIC src/engram/spec.cpp src/engram/hasher.cpp src/engram/row_reader.cpp src/engram/read_ahead.cpp)
target_include_directories(strix_engram PUBLIC include PRIVATE src)
target_compile_features(strix_engram PUBLIC cxx_std_20)
target_link_libraries(strix_engram PUBLIC Threads::Threads)
strix_target_warnings(strix_engram)
# Hy4-flash Engram adapter: the shared Spec from the pack metadata (no hashing code).
add_library(strix_hy4_engram STATIC src/hy4/hy4_engram_spec.cpp)
target_link_libraries(strix_hy4_engram PUBLIC strix_engram)
strix_target_warnings(strix_hy4_engram)
# deepseek41 layer sites (mHC, norms, RoPE, projection sites, embedding/head, Engram): host reference
# and the DS41 Engram adapter of strix_engram, with the same strict arithmetic flags as the formats.
add_library(strix_ds41_layer STATIC src/ds41_layer.cpp src/ds41_engram.cpp)
target_include_directories(strix_ds41_layer PUBLIC include)
target_compile_features(strix_ds41_layer PUBLIC cxx_std_20)
target_link_libraries(strix_ds41_layer PUBLIC strix_engram strix_ds41_formats strix_family_deepseek41 Threads::Threads)
strix_target_warnings(strix_ds41_layer)
if(NOT MSVC)
    target_compile_options(strix_ds41_layer PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-layer-tests tests/ds41_layer_tests.cpp)
    target_link_libraries(strix-ds41-layer-tests PRIVATE strix_ds41_layer)
    strix_target_warnings(strix-ds41-layer-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-layer-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-layer-tests COMMAND strix-ds41-layer-tests)
    add_executable(strix-engram-core-tests tests/engram_core_tests.cpp)
    target_link_libraries(strix-engram-core-tests PRIVATE strix_ds41_layer)
    if(TARGET strix_hy4_engram)  # the Hy4 Spec adapter (model package) adds its fixture when present
        target_link_libraries(strix-engram-core-tests PRIVATE strix_hy4_engram)
        target_compile_definitions(strix-engram-core-tests PRIVATE STRIX_ENGRAM_HY4=1)
    endif()
    strix_target_warnings(strix-engram-core-tests)
    add_test(NAME strix-engram-core-tests COMMAND strix-engram-core-tests)
    add_executable(strix-engram-inject-tests tests/engram_inject_tests.cpp)
    target_link_libraries(strix-engram-inject-tests PRIVATE strix_ds41_layer)
    strix_target_warnings(strix-engram-inject-tests)
    # Bitwise declared-order checks: the transcribed reference and the header-inline shared reference compile in this
    # TU, the re-bound ds41::engram_inject in strix_ds41_layer (-ffp-contract=off). Without the same flag a toolchain
    # whose target has FMA fuses multiply-adds in this TU only and the arms differ.
    if(NOT MSVC)
        target_compile_options(strix-engram-inject-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-engram-inject-tests COMMAND strix-engram-inject-tests)
    add_executable(strix-engram-row-prep-tests tests/engram_row_prep_tests.cpp)
    target_link_libraries(strix-engram-row-prep-tests PRIVATE strix_ds41_layer)
    strix_target_warnings(strix-engram-row-prep-tests)
    add_test(NAME strix-engram-row-prep-tests COMMAND strix-engram-row-prep-tests)
    add_executable(strix-ds41-engram-reader-bench tests/ds41_engram_reader_bench.cpp)
    target_link_libraries(strix-ds41-engram-reader-bench PRIVATE strix_ds41_layer)
    strix_target_warnings(strix-ds41-engram-reader-bench)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(strix-ds41-engram-cache-tests tests/ds41_engram_cache_tests.cpp)
        target_link_libraries(strix-ds41-engram-cache-tests PRIVATE strix_ds41_layer)
        strix_target_warnings(strix-ds41-engram-cache-tests)
        add_test(NAME strix-ds41-engram-cache-tests COMMAND strix-ds41-engram-cache-tests)
        add_executable(strix-ds41-engram-trace-bench tests/ds41_engram_trace_bench.cpp)
        target_link_libraries(strix-ds41-engram-trace-bench PRIVATE strix_ds41_layer)
        strix_target_warnings(strix-ds41-engram-trace-bench)
    endif()
endif()
# deepseek41 decode attention: isolated host references and device kernels with strict arithmetic flags.
add_library(strix_ds41_attention STATIC src/ds41_attention.cpp)
target_include_directories(strix_ds41_attention PUBLIC include)
target_link_libraries(strix_ds41_attention PUBLIC strix_family_deepseek41)
# Hyper-connection contract (include/strix/hc): host oracle of the shared HC arithmetic (DS41 mHC, Hy-V4 iHC).
add_library(strix_hc STATIC src/hc/hc_ref.cpp)
target_include_directories(strix_hc PUBLIC include)
target_compile_features(strix_hc PUBLIC cxx_std_20)
strix_target_warnings(strix_hc)
if(NOT MSVC)
    target_compile_options(strix_hc PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-hc-contract-tests tests/hc_contract_tests.cpp)
    target_link_libraries(strix-hc-contract-tests PRIVATE strix_hc strix_ds41_layer)
    strix_target_warnings(strix-hc-contract-tests)
    if(NOT MSVC)
        target_compile_options(strix-hc-contract-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-hc-contract-tests COMMAND strix-hc-contract-tests)
    add_executable(strix-hc-geometry-tests tests/hc_geometry_tests.cpp)
    target_include_directories(strix-hc-geometry-tests PRIVATE include src)
    target_compile_features(strix-hc-geometry-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hc-geometry-tests)
    if(MSVC)
        target_compile_options(strix-hc-geometry-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-hc-geometry-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-hc-geometry-tests COMMAND strix-hc-geometry-tests)
    set_tests_properties(strix-hc-geometry-tests PROPERTIES LABELS cpu)
    add_executable(strix-hc-lane-chain-tests tests/hc_lane_chain_tests.cpp)
    target_include_directories(strix-hc-lane-chain-tests PRIVATE include src)
    target_compile_features(strix-hc-lane-chain-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hc-lane-chain-tests)
    if(MSVC)
        target_compile_options(strix-hc-lane-chain-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-hc-lane-chain-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    if(MSVC)
        target_compile_options(strix-hc-lane-chain-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-hc-lane-chain-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-hc-lane-chain-tests COMMAND strix-hc-lane-chain-tests)
    set_tests_properties(strix-hc-lane-chain-tests PROPERTIES LABELS cpu)
    add_executable(strix-hc-packed-forms-tests tests/hc_packed_forms_tests.cpp)
    target_include_directories(strix-hc-packed-forms-tests PRIVATE include src)
    target_compile_features(strix-hc-packed-forms-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hc-packed-forms-tests)
    if(MSVC)
        target_compile_options(strix-hc-packed-forms-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-hc-packed-forms-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-hc-packed-forms-tests COMMAND strix-hc-packed-forms-tests)
    set_tests_properties(strix-hc-packed-forms-tests PROPERTIES LABELS cpu)
    add_executable(strix-hc-projection-tests tests/hc_projection_tests.cpp)
    target_include_directories(strix-hc-projection-tests PRIVATE include src)
    target_compile_features(strix-hc-projection-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hc-projection-tests)
    if(MSVC)
        target_compile_options(strix-hc-projection-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-hc-projection-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-hc-projection-tests COMMAND strix-hc-projection-tests)
    set_tests_properties(strix-hc-projection-tests PROPERTIES LABELS cpu)
endif()
target_compile_features(strix_ds41_attention PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_attention)
if(NOT MSVC)
    target_compile_options(strix_ds41_attention PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-attention-tests tests/ds41_attention_tests.cpp)
    target_link_libraries(strix-ds41-attention-tests PRIVATE strix_ds41_attention strix_ds41_formats)
    strix_target_warnings(strix-ds41-attention-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-attention-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-attention-tests COMMAND strix-ds41-attention-tests)
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_attention_hip STATIC src/hip_ds41_attention.hip)
    target_link_libraries(strix_ds41_attention_hip PUBLIC strix_ds41_attention "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_attention_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    set(strix_ds41_attention_hip_targets strix_ds41_attention_hip)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-attention-tests tests/hip_ds41_attention_tests.hip)
        target_link_libraries(strix-hip-ds41-attention-tests PRIVATE strix_ds41_attention_hip strix_ds41_hip)
        add_test(NAME strix-hip-ds41-attention-tests COMMAND strix-hip-ds41-attention-tests)
        list(APPEND strix_ds41_attention_hip_targets strix-hip-ds41-attention-tests)
        # FAST lever ds41-attn-f16wmma-v1: error bounds vs the exact attention + timing of both
        add_executable(strix-hip-ds41-fast-attn-tests tests/hip_ds41_fast_attn_tests.hip)
        target_link_libraries(strix-hip-ds41-fast-attn-tests PRIVATE strix_ds41_attention_hip)
        add_test(NAME strix-hip-ds41-fast-attn-tests COMMAND strix-hip-ds41-fast-attn-tests --no-timing)
        list(APPEND strix_ds41_attention_hip_targets strix-hip-ds41-fast-attn-tests)
        # timing build: the same kernels with per-block clocks (STRIX_DS41_ATTN_TIMING), not a test
        add_executable(strix-hip-ds41-attention-timing tests/hip_ds41_attention_timing.hip src/hip_ds41_attention.hip)
        target_link_libraries(strix-hip-ds41-attention-timing PRIVATE strix_ds41_attention "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix-hip-ds41-attention-timing PRIVATE include "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_definitions(strix-hip-ds41-attention-timing PRIVATE STRIX_DS41_ATTN_TIMING)
        list(APPEND strix_ds41_attention_hip_targets strix-hip-ds41-attention-timing)
    endif()
    foreach(target ${strix_ds41_attention_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(${target} PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        target_include_directories(${target} PRIVATE src)
        strix_target_warnings(${target})
    endforeach()
endif()
# deepseek41 trimmed-model host reference (integration): plan interpreter over the formats/MoE pieces.
add_library(strix_ds41_runtime STATIC src/deepseek41_host_forward.cpp src/deepseek41_attention_host.cpp src/ds41_weight_overlay.cpp
    src/deepseek41_attention_state.cpp src/deepseek41_prefix_cache.cpp src/deepseek41_pair_defaults.cpp)
target_link_libraries(strix_ds41_runtime PUBLIC strix_ds41_layer strix_ds41_formats strix_ds41_attention strix_family_deepseek41 strix_state_cache)
strix_target_warnings(strix_ds41_runtime)
if(NOT MSVC)
    target_compile_options(strix_ds41_runtime PRIVATE -ffp-contract=off)
endif()
# Session compatibility uses checkpoint loaders and serving JSON, outside the CUDA replica.
if(NOT STRIX_CUDA_DS41_ONLY)
    add_library(strix_ds41_session STATIC src/deepseek41_session_cache.cpp)
    target_link_libraries(strix_ds41_session PUBLIC strix_ds41_runtime strix_ds41_trellis strix_ds41_native_fp8 strix_state_cache)
    strix_target_warnings(strix_ds41_session)
    if(NOT MSVC)
        target_compile_options(strix_ds41_session PRIVATE -ffp-contract=off)
    endif()
endif()
# Versioned offline rank-local expert storage, shared by decode and batch.
add_library(strix_ds41_prepared_load STATIC src/ds41_prepared_load.cpp)
target_include_directories(strix_ds41_prepared_load PUBLIC include)
target_compile_features(strix_ds41_prepared_load PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(strix_ds41_prepared_load PUBLIC strix_ds41_runtime Threads::Threads)
# Prepared payload verification must not silently use a slow scalar SHA when the
# accelerated backend is installed. This leaves the full-file hashing option unchanged.
if(NOT TARGET OpenSSL::Crypto)
    find_package(OpenSSL QUIET COMPONENTS Crypto)
endif()
if(TARGET OpenSSL::Crypto)
    target_compile_definitions(strix_ds41_prepared_load PRIVATE STRIX_PREPARED_OPENSSL=1)
    target_link_libraries(strix_ds41_prepared_load PRIVATE OpenSSL::Crypto)
    message(STATUS "Prepared-load chunk SHA-256: OpenSSL (v1 full verification)")
else()
    message(STATUS "Prepared-load chunk SHA-256: portable (v1 full verification, parallel chunks)")
endif()
strix_target_warnings(strix_ds41_prepared_load)
add_executable(strix-ds41-prepare-load apps/ds41_prepare_load_main.cpp)
target_link_libraries(strix-ds41-prepare-load PRIVATE strix_ds41_prepared_load)
strix_target_warnings(strix-ds41-prepare-load)
if(BUILD_TESTING)
    add_executable(strix-ds41-prepared-load-tests tests/ds41_prepared_load_tests.cpp)
    target_link_libraries(strix-ds41-prepared-load-tests PRIVATE strix_ds41_prepared_load)
    strix_target_warnings(strix-ds41-prepared-load-tests)
    add_test(NAME strix-ds41-prepared-load-tests COMMAND strix-ds41-prepared-load-tests)
endif()
add_executable(strix-deepseek41-trimmed apps/deepseek41_trimmed_main.cpp)
target_link_libraries(strix-deepseek41-trimmed PRIVATE strix_ds41_runtime)
strix_target_warnings(strix-deepseek41-trimmed)
# attention-state handoff over TCP (prefill host -> decode rank 0), with --synth states of the released plan
add_executable(strix-ds41-state-send apps/ds41_state_send_main.cpp)
target_link_libraries(strix-ds41-state-send PRIVATE strix_ds41_runtime)
target_compile_definitions(strix-ds41-state-send PRIVATE STRIX_DS41_STATE_SYNTH=1)
if(WIN32)
    target_link_libraries(strix-ds41-state-send PRIVATE ws2_32)
endif()
strix_target_warnings(strix-ds41-state-send)
# link utilisation of the shared hand-off transport (strix::handoff): piece size x streams x codec sweeps, CPU cost
add_executable(strix-xfer-probe apps/xfer_probe_main.cpp src/handoff.cpp src/bp_codec.cpp)
target_include_directories(strix-xfer-probe PRIVATE include)
target_link_libraries(strix-xfer-probe PRIVATE Threads::Threads)
target_compile_features(strix-xfer-probe PRIVATE cxx_std_20)
if(WIN32)
    target_link_libraries(strix-xfer-probe PRIVATE ws2_32)
endif()
strix_target_warnings(strix-xfer-probe)
if(BUILD_TESTING)
    add_executable(strix-deepseek41-attention-state-tests tests/deepseek41_attention_state_tests.cpp)
    target_link_libraries(strix-deepseek41-attention-state-tests PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-deepseek41-attention-state-tests)
    add_test(NAME strix-deepseek41-attention-state-tests COMMAND strix-deepseek41-attention-state-tests)
endif()
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)
    add_executable(strix-serve-session-disabled-tests tests/serve_session_disabled_tests.cpp)
    target_link_libraries(strix-serve-session-disabled-tests PRIVATE strix_serve)
    target_sources(strix-serve-session-disabled-tests PRIVATE src/deepseek41_pair_defaults.cpp)
    strix_target_warnings(strix-serve-session-disabled-tests)
    add_test(NAME strix-serve-session-disabled-tests COMMAND strix-serve-session-disabled-tests --session-cache off)
    add_test(NAME strix-serve-session-enabled-tests COMMAND strix-serve-session-disabled-tests --enabled)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-serve-session-no-session-identity-tests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/serve_session_defaults_compare.py $<TARGET_FILE:strix-serve-session-disabled-tests>)
    endif()
    add_executable(strix-deepseek41-session-cache-tests tests/deepseek41_session_cache_tests.cpp src/ds41_file_reader.cpp)
    target_link_libraries(strix-deepseek41-session-cache-tests PRIVATE strix_ds41_session strix_ds41_dspark Threads::Threads)
    strix_target_warnings(strix-deepseek41-session-cache-tests)
    if(MSVC)
        target_compile_options(strix-deepseek41-session-cache-tests PRIVATE /UNDEBUG)
    else()
        target_compile_options(strix-deepseek41-session-cache-tests PRIVATE -UNDEBUG)
    endif()
    set(STRIX_DS41_CHECKPOINT_REFERENCE_DIR "" CACHE PATH "Optional independent DS41 host checkpoint entry files")
    add_test(NAME strix-deepseek41-session-cache-tests COMMAND strix-deepseek41-session-cache-tests "${STRIX_DS41_CHECKPOINT_REFERENCE_DIR}")
    set_tests_properties(strix-deepseek41-session-cache-tests PROPERTIES TIMEOUT 120)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-deepseek41-session-cache-restart-tests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/deepseek41_session_cache_restart_test.py $<TARGET_FILE:strix-deepseek41-session-cache-tests>)
        set_tests_properties(strix-deepseek41-session-cache-restart-tests PROPERTIES TIMEOUT 120)
        find_program(STRIX_SESSION_TEST_BASH bash)
        if(STRIX_SESSION_TEST_BASH)
            add_test(NAME strix-deepseek41-session-cache-gate-tests COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/deepseek41_session_cache_gate_tests.py)
        endif()
    endif()
endif()
if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-deepseek41-prefix-probe-tests COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/ds41_prefix_cache_probe_tests.py)
    endif()
    add_executable(strix-deepseek41-serve-cache-policy-tests tests/deepseek41_serve_cache_policy_tests.cpp)
    target_include_directories(strix-deepseek41-serve-cache-policy-tests PRIVATE include)
    target_compile_features(strix-deepseek41-serve-cache-policy-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-deepseek41-serve-cache-policy-tests)
    add_test(NAME strix-deepseek41-serve-cache-policy-tests COMMAND strix-deepseek41-serve-cache-policy-tests)
    add_executable(strix-repeat-digests-tests tests/repeat_digests_tests.cpp)
    target_include_directories(strix-repeat-digests-tests PRIVATE include)
    target_compile_features(strix-repeat-digests-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-repeat-digests-tests)
    add_test(NAME strix-repeat-digests-tests COMMAND strix-repeat-digests-tests)
    add_subdirectory(tests/repeat_digests_host)
    add_executable(strix-deepseek41-prefix-cache-tests tests/deepseek41_prefix_cache_tests.cpp)
    target_link_libraries(strix-deepseek41-prefix-cache-tests PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-deepseek41-prefix-cache-tests)
    add_test(NAME strix-deepseek41-prefix-cache-tests COMMAND strix-deepseek41-prefix-cache-tests)
    add_subdirectory(tests/eager_code_load_host)
    add_subdirectory(tests/ds41_prefix_checkpoint_io)
    add_subdirectory(tests/ds41_prefix_owner)
    add_subdirectory(tests/pr584_host)
    add_subdirectory(tests/minimax_ab_qualify_host)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_subdirectory(tests/pr605_host)
    endif()
    add_subdirectory(tests/ds41_dense_forms_host)
    add_subdirectory(tests/router_wmma_host)
    add_executable(strix-deepseek41-pair-defaults-tests tests/deepseek41_pair_defaults_tests.cpp)
    target_link_libraries(strix-deepseek41-pair-defaults-tests PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-deepseek41-pair-defaults-tests)
    add_test(NAME strix-deepseek41-pair-defaults-tests COMMAND strix-deepseek41-pair-defaults-tests)
endif()
if(BUILD_TESTING AND NOT STRIX_CUDA_DS41_ONLY)
    add_subdirectory(tests/ds41_session_owner)
    add_executable(strix-deepseek41-serve-sched-tests tests/deepseek41_serve_sched_tests.cpp)
    target_link_libraries(strix-deepseek41-serve-sched-tests PRIVATE strix_ds41_runtime strix_serve)
    strix_target_warnings(strix-deepseek41-serve-sched-tests)
    add_test(NAME strix-deepseek41-serve-sched-tests COMMAND strix-deepseek41-serve-sched-tests)
    add_executable(strix-deepseek41-serve-observe-tests tests/deepseek41_serve_observe_tests.cpp)
    target_include_directories(strix-deepseek41-serve-observe-tests PRIVATE include)
    target_compile_features(strix-deepseek41-serve-observe-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-deepseek41-serve-observe-tests PRIVATE strix_serve)
    strix_target_warnings(strix-deepseek41-serve-observe-tests)
    add_test(NAME strix-deepseek41-serve-observe-tests COMMAND strix-deepseek41-serve-observe-tests)
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_hip STATIC src/hip_ds41_formats.hip src/hip_ds41_routes_fast.hip src/hip_ds41_expert_tp.hip)
    target_link_libraries(strix_ds41_hip PUBLIC strix_ds41_formats "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-formats-tests tests/hip_ds41_formats_tests.hip)
        target_link_libraries(strix-hip-ds41-formats-tests PRIVATE strix_ds41_hip strix_dist strix_core)
        add_test(NAME strix-hip-ds41-formats-tests COMMAND strix-hip-ds41-formats-tests)
        # A8 rows stored by their producer (src/ops/a8: norm store, sigmoid gate / SwiGLU quantiser) and the strided rmsnorm ==
        # the separate passes they replace (bitwise).
        add_executable(strix-hip-a8-producer-tests tests/hip_a8_producer_tests.hip)
        target_include_directories(strix-hip-a8-producer-tests PRIVATE src)
        target_link_libraries(strix-hip-a8-producer-tests PRIVATE strix_ds41_hip strix_dist strix_core)
        add_test(NAME strix-hip-a8-producer-tests COMMAND strix-hip-a8-producer-tests)
        set_tests_properties(strix-hip-a8-producer-tests PROPERTIES SKIP_RETURN_CODE 77)
        set(strix_ds41_hip_targets strix_ds41_hip strix-hip-ds41-formats-tests strix-hip-a8-producer-tests)
    else()
        set(strix_ds41_hip_targets strix_ds41_hip)
    endif()
    add_library(strix_ds41_layer_hip STATIC src/hip_ds41_layer.hip src/hip_ds41_f16_rows.hip src/hip_ds41_rn_wide.hip src/hip_ds41_rn_table.hip src/hip_ds41_f8_rows_fast.hip)
    target_link_libraries(strix_ds41_layer_hip PUBLIC strix_ds41_hip strix_ds41_layer strix_ds41_prefill_hip)
    target_include_directories(strix_ds41_layer_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    if(BUILD_TESTING)
        target_compile_definitions(strix_ds41_layer_hip PRIVATE STRIX_DS41_ENGRAM_DIAG=1)
    endif()
    list(APPEND strix_ds41_hip_targets strix_ds41_layer_hip)
    if(BUILD_TESTING)
        # FAST lever ds41-dense-f16-v1 (shared policy dense-q8a8-f16-v1): error vs the exact Q8_0 x A8 products + timing
        add_executable(strix-hip-ds41-fast-dense-tests tests/hip_ds41_fast_dense_tests.hip)
        target_link_libraries(strix-hip-ds41-fast-dense-tests PRIVATE strix_ds41_layer_hip)
        target_include_directories(strix-hip-ds41-fast-dense-tests PRIVATE src)
        add_test(NAME strix-hip-ds41-fast-dense-tests COMMAND strix-hip-ds41-fast-dense-tests --no-timing)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-fast-dense-tests)
    endif()
    option(STRIX_DS41_PREFILL_FP16_LT "Build optional FP16 dense prefill backend" OFF)
    if(STRIX_DS41_PREFILL_FP16_LT)
        # Same toolkit as the HIP compiler: ROCM_PATH, then the compiler's ROCm root, then the system default.
        find_library(STRIX_HIPBLASLT_LIBRARY NAMES hipblaslt
            HINTS "$ENV{ROCM_PATH}" "${CMAKE_HIP_COMPILER_ROCM_ROOT}" PATH_SUFFIXES lib lib64
            PATHS /opt/rocm /opt/rocm/core-10.1 REQUIRED)
        target_sources(strix_ds41_layer_hip PRIVATE src/hip_ds41_dense_fp16.hip)
        target_link_libraries(strix_ds41_layer_hip PUBLIC "${STRIX_HIPBLASLT_LIBRARY}")
        get_filename_component(strix_hipblaslt_real "${STRIX_HIPBLASLT_LIBRARY}" REALPATH)
        get_filename_component(strix_hipblaslt_dir "${strix_hipblaslt_real}" DIRECTORY)
        target_link_options(strix_ds41_layer_hip INTERFACE "-Wl,-rpath,${strix_hipblaslt_dir}")

        set_property(SOURCE src/hip_deepseek41_prefill.hip APPEND PROPERTY COMPILE_DEFINITIONS STRIX_DS41_PREFILL_FP16_LT=1)
        add_executable(strix-hip-ds41-approx-prefill-tests tests/hip_ds41_approx_prefill_tests.hip)
        target_link_libraries(strix-hip-ds41-approx-prefill-tests PRIVATE strix_ds41_layer_hip strix_ds41_attention_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-approx-prefill-tests)

    endif()
    # Trellis dense sites (ds41_trellis.hpp): device side on the external trellis library (C ABI 1, a static archive
    # checked by its SHA256). No default location: pass -DSTRIX_DS41_TRELLIS_LIBRARY=<archive> and
    # -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=<dir with strix_trellis.h>. ON by default only where both are given and exist;
    # OFF builds a stub that refuses STRIX_DS41_TRELLIS at load time, so every other build keeps working.
    set(STRIX_DS41_TRELLIS_LIBRARY "" CACHE FILEPATH "Trellis library archive (with the dense-r1 and grouped GEMV APIs for the default dense_r1 adapter)")
    set(STRIX_DS41_TRELLIS_INCLUDE_DIR "" CACHE PATH "Directory with strix_trellis.h")
    if(STRIX_DS41_TRELLIS_LIBRARY AND STRIX_DS41_TRELLIS_INCLUDE_DIR AND EXISTS "${STRIX_DS41_TRELLIS_LIBRARY}"
       AND EXISTS "${STRIX_DS41_TRELLIS_INCLUDE_DIR}/strix_trellis.h")
        set(strix_ds41_trellis_default ON)
    else()
        set(strix_ds41_trellis_default OFF)
    endif()
    option(STRIX_DS41_TRELLIS "Link the trellis library for the deepseek41 dense sites" ${strix_ds41_trellis_default})
    set(STRIX_DS41_TRELLIS_API "dense_r1" CACHE STRING "Trellis adapter: dense_r1 (qualified production identity), v12, tp03, splitk04")
    set_property(CACHE STRIX_DS41_TRELLIS_API PROPERTY STRINGS v12 tp03 splitk04 dense_r1)
    if(NOT STRIX_DS41_TRELLIS_API MATCHES "^(v12|tp03|splitk04|dense_r1)$")
        message(FATAL_ERROR "Unknown STRIX_DS41_TRELLIS_API=${STRIX_DS41_TRELLIS_API}")
    endif()
    add_library(strix_ds41_trellis_hip STATIC src/hip_ds41_trellis.hip src/hip_ds41_trellis_group.hip src/hip_ds41_shexp_tp.hip)
    target_link_libraries(strix_ds41_trellis_hip PUBLIC strix_ds41_trellis strix_ds41_formats "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_trellis_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}" PRIVATE src)
    if(STRIX_DS41_TRELLIS)
        if(NOT EXISTS "${STRIX_DS41_TRELLIS_LIBRARY}" OR NOT EXISTS "${STRIX_DS41_TRELLIS_INCLUDE_DIR}/strix_trellis.h")
            message(FATAL_ERROR "STRIX_DS41_TRELLIS=ON needs STRIX_DS41_TRELLIS_LIBRARY (archive) and "
                                "STRIX_DS41_TRELLIS_INCLUDE_DIR (with strix_trellis.h)")
        endif()
        if(NOT STRIX_DS41_TRELLIS_API STREQUAL "v12")
            target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_PREPARED=1)
        endif()
        if(STRIX_DS41_TRELLIS_API MATCHES "^(splitk04|dense_r1)$")
            target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_SPLITK=1)
        endif()
        if(STRIX_DS41_TRELLIS_API STREQUAL "dense_r1")
            target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_DENSE_R1=1)
            # the apps' production defaults (deepseek41_pair_defaults.hpp) put the trellis packs on only with this API
            target_compile_definitions(strix_ds41_trellis_hip PUBLIC STRIX_DS41_PAIR_DENSE_R1=1)
        endif()
        message(STATUS "deepseek41 trellis: ${STRIX_DS41_TRELLIS_LIBRARY}")
        target_compile_definitions(strix_ds41_trellis_hip PRIVATE STRIX_DS41_TRELLIS_LIB=1)
        target_include_directories(strix_ds41_trellis_hip SYSTEM PUBLIC "${STRIX_DS41_TRELLIS_INCLUDE_DIR}")
        target_link_libraries(strix_ds41_trellis_hip PUBLIC "${STRIX_DS41_TRELLIS_LIBRARY}")
    endif()
    list(APPEND strix_ds41_hip_targets strix_ds41_trellis_hip)
    if(BUILD_TESTING AND STRIX_DS41_TRELLIS)
        # R-invariance of the library rows (GEMV R = 1..8, GEMM, the chunked linear) on real pack tensors; run explicitly on a GPU
        add_executable(strix-hip-ds41-trellis-tests tests/hip_ds41_trellis_tests.hip)
        target_link_libraries(strix-hip-ds41-trellis-tests PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-tests)
        # Persistent split-K counters: changing layouts and generic GEMV must not overwrite another plan's counters.
        # Explicit invocation on a GPU only; no CTest registration.
        add_executable(strix-hip-ds41-trellis-workspace-tests tests/hip_ds41_trellis_workspace_tests.hip)
        target_link_libraries(strix-hip-ds41-trellis-workspace-tests PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-workspace-tests)
        # trellis LM head adapter (STRIX_DS41_TRELLIS_HEADOUT): in-place output / padding vs today's staging; run explicitly on a GPU
        add_executable(strix-hip-ds41-trellis-headout-bench tests/hip_ds41_trellis_headout_bench.hip)
        target_link_libraries(strix-hip-ds41-trellis-headout-bench PRIVATE strix_ds41_trellis_hip strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-headout-bench)
        # TrellisGroupRunner's grouped-library path against a shim of the requested grouped strided GEMV; run explicitly on a GPU
        add_executable(strix-hip-ds41-trellis-group-shim-tests tests/hip_ds41_trellis_group_shim_tests.hip)
        target_link_libraries(strix-hip-ds41-trellis-group-shim-tests PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-trellis-group-shim-tests)
        # shared expert TP on the trellis library (ds41-trellis-k5-v1:shexp-tp): exactness of the lean chain, cold timing
        add_executable(strix-hip-ds41-shexp-tp-bench tests/hip_ds41_shexp_tp_bench.hip)
        target_link_libraries(strix-hip-ds41-shexp-tp-bench PRIVATE strix_ds41_trellis_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-shexp-tp-bench)
    endif()
    # trimmed-model device forward (P4) and the host/device lockstep gate
    add_library(strix_ds41_device_forward STATIC src/hip_deepseek41_device_forward.hip)
    target_link_libraries(strix_ds41_device_forward PUBLIC strix_ds41_prepared_load strix_ds41_layer_hip strix_ds41_attention_hip strix_ds41_runtime strix_dist
        strix_ds41_native_fp8 strix_ds41_pipeline strix_ds41_trellis_hip)
    target_include_directories(strix_ds41_device_forward PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    list(APPEND strix_ds41_hip_targets strix_ds41_device_forward)
    add_executable(strix-deepseek41-lockstep apps/deepseek41_lockstep_main.cpp)
    target_link_libraries(strix-deepseek41-lockstep PRIVATE strix_ds41_device_forward)
    strix_target_warnings(strix-deepseek41-lockstep)
    # one rank of the two-node decode (expert halves + split head over the communicator), or both ranks in-process
    add_executable(strix-deepseek41-rank apps/deepseek41_rank_main.cpp)
    target_link_libraries(strix-deepseek41-rank PRIVATE strix_ds41_device_forward strix_dist strix_ds41_pipeline_front strix_ds41_session strix_deepseek41_model_description)
    strix_target_warnings(strix-deepseek41-rank)
    # the FRONT of a pipeline (layers [0, K) here, the rest on a strix-deepseek41-rank --role pipeline-stage TAIL)
    add_executable(strix-deepseek41-pipeline-front apps/deepseek41_pipeline_front_main.cpp)
    target_link_libraries(strix-deepseek41-pipeline-front PRIVATE strix_ds41_device_forward strix_ds41_pipeline_front)
    strix_target_warnings(strix-deepseek41-pipeline-front)
    # requantization quality gate: teacher-forced windows, reference slices and per-row statistics, calibration sums
    add_executable(strix-deepseek41-requant-eval apps/deepseek41_requant_eval_main.cpp)
    target_link_libraries(strix-deepseek41-requant-eval PRIVATE strix_ds41_device_forward strix_dist)
    strix_target_warnings(strix-deepseek41-requant-eval)
    # several requests per forward step (BatchForward), its gate/measurement app and the DeepSeek HTTP server
    # the DS41 Channel (xport::Channel<prefill_xport::Traits>): prefill lease and decode exchanges (STRIX_DS41_DECODE_XPORT)
    add_library(strix_ds41_xport STATIC src/hip_deepseek41_prefill_xport.hip)
    target_link_libraries(strix_ds41_xport PUBLIC strix_dist)
    target_include_directories(strix_ds41_xport PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    list(APPEND strix_ds41_hip_targets strix_ds41_xport)
    add_library(strix_ds41_batch_forward STATIC src/hip_deepseek41_batch_forward.hip)
    target_link_libraries(strix_ds41_batch_forward PUBLIC strix_hip_logprob strix_hip_token_sampling)
    target_link_libraries(strix_ds41_batch_forward PUBLIC strix_ds41_xport strix_ds41_prepared_load strix_ds41_layer_hip strix_ds41_attention_hip strix_ds41_runtime strix_dist strix_ds41_trellis_hip strix_ds41_pipeline)
    target_include_directories(strix_ds41_batch_forward PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    list(APPEND strix_ds41_hip_targets strix_ds41_batch_forward)
    add_executable(strix-deepseek41-batch apps/deepseek41_batch_main.cpp)
    target_link_libraries(strix-deepseek41-batch PRIVATE strix_ds41_batch_dspark strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-batch)
    add_executable(strix-deepseek41-serve apps/deepseek41_serve_main.cpp src/handoff.cpp src/bp_codec.cpp)  # --import-frames
    target_link_libraries(strix-deepseek41-serve PRIVATE strix_ds41_batch_dspark strix_ds41_batch_forward strix_ds41_device_prefill strix_dist strix_serve strix_core strix_ds41_pipeline strix_ds41_session strix_deploy strix_ds41_http_session_owner)
    strix_target_warnings(strix-deepseek41-serve)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-layer-tests tests/hip_ds41_layer_tests.hip)
        add_executable(strix-hip-ds41-r1fuse-tests tests/hip_ds41_r1fuse_tests.hip)
        target_link_libraries(strix-hip-ds41-r1fuse-tests PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-r1fuse-tests)
        add_test(NAME strix-hip-ds41-r1fuse-tests COMMAND strix-hip-ds41-r1fuse-tests)
        add_executable(strix-hip-ds41-gaps-bench tests/hip_ds41_gaps_bench.hip)
        target_link_libraries(strix-hip-ds41-gaps-bench PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-gaps-bench)
        add_executable(strix-hip-ds41-ready-tests tests/hip_ds41_ready_tests.hip)
        target_link_libraries(strix-hip-ds41-ready-tests PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-ready-tests)
        add_executable(strix-hip-ds41-ready-bench tests/hip_ds41_ready_bench.hip)
        target_link_libraries(strix-hip-ds41-ready-bench PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-ready-bench)
        target_link_libraries(strix-hip-ds41-layer-tests PRIVATE strix_ds41_layer_hip)
        add_test(NAME strix-hip-ds41-layer-tests COMMAND strix-hip-ds41-layer-tests)
        add_executable(strix-hip-ds41-f16-rows-tests tests/hip_ds41_f16_rows_tests.hip)
        target_link_libraries(strix-hip-ds41-f16-rows-tests PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        add_test(NAME strix-hip-ds41-f16-rows-tests COMMAND strix-hip-ds41-f16-rows-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-f16-rows-tests)
        add_executable(strix-hip-ds41-engram-f16-pf-tests tests/hip_ds41_engram_f16_pf_tests.hip)
        target_link_libraries(strix-hip-ds41-engram-f16-pf-tests PRIVATE strix_ds41_layer_hip)
        add_test(NAME strix-hip-ds41-engram-f16-pf-tests COMMAND strix-hip-ds41-engram-f16-pf-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-engram-f16-pf-tests)
        add_executable(strix-hip-ds41-layer-bench tests/hip_ds41_layer_bench.hip)
        target_link_libraries(strix-hip-ds41-layer-bench PRIVATE strix_ds41_layer_hip)
        add_executable(strix-hip-ds41-rn-wide-bench tests/hip_ds41_rn_wide_bench.hip)
        target_link_libraries(strix-hip-ds41-rn-wide-bench PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-rn-wide-bench)
        # trellis LM head bench: only with an external strix_trellis static library (-DSTRIX_DS41_TRELLIS_HEAD_BENCH_LIB=<.a>
        # -DSTRIX_DS41_TRELLIS_HEAD_BENCH_INCLUDE=<dir of strix_trellis.h>)
        if(STRIX_DS41_TRELLIS_HEAD_BENCH_LIB)
            add_executable(strix-hip-ds41-trellis-head-bench tests/hip_ds41_trellis_head_bench.hip)
            target_include_directories(strix-hip-ds41-trellis-head-bench PRIVATE "${STRIX_DS41_TRELLIS_HEAD_BENCH_INCLUDE}")
            target_link_libraries(strix-hip-ds41-trellis-head-bench PRIVATE strix_ds41_layer_hip strix_ds41_formats
                "${STRIX_DS41_TRELLIS_HEAD_BENCH_LIB}")
        endif()
        add_executable(strix-hip-ds41-rn-sel-bench tests/hip_ds41_rn_sel_bench.hip)
        target_link_libraries(strix-hip-ds41-rn-sel-bench PRIVATE strix_ds41_layer_hip)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-rn-sel-bench)
        # Trellis attention projections bench (needs the trellis library, see STRIX_DS41_TRELLIS above).
        if(STRIX_DS41_TRELLIS)
            add_executable(strix-hip-ds41-trellis-attn-bench tests/hip_ds41_trellis_attn_bench.hip)
            target_include_directories(strix-hip-ds41-trellis-attn-bench SYSTEM PRIVATE "${STRIX_DS41_TRELLIS_INCLUDE_DIR}")
            target_link_libraries(strix-hip-ds41-trellis-attn-bench PRIVATE strix_ds41_layer_hip "${STRIX_DS41_TRELLIS_LIBRARY}")
            # o_a adapter (per-group calls vs the grouped runner, STRIX_DS41_TRELLIS_OA_GROUP) on the real pack, + today's U6
            add_executable(strix-hip-ds41-trellis-oa-bench tests/hip_ds41_trellis_oa_bench.hip)
            target_link_libraries(strix-hip-ds41-trellis-oa-bench PRIVATE strix_ds41_trellis_hip strix_ds41_layer_hip)
        endif()
        add_executable(strix-hip-ds41-engram-bench tests/hip_ds41_engram_bench.hip)
        target_link_libraries(strix-hip-ds41-engram-bench PRIVATE strix_ds41_layer_hip)
        add_executable(strix-hip-ds41-engram-kv-bench tests/hip_ds41_engram_kv_bench.hip)
        target_link_libraries(strix-hip-ds41-engram-kv-bench PRIVATE strix_ds41_layer_hip strix_ds41_formats)
        add_executable(strix-hip-ds41-layer-prefill-bench tests/hip_ds41_layer_prefill_bench.hip)
        target_link_libraries(strix-hip-ds41-layer-prefill-bench PRIVATE strix_ds41_layer_hip)
        # native FP8 x A8 sites and the shared-expert SwiGLU + A8 epilogue against product_fp8 / quantize_a8, bitwise
        add_executable(strix-hip-ds41-native-fp8-tests tests/hip_ds41_native_fp8_tests.hip)
        target_link_libraries(strix-hip-ds41-native-fp8-tests PRIVATE strix_ds41_layer_hip strix_ds41_native_fp8)
        add_test(NAME strix-hip-ds41-native-fp8-tests COMMAND strix-hip-ds41-native-fp8-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-native-fp8-tests)
        add_executable(strix-hip-ds41-f16-wide-tests tests/hip_ds41_f16_wide_tests.hip)
        target_link_libraries(strix-hip-ds41-f16-wide-tests PRIVATE strix_ds41_layer_hip)
        add_test(NAME strix-hip-ds41-f16-wide-tests COMMAND strix-hip-ds41-f16-wide-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-layer-tests strix-hip-ds41-layer-bench strix-hip-ds41-engram-bench
            strix-hip-ds41-engram-kv-bench
            strix-hip-ds41-layer-prefill-bench strix-hip-ds41-f16-wide-tests)
        # routed/shared expert and head kernels at the runner's launch shapes: timings and form checks
        add_executable(strix-hip-ds41-moe-bench tests/hip_ds41_moe_bench.hip)
        target_link_libraries(strix-hip-ds41-moe-bench PRIVATE strix_ds41_layer_hip strix_core)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-moe-bench)
        add_executable(strix-hip-ds41-expert-once-bench tests/hip_ds41_expert_once_bench.hip)
        target_link_libraries(strix-hip-ds41-expert-once-bench PRIVATE strix_ds41_layer_hip strix_core)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-expert-once-bench)
        # router fast form, paired F16 projections, compress ahead: against the current kernels (STRIX_DS41_SMALL_OPS_BENCH=1: timings)
        add_executable(strix-hip-ds41-small-ops-tests tests/hip_ds41_small_ops_tests.hip)
        target_link_libraries(strix-hip-ds41-small-ops-tests PRIVATE strix_ds41_layer_hip strix_ds41_attention_hip strix_ds41_formats)
        add_test(NAME strix-hip-ds41-small-ops-tests COMMAND strix-hip-ds41-small-ops-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-ds41-small-ops-tests)
    endif()
    if(BUILD_TESTING)
        # Shared HIP hyper-connection core (src/core/hc_core.hpp) vs the host oracle (strix_hc), bitwise: Hy-V4 Flash
        # sublayer + head (identity comb, F32 fn, bf16 store) and DeepSeek V4.1 (Sinkhorn, F16 fn) Specs, synthetic data.
        add_executable(strix-hip-hc-core-tests tests/hip_hc_core_tests.hip)
        target_link_libraries(strix-hip-hc-core-tests PRIVATE strix_hc strix_ds41_layer "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix-hip-hc-core-tests PRIVATE include src "${STRIX_HIP_INCLUDE_DIR}")
        add_test(NAME strix-hip-hc-core-tests COMMAND strix-hip-hc-core-tests)
        list(APPEND strix_ds41_hip_targets strix-hip-hc-core-tests)
    endif()
    foreach(target ${strix_ds41_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(${target} PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(${target})
    endforeach()
endif()

# deepseek41 prefill: routed experts of T rows (WMMA iu8, decode-identical arithmetic) and the exact
# suffix replay plan.
add_library(strix_ds41_prefill STATIC src/ds41_prefill.cpp)
target_include_directories(strix_ds41_prefill PUBLIC include)
target_compile_features(strix_ds41_prefill PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_prefill)
if(BUILD_TESTING)
    add_executable(strix-ds41-integer-operand-host-tests tests/ds41_integer_operand_host_tests.cpp)
    target_include_directories(strix-ds41-integer-operand-host-tests PRIVATE
        tests/ds41_integer_host src include)
    target_compile_features(strix-ds41-integer-operand-host-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(strix-ds41-integer-operand-host-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-ds41-integer-operand-host-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-integer-operand-host-tests COMMAND strix-ds41-integer-operand-host-tests)
    add_executable(strix-ds41-prefill-tests tests/ds41_prefill_tests.cpp)
    target_link_libraries(strix-ds41-prefill-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-prefill-tests)
    add_test(NAME strix-ds41-prefill-tests COMMAND strix-ds41-prefill-tests)
    add_executable(strix-ds41-prefill-tuning-tests tests/ds41_prefill_tuning_tests.cpp)
    target_link_libraries(strix-ds41-prefill-tuning-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-prefill-tuning-tests)
    add_test(NAME strix-ds41-prefill-tuning-tests COMMAND strix-ds41-prefill-tuning-tests)
    add_executable(strix-ds41-host-pin-admission-tests tests/ds41_host_pin_admission_tests.cpp)
    target_link_libraries(strix-ds41-host-pin-admission-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-host-pin-admission-tests)
    add_test(NAME strix-ds41-host-pin-admission-tests COMMAND strix-ds41-host-pin-admission-tests)
    add_executable(strix-ds41-host-pin-plan-tests tests/ds41_host_pin_plan_tests.cpp)
    target_link_libraries(strix-ds41-host-pin-plan-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-host-pin-plan-tests)
    add_test(NAME strix-ds41-host-pin-plan-tests COMMAND strix-ds41-host-pin-plan-tests)
    add_subdirectory(tests/ds41_host_pin_driver_host)
    add_executable(strix-ds41-host-promote-tests tests/ds41_host_promote_tests.cpp)
    target_link_libraries(strix-ds41-host-promote-tests PRIVATE strix_ds41_prefill)
    strix_target_warnings(strix-ds41-host-promote-tests)
    add_test(NAME strix-ds41-host-promote-tests COMMAND strix-ds41-host-promote-tests)
endif()

# The pipeline wire (serving front end, TCP) and the DSpark drafter are not part of the CUDA-only configure.
if(NOT STRIX_CUDA_DS41_ONLY)
# deepseek41 heterogeneous pipeline (portable): the strix/ds41-pipeline-v1 wire between pipeline stages (messages, the
# cut's side sections, TCP) and the tensormeld/engine-describe-v1 description generated from the model file header.
add_library(strix_ds41_pipeline STATIC src/ds41_pipeline_wire.cpp src/ds41_pipeline_multi.cpp src/ds41_pipeline_dspark.cpp src/deepseek41_describe.cpp)
target_link_libraries(strix_ds41_pipeline PUBLIC strix_family_deepseek41 strix_ds41_prefill strix_serve_json)
strix_target_warnings(strix_ds41_pipeline)
# the engine revision --describe-json reports: -DSTRIX_SOURCE_REVISION=..., else the tree's .source-revision file
if(NOT STRIX_SOURCE_REVISION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.source-revision")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/.source-revision" STRIX_SOURCE_REVISION LIMIT_COUNT 1)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/.source-revision")
endif()
if(STRIX_SOURCE_REVISION)
    set_source_files_properties(src/deepseek41_describe.cpp PROPERTIES COMPILE_DEFINITIONS "STRIX_SOURCE_REVISION=\"${STRIX_SOURCE_REVISION}\"")
endif()
# the FRONT side (backend interface, TAIL client, state manifest)
add_library(strix_ds41_pipeline_front STATIC src/ds41_pipeline_front.cpp)
target_link_libraries(strix_ds41_pipeline_front PUBLIC strix_ds41_pipeline strix_ds41_runtime)
strix_target_warnings(strix_ds41_pipeline_front)
# a FRONT without layers: replays recorded STEP payloads to a TAIL (exactness and TAIL timing without a FRONT's compute)
if(NOT WIN32)  # pipeline Connection/Listener implement POSIX sockets only
    add_executable(strix-deepseek41-pipeline-replay apps/deepseek41_pipeline_replay_main.cpp)
    target_link_libraries(strix-deepseek41-pipeline-replay PRIVATE strix_ds41_pipeline)
    strix_target_warnings(strix-deepseek41-pipeline-replay)
endif()
if(BUILD_TESTING)
    if(NOT WIN32)  # pipeline Connection/Listener implement POSIX sockets only
        add_executable(strix-ds41-pipeline-wire-tests tests/ds41_pipeline_wire_tests.cpp)
        target_link_libraries(strix-ds41-pipeline-wire-tests PRIVATE strix_ds41_pipeline_front)
        strix_target_warnings(strix-ds41-pipeline-wire-tests)
        add_test(NAME strix-ds41-pipeline-wire-tests COMMAND strix-ds41-pipeline-wire-tests)
        add_executable(strix-ds41-pipeline-multi-tests tests/ds41_pipeline_multi_tests.cpp)
        target_link_libraries(strix-ds41-pipeline-multi-tests PRIVATE strix_ds41_pipeline)
        strix_target_warnings(strix-ds41-pipeline-multi-tests)
        add_test(NAME strix-ds41-pipeline-multi-tests COMMAND strix-ds41-pipeline-multi-tests)
    endif()
    add_executable(strix-ds41-pipeline-dspark-tests tests/ds41_pipeline_dspark_tests.cpp)
    target_link_libraries(strix-ds41-pipeline-dspark-tests PRIVATE strix_ds41_pipeline)
    strix_target_warnings(strix-ds41-pipeline-dspark-tests)
    add_executable(strix-deepseek41-describe-tests tests/deepseek41_describe_tests.cpp)
    target_link_libraries(strix-deepseek41-describe-tests PRIVATE strix_ds41_pipeline)
    strix_target_warnings(strix-deepseek41-describe-tests)
    add_test(NAME strix-deepseek41-describe-tests COMMAND strix-deepseek41-describe-tests)
    # length routing of the prefill (local two-node prefill vs offload + state handoff; header only)
    add_executable(strix-ds41-prefill-route-tests tests/ds41_prefill_route_tests.cpp)
    target_link_libraries(strix-ds41-prefill-route-tests PRIVATE strix_serve_json)
    strix_target_warnings(strix-ds41-prefill-route-tests)
    add_test(NAME strix-ds41-prefill-route-tests COMMAND strix-ds41-prefill-route-tests)
endif()
# deepseek41 expert server (portable part): the strix/ds41-expert-rpc-v1 wire between a parts-host and expert servers
# (POSIX or Winsock), the server's session loop (direct or relay over two ranks) and its CPU reference backend, and the
# round-trip probe. The HIP backend is below (STRIX_ENABLE_HIP_RUNTIME). Strict FP: the server folds and the reference
# combine must keep the one-device combine's bits.
add_library(strix_ds41_xrpc STATIC src/ds41_expert_rpc.cpp)
target_include_directories(strix_ds41_xrpc PUBLIC include)
target_compile_features(strix_ds41_xrpc PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_xrpc)
add_library(strix_ds41_xs STATIC src/ds41_expert_server.cpp)
target_link_libraries(strix_ds41_xs PUBLIC strix_ds41_xrpc strix_ds41_formats strix_core Threads::Threads)
strix_target_warnings(strix_ds41_xs)
add_executable(strix-ds41-xs-probe apps/ds41_xs_probe_main.cpp)
target_link_libraries(strix-ds41-xs-probe PRIVATE strix_ds41_xrpc Threads::Threads)
strix_target_warnings(strix-ds41-xs-probe)
add_executable(strix-ds41-xs-check apps/ds41_xs_check_main.cpp)
target_link_libraries(strix-ds41-xs-check PRIVATE strix_ds41_xs)
strix_target_warnings(strix-ds41-xs-check)
if(NOT STRIX_ENABLE_HIP_RUNTIME)
    add_executable(strix-deepseek41-expert-server apps/deepseek41_expert_server_main.cpp)
    target_link_libraries(strix-deepseek41-expert-server PRIVATE strix_ds41_xs)
    strix_target_warnings(strix-deepseek41-expert-server)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-expert-rpc-tests tests/ds41_expert_rpc_tests.cpp)
    target_link_libraries(strix-ds41-expert-rpc-tests PRIVATE strix_ds41_xs)
    strix_target_warnings(strix-ds41-expert-rpc-tests)
    add_test(NAME strix-ds41-expert-rpc-tests COMMAND strix-ds41-expert-rpc-tests)
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    # the HIP backend (routed experts resident on a gfx1151 node) and the server with relay over USB4
    add_library(strix_ds41_xs_hip STATIC src/hip_ds41_expert_backend.hip)
    target_link_libraries(strix_ds41_xs_hip PUBLIC strix_ds41_xs strix_ds41_hip strix_device_hip "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_xs_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    set_target_properties(strix_ds41_xs_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_compile_options(strix_ds41_xs_hip PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
        -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
    strix_target_warnings(strix_ds41_xs_hip)
    add_executable(strix-deepseek41-expert-server apps/deepseek41_expert_server_main.cpp)
    target_link_libraries(strix-deepseek41-expert-server PRIVATE strix_ds41_xs_hip strix_dist)
    target_compile_definitions(strix-deepseek41-expert-server PRIVATE STRIX_XS_WITH_HIP)
    strix_target_warnings(strix-deepseek41-expert-server)
endif()
if(NOT MSVC)
    foreach(target strix_ds41_xrpc strix_ds41_xs)
        target_compile_options(${target} PRIVATE -ffp-contract=off)
    endforeach()
    target_compile_options(strix-ds41-xs-check PRIVATE -ffp-contract=off)
    if(BUILD_TESTING)
        target_compile_options(strix-ds41-expert-rpc-tests PRIVATE -ffp-contract=off)
    endif()
endif()
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_prefill_hip STATIC src/hip_ds41_prefill.hip)
    target_link_libraries(strix_ds41_prefill_hip PUBLIC strix_ds41_prefill strix_ds41_hip strix_ds41_formats "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_ds41_prefill_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    # Two-rank prefill of one MoE layer (expert halves, half sums exchanged per chunk).
    add_executable(strix-ds41-prefill-pair apps/ds41_prefill_pair_main.hip)
    target_link_libraries(strix-ds41-prefill-pair PRIVATE strix_ds41_prefill_hip strix_ds41_hip strix_dist strix_core)
    set(strix_ds41_prefill_hip_targets strix_ds41_prefill_hip strix-ds41-prefill-pair)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-prefill-tests tests/hip_ds41_prefill_tests.hip)
        target_link_libraries(strix-hip-ds41-prefill-tests PRIVATE strix_ds41_prefill_hip strix_ds41_hip strix_core)
        add_test(NAME strix-hip-ds41-prefill-tests COMMAND strix-hip-ds41-prefill-tests)
        # FAST lever ds41-moe-relaxed-v1: error vs the exact batch kernels + timings
        add_executable(strix-hip-ds41-fast-moe-tests tests/hip_ds41_fast_moe_tests.hip)
        target_link_libraries(strix-hip-ds41-fast-moe-tests PRIVATE strix_ds41_prefill_hip strix_ds41_hip strix_core)
        add_test(NAME strix-hip-ds41-fast-moe-tests COMMAND strix-hip-ds41-fast-moe-tests --no-timing)
        list(APPEND strix_ds41_prefill_hip_targets strix-hip-ds41-fast-moe-tests)
        list(APPEND strix_ds41_prefill_hip_targets strix-hip-ds41-prefill-tests)
    endif()
    foreach(target ${strix_ds41_prefill_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(${target} PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(${target})
    endforeach()
endif()

# deepseek41 DSpark drafter: host reference (declared arithmetic) and device kernels.
add_library(strix_ds41_dspark STATIC src/ds41_dspark.cpp src/ds41_dspark_files.cpp src/ds41_dspark_driver.cpp src/ds41_dspark_vocab.cpp)
target_link_libraries(strix_ds41_dspark PUBLIC strix_ds41_formats strix_core)
find_package(Threads REQUIRED)
target_link_libraries(strix_ds41_dspark PUBLIC Threads::Threads)
strix_target_warnings(strix_ds41_dspark)
if(NOT MSVC)
    target_compile_options(strix_ds41_dspark PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    add_executable(strix-ds41-dspark-tests tests/ds41_dspark_tests.cpp)
    target_link_libraries(strix-ds41-dspark-tests PRIVATE strix_ds41_dspark strix_core)
    strix_target_warnings(strix-ds41-dspark-tests)
    if(NOT MSVC)
        target_compile_options(strix-ds41-dspark-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-ds41-dspark-tests COMMAND strix-ds41-dspark-tests)
    add_executable(strix-ds41-dspark-table-tests tests/ds41_dspark_table_tests.cpp)
    target_link_libraries(strix-ds41-dspark-table-tests PRIVATE strix_ds41_dspark strix_core)
    strix_target_warnings(strix-ds41-dspark-table-tests)
    add_test(NAME strix-ds41-dspark-table-tests COMMAND strix-ds41-dspark-table-tests)
endif()
add_executable(strix-deepseek41-dspark apps/deepseek41_dspark_main.cpp)
target_link_libraries(strix-deepseek41-dspark PRIVATE strix_ds41_runtime strix_ds41_dspark)
strix_target_warnings(strix-deepseek41-dspark)
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_dspark_hip STATIC src/hip_ds41_dspark.hip)
    target_link_libraries(strix_ds41_dspark_hip PUBLIC strix_ds41_dspark strix_ds41_hip)
    set(strix_ds41_dspark_hip_targets strix_ds41_dspark_hip)
    if(BUILD_TESTING)
        add_executable(strix-hip-ds41-dspark-tests tests/hip_ds41_dspark_tests.hip)
        target_link_libraries(strix-hip-ds41-dspark-tests PRIVATE strix_ds41_dspark_hip strix_dist strix_core)
        add_test(NAME strix-hip-ds41-dspark-tests COMMAND strix-hip-ds41-dspark-tests)
        list(APPEND strix_ds41_dspark_hip_targets strix-hip-ds41-dspark-tests)
    endif()
    add_executable(strix-ds41-dspark-rank apps/ds41_dspark_rank_main.hip)
    target_link_libraries(strix-ds41-dspark-rank PRIVATE strix_ds41_dspark_hip strix_dist)
    list(APPEND strix_ds41_dspark_hip_targets strix-ds41-dspark-rank)
    # DSpark windows inside BatchForward (several requests per step): drafting contexts per request, the planner
    add_library(strix_ds41_batch_dspark STATIC src/hip_deepseek41_batch_dspark.hip)
    target_link_libraries(strix_ds41_batch_dspark PUBLIC strix_ds41_dspark_hip strix_ds41_batch_forward strix_lookup strix_spec)
    list(APPEND strix_ds41_dspark_hip_targets strix_ds41_batch_dspark)
    add_executable(strix-deepseek41-dspark-rank apps/deepseek41_dspark_rank_main.hip)
    target_link_libraries(strix-deepseek41-dspark-rank PRIVATE strix_ds41_dspark_hip strix_ds41_device_forward strix_dist)
    list(APPEND strix_ds41_dspark_hip_targets strix-deepseek41-dspark-rank)
    add_executable(strix-ds41-dspark-replay apps/ds41_dspark_replay_main.hip)
    target_link_libraries(strix-ds41-dspark-replay PRIVATE strix_ds41_dspark_hip)
    list(APPEND strix_ds41_dspark_hip_targets strix-ds41-dspark-replay)
    # the drafter of a pipeline TAIL (speculative extension of strix/ds41-pipeline-v1)
    add_library(strix_ds41_spec_tail STATIC src/hip_ds41_spec_tail.hip)
    target_link_libraries(strix_ds41_spec_tail PUBLIC strix_ds41_dspark_hip)
    list(APPEND strix_ds41_dspark_hip_targets strix_ds41_spec_tail)
    if(TARGET strix-deepseek41-rank)
        target_link_libraries(strix-deepseek41-rank PRIVATE strix_ds41_spec_tail)
    endif()
    foreach(target ${strix_ds41_dspark_hip_targets})
        set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(${target} PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
            -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
        strix_target_warnings(${target})
    endforeach()
endif()
# deepseek41 device prefill: T rows of the decode plan per layer (one node or one of two ranks) and its app.
if(STRIX_ENABLE_HIP_RUNTIME)
    add_library(strix_ds41_device_prefill STATIC src/hip_deepseek41_prefill.hip)
    target_link_libraries(strix_ds41_device_prefill PUBLIC strix_ds41_xport strix_ds41_device_forward strix_ds41_prefill_hip strix_ds41_layer_hip
        strix_ds41_attention_hip strix_dist)
    target_include_directories(strix_ds41_device_prefill PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
    set_target_properties(strix_ds41_device_prefill PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_compile_options(strix_ds41_device_prefill PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
        -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
    strix_target_warnings(strix_ds41_device_prefill)
    add_executable(strix-deepseek41-prefill apps/deepseek41_prefill_main.cpp)
    target_link_libraries(strix-deepseek41-prefill PRIVATE strix_ds41_device_prefill strix_ds41_device_forward strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-prefill)
    add_executable(strix-deepseek41-prefill-quality apps/deepseek41_prefill_quality_main.cpp)
    target_link_libraries(strix-deepseek41-prefill-quality PRIVATE strix_ds41_device_prefill strix_ds41_runtime strix_dist)
    strix_target_warnings(strix-deepseek41-prefill-quality)
    add_executable(strix-deepseek41-prefill-continue apps/deepseek41_prefill_continue_main.cpp)
    target_link_libraries(strix-deepseek41-prefill-continue PRIVATE strix_ds41_device_prefill strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-prefill-continue)

    # long context: the device prefill's attention state handed to the decode forward (or read from a file), then decode
    add_executable(strix-deepseek41-longctx apps/deepseek41_longctx_main.cpp)
    target_link_libraries(strix-deepseek41-longctx PRIVATE strix_ds41_device_prefill strix_ds41_device_forward strix_dist)
    strix_target_warnings(strix-deepseek41-longctx)
    # prefix cache gate: a prompt resumed from cached block states == its fresh prefill (tokens, logits, state digest)
    add_executable(strix-deepseek41-prefix apps/deepseek41_prefix_main.cpp)
    target_link_libraries(strix-deepseek41-prefix PRIVATE strix_ds41_device_prefill strix_ds41_batch_forward strix_dist)
    strix_target_warnings(strix-deepseek41-prefix)
endif()
endif()  # NOT STRIX_CUDA_DS41_ONLY
# Host-forward capture for the CUDA layer-core check (host only; also built with the CUDA backend).
if(STRIX_DS41_TAP_CAPTURE OR STRIX_ENABLE_CUDA_DS41)
    add_executable(strix-ds41-tap-capture tests/cuda/ds41_core_tap_capture.cpp)
    target_link_libraries(strix-ds41-tap-capture PRIVATE strix_ds41_runtime)
    strix_target_warnings(strix-ds41-tap-capture)
endif()
# strix/ds41-expert-rpc-v1 (std only, POSIX or Winsock): the wire between a parts-host and expert servers.
# Strict FP: the reference combine must keep the one-device combine's bits. Defined once: the expert server block above
# defines it when that part of the tree is configured.
if(NOT TARGET strix_ds41_xrpc)
    add_library(strix_ds41_xrpc STATIC src/ds41_expert_rpc.cpp)
    target_include_directories(strix_ds41_xrpc PUBLIC include)
    target_compile_features(strix_ds41_xrpc PUBLIC cxx_std_20)
    strix_target_warnings(strix_ds41_xrpc)
    if(NOT MSVC)
        target_compile_options(strix_ds41_xrpc PRIVATE -ffp-contract=off)
    endif()
endif()
# Pure host resolver: available to inspection/tests without a CUDA toolchain.
add_library(strix_ds41_rtx_profile STATIC src/ds41_rtx_profile.cpp)
target_include_directories(strix_ds41_rtx_profile PUBLIC include)
target_compile_features(strix_ds41_rtx_profile PUBLIC cxx_std_20)
strix_target_warnings(strix_ds41_rtx_profile)
if(BUILD_TESTING)
    add_executable(strix-ds41-rtx-profile-tests tests/ds41_rtx_profile_tests.cpp)
    target_link_libraries(strix-ds41-rtx-profile-tests PRIVATE strix_ds41_rtx_profile)
    strix_target_warnings(strix-ds41-rtx-profile-tests)
    add_test(NAME strix-ds41-rtx-profile-tests COMMAND strix-ds41-rtx-profile-tests)
    set_tests_properties(strix-ds41-rtx-profile-tests PROPERTIES LABELS "cpu;profile")
endif()
# deepseek41 CUDA backend (sm_120, sm_89, sm_86): layer kernels with the declared arithmetic of ds41-exact-v1,
# checked against the host references. Off by default; nothing else depends on it. The kernels use the INT8
# mma.sync / ldmatrix / cp.async these architectures share (same bits on each); select_device() refuses a device
# the build has no code for.
if(STRIX_ENABLE_CUDA_DS41)
    add_executable(strix-cuda-ds41-attention-alloc-tests tests/cuda/ds41_attention_alloc_tests.cpp)
    target_link_libraries(strix-cuda-ds41-attention-alloc-tests PRIVATE strix_ds41_attention strix_ds41_prefill)
    strix_target_warnings(strix-cuda-ds41-attention-alloc-tests)
    if(BUILD_TESTING)
        add_test(NAME strix-cuda-ds41-attention-alloc-tests COMMAND strix-cuda-ds41-attention-alloc-tests)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "120a;89;86")  # 120a: sm_120 plus the e4m3 mma of the FAST prefill GEMM
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/StrixCudaBackend.cmake)
    set(STRIX_DS41_CUDA_SOURCES src/cuda/models/ds41/ds41_cuda_common.cu src/cuda/models/ds41/ds41_core.cu src/cuda/models/ds41/ds41_core_norm.cu
        src/cuda/models/ds41/ds41_core_moe.cu src/cuda/models/ds41/ds41_core_io.cu ${STRIX_CUDA_CORE_DENSE_SOURCES} src/cuda/models/ds41/ds41_core_mma.cu src/cuda/models/ds41/ds41_router_v2.cu
        src/cuda/models/ds41/ds41_core_layer.cu)
    # Every deepseek41 CUDA binary resolves its device through the start gate before its first allocation.
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/StrixDevice.cmake)
    add_library(strix_ds41_cuda STATIC ${STRIX_DS41_CUDA_SOURCES})
    target_link_libraries(strix_ds41_cuda PUBLIC strix_ds41_layer strix_ds41_formats strix_family_deepseek41 strix_device strix_cuda_core CUDA::cudart)
    target_include_directories(strix_ds41_cuda PUBLIC include)
    set(strix_ds41_cuda_targets strix_ds41_cuda)
    # CUDA backend core, model-agnostic: logprob partials of device rows (same contract as src/hip_logprob_partials.hip)
    add_library(strix_cuda_logprob STATIC src/cuda/logprob_partials.cu)
    target_link_libraries(strix_cuda_logprob PUBLIC strix_core CUDA::cudart)
    target_include_directories(strix_cuda_logprob PUBLIC include)
    target_compile_options(strix_cuda_logprob PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    list(APPEND strix_ds41_cuda_targets strix_cuda_logprob)
    if(BUILD_TESTING)
        add_executable(strix-cuda-logprob-partials-tests tests/cuda/logprob_partials_tests.cu)
        target_link_libraries(strix-cuda-logprob-partials-tests PRIVATE strix_cuda_logprob strix_core)
        target_compile_options(strix-cuda-logprob-partials-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
        add_test(NAME strix-cuda-logprob-partials-tests COMMAND strix-cuda-logprob-partials-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-logprob-partials-tests)
    endif()
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-core-tests tests/cuda/ds41_core_tests.cu)
        target_link_libraries(strix-cuda-ds41-core-tests PRIVATE strix_ds41_cuda strix_ds41_layer strix_ds41_formats)
        add_test(NAME strix-cuda-ds41-core-tests COMMAND strix-cuda-ds41-core-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-tests)
        # real weights (STRIX_DS41_FIXTURE_GGUF: dense tensors of layers 0-2, embedding, head); skips without it
        add_executable(strix-cuda-ds41-core-real-tests tests/cuda/ds41_core_real_tests.cu)
        target_link_libraries(strix-cuda-ds41-core-real-tests PRIVATE strix_ds41_cuda strix_family_deepseek41)
        add_test(NAME strix-cuda-ds41-core-real-tests COMMAND strix-cuda-ds41-core-real-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-real-tests)
        # timing of the layer core on real weights (not a test)
        add_executable(strix-cuda-ds41-core-bench tests/cuda/ds41_core_bench.cu)
        target_link_libraries(strix-cuda-ds41-core-bench PRIVATE strix_ds41_cuda strix_family_deepseek41)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-bench)
        # replay of a host-forward capture on the device (needs a capture and the weights; not a ctest)
        add_executable(strix-cuda-ds41-core-tap-check tests/cuda/ds41_core_tap_check.cu)
        target_link_libraries(strix-cuda-ds41-core-tap-check PRIVATE strix_ds41_cuda strix_family_deepseek41)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-core-tap-check)
        add_executable(strix-cuda-ds41-decode-fusions-tests tests/cuda/ds41_decode_fusions_tests.cu)
        target_link_libraries(strix-cuda-ds41-decode-fusions-tests PRIVATE strix_ds41_cuda)
        add_test(NAME strix-cuda-ds41-decode-fusions-tests COMMAND strix-cuda-ds41-decode-fusions-tests)
        set_tests_properties(strix-cuda-ds41-decode-fusions-tests PROPERTIES SKIP_RETURN_CODE 77)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-decode-fusions-tests)
    endif()
    # routed experts: decode GEMVs, prefill grouped GEMM, device routes and the combine (docs/models/deepseek-v4.1/cuda-numeric-identity.md)
    add_library(strix_ds41_cuda_moe STATIC src/cuda/models/ds41/ds41_moe.cu src/cuda/models/ds41/ds41_moe_grouped.cu src/cuda/models/ds41/ds41_moe_fast.cu src/cuda/models/ds41/ds41_moe_v2.cu)
    target_link_libraries(strix_ds41_cuda_moe PUBLIC strix_ds41_cuda strix_ds41_formats)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_moe)
    # v1 vs v2 decode kernels on real tensors (bitwise + interleaved timing)
    add_executable(strix-cuda-ds41-kern-bench tests/cuda/ds41_kern_bench.cu src/gguf_reader.cpp)
    target_link_libraries(strix-cuda-ds41-kern-bench PRIVATE strix_ds41_cuda_moe strix_ds41_cuda)
    set(strix_dsrtxk_bench_needs_parts_host ON)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-kern-bench)
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-moe-tests tests/cuda/ds41_moe_tests.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-tests PRIVATE strix_ds41_cuda_moe)
        add_test(NAME strix-cuda-ds41-moe-tests COMMAND strix-cuda-ds41-moe-tests)
        add_executable(strix-cuda-ds41-moe-grouped-tests tests/cuda/ds41_moe_grouped_tests.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-grouped-tests PRIVATE strix_ds41_cuda_moe)
        add_test(NAME strix-cuda-ds41-moe-grouped-tests COMMAND strix-cuda-ds41-moe-grouped-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-moe-grouped-tests)
        add_executable(strix-cuda-ds41-moe-bench tests/cuda/ds41_moe_bench.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-bench PRIVATE strix_ds41_cuda_moe)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-moe-tests strix-cuda-ds41-moe-bench)
        # the prefill fast arm against the exact path on a real layer (needs the experts GGUF; --bench: ms per layer)
        add_executable(strix-cuda-ds41-moe-fast-tests tests/cuda/ds41_moe_fast_tests.cu src/gguf_reader.cpp)
        target_link_libraries(strix-cuda-ds41-moe-fast-tests PRIVATE strix_ds41_cuda_moe)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-moe-fast-tests)
    endif()
    add_library(strix_ds41_cuda_attn STATIC src/cuda/models/ds41/ds41_attention.cu src/cuda/models/ds41/ds41_attention_state.cu src/cuda/models/ds41/ds41_attention_export.cu
        src/cuda/models/ds41/ds41_fast_attention.cu src/cuda/models/ds41/ds41_fast_index.cu)
    target_link_libraries(strix_ds41_cuda_attn PUBLIC strix_ds41_cuda strix_ds41_attention strix_ds41_runtime)  # runtime: the decode state format
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_attn)
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-attention-tests tests/cuda/ds41_attention_tests.cu)
        target_link_libraries(strix-cuda-ds41-attention-tests PRIVATE strix_ds41_cuda_attn strix_ds41_layer strix_ds41_runtime)
        add_test(NAME strix-cuda-ds41-attention-tests COMMAND strix-cuda-ds41-attention-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-attention-tests)
        # timing of the attention ops at prefill shapes (not a test)
        add_executable(strix-cuda-ds41-attention-prefill-bench tests/cuda/ds41_attention_prefill_bench.cu)
        target_link_libraries(strix-cuda-ds41-attention-prefill-bench PRIVATE strix_ds41_cuda_attn)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-attention-prefill-bench)
        # fast prefill attention against the exact kernels (error of o, fused A8 bitwise) and the times of both
        add_executable(strix-cuda-ds41-fast-attention-tests tests/cuda/ds41_fast_attention_tests.cu)
        target_link_libraries(strix-cuda-ds41-fast-attention-tests PRIVATE strix_ds41_cuda_attn)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-fast-attention-tests)
    endif()
    # prefill driver: the DevicePrefill plan on the CUDA kernels, tiered routed experts, Engram rows from the file
    add_library(strix_ds41_cuda_prefill STATIC src/cuda/models/ds41/ds41_prefill_driver.cu src/cuda/models/ds41/ds41_prefill_kernels.cu src/cuda/models/ds41/ds41_prefill_q8.cu
        src/cuda/models/ds41/ds41_dense_fast.cu src/cuda/models/ds41/ds41_expert_streamer.cu src/cuda/models/ds41/ds41_gpu_profile.cu src/ds41_file_reader.cpp)
    target_link_libraries(strix_ds41_cuda_prefill PUBLIC strix_ds41_cuda_attn strix_ds41_cuda_moe strix_ds41_cuda strix_ds41_prefill
        strix_ds41_runtime strix_ds41_layer strix_family_deepseek41 ${CMAKE_DL_LIBS})
    if(STRIX_TIERSTREAM_DIR AND strix_mfile_define)
        target_link_libraries(strix_ds41_cuda_prefill PUBLIC strix_tierstream)
    endif()
    add_executable(strix-cuda-ds41-prefill apps/cuda_ds41_prefill_main.cu)
    target_link_libraries(strix-cuda-ds41-prefill PRIVATE strix_ds41_cuda_prefill)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_prefill strix-cuda-ds41-prefill)
    add_executable(strix-cuda-ds41-decode apps/cuda_ds41_decode_main.cu)
    target_link_libraries(strix-cuda-ds41-decode PRIVATE strix_ds41_cuda_prefill)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-decode)
    # pipeline FRONT on the CUDA decode step: the strix/ds41-pipeline-v1 messages (the portable part of the wire) and its
    # own client socket
    if(STRIX_CUDA_DS41_ONLY)
        add_library(strix_ds41_pipeline_messages STATIC src/ds41_pipeline_wire.cpp src/ds41_pipeline_multi.cpp src/ds41_pipeline_dspark.cpp)
        target_link_libraries(strix_ds41_pipeline_messages PUBLIC strix_family_deepseek41)
        set(strix_ds41_pipeline_for_front strix_ds41_pipeline_messages)
    else()
        set(strix_ds41_pipeline_for_front strix_ds41_pipeline)
    endif()
    add_executable(strix-cuda-ds41-pipeline-front apps/cuda_ds41_pipeline_front_main.cu)
    target_link_libraries(strix-cuda-ds41-pipeline-front PRIVATE strix_ds41_cuda_prefill ${strix_ds41_pipeline_for_front})
    if(WIN32)
        target_link_libraries(strix-cuda-ds41-pipeline-front PRIVATE ws2_32 winmm avrt)
    endif()
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-pipeline-front)
    # the prefill resident behind an HTTP/1.1 service (POST /v1/prefill, state handoff to a decode server's import route)
    add_executable(strix-cuda-ds41-prefill-service apps/cuda_ds41_prefill_service_main.cu src/serve/json.cpp)
    target_include_directories(strix-cuda-ds41-prefill-service PRIVATE src/serve)
    target_link_libraries(strix-cuda-ds41-prefill-service PRIVATE strix_ds41_cuda_prefill strix_core strix_ds41_rtx_profile)
    if(WIN32)
        target_link_libraries(strix-cuda-ds41-prefill-service PRIVATE ws2_32 bcrypt)
    endif()
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-prefill-service)
    # FRONT of several requests (strix/ds41-pipeline-multi-v1): sequence slots of one CudaPrefill, driven by a batch TAIL
    add_executable(strix-cuda-ds41-multi-front apps/cuda_ds41_multi_front_main.cu)
    target_link_libraries(strix-cuda-ds41-multi-front PRIVATE strix_ds41_cuda_prefill ${strix_ds41_pipeline_for_front})
    if(WIN32)
        target_link_libraries(strix-cuda-ds41-multi-front PRIVATE ws2_32 winmm avrt)
    endif()
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-multi-front)
    if(BUILD_TESTING)
        # prefill-shape lane-dot kernels bitwise against the decode kernels and the host references (--bench: times)
        add_executable(strix-cuda-ds41-prefill-kernels-tests tests/cuda/ds41_prefill_kernels_tests.cu)
        target_link_libraries(strix-cuda-ds41-prefill-kernels-tests PRIVATE strix_ds41_cuda_prefill)
        add_test(NAME strix-cuda-ds41-prefill-kernels-tests COMMAND strix-cuda-ds41-prefill-kernels-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-prefill-kernels-tests)
        # prefill-shape Q8_0 x A8 sites bitwise against the tiled kernel (--bench: times)
        add_executable(strix-cuda-ds41-prefill-q8-tests tests/cuda/ds41_prefill_q8_tests.cu)
        target_link_libraries(strix-cuda-ds41-prefill-q8-tests PRIVATE strix_ds41_cuda_prefill)
        add_test(NAME strix-cuda-ds41-prefill-q8-tests COMMAND strix-cuda-ds41-prefill-q8-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-prefill-q8-tests)
        # sustained RAM -> device bandwidth of the expert transfer shapes (not a test)
        add_executable(strix-cuda-ds41-xfer-bench tests/cuda/ds41_xfer_bench.cu)
        target_link_libraries(strix-cuda-ds41-xfer-bench PRIVATE strix_ds41_cuda)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-xfer-bench)
        # fast (relaxed FP32) dense sites against the exact kernels: errors per site (--bench: times)
        add_executable(strix-cuda-ds41-dense-fast-tests tests/cuda/ds41_dense_fast_tests.cu)
        target_link_libraries(strix-cuda-ds41-dense-fast-tests PRIVATE strix_ds41_cuda_prefill)
        add_test(NAME strix-cuda-ds41-dense-fast-tests COMMAND strix-cuda-ds41-dense-fast-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-dense-fast-tests)
    endif()
    # the Engram pages a prompt reads (to complete a partial model copy for that prompt)
    add_executable(strix-ds41-engram-pages apps/ds41_engram_pages_main.cpp)
    target_link_libraries(strix-ds41-engram-pages PRIVATE strix_ds41_layer strix_family_deepseek41)
    list(APPEND strix_ds41_cuda_targets strix-ds41-engram-pages)
    # parts-host: resident-expert routes, the seeded combine and the layer driver with expert servers
    add_library(strix_ds41_cuda_parts_host STATIC src/cuda/models/ds41/ds41_parts_host.cu src/cuda/models/ds41/ds41_parts_host_driver.cpp)
    target_link_libraries(strix_ds41_cuda_parts_host PUBLIC strix_ds41_cuda_moe strix_ds41_xrpc)
    if(strix_dsrtxk_bench_needs_parts_host)
        target_link_libraries(strix-cuda-ds41-kern-bench PRIVATE strix_ds41_cuda_parts_host)
    endif()
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_parts_host)
    # residency from a hot-expert placement (static + dynamic slots) and the decode replay over recorded routes
    add_library(strix_ds41_cuda_hot_residency STATIC src/cuda/models/ds41/ds41_hot_residency.cpp src/cuda/models/ds41/ds41_hot_site.cu src/gguf_reader.cpp)
    target_link_libraries(strix_ds41_cuda_hot_residency PUBLIC strix_ds41_cuda_parts_host strix_ds41_hot_placement CUDA::cuda_driver)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_hot_residency)
    add_executable(strix-cuda-ds41-parts-host-check apps/cuda_ds41_parts_host_check.cpp)
    target_link_libraries(strix-cuda-ds41-parts-host-check PRIVATE strix_ds41_cuda_hot_residency strix_ds41_cuda)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-parts-host-check)
    # decode with dense + hot experts: CudaPrefill::step with the parts-host at every MoE site
    add_executable(strix-cuda-ds41-hot-decode apps/cuda_ds41_hot_decode.cu)
    target_link_libraries(strix-cuda-ds41-hot-decode PRIVATE strix_ds41_cuda_hot_residency strix_ds41_cuda_prefill)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-hot-decode)
    # independent replica with streamed routed experts (design notes) on tierstream's device-gated decode mode
    # (device arena + pinned host tier + lookahead): needs a CUDA build of tierstream (STRIX_TIERSTREAM_DIR)
    if(STRIX_TIERSTREAM_DIR)
        target_link_libraries(strix_tierstream INTERFACE CUDA::cudart CUDA::cuda_driver)
        add_library(strix_ds41_cuda_tier STATIC src/cuda/models/ds41/ds41_tier_site.cu)
        target_link_libraries(strix_ds41_cuda_tier PUBLIC strix_ds41_cuda_hot_residency strix_tierstream)
        list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_tier)
        add_executable(strix-cuda-ds41-replica apps/cuda_ds41_replica.cu)
        target_link_libraries(strix-cuda-ds41-replica PRIVATE strix_ds41_cuda_tier strix_ds41_cuda_prefill strix_ds41_cuda_dspark
                              strix_ds41_rtx_profile)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${STRIX_TIERSTREAM_LIBRARY}")
        file(SHA256 "${STRIX_TIERSTREAM_LIBRARY}" strix_tierstream_sha256)
        target_compile_definitions(strix-cuda-ds41-replica PRIVATE STRIX_TIERSTREAM_SHA256="${strix_tierstream_sha256}")
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-replica)
        if(BUILD_TESTING)
            add_executable(strix-cuda-ds41-rtx-schedule-tests tests/ds41_rtx_schedule_tests.cpp)
            target_include_directories(strix-cuda-ds41-rtx-schedule-tests PRIVATE include)
            target_compile_features(strix-cuda-ds41-rtx-schedule-tests PRIVATE cxx_std_20)
            # whole-device budgets: tierstream's ss_budget_plan is pure CPU; linked from the tree's CPU build when it
            # has one (no CUDA runtime in a CPU test), else from the configured library
            set(strix_tierstream_cpu_lib "${STRIX_TIERSTREAM_DIR}/build-cpu/tierstream${CMAKE_STATIC_LIBRARY_SUFFIX}")
            if(EXISTS "${strix_tierstream_cpu_lib}")
                target_link_libraries(strix-cuda-ds41-rtx-schedule-tests PRIVATE "${strix_tierstream_cpu_lib}" Threads::Threads
                                      "$<$<PLATFORM_ID:Windows>:avrt;winmm;bcrypt>")
                target_include_directories(strix-cuda-ds41-rtx-schedule-tests PRIVATE "${STRIX_TIERSTREAM_DIR}/include")
            else()
                target_link_libraries(strix-cuda-ds41-rtx-schedule-tests PRIVATE strix_tierstream)
            endif()
            add_test(NAME strix-cuda-ds41-rtx-schedule-tests COMMAND strix-cuda-ds41-rtx-schedule-tests)
            set_tests_properties(strix-cuda-ds41-rtx-schedule-tests PROPERTIES LABELS "cpu;schedule")
            add_executable(strix-cuda-ds41-serve-tests tests/ds41_replica_serve_tests.cpp)
            target_compile_features(strix-cuda-ds41-serve-tests PRIVATE cxx_std_20)
            add_test(NAME strix-cuda-ds41-serve-tests COMMAND strix-cuda-ds41-serve-tests)
            set_tests_properties(strix-cuda-ds41-serve-tests PROPERTIES LABELS "cpu;serve")
            add_executable(strix-cuda-ds41-census-tests tests/ds41_replica_census_tests.cpp)
            target_include_directories(strix-cuda-ds41-census-tests PRIVATE include)
            target_compile_features(strix-cuda-ds41-census-tests PRIVATE cxx_std_20)
            add_test(NAME strix-cuda-ds41-census-tests COMMAND strix-cuda-ds41-census-tests)
            set_tests_properties(strix-cuda-ds41-census-tests PROPERTIES LABELS "cpu;census")
            add_executable(strix-ds41-rtx-drafter-select-tests tests/ds41_rtx_drafter_select_tests.cpp)
            target_include_directories(strix-ds41-rtx-drafter-select-tests PRIVATE include)
            target_compile_features(strix-ds41-rtx-drafter-select-tests PRIVATE cxx_std_20)
            add_test(NAME strix-ds41-rtx-drafter-select-tests COMMAND strix-ds41-rtx-drafter-select-tests)
            set_tests_properties(strix-ds41-rtx-drafter-select-tests PROPERTIES LABELS "cpu;drafter")
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
            add_custom_target(strix-cuda-ds41-config-tests DEPENDS strix-cuda-ds41-replica)
            add_test(NAME strix-cuda-ds41-config-tests
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/cuda_ds41_config_test.py
                    $<TARGET_FILE:strix-cuda-ds41-replica>)
            set_tests_properties(strix-cuda-ds41-config-tests PROPERTIES LABELS "cpu;config" TIMEOUT 60)
        endif()

    endif()
    # DSpark drafter on CUDA (design notes): the host reference library (built here in the CUDA-only tree) and
    # the device drafter, bitwise against the host reference on the real weights
    if(STRIX_CUDA_DS41_ONLY)
        add_library(strix_ds41_dspark STATIC src/ds41_dspark.cpp src/ds41_dspark_files.cpp src/ds41_dspark_driver.cpp src/ds41_dspark_vocab.cpp)
        target_link_libraries(strix_ds41_dspark PUBLIC strix_ds41_formats strix_core Threads::Threads)
        list(APPEND strix_ds41_cuda_targets strix_ds41_dspark)
        # the window controller's survival table + token identity under any table (CPU only)
        add_executable(strix-ds41-dspark-table-tests tests/ds41_dspark_table_tests.cpp)
        target_link_libraries(strix-ds41-dspark-table-tests PRIVATE strix_ds41_dspark)
        list(APPEND strix_ds41_cuda_targets strix-ds41-dspark-table-tests)
    endif()
    add_library(strix_ds41_cuda_dspark STATIC src/cuda/models/ds41/ds41_dspark.cu)
    target_link_libraries(strix_ds41_cuda_dspark PUBLIC strix_ds41_cuda_moe strix_ds41_cuda strix_ds41_dspark)
    list(APPEND strix_ds41_cuda_targets strix_ds41_cuda_dspark)
    add_executable(strix-cuda-ds41-dspark-tests tests/cuda/ds41_dspark_tests.cu)
    target_link_libraries(strix-cuda-ds41-dspark-tests PRIVATE strix_ds41_cuda_dspark)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-dspark-tests)
    add_executable(strix-cuda-ds41-hot-replay apps/cuda_ds41_hot_replay.cu)
    target_link_libraries(strix-cuda-ds41-hot-replay PRIVATE strix_ds41_cuda_hot_residency strix_ds41_cuda)
    list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-hot-replay)
    if(BUILD_TESTING)
        add_executable(strix-cuda-ds41-parts-host-tests tests/cuda/ds41_parts_host_tests.cu)
        target_link_libraries(strix-cuda-ds41-parts-host-tests PRIVATE strix_ds41_cuda_parts_host)
        add_test(NAME strix-cuda-ds41-parts-host-tests COMMAND strix-cuda-ds41-parts-host-tests)
        list(APPEND strix_ds41_cuda_targets strix-cuda-ds41-parts-host-tests)
    endif()
    # Strict arithmetic: no contraction into FMA, IEEE division and square root, denormals kept.
    if(STRIX_ENABLE_CUDA_TRELLIS)
        add_library(strix_cuda_trellis STATIC ${STRIX_CUDA_CORE_TRELLIS_SOURCES})
        target_link_libraries(strix_cuda_trellis PUBLIC strix_ds41_cuda)
        list(APPEND strix_ds41_cuda_targets strix_cuda_trellis)
        # the decode step's trellis shared expert (CudaPrefillOptions::trellis_shexp)
        target_link_libraries(strix_ds41_cuda_prefill PUBLIC strix_cuda_trellis)
        target_compile_definitions(strix_ds41_cuda_prefill PRIVATE STRIX_DS41_CUDA_TRELLIS=1)
        if(BUILD_TESTING)
            add_executable(strix-cuda-trellis-tests tests/cuda/trellis_tests.cu)
            target_link_libraries(strix-cuda-trellis-tests PRIVATE strix_cuda_trellis)
            list(APPEND strix_ds41_cuda_targets strix-cuda-trellis-tests)
            # the decode step's trellis shared expert vs the pair's per-row outputs (a dump from the AMD node)
            add_executable(strix-cuda-trellis-shexp-dump-check tests/cuda/trellis_shexp_dump_check.cu)
            target_link_libraries(strix-cuda-trellis-shexp-dump-check PRIVATE strix_cuda_trellis)
            list(APPEND strix_ds41_cuda_targets strix-cuda-trellis-shexp-dump-check)
            add_test(NAME strix-cuda-trellis-contract-tests COMMAND strix-cuda-trellis-tests --contract-only)
            # Native runs require explicit external fixtures; do not silently
            # pass/skip a missing oracle or select a shared GPU during configure.
        endif()
    endif()
    foreach(target ${strix_ds41_cuda_targets})
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CXX_STANDARD 20)
        if(MSVC)
            # MSVC host compiler: /fp:precise (no contraction, see STRIX_CUDA_DS41_ONLY) in both the .cu host
            # passes and the plain C++ sources; the device flags are the same as on Linux.
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -prec-div=true -prec-sqrt=true -ftz=false
                    -Xcompiler=/fp:precise,/Zc:__cplusplus,/Zc:preprocessor,/permissive-,/utf-8,/bigobj,/EHsc>
                $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
        else()
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -prec-div=true -prec-sqrt=true -ftz=false -Xcompiler=-ffp-contract=off,-Wall,-Wextra>
                $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off -Wall -Wextra>)
        endif()
    endforeach()
endif()
endif()  # STRIX_FAMILY_DEEPSEEK41

# minimax-m3 bring-up: MSA and fused post-exchange host references, their gfx1151 kernels, and the layer gates.
if(STRIX_FAMILY_MINIMAX)
    # Defaults-on resolver (host only): the default-ON items, STRIX_MM_BASELINE=1 master switch, --print-effective-config.
    option(STRIX_MINIMAX_GU4_SCALAR "MiniMax gu4 MoE expert addressing (exact, 6a8d4ab6); OFF = the former reader" ON)
    if(STRIX_MINIMAX_GU4_SCALAR)
        set(strix_minimax_gu4 1)
    else()
        set(strix_minimax_gu4 0)
    endif()
    add_library(strix_minimax_defaults STATIC src/minimax/engine/mm_defaults.cpp)
    target_include_directories(strix_minimax_defaults PUBLIC src include)
    target_compile_features(strix_minimax_defaults PUBLIC cxx_std_20)
    target_compile_definitions(strix_minimax_defaults PRIVATE STRIX_MM_GU4_SCALAR=${strix_minimax_gu4})
    strix_target_warnings(strix_minimax_defaults)
    if(BUILD_TESTING)
        add_executable(strix-minimax-index-layout-tests tests/minimax_index_layout_tests.cpp)
        target_include_directories(strix-minimax-index-layout-tests PRIVATE src include)
        target_compile_features(strix-minimax-index-layout-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-minimax-index-layout-tests)
        add_test(NAME strix-minimax-index-layout-tests COMMAND strix-minimax-index-layout-tests)
        add_executable(strix-minimax-auto-tests tests/minimax_auto_tests.cpp src/serve/json.cpp)
        target_include_directories(strix-minimax-auto-tests PRIVATE src include)
        target_compile_features(strix-minimax-auto-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-minimax-auto-tests)
        add_test(NAME strix-minimax-auto-tests COMMAND strix-minimax-auto-tests)
        # Chunk-invariant order contract: host reference of the bound K chain across chunk sizes and tails.
        add_executable(strix-chunk-order-contract-tests tests/chunk_order_contract_tests.cpp)
        target_include_directories(strix-chunk-order-contract-tests PRIVATE src include)
        target_compile_features(strix-chunk-order-contract-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-chunk-order-contract-tests)
        add_test(NAME strix-chunk-order-contract-tests COMMAND strix-chunk-order-contract-tests)
        # Pure host policy census; source receipts reject unaudited selectors.
        add_executable(strix-minimax-route-identity-tests tests/minimax_route_identity_tests.cpp)
        target_link_libraries(strix-minimax-route-identity-tests PRIVATE strix_minimax_defaults)
        target_compile_definitions(strix-minimax-route-identity-tests PRIVATE STRIX_ROUTE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        strix_target_warnings(strix-minimax-route-identity-tests)
        if(NOT MSVC)
            target_compile_options(strix-minimax-route-identity-tests PRIVATE -Wsign-conversion)
        endif()
        add_test(NAME strix-minimax-route-identity-tests COMMAND strix-minimax-route-identity-tests)
        add_test(NAME strix-minimax-route-source-audit COMMAND strix-minimax-route-identity-tests --audit-sources)
        set_tests_properties(strix-minimax-route-source-audit PROPERTIES LABELS "cpu;profile" TIMEOUT 60)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME strix-minimax-route-source-audit-negative
                COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_route_source_audit_tests.py"
                    "$<TARGET_FILE:strix-minimax-route-identity-tests>")
            set_tests_properties(strix-minimax-route-source-audit-negative PROPERTIES LABELS "cpu;profile" TIMEOUT 120)
        endif()
        # Separate queue object: test clock/work counters never enter production or DeepSeek.
        add_executable(strix-minimax-lead-tests tests/minimax_lead_tests.cpp src/serve/engine.cpp src/serve/json.cpp
            src/minimax/dspark/minimax_dspark_host.cpp)
        target_compile_definitions(strix-minimax-lead-tests PRIVATE STRIX_SERVE_QUEUE_TEST=1)
        if(MSVC)
            # Host-only spelling of the same rounding operation; no device TU.
            target_compile_definitions(strix-minimax-lead-tests PRIVATE __builtin_rintf=rintf)
        endif()
        target_link_libraries(strix-minimax-lead-tests PRIVATE strix_lookup strix_session_owner)
        target_include_directories(strix-minimax-lead-tests PRIVATE src include)
        target_compile_features(strix-minimax-lead-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-minimax-lead-tests)
        add_test(NAME strix-minimax-lead-tests COMMAND strix-minimax-lead-tests)
        add_executable(strix-minimax-sched-tests tests/minimax_sched_tests.cpp)
        target_link_libraries(strix-minimax-sched-tests PRIVATE strix_serve)
        target_include_directories(strix-minimax-sched-tests PRIVATE src include)
        strix_target_warnings(strix-minimax-sched-tests)
        add_test(NAME strix-minimax-sched-tests COMMAND strix-minimax-sched-tests)
        add_executable(strix-minimax-defaults-tests tests/minimax_defaults_tests.cpp)
        target_link_libraries(strix-minimax-defaults-tests PRIVATE strix_minimax_defaults)
        strix_target_warnings(strix-minimax-defaults-tests)
        add_test(NAME strix-minimax-defaults-tests COMMAND strix-minimax-defaults-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_production_receipt.txt)
    endif()
    add_library(strix_minimax_model_description STATIC src/minimax/model_description.cpp src/minimax/model_graph.cpp)
    target_include_directories(strix_minimax_model_description PUBLIC include)
    target_link_libraries(strix_minimax_model_description PUBLIC strix_model_description strix_serve_json strix_model_graph strix_minimax_defaults)
    target_compile_features(strix_minimax_model_description PUBLIC cxx_std_20)
    strix_target_warnings(strix_minimax_model_description)
    if(BUILD_TESTING)
        add_executable(strix-minimax-model-description-tests tests/minimax_model_description_tests.cpp)
        target_link_libraries(strix-minimax-model-description-tests PRIVATE strix_minimax_model_description strix_minimax_assets)
        strix_target_warnings(strix-minimax-model-description-tests)
        add_test(NAME strix-minimax-model-description-tests COMMAND strix-minimax-model-description-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/models/minimax-m3.1-flash.toml)
        add_executable(strix-minimax-model-graph-tests tests/minimax_model_graph_tests.cpp)
        target_link_libraries(strix-minimax-model-graph-tests PRIVATE strix_minimax_model_description)
        strix_target_warnings(strix-minimax-model-graph-tests)
        add_test(NAME strix-minimax-model-graph-tests COMMAND strix-minimax-model-graph-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/models/minimax-m3.1-flash.toml)
    endif()
    add_library(strix_minimax_assets STATIC src/minimax/model_assets.cpp)
    target_include_directories(strix_minimax_assets PUBLIC include)
    target_link_libraries(strix_minimax_assets PUBLIC strix_serve_json strix_minimax_model_description)
    target_compile_features(strix_minimax_assets PUBLIC cxx_std_20)
    strix_target_warnings(strix_minimax_assets)

    add_library(strix_minimax_session STATIC src/minimax/session/session_cache.cpp)
    target_include_directories(strix_minimax_session PUBLIC include)
    target_link_libraries(strix_minimax_session PUBLIC strix_core strix_minimax_assets strix_session)
    target_compile_features(strix_minimax_session PUBLIC cxx_std_20)
    strix_target_warnings(strix_minimax_session)
    if(BUILD_TESTING)
        add_executable(strix-minimax-session-tests tests/minimax_session_cache_tests.cpp)
        target_link_libraries(strix-minimax-session-tests PRIVATE strix_minimax_session Threads::Threads)
        if(NOT WIN32)
            # Exercise the production pack reader with tiny host-only fixtures.
            target_sources(strix-minimax-session-tests PRIVATE src/minimax/engine/mm_pack.cpp)
            target_include_directories(strix-minimax-session-tests PRIVATE src)
        endif()
        strix_target_warnings(strix-minimax-session-tests)
        add_test(NAME strix-minimax-session-tests COMMAND strix-minimax-session-tests)
        add_executable(strix-minimax-session-api-tests tests/minimax_session_api_tests.cpp)
        target_link_libraries(strix-minimax-session-api-tests PRIVATE strix_serve Threads::Threads)
        target_include_directories(strix-minimax-session-api-tests PRIVATE src)
        strix_target_warnings(strix-minimax-session-api-tests)
        add_test(NAME strix-minimax-session-api-tests COMMAND strix-minimax-session-api-tests)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME minimax-session-gate-helpers COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_session_cache_gate_tests.py)
            set_tests_properties(minimax-session-gate-helpers PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
            add_test(NAME minimax-session-phasing COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_session_phasing_tests.py)
            add_test(NAME minimax-session-default-observer COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_session_default_probe_tests.py)
            # The prefix-reuse exactness gate itself needs the running pair; only its comparison logic is checked here.
            add_test(NAME minimax-prefix-cache-gate-selftest COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_prefix_cache_gpu_gate.py self-test)
        endif()
    endif()
    # MiniMax tokenizer (tokenizer.json BPE: o200k-style split, NFC, byte level) and chat template, host only.
    add_library(strix_minimax_serve STATIC src/minimax/serve/minimax_tokenizer.cpp src/minimax/serve/minimax_chat.cpp
        src/minimax/serve/minimax_observe.cpp src/minimax/serve/minimax_api.cpp)
    target_include_directories(strix_minimax_serve PUBLIC src include)
    target_compile_features(strix_minimax_serve PUBLIC cxx_std_20)
    target_link_libraries(strix_minimax_serve PUBLIC strix_serve strix_lookup)
    strix_target_warnings(strix_minimax_serve)

    add_library(strix_minimax_session_service STATIC
        src/minimax/serve/session_service.cpp src/minimax/serve/serve_runtime.cpp)
    target_link_libraries(strix_minimax_session_service PUBLIC strix_minimax_serve
        strix_minimax_model_description strix_minimax_defaults strix_engine_deploy strix_session_runtime)
    target_include_directories(strix_minimax_session_service PUBLIC src include)
    target_compile_features(strix_minimax_session_service PUBLIC cxx_std_20)
    target_compile_definitions(strix_minimax_session_service PRIVATE STRIX_SERVE_REGISTRY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/minimax/serve/session_engine.cpp")
        target_sources(strix_minimax_session_service PRIVATE src/minimax/serve/session_engine.cpp)
        if(TARGET strix-minimax-lead-tests)
            target_sources(strix-minimax-lead-tests PRIVATE src/minimax/serve/session_engine.cpp)
        endif()
    else()
        message(STATUS "MiniMax shared owner source absent: native integration pending")
    endif()
    if(MSVC)
        target_compile_definitions(strix_minimax_session_service PUBLIC __builtin_rintf=rintf)
        target_compile_definitions(strix_minimax_session_service PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix_minimax_session_service)
    if(BUILD_TESTING)
        add_subdirectory(tests/minimax_service)
    endif()
    if(BUILD_TESTING)
        add_executable(strix-minimax-tool-choice-none-tests tests/serve_tool_choice_none_tests.cpp)
        target_compile_definitions(strix-minimax-tool-choice-none-tests PRIVATE STRIX_TOOLNONE_MINIMAX)
        target_link_libraries(strix-minimax-tool-choice-none-tests PRIVATE strix_minimax_serve)
        strix_target_warnings(strix-minimax-tool-choice-none-tests)
        add_test(NAME strix-minimax-tool-choice-none-tests COMMAND strix-minimax-tool-choice-none-tests)
        add_executable(strix-minimax-tokenizer-tests tests/minimax_tokenizer_tests.cpp)
        target_compile_definitions(strix-minimax-tokenizer-tests PRIVATE
            STRIX_MINIMAX_MEMO_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/minimax_tokenizer_memo.json")
        target_link_libraries(strix-minimax-tokenizer-tests PRIVATE strix_minimax_serve)
        strix_target_warnings(strix-minimax-tokenizer-tests)
        set(STRIX_MINIMAX_TOKENIZER_JSON "" CACHE FILEPATH "MiniMax tokenizer.json (optional; enables encode/decode checks)")
        add_test(NAME strix-minimax-tokenizer-tests COMMAND strix-minimax-tokenizer-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/src/minimax/serve/fixtures/minimax_tokenizer_fixtures.json ${STRIX_MINIMAX_TOKENIZER_JSON})
        # serve HTTP API and observability (request ids, status ring, metrics text, armed histograms), host only
        add_executable(strix-minimax-observe-tests tests/minimax_observe_tests.cpp)
        target_link_libraries(strix-minimax-observe-tests PRIVATE strix_minimax_serve)
        strix_target_warnings(strix-minimax-observe-tests)
        add_test(NAME strix-minimax-observe-tests COMMAND strix-minimax-observe-tests)
        add_executable(strix-minimax-api-tests tests/minimax_api_tests.cpp)
        target_link_libraries(strix-minimax-api-tests PRIVATE strix_minimax_serve)
        strix_target_warnings(strix-minimax-api-tests)
        add_test(NAME strix-minimax-api-tests COMMAND strix-minimax-api-tests)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME strix-minimax-serve-check-tests
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tools/minimax/serve_observe_check_test.py)
        endif()
    endif()
    add_library(strix_minimax_msa STATIC src/minimax_msa.cpp src/minimax_moe.cpp)
    target_include_directories(strix_minimax_msa PUBLIC include)
    target_compile_features(strix_minimax_msa PUBLIC cxx_std_20)
    strix_target_warnings(strix_minimax_msa)
    if(NOT MSVC)
        target_compile_options(strix_minimax_msa PRIVATE -ffp-contract=off)
    endif()
    if(BUILD_TESTING)
        add_executable(strix-minimax-msa-tests tests/minimax_msa_tests.cpp)
        target_link_libraries(strix-minimax-msa-tests PRIVATE strix_minimax_msa)
        strix_target_warnings(strix-minimax-msa-tests)
        if(NOT MSVC)
            target_compile_options(strix-minimax-msa-tests PRIVATE -ffp-contract=off)
        endif()
        add_test(NAME strix-minimax-msa-tests COMMAND strix-minimax-msa-tests)
        add_executable(strix-minimax-inblock-layout-tests tests/minimax_inblock_layout_tests.cpp)
        target_compile_features(strix-minimax-inblock-layout-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-minimax-inblock-layout-tests)
        add_test(NAME strix-minimax-inblock-layout-tests COMMAND strix-minimax-inblock-layout-tests)
    endif()
    if(STRIX_ENABLE_HIP_RUNTIME)
        add_library(strix_minimax_msa_hip STATIC src/hip_minimax_msa.hip src/hip_minimax_moe.hip)
        target_link_libraries(strix_minimax_msa_hip PUBLIC strix_minimax_msa "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix_minimax_msa_hip PUBLIC include "${STRIX_HIP_INCLUDE_DIR}")
        set(strix_minimax_msa_hip_targets strix_minimax_msa_hip)
        # Optimized decode MoE / dense EXL3 kernels (src/minimax/moe/, minimax/moe-opt 92e664f8):
        # self-contained, no library link.
        add_library(strix_minimax_moe_opt_hip STATIC src/minimax/moe/mmx_moe.hip src/minimax/moe/mmx_prefill.hip
            src/minimax/rtxpf/fp32_identity.hip)
        # FP32 prefill identity: no contraction, the RTX PRO build must produce the same bits.
        set_source_files_properties(src/minimax/rtxpf/fp32_identity.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
        target_link_libraries(strix_minimax_moe_opt_hip PUBLIC strix_minimax_msa "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix_minimax_moe_opt_hip PUBLIC src include "${STRIX_HIP_INCLUDE_DIR}")
        target_compile_definitions(strix_minimax_moe_opt_hip PRIVATE STRIX_MM_GU4_SCALAR=${strix_minimax_gu4})
        list(APPEND strix_minimax_msa_hip_targets strix_minimax_moe_opt_hip)
        # The one strix-trellis tree of the MiniMax build: include/strix_trellis.h + build-qualification-v03/libstrix_trellis.a.
        set(STRIX_MINIMAX_TRELLIS_DIR ""
            CACHE PATH "strix-trellis 0.3 tree: include/strix_trellis.h and build-qualification-v03/libstrix_trellis.a")
        set(strix_minimax_trellis_lib "${STRIX_MINIMAX_TRELLIS_DIR}/build-qualification-v03/libstrix_trellis.a")
        # FAST_I4 (STRIX_MMX_MODEL_FORMAT=fast_i4, opt-in, lossy): the grouped IU4 kernel and convert_exl3_iu4 of the one
        # shared tiled core (src/core, archive strix_tiled_core; GLM and DS41 use the same copy).
        target_link_libraries(strix_minimax_moe_opt_hip PUBLIC strix_tiled_core)
        set_property(SOURCE src/minimax/moe/mmx_prefill.hip APPEND PROPERTY COMPILE_DEFINITIONS STRIX_MMX_IU4_LIB=1)
        if(BUILD_TESTING)
            add_executable(strix-hip-minimax-msa-tests tests/hip_minimax_msa_tests.hip)
            target_link_libraries(strix-hip-minimax-msa-tests PRIVATE strix_minimax_msa_hip)
            add_test(NAME strix-hip-minimax-msa-tests COMMAND strix-hip-minimax-msa-tests)
            list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-msa-tests)
            add_executable(strix-hip-minimax-moe-tests tests/hip_minimax_moe_tests.hip)
            target_link_libraries(strix-hip-minimax-moe-tests PRIVATE strix_minimax_msa_hip)
            add_test(NAME strix-hip-minimax-moe-tests COMMAND strix-hip-minimax-moe-tests)
            list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-moe-tests)
            add_executable(strix-hip-mm-router-wmma-tests tests/hip_mm_router_wmma_tests.hip)  # FAST mm-pf-router-wmma-v1: forms, fallback, drift, timing
            target_link_libraries(strix-hip-mm-router-wmma-tests PRIVATE strix_minimax_msa_hip)
            add_test(NAME strix-hip-mm-router-wmma-tests COMMAND strix-hip-mm-router-wmma-tests)
            list(APPEND strix_minimax_msa_hip_targets strix-hip-mm-router-wmma-tests)
            add_executable(strix-hip-mm-fp32id-tests tests/hip_mm_fp32_identity_tests.hip)
            target_link_libraries(strix-hip-mm-fp32id-tests PRIVATE strix_minimax_moe_opt_hip)
            target_compile_options(strix-hip-mm-fp32id-tests PRIVATE -ffp-contract=off)
            add_test(NAME strix-hip-mm-fp32id-tests COMMAND strix-hip-mm-fp32id-tests)
            list(APPEND strix_minimax_msa_hip_targets strix-hip-mm-fp32id-tests)
        endif()
        # Layer gates G3 (one node) and G4 (two-rank TP): only with an external strix-trellis 0.3 tree (header + archive).
        if(BUILD_TESTING AND EXISTS "${STRIX_MINIMAX_TRELLIS_DIR}/include/strix_trellis.h" AND EXISTS "${strix_minimax_trellis_lib}")
            add_executable(strix-hip-minimax-g3 tests/hip_minimax_g3.hip)
            target_link_libraries(strix-hip-minimax-g3 PRIVATE strix_minimax_msa_hip "${strix_minimax_trellis_lib}")
            target_include_directories(strix-hip-minimax-g3 PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
            list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-g3)
            add_executable(strix-hip-mmx-moe-bench tests/hip_mmx_moe_bench.hip)
            target_link_libraries(strix-hip-mmx-moe-bench PRIVATE strix_minimax_moe_opt_hip strix_minimax_msa_hip "${strix_minimax_trellis_lib}")
            target_include_directories(strix-hip-mmx-moe-bench PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
            list(APPEND strix_minimax_msa_hip_targets strix-hip-mmx-moe-bench)
            if(TARGET strix_dist)
                add_executable(strix-hip-minimax-g4 tests/hip_minimax_g4.hip)
                target_link_libraries(strix-hip-minimax-g4 PRIVATE strix_minimax_msa_hip strix_dist "${strix_minimax_trellis_lib}")
                target_include_directories(strix-hip-minimax-g4 PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
                list(APPEND strix_minimax_msa_hip_targets strix-hip-minimax-g4)
            endif()
        endif()
        # MiniMax engine (fork of the DS41 TP2 engine): resident TP2 forward, gates, sessions, batched decode.
        if(TARGET strix_dist AND EXISTS "${STRIX_MINIMAX_TRELLIS_DIR}/include/strix_trellis.h" AND EXISTS "${strix_minimax_trellis_lib}")
            # Prefill exchange module (src/minimax/pfx): chunked wire, CRC lanes, in-place slots.
            add_library(strix_minimax_pfx STATIC src/minimax/pfx/pf_exchange.hip src/minimax/pfx/pf_crc.cpp)
            target_link_libraries(strix_minimax_pfx PUBLIC strix_dist "${STRIX_HIP_RUNTIME_LIBRARY}" Threads::Threads)
            target_include_directories(strix_minimax_pfx PUBLIC include src "${STRIX_HIP_INCLUDE_DIR}")
            set_target_properties(strix_minimax_pfx PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_features(strix_minimax_pfx PUBLIC cxx_std_20)
            target_compile_options(strix_minimax_pfx PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${STRIX_HIP_WAVE32_FLAG}>)
            strix_target_warnings(strix_minimax_pfx)
            if(BUILD_TESTING)
                add_executable(strix-hip-mm-pfx-tests tests/hip_mm_pfx_tests.hip)
                target_link_libraries(strix-hip-mm-pfx-tests PRIVATE strix_minimax_pfx)
                set_target_properties(strix-hip-mm-pfx-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                target_compile_options(strix-hip-mm-pfx-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG})
            endif()
            add_library(strix_minimax_engine STATIC src/minimax/engine/engine.hip src/minimax/engine/draft_lin.hip src/minimax/engine/mm_pack.cpp
                src/minimax/attn/msa_fused.hip src/minimax/attn/index_proj.hip src/minimax/xfer/minimax_xfer.hip)
            target_link_libraries(strix_minimax_engine PUBLIC strix_xport strix_minimax_assets strix_minimax_defaults strix_minimax_msa_hip strix_minimax_moe_opt_hip strix_minimax_pfx strix_dist strix_deploy strix_engine_deploy "${strix_minimax_trellis_lib}")
            target_include_directories(strix_minimax_engine PUBLIC include src "${STRIX_HIP_INCLUDE_DIR}"
                PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
            set_target_properties(strix_minimax_engine PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_features(strix_minimax_engine PUBLIC cxx_std_20)
            target_compile_options(strix_minimax_engine PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${STRIX_HIP_WAVE32_FLAG} -fno-fast-math
                -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee>)
            strix_target_warnings(strix_minimax_engine)
            if(BUILD_TESTING)  # FAST shared WMMA indexer core: gate + timing against the exact Q64 index kernel
                add_executable(strix-hip-idx-wmma-mm-tests tests/hip_idx_wmma_tests.hip)
                target_link_libraries(strix-hip-idx-wmma-mm-tests PRIVATE strix_minimax_engine)
                target_compile_definitions(strix-hip-idx-wmma-mm-tests PRIVATE IDXW_MM)
                set_target_properties(strix-hip-idx-wmma-mm-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                target_compile_options(strix-hip-idx-wmma-mm-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG})
            endif()
            # DSpark speculative draft (src/minimax/dspark): host reference + gfx1151 kernels.
            add_library(strix_minimax_dspark STATIC src/minimax/dspark/hip_minimax_dspark.hip src/minimax/dspark/markov_cand.hip src/minimax/dspark/minimax_dspark_host.cpp
                src/minimax/dspark/minimax_dspark_files.cpp src/minimax/dspark/minimax_ngram.cpp)
            target_include_directories(strix_minimax_dspark PUBLIC src/minimax/dspark "${STRIX_HIP_INCLUDE_DIR}")
            target_link_libraries(strix_minimax_dspark PUBLIC strix_minimax_assets "${STRIX_HIP_RUNTIME_LIBRARY}" Threads::Threads)
            set_target_properties(strix_minimax_dspark PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_features(strix_minimax_dspark PUBLIC cxx_std_20)
            target_compile_options(strix_minimax_dspark PRIVATE -fno-fast-math -ffp-contract=off $<$<COMPILE_LANGUAGE:HIP>:${STRIX_HIP_WAVE32_FLAG}
                -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee>)
            strix_target_warnings(strix_minimax_dspark)
            if(BUILD_TESTING)
                add_executable(strix-mm-dspark-tests src/minimax/dspark/tests/minimax_dspark_tests.cpp)
                target_link_libraries(strix-mm-dspark-tests PRIVATE strix_minimax_dspark)
                target_compile_options(strix-mm-dspark-tests PRIVATE -ffp-contract=off)
                add_test(NAME strix-mm-dspark-tests COMMAND strix-mm-dspark-tests)
                add_executable(strix-hip-mm-markov-cand-tests src/minimax/dspark/tests/hip_markov_cand_tests.hip)
                target_link_libraries(strix-hip-mm-markov-cand-tests PRIVATE strix_minimax_dspark Threads::Threads)
                set_target_properties(strix-hip-mm-markov-cand-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                target_compile_options(strix-hip-mm-markov-cand-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
                    -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
                add_executable(strix-hip-mm-dspark-tests src/minimax/dspark/tests/hip_minimax_dspark_tests.hip)
                target_link_libraries(strix-hip-mm-dspark-tests PRIVATE strix_minimax_dspark)
                set_target_properties(strix-hip-mm-dspark-tests PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                target_compile_options(strix-hip-mm-dspark-tests PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
                    -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            endif()
            add_library(strix_minimax_session_runtime STATIC src/minimax/session/runtime.cpp)
            target_link_libraries(strix_minimax_session_runtime PUBLIC strix_minimax_session strix_minimax_engine strix_minimax_dspark)
            target_include_directories(strix_minimax_session_runtime PUBLIC src)
            target_compile_features(strix_minimax_session_runtime PUBLIC cxx_std_20)
            strix_target_warnings(strix_minimax_session_runtime)
            # Native setup/protocol is extracted once and remains available with legacy mains OFF.
            target_sources(strix_minimax_session_service PRIVATE src/minimax/serve/serve_runner.cpp)
            target_compile_definitions(strix_minimax_session_service PRIVATE STRIX_MINIMAX_NATIVE=1)
            target_link_libraries(strix_minimax_session_service PRIVATE strix_minimax_engine
                strix_minimax_dspark strix_minimax_session_runtime)
            add_executable(strix-minimax-rank apps/minimax_rank_main.cpp src/bp_codec.cpp src/handoff.cpp)  # bp-v1: RTX hand-off frames
            target_link_libraries(strix-minimax-rank PRIVATE strix_minimax_engine strix_minimax_dspark strix_minimax_serve strix_minimax_session_runtime strix_device_hip)
            target_compile_features(strix-minimax-rank PRIVATE cxx_std_20)
            strix_target_warnings(strix-minimax-rank)
            if(BUILD_TESTING)  # drafter EXL3 linears: bitwise A/B + ABA timing
                add_executable(strix-hip-mmx-draft-bench tests/hip_mmx_draft_bench.hip)
                target_link_libraries(strix-hip-mmx-draft-bench PRIVATE strix_minimax_engine strix_minimax_dspark)
                target_include_directories(strix-hip-mmx-draft-bench PRIVATE "${STRIX_MINIMAX_TRELLIS_DIR}/include")
                set_target_properties(strix-hip-mmx-draft-bench PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                target_compile_options(strix-hip-mmx-draft-bench PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off)
            endif()
            if(STRIX_LEGACY_MAINS)
                add_executable(strix-minimax-serve apps/minimax_serve_main.cpp)
                target_link_libraries(strix-minimax-serve PRIVATE strix_session_shell strix_minimax_session_service)
                target_compile_features(strix-minimax-serve PRIVATE cxx_std_20)
                strix_target_warnings(strix-minimax-serve)
            endif()
        endif()
        foreach(target ${strix_minimax_msa_hip_targets})
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(${target} PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
                -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
            strix_target_warnings(${target})
        endforeach()
    endif()
endif()
option(STRIX_FAMILY_H3 "Build the MiniMax H3 audio+video DiT family modules (opt-in)" OFF)
if(STRIX_FAMILY_H3)
    file(GLOB strix_h3_cmake "${CMAKE_CURRENT_SOURCE_DIR}/cmake/h3_*.cmake")
    list(SORT strix_h3_cmake)
    foreach(f IN LISTS strix_h3_cmake)
        include(${f})
    endforeach()
endif()
option(STRIX_FAMILY_GEMMA4 "Build the gemma4 (Nemotron 3.5 Bolt E4B) family modules" OFF)
include(cmake/StrixCpuLinear.cmake)
if(STRIX_FAMILY_GEMMA4)
    file(GLOB strix_gemma4_cmake "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gemma4_*.cmake")
    foreach(f IN LISTS strix_gemma4_cmake)
        include(${f})
    endforeach()
endif()

# GLM-5.3-Flash engine (glm/engine): Tier A host emulation + CPU semantic test, Tier A / DSA device
# kernels + gate-K tests, and the TP2 engine (strix-glm5-rank) on strix-trellis (design notes).
option(STRIX_FAMILY_GLM5 "Build the GLM-5.3-Flash engine (glm/engine)" OFF)
if(STRIX_FAMILY_GLM5)
    if(BUILD_TESTING)
        add_executable(strix-glm5-moe-traits-host-tests tests/glm_moe_traits_host_tests.cpp)
        target_include_directories(strix-glm5-moe-traits-host-tests PRIVATE src)
        target_link_libraries(strix-glm5-moe-traits-host-tests PRIVATE strix_model_data)
        target_compile_features(strix-glm5-moe-traits-host-tests PRIVATE cxx_std_20)
        if(MSVC)
            target_compile_definitions(strix-glm5-moe-traits-host-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
            target_compile_options(strix-glm5-moe-traits-host-tests PRIVATE /W4 /WX /fp:strict)
        else()
            target_compile_options(strix-glm5-moe-traits-host-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math)
        endif()
        add_test(NAME strix-glm5-moe-traits-host-tests COMMAND strix-glm5-moe-traits-host-tests)
        set_tests_properties(strix-glm5-moe-traits-host-tests PROPERTIES LABELS "cpu;glm;exactness")
    endif()
    add_library(strix_glm5_model_description STATIC
        src/glm/load/glm_model_description.cpp src/glm/load/glm_rank_plan_graph.cpp)
    target_include_directories(strix_glm5_model_description PUBLIC src include)
    target_link_libraries(strix_glm5_model_description PUBLIC strix_model_description strix_serve_json strix_model_graph)
    target_compile_features(strix_glm5_model_description PUBLIC cxx_std_20)
    strix_target_warnings(strix_glm5_model_description)
    if(BUILD_TESTING)
        add_executable(strix-glm5-model-graph-tests tests/glm_model_graph_tests.cpp)
        target_link_libraries(strix-glm5-model-graph-tests PRIVATE strix_glm5_model_description)
        strix_target_warnings(strix-glm5-model-graph-tests)
        add_test(NAME strix-glm5-model-graph-tests COMMAND strix-glm5-model-graph-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/models/glm-5.3-flash.toml)
        set_tests_properties(strix-glm5-model-graph-tests PROPERTIES LABELS "cpu;model-graph")
        add_executable(strix-glm5-model-description-tests tests/glm_model_description_tests.cpp)
        target_link_libraries(strix-glm5-model-description-tests PRIVATE strix_glm5_model_description)
        strix_target_warnings(strix-glm5-model-description-tests)
        add_test(NAME strix-glm5-model-description-tests COMMAND strix-glm5-model-description-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/models/glm-5.3-flash.toml)
        set_tests_properties(strix-glm5-model-description-tests PROPERTIES LABELS "cpu;model-description")
    endif()
    add_library(strix_glm5_host STATIC src/glm/host/glm_host.cpp src/glm/spec/glm_spec_host.cpp src/glm/spec/glm_spec_merge.cpp)
    target_include_directories(strix_glm5_host PUBLIC src include)
    target_compile_features(strix_glm5_host PUBLIC cxx_std_20)
    target_compile_options(strix_glm5_host PRIVATE -ffp-contract=off -fno-fast-math)
    strix_target_warnings(strix_glm5_host)
    if(BUILD_TESTING)
        add_executable(strix-glm5-host-cpu-tests tests/glm_host_cpu_tests.cpp)
        target_link_libraries(strix-glm5-host-cpu-tests PRIVATE strix_glm5_host)
        target_compile_options(strix-glm5-host-cpu-tests PRIVATE -ffp-contract=off -fno-fast-math)
        add_executable(strix-glm5-spec-cpu-tests tests/glm_spec_cpu_tests.cpp)
        target_link_libraries(strix-glm5-spec-cpu-tests PRIVATE strix_glm5_host strix_spec)
        target_compile_options(strix-glm5-spec-cpu-tests PRIVATE -ffp-contract=off -fno-fast-math)
        # load memory accounting (design notes): probe / admission / cost model / trace + watchdog / PackReader
        # streaming a synthetic multi-GiB pack (page cache and RSS bounded); host-only, Linux
        add_executable(strix-glm5-mem-cpu-tests tests/glm_mem_cpu_tests.cpp src/glm/engine/glm_mem.cpp src/glm/engine/glm_pack.cpp)
        target_include_directories(strix-glm5-mem-cpu-tests PRIVATE src include)
        target_compile_features(strix-glm5-mem-cpu-tests PRIVATE cxx_std_20)
        target_link_libraries(strix-glm5-mem-cpu-tests PRIVATE Threads::Threads strix_deploy)
        strix_target_warnings(strix-glm5-mem-cpu-tests)
        # strix-glm5-rank defaults: plain invocation == production receipt, off switches / baseline == old path (host-only)
        add_executable(strix-glm5-rank-defaults-cpu-tests tests/glm_rank_defaults_cpu_tests.cpp src/glm/engine/glm_mem.cpp)
        target_link_libraries(strix-glm5-rank-defaults-cpu-tests PRIVATE strix_glm5_host Threads::Threads strix_deploy)
        target_include_directories(strix-glm5-rank-defaults-cpu-tests PRIVATE apps)
        target_compile_features(strix-glm5-rank-defaults-cpu-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-glm5-rank-defaults-cpu-tests)
        add_test(NAME strix-glm5-rank-defaults-cpu-tests COMMAND strix-glm5-rank-defaults-cpu-tests)
        # GLM MLA dispatch (host-only): the real explicit/automatic entry points of glm_prefill.hip, compiled with a
        # recording stub; exact version APIs never take FAST, resolved switches are cached (no copied dispatch logic).
        file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/glm/prefill/glm_prefill.hip" strix_glm_mla_prefill_src)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/glm/prefill/glm_prefill.hip")
        string(FIND "${strix_glm_mla_prefill_src}" "\nvoid mla_attention_rows_versions(" strix_glm_mla_begin)
        if(strix_glm_mla_begin LESS 0)
            message(FATAL_ERROR "glm_prefill.hip: MLA explicit entry points missing")
        endif()
        string(SUBSTRING "${strix_glm_mla_prefill_src}" ${strix_glm_mla_begin} -1 strix_glm_mla_wrappers)
        string(REPLACE "}  // namespace strix::glm::pf" "" strix_glm_mla_wrappers "${strix_glm_mla_wrappers}")
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/glm_mla_dispatch/mla_wrappers.inc" "${strix_glm_mla_wrappers}")
        add_executable(strix-glm-mla-dispatch-host-tests tests/glm_mla_dispatch_host_tests.cpp)
        target_compile_features(strix-glm-mla-dispatch-host-tests PRIVATE cxx_std_20)
        target_include_directories(strix-glm-mla-dispatch-host-tests PRIVATE src include
            "${CMAKE_CURRENT_BINARY_DIR}/glm_mla_dispatch")
        strix_target_warnings(strix-glm-mla-dispatch-host-tests)
        foreach(mode inherit off force)
            add_test(NAME strix-glm-mla-dispatch-host-tests-${mode} COMMAND strix-glm-mla-dispatch-host-tests ${mode})
            set_tests_properties(strix-glm-mla-dispatch-host-tests-${mode} PROPERTIES TIMEOUT 10)
        endforeach()
        # Compile the real overlap diagnostic guard with host-only recording work.
        string(FIND "${strix_glm_mla_prefill_src}" "static void topk_overlap_diag(" strix_glm_overlap_begin)
        string(FIND "${strix_glm_mla_prefill_src}" "    unsigned long long z[2][4] = {};" strix_glm_overlap_end)
        if(strix_glm_overlap_begin LESS 0 OR strix_glm_overlap_end LESS 0 OR
           strix_glm_overlap_end LESS_EQUAL strix_glm_overlap_begin)
            message(FATAL_ERROR "glm_prefill.hip: overlap host guard boundaries missing")
        endif()
        math(EXPR strix_glm_overlap_length "${strix_glm_overlap_end} - ${strix_glm_overlap_begin}")
        string(SUBSTRING "${strix_glm_mla_prefill_src}" ${strix_glm_overlap_begin}
            ${strix_glm_overlap_length} strix_glm_overlap_guard)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/glm_env_arms/overlap_guard.inc"
            "${strix_glm_overlap_guard}(void)tokens; (void)n; (void)C; (void)s; ++diagnostic_calls;\n}\n")
        add_executable(strix-glm-env-arms-host-tests tests/glm_env_arms_host_tests.cpp)
        target_compile_features(strix-glm-env-arms-host-tests PRIVATE cxx_std_20)
        target_include_directories(strix-glm-env-arms-host-tests PRIVATE include
            "${CMAKE_CURRENT_BINARY_DIR}/glm_env_arms")
        if(MSVC)
            target_compile_definitions(strix-glm-env-arms-host-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        strix_target_warnings(strix-glm-env-arms-host-tests)
        foreach(case no-arms governed-snapshot overlap-cache)
            add_test(NAME strix-glm-env-arms-host-tests-${case} COMMAND strix-glm-env-arms-host-tests ${case})
            set_tests_properties(strix-glm-env-arms-host-tests-${case} PROPERTIES TIMEOUT 10)
        endforeach()
        if(TARGET strix_serve)
            # strix-glm5-serve control channel, lead / follower protocol and scheduling on a fake engine (host-only)
            if(NOT MSVC)  # the GLM host arithmetic uses compiler builtins
                add_subdirectory(tests/glm_session_owner)
            endif()
            add_executable(strix-glm5-serve-cpu-tests tests/glm_serve_cpu_tests.cpp)
            target_link_libraries(strix-glm5-serve-cpu-tests PRIVATE strix_serve strix_glm5_host strix_core Threads::Threads strix_glm_session_owner)
            target_include_directories(strix-glm5-serve-cpu-tests PRIVATE src include)
            target_compile_features(strix-glm5-serve-cpu-tests PRIVATE cxx_std_20)
            strix_target_warnings(strix-glm5-serve-cpu-tests)
            add_test(NAME strix-glm5-serve-cpu-tests COMMAND strix-glm5-serve-cpu-tests)
        endif()
    endif()
    if(STRIX_ENABLE_HIP_RUNTIME)
        set(strix_glm5_hip_flags ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt
            -fdenormal-fp-math=ieee -fno-gpu-flush-denormals-to-zero)
        add_library(strix_glm5_kernels STATIC src/glm/kernels/glm_tier_a.hip src/glm/kernels/glm_dsa.hip src/glm/attn/glm_attn_fast.hip src/glm/spec/glm_spec_dev.hip
            src/glm/prefill/glm_prefill.hip src/glm/xfer/glm_post_fast.hip src/glm/moe/glm_moe.hip src/glm/engine/glm_r1b.hip src/glm/attn/glm_attn_tierb.hip)
        target_link_libraries(strix_glm5_kernels PUBLIC strix_glm5_host "${STRIX_HIP_RUNTIME_LIBRARY}")
        target_include_directories(strix_glm5_kernels PUBLIC src include "${STRIX_HIP_INCLUDE_DIR}")
        set(strix_glm5_hip_targets strix_glm5_kernels)
        if(BUILD_TESTING)
            add_executable(strix-hip-glm5-tier-a-tests tests/hip_glm_tier_a_tests.hip)
            target_link_libraries(strix-hip-glm5-tier-a-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-dsa-tests tests/hip_glm_dsa_tests.hip)
            target_link_libraries(strix-hip-glm5-dsa-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-attn-fast-tests tests/hip_glm_attn_fast_tests.hip)
            target_link_libraries(strix-hip-glm5-attn-fast-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-spec-tests tests/hip_glm_spec_tests.hip)
            target_link_libraries(strix-hip-glm5-spec-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-r1b-tests tests/hip_glm_r1b_tests.hip)
            target_link_libraries(strix-hip-glm5-r1b-tests PRIVATE strix_glm5_kernels)
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-r1b-tests)
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-tier-a-tests strix-hip-glm5-dsa-tests strix-hip-glm5-attn-fast-tests
                 strix-hip-glm5-spec-tests)
        endif()
        set(STRIX_GLM5_TRELLIS_DIR ""
            CACHE PATH "strix-trellis tree linked by the GLM engine (Tier B reference): include/strix_trellis.h + build-qualification-v03/libstrix_trellis.a")
        set(strix_glm5_trellis_lib "${STRIX_GLM5_TRELLIS_DIR}/build-qualification-v03/libstrix_trellis.a")
        if(TARGET strix_dist AND EXISTS "${STRIX_GLM5_TRELLIS_DIR}/include/strix_trellis.h" AND EXISTS "${strix_glm5_trellis_lib}")
            add_library(strix_glm5_engine STATIC src/glm/engine/engine.hip src/glm/engine/glm_fast_attn.hip src/glm/engine/glm_fast_moe.hip
                src/glm/engine/glm_kvshard.hip
                src/glm/engine/glm_pack.cpp src/glm/engine/glm_mem.cpp
                src/glm/xfer/glm_xfer.hip src/glm/xfer/glm_xfer_exchange.hip src/glm/xfer/glm_post_exchange.hip
                src/glm/prefill/glm_pf_moe.hip src/glm/prefill/glm_pf_path.hip src/glm/prefill/glm_pf_exact.hip src/glm/prefill/glm_pf_exact_moe.hip)
            target_link_libraries(strix_glm5_engine PUBLIC strix_tiled_core)
            target_compile_definitions(strix_glm5_engine PRIVATE STRIX_GLM_PF_IU4_LIB=1)
            target_link_libraries(strix_glm5_kernels PUBLIC strix_tiled_core)
            target_compile_definitions(strix_glm5_kernels PRIVATE STRIX_GLM_FAST_I4_CORE=1)
            target_link_libraries(strix_glm5_engine PUBLIC strix_xport strix_glm5_kernels strix_glm5_model_description strix_dist strix_deploy "${strix_glm5_trellis_lib}")
            target_sources(strix_glm5_engine PRIVATE src/glm/load/glm_rank_plan.cpp)
            set_source_files_properties(src/glm/engine/glm_kvshard.hip PROPERTIES
                COMPILE_OPTIONS "-Werror;-Wshadow;-Wsign-conversion;-Wunused-function;-Wcuda-compat")
            target_include_directories(strix_glm5_engine PUBLIC include src "${STRIX_HIP_INCLUDE_DIR}"
                PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix_glm5_engine)
            if(BUILD_TESTING)
                add_executable(strix-hip-glm5-post-exchange-rows-tests tests/hip_glm_post_exchange_rows_tests.hip)
                target_link_libraries(strix-hip-glm5-post-exchange-rows-tests PRIVATE strix_glm5_engine)
                list(APPEND strix_glm5_hip_targets strix-hip-glm5-post-exchange-rows-tests)
                # FastAttn multi-slot verify windows == single-slot verify calls, bitwise
                add_executable(strix-hip-glm5-fast-verify-tests tests/hip_glm_fast_verify_tests.hip)
                target_link_libraries(strix-hip-glm5-fast-verify-tests PRIVATE strix_glm5_engine)
                list(APPEND strix_glm5_hip_targets strix-hip-glm5-fast-verify-tests)
            endif()
            add_executable(strix-glm5-g3-replay tests/glm_g3_replay.hip)
            target_link_libraries(strix-glm5-g3-replay PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-g3-replay PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-glm5-g3-replay)
            # fused attention Tier B: gate (bitwise vs the linked strix_gemv) + bench
            add_executable(strix-hip-glm5-attn-tierb-tests tests/hip_glm_attn_tierb_tests.hip)
            target_link_libraries(strix-hip-glm5-attn-tierb-tests PRIVATE strix_glm5_kernels "${strix_glm5_trellis_lib}")
            target_include_directories(strix-hip-glm5-attn-tierb-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-attn-tierb-tests)
            add_executable(strix-glm5-gate-d-digest tests/glm_gate_d_digest.hip)
            target_link_libraries(strix-glm5-gate-d-digest PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-gate-d-digest PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_include_directories(strix-glm5-gate-d-digest SYSTEM PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/src")
            list(APPEND strix_glm5_hip_targets strix-glm5-gate-d-digest)
            add_executable(strix-glm5-decode-q tools/glm/omath/decode_q.hip)
            target_link_libraries(strix-glm5-decode-q PRIVATE strix_glm5_engine)
            target_include_directories(strix-glm5-decode-q PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_include_directories(strix-glm5-decode-q SYSTEM PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/src")
            list(APPEND strix_glm5_hip_targets strix-glm5-decode-q)
            # chunked prefill gate / timing app and the prefill gates (design notes)
            add_executable(strix-glm5-pf apps/glm_pf_main.cpp)
            target_link_libraries(strix-glm5-pf PRIVATE strix_glm5_engine strix_device_hip)
            target_include_directories(strix-glm5-pf PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_compile_features(strix-glm5-pf PRIVATE cxx_std_20)
            strix_target_warnings(strix-glm5-pf)
            add_executable(strix-hip-glm5-prefill-tests tests/hip_glm_prefill_tests.hip)
            target_link_libraries(strix-hip-glm5-prefill-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-pf-tierb-tests tests/hip_glm_pf_tierb_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-tierb-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-tierb-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            add_executable(strix-hip-glm5-pf-exact-tests tests/hip_glm_pf_exact_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-exact-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-exact-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-exact-tests)
            add_executable(strix-hip-glm5-pf-layer-bench tests/hip_glm_pf_layer_bench.hip)  # MLA / layer stage timing (no gate)
            target_link_libraries(strix-hip-glm5-pf-layer-bench PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-layer-bench PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            add_executable(strix-hip-attn-core-sparse-tests tests/hip_attn_core_sparse_tests.hip)  # shared sparse flash core vs FP64 (GLM MLA + MSA shapes)
            target_link_libraries(strix-hip-attn-core-sparse-tests PRIVATE strix_glm5_kernels)
            add_executable(strix-hip-glm5-pf-exact-moe-tests tests/hip_glm_pf_exact_moe_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-exact-moe-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-exact-moe-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-exact-moe-tests)
            add_executable(strix-hip-glm5-pf-wmma-tests tests/hip_glm_pf_wmma_tests.hip)  # FAST glm-pf-wmma-x drift gate + timing
            target_link_libraries(strix-hip-glm5-pf-wmma-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-wmma-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-wmma-tests)
            add_executable(strix-hip-glm5-pf-iu4-chain-tests tests/hip_glm_pf_iu4_chain_tests.hip)  # FAST_I4 chain plan policy: bits + timing
            target_link_libraries(strix-hip-glm5-pf-iu4-chain-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-iu4-chain-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-iu4-chain-tests)
            add_executable(strix-hip-glm5-pf-kda-wmma-tests tests/hip_glm_pf_kda_wmma_tests.hip)  # FAST glm-pf-kda-wmma-v1 contract + drift gate + timing
            target_link_libraries(strix-hip-glm5-pf-kda-wmma-tests PRIVATE strix_glm5_engine)
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-kda-wmma-tests)
            add_executable(strix-hip-glm5-pf-router-wmma-tests tests/hip_glm_pf_router_wmma_tests.hip)  # FAST glm-pf-router-wmma-v1: forms, fallback, drift, timing
            target_link_libraries(strix-hip-glm5-pf-router-wmma-tests PRIVATE strix_glm5_engine)
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-router-wmma-tests)
            add_executable(strix-hip-glm5-pf-mla-wmma-tests tests/hip_glm_pf_mla_wmma_tests.hip)  # FAST glm-pf-mla-wmma-v1 contract + drift gate + timing
            target_link_libraries(strix-hip-glm5-pf-mla-wmma-tests PRIVATE strix_glm5_engine)
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-pf-mla-wmma-tests)
            add_executable(strix-hip-fma-rows-tests tests/hip_fma_rows_tests.hip)  # shared exact FMA-rows core: bitwise gate + timing
            target_link_libraries(strix-hip-fma-rows-tests PRIVATE strix_glm5_engine)
            list(APPEND strix_glm5_hip_targets strix-hip-fma-rows-tests)
            add_executable(strix-hip-glm5-pf-moe-tests tests/hip_glm_pf_moe_tests.hip)
            target_link_libraries(strix-hip-glm5-pf-moe-tests PRIVATE strix_glm5_engine)
            target_include_directories(strix-hip-glm5-pf-moe-tests PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            list(APPEND strix_glm5_hip_targets strix-hip-glm5-prefill-tests strix-hip-glm5-pf-tierb-tests strix-hip-glm5-pf-moe-tests)
            add_executable(strix-hip-idx-wmma-glm-tests tests/hip_idx_wmma_tests.hip)  # FAST shared WMMA indexer core: gate + timing
            target_link_libraries(strix-hip-idx-wmma-glm-tests PRIVATE strix_glm5_kernels)
            target_compile_definitions(strix-hip-idx-wmma-glm-tests PRIVATE IDXW_GLM)
            list(APPEND strix_glm5_hip_targets strix-hip-idx-wmma-glm-tests)
            add_executable(strix-glm5-rank apps/glm_rank_main.cpp src/glm/spec/glm_spec_run.cpp)
            target_link_libraries(strix-glm5-rank PRIVATE strix_spec strix_glm5_engine strix_device_profile strix_device_hip)
            target_include_directories(strix-glm5-rank PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
            target_compile_features(strix-glm5-rank PRIVATE cxx_std_20)
            strix_target_warnings(strix-glm5-rank)
            if(TARGET strix_serve)
                # TP2 HTTP server with continuous batched decode (rank 0 lead + HTTP, rank 1 follower)
                add_executable(strix-glm5-serve apps/glm_serve_main.cpp src/glm/spec/glm_spec_run.cpp)
                target_link_libraries(strix-glm5-serve PRIVATE strix_spec strix_glm5_engine strix_serve strix_glm_session_owner strix_device_hip)
                target_include_directories(strix-glm5-serve PRIVATE "${STRIX_GLM5_TRELLIS_DIR}/include")
                target_compile_features(strix-glm5-serve PRIVATE cxx_std_20)
                strix_target_warnings(strix-glm5-serve)
            endif()
        endif()
        foreach(target ${strix_glm5_hip_targets})
            set_target_properties(${target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${strix_glm5_hip_flags}>)
            strix_target_warnings(${target})
        endforeach()
        foreach(target strix-hip-glm5-prefill-tests strix-hip-glm5-pf-tierb-tests strix-hip-glm5-pf-moe-tests strix-hip-glm5-pf-exact-tests strix-hip-glm5-pf-exact-moe-tests)  # prefill gates
            if(TARGET ${target})
                target_compile_options(${target} PRIVATE -Wno-sign-conversion)
            endif()
        endforeach()
        if(TARGET strix-hip-glm5-attn-fast-tests)  # attention gate compiled without -Wsign-conversion
            target_compile_options(strix-hip-glm5-attn-fast-tests PRIVATE -Wno-sign-conversion)
        endif()
    endif()
endif()

# Host-only RTX session envelope/LRU tests (also available in CUDA-only builds).
if(BUILD_TESTING AND STRIX_FAMILY_DEEPSEEK41)
    add_executable(strix-ds41-session-cache-tests tests/ds41_session_cache_test.cpp src/deepseek41_attention_state.cpp)
    target_link_libraries(strix-ds41-session-cache-tests PRIVATE strix_core strix_ds41_attention strix_ds41_formats Threads::Threads)
    target_include_directories(strix-ds41-session-cache-tests PRIVATE include)
    target_compile_features(strix-ds41-session-cache-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_definitions(strix-ds41-session-cache-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-ds41-session-cache-tests)
    add_test(NAME strix-ds41-session-cache-tests COMMAND strix-ds41-session-cache-tests
        "${CMAKE_CURRENT_BINARY_DIR}/session-test-${CMAKE_BUILD_TYPE}")
    add_executable(strix-ds41-session-admission-tests tests/ds41_session_admission_test.cpp)
    target_link_libraries(strix-ds41-session-admission-tests PRIVATE strix_core)
    target_include_directories(strix-ds41-session-admission-tests PRIVATE include)
    target_compile_features(strix-ds41-session-admission-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-ds41-session-admission-tests)
    add_test(NAME strix-ds41-session-admission-tests COMMAND strix-ds41-session-admission-tests
        "${CMAKE_CURRENT_BINARY_DIR}/session-admission-test-${CMAKE_BUILD_TYPE}")
    set_tests_properties(strix-ds41-session-admission-tests PROPERTIES LABELS "cpu;session" TIMEOUT 120)
    foreach(checkpoint_test strix-ds41-session-cache-tests strix-ds41-session-admission-tests)
        if(MSVC)
            target_compile_options(${checkpoint_test} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${checkpoint_test} PRIVATE -UNDEBUG)
        endif()
    endforeach()
    set(STRIX_CUDA_DS41_CHECKPOINT_REFERENCE_DIR "" CACHE PATH "Optional independent CUDA DS41 host checkpoint reference directory")
    if(STRIX_CUDA_DS41_CHECKPOINT_REFERENCE_DIR)
        add_test(NAME strix-ds41-session-reference-tests COMMAND strix-ds41-session-cache-tests
            --verify "${STRIX_CUDA_DS41_CHECKPOINT_REFERENCE_DIR}")
        set_tests_properties(strix-ds41-session-reference-tests PROPERTIES LABELS "cpu;session" TIMEOUT 120)
    endif()
endif()

# Full-content hash checks are also available in the CUDA-only host library.
if(BUILD_TESTING AND STRIX_CUDA_DS41_ONLY)
    add_executable(strix-model-content-hash-tests tests/model_content_hash_tests.cpp)
    target_link_libraries(strix-model-content-hash-tests PRIVATE strix_core)
    strix_target_warnings(strix-model-content-hash-tests)
    add_test(NAME strix-model-content-hash-tests COMMAND strix-model-content-hash-tests)
endif()
if(BUILD_TESTING AND STRIX_FAMILY_DEEPSEEK41)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME strix-rtx-session-gate-tests
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rtx_session_gate.py)
    set_tests_properties(strix-ds41-session-cache-tests strix-rtx-session-gate-tests PROPERTIES LABELS "cpu;session" TIMEOUT 120)
endif()
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME strix-mmflash-ranges-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mmflash_ranges.py)
        add_test(NAME strix-rtx-gpu-guard-tests
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_guard.py)
    endif()
endif()

# Host-only layer-split wire-layout gate; native Qwen execution is qualified separately.
if(BUILD_TESTING)
    add_executable(strix-qwen-prefill-pipeline-tests tests/qwen_prefill_pipeline_tests.cpp)
    target_include_directories(strix-qwen-prefill-pipeline-tests PRIVATE include)
    target_compile_features(strix-qwen-prefill-pipeline-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(strix-qwen-prefill-pipeline-tests PRIVATE /W4 /WX)
    else()
        target_compile_options(strix-qwen-prefill-pipeline-tests PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion)
    endif()
    add_test(NAME qwen-prefill-pipeline COMMAND strix-qwen-prefill-pipeline-tests)
endif()
if(BUILD_TESTING)
    find_package(Threads REQUIRED)
    add_executable(strix-qwen-prefill-handoff-tests tests/qwen_prefill_handoff_tests.cpp)
    target_include_directories(strix-qwen-prefill-handoff-tests PRIVATE include)
    target_compile_features(strix-qwen-prefill-handoff-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-qwen-prefill-handoff-tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(strix-qwen-prefill-handoff-tests PRIVATE /W4 /WX /EHsc /wd4996)
    else()
        target_compile_options(strix-qwen-prefill-handoff-tests PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion)
    endif()
    add_test(NAME qwen-prefill-handoff COMMAND strix-qwen-prefill-handoff-tests)
    add_executable(strix-pair-regime-tests tests/pair_regime_tests.cpp)
    target_include_directories(strix-pair-regime-tests PRIVATE include)
    target_compile_features(strix-pair-regime-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-regime-tests)
    add_test(NAME strix-pair-regime-tests COMMAND strix-pair-regime-tests)
    add_executable(strix-pair-layercut-pack-tests tests/pair_layercut_pack_tests.cpp src/cuda/models/minimax/mm_rtx_pack.cpp)
    target_include_directories(strix-pair-layercut-pack-tests PRIVATE src)
    target_compile_features(strix-pair-layercut-pack-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(strix-pair-layercut-pack-tests PRIVATE /wd4996)
    endif()
    add_test(NAME strix-pair-layercut-pack-tests COMMAND strix-pair-layercut-pack-tests "${CMAKE_CURRENT_BINARY_DIR}/layercut-pack-fixture")
    add_executable(strix-pair-layercut-tests tests/pair_layercut_tests.cpp)
    target_include_directories(strix-pair-layercut-tests PRIVATE include)
    target_compile_features(strix-pair-layercut-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-layercut-tests)
    add_test(NAME strix-pair-layercut-tests COMMAND strix-pair-layercut-tests)
    add_executable(strix-handoff-lanes-tests tests/handoff_lanes_tests.cpp src/handoff.cpp src/bp_codec.cpp)
    target_include_directories(strix-handoff-lanes-tests PRIVATE include)
    target_link_libraries(strix-handoff-lanes-tests PRIVATE Threads::Threads)
    if(WIN32)
        target_link_libraries(strix-handoff-lanes-tests PRIVATE ws2_32)
    endif()
    target_compile_features(strix-handoff-lanes-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-handoff-lanes-tests)
    add_test(NAME strix-handoff-lanes-tests COMMAND strix-handoff-lanes-tests)
    add_test(NAME strix-handoff-lanes-failed-write COMMAND strix-handoff-lanes-tests --failure)
    add_test(NAME strix-handoff-lanes-rejection COMMAND strix-handoff-lanes-tests --rejection)
    add_test(NAME strix-handoff-lanes-stopped COMMAND strix-handoff-lanes-tests --stopped)
    add_test(NAME strix-handoff-lanes-mixed COMMAND strix-handoff-lanes-tests --mixed)
    set_tests_properties(strix-handoff-lanes-tests strix-handoff-lanes-failed-write
        strix-handoff-lanes-rejection strix-handoff-lanes-stopped strix-handoff-lanes-mixed PROPERTIES TIMEOUT 25)
    add_executable(strix-pair-stage-pipes-tests tests/pair_stage_pipes_tests.cpp)
    target_include_directories(strix-pair-stage-pipes-tests PRIVATE include)
    target_compile_features(strix-pair-stage-pipes-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-pair-stage-pipes-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-pair-stage-pipes-tests)
    add_test(NAME strix-pair-stage-pipes-tests COMMAND strix-pair-stage-pipes-tests)
    add_executable(strix-pair-chain-tests tests/pair_chain_tests.cpp)
    target_include_directories(strix-pair-chain-tests PRIVATE include)
    target_compile_features(strix-pair-chain-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-chain-tests)
    add_test(NAME strix-pair-chain-tests COMMAND strix-pair-chain-tests)
    add_executable(strix-pair-window-overlap-tests tests/pair_window_overlap_tests.cpp)
    target_include_directories(strix-pair-window-overlap-tests PRIVATE include)
    target_compile_features(strix-pair-window-overlap-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-window-overlap-tests)
    add_test(NAME strix-pair-window-overlap-tests COMMAND strix-pair-window-overlap-tests)
    add_executable(strix-qwen35-pair-plan-tests tests/qwen35_pair_plan_tests.cpp)
    target_include_directories(strix-qwen35-pair-plan-tests PRIVATE include)
    target_compile_features(strix-qwen35-pair-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-qwen35-pair-plan-tests)
    add_test(NAME strix-qwen35-pair-plan-tests COMMAND strix-qwen35-pair-plan-tests)
    # Hy4-flash placement on the shared planners (pair/placement_inputs.hpp + chain + column_split) over the pack table.
    add_executable(strix-hy4-placement-tests tests/hy4_placement_tests.cpp)
    target_include_directories(strix-hy4-placement-tests PRIVATE include)
    target_compile_features(strix-hy4-placement-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hy4-placement-tests)
    add_test(NAME strix-hy4-placement-tests COMMAND strix-hy4-placement-tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hy4/q4km-component-bytes.tsv)
    # Hy4-flash exact tensor-parallel layout (strix/hy4/tp_layout.hpp): shares tile, MoE join, planner joins = step gathers.
    add_executable(strix-hy4-tp-layout-tests tests/hy4_tp_layout_tests.cpp)
    target_include_directories(strix-hy4-tp-layout-tests PRIVATE include)
    target_compile_features(strix-hy4-tp-layout-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-hy4-tp-layout-tests)
    add_test(NAME strix-hy4-tp-layout-tests COMMAND strix-hy4-tp-layout-tests)
    # Shared prompt windows (strix/session/prompt_windows.hpp) and the two-stage window picker.
    add_executable(strix-prompt-windows-tests tests/prompt_windows_tests.cpp)
    target_include_directories(strix-prompt-windows-tests PRIVATE include)
    target_compile_features(strix-prompt-windows-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-prompt-windows-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-prompt-windows-tests)
    add_test(NAME strix-prompt-windows-tests COMMAND strix-prompt-windows-tests)
    # N-rank layer pipeline (strix/pair/layer_pipeline.hpp): composed prefill + decode == one rank, bytes, N = 1..4.
    add_executable(strix-pair-layer-pipeline-tests tests/pair_layer_pipeline_tests.cpp)
    target_include_directories(strix-pair-layer-pipeline-tests PRIVATE include)
    target_compile_features(strix-pair-layer-pipeline-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-pair-layer-pipeline-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-pair-layer-pipeline-tests)
    add_test(NAME strix-pair-layer-pipeline-tests COMMAND strix-pair-layer-pipeline-tests)
    # Exact tensor-parallel decode shares (strix/pair/column_split.hpp): units by rate, joined shares == one rank.
    add_executable(strix-pair-column-split-tests tests/pair_column_split_tests.cpp)
    target_include_directories(strix-pair-column-split-tests PRIVATE include)
    target_compile_features(strix-pair-column-split-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-column-split-tests)
    add_test(NAME strix-pair-column-split-tests COMMAND strix-pair-column-split-tests)
    # One stage description, per-regime split plan (strix/pair/split_plan.hpp): dense binding == its hand model,
    # slot joins bitwise == one rank, regime -> form, bit-changing joins refused.
    add_executable(strix-pair-split-plan-tests tests/pair_split_plan_tests.cpp)
    target_include_directories(strix-pair-split-plan-tests PRIVATE include)
    target_compile_features(strix-pair-split-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-split-plan-tests)
    add_test(NAME strix-pair-split-plan-tests COMMAND strix-pair-split-plan-tests)
    # Rank-local head layouts + all-gather join + split argmax (strix/pair/tp_layout.hpp): joined == one rank.
    add_executable(strix-pair-tp-layout-tests tests/pair_tp_layout_tests.cpp)
    target_include_directories(strix-pair-tp-layout-tests PRIVATE include ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(strix-pair-tp-layout-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-tp-layout-tests)
    add_test(NAME strix-pair-tp-layout-tests COMMAND strix-pair-tp-layout-tests)
    add_executable(strix-tp-join-sum-tests tests/tp_join_sum_tests.cpp)
    target_include_directories(strix-tp-join-sum-tests PRIVATE src)
    target_compile_features(strix-tp-join-sum-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-tp-join-sum-tests)
    add_test(NAME strix-tp-join-sum-tests COMMAND strix-tp-join-sum-tests)
    add_executable(strix-pair-gather-fn-tests tests/pair_gather_fn_tests.cpp)
    target_include_directories(strix-pair-gather-fn-tests PRIVATE include)
    target_compile_features(strix-pair-gather-fn-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-gather-fn-tests)
    add_test(NAME strix-pair-gather-fn-tests COMMAND strix-pair-gather-fn-tests)
    # Shared TP gather link (routes, slot swap, host waits, tag sequence) + placement inputs (rates, measured trip cost).
    add_executable(strix-pair-tp-gather-tests tests/pair_tp_gather_tests.cpp)
    target_include_directories(strix-pair-tp-gather-tests PRIVATE include)
    target_compile_features(strix-pair-tp-gather-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-tp-gather-tests)
    add_test(NAME strix-pair-tp-gather-tests COMMAND strix-pair-tp-gather-tests)
    # One resolved device / link view (device::Set): rates, sources, measured trip classes, digest.
    add_executable(strix-device-set-tests tests/device_set_tests.cpp)
    target_include_directories(strix-device-set-tests PRIVATE include)
    target_compile_features(strix-device-set-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-device-set-tests)
    add_test(NAME strix-device-set-tests COMMAND strix-device-set-tests)
    # Cost-table device key (device/cost_identity.hpp): HIP text equals the stored Qwen MTP key; every fact keyed.
    add_executable(strix-device-cost-identity-tests tests/device_cost_identity_tests.cpp)
    target_include_directories(strix-device-cost-identity-tests PRIVATE include)
    target_compile_features(strix-device-cost-identity-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-device-cost-identity-tests)
    add_test(NAME strix-device-cost-identity-tests COMMAND strix-device-cost-identity-tests)
    # One planner entry (pair::planner::plan): measured-cost rules, gain bar, refusal reasons, receipt line.
    add_executable(strix-pair-planner-tests tests/pair_planner_tests.cpp)
    target_include_directories(strix-pair-planner-tests PRIVATE include)
    target_compile_features(strix-pair-planner-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-pair-planner-tests)
    add_test(NAME strix-pair-planner-tests COMMAND strix-pair-planner-tests)
    # Interleaved virtual stages of the layer pipeline (StageMap on two ranks): prefill + decode bitwise == one rank.
    add_executable(strix-pair-layer-stages-tests tests/pair_layer_stages_tests.cpp)
    target_include_directories(strix-pair-layer-stages-tests PRIVATE include)
    target_compile_features(strix-pair-layer-stages-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-pair-layer-stages-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-pair-layer-stages-tests)
    add_test(NAME strix-pair-layer-stages-tests COMMAND strix-pair-layer-stages-tests)
    set_tests_properties(strix-pair-layer-stages-tests PROPERTIES TIMEOUT 60)
    # Ordered logical channels over one pair transport (both stage-pipe directions on one link).
    add_executable(strix-pair-link-mux-tests tests/pair_link_mux_tests.cpp)
    target_include_directories(strix-pair-link-mux-tests PRIVATE include)
    target_compile_features(strix-pair-link-mux-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-pair-link-mux-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-pair-link-mux-tests)
    add_test(NAME strix-pair-link-mux-tests COMMAND strix-pair-link-mux-tests)
    set_tests_properties(strix-pair-link-mux-tests PROPERTIES TIMEOUT 60)
    # Stage-prefill state re-laid out for a replicated-state (TP) decode: every layer on both ranks, owner's bytes.
    add_executable(strix-pair-relayout-tests tests/pair_relayout_tests.cpp)
    target_include_directories(strix-pair-relayout-tests PRIVATE include)
    target_compile_features(strix-pair-relayout-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-pair-relayout-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-pair-relayout-tests)
    add_test(NAME strix-pair-relayout-tests COMMAND strix-pair-relayout-tests)
    set_tests_properties(strix-pair-relayout-tests PROPERTIES TIMEOUT 60)
    # Pair context-parallel window ring: both ranks end with the one-rank caches, odd / even windows, ring wrap.
    add_executable(strix-pair-window-ring-tests tests/pair_window_ring_tests.cpp)
    target_include_directories(strix-pair-window-ring-tests PRIVATE include)
    target_compile_features(strix-pair-window-ring-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-pair-window-ring-tests PRIVATE Threads::Threads)
    strix_target_warnings(strix-pair-window-ring-tests)
    add_test(NAME strix-pair-window-ring-tests COMMAND strix-pair-window-ring-tests)
    set_tests_properties(strix-pair-window-ring-tests PROPERTIES TIMEOUT 60)
    add_executable(strix-qwen-chain-plan apps/qwen_chain_plan_main.cpp)
    target_include_directories(strix-qwen-chain-plan PRIVATE include)
    target_compile_features(strix-qwen-chain-plan PRIVATE cxx_std_20)
    target_link_libraries(strix-qwen-chain-plan PRIVATE strix_serve_json)
    if(MSVC)
        target_compile_options(strix-qwen-chain-plan PRIVATE /W4 /WX /EHsc /wd4996)
    else()
        target_compile_options(strix-qwen-chain-plan PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion)
    endif()
    add_test(NAME strix-qwen-chain-plan-1gbe COMMAND strix-qwen-chain-plan --rows 32768 --card-rate 4320
        --node-rate 1600 --link-mbps 118 --decoder card)
    set_tests_properties(strix-qwen-chain-plan-1gbe PROPERTIES
        PASS_REGULAR_EXPRESSION "chosen card[.][.]card decoder=card cuts=0/48")
    add_test(NAME strix-qwen-chain-plan-invalid-numbers COMMAND ${CMAKE_COMMAND}
        -DPLANNER=$<TARGET_FILE:strix-qwen-chain-plan> -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_chain_cli.cmake)
    # Shared placement over N ranks from the topology declaration: N-stage balance (balance_split = its N = 2 form,
    # bit-identical), chain planner topology binding (1-4 ranks, unequal memory / speed, slow link, lab plan unchanged),
    # Qwen MoE cut with rank speeds, host mesh collectives and facts exchange over N.
    if(TARGET strix_device)
        add_executable(strix-pair-topology-plan-tests tests/pair_topology_plan_tests.cpp)
        target_include_directories(strix-pair-topology-plan-tests PRIVATE include src)
        target_compile_features(strix-pair-topology-plan-tests PRIVATE cxx_std_20)
        target_link_libraries(strix-pair-topology-plan-tests PRIVATE strix_device strix_serve_json Threads::Threads)
        strix_target_warnings(strix-pair-topology-plan-tests)
        add_test(NAME strix-pair-topology-plan-tests COMMAND strix-pair-topology-plan-tests)
    endif()
    add_executable(strix-qwen-prefill-pair-tests tests/qwen_prefill_pair_tests.cpp)
    target_include_directories(strix-qwen-prefill-pair-tests PRIVATE include src)
    target_compile_features(strix-qwen-prefill-pair-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-qwen-prefill-pair-tests PRIVATE Threads::Threads strix_serve_json)
    if(MSVC)
        target_compile_options(strix-qwen-prefill-pair-tests PRIVATE /W4 /WX /EHsc /wd4996)
    else()
        target_compile_options(strix-qwen-prefill-pair-tests PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion)
    endif()
    add_test(NAME qwen-prefill-pair COMMAND strix-qwen-prefill-pair-tests)
    add_executable(strix-qwen-prefill-staging-crc-tests tests/qwen_prefill_staging_crc_tests.cpp)
    target_include_directories(strix-qwen-prefill-staging-crc-tests PRIVATE tests/qwen_prefill_staging_shim include src)
    target_compile_features(strix-qwen-prefill-staging-crc-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-qwen-prefill-staging-crc-tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(strix-qwen-prefill-staging-crc-tests PRIVATE /W4 /WX /EHsc /wd4996)
    else()
        target_compile_options(strix-qwen-prefill-staging-crc-tests PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion -Wunused-function)
    endif()
    add_test(NAME qwen-prefill-staging-crc COMMAND strix-qwen-prefill-staging-crc-tests)
    add_executable(strix-qwen-prefill-pipe-dump-tests tests/qwen_prefill_pipe_dump_tests.cpp)
    target_include_directories(strix-qwen-prefill-pipe-dump-tests PRIVATE include)
    target_compile_features(strix-qwen-prefill-pipe-dump-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-qwen-prefill-pipe-dump-tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(strix-qwen-prefill-pipe-dump-tests PRIVATE /W4 /WX /EHsc /wd4996)
    else()
        target_compile_options(strix-qwen-prefill-pipe-dump-tests PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion)
    endif()
    add_test(NAME qwen-prefill-pipe-dump COMMAND strix-qwen-prefill-pipe-dump-tests)
    set_tests_properties(qwen-prefill-pipe-dump PROPERTIES
        ENVIRONMENT "STRIX_QWEN_PF_PIPE_DUMP=${CMAKE_CURRENT_BINARY_DIR}/qwen-prefill-pipe-dump")
endif()

# Reuse the extracted host identity/cache provider in the existing Gemma model
# archive. Keep one owner of these production translation units per link graph.
function(strix_link_host_cache_adopters)
    if(TARGET strix_gemma4_model)
        get_target_property(cache_adopter_sources strix_gemma4_model SOURCES)
        list(FILTER cache_adopter_sources EXCLUDE REGEX "(^|/)(model_content_hash|model_runtime_binding)\\.cpp$")
        set_property(TARGET strix_gemma4_model PROPERTY SOURCES "${cache_adopter_sources}")
        target_link_libraries(strix_gemma4_model PUBLIC strix_state_cache)
    endif()
endfunction()
cmake_language(DEFER CALL strix_link_host_cache_adopters)
# Shared expert residency planner (host only, always built).
include("${CMAKE_CURRENT_LIST_DIR}/cmake/StrixResidency.cmake")
if(BUILD_TESTING)
    # serve == bench residency command line (strix/residency/residency_cli.hpp); host only
    add_executable(strix-expert-residency-parity-tests tests/expert_residency_parity_tests.cpp)
    target_link_libraries(strix-expert-residency-parity-tests PRIVATE strix_residency)
    strix_target_warnings(strix-expert-residency-parity-tests)
    add_test(NAME strix-expert-residency-parity-tests COMMAND strix-expert-residency-parity-tests)
endif()
# CPU backend (opt-in, docs/architecture/backends.md).
if(STRIX_BACKEND_CPU)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/StrixBackendCpu.cmake")
endif()

# Host fixtures consuming the onfly census also need its source-local core header.
set_property(SOURCE
    tests/hip_http_batch_session_tests.cpp
    tests/onfly_i4_tests.cpp
    tests/ple_rolling_tests.cpp
    tests/qwen_effective_config_tests.cpp
    tests/qwen_mtp_bundle_tests.cpp
    tests/qwen_product_profile_tests.cpp
    tests/qwen_profile_resolution_tests.cpp
    tests/qwen_router_bf16_tests.cpp
    tests/qwen_startup_policy_tests.cpp
    tests/qwen_yarn_host_compile_tests.cpp
    tests/qwen_yarn_schedule_tests.cpp
    tests/rotary_descriptor_tests.cpp
    tests/serve_api_tests.cpp
    tests/turbo_profile_tests.cpp
    APPEND PROPERTY INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src")
# Shared CUDA core (opt-in).
if(STRIX_CUDA_CORE)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/CudaCore.cmake")
endif()
# Serving-spec fragments (cmake/servespec/*.cmake): shared serving sources and their host tests. Included after
# the optional HIP and CUDA targets exist, so a fragment can bind them (cmake/servespec/hook.cmake contract).
include("${CMAKE_CURRENT_LIST_DIR}/cmake/servespec/hook.cmake")
# Shared MoE router policy (include/strix/moe/route_policy.hpp, header-only): presets, ties, keys, and the family
# byte-compares against the DS41 / MiniMax / GLM-5 host routers when those families are built.
if(BUILD_TESTING)
    add_executable(strix-route-policy-tests tests/route_policy_tests.cpp)
    target_link_libraries(strix-route-policy-tests PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-route-policy-tests)
    if(TARGET strix_minimax_msa)
        target_link_libraries(strix-route-policy-tests PRIVATE strix_minimax_msa)
        target_compile_definitions(strix-route-policy-tests PRIVATE STRIX_RP_MINIMAX=1)
    endif()
    if(TARGET strix_hy4_ref)
        target_link_libraries(strix-route-policy-tests PRIVATE strix_hy4_ref)
        target_compile_definitions(strix-route-policy-tests PRIVATE STRIX_RP_HY4=1)
    endif()
    if(TARGET strix_glm5_host)
        target_link_libraries(strix-route-policy-tests PRIVATE strix_glm5_host)
        target_compile_definitions(strix-route-policy-tests PRIVATE STRIX_RP_GLM5=1)
    endif()
    if(NOT MSVC)
        target_compile_options(strix-route-policy-tests PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-route-policy-tests COMMAND strix-route-policy-tests)
    # Shared route projection (include/strix/moe/route_project.hpp): unfused lane chain + per-row bias class against the
    # DS41 host router and the scalar form of its HIP score kernels.
    add_executable(strix-route-lane-dot-test tests/moe_core/route_lane_dot_test.cpp)
    target_link_libraries(strix-route-lane-dot-test PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-route-lane-dot-test)
    if(NOT MSVC)
        target_compile_options(strix-route-lane-dot-test PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME strix-route-lane-dot-test COMMAND strix-route-lane-dot-test)
    set_tests_properties(strix-route-lane-dot-test PROPERTIES LABELS "cpu;moe;exactness")
    # The unfused step must not depend on the host contraction flags: this TU keeps the compiler's default contraction
    # (GCC gnu++20: contract across expressions; Clang: within an expression, the literal acc + w * x) and enables FMA
    # where it can. No explicit contraction flag: nothing in the build asks for the mode that ignores the
    # contract(off) pragmas (tests/dense_decode_exact_form_tests.py).
    add_executable(strix-route-lane-dot-contract-test tests/moe_core/route_lane_dot_contract_test.cpp)
    target_link_libraries(strix-route-lane-dot-contract-test PRIVATE strix_ds41_formats)
    strix_target_warnings(strix-route-lane-dot-contract-test)
    if(NOT MSVC)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            target_compile_options(strix-route-lane-dot-contract-test PRIVATE -mfma)
        endif()
    endif()
    add_test(NAME strix-route-lane-dot-contract-test COMMAND strix-route-lane-dot-contract-test)
    set_tests_properties(strix-route-lane-dot-contract-test PROPERTIES LABELS "cpu;moe;exactness")
endif()

# Hy4-flash HIP forward (src/hy4/hip: the stage map of hy4_hip_plan.hpp on shared entries; no kernel of its own) and the
# one-node correctness runner strix-hy4-rank. Needs the HIP runtime and the A8 routed-expert kernels
# (-DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON).
option(STRIX_FAMILY_HY4 "Build the Hy4-flash HIP forward and strix-hy4-rank" OFF)
if(STRIX_FAMILY_HY4)
    if(NOT STRIX_ENABLE_HIP_RUNTIME OR NOT TARGET strix_q4k_a8_hip_probe)
        message(FATAL_ERROR "STRIX_FAMILY_HY4 needs STRIX_ENABLE_HIP_RUNTIME=ON and STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON")
    endif()
    add_library(strix_hy4_hip STATIC src/hy4/hip/hy4_hip_forward.hip)
    set_target_properties(strix_hy4_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_include_directories(strix_hy4_hip PUBLIC include src "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_options(strix_hy4_hip PRIVATE ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off
        -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee)
    target_link_libraries(strix_hy4_hip PUBLIC strix_hy4_ref strix_hip_runtime strix_q4k_a8_hip_probe strix_engram strix_ds41_hip)
    target_link_libraries(strix_hy4_hip PRIVATE strix_state_cache strix_tiled_core)  # state image files; head Q8_0 transcode
    strix_target_warnings(strix_hy4_hip)
    add_executable(strix-hy4-rank apps/hy4_rank_main.cpp)
    target_link_libraries(strix-hy4-rank PRIVATE strix_hy4_hip strix_dist strix_device strix_spec strix_tp_xport)  # pair: shared layer pipeline + link (gather channel); copy proposer
    strix_target_warnings(strix-hy4-rank)
    add_executable(strix-hy4-rows apps/hy4_rows_main.cpp)  # verify rows bench + speculative decode (hy4_hip_rows)
    target_link_libraries(strix-hy4-rows PRIVATE strix_hy4_hip strix_spec strix_dist strix_tp_xport)  # --tp-rank: pair gather + round messages
    strix_target_warnings(strix-hy4-rows)
endif()

# --- core packages (CORE-SPEC 17.4) ---
include(cmake/packages/P0-03.cmake)
include(cmake/packages/SCHEMA.cmake)
include(cmake/packages/K1.cmake)
include(cmake/packages/K1c.cmake)
include(cmake/packages/K2c.cmake)
include(cmake/packages/X1.cmake)
include(cmake/packages/D4.cmake)
include(cmake/packages/D3.cmake)
include(cmake/packages/B3Segments.cmake)
include(cmake/packages/D1.cmake)
include(cmake/packages/D3o.cmake)
include(cmake/packages/K3c.cmake)
include(cmake/packages/R1Tail.cmake)
include(cmake/packages/K4c.cmake)
include(cmake/packages/K5c.cmake)
include(cmake/packages/K5A.cmake)
include(cmake/packages/K5B.cmake)
include(cmake/packages/S1.cmake)
include(cmake/packages/X2c.cmake)
include(cmake/packages/K6c.cmake)
include(cmake/packages/X3.cmake)
include(cmake/packages/W1.cmake)
include(cmake/packages/W2.cmake)
include(cmake/packages/D2.cmake)
include(cmake/packages/K11c.cmake)
include(cmake/packages/X2r.cmake)
include(cmake/packages/T1.cmake)
include(cmake/packages/K9c.cmake)
include(cmake/packages/K9G.cmake)
include(cmake/packages/K1bind.cmake)
include(cmake/packages/PreparedGqaFold.cmake)
include(cmake/packages/K8c.cmake)
include(cmake/packages/R12.cmake)
include(cmake/packages/X5.cmake)
include(cmake/packages/P1.cmake)
include(cmake/packages/E1x.cmake)
include(cmake/packages/TRANSITION.cmake)
include(cmake/packages/E2i.cmake)
include(cmake/packages/E2x.cmake)
include(cmake/packages/L2.cmake)
include(cmake/packages/K10c.cmake)
include(cmake/packages/X6.cmake)
include(cmake/packages/EPI.cmake)
include(cmake/packages/MIXLAT.cmake)
include(cmake/packages/HIPBB.cmake)
include(cmake/packages/LANESHIP.cmake)
include(cmake/packages/MOER.cmake)
include(cmake/packages/X6R.cmake)
include(cmake/packages/N1.cmake)
include(cmake/packages/X2W.cmake)
include(cmake/packages/F3G.cmake)
include(cmake/packages/WR1.cmake)
include(cmake/packages/WR2.cmake)
include(cmake/packages/K3R.cmake)
include(cmake/packages/MMSEL.cmake)
include(cmake/packages/QSA.cmake)
include(cmake/packages/RING.cmake)
include(cmake/packages/CHANCE.cmake)
include(cmake/packages/PDL.cmake)
include(cmake/packages/B1.cmake)
include(cmake/packages/DTAP.cmake)
include(cmake/packages/LCOST.cmake)
