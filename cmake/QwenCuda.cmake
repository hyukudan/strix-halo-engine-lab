enable_language(CUDA)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/StrixCudaBackend.cmake)
add_library(strix_qwen_cuda STATIC
    src/cuda/models/qwen/qwen_prefill_driver.cu src/cuda/models/qwen/qwen_hc.cu src/cuda/models/qwen/qwen_dense.cu
    src/cuda/models/qwen/qwen_gdn.cu src/cuda/models/qwen/qwen_qsa.cu src/cuda/models/qwen/qwen_moe.cu
    src/cuda/models/qwen/qwen_plemtp.cu src/cuda/models/qwen/qwen_mtp_draft.cu src/cuda/models/qwen/qwen_transcode.cu src/cuda/tiered_experts.cu
    ${STRIX_CUDA_CORE_AMD_MATH_SOURCES} src/gguf_reader.cpp
    src/cuda/core/nvfp4_a8.cu src/cuda/core/nvfp4_a8_reference.cpp src/cuda/core/nvfp4_reference.cpp)
target_include_directories(strix_qwen_cuda PUBLIC include src/cuda/models/qwen)
# Tiered expert residency: the host planner/resolver and the CPU routed-expert executor (bitwise = the card's A8 chains).
# Only the library targets are needed; the Qwen-only build has no strix_target_warnings, so a no-op stands in for it.
if(NOT COMMAND strix_target_warnings)
    function(strix_target_warnings target)
    endfunction()
endif()
include(${CMAKE_CURRENT_LIST_DIR}/StrixResidency.cmake)
if(NOT TARGET strix_backend_cpu)
    include(${CMAKE_CURRENT_LIST_DIR}/StrixBackendCpu.cmake)
endif()
target_link_libraries(strix_qwen_cuda PUBLIC strix_residency strix_backend_cpu)
# Every Qwen CUDA binary resolves its device through the start gate (strix-devices-v1) before its first allocation.
include(${CMAKE_CURRENT_LIST_DIR}/StrixDevice.cmake)
target_link_libraries(strix_qwen_cuda PUBLIC strix_device strix_cuda_core)
target_compile_options(strix_qwen_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_qwen_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
# Handoff library and the serving front end of the node (OpenAI/Anthropic API, SSE, GGUF BPE tokenizer, Qwen chat
# template) for the RTX engine: strix_qwen_handoff and strix_serve_win, with the libraries they call linked by name.
include(${CMAKE_CURRENT_LIST_DIR}/StrixServeStandalone.cmake)
strix_add_standalone_serve(strix_serve_win strix_qwen_handoff)
include(${CMAKE_CURRENT_LIST_DIR}/StrixBuildInfo.cmake)  # generated/strix/build_info.hpp for src/preflight.cpp
add_executable(strix-cuda-qwen-prefill-service apps/cuda_qwen_prefill_service_main.cu)
target_link_libraries(strix-cuda-qwen-prefill-service PRIVATE strix_qwen_cuda strix_qwen_handoff)
add_executable(qwen_prefill_e2e tests/cuda/qwen_prefill_e2e.cu)
target_link_libraries(qwen_prefill_e2e PRIVATE strix_qwen_cuda strix_qwen_handoff)
add_executable(qwen_prefill_schedule_test tests/cuda/qwen_prefill_schedule_test.cpp)
target_include_directories(qwen_prefill_schedule_test PRIVATE include)
target_link_libraries(qwen_prefill_schedule_test PRIVATE Threads::Threads)
add_executable(qwen_handoff_codec_test tests/cuda/qwen_handoff_codec_test.cpp)
target_link_libraries(qwen_handoff_codec_test PRIVATE strix_qwen_handoff)
enable_testing()
add_test(NAME qwen_prefill_schedule COMMAND qwen_prefill_schedule_test)
add_test(NAME qwen_handoff_codec COMMAND qwen_handoff_codec_test)
add_executable(qwen_decode_e2e tests/cuda/qwen_decode_e2e.cu)
target_link_libraries(qwen_decode_e2e PRIVATE strix_qwen_cuda strix_qwen_handoff)
# Dense IMMA site tests, --bench-shape / --bench-rows microbenches (GPU index 0 only). An instrumented copy of the
# production qwen_dense.cu (dispatch counter) is compiled in with the library's options; the archive's dense object is not linked.
add_executable(qwen_dense_tests tests/cuda/qwen_dense_tests.cu src/cuda/models/qwen/qwen_dense.cu)
target_compile_definitions(qwen_dense_tests PRIVATE STRIX_QWEN_DENSE_TEST_DISPATCH)
target_compile_options(qwen_dense_tests PRIVATE $<TARGET_PROPERTY:strix_qwen_cuda,COMPILE_OPTIONS>)
target_link_libraries(qwen_dense_tests PRIVATE strix_qwen_cuda)
add_test(NAME qwen_dense_host COMMAND qwen_dense_tests)
add_executable(qwen_hc_workspace_host_test tests/cuda/qwen_hc_workspace_host_test.cpp)
target_include_directories(qwen_hc_workspace_host_test PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_test(NAME qwen_hc_workspace_host COMMAND qwen_hc_workspace_host_test)
find_package(Python3 QUIET COMPONENTS Interpreter)
find_program(STRIX_IDENT_BASH bash)
if(Python3_Interpreter_FOUND AND STRIX_IDENT_BASH)
    add_test(NAME qwen_ident_host COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/qwen_ident_host_test.py)
endif()
target_include_directories(qwen_dense_tests PRIVATE tests/cuda)
set_source_files_properties(tests/cuda/qwen_dense_tests.cu PROPERTIES COMPILE_OPTIONS "--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false")
if(MSVC)
    target_compile_options(qwen_dense_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
# TURBO onfly (qwen-turbo-v1): shared nvfp4-a8-v1 core vs its host oracles, the Qwen dense binding, and the one
# resolver (serve == CLI).
add_executable(nvfp4_a8_test tests/cuda/nvfp4_a8_test.cu)
target_link_libraries(nvfp4_a8_test PRIVATE strix_qwen_cuda)
target_compile_options(nvfp4_a8_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
add_test(NAME nvfp4_a8 COMMAND nvfp4_a8_test)
add_executable(nvfp4_a8_bench tests/cuda/nvfp4_a8_bench.cu)
target_link_libraries(nvfp4_a8_bench PRIVATE strix_qwen_cuda)
add_executable(qwen_turbo_resolver_test tests/cuda/qwen_turbo_resolver_test.cu)
target_link_libraries(qwen_turbo_resolver_test PRIVATE strix_qwen_cuda strix_qwen_prefix)
add_test(NAME qwen_turbo_resolver COMMAND qwen_turbo_resolver_test)
add_executable(qwen_decode_bench tests/cuda/qwen_decode_bench.cu)
target_link_libraries(qwen_decode_bench PRIVATE strix_qwen_cuda)
foreach(arm IN ITEMS 0 1)
    add_test(NAME qwen_dense_rows16_${arm} COMMAND ${CMAKE_COMMAND} -E env
        STRIX_QRTX_DENSE_ROWS16=${arm} STRIX_CUDA_DECODE_ROWS_MAX=16 STRIX_QRTX_DECODE_KERNELS=1
        STRIX_QRTX_ROWS_KSPLIT=1 STRIX_QRTX_ROWS_CP=0 STRIX_QRTX_KSO=4
        $<TARGET_FILE:qwen_dense_tests> --rows16 ${arm})
    set_tests_properties(qwen_dense_rows16_${arm} PROPERTIES LABELS gpu RUN_SERIAL TRUE)
endforeach()
add_executable(strix-cuda-qwen-serve apps/cuda_qwen_serve_main.cu)
# Shared MTP window controllers (also linked by the HIP engine): measured-cost depth/rows policy and its host adapter.
add_library(strix_mtp_depth STATIC src/mtp_window_controller.cpp src/mtp_depth_window_controller.cpp src/serve/row_budget.cpp
            src/mtp_context_draft.cpp)
target_include_directories(strix_mtp_depth PUBLIC include)
add_executable(qwen_mtp_depth_host_test tests/cuda/qwen_mtp_depth_host_test.cpp)
target_link_libraries(qwen_mtp_depth_host_test PRIVATE strix_mtp_depth)
add_test(NAME qwen_mtp_depth_host COMMAND qwen_mtp_depth_host_test)
add_executable(qwen_mtp_draft_split_test tests/mtp_draft_split_tests.cpp)
target_link_libraries(qwen_mtp_draft_split_test PRIVATE strix_mtp_depth)
add_test(NAME qwen_mtp_draft_split COMMAND qwen_mtp_draft_split_test)
target_link_libraries(strix-cuda-qwen-serve PRIVATE strix_qwen_cuda strix_serve_win strix_mtp_depth)
if(MSVC)
    target_compile_options(strix-cuda-qwen-serve PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_executable(qwen_quant_host_test tests/cuda/qwen_quant_host_test.cpp src/ggml_dequant.cpp src/iq4_nl.cpp src/gguf_reader.cpp)
target_include_directories(qwen_quant_host_test PRIVATE include)
add_executable(qwen_quant_device_test tests/cuda/qwen_quant_device_test.cu src/ggml_dequant.cpp src/iq4_nl.cpp)
target_link_libraries(qwen_quant_device_test PRIVATE strix_qwen_cuda)
target_compile_options(qwen_quant_device_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
# Multi-session decode: cohort window packing (host test) and the cohort gate/bench (GPU).
add_executable(qwen_cohort_plan_test tests/cuda/qwen_cohort_plan_test.cpp)
target_include_directories(qwen_cohort_plan_test PRIVATE include)
add_test(NAME qwen_cohort_plan COMMAND qwen_cohort_plan_test)
# per-request session reservation + the serve's ONE RESOLVER check (host only)
add_executable(qwen_session_reservation_test tests/session_reservation_tests.cpp)
target_include_directories(qwen_session_reservation_test PRIVATE include)
add_test(NAME qwen_session_reservation COMMAND qwen_session_reservation_test)
add_executable(qwen_cohort_e2e tests/cuda/qwen_cohort_e2e.cu)
target_link_libraries(qwen_cohort_e2e PRIVATE strix_qwen_cuda strix_qwen_handoff strix_mtp_depth)
# Multi-segment QSA core == per-segment layer_core on random data (device, no model): qwen_qsa_multi_test <tables dir>
add_executable(qwen_qsa_multi_test tests/cuda/qwen_qsa_multi_test.cu)
target_link_libraries(qwen_qsa_multi_test PRIVATE strix_qwen_cuda)
target_compile_options(qwen_qsa_multi_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
# QSA FAST attention form (qsa_mma) vs exact attention on random inputs + timing: qwen_qsa_mma_test <tables dir>
add_executable(qwen_qsa_mma_test tests/cuda/qwen_qsa_mma_test.cu)
target_link_libraries(qwen_qsa_mma_test PRIVATE strix_qwen_cuda)
target_compile_options(qwen_qsa_mma_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

# Prefix storage and compatibility planning are host-only; no CUDA initialization in their tests.
add_library(strix_qwen_prefix STATIC src/prefix_state_cache.cpp src/cuda/models/qwen/qwen_prefix_identity.cpp)
target_include_directories(strix_qwen_prefix PUBLIC include)
target_link_libraries(strix_qwen_prefix PUBLIC strix_qwen_handoff)
target_link_libraries(strix-cuda-qwen-serve PRIVATE strix_qwen_prefix)
add_executable(qwen_prefix_cache_host_test tests/cuda/qwen_prefix_cache_host_test.cpp)
target_link_libraries(qwen_prefix_cache_host_test PRIVATE strix_qwen_prefix)
target_compile_definitions(qwen_prefix_cache_host_test PRIVATE STRIX_PREFIX_TEST_SOURCE="${CMAKE_SOURCE_DIR}")
add_test(NAME qwen_prefix_cache_host COMMAND qwen_prefix_cache_host_test)
add_executable(qwen_prefix_cache_e2e tests/cuda/qwen_prefix_cache_e2e.cu)
target_link_libraries(qwen_prefix_cache_e2e PRIVATE strix_qwen_cuda strix_qwen_prefix)
if(MSVC)
    target_compile_options(qwen_prefix_cache_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Explicit unqualified same-pack storage candidate. The operator target uses
# the actual production CUDA down/merge implementations from strix_qwen_cuda.
add_executable(qwen_route_bf16_operator tests/cuda/qwen_moe_tests.cu)
target_include_directories(qwen_route_bf16_operator PRIVATE tests/cuda)
target_link_libraries(qwen_route_bf16_operator PRIVATE strix_qwen_cuda)
target_compile_options(qwen_route_bf16_operator PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
add_executable(qwen_route_bf16_e2e tests/cuda/qwen_route_bf16_e2e.cu)
target_link_libraries(qwen_route_bf16_e2e PRIVATE strix_qwen_cuda)
target_compile_options(qwen_route_bf16_e2e PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    foreach(target qwen_route_bf16_operator qwen_route_bf16_e2e)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endforeach()
else()
    # Existing Qwen sources use the Windows spelling of 64-bit stdio.
    # POSIX aliases preserve the byte offsets and do not alter GPU arithmetic.
    target_compile_definitions(strix_qwen_cuda PUBLIC _fseeki64=fseeko _ftelli64=ftello)
    foreach(target qwen_route_bf16_operator qwen_route_bf16_e2e)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
    endforeach()
endif()

# Armed diagnostic: actual production Q8 proposer, BF16 target and verification.
add_executable(qwen_bf16_mtp_e2e tests/cuda/qwen_bf16_mtp_e2e.cu)
target_link_libraries(qwen_bf16_mtp_e2e PRIVATE strix_qwen_cuda)
target_compile_options(qwen_bf16_mtp_e2e PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(qwen_bf16_mtp_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
else()
    target_compile_options(qwen_bf16_mtp_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
endif()

# rtxmax exact decode kernels: each new kernel is byte-compared with the old form on random data (device, no model).
foreach(_qx draft_step gdn_decode hc_rows qsa_decode rows_glue)
    add_executable(qwen_${_qx}_exact_test tests/cuda/qwen_${_qx}_exact_test.cu)
    target_link_libraries(qwen_${_qx}_exact_test PRIVATE strix_qwen_cuda)
    target_compile_options(qwen_${_qx}_exact_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endforeach()
# 16-bit recurrent state (STRIX_RECURRENT_STATE=f16|bf16) in the CUDA GDN recurrence: codec bit-equality with the host,
# per-row rounding equalities, f32 incumbent bits. Needs the GPU (index 0 only: set CUDA_VISIBLE_DEVICES=0).
add_executable(qwen_gdn_state16_test tests/cuda/qwen_gdn_state16_test.cu)
target_link_libraries(qwen_gdn_state16_test PRIVATE strix_qwen_cuda)
target_compile_options(qwen_gdn_state16_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(qwen_gdn_state16_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_test(NAME qwen_gdn_state16 COMMAND qwen_gdn_state16_test)
set_tests_properties(qwen_gdn_state16 PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")

# Profile resolution of the CUDA Qwen apps: one resolver, one engine-option parser, named serve/CLI differences (host-only).
add_executable(qwen_cuda_profile_host_test tests/cuda/qwen_cuda_profile_host_test.cpp)
target_link_libraries(qwen_cuda_profile_host_test PRIVATE strix_qwen_prefix)
add_test(NAME qwen_cuda_profile_host COMMAND qwen_cuda_profile_host_test)
# Gate kit: teacher-forced logit dump (layout of tools/strix-gate/fast_logit_gate.py); see scripts/bench/cuda/qwen_fast/README.md.
add_executable(qwen_tf_logits tests/cuda/qwen_tf_logits.cu)
target_link_libraries(qwen_tf_logits PRIVATE strix_qwen_cuda strix_qwen_handoff)
if(MSVC)
    target_compile_options(qwen_tf_logits PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
# GPU unit test of tiered expert residency: full route vs tiered route (CPU executor) on random weights, bitwise.
add_executable(qwen_tiered_moe_tests tests/cuda/qwen_tiered_moe_tests.cu)
target_link_libraries(qwen_tiered_moe_tests PRIVATE strix_qwen_cuda)
target_compile_options(qwen_tiered_moe_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(qwen_tiered_moe_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# serve == bench residency command line (host only, no CUDA)
add_executable(strix-expert-residency-parity-tests tests/expert_residency_parity_tests.cpp)
target_link_libraries(strix-expert-residency-parity-tests PRIVATE strix_residency)
add_test(NAME strix-expert-residency-parity-tests COMMAND strix-expert-residency-parity-tests)
