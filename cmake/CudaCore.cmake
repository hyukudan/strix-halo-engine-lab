# Shared CUDA core (model-agnostic kernels: dense rows, norms, rope, decode attention, GDN decode).
# Usable from the full build (-DSTRIX_CUDA_CORE=ON) and standalone (-DSTRIX_CUDA_CORE_ONLY=ON, CUDA language only).
if(TARGET strix_cuda_core_kernels)
    return()
endif()
enable_language(CUDA)
include(CTest)
include("${CMAKE_CURRENT_LIST_DIR}/CudaTokenSampling.cmake")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()
get_filename_component(_cc_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# NVFP4 (TURBO weights) needs sm_120a PTX (cvt.e2m1x2): own object library, built for 120a only.
add_library(strix_cuda_core_fp4 OBJECT ${_cc_root}/src/cuda/core/dense/nvfp4.cu
    ${_cc_root}/src/cuda/core/nvfp4_tile.cu  # FAST4 W4A4 tile GEMM (shared CUDA core, sm_120 block-scaled e2m1)
    ${_cc_root}/src/cuda/core/grouped_w4a4.cu)  # FAST W4A4 grouped expert projection (moe_core/grouped_w4a4.hpp)
set_target_properties(strix_cuda_core_fp4 PROPERTIES CUDA_ARCHITECTURES "120a-real")
target_include_directories(strix_cuda_core_fp4 PUBLIC ${_cc_root}/include)
target_include_directories(strix_cuda_core_fp4 PRIVATE ${_cc_root}/src)
target_compile_options(strix_cuda_core_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_cuda_core_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_library(strix_cuda_core_kernels STATIC
    $<TARGET_OBJECTS:strix_cuda_core_fp4>
    ${_cc_root}/src/cuda/core/dense/dense.cu
    ${_cc_root}/src/cuda/core/dense/dense_mma.cu
    ${_cc_root}/src/cuda/core/dense/dense_big.cu
    ${_cc_root}/src/cuda/core/norms/norms.cu
    ${_cc_root}/src/cuda/core/norms/rope.cu
    ${_cc_root}/src/cuda/core/vision/vision_tower.cu  # FP32 vision tower kernels (vision::CudaEncoder)
    ${_cc_root}/src/cuda/core/attn_production.cu
    ${_cc_root}/src/cuda/core/attn_flash.cu
    ${_cc_root}/src/cuda/core/attn_flash_decode_split.cu  # FAST decode key split of the prepared D128 flash
    ${_cc_root}/src/cuda/core/attn_flash8.cu
    ${_cc_root}/src/cuda/core/recur_parity.cu
    ${_cc_root}/src/cuda/core/recur_chunk.cu  # FAST chunked GDN prefill (gdn-chunk16-f32-v1)
    ${_cc_root}/src/cuda/core/step_inputs.cu  # X3 step-input scatter (plain / tree / fed rows / slots)
    ${_cc_root}/src/cuda/core/lookup_cuda.cu  # K10c token lookup family
    ${_cc_root}/src/cuda/core/embed_splice.cu)  # provided embedding rows (session/embed_splice_device.hpp)
target_link_libraries(strix_cuda_core_kernels PUBLIC strix_cuda_token_sampling)
target_include_directories(strix_cuda_core_kernels PUBLIC ${_cc_root}/include PRIVATE ${_cc_root}/src)
# Identity: no FMA contraction, IEEE division and sqrt, no flush-to-zero (as the existing Qwen CUDA build).
target_compile_options(strix_cuda_core_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_cuda_core_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/StrixDevice.cmake)
# qwen35 CUDA engine (reusable library: forward_rows / commit / MTP) + the decode CLI.
add_library(strix_qwen35_cuda STATIC ${_cc_root}/src/qwen35/cuda/engine.cu ${_cc_root}/src/gguf_reader.cpp
    ${_cc_root}/src/qwen35/config.cpp ${_cc_root}/src/qwen35/family.cpp ${_cc_root}/src/model_family_registry.cpp)
target_link_libraries(strix_qwen35_cuda PUBLIC strix_cuda_core_kernels)
target_include_directories(strix_qwen35_cuda PUBLIC ${_cc_root}/include)
target_compile_options(strix_qwen35_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_qwen35_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_executable(strix-cuda-qwen35-decode ${_cc_root}/apps/cuda_qwen35_decode_main.cu)
target_link_libraries(strix-cuda-qwen35-decode PRIVATE strix_qwen35_cuda strix_device)
target_compile_options(strix-cuda-qwen35-decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix-cuda-qwen35-decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Shared multi-request decoder (serve/cohort_decode) bound to the qwen35 slots: N requests per forward, ids == solo.
add_executable(strix-cuda-qwen35-cohort ${_cc_root}/apps/cuda_qwen35_cohort_main.cu ${_cc_root}/src/serve/cohort_decode.cpp
    ${_cc_root}/src/serve/row_budget.cpp ${_cc_root}/src/serve/step_planner.cpp ${_cc_root}/src/mtp_context_draft.cpp
    ${_cc_root}/src/token_sampling.cpp ${_cc_root}/src/speculation_rng.cpp
    ${_cc_root}/src/mtp_selection_transaction.cpp)  # shared window selection (sampled/constrained lanes)
target_link_libraries(strix-cuda-qwen35-cohort PRIVATE strix_qwen35_cuda)
if(MSVC)
    target_compile_options(strix-cuda-qwen35-cohort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

add_executable(cuda_core_test ${_cc_root}/tests/cuda_core/cuda_core_test.cu)
target_link_libraries(cuda_core_test PRIVATE strix_cuda_core_kernels)
target_compile_options(cuda_core_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

add_executable(qwen35_cuda_engine_test ${_cc_root}/tests/qwen35/cuda_engine_test.cu)
target_link_libraries(qwen35_cuda_engine_test PRIVATE strix_qwen35_cuda strix_spec)  # strix_spec: token-tree walk (accept_tree)
target_compile_options(qwen35_cuda_engine_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(qwen35_cuda_engine_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/Qwen35Spec.cmake)
add_executable(cuda_core_nvfp4_test ${_cc_root}/tests/cuda_core/nvfp4_test.cu)
target_link_libraries(cuda_core_nvfp4_test PRIVATE strix_cuda_core_kernels)
target_compile_options(cuda_core_nvfp4_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_nvfp4_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Shared R1 MoE decode kernels over launch geometries (columns per warp, runs in flight): bits vs the one-column one-run
# geometry + us per launch on synthetic top-8 experts (--check-only is the identity form).
add_executable(cuda_core_moe_r1_geometry_bench ${_cc_root}/tests/cuda_core/moe_r1_geometry_bench.cu)
target_include_directories(cuda_core_moe_r1_geometry_bench PRIVATE ${_cc_root}/src ${_cc_root}/include)
target_compile_features(cuda_core_moe_r1_geometry_bench PRIVATE cxx_std_20)
target_compile_options(cuda_core_moe_r1_geometry_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_moe_r1_geometry_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8,/wd4702>)
endif()

# Shared R1 MoE decode route + plan forms: inline-routed gate_up and the expert-union rows launches, bitwise vs the
# route launch + rows launches, then graph-captured visit timings (--check-only is the identity form).
add_executable(cuda_core_moe_r1_route_union_bench ${_cc_root}/tests/cuda_core/moe_r1_route_union_bench.cu)
target_include_directories(cuda_core_moe_r1_route_union_bench PRIVATE ${_cc_root}/src ${_cc_root}/include)
target_compile_features(cuda_core_moe_r1_route_union_bench PRIVATE cxx_std_20)
target_compile_options(cuda_core_moe_r1_route_union_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_moe_r1_route_union_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8,/wd4702>)
endif()

# Shared greedy head pick: identity vs the host reference + us per call (--check-only is the ctest form).
add_executable(cuda_core_head_argmax_bench ${_cc_root}/tests/cuda_core/head_argmax_bench.cu)
target_include_directories(cuda_core_head_argmax_bench PRIVATE ${_cc_root}/include)
target_compile_features(cuda_core_head_argmax_bench PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(cuda_core_head_argmax_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# EXACT screened greedy LM head (src/cuda/core/head_screened.cuh): device test (monotone G, mapped-row identity vs the incumbent
# R=1 kernel, screened Pick == full head) and, with --bench, full vs screened timing (refused unless the tests PASS).
add_executable(cuda_core_screened_head_test ${_cc_root}/tests/cuda_core/screened_head_cuda.cu)
target_link_libraries(cuda_core_screened_head_test PRIVATE strix_cuda_core_kernels)
target_include_directories(cuda_core_screened_head_test PRIVATE ${_cc_root}/include ${_cc_root}/src)
target_compile_options(cuda_core_screened_head_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_screened_head_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Long-context prefill attention bench + pipe identity check (scripts: --check-only is the ctest form).
add_executable(cuda_core_flash_bench ${_cc_root}/tests/cuda_core/flash_bench.cu)
target_link_libraries(cuda_core_flash_bench PRIVATE strix_cuda_core_kernels)
if(MSVC)
    target_compile_options(cuda_core_flash_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
# Prepared D128 decode key split vs the prepared flash step path: tolerance vs a double reference, determinism, timing.
add_executable(cuda_core_decode_split_bench ${_cc_root}/tests/cuda_core/decode_split_bench.cu)
target_link_libraries(cuda_core_decode_split_bench PRIVATE strix_cuda_core_kernels)
if(MSVC)
    target_compile_options(cuda_core_decode_split_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
# Lanes form of the decode key split: every lane bitwise equal to the one-row launch of that row (no model).
add_executable(cuda_core_decode_split_lanes_fixture ${_cc_root}/tests/cuda_core/decode_split_lanes_fixture.cu)
target_link_libraries(cuda_core_decode_split_lanes_fixture PRIVATE strix_cuda_core_kernels)
if(MSVC)
    target_compile_options(cuda_core_decode_split_lanes_fixture PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_executable(cuda_core_dense_big_test ${_cc_root}/tests/cuda_core/dense_big_test.cu)
target_link_libraries(cuda_core_dense_big_test PRIVATE strix_cuda_core_kernels)
target_compile_options(cuda_core_dense_big_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_dense_big_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Device fixture: grouped F1 rows launch (8 jobs, different inputs) == separate legacy launches, bitwise.
add_executable(cuda_core_rows_grouped_test ${_cc_root}/tests/rows/rows_grouped_test.cu)
target_link_libraries(cuda_core_rows_grouped_test PRIVATE strix_cuda_core_kernels)
target_include_directories(cuda_core_rows_grouped_test PRIVATE ${_cc_root}/include ${_cc_root}/src)
target_compile_options(cuda_core_rows_grouped_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_rows_grouped_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Host-only: converted-pack GGUF writer round trip (no GPU).
add_executable(gguf_pack_writer_test ${_cc_root}/tests/gguf_pack_writer_test.cpp ${_cc_root}/src/gguf_reader.cpp)
target_include_directories(gguf_pack_writer_test PRIVATE ${_cc_root}/include)

if(MSVC)
    set(_cc_mini_warnings -Xcompiler=/W4)
    set(_cc_mini_host_werror -Xcompiler=/WX)
else()
    # nvcc emits GNU line markers in its generated host C++; -Wpedantic rejects that output.
    # Ordinary CXX targets retain the full warning set, and CUDA keeps conversion/shadow and Werror.
    set(_cc_mini_warnings -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow)
    set(_cc_mini_host_werror -Xcompiler=-Werror)
endif()
if(NOT DEFINED STRIX_WARNINGS_AS_ERRORS OR STRIX_WARNINGS_AS_ERRORS)
    list(APPEND _cc_mini_warnings ${_cc_mini_host_werror} -Werror=all-warnings)
endif()

if(TARGET strix_inkling2_mini_checkpoint)
    add_executable(strix-inkling2-mini-cuda ${_cc_root}/apps/inkling2_mini_cuda.cu)
    target_include_directories(strix-inkling2-mini-cuda PRIVATE ${_cc_root}/src)
    target_link_libraries(strix-inkling2-mini-cuda PRIVATE strix_cuda_core_kernels strix_inkling2_mini_checkpoint strix_core strix_serve)
    target_compile_options(strix-inkling2-mini-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false;--expt-relaxed-constexpr>)
    target_compile_options(strix-inkling2-mini-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${_cc_mini_warnings}>")
    if(MSVC)
        target_compile_options(strix-inkling2-mini-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
endif()

if(BUILD_TESTING)
    # Grouped BF16 expert projection, tall slab schedule: device fixture (every form bitwise the staged schedule) + no-model
    # bench (--bench [tokens]).
    add_executable(strix-cuda-grouped-tall-test ${_cc_root}/tests/moe_core/grouped_tall_cuda.cu)
    target_include_directories(strix-cuda-grouped-tall-test PRIVATE ${_cc_root}/src ${_cc_root}/include)
    target_compile_options(strix-cuda-grouped-tall-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    target_compile_options(strix-cuda-grouped-tall-test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${_cc_mini_warnings}>")
    if(MSVC)
        target_compile_options(strix-cuda-grouped-tall-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    add_test(NAME strix-cuda-grouped-tall-test COMMAND strix-cuda-grouped-tall-test)
    set_tests_properties(strix-cuda-grouped-tall-test PROPERTIES LABELS "gpu;cuda;moe;exactness")
    # Grouped W4A4 expert tile: device fixture (quantiser bitwise, projection bound) + no-model bench (--bench). sm_120a only.
    add_executable(strix-cuda-grouped-w4a4-test ${_cc_root}/tests/moe_core/grouped_w4a4_cuda.cu)
    target_link_libraries(strix-cuda-grouped-w4a4-test PRIVATE strix_cuda_core_kernels)
    target_include_directories(strix-cuda-grouped-w4a4-test PRIVATE ${_cc_root}/src ${_cc_root}/include)
    set_target_properties(strix-cuda-grouped-w4a4-test PROPERTIES CUDA_ARCHITECTURES "120a-real")
    target_compile_options(strix-cuda-grouped-w4a4-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-cuda-grouped-w4a4-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
endif()
if(BUILD_TESTING)
    add_executable(strix-cuda-inkling2-mini-dense-test ${_cc_root}/tests/inkling2/mini_dense_cuda.cu)
    target_link_libraries(strix-cuda-inkling2-mini-dense-test PRIVATE strix_cuda_core_kernels)
    add_executable(strix-cuda-inkling2-mini-residual-test ${_cc_root}/tests/inkling2/mini_residual_device.cu)
    target_link_libraries(strix-cuda-inkling2-mini-residual-test PRIVATE strix_cuda_core_kernels)
    add_executable(strix-cuda-inkling2-mini-short-conv-test ${_cc_root}/tests/inkling2/mini_short_conv_cuda.cu)
    add_executable(strix-cuda-inkling2-mini-step-graph-test ${_cc_root}/tests/inkling2/step_graph_cuda.cu)
    add_executable(strix-cuda-inkling2-mini-attention-test ${_cc_root}/tests/inkling2/mini_attention_reference_cuda.cu)
    target_link_libraries(strix-cuda-inkling2-mini-attention-test PRIVATE strix_cuda_core_kernels)
    add_executable(strix-cuda-resident-lowbit-refusal-test ${_cc_root}/tests/cuda/resident_lowbit_refusal_tests.cu)
    target_link_libraries(strix-cuda-resident-lowbit-refusal-test PRIVATE strix_cuda_core_kernels)
    set(_cc_mini_tests strix-cuda-inkling2-mini-short-conv-test strix-cuda-inkling2-mini-attention-test
        strix-cuda-inkling2-mini-dense-test strix-cuda-inkling2-mini-residual-test strix-cuda-inkling2-mini-step-graph-test
        strix-cuda-resident-lowbit-refusal-test)
    if(TARGET strix_inkling2_mini_checkpoint)
        add_executable(strix-cuda-inkling2-mini-dspark-test ${_cc_root}/tests/inkling2/mini_dspark_device.cu)
        target_link_libraries(strix-cuda-inkling2-mini-dspark-test PRIVATE
            strix_cuda_core_kernels strix_inkling2_mini_checkpoint strix_core strix_serve)
        list(APPEND _cc_mini_tests strix-cuda-inkling2-mini-dspark-test)
    endif()
    if(TARGET strix_core)
        add_executable(strix-cuda-inkling2-mini-moe-test ${_cc_root}/tests/inkling2/mini_moe_cuda.cu)
        target_link_libraries(strix-cuda-inkling2-mini-moe-test PRIVATE strix_core strix_cuda_core_kernels)
        # Same C4702 exemption as the root CXX flags: inlined constant format branches are unreachable in the host pass.
        if(MSVC)
            target_compile_options(strix-cuda-inkling2-mini-moe-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4702>)
        endif()
        # Use the requested architecture for a Windows SM120a build, including this fixture.
        if(NOT MSVC OR NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^120a(-real)?$")
            set_target_properties(strix-cuda-inkling2-mini-moe-test PROPERTIES CUDA_ARCHITECTURES "80;89;120")
        endif()
        list(APPEND _cc_mini_tests strix-cuda-inkling2-mini-moe-test)
    endif()
    if(TARGET strix_inkling2_mini_checkpoint)
        set(_cc_mini_verify_target strix-cuda-inkling2-mini-verify-rows-test)
        add_executable(${_cc_mini_verify_target} ${_cc_root}/tests/inkling2/mini_verify_rows_cuda.cu)
        target_link_libraries(${_cc_mini_verify_target} PRIVATE
            strix_cuda_core_kernels strix_inkling2_mini_checkpoint strix_core)
        # Serving cohort lanes vs each lane alone: bitwise logits, appended K/V rows, histories + build digests.
        add_executable(strix-cuda-inkling2-mini-cohort-lanes-test ${_cc_root}/tests/inkling2/mini_cohort_lanes_cuda.cu)
        target_link_libraries(strix-cuda-inkling2-mini-cohort-lanes-test PRIVATE
            strix_cuda_core_kernels strix_inkling2_mini_checkpoint strix_core)
        list(APPEND _cc_mini_verify_target strix-cuda-inkling2-mini-cohort-lanes-test)
        # A sampled serving lane reads the scores of its own (screened) step; needs the model checkpoint.
        add_executable(strix-cuda-inkling2-mini-screened-scores-test
            ${_cc_root}/tests/inkling2/mini_screened_scores_cuda.cu)
        target_link_libraries(strix-cuda-inkling2-mini-screened-scores-test PRIVATE
            strix_cuda_core_kernels strix_inkling2_mini_checkpoint strix_core)
        list(APPEND _cc_mini_verify_target strix-cuda-inkling2-mini-screened-scores-test)
    endif()
    foreach(t IN LISTS _cc_mini_tests _cc_mini_verify_target)
        target_include_directories(${t} PRIVATE ${_cc_root}/src ${_cc_root}/include)
        target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
        target_compile_options(${t} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${_cc_mini_warnings}>")
        if(MSVC)
            target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
        endif()
    endforeach()
    # The verify fixture requires a separately supplied model checkpoint.
    foreach(t IN LISTS _cc_mini_tests)
        add_test(NAME ${t} COMMAND ${t})
        set_tests_properties(${t} PROPERTIES LABELS "gpu;cuda;inkling2")
    endforeach()
    add_test(NAME gguf_pack_writer_test COMMAND gguf_pack_writer_test)
    foreach(t cuda_core_test qwen35_cuda_engine_test cuda_core_nvfp4_test cuda_core_dense_big_test)
        add_test(NAME ${t} COMMAND ${t})
        set_tests_properties(${t} PROPERTIES LABELS "gpu;cuda")
    endforeach()
    add_test(NAME cuda_core_flash_bench COMMAND cuda_core_flash_bench --check-only)  # flash pipes bitwise identical, flash8 finite
    # host only (no CUDA call): every launch of the decode split bench is admitted by the engine's launch predicates
    add_test(NAME cuda_core_decode_split_bench_args COMMAND cuda_core_decode_split_bench --check-args)
    set_tests_properties(cuda_core_flash_bench PROPERTIES LABELS "gpu;cuda")
endif()

# FAST8 needs a feature-specific image and must not select a generic SM120
# image whose MMA helper traps. Other device generations are rejected at runtime.
get_target_property(_cc_archs strix_cuda_core_kernels CUDA_ARCHITECTURES)
if(_cc_archs MATCHES "(^|;)120a(-real)?(;|$)" AND NOT _cc_archs MATCHES "(^|;)120(-real|-virtual)?(;|$)")
    target_compile_definitions(strix_cuda_core_kernels PRIVATE STRIX_CUDA_FAST8_SM120A=1)
endif()
