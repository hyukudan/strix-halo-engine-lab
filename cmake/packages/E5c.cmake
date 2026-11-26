if(BUILD_TESTING AND STRIX_FAMILY_MINIMAX)
    add_executable(strix-minimax-rtx-binding-tests tests/minimax_rtx_binding_tests.cpp)
    target_include_directories(strix-minimax-rtx-binding-tests PRIVATE src)
    target_compile_features(strix-minimax-rtx-binding-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-minimax-rtx-binding-tests)
    add_test(NAME strix-minimax-rtx-binding-tests COMMAND strix-minimax-rtx-binding-tests)
    set_tests_properties(strix-minimax-rtx-binding-tests PROPERTIES LABELS "cpu")
endif()

option(STRIX_MINIMAX_CUDA "Build the Windows MiniMax RTX prefill producer and its explicit device fixtures" OFF)
if(STRIX_MINIMAX_CUDA)
    if(NOT WIN32 OR NOT TARGET strix_cuda_core_kernels)
        message(FATAL_ERROR "STRIX_MINIMAX_CUDA requires Windows and STRIX_CUDA_CORE=ON")
    endif()

    set(mm_cuda src/cuda/models/minimax)
    add_library(strix_minimax_rtx_kernels STATIC ${mm_cuda}/fp32_identity.cu ${mm_cuda}/mm_rtx_kernels.cu)
    target_link_libraries(strix_minimax_rtx_kernels PRIVATE strix_cuda_core_kernels)
    add_executable(mm-rtx-prefill ${mm_cuda}/mm_rtx_prefill_main.cu ${mm_cuda}/mm_rtx_pack.cpp ${mm_cuda}/mm_rtx_weights.cpp
        src/bp_codec.cpp src/handoff.cpp)
    target_link_libraries(mm-rtx-prefill PRIVATE strix_minimax_rtx_kernels strix_serve_json ws2_32)

    add_executable(mm_fp32_identity_tests ${mm_cuda}/mm_fp32_identity_tests.cu)
    add_executable(mm_fp32_core_bench ${mm_cuda}/bench/core_bench.cu)
    add_executable(strix-minimax-rtx-norm-fixture tests/cuda_core/minimax_norm_fixture.cu)
    target_link_libraries(strix-minimax-rtx-norm-fixture PRIVATE strix_minimax_rtx_kernels)
    add_executable(strix-minimax-rtx-route-fixture tests/cuda_core/minimax_route_fixture.cu)
    target_link_libraries(strix-minimax-rtx-route-fixture PRIVATE strix_minimax_rtx_kernels)
    add_executable(strix-minimax-rtx-index-fixture tests/cuda_core/minimax_index_fixture.cu)
    target_link_libraries(strix-minimax-rtx-index-fixture PRIVATE strix_minimax_rtx_kernels)
    add_executable(strix-minimax-rtx-prep-fixture tests/cuda_core/minimax_prep_fixture.cu)
    target_link_libraries(strix-minimax-rtx-prep-fixture PRIVATE strix_minimax_rtx_kernels)
    add_executable(strix-minimax-rtx-attn-fixture tests/cuda_core/minimax_attn_fixture.cu)
    target_link_libraries(strix-minimax-rtx-attn-fixture PRIVATE strix_minimax_rtx_kernels)
    if(BUILD_TESTING)
        add_test(NAME strix-minimax-rtx-index-fixture COMMAND strix-minimax-rtx-index-fixture)
        set_tests_properties(strix-minimax-rtx-index-fixture PROPERTIES LABELS "cuda;gpu")
        add_test(NAME strix-minimax-rtx-prep-fixture COMMAND strix-minimax-rtx-prep-fixture)
        set_tests_properties(strix-minimax-rtx-prep-fixture PROPERTIES LABELS "cuda;gpu" TIMEOUT 60)
        add_test(NAME strix-minimax-rtx-attn-fixture COMMAND strix-minimax-rtx-attn-fixture)
        set_tests_properties(strix-minimax-rtx-attn-fixture PROPERTIES LABELS "cuda;gpu" TIMEOUT 60)
    endif()
    foreach(mm_target IN ITEMS strix_minimax_rtx_kernels mm-rtx-prefill mm_fp32_identity_tests mm_fp32_core_bench
                               strix-minimax-rtx-norm-fixture strix-minimax-rtx-route-fixture strix-minimax-rtx-index-fixture
                               strix-minimax-rtx-prep-fixture strix-minimax-rtx-attn-fixture)
        target_include_directories(${mm_target} PRIVATE include src)
        set_target_properties(${mm_target} PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "120a-real" CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
        target_compile_options(${mm_target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:strict,/EHsc,/Zc:preprocessor,/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/fp:strict>)
    endforeach()
    unset(mm_cuda)
    # The norm and route fixtures and the bench are operator-run targets.
endif()
