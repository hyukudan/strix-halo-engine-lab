# B3 row-owner transition (residual add + bank readout + input norms + a8g32 image in one CUDA decode leaf).
if(TARGET strix_exec_cuda_decode_catalog)
    if(BUILD_TESTING)
        # Host only: bind() records launches, nothing runs on a GPU.
        add_executable(strix-cuda-decode-transition-tests tests/exec/cuda_decode_transition_tests.cpp)
        target_include_directories(strix-cuda-decode-transition-tests PRIVATE include src)
        target_link_libraries(strix-cuda-decode-transition-tests PRIVATE strix_exec strix_exec_cuda_decode_catalog)
        target_compile_features(strix-cuda-decode-transition-tests PRIVATE cxx_std_20)
        strix_target_warnings(strix-cuda-decode-transition-tests)
        add_test(NAME cuda_decode_transition_host COMMAND strix-cuda-decode-transition-tests)
        set_tests_properties(cuda_decode_transition_host PROPERTIES LABELS "cpu;plan" TIMEOUT 30)
    endif()

    option(STRIX_TRANSITION_CUDA_FIXTURE "Build the sm_120a TRANSITION fixture and no-model bench" OFF)
    if(STRIX_TRANSITION_CUDA_FIXTURE)
        add_executable(strix-transition-cuda-fixture tests/cuda_core/transition_cuda_fixture.cu)
        target_include_directories(strix-transition-cuda-fixture PRIVATE include src)
        target_link_libraries(strix-transition-cuda-fixture PRIVATE strix_exec strix_exec_cuda_decode_catalog
            strix_cuda_core_kernels)
        set_target_properties(strix-transition-cuda-fixture PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 120a-real)
        target_compile_options(strix-transition-cuda-fixture PRIVATE --fmad=false --prec-div=true --prec-sqrt=true
            --ftz=false)
        if(MSVC)
            target_compile_options(strix-transition-cuda-fixture PRIVATE -Xcompiler=/W4,/WX,/EHsc,/Zc:preprocessor,/utf-8)
        else()
            target_compile_options(strix-transition-cuda-fixture PRIVATE -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-Werror)
        endif()
        if(STRIX_WARNINGS_AS_ERRORS)
            target_compile_options(strix-transition-cuda-fixture PRIVATE -Werror=all-warnings)
        endif()
        # Device execution is an operator task. Never register this binary with CTest.
    endif()
endif()
