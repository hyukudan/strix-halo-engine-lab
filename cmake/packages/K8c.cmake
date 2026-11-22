if(BUILD_TESTING)
    add_executable(strix-k8c-mix-host-test tests/cuda_core/mix_host_test.cpp)
    target_include_directories(strix-k8c-mix-host-test PRIVATE src)
    target_link_libraries(strix-k8c-mix-host-test PRIVATE strix_exec_catalog)
    strix_target_warnings(strix-k8c-mix-host-test)
    if(MSVC)
        target_compile_options(strix-k8c-mix-host-test PRIVATE /fp:precise)
    else()
        target_compile_options(strix-k8c-mix-host-test PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-k8c-mix-host-test COMMAND strix-k8c-mix-host-test)
    set_tests_properties(strix-k8c-mix-host-test PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_K8C_CUDA_FIXTURE "Compile the K8c residual readout fixture for sm_120a" OFF)
if(STRIX_K8C_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-k8c-mix-cuda-fixture tests/cuda_core/mix_cuda_fixture.cu)
    target_include_directories(strix-k8c-mix-cuda-fixture PRIVATE include src)
    set_target_properties(strix-k8c-mix-cuda-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real")
    target_compile_options(strix-k8c-mix-cuda-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-k8c-mix-cuda-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8,/fp:precise>)
    else()
        target_compile_options(strix-k8c-mix-cuda-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math,-ffp-contract=off>)
    endif()
    # Operator execution only. CTest runs the CPU oracle and catalogue checks.
endif()
