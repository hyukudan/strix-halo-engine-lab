if(BUILD_TESTING)
    add_executable(strix-k9c-recur-plan-test tests/cuda_core/recur_plan_test.cpp tests/cuda_core/recur_cases.cpp)
    target_include_directories(strix-k9c-recur-plan-test PRIVATE src)
    target_link_libraries(strix-k9c-recur-plan-test PRIVATE strix_state)
    strix_target_warnings(strix-k9c-recur-plan-test)
    add_test(NAME strix-k9c-recur-plan-test COMMAND strix-k9c-recur-plan-test)
    set_tests_properties(strix-k9c-recur-plan-test PROPERTIES LABELS "cpu;plan;S1")
endif()

option(STRIX_K9C_CUDA_FIXTURE "Build the K9c recurrent CUDA conformance fixture for sm_120a" OFF)
if(STRIX_K9C_CUDA_FIXTURE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(strix-k9c-recur-fixture
        tests/cuda_core/recur_fixture.cu
        tests/cuda_core/recur_cases.cpp
        src/cuda/core/recur_cuda.cu
        src/cuda_core/gdn/gdn.cu
        src/cuda_core/gdn/gdn_chunk.cu)
    target_include_directories(strix-k9c-recur-fixture PRIVATE src)
    target_link_libraries(strix-k9c-recur-fixture PRIVATE strix_state CUDA::cudart)
    set_target_properties(strix-k9c-recur-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real" CUDA_RUNTIME_LIBRARY None)
    target_compile_options(strix-k9c-recur-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-k9c-recur-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    # Native execution is operator initiated. CTest only runs the CPU mapping test.
endif()
