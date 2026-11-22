if(BUILD_TESTING)
    add_executable(strix-mixlat-host-test tests/cuda_core/mixlat_host_test.cpp)
    target_include_directories(strix-mixlat-host-test PRIVATE src)
    target_link_libraries(strix-mixlat-host-test PRIVATE strix_exec_catalog)
    strix_target_warnings(strix-mixlat-host-test)
    add_test(NAME strix-mixlat-host-test COMMAND strix-mixlat-host-test)
    set_tests_properties(strix-mixlat-host-test PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_MIXLAT_CUDA_FIXTURE "Build the sm_120a MIXLAT fixture and no-model entry cases" OFF)
if(STRIX_MIXLAT_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-mixlat-cuda-fixture tests/cuda_core/mixlat_cuda_fixture.cu
        tests/cuda_core/mixlat_cuda_include_order.cu
        src/cuda/core/dense/dense_mma.cu)
    target_include_directories(strix-mixlat-cuda-fixture PRIVATE include src)
    target_link_libraries(strix-mixlat-cuda-fixture PRIVATE strix_exec_catalog)
    set_target_properties(strix-mixlat-cuda-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 120a-real)
    target_compile_options(strix-mixlat-cuda-fixture PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
    if(MSVC)
        # Match the project's C4702 suppression for the incumbent constexpr return in dense_mma.cu.
        target_compile_options(strix-mixlat-cuda-fixture PRIVATE -Xcompiler=/W4,/WX,/wd4702,/EHsc,/Zc:preprocessor,/utf-8)
    else()
        # nvcc emits GNU line markers; ordinary CXX warnings remain unchanged.
        target_compile_options(strix-mixlat-cuda-fixture PRIVATE -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-Werror)
    endif()
    if(STRIX_WARNINGS_AS_ERRORS)
        target_compile_options(strix-mixlat-cuda-fixture PRIVATE -Werror=all-warnings)
    endif()
    # Device execution is an operator task. Never register this binary with CTest.
endif()
