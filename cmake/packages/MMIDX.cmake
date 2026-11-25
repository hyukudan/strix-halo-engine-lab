if(BUILD_TESTING)
    add_executable(strix-mmidx-host-test tests/rows/mmidx_host_test.cpp)
    target_include_directories(strix-mmidx-host-test PRIVATE src)
    target_compile_features(strix-mmidx-host-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-mmidx-host-test)
    add_test(NAME strix-mmidx-host-test COMMAND strix-mmidx-host-test)
    set_tests_properties(strix-mmidx-host-test PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_MMIDX_CUDA_FIXTURE "Build the input-major F32 rows conformance fixture for sm_120a" OFF)
if(STRIX_MMIDX_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-mmidx-cuda-fixture tests/rows/mmidx_cuda_fixture.cu
        src/cuda/core/dense/dense.cu src/cuda/core/dense/dense_mma.cu src/cuda/core/dense/nvfp4.cu)
    target_include_directories(strix-mmidx-cuda-fixture PRIVATE include src)
    set_target_properties(strix-mmidx-cuda-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real")
    target_compile_options(strix-mmidx-cuda-fixture PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
    if(MSVC)
        target_compile_options(strix-mmidx-cuda-fixture PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
    endif()
    # The device operator runs this executable separately from CTest.
endif()
