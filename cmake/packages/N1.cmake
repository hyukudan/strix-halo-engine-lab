if(BUILD_TESTING)
    add_executable(strix-n1-gfx1151-host-test tests/numeric_math_gfx1151_test.cpp)
    target_include_directories(strix-n1-gfx1151-host-test PRIVATE include)
    target_compile_features(strix-n1-gfx1151-host-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-n1-gfx1151-host-test)
    if(MSVC)
        target_compile_options(strix-n1-gfx1151-host-test PRIVATE /fp:strict)
    else()
        target_compile_options(strix-n1-gfx1151-host-test PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-n1-gfx1151-host-test COMMAND strix-n1-gfx1151-host-test)
    set_tests_properties(strix-n1-gfx1151-host-test PROPERTIES LABELS "cpu;N1")
endif()

option(STRIX_N1_CUDA_FIXTURE "Compile the N1 gfx1151 CUDA fixture for sm_120a" OFF)
if(STRIX_N1_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-n1-gfx1151-cuda-fixture
        tests/cuda_core/numeric_math_gfx1151_fixture.cu
        src/cuda/core/amd_math_tables.cu)
    target_include_directories(strix-n1-gfx1151-cuda-fixture PRIVATE include src)
    set_target_properties(strix-n1-gfx1151-cuda-fixture PROPERTIES
        CUDA_ARCHITECTURES "120a-real" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strix-n1-gfx1151-cuda-fixture PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
    if(MSVC)
        target_compile_options(strix-n1-gfx1151-cuda-fixture PRIVATE
            -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
    endif()
    # Native execution is operator initiated and needs the gfx1151 probe assets.
endif()
