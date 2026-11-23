if(BUILD_TESTING)
    add_executable(strix-k9g-decode-plan-test tests/cuda_core/recur_decode_plan_test.cpp)
    target_include_directories(strix-k9g-decode-plan-test PRIVATE include src)
    target_compile_features(strix-k9g-decode-plan-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-k9g-decode-plan-test)
    add_test(NAME strix-k9g-decode-plan-test COMMAND strix-k9g-decode-plan-test)
    set_tests_properties(strix-k9g-decode-plan-test PROPERTIES LABELS "cpu;K9G")
endif()

option(STRIX_K9G_CUDA_FIXTURE "Compile the one-row gfx1151 GDN comparison for sm_120a" OFF)
if(STRIX_K9G_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-k9g-decode-fixture
        tests/cuda_core/recur_decode_fixture.cu
        src/cuda/core/recur_decode.cu
        src/cuda/core/amd_math_tables.cu)
    target_include_directories(strix-k9g-decode-fixture PRIVATE include src)
    set_target_properties(strix-k9g-decode-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real")
    target_compile_options(strix-k9g-decode-fixture PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
    if(MSVC)
        target_compile_options(strix-k9g-decode-fixture PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
    endif()
    # The operator runs this fixture with TABLES_DIR. CTest runs only the host test.
endif()
