if(BUILD_TESTING)
    add_executable(strix-k5a-select-rows-tests tests/cuda_core/select_rows_tests.cpp)
    target_include_directories(strix-k5a-select-rows-tests PRIVATE src include)
    target_compile_features(strix-k5a-select-rows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-k5a-select-rows-tests)
    add_test(NAME strix-k5a-select-rows-tests COMMAND strix-k5a-select-rows-tests)
    set_tests_properties(strix-k5a-select-rows-tests PROPERTIES LABELS "cpu" TIMEOUT 30)
endif()

option(STRIX_K5A_CUDA_FIXTURE "Build the K5A selection row producer fixture for sm_120a" OFF)
if(STRIX_K5A_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-k5a-select-fixture tests/cuda_core/select_indexed_fixture.cu)
    target_include_directories(strix-k5a-select-fixture PRIVATE src include)
    set_target_properties(strix-k5a-select-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strix-k5a-select-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false;--Werror=all-warnings>)
    if(MSVC)
        target_compile_options(strix-k5a-select-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8,/W4,/WX>)
    endif()
    # The fixture runs only by explicit operator invocation.
endif()
