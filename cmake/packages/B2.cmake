include(${CMAKE_CURRENT_LIST_DIR}/../CoreRowsPlan.cmake)
if(BUILD_TESTING)
    add_executable(strix-b2-rows-tests tests/rows/b2_rows_tests.cpp)
    target_link_libraries(strix-b2-rows-tests PRIVATE strix_core_rows_plan)
    target_include_directories(strix-b2-rows-tests PRIVATE include src)
    target_compile_features(strix-b2-rows-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-b2-rows-tests)
    add_test(NAME strix-b2-rows-tests COMMAND strix-b2-rows-tests)
    set_tests_properties(strix-b2-rows-tests PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_B2_CUDA_FIXTURE "Build the B2 native conformance fixture for sm_120a" OFF)
if(STRIX_B2_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-b2-rows-fixture EXCLUDE_FROM_ALL
        tests/rows/b2_rows_fixture.cu
        src/core/rows_plan.cpp  # standalone fixture: own static CRT, links neither strix_core_rows_plan nor the CUDA core
        src/cuda/core/dense/dense_mma.cu
        src/cuda/core/dense/nvfp4.cu)
    target_include_directories(strix-b2-rows-fixture PRIVATE include src)
    target_compile_features(strix-b2-rows-fixture PRIVATE cxx_std_20)
    set_target_properties(strix-b2-rows-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_RUNTIME_LIBRARY Static)
    target_compile_options(strix-b2-rows-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        set_property(TARGET strix-b2-rows-fixture PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_options(strix-b2-rows-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    # Native execution is by explicit operator invocation only.
endif()
