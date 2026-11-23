if(BUILD_TESTING)
    add_executable(strix-qsa-launch-plan-tests tests/cuda_core/attn_rowgroup_plan_tests.cpp)
    target_include_directories(strix-qsa-launch-plan-tests PRIVATE src)
    target_compile_features(strix-qsa-launch-plan-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(strix-qsa-launch-plan-tests PRIVATE /W4 /WX)
    else()
        target_compile_options(strix-qsa-launch-plan-tests PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
    endif()
    add_test(NAME strix-qsa-launch-plan-tests COMMAND strix-qsa-launch-plan-tests)
    set_tests_properties(strix-qsa-launch-plan-tests PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_QSA_CUDA_FIXTURE "Compile the QSA launch identity fixture for sm_120a" OFF)
if(STRIX_QSA_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-qsa-launch-fixture tests/cuda_core/attn_rowgroup_fixture.cu)
    target_include_directories(strix-qsa-launch-fixture PRIVATE include src)
    set_target_properties(strix-qsa-launch-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strix-qsa-launch-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-qsa-launch-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    # Device execution is an explicit operator step, outside CTest.
endif()
