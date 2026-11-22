if(BUILD_TESTING)
    add_executable(strix-k4c-attn-plan-tests tests/cuda_core/attn_plan_tests.cpp)
    target_include_directories(strix-k4c-attn-plan-tests PRIVATE src)
    target_compile_features(strix-k4c-attn-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-k4c-attn-plan-tests)
    add_test(NAME strix-k4c-attn-plan-tests COMMAND strix-k4c-attn-plan-tests)
    set_tests_properties(strix-k4c-attn-plan-tests PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_K4C_CUDA_FIXTURE "Compile the K4c native conformance fixture for sm_120a" OFF)
if(STRIX_K4C_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-k4c-attn-fixture
        tests/cuda_core/attn_fixture.cu
        src/cuda/core/attn_decode.cu
        src/cuda/core/attn_production.cu
        src/cuda/core/attn_flash.cu
        src/cuda/core/attn_flash8.cu)
    target_include_directories(strix-k4c-attn-fixture PRIVATE include src)
    target_compile_definitions(strix-k4c-attn-fixture PRIVATE STRIX_CUDA_FAST8_SM120A=1)
    set_target_properties(strix-k4c-attn-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strix-k4c-attn-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-k4c-attn-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    # Lane-batched prepared attention identity (bitwise vs per-lane launches), production FP flags.
    add_executable(strix-attn-lanes-fixture
        tests/cuda_core/attn_lanes_fixture.cu
        src/cuda/core/attn_decode.cu
        src/cuda/core/attn_production.cu
        src/cuda/core/attn_flash.cu
        src/cuda/core/attn_flash8.cu)
    target_include_directories(strix-attn-lanes-fixture PRIVATE include src)
    target_compile_definitions(strix-attn-lanes-fixture PRIVATE STRIX_CUDA_FAST8_SM120A=1)
    set_target_properties(strix-attn-lanes-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    if(MSVC)
        target_compile_options(strix-attn-lanes-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    # Run this fixture only by explicit operator invocation.
endif()
