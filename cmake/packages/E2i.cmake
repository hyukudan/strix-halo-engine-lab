if(BUILD_TESTING)
    add_executable(strix-e2i-step-inputs-tests tests/cuda_core/step_inputs_tests.cpp)
    target_include_directories(strix-e2i-step-inputs-tests PRIVATE include src)
    target_compile_features(strix-e2i-step-inputs-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-e2i-step-inputs-tests)
    add_test(NAME strix-e2i-step-inputs-tests COMMAND strix-e2i-step-inputs-tests)
    set_tests_properties(strix-e2i-step-inputs-tests PROPERTIES LABELS "cpu")
endif()

option(STRIX_E2I_CUDA_FIXTURE "Build the E2i CUDA conformance fixture for sm_120a" OFF)
if(STRIX_E2I_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-e2i-step-inputs-fixture tests/cuda_core/step_inputs_fixture.cu src/cuda/core/step_inputs.cu)
    target_include_directories(strix-e2i-step-inputs-fixture PRIVATE include src)
    set_target_properties(strix-e2i-step-inputs-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strix-e2i-step-inputs-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-e2i-step-inputs-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    # Native execution is operator-initiated. The fixture is not registered with CTest.
endif()
