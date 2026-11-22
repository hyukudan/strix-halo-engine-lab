# K11c host checks and opt-in, compile-only-by-default native fixture.
if(BUILD_TESTING)
    add_executable(strix-k11c-head-tests tests/cuda_core/head_emit_tests.cpp src/logprob_partials.cpp)
    target_include_directories(strix-k11c-head-tests PRIVATE include src)
    target_compile_features(strix-k11c-head-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-k11c-head-tests)
    add_test(NAME strix-k11c-head-tests COMMAND strix-k11c-head-tests)
endif()

option(STRIX_K11C_CUDA_FIXTURE "Compile the K11c native conformance fixture for sm_120a" OFF)
if(STRIX_K11C_CUDA_FIXTURE)
    enable_language(CUDA)

    # Compile the unchanged private Markov head kernels as a fixture oracle, without linking the engine.
    set(k11c_markov_source "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/models/ds41/ds41_dspark.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${k11c_markov_source}")
    file(READ "${k11c_markov_source}" k11c_markov_text)
    function(strix_k11c_incumbent_slice first last out)
        string(FIND "${k11c_markov_text}" "${first}" begin)
        string(FIND "${k11c_markov_text}" "${last}" end)
        if(begin LESS 0 OR end LESS_EQUAL begin)
            message(FATAL_ERROR "K11c fixture: ds41_dspark.cu oracle boundary changed at ${first}")
        endif()
        math(EXPR length "${end} - ${begin}")
        string(SUBSTRING "${k11c_markov_text}" ${begin} ${length} body)
        set(${out} "${body}" PARENT_SCOPE)
    endfunction()
    strix_k11c_incumbent_slice("__device__ __forceinline__ float f16_device" "__device__ __forceinline__ float entry_value_device" k11c_markov_math)
    strix_k11c_incumbent_slice("struct DraftParams {" "// Decodes the keys of a block attention" k11c_markov_params)
    strix_k11c_incumbent_slice("struct HeadView {" "// Gathers Q8_0 head rows" k11c_markov_pick)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/k11c_markov_incumbent.cuh"
        CONTENT "${k11c_markov_math}\n${k11c_markov_params}\n${k11c_markov_pick}")

    add_executable(strix-k11c-head-fixture tests/cuda_core/head_emit_fixture.cu
        src/cuda/logprob_partials.cu)
    target_include_directories(strix-k11c-head-fixture PRIVATE include src "${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties(strix-k11c-head-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strix-k11c-head-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-k11c-head-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
    # Explicit operator execution only. This device fixture is never registered with CTest.
endif()
