if(BUILD_TESTING)
    add_executable(strix-cuda-chain-key-test tests/cuda_core/chain_key_test.cpp)
    target_include_directories(strix-cuda-chain-key-test PRIVATE src)
    target_link_libraries(strix-cuda-chain-key-test PRIVATE strix_stage_graph)
    strix_target_warnings(strix-cuda-chain-key-test)
    add_test(NAME strix-cuda-chain-key-test COMMAND strix-cuda-chain-key-test)
    set_tests_properties(strix-cuda-chain-key-test PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_K6C_CUDA_FIXTURES "Build the CUDA chain device fixture" OFF)
if(STRIX_K6C_CUDA_FIXTURES)
    enable_language(CUDA)
    add_executable(strix-cuda-chain-fixtures
        tests/cuda_core/chain_fixtures.cu
        src/cuda_core/norms/norms.cu
        src/cuda_core/norms/rope.cu
        src/cuda/models/ds41/ds41_core_norm.cu)
    target_include_directories(strix-cuda-chain-fixtures PRIVATE include src)
    set_target_properties(strix-cuda-chain-fixtures PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "120a-real")
    target_compile_options(strix-cuda-chain-fixtures PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-cuda-chain-fixtures PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endif()
endif()
