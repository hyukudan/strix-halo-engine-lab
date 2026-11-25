if(BUILD_TESTING)
    add_executable(strix-b6-attn-tests tests/cuda_core/attn_position_split_tests.cpp)
    target_include_directories(strix-b6-attn-tests PRIVATE src)
    target_compile_features(strix-b6-attn-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-b6-attn-tests)
    if(MSVC)
        target_compile_options(strix-b6-attn-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-b6-attn-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-b6-attn-tests COMMAND strix-b6-attn-tests)
    set_tests_properties(strix-b6-attn-tests PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_B6_CUDA_FIXTURE "Compile the B6 position-split fixture for sm_120a" OFF)
if(STRIX_B6_CUDA_FIXTURE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(strix-b6-attn-fixture
        tests/cuda_core/attn_position_split_fixture.cu
        src/cuda/core/attn_decode.cu
        src/cuda/core/attn_production.cu
        src/cuda/core/attn_flash.cu
        src/cuda/core/attn_flash8.cu)
    target_include_directories(strix-b6-attn-fixture PRIVATE include src)
    target_link_libraries(strix-b6-attn-fixture PRIVATE CUDA::cudart)
    target_compile_definitions(strix-b6-attn-fixture PRIVATE STRIX_CUDA_FAST8_SM120A=1)
    set_target_properties(strix-b6-attn-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY None)
    target_compile_options(strix-b6-attn-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-b6-attn-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8,/fp:strict>)
    else()
        target_compile_options(strix-b6-attn-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math,-ffp-contract=off>)
    endif()
    # Native execution is operator-invoked and has no CTest registration.
endif()
