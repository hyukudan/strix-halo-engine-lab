if(BUILD_TESTING)
    add_executable(strix-wr2-host-tests tests/cuda_core/wr2_host_tests.cpp)
    target_include_directories(strix-wr2-host-tests PRIVATE src)
    target_link_libraries(strix-wr2-host-tests PRIVATE strix_exec)
    strix_target_warnings(strix-wr2-host-tests)
    add_test(NAME wr2_host COMMAND strix-wr2-host-tests)
    set_tests_properties(wr2_host PROPERTIES LABELS "cpu;plan" TIMEOUT 30)
endif()

if(TARGET strix_exec_cuda_decode_catalog)
    add_library(strix_exec_cuda_wr2_packed OBJECT src/exec/cuda_decode_wr2_packed.cpp)
    target_include_directories(strix_exec_cuda_wr2_packed PRIVATE include src)
    target_compile_features(strix_exec_cuda_wr2_packed PRIVATE cxx_std_20)
    strix_target_warnings(strix_exec_cuda_wr2_packed)
    target_sources(strix_exec_cuda_decode_catalog PRIVATE $<TARGET_OBJECTS:strix_exec_cuda_wr2_packed>)
endif()
if(TARGET strix_cuda_core_kernels)
    target_sources(strix_cuda_core_kernels PRIVATE src/cuda/core/moe_microscaled_step.cu)
endif()

option(STRIX_WR2_CUDA_FIXTURE "Build the WR2 CUDA decode leaf conformance fixture" OFF)
if(STRIX_WR2_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-wr2-cuda-fixture tests/cuda_core/wr2_fixture.cu
        src/cuda/core/attn_production.cu src/cuda/core/attn_flash.cu
        src/cuda/core/moe_microscaled_step.cu src/exec/cuda_decode_wr2_packed.cpp)
    target_include_directories(strix-wr2-cuda-fixture PRIVATE include src)
    set_target_properties(strix-wr2-cuda-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 120a-real)
    target_compile_features(strix-wr2-cuda-fixture PRIVATE cxx_std_20)
    target_compile_options(strix-wr2-cuda-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        set_target_properties(strix-wr2-cuda-fixture PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_options(strix-wr2-cuda-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/WX,/wd4127,/EHsc,/Zc:preprocessor,/utf-8>)
    else()
        target_compile_options(strix-wr2-cuda-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow;-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-Wconversion,-Wshadow,-Werror>)
    endif()
    # Device execution is explicit and is never part of the CPU suite.
endif()
