# Shared CUDA core (model-agnostic kernels: dense rows, norms, rope, decode attention, GDN decode).
# Usable from the full build (-DSTRIX_CUDA_CORE=ON) and standalone (-DSTRIX_CUDA_CORE_ONLY=ON, CUDA language only).
if(TARGET strix_cuda_core)
    return()
endif()
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()
get_filename_component(_cc_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(strix_cuda_core STATIC
    ${_cc_root}/src/cuda_core/dense/dense.cu
    ${_cc_root}/src/cuda_core/norms/norms.cu
    ${_cc_root}/src/cuda_core/norms/rope.cu
    ${_cc_root}/src/cuda_core/attention/attention.cu
    ${_cc_root}/src/cuda_core/gdn/gdn.cu)
target_include_directories(strix_cuda_core PUBLIC ${_cc_root}/include)
# Identity: no FMA contraction, IEEE division and sqrt, no flush-to-zero (as the existing Qwen CUDA build).
target_compile_options(strix_cuda_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_cuda_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/StrixDevice.cmake)
add_executable(strix-cuda-qwen35-decode ${_cc_root}/apps/cuda_qwen35_decode_main.cu ${_cc_root}/src/gguf_reader.cpp)
target_link_libraries(strix-cuda-qwen35-decode PRIVATE strix_cuda_core strix_device)
target_include_directories(strix-cuda-qwen35-decode PRIVATE ${_cc_root}/include)
target_compile_options(strix-cuda-qwen35-decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix-cuda-qwen35-decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

add_executable(cuda_core_test ${_cc_root}/tests/cuda_core/cuda_core_test.cu)
target_link_libraries(cuda_core_test PRIVATE strix_cuda_core)
target_compile_options(cuda_core_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
