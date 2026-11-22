# Configure-only check: both CUDA core layers coexist in one build tree.
#   cmake/StrixCudaBackend.cmake : INTERFACE strix_cuda_core (src/cuda/core headers; CUDA Qwen MoE / DS41 libraries)
#   cmake/CudaCore.cmake         : STATIC strix_cuda_core_kernels (src/cuda/core) + the qwen35 CUDA engine and CLI
# The full build includes them in this order when STRIX_ENABLE_CUDA_DS41 and STRIX_CUDA_CORE are both ON
# (CMakeLists.txt StrixCudaBackend before CudaCore). Fails if any target of the second layer is missing.
# Run: cmake -S tests/cmake/cuda_core_coexist -B <build dir>   (needs a CUDA toolchain; nothing is compiled)
cmake_minimum_required(VERSION 3.24)
project(cuda_core_coexist LANGUAGES CXX)
get_filename_component(_coexist_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include(${_coexist_root}/cmake/StrixCudaBackend.cmake)
include(${_coexist_root}/cmake/CudaCore.cmake)
include(${_coexist_root}/cmake/CudaCore.cmake)  # a second include is a no-op (include_guard)
foreach(t strix_qwen35_cuda strix-cuda-qwen35-decode strix_cuda_core strix_cuda_core_kernels)
    if(NOT TARGET ${t})
        message(FATAL_ERROR "cuda_core_coexist: FAIL, target ${t} missing")
    endif()
endforeach()
get_target_property(_iface strix_cuda_core TYPE)
get_target_property(_static strix_cuda_core_kernels TYPE)
get_target_property(_links strix_qwen35_cuda LINK_LIBRARIES)
if(NOT _iface STREQUAL "INTERFACE_LIBRARY" OR NOT _static STREQUAL "STATIC_LIBRARY" OR NOT "strix_cuda_core_kernels" IN_LIST _links)
    message(FATAL_ERROR "cuda_core_coexist: FAIL, types ${_iface} / ${_static}, qwen35 links ${_links}")
endif()
message(STATUS "cuda_core_coexist: PASS (strix_cuda_core ${_iface}, strix_cuda_core_kernels ${_static})")
