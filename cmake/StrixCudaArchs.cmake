include_guard(GLOBAL)

set(STRIX_CUDA_NVFP4_ARCHITECTURES "120a" CACHE STRING
    "Native NVFP4 images (120a;121a); empty builds the portable CUDA path only")

# Explicit generic architectures can come from CMake or CUDAARCHS. Keep the
# feature images independent: SM120 code is not an SM120a image.
macro(strix_cuda_arch_defaults)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "120a-real")
    endif()
endmacro()

function(strix_cuda_nvfp4_images target real_only)
    set(images)
    foreach(arch IN LISTS STRIX_CUDA_NVFP4_ARCHITECTURES)
        if(NOT arch MATCHES "^(120|121)a$")
            message(FATAL_ERROR
                "STRIX_CUDA_NVFP4_ARCHITECTURES expects native images 120a or 121a; got ${arch}")
        endif()
        target_compile_definitions(${target} PRIVATE STRIX_CUDA_NVFP4_SM${CMAKE_MATCH_1}=1)
        if(real_only)
            list(APPEND images "${arch}-real")
        else()
            list(APPEND images "${arch}")
        endif()
    endforeach()
    if(images)
        set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${images}")
        target_compile_definitions(${target} PRIVATE STRIX_CUDA_NATIVE_NVFP4=1)
    endif()
endfunction()
