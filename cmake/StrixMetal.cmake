option(STRIX_BACKEND_METAL "Build the macOS Metal reference scaffold" OFF)
include("${CMAKE_CURRENT_LIST_DIR}/StrixGpuPort.cmake")
if(NOT STRIX_BACKEND_METAL)
    return()
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "The Metal reference scaffold requires macOS")
endif()

enable_language(OBJCXX)
find_library(STRIX_METAL_FRAMEWORK Metal REQUIRED)
find_library(STRIX_FOUNDATION_FRAMEWORK Foundation REQUIRED)
set(_metal_shader "${PROJECT_SOURCE_DIR}/src/metal/core/rms_norm.metal")
set(_metal_header "${CMAKE_CURRENT_BINARY_DIR}/strix_metal_rms_shader.hpp")
set(_metal_embed "${CMAKE_CURRENT_LIST_DIR}/StrixGpuPortShader.cmake")
add_custom_command(OUTPUT "${_metal_header}"
    COMMAND "${CMAKE_COMMAND}" -DMODE=msl "-DINPUT=${_metal_shader}" "-DOUTPUT=${_metal_header}"
        -P "${_metal_embed}"
    DEPENDS "${_metal_shader}" "${_metal_embed}" VERBATIM)
add_library(strix_metal_core STATIC src/metal/core/backend.cpp src/metal/core/backend_metal.mm "${_metal_header}")
set_target_properties(strix_metal_core PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED ON)
target_link_libraries(strix_metal_core PUBLIC strix_gpu_port
    PRIVATE "${STRIX_METAL_FRAMEWORK}" "${STRIX_FOUNDATION_FRAMEWORK}")
target_include_directories(strix_metal_core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(strix_metal_core PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(strix_metal_core PRIVATE STRIX_HAVE_METAL=1)
strix_target_warnings(strix_metal_core)
