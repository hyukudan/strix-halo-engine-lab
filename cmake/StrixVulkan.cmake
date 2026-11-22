option(STRIX_BACKEND_VULKAN "Build the Vulkan reference scaffold" OFF)
include("${CMAKE_CURRENT_LIST_DIR}/StrixGpuPort.cmake")
if(NOT STRIX_BACKEND_VULKAN)
    return()
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The Vulkan reference scaffold currently targets Linux")
endif()

find_path(STRIX_VULKAN_HEADERS NAMES vulkan/vulkan.h
    HINTS "$ENV{VULKAN_SDK}" PATH_SUFFIXES include REQUIRED)
find_program(STRIX_VULKAN_GLSL_COMPILER NAMES glslangValidator glslc
    HINTS "$ENV{VULKAN_SDK}" PATH_SUFFIXES bin)
set(_vulkan_shader "${PROJECT_SOURCE_DIR}/src/vulkan/core/rms_norm.comp")
set(_vulkan_header "${CMAKE_CURRENT_BINARY_DIR}/strix_vulkan_rms_shader.hpp")
set(_vulkan_embed "${CMAKE_CURRENT_LIST_DIR}/StrixGpuPortShader.cmake")
add_library(strix_vulkan_core STATIC src/vulkan/core/backend.cpp
    src/vulkan/core/backend_vulkan.cpp "${_vulkan_header}")
target_link_libraries(strix_vulkan_core PUBLIC strix_gpu_port PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(strix_vulkan_core SYSTEM PRIVATE "${STRIX_VULKAN_HEADERS}")
target_include_directories(strix_vulkan_core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(strix_vulkan_core PRIVATE STRIX_HAVE_VULKAN=1)
strix_target_warnings(strix_vulkan_core)

if(STRIX_VULKAN_GLSL_COMPILER)
    get_filename_component(_vulkan_compiler "${STRIX_VULKAN_GLSL_COMPILER}" NAME_WE)
    set(_vulkan_spirv "${CMAKE_CURRENT_BINARY_DIR}/strix_vulkan_rms.spv")
    if(_vulkan_compiler STREQUAL "glslc")
        set(_vulkan_flags --target-env=vulkan1.1)
    else()
        set(_vulkan_flags -V --target-env vulkan1.1)
    endif()
    add_custom_command(OUTPUT "${_vulkan_spirv}"
        COMMAND "${STRIX_VULKAN_GLSL_COMPILER}" ${_vulkan_flags} -o "${_vulkan_spirv}" "${_vulkan_shader}"
        DEPENDS "${_vulkan_shader}" VERBATIM)
    set(_vulkan_mode spirv)
    set(_vulkan_input "${_vulkan_spirv}")
    target_compile_definitions(strix_vulkan_core PRIVATE STRIX_VULKAN_HAS_SPIRV=1)
else()
    message(STATUS "Vulkan GLSL compiler unavailable: embedding checked source; RMSNorm dispatch will refuse")
    set(_vulkan_mode glsl)
    set(_vulkan_input "${_vulkan_shader}")
endif()
add_custom_command(OUTPUT "${_vulkan_header}"
    COMMAND "${CMAKE_COMMAND}" "-DMODE=${_vulkan_mode}" "-DINPUT=${_vulkan_input}"
        "-DOUTPUT=${_vulkan_header}" -P "${_vulkan_embed}"
    DEPENDS "${_vulkan_input}" "${_vulkan_embed}" VERBATIM)
