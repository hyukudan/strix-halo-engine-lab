cmake_minimum_required(VERSION 3.21)
project(strix_dense_a8_target_host_test LANGUAGES CXX)

set(STRIX_HIP_ARCHS "gfx1151;gfx1200;gfx942;gfx950")
set(CMAKE_HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
add_library(strix_hip_runtime INTERFACE)
function(strix_target_warnings target)
endfunction()

include(../../../cmake/StrixDenseA8WmmaTest.cmake)
get_target_property(archs strix-hip-dense-a8-wmma-tests HIP_ARCHITECTURES)
if(NOT archs STREQUAL "gfx1151")
    message(FATAL_ERROR "Dense A8 WMMA test must target gfx1151 alone, got '${archs}'")
endif()

# This fixture only configures the target. The HIP language stays disabled.
set_target_properties(strix-hip-dense-a8-wmma-tests PROPERTIES LINKER_LANGUAGE CXX)
