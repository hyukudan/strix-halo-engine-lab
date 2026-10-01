cmake_minimum_required(VERSION 3.24)

project(strix_uma_residency_lab LANGUAGES CXX)

option(STRIX_UMA_ENABLE_HIP "Build the HIP residency probe" OFF)

function(strix_lab_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
    endif()
endfunction()

add_executable(strix-uma-host host_residency.cpp)
target_compile_features(strix-uma-host PRIVATE cxx_std_20)
strix_lab_warnings(strix-uma-host)

if(STRIX_UMA_ENABLE_HIP)
    enable_language(HIP)
    add_executable(strix-uma-hip hip_residency.hip)
    set_target_properties(strix-uma-hip PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
        HIP_ARCHITECTURES gfx1151
    )
endif()
