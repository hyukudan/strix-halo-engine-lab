cmake_minimum_required(VERSION 3.24)

project(strix_packed_projection_lab LANGUAGES CXX HIP)

find_package(hip REQUIRED CONFIG)

function(configure_projection_target target wave_size)
    add_executable(${target} q8_projection.hip)
    set_target_properties(${target} PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
        HIP_ARCHITECTURES gfx1151
    )
    target_compile_definitions(${target} PRIVATE STRIX_WAVE_SIZE=${wave_size})
    target_compile_options(
        ${target} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wall>
        $<$<COMPILE_LANGUAGE:HIP>:-Wextra>
    )
    target_link_libraries(${target} PRIVATE hip::host)
endfunction()

configure_projection_target(strix-q8-projection-w32 32)

configure_projection_target(strix-q8-projection-w64 64)
target_compile_options(
    strix-q8-projection-w64 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize64>
)
