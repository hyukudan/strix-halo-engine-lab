cmake_minimum_required(VERSION 3.24)

project(strix_basic_kernel_lab LANGUAGES CXX HIP)

find_package(hip REQUIRED CONFIG)

function(add_strix_basic_kernel_target target wave_size)
    add_executable(${target} basic_kernels.hip)
    set_target_properties(${target} PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
        HIP_ARCHITECTURES gfx1151
    )
    target_compile_definitions(${target} PRIVATE STRIX_WAVE_SIZE=${wave_size})
    target_compile_options(
        ${target} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize${wave_size}>
        $<$<COMPILE_LANGUAGE:HIP>:-Wall>
        $<$<COMPILE_LANGUAGE:HIP>:-Wextra>
    )
    target_link_libraries(${target} PRIVATE hip::host)
endfunction()

add_strix_basic_kernel_target(strix-basic-kernels-w32 32)
add_strix_basic_kernel_target(strix-basic-kernels-w64 64)
