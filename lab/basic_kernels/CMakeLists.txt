cmake_minimum_required(VERSION 3.24)

project(strix_basic_kernel_lab LANGUAGES CXX HIP)


function(configure_strix_basic_kernel_target target wave_size)
    add_executable(${target} basic_kernels.hip)
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
endfunction()

configure_strix_basic_kernel_target(strix-basic-kernels-w32 32)

configure_strix_basic_kernel_target(strix-basic-kernels-w64 64)
target_compile_options(
    strix-basic-kernels-w64 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize64>
)
