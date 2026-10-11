cmake_minimum_required(VERSION 3.24)

project(strix_packed_projection_lab LANGUAGES CXX HIP)

add_library(projection_arithmetic STATIC ../../src/projection_reduction.cpp)
target_include_directories(projection_arithmetic PUBLIC ../../include)
target_compile_features(projection_arithmetic PUBLIC cxx_std_20)
target_compile_options(projection_arithmetic PRIVATE -Wall -Wextra -Werror -ffp-contract=off)


function(configure_projection_target target source wave_size)
    add_executable(${target} ${source})
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
        $<$<COMPILE_LANGUAGE:HIP>:-Werror>
    )
endfunction()

configure_projection_target(
    strix-q8-projection-w32 q8_projection.hip 32)
configure_projection_target(
    strix-q8-projection-w64 q8_projection.hip 64)
target_compile_options(
    strix-q8-projection-w64 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize64>
)

configure_projection_target(
    strix-iq4-xs-projection-w32 iq4_xs_projection.hip 32)
configure_projection_target(
    strix-iq4-xs-projection-w64 iq4_xs_projection.hip 64)
foreach(target strix-iq4-xs-projection-w32 strix-iq4-xs-projection-w64)
    target_link_libraries(${target} PRIVATE projection_arithmetic)
    target_compile_options(${target} PRIVATE -ffp-contract=off)
endforeach()
target_compile_options(
    strix-iq4-xs-projection-w64 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize64>
)
