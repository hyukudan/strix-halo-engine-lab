cmake_minimum_required(VERSION 3.24)
project(strix_low_precision_reference LANGUAGES CXX)

add_executable(strix-low-precision-reference
    reference_main.cpp
    ../../src/low_precision.cpp
    ../../src/iq4_xs.cpp
)
target_include_directories(strix-low-precision-reference PRIVATE ../../include)
target_compile_features(strix-low-precision-reference PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(strix-low-precision-reference PRIVATE /W4 /WX /fp:strict)
else()
    target_compile_options(strix-low-precision-reference PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -ffp-contract=off)
endif()
