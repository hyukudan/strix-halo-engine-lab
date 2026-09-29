cmake_minimum_required(VERSION 3.24)

project(strix_transport_lab LANGUAGES CXX)

add_executable(strix-transport-tcp tcp_baseline.cpp)
target_compile_features(strix-transport-tcp PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(strix-transport-tcp PRIVATE /W4 /WX)
else()
    target_compile_options(
        strix-transport-tcp PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
    )
endif()
