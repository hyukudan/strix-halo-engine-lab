cmake_minimum_required(VERSION 3.24)

project(strix_sparse_lookup_lab LANGUAGES CXX)

add_executable(strix-sparse-lookup sparse_lookup.cpp)
target_compile_features(strix-sparse-lookup PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(strix-sparse-lookup PRIVATE /W4 /WX)
else()
    target_compile_options(
        strix-sparse-lookup PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
    )
endif()
