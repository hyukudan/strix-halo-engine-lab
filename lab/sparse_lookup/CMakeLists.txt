cmake_minimum_required(VERSION 3.24)

project(strix_sparse_lookup_lab LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
    strix-sparse-lookup
    sparse_lookup.cpp
    ../../src/sparse_lookup.cpp
    ../../src/sparse_lookup_executor.cpp
)
target_include_directories(
    strix-sparse-lookup
    PRIVATE ../../include
)
target_compile_features(strix-sparse-lookup PRIVATE cxx_std_20)
target_link_libraries(strix-sparse-lookup PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(strix-sparse-lookup PRIVATE /W4 /WX)
else()
    target_compile_options(
        strix-sparse-lookup PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
    )
endif()
