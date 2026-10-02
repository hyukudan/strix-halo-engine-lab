cmake_minimum_required(VERSION 3.24)

project(strix_moe_q8_0_lab LANGUAGES CXX HIP)

find_package(hip REQUIRED CONFIG)

add_executable(strix-moe-q8-0 moe_q8_0.hip)
set_target_properties(strix-moe-q8-0 PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES gfx1151
)
target_link_libraries(strix-moe-q8-0 PRIVATE hip::host)

if(MSVC)
    target_compile_options(strix-moe-q8-0 PRIVATE /W4 /WX)
else()
    target_compile_options(
        strix-moe-q8-0 PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
    )
endif()
