cmake_minimum_required(VERSION 3.24)

project(strix_moe_iq4_xs_lab LANGUAGES CXX HIP)

find_package(hip REQUIRED CONFIG)

add_executable(strix-moe-iq4-xs moe_iq4_xs.hip)
set_target_properties(strix-moe-iq4-xs PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES gfx1151
)
target_link_libraries(strix-moe-iq4-xs PRIVATE hip::host)

if(MSVC)
    target_compile_options(strix-moe-iq4-xs PRIVATE /W4)
else()
    target_compile_options(
        strix-moe-iq4-xs PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()
