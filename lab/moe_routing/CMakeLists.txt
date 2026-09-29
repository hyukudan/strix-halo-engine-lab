cmake_minimum_required(VERSION 3.24)

project(strix_moe_routing_lab LANGUAGES CXX HIP)

find_package(hip REQUIRED CONFIG)

add_executable(strix-moe-routing moe_routing.hip)
set_target_properties(strix-moe-routing PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES gfx1151
)
target_link_libraries(strix-moe-routing PRIVATE hip::host)

if(MSVC)
    target_compile_options(strix-moe-routing PRIVATE /W4)
else()
    target_compile_options(
        strix-moe-routing PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()
