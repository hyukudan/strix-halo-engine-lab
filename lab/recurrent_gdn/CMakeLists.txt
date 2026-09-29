cmake_minimum_required(VERSION 3.24)

project(strix_recurrent_gdn_lab LANGUAGES CXX HIP)

find_package(hip REQUIRED CONFIG)

add_executable(strix-recurrent-gdn recurrent_gdn.hip)
set_target_properties(strix-recurrent-gdn PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES gfx1151
)
target_link_libraries(strix-recurrent-gdn PRIVATE hip::host)

if(MSVC)
    target_compile_options(strix-recurrent-gdn PRIVATE /W4)
else()
    target_compile_options(
        strix-recurrent-gdn PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()
