cmake_minimum_required(VERSION 3.24)
project(strix_integer_projection_lab LANGUAGES CXX)
option(STRIX_INTEGER_PROJECTION_NATIVE "Build opt-in, unqualified gfx1151 HIP experiment" OFF)
add_library(integer_projection_reference STATIC
    ../../src/integer_projection.cpp ../../src/integer_projection_execution.cpp
    ../../src/low_precision.cpp ../../src/iq4_xs.cpp)
target_include_directories(integer_projection_reference PUBLIC ../../include)
target_compile_features(integer_projection_reference PUBLIC cxx_std_20)
target_compile_options(integer_projection_reference PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
add_executable(integer-projection-host-tests ../../tests/integer_projection_tests.cpp)
target_link_libraries(integer-projection-host-tests PRIVATE integer_projection_reference)
target_compile_options(integer-projection-host-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
add_executable(integer-projection-contract-tests ../../tests/integer_projection_execution_tests.cpp)
target_link_libraries(integer-projection-contract-tests PRIVATE integer_projection_reference)
target_compile_options(integer-projection-contract-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
if(STRIX_INTEGER_PROJECTION_NATIVE)
    enable_language(HIP)
    add_library(hip_integer_projection STATIC ../../src/hip_integer_projection.hip)
    target_link_libraries(hip_integer_projection PUBLIC integer_projection_reference)
    set_target_properties(hip_integer_projection PROPERTIES HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
    target_compile_options(hip_integer_projection PRIVATE
        -Wall -Wextra -Werror -ffp-contract=off -mno-wavefrontsize64)
    add_executable(integer-projection-native integer_projection.hip)
    target_link_libraries(integer-projection-native PRIVATE hip_integer_projection)
    set_target_properties(integer-projection-native PROPERTIES HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
    target_compile_options(integer-projection-native PRIVATE
        -Wall -Wextra -Werror -ffp-contract=off -mno-wavefrontsize64)
endif()
