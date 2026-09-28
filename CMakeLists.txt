cmake_minimum_required(VERSION 3.24)

project(strix_engine_lab VERSION 0.1.0 LANGUAGES CXX)

option(STRIX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

add_library(strix_core STATIC
    src/memory_plan.cpp
    src/model_contract.cpp
    src/oracles.cpp
)
target_include_directories(strix_core PUBLIC include)
target_compile_features(strix_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(strix_core PRIVATE /W4)
    if(STRIX_WARNINGS_AS_ERRORS)
        target_compile_options(strix_core PRIVATE /WX)
    endif()
else()
    target_compile_options(strix_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(STRIX_WARNINGS_AS_ERRORS)
        target_compile_options(strix_core PRIVATE -Werror)
    endif()
endif()

add_executable(strix-memory-plan apps/memory_plan_main.cpp)
target_link_libraries(strix-memory-plan PRIVATE strix_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(strix-core-tests tests/core_tests.cpp)
    target_link_libraries(strix-core-tests PRIVATE strix_core)
    add_test(NAME strix-core-tests COMMAND strix-core-tests)
endif()
