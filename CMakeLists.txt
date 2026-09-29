cmake_minimum_required(VERSION 3.24)

project(strix_engine_lab VERSION 0.1.0 LANGUAGES CXX)

option(STRIX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

function(strix_target_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(STRIX_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(STRIX_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(strix_core STATIC
    src/gguf_reader.cpp
    src/memory_plan.cpp
    src/model_contract.cpp
    src/model_manifest.cpp
    src/oracles.cpp
    src/platform_probe.cpp
)
target_include_directories(strix_core PUBLIC include)
target_compile_features(strix_core PUBLIC cxx_std_20)
strix_target_warnings(strix_core)

add_executable(strix-memory-plan apps/memory_plan_main.cpp)
target_link_libraries(strix-memory-plan PRIVATE strix_core)
strix_target_warnings(strix-memory-plan)

add_executable(strix-inspect-gguf apps/inspect_gguf_main.cpp)
target_link_libraries(strix-inspect-gguf PRIVATE strix_core)
strix_target_warnings(strix-inspect-gguf)

add_executable(strix-model-manifest apps/model_manifest_main.cpp)
target_link_libraries(strix-model-manifest PRIVATE strix_core)
strix_target_warnings(strix-model-manifest)

add_executable(strix-platform-probe apps/platform_probe_main.cpp)
target_link_libraries(strix-platform-probe PRIVATE strix_core)
strix_target_warnings(strix-platform-probe)

include(CTest)
if(BUILD_TESTING)
    add_executable(strix-core-tests tests/core_tests.cpp)
    target_link_libraries(strix-core-tests PRIVATE strix_core)
    strix_target_warnings(strix-core-tests)
    add_test(NAME strix-core-tests COMMAND strix-core-tests)

    add_executable(strix-manifest-platform-tests tests/manifest_platform_tests.cpp)
    target_link_libraries(strix-manifest-platform-tests PRIVATE strix_core)
    strix_target_warnings(strix-manifest-platform-tests)
    add_test(NAME strix-manifest-platform-tests COMMAND strix-manifest-platform-tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME strix-python-oracle-tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/oracles/test_qwen4_exp_oracle.py
        )
    endif()
endif()
