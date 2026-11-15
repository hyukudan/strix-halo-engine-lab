cmake_minimum_required(VERSION 3.24)
project(strix_qwen27_host LANGUAGES CXX)
include(CTest)
include(../../cmake/StrixDenseModels.cmake)
# Exercise the existing Gemma API after extracting its file/format reader.
add_executable(strix-gemma4-shared-reader-tests ../gemma4_shared_reader_tests.cpp
    ../../src/gemma4/model/safetensors.cpp)
target_link_libraries(strix-gemma4-shared-reader-tests PRIVATE strix_weight_archive)
target_include_directories(strix-gemma4-shared-reader-tests PRIVATE ../../src)
if(MSVC)
    target_compile_options(strix-gemma4-shared-reader-tests PRIVATE /W4 /WX)
endif()
add_test(NAME strix-gemma4-shared-reader-tests COMMAND strix-gemma4-shared-reader-tests)
