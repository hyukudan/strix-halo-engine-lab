cmake_minimum_required(VERSION 3.24)
project(mfile_contract_stub LANGUAGES CXX)
if(NOT STRIX_TIERSTREAM_DIR)
    message(FATAL_ERROR "STRIX_TIERSTREAM_DIR must name the API skeleton source tree")
endif()
# An isolated test double, never part of the production build. Reuse the API skeleton's hash/JSON helpers.
add_library(tierstream STATIC mfile_stub.cpp
    "${STRIX_TIERSTREAM_DIR}/src/json.cpp"
    "${STRIX_TIERSTREAM_DIR}/src/raw_file.cpp"
    "${STRIX_TIERSTREAM_DIR}/src/xxh64.cpp"
    "${STRIX_TIERSTREAM_DIR}/src/sha256.cpp")
target_compile_features(tierstream PRIVATE cxx_std_20)
target_include_directories(tierstream PRIVATE "${STRIX_TIERSTREAM_DIR}/include" "${STRIX_TIERSTREAM_DIR}/src")
if(WIN32)
    target_link_libraries(tierstream PRIVATE bcrypt)
    target_compile_definitions(tierstream PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()
