# Block drafting (strix_spec + strix_lookup + host tests + the qwen35 speculative decode app) for the CUDA-only build.
# Included from CudaCore.cmake; the full build defines strix_spec in CMakeLists.txt.
if(NOT TARGET strix_spec)
    add_library(strix_lookup STATIC ${_cc_root}/src/lookup/index.cpp)
    target_include_directories(strix_lookup PUBLIC ${_cc_root}/include)
    target_compile_features(strix_lookup PUBLIC cxx_std_20)
    add_library(strix_spec STATIC ${_cc_root}/src/spec/drafting.cpp ${_cc_root}/src/spec/block_drafter.cpp)
    target_link_libraries(strix_spec PUBLIC strix_lookup)
endif()
add_executable(strix-spec-block-drafter-tests ${_cc_root}/tests/spec_block_drafter_tests.cpp)
target_link_libraries(strix-spec-block-drafter-tests PRIVATE strix_spec)
add_executable(strix-spec-drafting-tests ${_cc_root}/tests/spec_drafting_tests.cpp)
target_link_libraries(strix-spec-drafting-tests PRIVATE strix_spec)
