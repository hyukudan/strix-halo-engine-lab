# The shared sampling-v1 host library: policy, reference transform, path choice and the counter RNG. strix_core,
# strix_spec and the standalone CUDA backends link this one target.
if(NOT TARGET strix_token_sampling)
    find_package(Threads REQUIRED)
    add_library(strix_token_sampling STATIC
        "${CMAKE_CURRENT_LIST_DIR}/../src/token_sampling.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/token_sampling_device.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/speculation_rng.cpp")
    target_include_directories(strix_token_sampling PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../include")
    target_compile_features(strix_token_sampling PUBLIC cxx_std_20)
    target_link_libraries(strix_token_sampling PUBLIC Threads::Threads)
    if(COMMAND strix_apply_target_warnings)
        strix_apply_target_warnings(strix_token_sampling)
    elseif(COMMAND strix_target_warnings)
        strix_target_warnings(strix_token_sampling)
    endif()
endif()
