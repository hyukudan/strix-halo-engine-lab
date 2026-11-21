if(NOT TARGET strix_token_sampling)
    add_library(strix_token_sampling STATIC
        "${CMAKE_CURRENT_LIST_DIR}/../src/token_sampling.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/speculation_rng.cpp")
    target_include_directories(strix_token_sampling PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../include")
    target_compile_features(strix_token_sampling PUBLIC cxx_std_20)
    if(COMMAND strix_target_warnings)
        strix_target_warnings(strix_token_sampling)
    endif()
endif()
