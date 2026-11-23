if(BUILD_TESTING)
    add_executable(strix-prepared-gqa-round-fold-test tests/attn_core/prepared_gqa_round_fold_test.cpp)
    target_include_directories(strix-prepared-gqa-round-fold-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(strix-prepared-gqa-round-fold-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-prepared-gqa-round-fold-test)
    add_test(NAME strix-prepared-gqa-round-fold-test COMMAND strix-prepared-gqa-round-fold-test)
endif()
