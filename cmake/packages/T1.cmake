if(BUILD_TESTING)
    add_executable(strix-profile-table-tests tests/profile_table_tests.cpp
        src/gemma4/engine/profile.cpp src/cuda/models/glm/producer/profile.cpp
        src/deepseek41_pair_defaults.cpp src/minimax/engine/mm_defaults.cpp)
    target_include_directories(strix-profile-table-tests PRIVATE src apps)
    target_link_libraries(strix-profile-table-tests PRIVATE strix_ds41_rtx_profile
        strix_serve_json strix_device strix_model_registry)
    strix_target_warnings(strix-profile-table-tests)
    # The GLM producer profile table is compiled here only as an identity oracle; its own CUDA build does not enable
    # -Wmissing-field-initializers (designated rows leave trailing fields at their defaults).
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(src/cuda/models/glm/producer/profile.cpp PROPERTIES
            COMPILE_OPTIONS -Wno-missing-field-initializers)
    endif()

    foreach(case qwen_hip qwen_cuda gemma4 ds41_rtx glm_cuda matrix)
        add_test(NAME profile_table_identity_${case}
            COMMAND "${CMAKE_COMMAND}"
                "-DEXECUTABLE=$<TARGET_FILE:strix-profile-table-tests>"
                "-DCASE=${case}"
                "-DGOLDEN=${PROJECT_SOURCE_DIR}/tests/golden/profile_resolution.sha256"
                "-DREGISTRY=${PROJECT_SOURCE_DIR}/models/registry.toml"
                -P "${PROJECT_SOURCE_DIR}/tests/profile_table_identity.cmake")
        set_tests_properties(profile_table_identity_${case} PROPERTIES LABELS "host;T1"
            ENVIRONMENT "STRIX_INTERNAL_MODEL_PRESET=0")
    endforeach()
endif()
