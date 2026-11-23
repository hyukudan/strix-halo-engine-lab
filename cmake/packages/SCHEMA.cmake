if(TARGET strix_model_constants)
    target_link_libraries(strix_model_constants PRIVATE strix-weight)

    # Compose model lever catalogues in the generator tool.
    target_include_directories(strix-model-gen PRIVATE src)
    target_link_libraries(strix-model-gen PRIVATE strix_ds41_rtx_profile)

    if(BUILD_TESTING)
        # The drift test validates profile tables with the generator's lever names (tools/model_gen_levers.hpp).
        target_include_directories(strix-model-constants-tests PRIVATE src)
        target_link_libraries(strix-model-constants-tests PRIVATE strix_ds41_rtx_profile)
        add_executable(strix-model-schema-tests tests/model_schema_tests.cpp)
        target_include_directories(strix-model-schema-tests PRIVATE src)
        target_link_libraries(strix-model-schema-tests PRIVATE strix_model_constants)
        target_compile_definitions(strix-model-schema-tests PRIVATE STRIX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        strix_target_warnings(strix-model-schema-tests)
        add_test(NAME strix-model-schema-tests COMMAND strix-model-schema-tests)
        set_tests_properties(strix-model-schema-tests PROPERTIES LABELS "cpu" TIMEOUT 30)
    endif()
endif()
