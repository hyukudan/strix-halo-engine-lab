if(TARGET strix_stage_graph AND TARGET strix_model_description)
    add_library(strix_model_decl STATIC src/graph/model_decl.cpp)
    target_link_libraries(strix_model_decl PUBLIC strix_stage_graph strix_model_description)
    strix_target_warnings(strix_model_decl)

    if(BUILD_TESTING)
        add_executable(strix-model-decl-tests tests/model_decl_tests.cpp)
        target_include_directories(strix-model-decl-tests PRIVATE src)
        target_link_libraries(strix-model-decl-tests PRIVATE strix_model_decl strix_model_graph_stage)
        strix_target_warnings(strix-model-decl-tests)
        add_test(NAME strix-model-decl-tests COMMAND strix-model-decl-tests
            "${CMAKE_CURRENT_SOURCE_DIR}/data/graph/templates"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/model_decl")
        set_tests_properties(strix-model-decl-tests PROPERTIES LABELS "cpu;plan;D1" TIMEOUT 30)
    endif()
endif()
