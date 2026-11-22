if(BUILD_TESTING)
    add_executable(strix-model-decl-inline-tables-tests tests/model_decl_inline_tables_tests.cpp)
    target_link_libraries(strix-model-decl-inline-tables-tests PRIVATE strix_model_description)
    strix_target_warnings(strix-model-decl-inline-tables-tests)
    add_test(NAME strix-model-decl-inline-tables-tests COMMAND strix-model-decl-inline-tables-tests)

    set_tests_properties(strix-model-decl-inline-tables-tests PROPERTIES LABELS "cpu;plan;D3o" TIMEOUT 30)

    add_executable(strix-indexed-fusion-tests tests/graph/indexed_fusion_tests.cpp)
    target_link_libraries(strix-indexed-fusion-tests PRIVATE strix_stage_graph)
    strix_target_warnings(strix-indexed-fusion-tests)
    add_test(NAME strix-indexed-fusion-tests COMMAND strix-indexed-fusion-tests)
    set_tests_properties(strix-indexed-fusion-tests PROPERTIES LABELS "cpu;plan;D3o" TIMEOUT 30)
endif()
