target_sources(strix_stage_graph PRIVATE "${PROJECT_SOURCE_DIR}/src/graph/chain.cpp")

if(BUILD_TESTING)
    add_executable(strix-graph-chain-tests "${PROJECT_SOURCE_DIR}/tests/graph_chain_tests.cpp")
    target_link_libraries(strix-graph-chain-tests PRIVATE strix_model_graph_stage strix_qwen_model_description)
    if(TARGET strix_deepseek41_model_description)
        target_link_libraries(strix-graph-chain-tests PRIVATE strix_deepseek41_model_description)
        target_compile_definitions(strix-graph-chain-tests PRIVATE STRIX_D3_HAS_DS41=1)
    endif()
    strix_target_warnings(strix-graph-chain-tests)
    add_test(NAME strix-graph-chain-tests COMMAND strix-graph-chain-tests
        "${PROJECT_SOURCE_DIR}/models/qwen3.8-flash-next.toml")
    set_tests_properties(strix-graph-chain-tests PROPERTIES LABELS "cpu;plan")
endif()
