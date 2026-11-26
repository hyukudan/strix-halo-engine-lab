# B3 mechanism 7: program segments (prelude / core x repeats / coda), launch budget and per-step receipt counts.
target_sources(strix_stage_graph PRIVATE "${PROJECT_SOURCE_DIR}/src/graph/program_segments.cpp")

if(BUILD_TESTING)
    add_executable(strix-step-graph-runner-tests tests/graph/step_runner_tests.cpp)
    target_include_directories(strix-step-graph-runner-tests PRIVATE include tests/stubs/cuda_status)
    target_compile_definitions(strix-step-graph-runner-tests PRIVATE STRIX_STRICT_GRAPH_CAPTURE=1)
    target_compile_features(strix-step-graph-runner-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-step-graph-runner-tests)
    add_test(NAME strix-step-graph-runner-tests COMMAND strix-step-graph-runner-tests)
    set_tests_properties(strix-step-graph-runner-tests PROPERTIES LABELS "cpu;plan")

    add_executable(strix-program-segments-tests "${PROJECT_SOURCE_DIR}/tests/graph/program_segments_tests.cpp")
    target_link_libraries(strix-program-segments-tests PRIVATE strix_stage_graph)
    strix_target_warnings(strix-program-segments-tests)
    add_test(NAME strix-program-segments-tests COMMAND strix-program-segments-tests)
    set_tests_properties(strix-program-segments-tests PROPERTIES LABELS "cpu;plan")
endif()
