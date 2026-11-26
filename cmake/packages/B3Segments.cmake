# B3 mechanism 7: program segments (prelude / core x repeats / coda), launch budget and per-step receipt counts.
# The stage graph is not configured in the CUDA DeepSeek-only tree (STRIX_CUDA_DS41_ONLY); nothing here applies there.
if(NOT TARGET strix_stage_graph)
    return()
endif()
target_sources(strix_stage_graph PRIVATE "${PROJECT_SOURCE_DIR}/src/graph/program_segments.cpp")

if(BUILD_TESTING)
    add_executable(strix-program-segments-tests "${PROJECT_SOURCE_DIR}/tests/graph/program_segments_tests.cpp")
    target_link_libraries(strix-program-segments-tests PRIVATE strix_stage_graph)
    strix_target_warnings(strix-program-segments-tests)
    add_test(NAME strix-program-segments-tests COMMAND strix-program-segments-tests)
    set_tests_properties(strix-program-segments-tests PROPERTIES LABELS "cpu;plan")
endif()
