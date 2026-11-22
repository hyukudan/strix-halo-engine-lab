target_sources(strix_exec PRIVATE src/exec/descriptor.cpp)

if(BUILD_TESTING)
    add_executable(strix-exec-descriptor-tests tests/exec_descriptor_tests.cpp)
    target_link_libraries(strix-exec-descriptor-tests PRIVATE strix_exec)
    strix_target_warnings(strix-exec-descriptor-tests)
    add_test(NAME row_invariance COMMAND strix-exec-descriptor-tests)
    set_tests_properties(row_invariance PROPERTIES LABELS "cpu;plan" TIMEOUT 30)
endif()
