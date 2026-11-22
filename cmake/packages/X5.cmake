add_library(strix_generic_engine STATIC src/exec/generic_engine.cpp)
target_link_libraries(strix_generic_engine PUBLIC strix_exec strix_state strix_weight_load strix_session
    PRIVATE strix_exec_cpu)
strix_target_warnings(strix_generic_engine)

if(BUILD_TESTING)
    add_executable(strix-serve-conformance-generic tests/serve_conformance_generic.cpp)
    target_link_libraries(strix-serve-conformance-generic PRIVATE strix_generic_engine strix_exec_cpu
        strix_session_owner strix_serve)
    strix_target_warnings(strix-serve-conformance-generic)
    add_test(NAME serve_conformance_generic COMMAND strix-serve-conformance-generic)
    set_tests_properties(serve_conformance_generic PROPERTIES LABELS "cpu;session;X5" TIMEOUT 30)
endif()
