# B8: tails of the microscaled R1 down-combine kernel (MoE tail folded into the down epilogue).
if(BUILD_TESTING)
    add_executable(strix-r1-tail-test tests/moe_core/r1_tail_test.cpp)
    target_compile_features(strix-r1-tail-test PRIVATE cxx_std_20)
    target_include_directories(strix-r1-tail-test PRIVATE include src)
    strix_target_warnings(strix-r1-tail-test)
    add_test(NAME strix-r1-tail-test COMMAND strix-r1-tail-test)
    set_tests_properties(strix-r1-tail-test PROPERTIES LABELS "cpu;moe;exactness")
endif()

# B8: route sources (inline checked route in gate_up) and the expert union plan of the R1 rows launches.
if(BUILD_TESTING)
    add_executable(strix-r1-route-test tests/moe_core/r1_route_test.cpp)
    target_compile_features(strix-r1-route-test PRIVATE cxx_std_20)
    target_include_directories(strix-r1-route-test PRIVATE include src)
    strix_target_warnings(strix-r1-route-test)
    add_test(NAME strix-r1-route-test COMMAND strix-r1-route-test)
    set_tests_properties(strix-r1-route-test PROPERTIES LABELS "cpu;moe;exactness")
endif()
