# K1 rows family (F1): backend-neutral planner tests (host only). Device fixtures live with the backend cores.
if(BUILD_TESTING)
    add_executable(strix-rows-plan-tests tests/rows/rows_plan_tests.cpp)
    target_include_directories(strix-rows-plan-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(strix-rows-plan-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-rows-plan-tests)
    add_test(NAME strix-rows-plan-tests COMMAND strix-rows-plan-tests)
endif()
