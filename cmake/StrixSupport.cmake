add_library(strix_support_matrix STATIC src/support/matrix.cpp)
target_link_libraries(strix_support_matrix PRIVATE strix_model_registry)
target_include_directories(strix_support_matrix PUBLIC include)
target_compile_features(strix_support_matrix PUBLIC cxx_std_20)
strix_target_warnings(strix_support_matrix)

add_executable(strix-support-matrix apps/support_matrix_main.cpp)
target_link_libraries(strix-support-matrix PRIVATE strix_support_matrix)
strix_target_warnings(strix-support-matrix)

if(BUILD_TESTING)
    add_executable(strix-support-matrix-tests tests/support_matrix_tests.cpp)
    add_dependencies(strix-support-matrix-tests strix-support-matrix)
    target_link_libraries(strix-support-matrix-tests PRIVATE strix_support_matrix strix_model_description)
    strix_target_warnings(strix-support-matrix-tests)
    add_test(NAME strix-support-matrix-validation COMMAND strix-support-matrix-tests "${PROJECT_SOURCE_DIR}")
    add_test(NAME strix-support-matrix-check COMMAND strix-support-matrix --check --root "${PROJECT_SOURCE_DIR}")
    add_test(NAME strix-support-matrix-docs COMMAND "${CMAKE_COMMAND}"
        "-DTOOL=$<TARGET_FILE:strix-support-matrix>" "-DROOT=${PROJECT_SOURCE_DIR}"
        "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/support-matrix.rendered.md"
        -P "${CMAKE_CURRENT_LIST_DIR}/CheckSupportMatrixDocs.cmake")
    set_tests_properties(strix-support-matrix-validation strix-support-matrix-check strix-support-matrix-docs
        PROPERTIES LABELS "cpu;support" TIMEOUT 30)
endif()
