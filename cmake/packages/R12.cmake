if(BUILD_TESTING AND TARGET strix_serve_json)
    add_executable(strix-receipt-tests tests/kit/receipt_tests.cpp)
    add_executable(strix-conformance-kit-tests tests/kit/conformance_tests.cpp)
    foreach(target strix-receipt-tests strix-conformance-kit-tests)
        target_link_libraries(${target} PRIVATE strix_serve_json)
        target_compile_features(${target} PRIVATE cxx_std_20)
        strix_target_warnings(${target})
    endforeach()

    set(r12_receipts "${CMAKE_CURRENT_BINARY_DIR}/r12-receipts/$<CONFIG>")
    add_test(NAME receipt_identity COMMAND strix-receipt-tests "${r12_receipts}")
    set_tests_properties(receipt_identity PROPERTIES FIXTURES_SETUP r12_receipts LABELS "host;R1")
    foreach(case capture_compare kernel_conformance chain_fixtures row_invariance)
        add_test(NAME r12_${case} COMMAND strix-conformance-kit-tests ${case} "${CMAKE_CURRENT_SOURCE_DIR}/tests/kit/seeded")
        set_tests_properties(r12_${case} PROPERTIES LABELS "host;R2")
    endforeach()

    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME receipt_schema COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/receipt-schema/check.py"
            "${r12_receipts}/engine.json" "${r12_receipts}/route.json" "${r12_receipts}/complete.json")
        add_test(NAME receipt_schema_rejects_invalid COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/kit/receipt_schema_tests.py" "${r12_receipts}/complete.json")
        set_tests_properties(receipt_schema receipt_schema_rejects_invalid PROPERTIES FIXTURES_REQUIRED r12_receipts LABELS "host;R1")
        add_test(NAME r12_lint_report COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/lint/check.py" --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --report "${CMAKE_CURRENT_BINARY_DIR}/r12-lint-$<CONFIG>.txt")
        add_test(NAME r12_lint_rules COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/kit/lint_tests.py" "${CMAKE_CURRENT_BINARY_DIR}/r12-lint-selftest-$<CONFIG>.txt")
        set_tests_properties(r12_lint_report r12_lint_rules PROPERTIES LABELS "host;R2")
    endif()
    unset(r12_receipts)
endif()
