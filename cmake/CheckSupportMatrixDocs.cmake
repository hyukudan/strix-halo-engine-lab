execute_process(COMMAND "${TOOL}" --markdown --root "${ROOT}"
    OUTPUT_FILE "${OUTPUT}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files --ignore-eol
    "${ROOT}/docs/support-matrix.md" "${OUTPUT}" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "docs/support-matrix.md differs from strix-support-matrix --markdown")
endif()
