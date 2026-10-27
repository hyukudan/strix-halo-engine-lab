execute_process(COMMAND "${FIXTURE}" --watchdog-self-test
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 5)
if(NOT result STREQUAL "124" OR NOT error MATCHES "TIMEOUT site=watchdog-self-test")
    message(FATAL_ERROR "watchdog did not terminate the fixture: ${result}\n${output}${error}")
endif()
message(STATUS "watchdog exited 124: ${error}")
