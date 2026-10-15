execute_process(COMMAND "${BINARY}" absent.gguf 1 --virtual-q8-gdn-output --virtual-q8-gdn-output
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "duplicate virtual Q8 GDN output option")
    message(FATAL_ERROR "Unexpected duplicate output CLI result: ${result} ${output} ${error}")
endif()
