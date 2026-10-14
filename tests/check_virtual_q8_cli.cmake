execute_process(COMMAND "${BINARY}" missing-model.gguf 1
    --virtual-q8-lm-head --virtual-q8-lm-head
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "duplicate virtual Q8 LM-head option")
    message(FATAL_ERROR "Duplicate virtual flag must reject during parsing: ${result} ${output} ${error}")
endif()
