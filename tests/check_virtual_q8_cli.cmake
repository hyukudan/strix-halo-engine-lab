execute_process(COMMAND "${BINARY}" missing-model.gguf 1
    --virtual-q8-lm-head --virtual-q8-lm-head
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "duplicate virtual Q8 LM-head option")
    message(FATAL_ERROR "Duplicate virtual flag must reject during parsing: ${result} ${output} ${error}")
endif()

execute_process(COMMAND "${BINARY}" missing-model.gguf 1
    --virtual-q8-gdn-input --virtual-q8-gdn-input
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "duplicate virtual Q8 GDN input option")
    message(FATAL_ERROR "Duplicate GDN virtual flag must reject during parsing: ${result} ${output} ${error}")
endif()

execute_process(COMMAND "${BINARY}" missing-model.gguf 1
    --prefetch-q8-gdn-input --prefetch-q8-gdn-input
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "duplicate prefetch Q8 GDN input option")
    message(FATAL_ERROR "Duplicate prefetch flag must reject during parsing: ${result} ${output} ${error}")
endif()
execute_process(COMMAND "${BINARY}" missing-model.gguf 1 --prefetch-q8-gdn-input
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "prefetch-q8-gdn-input requires --virtual-q8-gdn-input")
    message(FATAL_ERROR "Prefetch dependency must reject before GGUF: ${result} ${output} ${error}")
endif()
foreach(first_prefetch ON OFF)
    if(first_prefetch)
        set(options --prefetch-q8-gdn-input --virtual-q8-gdn-input)
    else()
        set(options --virtual-q8-gdn-input --prefetch-q8-gdn-input)
    endif()
    execute_process(COMMAND "${BINARY}" missing-model.gguf 1 ${options}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "cannot open GGUF file")
        message(FATAL_ERROR "Both flag orders must reach missing GGUF after valid parsing: ${result} ${output} ${error}")
    endif()
endforeach()
