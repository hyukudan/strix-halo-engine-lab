string(RANDOM LENGTH 12 suffix)
set(directory "${CMAKE_CURRENT_BINARY_DIR}/mini-cli-test-${suffix}")
file(MAKE_DIRECTORY "${directory}")
set(checkpoint "${directory}/checkpoint.safetensors")
set(output "${directory}/logits.f32")
file(WRITE "${checkpoint}" "invalid checkpoint")

execute_process(COMMAND "${DECODER}" "${checkpoint}" 1 --logits "${checkpoint}"
    RESULT_VARIABLE code ERROR_VARIABLE error)
if(NOT code EQUAL 1 OR NOT error MATCHES "logits output exists")
    message(FATAL_ERROR "decoder did not refuse an existing logits output: ${error}")
endif()
file(READ "${checkpoint}" payload)
if(NOT payload STREQUAL "invalid checkpoint")
    message(FATAL_ERROR "decoder modified the checkpoint")
endif()

execute_process(COMMAND "${DECODER}" "${checkpoint}" 1 --logits "${output}"
    RESULT_VARIABLE code ERROR_VARIABLE error)
if(NOT code EQUAL 1 OR error MATCHES "filesystem error|logits output must differ" OR EXISTS "${output}")
    message(FATAL_ERROR "fresh logits path prevented checkpoint validation: ${error}")
endif()
file(GLOB staged "${directory}/.strix-logits-*")
if(staged)
    message(FATAL_ERROR "a failed run left a staged logits dump: ${staged}")
endif()

execute_process(COMMAND "${DECODER}" "${checkpoint}" 1 --generate 2 --speculation verify --temperature 1
    RESULT_VARIABLE code ERROR_VARIABLE error)
if(NOT code EQUAL 1 OR NOT error MATCHES "--speculation verify requires greedy decoding")
    message(FATAL_ERROR "decoder admitted speculative sampling: ${error}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E env STRIX_MOE_EXPERTS_NVFP4=1
    "${DECODER}" "${checkpoint}" 1 --profile fast
    RESULT_VARIABLE code ERROR_VARIABLE error)
if(NOT code EQUAL 1 OR NOT error MATCHES "STRIX_MOE_EXPERTS_NVFP4=1 requires native resident model execution")
    message(FATAL_ERROR "CPU decoder did not refuse the narrow image before loading: ${error}")
endif()
file(REMOVE_RECURSE "${directory}")
