string(RANDOM LENGTH 12 suffix)
set(directory "${CMAKE_CURRENT_BINARY_DIR}/mini-cli-test-${suffix}")
file(MAKE_DIRECTORY "${directory}")
set(checkpoint "${directory}/checkpoint.safetensors")
set(output "${directory}/logits.f32")
file(WRITE "${checkpoint}" "invalid checkpoint")

execute_process(COMMAND "${DECODER}" "${checkpoint}" 1 --logits "${checkpoint}"
    RESULT_VARIABLE code ERROR_VARIABLE error)
if(NOT code EQUAL 1 OR NOT error MATCHES "logits output must differ from the checkpoint")
    message(FATAL_ERROR "decoder did not reject checkpoint as output: ${error}")
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
file(REMOVE_RECURSE "${directory}")
