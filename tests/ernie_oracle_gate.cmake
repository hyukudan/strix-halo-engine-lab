string(RANDOM LENGTH 20 ALPHABET 0123456789abcdef suffix)
set(trace "${CMAKE_CURRENT_BINARY_DIR}/ernie-oracle-${suffix}")
execute_process(
    COMMAND "${RUNNER}" "${CHECKPOINT}" "${TEXT}" "${trace}"
        --oracle --limit 64 --threads 12 --cache-mib 8192
        --ordered-simd --table-mxfp4 --parallel-mla
    RESULT_VARIABLE result)
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "ERNIE oracle execution failed: ${result}")
endif()
file(READ "${trace}/run.json" receipt)
string(JSON positions GET "${receipt}" positions)
string(JSON loss GET "${receipt}" mean_nll)
string(JSON accuracy GET "${receipt}" top1_hit_rate)
string(JSON unigram GET "${receipt}" unigram_nll)
foreach(value IN ITEMS positions loss accuracy unigram)
    if(NOT "${${value}}" MATCHES "^[0-9.eE+-]+$")
        message(FATAL_ERROR "ERNIE host oracle: receipt value ${value} is not numeric: '${${value}}'")
    endif()
endforeach()
if(NOT positions EQUAL 63 OR loss GREATER 2.10 OR loss GREATER_EQUAL unigram OR accuracy LESS 0.46)
    message(FATAL_ERROR "ERNIE host oracle: ${loss} nats, ${accuracy} top-1, ${positions} positions")
endif()
