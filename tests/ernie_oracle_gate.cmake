string(RANDOM LENGTH 20 ALPHABET 0123456789abcdef suffix)
set(trace "${CMAKE_CURRENT_BINARY_DIR}/ernie-oracle-${suffix}")
execute_process(
    COMMAND "${RUNNER}" "${CHECKPOINT}" "${TEXT}" "${trace}"
        --oracle --limit 64 --threads 12 --cache-mib 8192 --checkpoint-contract
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
if(NOT positions EQUAL 63 OR loss GREATER 2.5 OR loss GREATER_EQUAL unigram OR accuracy LESS 0.40)
    message(FATAL_ERROR "ERNIE host oracle: ${loss} nats, ${accuracy} top-1, ${positions} positions")
endif()
