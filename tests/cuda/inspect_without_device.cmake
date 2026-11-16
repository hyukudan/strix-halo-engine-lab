# Exercise the actual entry point: no admitted device, model or device config.
set(ENV{CUDA_VISIBLE_DEVICES} "")
execute_process(COMMAND "${SERVER}" --profile turbo --turbo-weights onfly
    --devices "${CMAKE_CURRENT_BINARY_DIR}/absent-device-config-712.toml"
    --device-id absent --print-effective-config
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 30)
if(NOT status EQUAL 0 OR NOT output MATCHES "qwen-cuda profile turbo"
   OR errors MATCHES "device:")
    message(FATAL_ERROR "Device-free inspection failed (${status}): ${output} ${errors}")
endif()
