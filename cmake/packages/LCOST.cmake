# LCOST: no-model bench of the HIP per-launch cost (stream issue gap, graph replay gap, host issue rate, host sync trip).
# Built only with the HIP runtime; not a ctest (it measures, it does not assert).
if(STRIX_ENABLE_HIP_RUNTIME)
    add_executable(strix-hip-launch-cost-bench tests/hip_launch_cost_bench.hip)
    target_link_libraries(strix-hip-launch-cost-bench PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
endif()
