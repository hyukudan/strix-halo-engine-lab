# Multi-request decode step lanes on HIP: device fixture, lane forms (short convolution, position-scaled clip)
# == the one-row stage of each lane, bit for bit (the HIP eager lane mode binds these forms).
if(STRIX_ENABLE_HIP_RUNTIME AND BUILD_TESTING)
    add_executable(strix-hip-step-lanes-test tests/core/step_lanes_hip.hip)
    target_link_libraries(strix-hip-step-lanes-test PRIVATE strix_tiled_core)
    strix_target_warnings(strix-hip-step-lanes-test)
    # The Inkling 2 Mini HIP engine's floating-point options (the lane kernels it binds).
    target_compile_options(strix-hip-step-lanes-test PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off
        -fno-fast-math -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
    set_target_properties(strix-hip-step-lanes-test PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    # Device execution is an operator task (node ticket); not registered with CTest.
endif()
