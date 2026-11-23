# Residual bank batch, HIP body (residual_mix::launch_bank_batch): device fixture, job == single-query readout bits.
if(STRIX_ENABLE_HIP_RUNTIME AND BUILD_TESTING)
    add_executable(strix-hip-residual-bank-batch-test tests/core/residual_bank_batch_hip.hip)
    target_link_libraries(strix-hip-residual-bank-batch-test PRIVATE strix_tiled_core)
    strix_target_warnings(strix-hip-residual-bank-batch-test)
    # The Inkling 2 Mini HIP engine's floating-point options (the readout kernels it binds).
    target_compile_options(strix-hip-residual-bank-batch-test PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off
        -fno-fast-math -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
    set_target_properties(strix-hip-residual-bank-batch-test PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    # Device execution is an operator task (node ticket); not registered with CTest.
endif()
