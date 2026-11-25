# Residual transition, HIP body (residual_mix::launch_transition): device fixture, transition == add + readout + RMS leaves bitwise.
if(STRIX_ENABLE_HIP_RUNTIME AND BUILD_TESTING)
    add_executable(strix-hip-residual-transition-test tests/core/residual_transition_hip.hip)
    target_link_libraries(strix-hip-residual-transition-test PRIVATE strix_tiled_core)
    strix_target_warnings(strix-hip-residual-transition-test)
    # The Inkling 2 Mini HIP engine's floating-point options (the readout and norm kernels it binds).
    target_compile_options(strix-hip-residual-transition-test PRIVATE -O3 -mno-wavefrontsize64 -ffp-contract=off
        -fno-fast-math -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
    set_target_properties(strix-hip-residual-transition-test PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    # Device execution is an operator task (node ticket); not registered with CTest.
endif()

# Host only: the admission rule of the HIP transition body (which shapes fuse, which keep the separate leaves).
if(BUILD_TESTING)
    add_executable(strix-residual-transition-admission-tests tests/core/residual_transition_admission_tests.cpp)
    target_include_directories(strix-residual-transition-admission-tests PRIVATE include src)
    target_compile_features(strix-residual-transition-admission-tests PRIVATE cxx_std_20)
    strix_target_warnings(strix-residual-transition-admission-tests)
    add_test(NAME residual_transition_admission COMMAND strix-residual-transition-admission-tests)
    set_tests_properties(residual_transition_admission PROPERTIES LABELS "cpu;plan" TIMEOUT 30)
endif()
