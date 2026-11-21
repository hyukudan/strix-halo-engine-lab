# Standalone dense A8 dot4 decode and IU8 WMMA prefill prototypes for gfx1151.
add_executable(strix-hip-dense-a8-wmma-tests "${CMAKE_CURRENT_LIST_DIR}/../tests/hip_dense_a8_wmma_tests.hip")
set_target_properties(strix-hip-dense-a8-wmma-tests PROPERTIES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
target_compile_options(strix-hip-dense-a8-wmma-tests PRIVATE
    ${STRIX_HIP_WAVE32_FLAG} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee)
# hip_q8_rows_gemv.hpp reads the Q6P32 registry (strix/q6p_registry.hpp).
target_link_libraries(strix-hip-dense-a8-wmma-tests PRIVATE strix_hip_runtime)
strix_target_warnings(strix-hip-dense-a8-wmma-tests)
add_test(NAME strix-hip-dense-a8-wmma-tests COMMAND strix-hip-dense-a8-wmma-tests)
