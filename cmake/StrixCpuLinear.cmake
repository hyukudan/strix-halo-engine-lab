# Baseline CPU feature detection and FP32 linear family, independent of accelerator SDKs.
if(TARGET strix_cpu_linear OR NOT (STRIX_BACKEND_CPU OR STRIX_FAMILY_GEMMA4 OR STRIX_CUDA_QWEN_ONLY))
    return()
endif()

get_filename_component(_strix_cpu_linear_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_strix_cpu_avx2 "${_strix_cpu_linear_root}/src/cpu/arch/avx2/dense.cpp")
set(_strix_cpu_avx512 "${_strix_cpu_linear_root}/src/cpu/arch/avx512/dense.cpp")
add_library(strix_cpu_linear STATIC
    "${_strix_cpu_linear_root}/src/cpu/core/isa.cpp"
    "${_strix_cpu_linear_root}/src/cpu/core/dense.cpp"
    "${_strix_cpu_avx2}" "${_strix_cpu_avx512}")
target_include_directories(strix_cpu_linear PUBLIC "${_strix_cpu_linear_root}/src")
target_compile_features(strix_cpu_linear PUBLIC cxx_std_20)
strix_target_warnings(strix_cpu_linear)
if(MSVC)
    target_compile_options(strix_cpu_linear PRIVATE /fp:strict)
else()
    target_compile_options(strix_cpu_linear PRIVATE -fno-fast-math -ffp-contract=off -frounding-math)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|X86_64|x64")
    if(MSVC)
        # MSVC emits the explicit intrinsics without enabling SIMD for unrelated code.
        target_compile_options(strix_cpu_linear PRIVATE /arch:SSE2)
    else()
        target_compile_options(strix_cpu_linear PRIVATE -march=x86-64)
        set_source_files_properties("${_strix_cpu_avx2}" PROPERTIES COMPILE_OPTIONS "-mavx2")
        set_source_files_properties("${_strix_cpu_avx512}" PROPERTIES COMPILE_OPTIONS "-mavx512f")
    endif()
endif()

if(BUILD_TESTING)
    add_executable(strix-cpu-isa-tests "${_strix_cpu_linear_root}/tests/cpu/cpu_isa_tests.cpp")
    target_link_libraries(strix-cpu-isa-tests PRIVATE strix_cpu_linear)
    strix_target_warnings(strix-cpu-isa-tests)
    add_test(NAME strix-cpu-isa-tests COMMAND strix-cpu-isa-tests)
    set_tests_properties(strix-cpu-isa-tests PROPERTIES LABELS "cpu;exactness")
endif()
