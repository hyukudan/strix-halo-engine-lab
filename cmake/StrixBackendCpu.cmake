# CPU backend (src/cpu): layer-2 backend of the device rule with ISA specialisations under
# src/cpu/arch/<isa>. Opt-in; nothing else in the build depends on it.
add_library(strix_backend_cpu STATIC
    src/cpu/core/reference.cpp
    src/cpu/core/select.cpp
    src/cpu/arch/avx512/kernels.cpp
    src/cpu/arch/neon/kernels.cpp)
target_include_directories(strix_backend_cpu PUBLIC src)
target_compile_features(strix_backend_cpu PUBLIC cxx_std_20)
strix_target_warnings(strix_backend_cpu)
if(MSVC)
    target_compile_options(strix_backend_cpu PRIVATE /fp:precise)
else()
    target_compile_options(strix_backend_cpu PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    find_package(Threads REQUIRED)
    # The ds41 Q8_0 / A8 host reference TU only (the rest of strix_ds41_formats is not needed).
    add_executable(strix-cpu-backend-host-tests tests/cpu_backend_host_tests.cpp src/ds41_formats.cpp)
    target_link_libraries(strix-cpu-backend-host-tests PRIVATE strix_backend_cpu Threads::Threads)
    target_include_directories(strix-cpu-backend-host-tests PRIVATE include)
    if(NOT MSVC)
        target_compile_options(strix-cpu-backend-host-tests PRIVATE -ffp-contract=off)
    endif()
    strix_target_warnings(strix-cpu-backend-host-tests)
    add_test(NAME strix-cpu-backend-host-tests COMMAND strix-cpu-backend-host-tests)
endif()
