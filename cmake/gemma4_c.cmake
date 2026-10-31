# Host-only DFlash arithmetic reference and diagnostic probe.
add_library(strix_gemma4_dflash STATIC
    src/gemma4/dflash/host.cpp
    src/gguf_reader.cpp)
target_include_directories(strix_gemma4_dflash PUBLIC src include)
target_compile_features(strix_gemma4_dflash PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(strix_gemma4_dflash PUBLIC /fp:strict /permissive- /Zc:__cplusplus /utf-8)
    target_compile_definitions(strix_gemma4_dflash PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(strix_gemma4_dflash PUBLIC -fno-fast-math -ffp-contract=off -frounding-math)
endif()

if(TARGET strix_gemma4_device)
    add_library(strix_gemma4_dflash_device STATIC src/gemma4/dflash/device.hip src/gemma4/dflash/controller.cpp)
    target_link_libraries(strix_gemma4_dflash_device PUBLIC strix_gemma4_dflash strix_gemma4_device)
    set_target_properties(strix_gemma4_dflash_device PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
    target_compile_options(strix_gemma4_dflash_device PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee;-fno-gpu-flush-denormals-to-zero>)
    set_property(SOURCE src/gemma4/dflash/device.hip APPEND PROPERTY COMPILE_OPTIONS -fno-rounding-math)
    strix_target_warnings(strix_gemma4_dflash_device)
    add_executable(strix-gemma4-dflash-device-probe src/gemma4/dflash/device_probe_main.cpp)
    target_link_libraries(strix-gemma4-dflash-device-probe PRIVATE strix_gemma4_dflash_device)
    strix_target_warnings(strix-gemma4-dflash-device-probe)
    if(BUILD_TESTING)
        add_executable(strix-gemma4-dflash-device-tests tests/gemma4_dflash/device_tests.cpp tests/gemma4_dflash/device_cycle_tests.cpp)
        target_link_libraries(strix-gemma4-dflash-device-tests PRIVATE strix_gemma4_dflash_device)
        strix_target_warnings(strix-gemma4-dflash-device-tests)
        add_test(NAME strix-gemma4-dflash-device-tests COMMAND strix-gemma4-dflash-device-tests
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_dflash/fixtures")
        add_test(NAME strix-gemma4-dflash-device-off-tests COMMAND strix-gemma4-dflash-device-tests
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_dflash/fixtures")
        set_tests_properties(strix-gemma4-dflash-device-tests strix-gemma4-dflash-device-off-tests
            PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4")
        set_tests_properties(strix-gemma4-dflash-device-off-tests PROPERTIES
            ENVIRONMENT "STRIX_G4_SHARED_WEIGHTS=0;STRIX_G4_NORM_GROUP=0;STRIX_G4_NORM_SEARCH=0;STRIX_G4_NORM_PREFETCH=0")
    endif()
elseif(TARGET strix_gemma4_ref)
    add_library(strix_gemma4_dflash_device_host_compile OBJECT EXCLUDE_FROM_ALL
        src/gemma4/dflash/controller.cpp src/gemma4/dflash/device_probe_main.cpp
        tests/gemma4_dflash/device_tests.cpp tests/gemma4_dflash/device_cycle_tests.cpp)
    target_link_libraries(strix_gemma4_dflash_device_host_compile PRIVATE strix_gemma4_dflash strix_gemma4_ref)
    strix_target_warnings(strix_gemma4_dflash_device_host_compile)
endif()
strix_target_warnings(strix_gemma4_dflash)
add_executable(strix-gemma4-dflash-probe src/gemma4/dflash/dflash_probe_main.cpp)
target_link_libraries(strix-gemma4-dflash-probe PRIVATE strix_gemma4_dflash)
strix_target_warnings(strix-gemma4-dflash-probe)

if(TARGET strix_gemma4_ref)
    add_executable(strix-gemma4-dflash-reference src/gemma4/dflash/dflash_reference_main.cpp)
    target_link_libraries(strix-gemma4-dflash-reference PRIVATE strix_gemma4_dflash strix_gemma4_ref)
    strix_target_warnings(strix-gemma4-dflash-reference)
endif()

if(BUILD_TESTING)
    foreach(part loader protocol forward)
        add_executable(strix-gemma4-dflash-${part}-tests tests/gemma4_dflash/${part}_tests.cpp)
        target_link_libraries(strix-gemma4-dflash-${part}-tests PRIVATE strix_gemma4_dflash)
        strix_target_warnings(strix-gemma4-dflash-${part}-tests)
        add_test(NAME strix-gemma4-dflash-${part}-tests COMMAND strix-gemma4-dflash-${part}-tests
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_dflash/fixtures")
        set_tests_properties(strix-gemma4-dflash-${part}-tests PROPERTIES TIMEOUT 60)
    endforeach()
    add_test(NAME strix-gemma4-dflash-off-tests COMMAND strix-gemma4-dflash-protocol-tests --off)
    set_tests_properties(strix-gemma4-dflash-off-tests PROPERTIES ENVIRONMENT "STRIX_G4_DFLASH=0" TIMEOUT 60)
endif()
