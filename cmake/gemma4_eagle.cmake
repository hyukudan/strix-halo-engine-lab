# e4b-eagle drafter: format loader, host g4-v1 reference, verification cycle helper and engine
# DraftSession adapter; optional HIP device drafter. Included by the gemma4_*.cmake glob after gemma4_c/d2.
if(TARGET strix_gemma4_dflash)
    # EAGLE fallback + MTP assistant (primary, src/gemma4/eagle/mtp) host code.
    add_library(strix_gemma4_eagle STATIC src/gemma4/eagle/host.cpp src/gemma4/eagle/session.cpp
        src/gemma4/eagle/mtp/host.cpp src/gemma4/eagle/mtp/session.cpp)
    target_link_libraries(strix_gemma4_eagle PUBLIC strix_gemma4_dflash strix_gemma4_model)
    if(MSVC)
        target_compile_definitions(strix_gemma4_eagle PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix_gemma4_eagle)

    if(TARGET strix_gemma4_device)
        add_library(strix_gemma4_eagle_device STATIC src/gemma4/eagle/device.hip src/gemma4/eagle/device_session.cpp
            src/gemma4/eagle/mtp/device.hip src/gemma4/eagle/mtp/device_session.cpp)
        target_link_libraries(strix_gemma4_eagle_device PUBLIC strix_gemma4_eagle strix_gemma4_device)
        set_target_properties(strix_gemma4_eagle_device PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES gfx1151)
        target_compile_options(strix_gemma4_eagle_device PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64;-fno-fast-math;-ffp-contract=off;-fhip-fp32-correctly-rounded-divide-sqrt;-fdenormal-fp-math=ieee;-fno-gpu-flush-denormals-to-zero>)
        set_property(SOURCE src/gemma4/eagle/device.hip src/gemma4/eagle/mtp/device.hip APPEND PROPERTY COMPILE_OPTIONS -fno-rounding-math)
        strix_target_warnings(strix_gemma4_eagle_device)
    else()
        # Host compiler check of the device-facing sources (no HIP runtime, never linked).
        add_library(strix_gemma4_eagle_device_host_compile OBJECT EXCLUDE_FROM_ALL
            src/gemma4/eagle/device_session.cpp tests/gemma4_eagle/device_tests.cpp
            src/gemma4/eagle/mtp/device_session.cpp tests/gemma4_eagle/mtp_device_tests.cpp)
        target_link_libraries(strix_gemma4_eagle_device_host_compile PRIVATE strix_gemma4_eagle)
        strix_target_warnings(strix_gemma4_eagle_device_host_compile)
    endif()

    if(BUILD_TESTING)
        add_executable(strix-gemma4-eagle-host-tests tests/gemma4_eagle/host_tests.cpp)
        target_link_libraries(strix-gemma4-eagle-host-tests PRIVATE strix_gemma4_eagle)
        strix_target_warnings(strix-gemma4-eagle-host-tests)
        add_test(NAME strix-gemma4-eagle-host-tests COMMAND strix-gemma4-eagle-host-tests)
        set_tests_properties(strix-gemma4-eagle-host-tests PROPERTIES TIMEOUT 600 LABELS "gemma4")
        add_executable(strix-gemma4-eagle-mtp-tests tests/gemma4_eagle/mtp_tests.cpp)
        target_link_libraries(strix-gemma4-eagle-mtp-tests PRIVATE strix_gemma4_eagle)
        strix_target_warnings(strix-gemma4-eagle-mtp-tests)
        add_test(NAME strix-gemma4-eagle-mtp-tests COMMAND strix-gemma4-eagle-mtp-tests)
        set_tests_properties(strix-gemma4-eagle-mtp-tests PROPERTIES TIMEOUT 240 LABELS "gemma4")
        if(TARGET strix_gemma4_engine)
            # End-to-end through engine::Coordinator; prints SKIP until the staged-rows draft API is present.
            add_executable(strix-gemma4-eagle-coordinator-tests tests/gemma4_eagle/coordinator_tests.cpp)
            target_link_libraries(strix-gemma4-eagle-coordinator-tests PRIVATE strix_gemma4_engine strix_gemma4_eagle)
            strix_target_warnings(strix-gemma4-eagle-coordinator-tests)
            add_test(NAME strix-gemma4-eagle-coordinator-tests COMMAND strix-gemma4-eagle-coordinator-tests)
            # The real-Coordinator e2e takes > 120 s on a busy host build (owner request).
            set_tests_properties(strix-gemma4-eagle-coordinator-tests PROPERTIES TIMEOUT 600 LABELS "gemma4")
            # Prompt lookup: n-gram index vs brute force; hybrid session through the real Coordinator (ids == OFF).
            add_executable(strix-gemma4-lookup-tests tests/gemma4_eagle/lookup_tests.cpp)
            target_link_libraries(strix-gemma4-lookup-tests PRIVATE strix_gemma4_engine strix_gemma4_eagle)
            strix_target_warnings(strix-gemma4-lookup-tests)
            add_test(NAME strix-gemma4-lookup-tests COMMAND strix-gemma4-lookup-tests)
            set_tests_properties(strix-gemma4-lookup-tests PROPERTIES TIMEOUT 300 LABELS "gemma4")
        endif()
        if(TARGET strix_gemma4_eagle_device)
            add_executable(strix-gemma4-eagle-device-tests tests/gemma4_eagle/device_tests.cpp)
            target_link_libraries(strix-gemma4-eagle-device-tests PRIVATE strix_gemma4_eagle_device)
            strix_target_warnings(strix-gemma4-eagle-device-tests)
            add_executable(strix-gemma4-eagle-mtp-device-tests tests/gemma4_eagle/mtp_device_tests.cpp)
            target_link_libraries(strix-gemma4-eagle-mtp-device-tests PRIVATE strix_gemma4_eagle_device)
            strix_target_warnings(strix-gemma4-eagle-mtp-device-tests)
            add_test(NAME strix-gemma4-eagle-mtp-device-tests COMMAND strix-gemma4-eagle-mtp-device-tests)
            set_tests_properties(strix-gemma4-eagle-mtp-device-tests PROPERTIES TIMEOUT 240 LABELS "gpu;gemma4")
            add_test(NAME strix-gemma4-eagle-device-tests COMMAND strix-gemma4-eagle-device-tests)
            add_test(NAME strix-gemma4-eagle-device-off-tests COMMAND strix-gemma4-eagle-device-tests)
            set_tests_properties(strix-gemma4-eagle-device-tests strix-gemma4-eagle-device-off-tests
                PROPERTIES TIMEOUT 240 LABELS "gpu;gemma4")
            set_tests_properties(strix-gemma4-eagle-device-off-tests PROPERTIES ENVIRONMENT
                "STRIX_G4_SHARED_WEIGHTS=0;STRIX_G4_NORM_GROUP=0;STRIX_G4_NORM_SEARCH=0;STRIX_G4_NORM_PREFETCH=0;STRIX_G4_ATTN_PREFETCH=0;STRIX_G4_CHUNK_DECODE=0")
        endif()
    endif()
endif()
