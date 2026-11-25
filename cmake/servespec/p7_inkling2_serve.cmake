if(TARGET strix_cohort_session AND TARGET strix_inkling2_mini_checkpoint)
    add_library(strix_inkling2_mini_service STATIC src/inkling2/inkling2_mini_serve.cpp)
    target_include_directories(strix_inkling2_mini_service PUBLIC src)
    target_link_libraries(strix_inkling2_mini_service PUBLIC strix_cohort_session PRIVATE strix-weight)
    strix_target_warnings(strix_inkling2_mini_service)
    if(BUILD_TESTING)
        add_executable(inkling2-mini-serve-test tests/inkling2/mini_serve_test.cpp)
        target_include_directories(inkling2-mini-serve-test PRIVATE apps)
        target_link_libraries(inkling2-mini-serve-test PRIVATE strix_inkling2_mini_service strix_session_shell)
        strix_target_warnings(inkling2-mini-serve-test)
        add_test(NAME inkling2-mini-serve COMMAND inkling2-mini-serve-test "${CMAKE_SOURCE_DIR}")
        add_test(NAME inkling2-mini-profile-parity COMMAND inkling2-mini-serve-test
            "${CMAKE_SOURCE_DIR}" --profile-parity)
        set_tests_properties(inkling2-mini-serve inkling2-mini-profile-parity
            PROPERTIES LABELS "cpu;inkling2" TIMEOUT 30)
        set(STRIX_INKLING2_SERVE_CHECKPOINT "" CACHE FILEPATH "Inkling checkpoint for real CPU cohort ID parity")
        add_executable(inkling2-mini-serve-cpu-model-test EXCLUDE_FROM_ALL tests/inkling2/mini_serve_cpu_model_test.cpp
            src/inkling2/inkling2_mini_host.cpp src/inkling2/inkling2_mini_dspark_reference.cpp)
        target_link_libraries(inkling2-mini-serve-cpu-model-test PRIVATE
            strix_inkling2_mini_checkpoint strix_cohort_session)
        strix_target_warnings(inkling2-mini-serve-cpu-model-test)
        if(MSVC)
            target_compile_options(inkling2-mini-serve-cpu-model-test PRIVATE /fp:strict)
        else()
            target_compile_options(inkling2-mini-serve-cpu-model-test PRIVATE -ffp-contract=off)
        endif()
        if(STRIX_INKLING2_SERVE_CHECKPOINT)
            set_property(TARGET inkling2-mini-serve-cpu-model-test PROPERTY EXCLUDE_FROM_ALL FALSE)
            add_test(NAME inkling2-mini-serve-cpu-model COMMAND inkling2-mini-serve-cpu-model-test
                "${STRIX_INKLING2_SERVE_CHECKPOINT}")
            set_tests_properties(inkling2-mini-serve-cpu-model PROPERTIES LABELS "cpu;inkling2" TIMEOUT 540)
        endif()
    endif()

    set(STRIX_INKLING2_SERVE_BACKEND "auto" CACHE STRING "Inkling backend in strix-serve (auto, hip, cuda)")
    set_property(CACHE STRIX_INKLING2_SERVE_BACKEND PROPERTY STRINGS auto hip cuda)
    set(_inkling_backend "${STRIX_INKLING2_SERVE_BACKEND}")
    if(_inkling_backend STREQUAL "auto")
        if(TARGET strix-inkling2-mini-hip)
            set(_inkling_backend hip)
        elseif(TARGET strix-inkling2-mini-cuda)
            set(_inkling_backend cuda)
        endif()
    endif()

    # Native Inkling decoder types are instantiated by one device compiler
    # per strix-serve binary. Both builds use the same family registration.
    if(_inkling_backend STREQUAL "hip" AND TARGET strix-inkling2-mini-hip)
        add_library(strix_inkling2_mini_serve_device STATIC src/inkling2/inkling2_mini_serve_hip.hip)
        target_link_libraries(strix_inkling2_mini_serve_device PRIVATE strix_inkling2_mini_service
            strix_inkling2_mini_checkpoint strix_tiled_core strix_hip_runtime strix_core strix_device_hip)
        strix_target_warnings(strix_inkling2_mini_serve_device)
        target_compile_options(strix_inkling2_mini_serve_device PRIVATE
            -O3 -mno-wavefrontsize64 -ffp-contract=off -fno-fast-math
            -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt)
        set_target_properties(strix_inkling2_mini_serve_device PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    elseif(_inkling_backend STREQUAL "cuda" AND TARGET strix-inkling2-mini-cuda)
        add_library(strix_inkling2_mini_serve_device STATIC src/inkling2/inkling2_mini_serve_cuda.cu)
        target_link_libraries(strix_inkling2_mini_serve_device PRIVATE strix_inkling2_mini_service
            strix_inkling2_mini_checkpoint strix_cuda_core_kernels strix_core strix_device)
        get_target_property(_inkling_cuda_options strix-inkling2-mini-cuda COMPILE_OPTIONS)
        target_compile_options(strix_inkling2_mini_serve_device PRIVATE ${_inkling_cuda_options})
    elseif(NOT _inkling_backend STREQUAL "auto")
        message(FATAL_ERROR "Requested Inkling serving backend '${_inkling_backend}' is not available")
    endif()
    if(TARGET strix_inkling2_mini_serve_device)
        # One serve engine row (key, namespace of install_session_service): run.cpp names no model.
        # The row is a host TU bound to the one device library this build selected.
        string(TOUPPER "${_inkling_backend}" _inkling_backend_upper)
        add_library(strix_inkling2_mini_serve_engine STATIC src/inkling2/inkling2_mini_serve_install.cpp)
        target_link_libraries(strix_inkling2_mini_serve_engine PUBLIC strix_inkling2_mini_service
            strix_inkling2_mini_serve_device)
        target_compile_definitions(strix_inkling2_mini_serve_engine PRIVATE
            STRIX_INKLING2_SERVE_${_inkling_backend_upper}=1)
        strix_target_warnings(strix_inkling2_mini_serve_engine)
        strix_serve_register_engine(inkling2_mini strix::inkling2::serve_engine)
        target_link_libraries(strix_session_shell PRIVATE strix_inkling2_mini_serve_engine)
        if(BUILD_TESTING)
            add_test(NAME inkling2-mini-serve-factory-${_inkling_backend}
                COMMAND strix-serve --model inkling2-mini --registry-root "${CMAKE_SOURCE_DIR}"
                    --family-inspect --checkpoint MODEL.safetensors --backend ${_inkling_backend})
            set_tests_properties(inkling2-mini-serve-factory-${_inkling_backend} PROPERTIES
                LABELS "cpu;inkling2" PASS_REGULAR_EXPRESSION "backend ${_inkling_backend} context")
        endif()
    endif()
endif()
