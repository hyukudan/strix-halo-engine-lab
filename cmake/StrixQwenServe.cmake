if(STRIX_ENABLE_HIP_RUNTIME OR BUILD_TESTING)
    add_library(strix_qwen_serve_options STATIC src/qwen/session_service.cpp)
    target_include_directories(strix_qwen_serve_options PUBLIC src PRIVATE apps)
    target_link_libraries(strix_qwen_serve_options PUBLIC strix_session_runtime
        PRIVATE strix_qwen_model_description strix_device)
    strix_target_warnings(strix_qwen_serve_options)
endif()

if(BUILD_TESTING)
    add_executable(strix-qwen-serve-adapter-link-tests tests/qwen_serve_adapter_link_tests.cpp)
    target_link_libraries(strix-qwen-serve-adapter-link-tests PRIVATE strix_qwen_serve_options)
    strix_target_warnings(strix-qwen-serve-adapter-link-tests)
    add_test(NAME strix-qwen-serve-adapter-link-tests COMMAND strix-qwen-serve-adapter-link-tests)
    set_tests_properties(strix-qwen-serve-adapter-link-tests PROPERTIES TIMEOUT 30 LABELS "cpu;serve;qwen")

    add_executable(strix-qwen-serve-factory-tests tests/qwen_serve_factory_tests.cpp)
    target_include_directories(strix-qwen-serve-factory-tests PRIVATE apps tests)
    target_link_libraries(strix-qwen-serve-factory-tests PRIVATE strix_qwen_serve_options strix_session_shell)
    strix_target_warnings(strix-qwen-serve-factory-tests)
    add_test(NAME strix-qwen-serve-factory-tests COMMAND strix-qwen-serve-factory-tests "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(strix-qwen-serve-factory-tests PROPERTIES TIMEOUT 30 LABELS "cpu;serve;qwen")
endif()

function(strix_link_qwen_serve)
    if(TARGET strix-hip-decode-token)
        # Compile the native setup once with the decoder's existing runtime and flags.
        add_library(strix_qwen_hip_decode STATIC apps/hip_decode_token_main.cpp)
        target_include_directories(strix_qwen_hip_decode PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
        target_link_libraries(strix_qwen_hip_decode PRIVATE strix_hip_runtime strix_serve strix_serve_json
            strix_assets strix_qwen_model_description strix_device_profile strix_qwen_http_session_owner
            strix_qwen_serve_options)
        target_compile_options(strix_qwen_hip_decode PRIVATE
            "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/include/strix/hip_runtime_telemetry.hpp\"")
        strix_target_warnings(strix_qwen_hip_decode)
        target_link_libraries(strix_qwen_serve_options PRIVATE strix_hip_runtime)
        target_link_libraries(strix-hip-decode-token PRIVATE strix_qwen_hip_decode)
        target_link_libraries(strix_session_shell PRIVATE strix_qwen_hip_decode)
        target_compile_definitions(strix_session_shell PRIVATE STRIX_SERVE_QWEN=1)
        if(TARGET strix-qwen-serve-factory-tests)
            target_link_libraries(strix-qwen-serve-factory-tests PRIVATE strix_hip_runtime)
            target_compile_definitions(strix-qwen-serve-factory-tests PRIVATE STRIX_EXPECT_QWEN_SERVICE=1)
        endif()
        # Every target that links the HIP runtime here needs the explicit architecture list (offline builds have no
        # detected default).
        if(DEFINED STRIX_HIP_ARCHS)
            foreach(t strix_qwen_serve_options strix_qwen_hip_decode strix-qwen-serve-adapter-link-tests
                    strix-qwen-serve-factory-tests)
                if(TARGET ${t})
                    set_target_properties(${t} PROPERTIES HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                endif()
            endforeach()
        endif()
    endif()
    if(TARGET strix-hip-mtp-model-qualification)
        target_link_libraries(strix-hip-mtp-model-qualification PRIVATE strix_qwen_serve_options)
    endif()
    if(TARGET strix-http-batch-session-tests)
        target_link_libraries(strix-http-batch-session-tests PRIVATE
            strix_qwen_http_session_owner strix_qwen_serve_options)
    endif()
endfunction()
cmake_language(DEFER CALL strix_link_qwen_serve)
