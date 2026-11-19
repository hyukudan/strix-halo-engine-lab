# Hy4-flash serve binding (strix-serve --model hy4_flash): host-only family options, shared-API binding, ServiceFactory and
# GGUF admission. Included by strix_session_link_services (cmake/StrixSession.cmake), which links it into the serve shell.
# No device code: create() refuses until a device binding (hy4_hipbind / hy4_cudabind) exists.
if(TARGET strix_serve AND TARGET strix-weight AND TARGET strix_session_runtime AND TARGET strix_core
   AND NOT TARGET strix_hy4_session_service)
    add_library(strix_hy4_session_service STATIC "${_strix_session_root}/src/hy4/serve/session_service.cpp")
    target_include_directories(strix_hy4_session_service PUBLIC "${_strix_session_root}/src")
    target_link_libraries(strix_hy4_session_service PUBLIC strix_session_runtime strix_serve strix_core strix-weight)
    strix_session_warnings(strix_hy4_session_service)
    if(BUILD_TESTING)
        add_executable(hy4_serve_tests "${_strix_session_root}/tests/hy4_serve_tests.cpp")
        target_link_libraries(hy4_serve_tests PRIVATE strix_hy4_session_service strix_session_shell)
        strix_session_warnings(hy4_serve_tests)
        add_test(NAME hy4-serve COMMAND hy4_serve_tests)
        set_tests_properties(hy4-serve PROPERTIES TIMEOUT 120 LABELS "cpu;serve;hy4")
    endif()
endif()
