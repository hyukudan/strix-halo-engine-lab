include_guard(GLOBAL)

if(BUILD_TESTING OR STRIX_BACKEND_VULKAN OR STRIX_BACKEND_METAL)
    add_library(strix_gpu_port STATIC src/core/gpu_port.cpp)
    target_include_directories(strix_gpu_port PUBLIC src)
    target_link_libraries(strix_gpu_port PUBLIC strix_session)
    strix_target_warnings(strix_gpu_port)
endif()

if(BUILD_TESTING)
    add_executable(strix-gpu-port-host-tests tests/gpu_port_host_tests.cpp
        src/vulkan/core/backend.cpp src/metal/core/backend.cpp)
    target_link_libraries(strix-gpu-port-host-tests PRIVATE strix_gpu_port)
    strix_target_warnings(strix-gpu-port-host-tests)
    foreach(_case capabilities refusal resources)
        add_test(NAME strix-gpu-port-${_case} COMMAND strix-gpu-port-host-tests ${_case})
        set_tests_properties(strix-gpu-port-${_case} PROPERTIES LABELS "cpu;backend")
    endforeach()
endif()
