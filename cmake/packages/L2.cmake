if(BUILD_TESTING)
    add_test(NAME strix-cuda-source-home
        COMMAND ${CMAKE_COMMAND} -DSTRIX_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_core/source_home.cmake)
    set_tests_properties(strix-cuda-source-home PROPERTIES LABELS "cpu;layering")
endif()
