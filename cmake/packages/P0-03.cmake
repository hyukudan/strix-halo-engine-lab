if(BUILD_TESTING)
    add_test(NAME strix-cuda-core-no-model-includes
        COMMAND ${CMAKE_COMMAND} -DSTRIX_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_core/no_model_includes.cmake)
    set_tests_properties(strix-cuda-core-no-model-includes PROPERTIES LABELS "cpu;layering")
endif()
