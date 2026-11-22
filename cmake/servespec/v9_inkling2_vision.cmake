# Inkling 2 Mini image tower binding on the shared vision runtime: tensor names, pad-to-40 geometry, folds and span.
if(TARGET strix_vision_encoder AND TARGET strix_vision_preprocess AND TARGET strix-weight)
    add_library(strix_inkling2_vision STATIC "${CMAKE_SOURCE_DIR}/src/inkling2/inkling2_vision_adapter.cpp")
    target_include_directories(strix_inkling2_vision PUBLIC "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(strix_inkling2_vision PUBLIC strix_vision_encoder strix_vision_preprocess strix-weight)
    if(MSVC)
        target_compile_definitions(strix_inkling2_vision PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix_inkling2_vision)
endif()

strix_servespec_host_test(NAME inkling2-vision-adapter
    SOURCES tests/inkling2/vision_adapter_test.cpp
    LIBS strix_inkling2_vision)
