# Inkling 2 image tower on the shared vision stack: the family adapter (strix_inkling2_vision) over the checkpoint's own
# safetensors (LookupWeights) and the real-file gate tool against tools/vision/inkling2_vision_ref.py dumps (manual tool,
# never run by CMake or CTest).
if(TARGET strix_vision_encoder AND TARGET strix_vision_preprocess AND TARGET strix_serve AND TARGET strix-weight)
    add_library(strix_inkling2_vision STATIC "${CMAKE_SOURCE_DIR}/src/inkling2/inkling2_vision.cpp")
    target_include_directories(strix_inkling2_vision PUBLIC "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(strix_inkling2_vision PUBLIC strix_vision_encoder strix_vision_preprocess strix_core strix_serve
        strix-weight)
    if(NOT MSVC)
        # Rows are cache content: no fused multiply-add in positions, folding or the projector activation.
        target_compile_options(strix_inkling2_vision PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    strix_target_warnings(strix_inkling2_vision)

    add_executable(strix-inkling2-vision-gate "${CMAKE_SOURCE_DIR}/tests/inkling2_vision_gate.cpp")
    target_link_libraries(strix-inkling2-vision-gate PRIVATE strix_inkling2_vision)
    strix_target_warnings(strix-inkling2-vision-gate)
    if(TARGET strix_vision_encoder_hip)
        add_executable(strix-inkling2-vision-gate-hip "${CMAKE_SOURCE_DIR}/tests/inkling2_vision_gate.cpp")
        target_compile_definitions(strix-inkling2-vision-gate-hip PRIVATE STRIX_INKLING2_VISION_GATE_HIP=1)
        target_link_libraries(strix-inkling2-vision-gate-hip PRIVATE strix_inkling2_vision strix_vision_encoder_hip)
        strix_target_warnings(strix-inkling2-vision-gate-hip)
    endif()
endif()

strix_servespec_host_test(NAME inkling2-vision
    SOURCES tests/inkling2_vision_tests.cpp
    LIBS strix_inkling2_vision)
strix_servespec_host_test(NAME inkling2-vision-adapter
    SOURCES tests/inkling2/vision_adapter_test.cpp
    LIBS strix_inkling2_vision)
if(TARGET inkling2-vision AND NOT MSVC)
    target_compile_options(inkling2-vision PRIVATE -ffp-contract=off -fno-fast-math)
endif()
