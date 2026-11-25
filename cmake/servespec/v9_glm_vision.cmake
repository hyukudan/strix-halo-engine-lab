# GLM-5.3 Flash image tower on the shared vision stack: the family adapter (strix_glm5_vision), its host test and the
# host tool that turns an image into an embed rows file (strix-glm5-vision; FP32 reference tower, no GPU).
if(TARGET strix_vision_gguf AND TARGET strix_vision_preprocess AND TARGET strix_model_data)
    add_library(strix_glm5_vision STATIC "${CMAKE_SOURCE_DIR}/src/glm/vision/glm_vision.cpp")
    target_include_directories(strix_glm5_vision PUBLIC "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(strix_glm5_vision PUBLIC strix_vision_gguf strix_vision_preprocess strix_model_data)
    if(NOT MSVC)
        # Rows are cache content and both ranks of a pair splice them: no fused multiply-add anywhere in the adapter.
        target_compile_options(strix_glm5_vision PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    strix_target_warnings(strix_glm5_vision)

    add_executable(strix-glm5-vision "${CMAKE_SOURCE_DIR}/apps/glm_vision_main.cpp")
    target_link_libraries(strix-glm5-vision PRIVATE strix_glm5_vision)
    strix_target_warnings(strix-glm5-vision)
endif()

strix_servespec_host_test(NAME glm5-vision
    SOURCES tests/glm_vision_tests.cpp src/session/embed_rows.cpp
    LIBS strix_glm5_vision)
if(TARGET glm5-vision)
    if(NOT MSVC)
        target_compile_options(glm5-vision PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    set_tests_properties(glm5-vision PROPERTIES LABELS "cpu;glm")
endif()

# Image rows in the GLM prefill (Engine::set_embed_rows): the engine binds the shared device splice; which rows a rank
# writes under sequence-parallel prefill is host code with its own test.
if(TARGET strix_glm5_engine AND TARGET strix_embed_splice_hip)
    target_link_libraries(strix_glm5_engine PUBLIC strix_embed_splice_hip)
endif()
strix_servespec_host_test(NAME glm5-embed-splice
    SOURCES tests/glm_embed_splice_cpu_tests.cpp
    LIBS strix_model_data)
if(TARGET glm5-embed-splice)
    target_include_directories(glm5-embed-splice PRIVATE "${CMAKE_SOURCE_DIR}/src")
    set_tests_properties(glm5-embed-splice PROPERTIES LABELS "cpu;glm;exactness")
endif()
