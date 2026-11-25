# Included after vise2e_media.cmake (name order), which defines the shared media bridge.
# --vision FILE of the GLM pair server: the family adapter, the weight reader and the shared media bridge. The host
# half (glm_serve_vision.hpp: the stack and the per-prompt embed frame / splice resolution) has its own host test.
if(TARGET strix-glm5-serve)
    foreach(strix_glm_vision_lib strix_glm5_vision strix_vision_api_media)
        if(TARGET ${strix_glm_vision_lib})
            target_link_libraries(strix-glm5-serve PRIVATE ${strix_glm_vision_lib})
        endif()
    endforeach()
endif()
strix_servespec_host_test(NAME glm5-serve-vision
    SOURCES tests/glm_serve_vision_tests.cpp
    LIBS strix_glm5_vision strix_vision_api_media strix_serve
    TIMEOUT 120)
if(TARGET glm5-serve-vision)
    target_include_directories(glm5-serve-vision PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(glm5-serve-vision PRIVATE "STRIX_VISION_FIXTURE_DIR=\"${CMAKE_SOURCE_DIR}/tests/vision/fixtures\"")
    set_tests_properties(glm5-serve-vision PROPERTIES LABELS "cpu;glm;session")
endif()
