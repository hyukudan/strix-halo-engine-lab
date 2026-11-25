# After vise2e_media.cmake (strix_vision_api_media): fragments are included in name order.
# Image input of the Inkling 2 Mini CLIs (apps/inkling2_mini_media.hpp): the shared serial media owner over this
# adapter, encoding on the shared device tower of the backend (vision::Encoder / vision::CudaEncoder). The HIP
# resident decoder always splices through the shared HIP kernel: MiniResidentDecoder::append calls splice_rows, so
# EVERY HIP target that instantiates the decoder (the CLI, the served decoder, the device tests that prefill) links
# strix_embed_splice_hip. A new HIP decoder user joins this list.
if(TARGET strix_embed_splice_hip)
    foreach(_ink_decoder_user strix-inkling2-mini-hip strix-hip-inkling2-mini-verify-rows-test
                              strix-hip-inkling2-mini-dspark-test strix-hip-inkling2-mini-screened-scores-test)
        if(TARGET ${_ink_decoder_user})
            target_link_libraries(${_ink_decoder_user} PRIVATE strix_embed_splice_hip)
        endif()
    endforeach()
    # The served HIP decoder (p7_inkling2_serve.cmake) instantiates the same resident decoder, splice included.
    if(TARGET strix_inkling2_mini_serve_device)
        get_target_property(_ink_serve_sources strix_inkling2_mini_serve_device SOURCES)
        if(_ink_serve_sources MATCHES "\\.hip")
            target_link_libraries(strix_inkling2_mini_serve_device PRIVATE strix_embed_splice_hip)
        endif()
    endif()
endif()

strix_servespec_host_test(NAME inkling2-vision-media
    SOURCES tests/inkling2/vision_media_test.cpp
    LIBS strix_inkling2_vision strix_vision_api_media strix_serve)
if(TARGET inkling2-vision-media)
    target_include_directories(inkling2-vision-media PRIVATE "${CMAKE_SOURCE_DIR}/apps")
    target_compile_definitions(inkling2-vision-media PRIVATE "STRIX_VISION_FIXTURE_DIR=\"${CMAKE_SOURCE_DIR}/tests/vision/fixtures\"")
endif()
if(TARGET strix_inkling2_vision AND TARGET strix_vision_api_media)
    if(TARGET strix-inkling2-mini-hip AND TARGET strix_vision_encoder_hip)
        target_link_libraries(strix-inkling2-mini-hip PRIVATE strix_inkling2_vision strix_vision_api_media strix_vision_encoder_hip)
        target_compile_definitions(strix-inkling2-mini-hip PRIVATE STRIX_INKLING2_MEDIA=1)
    endif()
    if(TARGET strix-inkling2-mini-cpu)  # the host reference of the device answers (FP32 host tower)
        target_link_libraries(strix-inkling2-mini-cpu PRIVATE strix_inkling2_vision strix_vision_api_media)
        target_compile_definitions(strix-inkling2-mini-cpu PRIVATE STRIX_INKLING2_MEDIA=1)
    endif()
    if(TARGET strix-inkling2-mini-cuda AND TARGET strix_vision_encoder_cuda)
        target_link_libraries(strix-inkling2-mini-cuda PRIVATE strix_inkling2_vision strix_vision_api_media strix_vision_encoder_cuda)
        target_compile_definitions(strix-inkling2-mini-cuda PRIVATE STRIX_INKLING2_MEDIA=1)
    endif()
endif()
