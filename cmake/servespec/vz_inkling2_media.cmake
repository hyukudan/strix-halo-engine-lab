# After vise2e_media.cmake (strix_vision_api_media): fragments are included in name order.
# Image input of the Inkling 2 Mini CLIs (apps/inkling2_mini_media.hpp): the shared serial media owner over this
# adapter, encoding on the shared device tower of the backend (vision::Encoder / vision::CudaEncoder). The HIP
# resident decoder always splices through the shared HIP kernel.
if(TARGET strix-inkling2-mini-hip AND TARGET strix_embed_splice_hip)
    target_link_libraries(strix-inkling2-mini-hip PRIVATE strix_embed_splice_hip)
endif()
if(TARGET strix_inkling2_vision AND TARGET strix_vision_api_media)
    if(TARGET strix-inkling2-mini-hip AND TARGET strix_vision_encoder_hip)
        target_link_libraries(strix-inkling2-mini-hip PRIVATE strix_inkling2_vision strix_vision_api_media strix_vision_encoder_hip)
        target_compile_definitions(strix-inkling2-mini-hip PRIVATE STRIX_INKLING2_MEDIA=1)
    endif()
    if(TARGET strix-inkling2-mini-cuda AND TARGET strix_vision_encoder_cuda)
        target_link_libraries(strix-inkling2-mini-cuda PRIVATE strix_inkling2_vision strix_vision_api_media strix_vision_encoder_cuda)
        target_compile_definitions(strix-inkling2-mini-cuda PRIVATE STRIX_INKLING2_MEDIA=1)
    endif()
endif()
