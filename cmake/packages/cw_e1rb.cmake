if(TARGET strix-inkling2-mini-cuda AND TARGET strix_exec_cuda)
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/apps/inkling2_mini_cuda.cu" PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(strix-inkling2-mini-cuda PRIVATE src/inkling2/inkling2_mini_verify_cli.cu)
endif()
