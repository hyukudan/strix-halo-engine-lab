if(TARGET strix-inkling2-program-tests)
    target_link_libraries(strix-inkling2-program-tests PRIVATE strix_exec_catalog)
endif()

option(STRIX_EPI_CUDA_FIXTURE "Build the CUDA decode epilogue conformance fixture" OFF)
if(STRIX_EPI_CUDA_FIXTURE)
    if(NOT TARGET strix_exec_cuda_decode_catalog)
        message(FATAL_ERROR "STRIX_EPI_CUDA_FIXTURE requires STRIX_CUDA_CORE and STRIX_EXEC_CUDA")
    endif()
    add_executable(strix-cuda-epilogue-fixture tests/cuda_core/epilogue_fixture.cu)
    target_include_directories(strix-cuda-epilogue-fixture PRIVATE include src)
    target_link_libraries(strix-cuda-epilogue-fixture PRIVATE strix_exec_cuda_decode_catalog)
    set_target_properties(strix-cuda-epilogue-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 120a-real)
    target_compile_options(strix-cuda-epilogue-fixture PRIVATE
        --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
    if(MSVC)
        target_compile_options(strix-cuda-epilogue-fixture PRIVATE -Xcompiler=/W4,/WX,/EHsc,/Zc:preprocessor,/utf-8)
    else()
        target_compile_options(strix-cuda-epilogue-fixture PRIVATE -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-Werror)
    endif()
    # The operator runs the device fixture explicitly.
endif()
