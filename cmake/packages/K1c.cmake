# The formats fixture is built on request and run by the device operator.
if(TARGET strix_cuda_core_kernels)
    add_executable(cuda_core_rows_formats_test EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/tests/rows/rows_formats_test.cu")
    target_link_libraries(cuda_core_rows_formats_test PRIVATE strix_cuda_core_kernels)
    target_include_directories(cuda_core_rows_formats_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set_target_properties(cuda_core_rows_formats_test PROPERTIES CUDA_ARCHITECTURES "120a-real")
    target_compile_options(cuda_core_rows_formats_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(cuda_core_rows_formats_test PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
    endif()
endif()
