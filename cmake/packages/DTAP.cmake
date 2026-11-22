# Shared position-indexed BF16 row store of step programs (drafter feature taps): device fixture.
# Explicit operator execution (RTX kit); CTest keeps the CPU targets only.
if(TARGET strix_exec_cuda_decode_catalog AND TARGET strix_cuda_core_kernels)
    add_executable(strix-step-row-store-cuda EXCLUDE_FROM_ALL tests/cuda_core/step_row_store_cuda.cu)
    target_include_directories(strix-step-row-store-cuda PRIVATE include src)
    target_link_libraries(strix-step-row-store-cuda PRIVATE strix_cuda_core_kernels)
    set_target_properties(strix-step-row-store-cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "120a-real")
    target_compile_options(strix-step-row-store-cuda PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
    if(MSVC)
        target_compile_options(strix-step-row-store-cuda PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
    endif()
endif()
