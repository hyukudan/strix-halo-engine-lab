if(BUILD_TESTING)
    add_executable(strix-k10c-lookup-host-test tests/cuda_core/lookup_host_test.cpp)
    target_include_directories(strix-k10c-lookup-host-test PRIVATE src)
    target_link_libraries(strix-k10c-lookup-host-test PRIVATE strix_exec_catalog)
    target_compile_features(strix-k10c-lookup-host-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-k10c-lookup-host-test)
    if(MSVC)
        target_compile_options(strix-k10c-lookup-host-test PRIVATE /fp:precise)
    else()
        target_compile_options(strix-k10c-lookup-host-test PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-k10c-lookup-host-test COMMAND strix-k10c-lookup-host-test)
    set_tests_properties(strix-k10c-lookup-host-test PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_K10C_CUDA_FIXTURE "Build the K10c lookup conformance fixture for sm_120a" OFF)
if(STRIX_K10C_CUDA_FIXTURE)
    enable_language(CUDA)

    # Follow K11c: compile the incumbent private kernel bodies directly from
    # their sources so this fixture can link without any production engine.
    function(strix_k10c_incumbent file first last out)
        set(path "${CMAKE_CURRENT_SOURCE_DIR}/src/${file}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
        file(READ "${path}" source)
        string(FIND "${source}" "${first}" begin)
        if(begin LESS 0)
            message(FATAL_ERROR "K10c fixture: ${file} start changed at ${first}")
        endif()
        string(SUBSTRING "${source}" ${begin} -1 tail)
        string(FIND "${tail}" "${last}" length)
        if(length LESS_EQUAL 0)
            message(FATAL_ERROR "K10c fixture: ${file} end changed at ${last}")
        endif()
        string(SUBSTRING "${tail}" 0 ${length} body)
        set(${out} "${body}" PARENT_SCOPE)
    endfunction()

    set(dense cuda_core/dense/dense.cu)
    strix_k10c_incumbent(${dense} "__global__ void embedding_kernel(" "}  // namespace" dense_kernel)
    strix_k10c_incumbent(${dense} "cudaError_t embedding_rows(" "cudaError_t argmax_rows(" dense_launch)

    set(hc cuda/models/qwen/qwen_hc.cu)
    strix_k10c_incumbent(${hc} "__device__ __forceinline__ std::uint16_t float_to_bf16(" "// sigmoid(x)" hc_math)
    strix_k10c_incumbent(${hc} "__global__ void embedding_kernel(" "__global__ void expand_kernel(" hc_kernel)

    set(mtp cuda/models/qwen/qwen_plemtp.cu)
    strix_k10c_incumbent(${mtp} "__host__ __device__ __forceinline__ std::uint16_t f2bf(" "// Host: hash contract" mtp_round)
    strix_k10c_incumbent(${mtp} "__device__ __forceinline__ float half_at(" "__global__ void mtp_q8_to_q4_kernel(" mtp_half)
    strix_k10c_incumbent(${mtp} "__global__ void mtp_embedding_kernel(" "// 2. mtp_input_norms" mtp_kernel)

    strix_k10c_incumbent(cuda/models/ds41/ds41_core_io.cu "__global__ void embed_kernel(" "// ---- Engram" ds41_kernel)
    set(dspark cuda/models/ds41/ds41_dspark.cu)
    strix_k10c_incumbent(${dspark} "__device__ __forceinline__ float f16_device(" "// Wave-order reduction" dspark_math)
    strix_k10c_incumbent(${dspark} "__global__ __launch_bounds__(256) void embed_block_kernel("
        "__global__ __launch_bounds__(32) void confidence_kernel(" dspark_kernel)

    set(glm cuda/models/glm/producer/runtime.cu)
    strix_k10c_incumbent(${glm} "__device__ float bf16_to_float(" "\n" glm_math)
    strix_k10c_incumbent(${glm} "__global__ void embed(" "__global__ void add(" glm_kernel)

    set(mm cuda/models/minimax/mm_rtx_kernels.cu)
    strix_k10c_incumbent(${mm} "__device__ __forceinline__ float u2f(" "__device__ __forceinline__ std::uint16_t float_to_bf16(" mm_math)
    strix_k10c_incumbent(${mm} "__global__ void embed_kernel(" "// Gemma norm" mm_kernel)

    set(nvfp4 cuda_core/dense/nvfp4.cu)
    strix_k10c_incumbent(${nvfp4} "__device__ __forceinline__ void e2m1_word(" "__device__ __forceinline__ std::uint32_t f32_to_e4m3(" fp4_convert)
    strix_k10c_incumbent(${nvfp4} "__device__ __forceinline__ std::uint32_t hmul2(" "__device__ __forceinline__ void mma16816(" fp4_mul)
    strix_k10c_incumbent(${nvfp4} "__global__ void __launch_bounds__(256) expand_f16_kernel(" "\n\nnamespace {" fp4_expand)

    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/k10c_lookup_incumbents.cuh" CONTENT
"namespace strix::cuda_core {
namespace {
${dense_kernel}
${fp4_convert}${fp4_mul}${fp4_expand}
${dense_launch}
}
namespace incumbent::qwen_hc {
using strix::cuda::qwen_hc::kHidden;
namespace weight = strix::cuda::weight;
${hc_math}${hc_kernel}
}
namespace incumbent::qwen_mtp {
using strix::cuda::qwen_plemtp::kHidden;
namespace weight = strix::cuda::weight;
${mtp_round}${mtp_half}${mtp_kernel}
}
namespace incumbent::ds41 {
using strix::ds41::f16_to_float;
${ds41_kernel}
}
namespace incumbent::dspark {
using namespace strix::ds41::dspark;
${dspark_math}${dspark_kernel}
}
namespace incumbent::glm {
using U16 = std::uint16_t;
using U32 = std::uint32_t;
${glm_math}
${glm_kernel}
}
namespace incumbent::minimax {
using strix::minimax::rtxpf::cuda::kHidden;
${mm_math}${mm_kernel}
}
")

    add_executable(strix-k10c-lookup-cuda-fixture tests/cuda_core/lookup_cuda_fixture.cu src/cuda/core/lookup_cuda.cu)
    target_include_directories(strix-k10c-lookup-cuda-fixture PRIVATE include src "${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties(strix-k10c-lookup-cuda-fixture PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real")
    target_compile_options(strix-k10c-lookup-cuda-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-k10c-lookup-cuda-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8,/fp:precise>)
    else()
        target_compile_options(strix-k10c-lookup-cuda-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math,-ffp-contract=off>)
    endif()
    # Node operator execution only. The CUDA executable is never a CTest test.
endif()
