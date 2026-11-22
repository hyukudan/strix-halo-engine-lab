# Copy test baselines verbatim from the checked-out incumbent source. Keeping
# the source slices in the build directory also avoids pulling in the model
# drivers or GLM's projection dependencies to test its score/select bodies.
function(k5c_slice out file start stop)
    set(file "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/models/${file}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${file}")
    file(READ "${file}" source)
    string(FIND "${source}" "${start}" first)
    string(FIND "${source}" "${stop}" last)
    if(first LESS 0 OR last LESS_EQUAL first)
        message(FATAL_ERROR "K5c fixture cannot locate source slice in ${file}: ${start}")
    endif()
    math(EXPR length "${last} - ${first}")
    string(SUBSTRING "${source}" ${first} ${length} body)
    set(${out} "${${out}}\n${body}" PARENT_SCOPE)
endfunction()

k5c_slice(qwen qwen/qwen_qsa.cu "constexpr unsigned kFull" "__device__ __forceinline__ float bf16f")
k5c_slice(qwen qwen/qwen_qsa.cu "template <unsigned P, unsigned Step = 0U>" "inline constexpr strix::lever::Decl score_tree_lever")
k5c_slice(qwen qwen/qwen_qsa.cu "constexpr unsigned kTreeRows" "constexpr unsigned kG =")
k5c_slice(ds ds41/ds41_attention.cu "struct QPrep {" "__global__ __launch_bounds__(1024) void index_prep_kernel")
k5c_slice(ds ds41/ds41_attention.cu "constexpr unsigned kScoreWarps" "// ---- attention ")
k5c_slice(ds ds41/ds41_fast_index.cu "constexpr unsigned kXWarps" "// ---- keys:")
k5c_slice(ds ds41/ds41_fast_index.cu "// ---- scores" "}  // namespace")
k5c_slice(glm glm/dsa.cu "__device__ U32 ordered(" "__global__ void rms(")
k5c_slice(glm glm/dsa.cu "template<int Kind>\n__device__ void index_dots" "// One CTA per query/chunk")
k5c_slice(mm minimax/mm_rtx_kernels.cu "constexpr float kEps" "__device__ float exp_f(")
k5c_slice(mm minimax/mm_rtx_kernels.cu "__device__ __forceinline__ float shx" "__global__ void embed_kernel(")
k5c_slice(mm minimax/mm_rtx_kernels.cu "__global__ void index_scores_kernel(" "// Sparse attention: one warp")
k5c_slice(mm minimax/mm_rtx_kernels.cu "constexpr unsigned kKs4" "// Sparse attention, shared chunks")

file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/k5c/select_incumbents.inc" CONTENT [=[
namespace strix::cuda::qwen_qsa {
@qwen@
#undef kInf
}
namespace strix::ds41::cuda::attn {
constexpr unsigned kFull = 0xffffffffU;
@ds@
}
namespace strix::glm::rtxpf {
using U32 = std::uint32_t;
using BF = __nv_bfloat16;
using namespace nvcuda;
inline constexpr unsigned dsa_max_tokens = 2051;
@glm@
}
namespace strix::minimax::rtxpf::cuda {
@mm@
}
]=] @ONLY)
