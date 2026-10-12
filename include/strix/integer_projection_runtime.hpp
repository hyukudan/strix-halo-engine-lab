#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/integer_projection_execution.hpp"
#include "strix/launch_dispatch.hpp"

namespace strix::integer_projection {
enum class RuntimeInput { bf16, fp32 };
struct RuntimeConfig {
    bool experimental_opt_in=false;
    std::uint32_t hidden_size=0, vocabulary_size=0, tile_columns=1024;
    RuntimeInput input=RuntimeInput::bf16;
    WeightEncoding weights=WeightEncoding::iq4_xs;
    low_precision::ActivationEncoding activations=low_precision::ActivationEncoding::symmetric_a8;
    Instruction instruction=Instruction::iu8;
    DeviceIdentity device;
    std::string approved_numerical_identity;
    std::uint64_t maximum_internal_bytes=0, maximum_retained_bytes=0;
};
[[nodiscard]] std::string runtime_numerical_identity(const RuntimeConfig&);
[[nodiscard]] std::string runtime_kernel_id(const RuntimeConfig&);
// Registers only an experimental, non-graph decode variant. No qualification
// parameter or default catalog selection can promote this alternate target.
void register_integer_projection_runtime_variant(const ModelContract&,
    const ExecutionBindingRequest&,const RuntimeConfig&,kernels::KernelRegistry&);
struct RuntimeAccounting {
    std::uint64_t checkpoint_bytes=0, internal_peak_bytes=0, staged_logits_bytes=0;
    std::uint64_t tile_count=0, preparation_elements=0, repacked_groups=0;
    std::uint32_t internal_allocation_count=12;
};
struct RuntimeOwners {
    std::shared_ptr<const AllocationPin> arena, weights;
};
struct RuntimeSnapshot {
    AllocationBinding arena, weights;
    std::uint64_t retained_bytes=0;
};
class RuntimePlan {
public:
    [[nodiscard]] static RuntimePlan bind(const CompiledSerialExecution&,
        const ExecutionRuntimeResources&,const RuntimeConfig&,const RuntimeOwners&);
    [[nodiscard]] const RuntimeConfig& config()const noexcept{return config_;}
    [[nodiscard]] const RuntimeAccounting& accounting()const noexcept{return accounting_;}
    [[nodiscard]] const std::string& identity()const noexcept{return identity_;}
    [[nodiscard]] std::uint64_t implementation_identity()const noexcept{return implementation_identity_;}
    [[nodiscard]] std::uint64_t execution_identity()const noexcept{return execution_identity_;}
    [[nodiscard]] std::uint32_t step_id()const noexcept{return step_id_;}
    [[nodiscard]] const ExecutionBufferAddress& input()const noexcept{return input_;}
    [[nodiscard]] const ExecutionBufferAddress& output()const noexcept{return output_;}
    [[nodiscard]] const RuntimeTensorBinding& checkpoint()const noexcept{return checkpoint_;}
    [[nodiscard]] ExecutionPlan tile(std::uint32_t first_column,std::uint64_t operation_generation)const;
    [[nodiscard]] RuntimeSnapshot validate_dispatch(const ExecutionDispatchContext&,const RuntimeOwners&)const;
    void validate_snapshot(const RuntimeOwners&,const RuntimeSnapshot&)const;
private:
    RuntimeConfig config_;
    RuntimeAccounting accounting_;
    std::string identity_;
    std::uint64_t implementation_identity_=0, execution_identity_=0, manifest_identity_=0, access_identity_=0;
    std::uint32_t step_id_=0, tensor_index_=0;
    ModelContentIdentity content_identity_;
    ExecutionMemoryBinding memory_;
    RuntimeTensorBinding checkpoint_;
    ManifestTensor tensor_;
    ExecutionBufferAddress input_,output_;
    AllocationBinding arena_owner_,weight_owner_;
};

// Exact integer-only IEEE binary32 -> binary64 widening, including every
// subnormal and signed zero. Nonfinite encodings stay nonfinite for rejection.
[[nodiscard]]
#if defined(__HIPCC__)
__host__ __device__
#endif
inline std::uint64_t binary64_widen_bits(std::uint32_t bits) {
    const auto sign=static_cast<std::uint64_t>(bits>>31U)<<63U;
    const auto exponent=(bits>>23U)&255U, fraction=bits&0x007fffffU;
    if(exponent==255U)return sign|0x7ff0000000000000ULL|(static_cast<std::uint64_t>(fraction)<<29U);
    if(exponent)return sign|(static_cast<std::uint64_t>(exponent+896U)<<52U)|(static_cast<std::uint64_t>(fraction)<<29U);
    if(!fraction)return sign;
    unsigned top=0;auto remaining=fraction;
    while(remaining>>1U){++top;remaining>>=1U;}
    return sign|(static_cast<std::uint64_t>(top+874U)<<52U)|
        (static_cast<std::uint64_t>(fraction^(1U<<top))<<(52U-top));
}
[[nodiscard]]
#if defined(__HIPCC__)
__host__ __device__
#endif
inline std::uint64_t binary64_widen_half_bits(std::uint16_t bits) {
    const auto sign=static_cast<std::uint64_t>(bits>>15U)<<63U;
    const auto exponent=(static_cast<unsigned>(bits)>>10U)&31U, fraction=static_cast<unsigned>(bits)&1023U;
    if(exponent==31U)return sign|0x7ff0000000000000ULL|(static_cast<std::uint64_t>(fraction)<<42U);
    if(exponent)return sign|(static_cast<std::uint64_t>(exponent+1008U)<<52U)|(static_cast<std::uint64_t>(fraction)<<42U);
    if(!fraction)return sign;
    unsigned top=0;auto remaining=fraction;
    while(remaining>>1U){++top;remaining>>=1U;}
    return sign|(static_cast<std::uint64_t>(top+999U)<<52U)|
        (static_cast<std::uint64_t>(fraction^(1U<<top))<<(52U-top));
}
// Integer-only IEEE binary64 -> binary32 round-to-nearest/ties-even conversion.
// This gives the HIP bridge explicit gradual-subnormal and signed-zero behavior
// independently of caller FENV and compiler float32 flush modes.
[[nodiscard]]
#if defined(__HIPCC__)
__host__ __device__
#endif
inline std::uint32_t binary32_rne_bits(std::uint64_t bits) {
    const auto sign=static_cast<std::uint32_t>(bits>>63U)<<31U;
    const auto exponent=static_cast<unsigned>((bits>>52U)&2047U);
    const auto fraction=bits&0x000fffffffffffffULL;
    if(exponent==2047U)return sign|0x7f800000U|(fraction?0x00400000U:0U);
    if(exponent==0U)return sign;
    int e=static_cast<int>(exponent)-1023;
    if(e < -150)return sign;
    if(e > 127)return sign|0x7f800000U;
    const auto significand=fraction|0x0010000000000000ULL;
    const unsigned shift=e>=-126?29U:static_cast<unsigned>(-e-97);
    auto rounded=significand>>shift;
    const auto remainder=significand&((1ULL<<shift)-1ULL), halfway=1ULL<<(shift-1U);
    if(remainder>halfway||(remainder==halfway&&(rounded&1U)))++rounded;
    if(e < -126)return sign|static_cast<std::uint32_t>(rounded);
    if(rounded==0x01000000ULL){rounded>>=1U;++e;}
    if(e>127)return sign|0x7f800000U;
    return sign|(static_cast<std::uint32_t>(e+127)<<23U)|static_cast<std::uint32_t>(rounded&0x007fffffULL);
}
} // namespace strix::integer_projection
