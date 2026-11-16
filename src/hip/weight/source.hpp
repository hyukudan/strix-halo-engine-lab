#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
namespace strix::weight { struct Status; struct WeightView; }
namespace strix::weight::hip {
struct Q8_0Tag {}; struct RawF16Tag {}; struct Q6P32Tag {};
struct NativeScalarTag {}; struct SignedA8Tag {}; struct F16TileTag {};
struct Bf16TileTag {}; struct Fp8TileTag {}; struct Nvfp4TileTag {};
struct Gfx1151 {}; struct LegacyRowsArithmetic {}; struct DenseQ8A8F16V1 {}; struct RawF16Arithmetic {};
struct MatrixView {
    const std::uint8_t* blocks;
    std::uint64_t row_stride_bytes, expert_stride_bytes;
    std::uint32_t k,n;
};
struct LaunchShape {
    std::uint64_t m,n,k,row_stride_bytes,activation_stride,output_stride,k_offset_bytes;
};
Status validate_q8_rows(const WeightView&,const LaunchShape&);
Status validate_q6_rows(const WeightView&,const LaunchShape&);
struct PacketSource {
    using View=MatrixView;
    template<class Core> __device__ __forceinline__ static void load_fragment(const View&,
        const typename Core::Coord& coord,typename Core::BFragment& fragment,typename Core::Scratch&) {
        fragment.load(coord.rows,coord.unit);
    }
};
template<class Storage,class Protocol,class Arch,class Arithmetic> struct OperandSource;
template<class Source,class Arithmetic,class ExistingCore> struct BoundCore:ExistingCore {
    using WeightSource=Source; using NumericContract=Arithmetic;
};
}  // namespace strix::weight::hip
