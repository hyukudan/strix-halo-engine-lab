#pragma once
// Independent CPU mirror of the declared HC-F RMS/dot trees. No engine calls.
// rsqrt/expf are intentionally not treated as CPU/native bitwise equivalents.
#include "q4k_a8_oracle.hpp"
#include <span>

namespace hc_deferred_norm_oracle {
inline constexpr unsigned hidden=2560,streams=4,width=10240,rank=320;
inline float bf16(std::uint16_t value) {return std::bit_cast<float>(static_cast<std::uint32_t>(value)<<16U);}
inline float rounded_add(float x,float y) {volatile float value=x+y;return value;}
inline float rounded_mul(float x,float y) {volatile float value=x*y;return value;}
inline float rounded_divide(float x,float y) {volatile float value=x/y;return value;}
inline float butterfly(std::array<float,32> lanes) {
    for (unsigned offset=16;offset;offset/=2U) {
        const auto before=lanes;
        for (unsigned lane=0;lane<32;++lane) lanes[lane]=rounded_add(before[lane],before[lane^offset]);
    }
    return lanes[0];
}
inline void input_geometry(std::span<const std::uint16_t> raw,std::span<const float> norm) {
    if (raw.size()!=width || norm.size()!=width) throw std::invalid_argument("HC-F oracle exact geometry");
}
inline std::array<float,4> rms_denominators(std::span<const std::uint16_t> raw,float epsilon) {
    if (raw.size()!=width || !std::isfinite(epsilon) || epsilon<=0) throw std::invalid_argument("HC-F RMS geometry/epsilon");
    std::array<float,16> tiles{};
    for (unsigned tile=0;tile<16;++tile) {
        std::array<float,32> lanes{};
        for (unsigned lane=0;lane<32;++lane) {
            float value=0;
            for (unsigned part=0;part<5;++part) for (unsigned j=0;j<4;++j) {
                const float x=bf16(raw[tile*640U+part*128U+lane*4U+j]);
                value=std::fma(x,x,value);
            }
            lanes[lane]=value;
        }
        tiles[tile]=butterfly(lanes);
    }
    std::array<float,4> result{};
    for (unsigned stream=0;stream<4;++stream) {
        float total=0;
        for (unsigned tile=0;tile<4;++tile) total=rounded_add(total,tiles[stream*4U+tile]);
        result[stream]=rounded_add(rounded_divide(total,2560.0F),epsilon);
    }
    return result;
}
inline float q8(std::span<const std::byte> weights,unsigned row,unsigned row_width,unsigned feature) {
    if (row_width%32U || weights.size()<(static_cast<std::size_t>(row)+1U)*(row_width/32U)*34U || feature>=row_width)
        throw std::invalid_argument("HC-F Q8 oracle bounds");
    const auto block=(static_cast<std::size_t>(row)*(row_width/32U)+feature/32U)*34U;
    const auto lo=std::to_integer<unsigned>(weights[block]),hi=std::to_integer<unsigned>(weights[block+1U]);
    const float scale=q4k_a8_oracle::half(static_cast<std::uint16_t>(lo|(hi<<8U)));
    const auto byte=std::to_integer<unsigned>(weights[block+2U+feature%32U]);
    const int code=byte<128U?static_cast<int>(byte):static_cast<int>(byte)-256;
    return rounded_mul(scale,static_cast<float>(code));
}
inline std::vector<float> scaled(std::span<const std::uint16_t> raw,std::span<const float> norm,
    std::span<const std::byte> down,std::span<const float> injection,std::span<const float> inverse,
    bool omit_gain=false,bool plain_cross_tile=false) {
    input_geometry(raw,norm);
    if (down.size()!=3481600U || inverse.size()!=4 || (!injection.empty() && injection.size()!=40960U))
        throw std::invalid_argument("HC-F projection oracle geometry");
    std::vector<float> result(rank+(injection.empty()?0U:streams));
    for (unsigned row=0;row<result.size();++row) {
        std::array<float,16> partial{};
        for (unsigned tile=0;tile<16;++tile) {
            std::array<float,32> lanes{};
            for (unsigned lane=0;lane<32;++lane) {
                float sum=0;
                for (unsigned part=0;part<5;++part) for (unsigned j=0;j<4;++j) {
                    const unsigned feature=tile*640U+part*128U+lane*4U+j;
                    const float x=omit_gain?bf16(raw[feature]):rounded_mul(bf16(raw[feature]),norm[feature]);
                    const float coefficient=row<rank?q8(down,row,width,feature):injection[(row-rank)*width+feature];
                    sum=std::fma(x,coefficient,sum);
                }
                lanes[lane]=sum;
            }
            partial[tile]=butterfly(lanes);
        }
        float total=0;
        for (unsigned tile=0;tile<16;++tile)
            total=plain_cross_tile?rounded_add(total,rounded_mul(partial[tile],inverse[tile/4U])):
                std::fma(partial[tile],inverse[tile/4U],total);
        result[row]=rounded_divide(total,4.0F);
    }
    return result;
}
// Exact sigmoid(0)=1/2 case. Use a zero-Q8 up matrix to isolate the newly
// recomputed normalized-value path from expf and unchanged projection arithmetic.
inline std::vector<std::uint16_t> output_with_zero_up(std::span<const std::uint16_t> raw,
    std::span<const float> norm,std::span<const float> inverse) {
    input_geometry(raw,norm);if (inverse.size()!=4) throw std::invalid_argument("HC-F inverse geometry");
    std::vector<std::uint16_t> result(hidden);
    for (unsigned feature=0;feature<hidden;++feature) {
        float mixed=0;
        for (unsigned stream=0;stream<streams;++stream) {
            const auto index=stream*hidden+feature;
            const float n=rounded_mul(rounded_mul(bf16(raw[index]),inverse[stream]),norm[index]);
            mixed=std::fma(n,0.5F,mixed);
        }
        const auto value=std::bit_cast<std::uint32_t>(rounded_divide(mixed,4.0F));
        result[feature]=static_cast<std::uint16_t>((value+0x7fffU+((value>>16U)&1U))>>16U);
    }
    return result;
}
} // namespace hc_deferred_norm_oracle
