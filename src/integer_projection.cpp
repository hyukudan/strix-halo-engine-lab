#include "strix/integer_projection.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::integer_projection {
namespace {
std::size_t product(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("integer projection allocation overflow");
    return a * b;
}
void environment() {
    if(!std::numeric_limits<double>::is_iec559 || std::fegetround()!=FE_TONEAREST)
        throw std::invalid_argument("integer projection requires binary64 nearest-even");
    volatile double tiny=std::numeric_limits<double>::denorm_min();
    volatile double two=2.0;
    if(tiny*two==0)throw std::invalid_argument("integer projection requires gradual subnormals");
}
void activation_encoding(low_precision::ActivationEncoding e) {
    if (e != low_precision::ActivationEncoding::symmetric_a8 &&
        e != low_precision::ActivationEncoding::symmetric_a4)
        throw std::invalid_argument("integer projection requires explicit A8 or A4");
}
int signed_code(std::uint32_t code, unsigned bits) {
    const auto half = std::uint32_t{1} << (bits - 1U);
    return code < half ? static_cast<int>(code) :
        static_cast<int>(code) - static_cast<int>(std::uint32_t{1} << bits);
}
}
void validate_shape(Shape s, WeightEncoding e) {
    std::size_t block = 0;
    switch (e) {
    case WeightEncoding::q4_0: block = 32; break;
    case WeightEncoding::iq4_xs: block = 256; break;
    default: throw std::invalid_argument("unknown integer projection weight encoding");
    }
    if (!s.rows || !s.columns || !s.inner || s.inner % block)
        throw std::invalid_argument("nonzero dimensions and exact storage blocks required");
    if (s.rows > UINT32_MAX - 16U || s.columns > UINT32_MAX - 16U || s.inner > UINT32_MAX)
        throw std::overflow_error("integer projection device dimension overflow");
    (void)product(product(s.rows, s.inner), sizeof(double));
    (void)product(product(s.columns, s.inner / 32), 18);
    (void)product(product(product(s.rows, s.columns), s.inner / 32), sizeof(double));
    std::size_t working=0;
    const auto add=[&](std::size_t bytes){
        if(bytes>max_working_bytes || working>max_working_bytes-bytes)
            throw std::length_error("integer projection exceeds 256 MiB laboratory working set");
        working+=bytes;
    };
    add(product(product(s.rows,s.inner),9)); // input f64 + activation codes
    add(product(product(s.rows,s.inner/32),12)); // scales and sums
    add(product(s.columns,s.inner/2));
    add(product(product(s.columns,s.inner/block),e==WeightEncoding::q4_0?18U:136U)); // original packed source
    add(product(product(s.columns,s.inner/32),8));
    add(product(product(s.rows,s.columns),8));
    add(product(product(product(s.rows,s.columns),s.inner/32),4));
    add(s.inner);add(product(s.inner/32,8)); // transient quantized row
}
Weights stage_weights(WeightEncoding e, std::size_t n, std::size_t k,
    std::span<const std::byte> checkpoint) {
    environment();
    validate_shape({1,n,k}, e);
    const std::size_t block = e == WeightEncoding::q4_0 ? 32 : 256;
    const std::size_t bytes = e == WeightEncoding::q4_0 ? 18 : 136;
    if (checkpoint.size() != product(product(n, k / block), bytes))
        throw std::invalid_argument("checkpoint length does not match exact storage shape");
    Weights w{e,n,k,{}, {}};
    w.packed.resize(product(n,k / 2));
    w.scales.resize(product(n,k / 32));
    for (std::size_t column=0; column<n; ++column) {
        for (std::size_t g=0; g<k/32; ++g) {
            const auto offset = (column*(k/block) + g/(block/32))*bytes;
            const auto lo = std::to_integer<unsigned>(checkpoint[offset]);
            const auto hi = std::to_integer<unsigned>(checkpoint[offset+1]);
            const auto d = static_cast<double>(iq4_xs::fp16_bits_to_float(
                static_cast<std::uint16_t>(lo | (hi<<8U))));
            const auto subgroup = g % (block/32);
            double scale = d;
            std::size_t qoffset = offset+2;
            if (e == WeightEncoding::iq4_xs) {
                const auto b = std::span<const std::byte,136>(checkpoint.data()+offset,136);
                scale *= static_cast<int>(iq4_xs::local_scale_code(b,subgroup))-32;
                qoffset = offset+8+subgroup*16;
            }
            if (!std::isfinite(scale)) throw std::invalid_argument("nonfinite packed weight scale");
            w.scales[column*(k/32)+g] = scale;
            for (std::size_t j=0;j<16;++j)
                w.packed[(column*(k/32)+g)*16+j] = std::to_integer<std::uint8_t>(checkpoint[qoffset+j]);
        }
    }
    return w;
}
Activations prepare(Shape s, low_precision::ActivationEncoding e, std::span<const double> input) {
    validate_shape(s,WeightEncoding::q4_0);
    activation_encoding(e);
    if (input.size()!=product(s.rows,s.inner)) throw std::invalid_argument("activation input length mismatch");
    Activations a{s,e,{}, {}, {}};
    a.codes.reserve(input.size());
    for (std::size_t r=0;r<s.rows;++r) {
        const auto row=low_precision::quantize_row(input.subspan(r*s.inner,s.inner),e,32);
        a.codes.insert(a.codes.end(),row.codes.begin(),row.codes.end());
        a.scales.insert(a.scales.end(),row.scales.begin(),row.scales.end());
        for (std::size_t g=0;g<s.inner/32;++g) {
            std::int32_t sum=0;
            for(std::size_t j=0;j<32;++j)
                sum+=signed_code(row.codes[g*32+j],e==low_precision::ActivationEncoding::symmetric_a8?8U:4U);
            a.sums.push_back(sum);
        }
    }
    return a;
}
void validate(const Weights& w,const Activations& a,Instruction instruction) {
    environment();
    validate_shape(a.shape,w.encoding);
    activation_encoding(a.encoding);
    if(instruction!=Instruction::iu8 && instruction!=Instruction::iu4)
        throw std::invalid_argument("unknown matrix instruction");
    if(instruction==Instruction::iu4 && (w.encoding!=WeightEncoding::q4_0 ||
        a.encoding!=low_precision::ActivationEncoding::symmetric_a4))
        throw std::invalid_argument("IU4 requires affine Q4_0 and A4; nonlinear indices are not integers");
    const auto s=a.shape;
    if(w.columns!=s.columns || w.inner!=s.inner || w.packed.size()!=product(s.columns,s.inner/2) ||
        w.scales.size()!=product(s.columns,s.inner/32) || a.codes.size()!=product(s.rows,s.inner) ||
        a.scales.size()!=product(s.rows,s.inner/32) || a.sums.size()!=a.scales.size())
        throw std::invalid_argument("integer projection staging size mismatch");
    for(double scale:w.scales) if(!std::isfinite(scale)) throw std::invalid_argument("nonfinite weight scale");
    for(double scale:a.scales) if(!std::isfinite(scale)||scale<=0) throw std::invalid_argument("invalid activation scale");
    for(std::size_t r=0;r<s.rows;++r) for(std::size_t g=0;g<s.inner/32;++g) {
        int sum=0;
        for(std::size_t j=0;j<32;++j) {
            const auto code=a.codes[r*s.inner+g*32+j];
            if(a.encoding==low_precision::ActivationEncoding::symmetric_a4 && code>15)
                throw std::invalid_argument("invalid A4 staging code");
            const auto q=activation_code(a,r,g*32+j);
            if(q== (a.encoding==low_precision::ActivationEncoding::symmetric_a8?-128:-8))
                throw std::invalid_argument("activation code outside symmetric quantizer range");
            sum+=q;
        }
        if(sum!=a.sums[r*(s.inner/32)+g]) throw std::invalid_argument("tampered activation group sum");
    }
}
std::string numerical_identity(WeightEncoding w,low_precision::ActivationEncoding a,Instruction i) {
    activation_encoding(a);
    validate_shape({1,1,w==WeightEncoding::iq4_xs?256U:32U},w);
    if(i!=Instruction::iu8 && i!=Instruction::iu4) throw std::invalid_argument("unknown matrix instruction");
    if(i==Instruction::iu4 && (w!=WeightEncoding::q4_0 || a!=low_precision::ActivationEncoding::symmetric_a4))
        throw std::invalid_argument("incompatible IU4 interpretation");
    return std::string("strix-integer-projection-v1:")+(w==WeightEncoding::q4_0?"q4_0-u4-z8":"iq4_xs-codebook-i8")+
        ":"+std::string(low_precision::name(a))+":"+(i==Instruction::iu4?"iu4":"iu8")+
        ":gfx1151-w32-g32-i32-no-clamp-i64-correction-f64-scale-left-to-right-no-contract:"+
        std::to_string(low_precision::quantization_identity(a,32));
}
int weight_code(const Weights& w,std::size_t n,std::size_t k) {
    if(w.encoding!=WeightEncoding::q4_0 && w.encoding!=WeightEncoding::iq4_xs)
        throw std::invalid_argument("unknown weight code interpretation");
    if(n>=w.columns || k>=w.inner) throw std::out_of_range("weight coordinate");
    const auto byte=w.packed.at((n*(w.inner/32)+k/32)*16+k%16);
    const auto code=k%32<16?byte&15U:byte>>4U;
    return w.encoding==WeightEncoding::q4_0?static_cast<int>(code):iq4_xs::codebook[code];
}
int activation_code(const Activations& a,std::size_t r,std::size_t k) {
    activation_encoding(a.encoding);
    if(r>=a.shape.rows||k>=a.shape.inner) throw std::out_of_range("activation coordinate");
    if(a.encoding==low_precision::ActivationEncoding::symmetric_a4&&a.codes.at(r*a.shape.inner+k)>15)
        throw std::invalid_argument("invalid A4 code");
    return signed_code(a.codes.at(r*a.shape.inner+k),a.encoding==low_precision::ActivationEncoding::symmetric_a8?8U:4U);
}
Result reference(const Weights& w,const Activations& a,Instruction i) {
    validate(w,a,i);
    const auto s=a.shape; const auto groups=s.inner/32;
    Result result; result.output.resize(product(s.rows,s.columns));
    result.groups.resize(product(result.output.size(),groups));
    for(std::size_t r=0;r<s.rows;++r) for(std::size_t n=0;n<s.columns;++n) {
        double output=0;
        for(std::size_t g=0;g<groups;++g) {
            std::int64_t dot=0;
            for(std::size_t j=0;j<32;++j) dot+=
                static_cast<std::int64_t>(weight_code(w,n,g*32+j)-(w.encoding==WeightEncoding::q4_0?8:0))*
                activation_code(a,r,g*32+j);
            // <=32*127*127=516128; reset each group, never across scales.
            result.groups[(r*s.columns+n)*groups+g]=static_cast<std::int32_t>(dot);
            const double scaled=(w.scales[n*groups+g]*a.scales[r*groups+g])*static_cast<double>(dot);
            output+=scaled;
            if(!std::isfinite(output)) throw std::overflow_error("nonfinite projection output");
        }
        result.output[r*s.columns+n]=output;
    }
    return result;
}
Tile fragment_panel(const Weights& w,const Activations& a,Instruction instruction,
    std::size_t rb,std::size_t nb,std::size_t kb) {
    validate(w,a,instruction);
    if(rb%16||nb%16||kb%16||rb>=a.shape.rows||nb>=w.columns||kb>=w.inner)
        throw std::invalid_argument("invalid WMMA panel origin");
    const unsigned bits=instruction==Instruction::iu4?4U:8U;
    const unsigned per=32U/bits;
    std::array<std::array<std::uint32_t,4>,32> af{},bf{};
    for(std::size_t lane=0;lane<32;++lane) for(unsigned k=0;k<16;++k) {
        const auto r=rb+lane%16, n=nb+lane%16;
        const int av=r<a.shape.rows?activation_code(a,r,kb+k):0;
        const int bv=n<w.columns?weight_code(w,n,kb+k):0;
        const auto mask=(std::uint32_t{1}<<bits)-1U;
        af[lane][k/per]|=(static_cast<std::uint32_t>(av)&mask)<<((k%per)*bits);
        bf[lane][k/per]|=(static_cast<std::uint32_t>(bv)&mask)<<((k%per)*bits);
    }
    Tile out{};
    for(std::size_t lane=0;lane<32;++lane) for(std::size_t v=0;v<8;++v) {
        const auto r=2*v+lane/16, n=lane%16;
        std::int32_t raw=0;
        for(unsigned k=0;k<16;++k) {
            const auto mask=(std::uint32_t{1}<<bits)-1U;
            const auto av=(af[r][k/per]>>((k%per)*bits))&mask;
            const auto bv=(bf[lane][k/per]>>((k%per)*bits))&mask;
            raw+=signed_code(av,bits)*(w.encoding==WeightEncoding::q4_0?static_cast<int>(bv):signed_code(bv,bits));
        }
        out[r*16+n]=raw;
    }
    return out;
}
} // namespace strix::integer_projection
