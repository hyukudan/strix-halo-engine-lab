#pragma once
// GSQ weight decode shared by every CUDA expert tier (exact native SIMT through ggml_sources, FAST bf16 MMA, TURBO i8 MMA)
// and by their CPU oracles. Host-compilable: the same expressions run in the host tests.
//
// code() is the reference definition of one weight code (unchanged). word() / scales() are the decode the kernels use:
// word-level and branch-free, bit-identical to code() / the reference scales for every byte pattern
// (tests/cuda/gsq_decode_host_test.cpp sweeps every field value of every type, the full codebooks and all 65,536 FP16
// scales). Why word-level: the four codes of a word share one codebook entry (IQ3_S / IQ3_XXS: one 32-bit grid entry;
// IQ2_S: one 32-bit half of a 64-bit entry), one sign nibble and one field byte, so a word needs one table read and a
// few SWAR operations instead of four table reads and four sign branches; with no branch region left, a kernel can issue
// the loads of several groups together (load() / word(raw, ...) split the field loads from the arithmetic).
#include "ggml_sources.cuh"
#include "strix/weight/formats/ggml.hpp"

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define STRIX_GSQ_HD __host__ __device__ __forceinline__
#else
#define STRIX_GSQ_HD inline
#endif

namespace strix::cuda::weight {

namespace gsq_detail {
// Little-endian 16-bit field at an even offset of a block (every GSQ block size is even, so a 2-byte aligned tensor
// keeps every even field 2-byte aligned on the device).
STRIX_GSQ_HD std::uint32_t ld16(const std::uint8_t* p) {
#ifdef __CUDA_ARCH__
    return *reinterpret_cast<const std::uint16_t*>(p);
#else
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8U);
#endif
}
STRIX_GSQ_HD std::uint32_t ld32(const std::uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16U); }
// Byte permute: result byte n = byte ((sel >> 4 n) & 7) of the 8-byte value {hi, lo}.
STRIX_GSQ_HD std::uint32_t prmt(std::uint32_t lo, std::uint32_t hi, std::uint32_t sel) {
#ifdef __CUDA_ARCH__
    return __byte_perm(lo, hi, sel);
#else
    const std::uint64_t v = (std::uint64_t(hi) << 32U) | lo;
    std::uint32_t r = 0;
    for (unsigned n = 0; n < 4U; ++n) r |= std::uint32_t((v >> (8U * ((sel >> (4U * n)) & 7U))) & 255U) << (8U * n);
    return r;
#endif
}
// Four codebook bytes (each 1..127) with sign bit n of `nib` negating byte n, as two's complement bytes. The multiply
// spreads bit n to bit 8 n without collisions; (g ^ 0xFF) + 1 = -g per byte and cannot carry because g >= 1.
STRIX_GSQ_HD std::uint32_t signed4(std::uint32_t grid, std::uint32_t nib) {
    const std::uint32_t one = (nib * 0x00204081U) & 0x01010101U;
    return (grid ^ (one * 0xFFU)) + one;
}
// The IQ4_NL code table (kvalues_iq4nl) as four little-endian words, values 4 w .. 4 w + 3 (host test checks them).
constexpr std::uint32_t kKv0 = 0xBFAD9881U, kKv1 = 0xF6EADDCFU, kKv2 = 0x26190D01U, kKv3 = 0x71594535U;
// Four 4-bit indices (one per byte) -> their four kvalues, by two byte permutes and a select on index bit 3.
STRIX_GSQ_HD std::uint32_t kvalue4(std::uint32_t nibbles) {
    const std::uint32_t s = nibbles & 0x07070707U, t = s | (s >> 4U), sel = (t & 0xFFU) | ((t >> 8U) & 0xFF00U);
    const std::uint32_t m = ((nibbles >> 3U) & 0x01010101U) * 0xFFU;
    return (prmt(kKv0, kKv1, sel) & ~m) | (prmt(kKv2, kKv3, sel) & m);
}
// Q2_0: four 2-bit fields of one byte -> codes field - 1 in four bytes (SWAR, no borrow across bytes).
STRIX_GSQ_HD std::uint32_t crumbs4(std::uint32_t v) {
    const std::uint32_t spread = (v & 3U) | ((v & 0xCU) << 6U) | ((v & 0x30U) << 12U) | ((v & 0xC0U) << 18U);
    return ((spread | 0x80808080U) - 0x01010101U) ^ 0x80808080U;
}
}  // namespace gsq_detail

// Codebook words of a GSQ type: IQ3_S 512 x u32, IQ3_XXS 256 x u32, IQ2_S 1,024 x u64 as 2,048 u32 halves (word 2 i + h
// = bits 32 h .. of entry i); the other types have no codebook.
template<unsigned T> struct GsqGrid {
    static constexpr unsigned words = T == 21 ? 512U : T == 18 ? 256U : T == 22 ? 2048U : 0U;
    STRIX_GSQ_HD static std::uint32_t word(unsigned i) {
        if constexpr (T == 21) return quant::detail::iq3s_grid(i);
        if constexpr (T == 18) return quant::detail::iq3xxs_grid(i);
        if constexpr (T == 22) return std::uint32_t(quant::detail::iq2s_grid(i >> 1U) >> (32U * (i & 1U)));
        return 0U;
    }
};
// Codebook views passed to GsqDecode::word: the shared constant tables, or a block's shared-memory copy (gsq_grid_fill).
template<unsigned T> struct GsqConstGrid {
    STRIX_GSQ_HD std::uint32_t operator()(unsigned i) const { return GsqGrid<T>::word(i); }
};
struct GsqSmemGrid {
    const std::uint32_t* words;
    STRIX_GSQ_HD std::uint32_t operator()(unsigned i) const { return words[i]; }
};
// Thread `t` of `threads` copies its share of the codebook to `dst` (GsqGrid<T>::words words); the caller synchronises.
template<unsigned T> STRIX_GSQ_HD void gsq_grid_fill(std::uint32_t* dst, unsigned t, unsigned threads) {
    if constexpr (GsqGrid<T>::words != 0U)
        for (unsigned i = t; i < GsqGrid<T>::words; i += threads) dst[i] = GsqGrid<T>::word(i);
}

// Scalar extraction is shared by the MMA and dp4a operand protocols. Grid
// placement is supplied by the canonical CUDA decoder, with its original notices.
template<unsigned T> struct GsqDecode {
    static_assert(T==18 || T==20 || T==21 || T==22 || T==23 || T==42);
    static constexpr unsigned be=::strix::weight::formats::gsq_block_elements(T);
    static constexpr unsigned bb=::strix::weight::formats::gsq_block_bytes(T);
    static_assert(bb%2==0, "16-bit field loads need even block sizes");
    static constexpr bool halves=T==22;
    static constexpr unsigned grid_words=GsqGrid<T>::words;
    STRIX_GSQ_HD static const std::uint8_t* block(const std::uint8_t* row,unsigned group,unsigned& sub) {
        sub=group%(be/32); return row+std::uint64_t(group/(be/32))*bb;
    }
    // Reference definition of code k (0..31) of a group (the CPU oracle path; kernels use word()).
    STRIX_GSQ_HD static int code(const std::uint8_t* row,unsigned group,unsigned k) {
        unsigned s; const auto* x=block(row,group,s);
        if constexpr(T==42) { const unsigned e=32*s+k; return int((x[2+e/4]>>(2*(e%4)))&3)-1; }
        if constexpr(T==20 || T==23) {
            const auto* qs=T==20?x+2:x+8+16*s;
            return int(quant::detail::kvalue((qs[k%16]>>(4*(k/16)))&15));
        }
        if constexpr(T==21) {
            using Iq=::strix::weight::formats::Iq3S;
            const unsigned l=k/8,j=k%8,qh=x[Iq::qh+s];
            const unsigned idx=x[Iq::qs+8*s+2*l+j/4]|((qh<<(8-2*l-j/4))&256);
            const unsigned grid=quant::detail::iq3s_grid(idx);
            return int((grid>>(8*(j%4)))&255)*((x[Iq::signs+4*s+l]>>j)&1?-1:1);
        }
        if constexpr(T==18) {
            const unsigned off=66+4*s;
            const unsigned aux=unsigned(x[off])|(unsigned(x[off+1])<<8)|(unsigned(x[off+2])<<16)|(unsigned(x[off+3])<<24);
            const unsigned l=k/8,j=k%8,sign=quant::detail::ksigns((aux>>(7*l))&127);
            const unsigned grid=quant::detail::iq3xxs_grid(x[2+8*s+2*l+j/4]);
            return int((grid>>(8*(j%4)))&255)*((sign>>j)&1?-1:1);
        }
        if constexpr(T==22) {
            const unsigned l=k/8,j=k%8,qh=x[66+s];
            const auto grid=quant::detail::iq2s_grid(x[2+4*s+l]|((qh<<(8-2*l))&0x300));
            return int((grid>>(8*j))&255)*((x[34+4*s+l]>>j)&1?-1:1);
        }
        return 0;
    }
    // Field bytes of one group that words i0, i1 (0..7, word i = codes 4 i .. 4 i + 3) and the group scales need. Loads
    // only: a kernel may issue them a group ahead. d: FP16 scale bits; a / b: per-group fields; w / v: per-word fields.
    //   IQ3_S  a scale byte, b qh, w qs byte (index low 8 bits), v sign byte    IQ3_XXS a aux word, w qs byte
    //   IQ2_S  a scale byte, b qh, w qs byte, v sign byte                       IQ4_XS  a high-scale bits, b scale byte, w 4 qs bytes
    //   IQ4_NL w 4 qs bytes                                                     Q2_0    w the word's byte
    struct Raw { std::uint32_t d,a,b,w[2],v[2]; };
    STRIX_GSQ_HD static Raw load(const std::uint8_t* row,unsigned group,unsigned i0,unsigned i1) {
        using namespace gsq_detail;
        unsigned s; const auto* x=block(row,group,s);
        Raw r{};
        r.d=ld16(x);
        const unsigned idx[2]={i0,i1};
        #pragma unroll
        for(unsigned n=0;n<2;++n) {
            const unsigned i=idx[n];
            if constexpr(T==21) { r.w[n]=x[2+8*s+i]; r.v[n]=x[74+4*s+i/2]; }
            if constexpr(T==18) r.w[n]=x[2+8*s+i];
            if constexpr(T==22) { r.w[n]=x[2+4*s+i/2]; r.v[n]=x[34+4*s+i/2]; }
            if constexpr(T==23) r.w[n]=ld32(x+8+16*s+4*(i%4));
            if constexpr(T==20) r.w[n]=ld32(x+2+4*(i%4));
            if constexpr(T==42) r.w[n]=x[2+8*s+i];
        }
        if constexpr(T==21) { r.a=x[106+s/2]; r.b=x[66+s]; }
        if constexpr(T==18) r.a=ld32(x+66+4*s);
        if constexpr(T==22) { r.a=x[74+s]; r.b=x[66+s]; }
        if constexpr(T==23) { r.a=ld16(x+2); r.b=x[4+s/2]; }
        return r;
    }
    // Word `index` from the fields loaded for slot n (= the index passed to load() in that slot): byte j = code(4 index + j).
    template<class Grid> STRIX_GSQ_HD static unsigned word(const Raw& r,unsigned n,unsigned index,const Grid& grid) {
        using namespace gsq_detail;
        const unsigned half=index%2;
        if constexpr(T==21) {
            const unsigned g=grid(r.w[n]|((r.b<<(8-index))&256));
            return signed4(g,(r.v[n]>>(4*half))&15);
        }
        if constexpr(T==18) {
            const unsigned sign=quant::detail::ksigns((r.a>>(7*(index/2)))&127);
            return signed4(grid(r.w[n]),(sign>>(4*half))&15);
        }
        if constexpr(T==22) {
            const unsigned idx=r.w[n]|((r.b<<(8-2*(index/2)))&0x300);
            return signed4(grid(2*idx+half),(r.v[n]>>(4*half))&15);
        }
        if constexpr(T==20 || T==23) return kvalue4((r.w[n]>>(4*(index/4)))&0x0F0F0F0FU);
        if constexpr(T==42) return crumbs4(r.w[n]);
        return 0;
    }
    STRIX_GSQ_HD static void scales(const Raw& r,unsigned group,float& a,float& b) {
        const unsigned s=group%(be/32);
        const float d=quant::detail::fp16(std::uint16_t(r.d));
        b=0.0F;
        if constexpr(T==20 || T==42) a=d;
        if constexpr(T==21) a=d*float(1+2*int((r.a>>(4*(s%2)))&15));
        if constexpr(T==18) a=(d*(0.5F+float(r.a>>28)))*0.5F;
        if constexpr(T==22) { a=(d*(0.5F+float(r.a&15)))*0.25F; b=(d*(0.5F+float(r.a>>4)))*0.25F; }
        if constexpr(T==23) {
            const unsigned ls=((r.b>>(4*(s%2)))&15)|(((r.a>>(2*s))&3)<<4);
            a=d*float(int(ls)-32);
        }
    }
    // One-shot forms (same values): the constant codebook, fields loaded here.
    STRIX_GSQ_HD static void scales(const std::uint8_t* row,unsigned group,float& a,float& b) {
        scales(load(row,group,0,0),group,a,b);
    }
    STRIX_GSQ_HD static unsigned word(const std::uint8_t* row,unsigned group,unsigned index) {
        return word(load(row,group,index,index),0,index,GsqConstGrid<T>{});
    }
};

} // namespace strix::cuda::weight
