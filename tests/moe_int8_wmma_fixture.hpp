#pragma once
// Independent FP64 mathematical oracle; no HIP or production decoder dependencies.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace i8fixture {
inline constexpr unsigned H=2560, I=640, E=512, K=10;
inline void require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
inline std::uint64_t mix(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30U))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27U))*0x94d049bb133111ebULL;return x^(x>>31U);
}
inline double half(const unsigned char* p) {
    const unsigned b=p[0]|(unsigned(p[1])<<8U),e=(b>>10U)&31U,f=b&1023U;
    const double v=e==31U?(f?std::numeric_limits<double>::quiet_NaN():std::numeric_limits<double>::infinity()):
        e?std::ldexp(double(1024U+f),int(e)-25):std::ldexp(double(f),-24);
    return b&32768U?-v:v;
}
inline unsigned bytes(unsigned type) { return type==12?144:type==7?24:34; }
inline unsigned elements(unsigned type) { return type==12?256:32; }
inline std::size_t row_bytes(unsigned type,unsigned width) { return std::size_t(width/elements(type))*bytes(type); }
struct Coeff { int q; double d,m; };
inline Coeff coefficient(const unsigned char* row,unsigned k,unsigned type) {
    const auto* b=row+(k/elements(type))*bytes(type);
    const unsigned j=k%elements(type);
    if(type==12) {
        const unsigned g=j/32,l=j%32;const auto* s=b+4;
        const unsigned sc=g<4?s[g]&63U:(s[g+4]&15U)|((unsigned(s[g-4])>>6U)<<4U);
        const unsigned mn=g<4?s[g+4]&63U:(unsigned(s[g+4])>>4U)|((unsigned(s[g])>>6U)<<4U);
        const unsigned q=b[16+(g/2)*32+l];
        return {int(g%2?q>>4U:q&15U),half(b)*sc,-half(b+2)*mn};
    }
    if(type==7) {
        const unsigned hi=(unsigned(b[4+j/8])>>(j%8))&1U;
        return {int(((j<16?b[8+j]&15U:b[8+j-16]>>4U))|(hi<<4U)),half(b),half(b+2)};
    }
    require(type==8,"unsupported reference format");
    return {int(static_cast<std::int8_t>(b[2+j])),half(b),0};
}
inline double weight(const unsigned char* row,unsigned k,unsigned type) {
    const auto c=coefficient(row,k,type);return c.d*c.q+c.m;
}
struct Group {
    std::array<signed char,32> q{};float delta=1;int sum=0;unsigned status=0;
};
// Match the two existing quantizers' different tiny rules deliberately.
inline Group quantize(const float* x,bool down) {
    Group g;float a=0;
    for(unsigned k=0;k<32;++k) { if(!std::isfinite(x[k]))g.status|=2U;else a=std::max(a,std::abs(x[k])); }
    const bool tiny=down&&a>0&&a<0x1p-119F;
    g.delta=a==0||tiny?1.0F:a/127.0F;
    if(!(g.delta>0)||!std::isfinite(g.delta)){g.status|=4U;g.delta=1;}
    for(unsigned k=0;k<32;++k) {
        const float v=g.status||tiny?0:std::nearbyint(x[k]/g.delta);
        g.q[k]=static_cast<signed char>(std::clamp(v,-127.0F,127.0F));g.sum+=g.q[k];
    }
    return g;
}
inline double silu(double g,double u) { return (g/(1+std::exp(-g)))*u; }
inline std::vector<Group> image(const std::vector<float>& x,bool down) {
    std::vector<Group> r(x.size()/32);for(std::size_t g=0;g<r.size();++g)r[g]=quantize(x.data()+32*g,down);return r;
}
inline double dot(const std::vector<float>& x,const unsigned char* w,unsigned type) {
    double d=0;for(unsigned k=0;k<x.size();++k)d+=double(x[k])*weight(w,k,type);return d;
}
inline double qdot(const std::vector<Group>& x,const unsigned char* w,unsigned type) {
    double d=0;
    for(unsigned g=0;g<x.size();++g) {
        int dot32=0,sum=0;const auto c=coefficient(w,g*32,type);
        for(unsigned k=0;k<32;++k){dot32+=int(x[g].q[k])*coefficient(w,g*32+k,type).q;sum+=x[g].q[k];}
        require(sum==x[g].sum,"reference code sum mismatch");
        d+=double(x[g].delta)*(c.d*dot32+c.m*sum);
    }
    return d;
}
struct Metrics {
    std::vector<double> errors;double maxabs=0,e2=0,r2=0,y2=0,yr=0,bias=0;
    void add(double y,double ref) {
        require(std::isfinite(y)&&std::isfinite(ref),"nonfinite reference or device output");
        const double e=y-ref;errors.push_back(std::abs(e));maxabs=std::max(maxabs,std::abs(e));
        e2+=e*e;r2+=ref*ref;y2+=y*y;yr+=y*ref;bias+=e;
    }
    void print(const char* arm,const char* reference,const char* stage,unsigned rows,unsigned m,unsigned n) {
        require(!errors.empty(),"empty accuracy sample");std::sort(errors.begin(),errors.end());
        const auto q=[&](double p){return errors[std::size_t(p*double(errors.size()-1))];};
        const double floor=1e-12;
        const double cosine=yr/std::max(floor,std::sqrt(r2*y2));
        std::printf("{\"accuracy\":\"%s\",\"reference\":\"%s\",\"stage\":\"%s\",\"T\":%u,\"M\":%u,\"N\":%u,\"samples\":%zu,\"max_abs\":%.9g,\"rmse\":%.9g,\"normalized_l2\":%.9g,\"cosine\":%.9g,\"cosine_error\":%.9g,\"bias\":%.9g,\"p50_abs\":%.9g,\"p95_abs\":%.9g,\"p99_abs\":%.9g,\"denominator_floor\":1e-12}\n",
            arm,reference,stage,rows,m,n,errors.size(),maxabs,std::sqrt(e2/double(errors.size())),
            std::sqrt(e2)/std::max(floor,std::sqrt(r2)),cosine,1-cosine,bias/double(errors.size()),q(.5),q(.95),q(.99));
    }
};
// Packed edge fixtures: all scale/min bit boundaries, Q5 high bits, signed Q8,
// zero/subnormal/negative scales and cancellation; kept separate from real weights.
inline std::vector<unsigned char> packed(unsigned type,unsigned blocks) {
    std::vector<unsigned char> out(std::size_t(blocks)*bytes(type));
    constexpr unsigned headers[]{0,0x8000,1,0x8001,0x1400,0x9400,0x2401,0xa401};
    for(unsigned b=0;b<blocks;++b) {
        auto* p=out.data()+std::size_t(b)*bytes(type);
        for(unsigned j=0;j<bytes(type);++j)p[j]=static_cast<unsigned char>(mix(std::uint64_t(b)*1024+j));
        for(unsigned h=0;h<(type==8?1U:2U);++h) {
            const unsigned v=headers[(b+h)%8];p[2*h]=static_cast<unsigned char>(v);p[2*h+1]=static_cast<unsigned char>(v>>8U);
        }
        if(type==12)for(unsigned j=0;j<12;++j)p[4+j]=static_cast<unsigned char>((b+j)&255U);
        if(type==7) {
            const unsigned mask=b%8==0?0U:b%8==1?0xffffffffU:unsigned(mix(b));
            for(unsigned j=0;j<4;++j)p[4+j]=static_cast<unsigned char>(mask>>(8*j));
        }
    }
    return out;
}
} // namespace i8fixture
