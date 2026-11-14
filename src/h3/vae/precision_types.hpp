#pragma once
#include "h3/vae/vae_half.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
namespace strix::h3::vae {
// Scalar policies share the reference operators while retaining every FP16 rounding point.
template<class T> struct Scalar;
template<> struct Scalar<half_bits> {
    static float get(half_bits v) { return h2f(v); }
    static half_bits set(float v) { return f2h(v); }
    static float round(float v) { return rh(v); }
    static half_bits sanitize(half_bits v) {
        if ((v&0x7c00U)!=0x7c00U) return v;
        return (v&0x03ffU) ? 0 : half_bits((v&0x8000U)|0x7bffU);
    }
};
template<> struct Scalar<float> {
    static float get(float v) { return v; }
    static float set(float v) { return v; }
    static float round(float v) { return v; }
    static float sanitize(float v) { return std::isnan(v) ? 0.0F : std::clamp(v,-3.4028234663852886e38F,3.4028234663852886e38F); }
};
template<class T> struct LinearT { int in=0,out=0; std::vector<T> w,b; };
template<class T> struct LayerT { std::vector<T> norm1,norm2,scale1,scale2; LinearT<T> qkv,out,w1,w2; };
template<class T> struct HeadT { LinearT<T> post_quant,x_embed,proj; std::vector<T> registers,norm_out_w,norm_out_b,latents_mean,latents_std; };
namespace enc {
template<class T> struct Conv2T { int cin=0,cout=0,k=0,kt=1; std::vector<T> w,b,w3; };
template<class T> struct NormT { std::vector<T> w,b; };
template<class T> struct ResT { NormT<T> n1,n2; Conv2T<T> c1,c2,nin; bool has_nin=false; };
template<class T> struct WeightsT { Conv2T<T> conv_in,conv_out,quant; ResT<T> blocks[6][2]; Conv2T<T> down[4]; NormT<T> norm_out; std::vector<T> latents_mean,latents_std; };
template<class T> struct MapT { int C=0,H=0,W=0; std::vector<T> v; };
}
}
