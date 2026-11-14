// Compare this receipt against the pre-change shared operators on a host build.
#include "h3/vae/enc_ref.hpp"
#include <iostream>
#include <cmath>
using namespace strix::h3::vae;
namespace {
std::uint64_t digest=14695981039346656037ULL;
void add(const H16& x) { for(auto h:x) { digest=(digest^(h&255))*1099511628211ULL;digest=(digest^(h>>8))*1099511628211ULL; } }
H16 values(std::size_t n,int salt) { H16 out(n);for(std::size_t i=0;i<n;++i)out[i]=f2h(float(int((i*13+salt)%31)-15)/32);return out; }
}
int main() {
    LinearH l;l.in=32;l.out=16;l.w=values(512,1);l.b=values(16,2);const auto x=values(96,4);
    add(linear(x,3,l));add(linear(x,3,l,true));
    const auto norm=values(32,1);add(rms_norm(x,3,32,&norm,1e-5F));add(layer_norm(x,3,32,norm,norm,1e-5F));
    const auto rope=rope_table(1,1,3);add(attention(values(3*kQkv,7),3,rope));
    HeadH h;h.post_quant.in=24;h.post_quant.out=24;h.post_quant.w=values(24*24,4);h.post_quant.b=values(24,1);
    add(post_quant(h,values(24*3,1),3));add(unpatchify(values(3*kProj,8),1,1,3));
    enc::Map m;m.C=32;m.H=4;m.W=4;m.v=values(32*16,3);
    enc::Conv2 c;c.cin=32;c.cout=32;c.k=3;c.w=values(32*32*9,2);c.b=values(32,5);
    add(enc::conv(c,m,1).v);add(enc::conv(c,m,2).v);add(enc::conv(c,m,1,&m).v);
    enc::Norm gn;gn.w=values(32,5);gn.b=values(32,7);add(enc::gn_silu(gn,m).v);
    add(enc::normalize_pixels(std::vector<float>(3*32*32,0.25F),32,32).v);
    std::cout<<std::hex<<digest<<'\n';
}
