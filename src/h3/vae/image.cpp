#include "h3/vae/image.hpp"
#include "h3/block/weights.hpp"
#include "h3/block/torch_rng.hpp"
#include "h3/vae/latent_stats.hpp"
#include <array>
namespace strix::h3::vae {
namespace {
using Map = enc::MapT<float>;
std::vector<float> read(const block::SafeTensors& st, const std::string& key) {
    return block::detail::read_as_f32(st,key,false);
}
LinearT<float> linear(const block::SafeTensors& st, const std::string& key) {
    const auto& t=st.at(key+".weight");
    if (t.shape.size()<2 || t.shape[0]<=0) throw std::invalid_argument("image VAE: malformed linear");
    LinearT<float> l; l.out=int(t.shape[0]); l.in=int(t.numel()/t.shape[0]);
    l.w=read(st,key+".weight"); l.b=read(st,key+".bias");
    if (l.b.size()!=std::size_t(l.out)) throw std::invalid_argument("image VAE: bias shape");
    return l;
}
enc::Conv2T<float> conv(const block::SafeTensors& st,const std::string& key) {
    const auto& t=st.at(key+".weight");
    if (t.shape.size()!=5 || t.shape[3]!=t.shape[4] || (t.shape[2]!=1 && t.shape[2]!=3))
        throw std::invalid_argument("image VAE: expected causal 3D convolution");
    enc::Conv2T<float> c; c.cout=int(t.shape[0]); c.cin=int(t.shape[1]); c.kt=int(t.shape[2]); c.k=int(t.shape[3]);
    const auto raw=read(st,key+".weight"); c.b=read(st,key+".bias");
    c.w.resize(std::size_t(c.cout)*c.cin*c.k*c.k);
    for (std::size_t i=0;i<std::size_t(c.cout)*c.cin;++i)
        std::copy_n(raw.data()+(i*c.kt+c.kt-1)*c.k*c.k,c.k*c.k,c.w.data()+i*c.k*c.k);
    return c;
}
enc::NormT<float> norm(const block::SafeTensors& st,const std::string& key) {
    return {read(st,key+".weight"),read(st,key+".bias")};
}
// Spatial tiling shares the video tile plan. Blend RAW neighbours above then left,
// cut their overlap, and assemble one image; no temporal padding or repeated frames.
std::vector<float> tiled(const std::vector<float>& x,int C,int H,int W,int output_scale,
                         const ImageVae::TileFn& fn) {
    const auto ty=split_tiles(H),tx=split_tiles(W);
    std::vector<std::vector<Map>> raw(ty.start.size());
    std::vector<Map> cropped;
    for (std::size_t iy=0;iy<ty.start.size();++iy) for (std::size_t ix=0;ix<tx.start.size();++ix) {
        const int h=ty.len[iy],w=tx.len[ix];
        std::vector<float> tile(std::size_t(C)*h*w);
        for(int c=0;c<C;++c) for(int y=0;y<h;++y)
            std::copy_n(x.data()+(std::size_t(c)*H+ty.start[iy]+y)*W+tx.start[ix],w,tile.data()+(std::size_t(c)*h+y)*w);
        Map m; m.H=h/output_scale; m.W=w/output_scale; m.v=fn(tile,h,w);
        const auto plane=std::size_t(m.H)*m.W;
        if (!plane || m.v.size()%plane) throw std::runtime_error("image VAE: invalid tile output");
        m.C=int(m.v.size()/plane); raw[iy].push_back(m);
        auto blend=[&](const Map& previous,int overlap,bool vertical) {
            for(int c=0;c<m.C;++c) for(int y=0;y<(vertical?overlap:m.H);++y) for(int xx=0;xx<(vertical?m.W:overlap);++xx) {
                const int p=vertical?y:xx;
                const float weight=float(p)/float(overlap);
                const int sy=vertical?previous.H-overlap+y:y, sx=vertical?xx:previous.W-overlap+xx;
                auto& v=m.v[(std::size_t(c)*m.H+y)*m.W+xx];
                v=previous.v[(std::size_t(c)*previous.H+sy)*previous.W+sx]*(1.0F-weight)+v*weight;
            }
        };
        if(iy) blend(raw[iy-1][ix],ty.overlap[iy-1]/output_scale,true);
        if(ix) blend(raw[iy][ix-1],tx.overlap[ix-1]/output_scale,false);
        Map crop; crop.C=m.C;
        crop.H=m.H-(iy+1<ty.start.size()?ty.overlap[iy]/output_scale:0);
        crop.W=m.W-(ix+1<tx.start.size()?tx.overlap[ix]/output_scale:0);
        crop.v.resize(std::size_t(crop.C)*crop.H*crop.W);
        for(int c=0;c<crop.C;++c) for(int y=0;y<crop.H;++y)
            std::copy_n(m.v.data()+(std::size_t(c)*m.H+y)*m.W,crop.W,crop.v.data()+(std::size_t(c)*crop.H+y)*crop.W);
        cropped.push_back(std::move(crop));
    }
    const int OH=H/output_scale,OW=W/output_scale,OC=cropped.front().C;
    std::vector<float> out(std::size_t(OC)*OH*OW);
    int y0=0; std::size_t i=0;
    for(std::size_t y=0;y<ty.start.size();++y) { int x0=0,advance=0;
        for(std::size_t xx=0;xx<tx.start.size();++xx) { const auto& m=cropped[i++]; advance=m.H;
            for(int c=0;c<OC;++c) for(int r=0;r<m.H;++r)
                std::copy_n(m.v.data()+(std::size_t(c)*m.H+r)*m.W,m.W,out.data()+(std::size_t(c)*OH+y0+r)*OW+x0);
            x0+=m.W;
        } y0+=advance;
    }
    return out;
}
}

std::vector<float> assemble_image_tiles(int H,int W,
    const std::function<std::vector<float>(std::size_t,int,int)>& tile) {
    if(H<32 || W<32 || H%32 || W%32) throw std::invalid_argument("image VAE: tile canvas shape");
    std::vector<float> canvas(std::size_t(H)*W,0);
    std::size_t index=0;
    return tiled(canvas,1,H,W,1,[&](const auto&,int h,int w) {
        auto out=tile(index++,h,w);
        if(out.size()!=std::size_t(3)*h*w) throw std::invalid_argument("image VAE: RGB tile shape");
        return out;
    });
}

RopeTable image_rope_table(int ny,int nx,bool fp32,int axes) {
    if(ny<1 || nx<1) throw std::invalid_argument("image VAE: empty rope geometry");
    if(axes!=2 && axes!=3) throw std::invalid_argument("image VAE: rope axes must be 2 or 3");
    RopeTable r; r.L=ny*nx+kSuffix; r.pairs=8*axes;
    r.c.assign(std::size_t(r.L)*r.pairs,1); r.s.assign(r.c.size(),0);
    auto round=[&](float v) { return fp32?v:rh(v); };
    const float time=round(float(2.0*(0.5/7.0)-1.0));
    for(int y=0;y<ny;++y) for(int x=0;x<nx;++x) {
        const float ids3[]={time,round(round(2*round(round(float(y)+0.5F)/float(ny)))-1),round(round(2*round(round(float(x)+0.5F)/float(nx)))-1)};
        const float* ids=ids3+(3-axes);  // the 2-axis (MiniMax-H3-Image) decoder has no time coordinate
        for(int a=0;a<axes;++a) for(int f=0;f<8;++f) {
            const float inv=round(1/std::pow(100.0F,float(f)*0.125F));
            const float angle=(float(2*3.14159265358979323846)*ids[a])*inv;
            const auto i=std::size_t(y*nx+x)*r.pairs+a*8+f;
            r.c[i]=round(std::cos(angle)); r.s[i]=round(std::sin(angle));
        }
    } return r;
}

ImageVae::ImageVae(const std::string& weights):st_(weights,block::CheckpointNames::image_vae) {
    head_.post_quant=linear(st_,"post_quant_conv"); head_.x_embed=linear(st_,"decoder.x_embedder");
    head_.proj=linear(st_,"decoder.proj_out");
    head_.registers=read(st_,"decoder.register_tokens");
    head_.norm_out_w=read(st_,"decoder.norm_out.weight"); head_.norm_out_b=read(st_,"decoder.norm_out.bias");
    // Packs without the statistics use the released constants. Both or neither.
    if(st_.find("latents_mean") || st_.find("latents_std")) { head_.latents_mean=read(st_,"latents_mean"); head_.latents_std=read(st_,"latents_std"); }
    else { head_.latents_mean.assign(std::begin(kLatentsMean),std::end(kLatentsMean)); head_.latents_std.assign(std::begin(kLatentsStd),std::end(kLatentsStd)); }
    // Width 2048 / 32 heads / 3-axis rope: p-h3-image. Width 1024 / 16 heads / 2-axis half-head rope: the released
    // MiniMax-H3-Image VAE (released 1024-wide image decoder). Head dim 64, FFN 4x and 24 layers in both.
    const int kDim=head_.x_embed.out, kQkv=3*kDim, kFf=4*kDim, kFf2=2*kFf;
    if(kDim!=1024 && kDim!=vae::kDim) throw std::invalid_argument("image VAE: unsupported decoder width");
    rope_axes_=kDim==1024?2:3;
    if(head_.post_quant.in!=24 || head_.post_quant.out!=24 || head_.x_embed.in!=24 || head_.x_embed.out!=kDim ||
       head_.proj.in!=kDim || head_.proj.out!=768 || head_.registers.size()!=4*kDim || head_.norm_out_w.size()!=kDim ||
       head_.norm_out_b.size()!=kDim || head_.latents_mean.size()!=24 || head_.latents_std.size()!=24)
        throw std::invalid_argument("image VAE: unsupported head geometry");
    for(auto sd:head_.latents_std) if(!(sd>0) || !std::isfinite(sd)) throw std::invalid_argument("image VAE: invalid latent scale");
    for(int i=0;i<24;++i) {
        const auto p="decoder.transformer_blocks."+std::to_string(i)+".";
        LayerT<float> l;
        l.norm1=read(st_,p+"norm1.weight"); l.norm2=read(st_,p+"norm2.weight");
        l.scale1=read(st_,p+"scale1"); l.scale2=read(st_,p+"scale2");
        l.qkv=linear(st_,p+"attn.to_qkv"); l.out=linear(st_,p+"attn.to_out");
        l.w1=linear(st_,p+"ff.w1"); l.w2=linear(st_,p+"ff.w2");
        if(l.norm1.size()!=kDim || l.norm2.size()!=kDim || l.scale1.size()!=kDim || l.scale2.size()!=kDim ||
           l.qkv.in!=kDim || l.qkv.out!=kQkv || l.out.in!=kDim || l.out.out!=kDim ||
           l.w1.in!=kDim || l.w1.out!=kFf2 || l.w2.in!=kFf || l.w2.out!=kDim)
            throw std::invalid_argument("image VAE: unsupported block geometry");
        layers_.push_back(std::move(l));
    }
    if(st_.find("decoder.transformer_blocks.24.norm1.weight")) throw std::invalid_argument("image VAE: expected 24 decoder layers");
}
void ImageVae::load_encoder() {
    if(encoder_) return;
    auto w=std::make_unique<enc::WeightsT<float>>();
    w->conv_in=conv(st_,"encoder.conv_in");
    for(int l=0;l<6;++l) {
        for(int b=0;b<2;++b) {
            const auto p="encoder.down."+std::to_string(l)+".block."+std::to_string(b)+".";
            auto& r=w->blocks[l][b]; r.n1=norm(st_,p+"norm1"); r.n2=norm(st_,p+"norm2");
            r.c1=conv(st_,p+"conv1"); r.c2=conv(st_,p+"conv2"); r.has_nin=st_.find(p+"nin_shortcut.weight")!=nullptr;
            if(r.has_nin) r.nin=conv(st_,p+"nin_shortcut");
        } if(l<4) w->down[l]=conv(st_,"encoder.down."+std::to_string(l)+".downsample.conv");
    }
    w->norm_out=norm(st_,"encoder.norm_out"); w->conv_out=conv(st_,"encoder.conv_out"); w->quant=conv(st_,"quant_conv");
    encoder_=std::move(w);
}
std::vector<float> ImageVae::posterior(const std::vector<float>& moments,const std::vector<float>& noise,
                                      const std::vector<float>& mean,const std::vector<float>& sd) {
    if(mean.size()!=24 || sd.size()!=24 || noise.empty() || noise.size()%24 || moments.size()!=2*noise.size())
        throw std::invalid_argument("image VAE: posterior shapes");
    std::vector<float> z(noise.size()); const auto plane=noise.size()/24;
    for(std::size_t i=0;i<z.size();++i) {
        if(!(sd[i/plane]>0)) throw std::invalid_argument("image VAE: posterior scale");
        const float scale=std::exp(0.5F*std::clamp(moments[i+z.size()],-30.0F,20.0F));
        z[i]=(std::fma(scale,noise[i],moments[i])-mean[i/plane])/sd[i/plane];
    } return z;
}
std::vector<float> ImageVae::encode(const std::vector<float>& pixels,int H,int W,const TileFn& fn) {
    enc::check_image(H,W,pixels.size());
    if(!fn) load_encoder();
    std::vector<float> input=pixels;
    for(auto& v:input) v=2*v-1;
    const auto m=tiled(input,3,H,W,16,[&](const auto& tile,int h,int w) {
        if(fn) return fn(tile,h,w);
        return precision::encode_tile<float>(*encoder_,precision::normalize_pixels<float>(tile,h,w)).v;
    });
    block::TorchCpuGenerator rng(42);
    return posterior(m,rng.randn(std::size_t(24)*(H/16)*(W/16)),head_.latents_mean,head_.latents_std);
}
std::vector<float> ImageVae::decode(const std::vector<float>& z,int H,int W,const TileFn& fn,const TileDispatch& dispatch) {
    if(H<32 || W<32 || H%32 || W%32 || z.size()!=std::size_t(24)*(H/16)*(W/16))
        throw std::invalid_argument("image VAE: latent shape");
    std::vector<float> den=z;
    const auto plane=std::size_t(H/16)*(W/16);
    for(std::size_t i=0;i<den.size();++i) den[i]=den[i]*head_.latents_std[i/plane]+head_.latents_mean[i/plane];
    const auto ty=split_tiles(H),tx=split_tiles(W);
    const auto count=ty.start.size()*tx.start.size();
    auto eval=[&](std::size_t index) {
        if(index>=count) throw std::out_of_range("image VAE: tile index");
        const auto iy=index/tx.start.size(),ix=index%tx.start.size();
        const int h=ty.len[iy],w=tx.len[ix];
        const int ny=h/16,nx=w/16; std::vector<float> box(std::size_t(24)*ny*nx);
        for(int c=0;c<24;++c) for(int y=0;y<ny;++y)
            std::copy_n(den.data()+(std::size_t(c)*(H/16)+ty.start[iy]/16+y)*(W/16)+tx.start[ix]/16,nx,box.data()+(std::size_t(c)*ny+y)*nx);
        const auto pq=precision::post_quant<float>(head_,box,ny*nx);
        if(fn) return fn(pq,ny,nx);
        auto x=precision::embed<float>(head_,pq,1,ny,nx);
        const auto rope=image_rope_table(ny,nx,true,rope_axes_);
        for(const auto& l:layers_) precision::layer_forward<float>(l,x,ny*nx+kSuffix,rope);
        return precision::head_out<float>(head_,x,1,ny,nx,nullptr,1);
    };
    std::vector<std::vector<float>> gathered;
    if(dispatch) {
        std::vector<std::size_t> sizes;
        for(auto h:ty.len) for(auto w:tx.len) sizes.push_back(std::size_t(3)*h*w);
        gathered=dispatch(sizes,eval);
        if(gathered.size()!=count) throw std::invalid_argument("image VAE: gathered tile count");
    }
    auto out=assemble_image_tiles(H,W,[&](std::size_t index,int,int) {
        return dispatch ? std::move(gathered[index]) : eval(index);
    });
    constexpr float mu[]={0.485F,0.456F,0.406F},sd[]={0.229F,0.224F,0.225F};
    for(std::size_t i=0;i<out.size();++i) out[i]=std::clamp(out[i]*sd[i/(std::size_t(H)*W)]+mu[i/(std::size_t(H)*W)],0.0F,1.0F);
    return out;
}
}
