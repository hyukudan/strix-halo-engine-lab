// CudaEncoder: the shared vision plan (encoder_plan.hpp) bound to the FP32 CUDA kernels of src/cuda/core/vision.
// Launch glue and memory only; no kernel lives here.
#include "strix/vision/encoder_cuda.hpp"

#include "../cuda/core/vision/vision_tower.hpp"
#include "encoder_plan.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace strix::vision {
namespace {

namespace cv = strix::cuda_core::vision;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("vision cuda: ") + what + ": " + cudaGetErrorString(e));
}

// Owning device allocation.
struct Dev {
    void* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    explicit Dev(std::size_t bytes) : n(bytes) {
        if (bytes) ck(cudaMalloc(&p, bytes), "alloc");
    }
    Dev(Dev&& o) noexcept : p(std::exchange(o.p, nullptr)), n(std::exchange(o.n, 0)) {}
    Dev& operator=(Dev&& o) noexcept {
        if (this != &o) {
            reset();
            p = std::exchange(o.p, nullptr);
            n = std::exchange(o.n, 0);
        }
        return *this;
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { reset(); }
    void reset() {
        if (p) cudaFree(p);
        p = nullptr;
        n = 0;
    }
    float* f() const { return static_cast<float*>(p); }
};

Dev upload(const Tensor& t) {
    if (!t) return {};
    std::vector<float> h(t.size());
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = t.at(i);
    Dev d(h.size() * 4U);
    ck(cudaMemcpy(d.p, h.data(), d.n, cudaMemcpyHostToDevice), "weight upload");
    return d;
}
struct Panel {
    Dev w, b;
    unsigned n = 0, k = 0;
};
Panel make_panel(const Tensor& w, const Tensor& bias) {
    if (!w) throw std::invalid_argument("vision cuda: missing weight");
    if (bias && bias.size() != w.rows) throw std::invalid_argument("vision cuda: bias shape");
    return Panel{upload(w), upload(bias), static_cast<unsigned>(w.rows), static_cast<unsigned>(w.cols)};
}

struct Block {
    Panel qkv, out, mlp1, mlp2;
    Dev n1w, n1b, n2w, n2b;
};

}  // namespace

bool cuda_supported(const EncoderConfig& c, std::string* why) {
    const auto no = [&](const char* m) {
        if (why) *why = m;
        return false;
    };
    if (!c.heads || c.width % c.heads) return no("width not divisible by heads");
    const unsigned d = c.head_dim();
    if (d == 0 || d > 128U) return no("head dimension above 128");
    if (c.has_rope() && d % 2U) return no("odd head dimension with rotary positions");
    return true;
}

struct CudaEncoder::Impl {
    EncoderConfig cfg;
    CudaEncoderOptions opt;
    bool loaded = false;
    Panel patch;
    std::vector<Block> blocks;
    Dev enw, enb, fnw, fnb;
    std::unordered_map<const void*, Panel> projector;
    Dev ws;
    std::size_t resident = 0;

    // Per-encode state the plan's steps read (valid inside encode only).
    unsigned N = 0;
    float *x = nullptr, *nrm = nullptr, *qkv = nullptr, *ao = nullptr, *mid = nullptr, *act = nullptr, *patches = nullptr, *pos = nullptr,
          *cs = nullptr, *sn = nullptr;
    const float* hidden = nullptr;

    cv::Norm norm_kind() const { return cfg.norm == EncoderConfig::Norm::layernorm ? cv::Norm::layernorm : cv::Norm::rmsnorm; }
    void norm(const float* in, float* out, const Dev& w, const Dev& b) const {
        ck(cv::norm_rows(in, out, N, cfg.width, w.f(), b.f(), cfg.eps, norm_kind(), nullptr), "norm");
    }
    void gemm(const Panel& p, const float* in, unsigned rows, float* out, unsigned ldy, bool add) const {
        ck(cv::gemm_nt(in, p.w.f(), p.b.f(), out, rows, p.n, p.k, ldy, add, nullptr), "gemm");
    }
    const Panel& panel_for(const Tensor& w, const Tensor& bias) {
        auto it = projector.find(w.data);
        if (it == projector.end()) it = projector.emplace(w.data, make_panel(w, bias)).first;
        if (it->second.k != w.cols || it->second.n != w.rows) throw std::invalid_argument("vision cuda: linear panel reuse with another shape");
        return it->second;
    }

    // Steps of the shared plan (encoder_plan.hpp).
    void embed() { gemm(patch, patches, N, x, cfg.width, false); }
    void add_positions() { ck(cv::add_rows(x, pos, std::size_t{N} * cfg.width, nullptr), "add positions"); }
    void norm_embed() { norm(x, x, enw, enb); }
    void attn_norm(unsigned l) { norm(x, nrm, blocks[l].n1w, blocks[l].n1b); }
    void qkv_step(unsigned l) { gemm(blocks[l].qkv, nrm, N, qkv, 3U * cfg.width, false); }
    void rope(unsigned) {
        ck(cv::rope_qkv(qkv, cs, sn, N, cfg.heads, cfg.head_dim(), cfg.rope_pairing == EncoderConfig::RopePairing::adjacent, nullptr), "rope");
    }
    void attend(unsigned, unsigned row0, unsigned n) { ck(cv::attention_tile(qkv, ao, row0, n, cfg.heads, cfg.head_dim(), nullptr), "attention"); }
    void out_proj(unsigned l) { gemm(blocks[l].out, ao, N, x, cfg.width, true); }
    void mlp_norm(unsigned l) { norm(x, nrm, blocks[l].n2w, blocks[l].n2b); }
    void mlp(unsigned l, unsigned r0, unsigned nr) {
        const Block& b = blocks[l];
        gemm(b.mlp1, nrm + std::size_t{r0} * cfg.width, nr, mid, cfg.mlp_out(), false);
        const cv::Act kind = cfg.mlp == EncoderConfig::Mlp::gelu ? cv::Act::gelu : cfg.mlp == EncoderConfig::Mlp::gelu_tanh ? cv::Act::gelu_tanh : cv::Act::swiglu;
        ck(cv::activate_rows(mid, act, nr, cfg.mlp_width, kind, cfg.swiglu_limit, cfg.swiglu_gate_first, nullptr), "activation");
        gemm(b.mlp2, act, nr, x + std::size_t{r0} * cfg.width, cfg.width, true);
    }
    void tap(unsigned, std::vector<std::vector<float>>& taps) { taps.push_back(download(x)); }
    void final_norm() {
        norm(x, nrm, fnw, fnb);
        hidden = nrm;
    }
    std::vector<float> download(const float* dev) const {
        std::vector<float> h(std::size_t{N} * cfg.width);
        ck(cudaMemcpy(h.data(), dev, h.size() * 4U, cudaMemcpyDeviceToHost), "download");
        return h;
    }
};

CudaEncoder::CudaEncoder(EncoderConfig cfg, CudaEncoderOptions opt) : impl_(std::make_unique<Impl>()) {
    validate(cfg);
    std::string why;
    if (!cuda_supported(cfg, &why)) throw std::invalid_argument("vision cuda: " + why);
    impl_->cfg = std::move(cfg);
    impl_->opt = opt;
}
CudaEncoder::~CudaEncoder() = default;

void CudaEncoder::load(const EncoderWeights& w) {
    auto& im = *impl_;
    validate(im.cfg, w);
    im.blocks.clear();
    im.projector.clear();
    im.ws.reset();
    im.patch = make_panel(w.patch_w, im.cfg.patch_bias ? w.patch_b : Tensor{});
    im.enw = upload(w.embed_norm_w);
    im.enb = upload(w.embed_norm_b);
    im.fnw = upload(w.final_norm_w);
    im.fnb = upload(w.final_norm_b);
    for (const auto& b : w.blocks) {
        Block d;
        d.qkv = make_panel(b.qkv_w, b.qkv_b);
        d.out = make_panel(b.out_w, b.out_b);
        d.mlp1 = make_panel(b.mlp1_w, b.mlp1_b);
        d.mlp2 = make_panel(b.mlp2_w, b.mlp2_b);
        d.n1w = upload(b.norm1_w);
        d.n1b = upload(b.norm1_b);
        d.n2w = upload(b.norm2_w);
        d.n2b = upload(b.norm2_b);
        im.blocks.push_back(std::move(d));
    }
    im.loaded = true;
}
void CudaEncoder::unload() {
    auto& im = *impl_;
    im.blocks.clear();
    im.projector.clear();
    im.ws.reset();
    im.patch = Panel{};
    im.enw.reset();
    im.enb.reset();
    im.fnw.reset();
    im.fnb.reset();
    im.loaded = false;
}
bool CudaEncoder::loaded() const { return impl_->loaded; }

std::size_t CudaEncoder::device_bytes() const {
    const auto& im = *impl_;
    std::size_t n = im.ws.n + im.patch.w.n + im.patch.b.n + im.enw.n + im.enb.n + im.fnw.n + im.fnb.n;
    for (const auto& b : im.blocks)
        n += b.qkv.w.n + b.qkv.b.n + b.out.w.n + b.out.b.n + b.mlp1.w.n + b.mlp1.b.n + b.mlp2.w.n + b.mlp2.b.n + b.n1w.n + b.n1b.n + b.n2w.n + b.n2b.n;
    for (const auto& kv : im.projector) n += kv.second.w.n + kv.second.b.n;
    return n;
}

EncoderOutput CudaEncoder::encode(const PatchGrid& grid, const PositionInputs& pos) {
    auto& im = *impl_;
    const auto& c = im.cfg;
    if (!im.loaded) throw std::logic_error("vision cuda: encode before load");
    validate(c, grid, pos);
    const std::vector<unsigned> tiles = tower::tile_ends(grid);
    const unsigned N = grid.rows;
    const bool addend = !pos.addend.empty();
    const unsigned chunk = tower::mlp_chunk_rows(c, N, im.opt.mlp_workspace_bytes);
    const tower::Buffers b = tower::buffers(c, N, chunk, addend);
    const std::size_t floats = b.total();
    if (im.ws.n < floats * 4U) {
        im.ws.reset();
        im.ws = Dev(floats * 4U);
    }
    im.N = N;
    float* cur = im.ws.f();
    const auto take = [&](std::size_t count) {
        float* r = count ? cur : nullptr;
        cur += count;
        return r;
    };
    im.x = take(b.x);
    im.nrm = take(b.nrm);
    im.qkv = take(b.qkv);
    im.ao = take(b.ao);
    im.mid = take(b.mid);
    im.act = take(b.act);
    im.patches = take(b.patches);
    im.pos = take(b.pos);
    im.cs = take(b.trig);
    im.sn = take(b.trig);
    ck(cudaMemcpy(im.patches, grid.data.data(), grid.data.size() * 4U, cudaMemcpyHostToDevice), "patch upload");
    if (addend) ck(cudaMemcpy(im.pos, pos.addend.data(), pos.addend.size() * 4U, cudaMemcpyHostToDevice), "position upload");
    if (c.has_rope()) {
        std::vector<float> cs(pos.angles.size()), sn(pos.angles.size());
        for (std::size_t i = 0; i < cs.size(); ++i) {
            cs[i] = static_cast<float>(std::cos(static_cast<double>(pos.angles[i])));
            sn[i] = static_cast<float>(std::sin(static_cast<double>(pos.angles[i])));
        }
        ck(cudaMemcpy(im.cs, cs.data(), cs.size() * 4U, cudaMemcpyHostToDevice), "angle upload");
        ck(cudaMemcpy(im.sn, sn.data(), sn.size() * 4U, cudaMemcpyHostToDevice), "angle upload");
    }
    EncoderOutput out;
    out.rows = N;
    out.width = c.width;
    struct Steps {  // the names the plan calls onto the implementation's step bodies
        CudaEncoder::Impl& i;
        void embed() { i.embed(); }
        void add_positions() { i.add_positions(); }
        void norm_embed() { i.norm_embed(); }
        void attn_norm(unsigned l) { i.attn_norm(l); }
        void qkv(unsigned l) { i.qkv_step(l); }
        void rope(unsigned l) { i.rope(l); }
        void attend(unsigned l, unsigned r0, unsigned n) { i.attend(l, r0, n); }
        void out_proj(unsigned l) { i.out_proj(l); }
        void mlp_norm(unsigned l) { i.mlp_norm(l); }
        void mlp(unsigned l, unsigned r0, unsigned nr) { i.mlp(l, r0, nr); }
        void tap(unsigned l, std::vector<std::vector<float>>& t) { i.tap(l, t); }
        void final_norm() { i.final_norm(); }
    } steps{im};
    im.hidden = im.x;
    tower::drive(c, tiles, N, chunk, addend, steps, out);
    ck(cudaDeviceSynchronize(), "encode");
    out.hidden = im.download(im.hidden);
    return out;
}

void CudaEncoder::prepare_linear(const Tensor& w, const Tensor& bias) { (void)impl_->panel_for(w, bias); }

void CudaEncoder::linear(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) {
    auto& im = *impl_;
    if (!x || !y) throw std::invalid_argument("vision cuda: linear operands");
    const Panel& p = im.panel_for(w, bias);
    Dev dx(rows * p.k * 4U), dy(rows * p.n * 4U);
    ck(cudaMemcpy(dx.p, x, dx.n, cudaMemcpyHostToDevice), "linear upload");
    ck(cv::gemm_nt(dx.f(), p.w.f(), p.b.f(), dy.f(), static_cast<unsigned>(rows), p.n, p.k, p.n, false, nullptr), "linear");
    ck(cudaMemcpy(y, dy.p, dy.n, cudaMemcpyDeviceToHost), "linear download");
}

}  // namespace strix::vision
