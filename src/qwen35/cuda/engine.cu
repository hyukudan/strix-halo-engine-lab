// Qwen3.8-27B dense family (`qwen35`) CUDA engine over the shared cuda_core. See include/strix/qwen35/cuda_engine.hpp.
// Home: specific (model layer composition); all arithmetic is in src/cuda_core (shared).
#include "strix/qwen35/cuda_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/nvfp4.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/rope.hpp"
#include "strix/gguf_reader.hpp"

namespace cc = strix::cuda_core;

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        const cudaError_t e__ = (x);                                                                           \
        if (e__ != cudaSuccess) {                                                                              \
            throw std::runtime_error(std::string(#x) + " failed: " + cudaGetErrorString(e__) + " at line " + \
                                     std::to_string(__LINE__));                                                \
        }                                                                                                      \
    } while (0)

namespace strix::qwen35 {
namespace {

// ---------------------------------------------------------------- config from GGUF (until config_from_gguf lands here)
std::uint64_t meta_u(const gguf::FileInfo& f, const std::string& k) {
    const auto* e = f.find_metadata(k);
    if (!e) throw std::runtime_error("missing metadata " + k);
    if (e->unsigned_value) return *e->unsigned_value;
    if (e->signed_value) return static_cast<std::uint64_t>(*e->signed_value);
    throw std::runtime_error("metadata not integer: " + k);
}
double meta_f(const gguf::FileInfo& f, const std::string& k) {
    const auto* e = f.find_metadata(k);
    if (!e || !e->floating_value) throw std::runtime_error("missing float metadata " + k);
    return *e->floating_value;
}
Config read_cfg(const gguf::FileInfo& f) {
    const auto* arch = f.find_metadata("general.architecture");
    if (!arch || !arch->string_value || *arch->string_value != "qwen35") throw std::runtime_error("architecture is not qwen35");
    Config c;
    c.layers = static_cast<std::uint32_t>(meta_u(f, "qwen35.block_count"));
    c.hidden = static_cast<std::uint32_t>(meta_u(f, "qwen35.embedding_length"));
    c.intermediate = static_cast<std::uint32_t>(meta_u(f, "qwen35.feed_forward_length"));
    c.heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.attention.head_count"));
    c.kv_heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.attention.head_count_kv"));
    c.head_dim = static_cast<std::uint32_t>(meta_u(f, "qwen35.attention.key_length"));
    c.rms_eps = static_cast<float>(meta_f(f, "qwen35.attention.layer_norm_rms_epsilon"));
    c.rope_theta = meta_f(f, "qwen35.rope.freq_base");
    c.rotary_dim = static_cast<std::uint32_t>(meta_u(f, "qwen35.rope.dimension_count"));
    c.gdn_conv_kernel = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.conv_kernel"));
    c.gdn_key_dim = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.state_size"));
    c.gdn_key_heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.group_count"));
    c.gdn_value_heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.time_step_rank"));
    c.gdn_value_dim = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.inner_size")) / c.gdn_value_heads;
    c.full_attention_interval = static_cast<std::uint32_t>(meta_u(f, "qwen35.full_attention_interval"));
    c.mtp_layers = f.find_metadata("qwen35.nextn_predict_layers") ? static_cast<std::uint32_t>(meta_u(f, "qwen35.nextn_predict_layers")) : 0;
    for (const auto& t : f.tensors)
        if (t.name == "token_embd.weight") c.vocab = static_cast<std::uint32_t>(t.dimensions.at(1));
    if (c.vocab == 0) throw std::runtime_error("token_embd.weight missing");
    return c;
}

// ---------------------------------------------------------------- tensors
struct PackTensorSpec {
    std::string name;
    int type = 0;  // ggml: 30 bf16, 8 q8_0, 0 f32
    std::uint64_t in = 0, out = 1, bytes = 0, file_off = 0;
    // TURBO onfly: the source is BF16 (src_bytes) and the stored copy is NVFP4 (bytes); gscale is set at conversion.
    bool fp4 = false;
    std::uint64_t src_bytes = 0;
    float gscale = 1.0f;
};

// Tensors that take the NVFP4 path: 2-D BF16 weights (every projection, the MTP eh_proj, optional lm_head), never the
// embedding table (row gather) nor norms / conv / state tensors.
bool fp4_eligible(const EngineOptions& o, const PackTensorSpec& t) {
    if (o.turbo_weights != TurboWeights::onfly || t.type != 30 || t.out <= 1) return false;
    if (t.name == "token_embd.weight") return false;
    if (t.name == "output.weight" && !o.turbo_lm_head) return false;
    if (t.name.size() < 7 || t.name.compare(t.name.size() - 7, 7, ".weight") != 0) return false;
    return cc::nvfp4_shape_ok(static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in));
}
void mark_fp4(const EngineOptions& o, std::vector<PackTensorSpec>& v) {
    for (auto& t : v) {
        if (!fp4_eligible(o, t)) continue;
        t.fp4 = true;
        t.src_bytes = t.bytes;
        t.bytes = cc::nvfp4_bytes(static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in));
    }
}

std::uint64_t bytes_of(int type, std::uint64_t in, std::uint64_t out) {
    if (type == 30) return in * out * 2;
    if (type == 8) return out * (in / 32) * 34;
    return in * out * 4;
}

// Synthetic fill: counter-based RNG (splitmix64), deterministic in (seed, tensor, index).
enum class Fill : int { weight = 0, affine = 1 };
struct FillSpec {
    Fill kind = Fill::weight;
    float base = 0.0f, amp = 1.0f;  // value = base + amp * u, u in [-1, 1)
};

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float urand(std::uint64_t seed, std::uint64_t idx) {
    const std::uint64_t h = mix64(seed ^ mix64(idx));
    return static_cast<float>(static_cast<std::int32_t>(h >> 40)) * (1.0f / 8388608.0f) - 1.0f;  // exact in float: 24-bit / 2^23 - 1
}

__global__ void fill_bf16_kernel(std::uint16_t* p, std::uint64_t n, std::uint64_t seed, float base, float amp) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        p[i] = __bfloat16_as_ushort(__float2bfloat16_rn(base + amp * urand(seed, i)));
}
__global__ void fill_f32_kernel(float* p, std::uint64_t n, std::uint64_t seed, float base, float amp) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        p[i] = base + amp * urand(seed, i);
}
// q8_0: block = {f16 d, int8 q[32]}; value ~ amp * u: q = round(127 * u), d = amp / 127 (x (0.75 + 0.25 u2) per block).
__global__ void fill_q8_kernel(unsigned char* p, std::uint64_t nblocks, std::uint64_t seed, float amp) {
    for (std::uint64_t b = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; b < nblocks;
         b += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        unsigned char* q = p + b * 34;
        const float d = amp / 127.0f * (0.75f + 0.25f * urand(seed ^ 0x5555ull, b));
        const std::uint16_t hb = __half_as_ushort(__float2half_rn(d));
        q[0] = static_cast<unsigned char>(hb & 0xFF);
        q[1] = static_cast<unsigned char>(hb >> 8);
        for (int j = 0; j < 32; ++j) q[2 + j] = static_cast<unsigned char>(static_cast<std::int8_t>(std::lrintf(127.0f * urand(seed, b * 32 + j))));
    }
}

std::uint64_t fnv(const std::string& s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : s) h = (h ^ ch) * 1099511628211ull;
    return h;
}

struct WeightStore {
    unsigned char* base = nullptr;
    std::uint64_t total = 0, trunk_bytes = 0;
    std::map<std::string, PackTensorSpec> spec;
    std::map<std::string, std::uint64_t> off;
    [[nodiscard]] bool has(const std::string& n) const { return off.count(n) != 0; }
    [[nodiscard]] const void* ptr(const std::string& n) const { return base + off.at(n); }
    [[nodiscard]] cc::DenseWeight dense(const std::string& n) const {
        const auto& t = spec.at(n);
        cc::DenseWeight w;
        w.data = ptr(n);
        w.in = static_cast<std::uint32_t>(t.in);
        w.out = static_cast<std::uint32_t>(t.out);
        if (t.fp4) {
            w.format = cc::WeightFormat::nvfp4;
            w.gscale = t.gscale;
            return w;
        }
        switch (t.type) {
            case 30: w.format = cc::WeightFormat::bf16; break;
            case 8: w.format = cc::WeightFormat::q8_0; break;
            case 0: w.format = cc::WeightFormat::f32; break;
            default: throw std::runtime_error("unsupported ggml type for " + n);
        }
        return w;
    }
    [[nodiscard]] const float* f32(const std::string& n) const {
        if (spec.at(n).type != 0) throw std::runtime_error(n + " must be F32");
        return static_cast<const float*>(ptr(n));
    }
};

void layout(WeightStore& W, const std::vector<PackTensorSpec>& specs, std::uint32_t layers) {
    std::uint64_t total = 0;
    for (const auto& t : specs) {
        W.spec[t.name] = t;
        W.off[t.name] = total;
        total += (t.bytes + 255) & ~255ULL;
        const bool mtp = t.name.rfind("blk." + std::to_string(layers) + ".", 0) == 0;
        if (t.name != "token_embd.weight" && !mtp) W.trunk_bytes += t.bytes;
    }
    W.total = total;
    CK(cudaMalloc(reinterpret_cast<void**>(&W.base), total));
}

std::vector<PackTensorSpec> specs_from_gguf(const gguf::FileInfo& f, std::uint32_t layers, bool load_mtp) {
    std::vector<PackTensorSpec> v;
    for (const auto& t : f.tensors) {
        const std::string pre = "blk." + std::to_string(layers) + ".";
        if (t.name.rfind(pre, 0) == 0 && !load_mtp) continue;
        if (t.name.rfind("blk.", 0) == 0) {
            const std::uint32_t idx = static_cast<std::uint32_t>(std::stoul(t.name.substr(4)));
            if (idx > layers) continue;
        }
        PackTensorSpec s;
        s.name = t.name;
        s.type = static_cast<int>(t.ggml_type);
        s.in = t.dimensions.at(0);
        s.out = t.dimensions.size() > 1 ? t.dimensions[1] : 1;
        s.bytes = t.byte_size;
        s.file_off = t.absolute_offset;
        v.push_back(s);
    }
    return v;
}

// Tensor list of the qwen35 contract (design section 2) plus the fill rule of each tensor.
std::vector<std::pair<PackTensorSpec, FillSpec>> synth_specs(const Config& c, PackFormat pack, bool with_mtp) {
    std::vector<std::pair<PackTensorSpec, FillSpec>> v;
    const int wt = pack == PackFormat::bf16 ? 30 : 8;
    auto W = [&](const std::string& n, std::uint64_t in, std::uint64_t out, float gain) {
        PackTensorSpec s;
        s.name = n;
        s.type = wt;
        s.in = in;
        s.out = out;
        s.bytes = bytes_of(wt, in, out);
        v.push_back({s, FillSpec{Fill::weight, 0.0f, gain * std::sqrt(3.0f / static_cast<float>(in))}});
    };
    auto F = [&](const std::string& n, std::uint64_t cnt, std::uint64_t cols, float base, float amp) {
        PackTensorSpec s;
        s.name = n;
        s.type = 0;
        s.in = cols;
        s.out = cnt;
        s.bytes = cnt * cols * 4;
        v.push_back({s, FillSpec{Fill::affine, base, amp}});
    };
    const std::uint64_t H = c.hidden;
    auto attn_block = [&](const std::string& p) {
        W(p + "attn_q.weight", H, c.q_gate_width(), 1.0f);
        W(p + "attn_k.weight", H, c.kv_width(), 1.0f);
        W(p + "attn_v.weight", H, c.kv_width(), 1.0f);
        W(p + "attn_output.weight", c.heads * c.head_dim, H, 0.5f);
        F(p + "attn_q_norm.weight", 1, c.head_dim, 1.0f, 0.1f);
        F(p + "attn_k_norm.weight", 1, c.head_dim, 1.0f, 0.1f);
    };
    auto ffn_block = [&](const std::string& p) {
        W(p + "ffn_gate.weight", H, c.intermediate, 1.0f);
        W(p + "ffn_up.weight", H, c.intermediate, 1.0f);
        W(p + "ffn_down.weight", c.intermediate, H, 0.5f);
    };
    W("token_embd.weight", H, c.vocab, 1.0f / std::sqrt(3.0f / static_cast<float>(H)));
    F("output_norm.weight", 1, H, 1.0f, 0.1f);
    W("output.weight", H, c.vocab, 1.0f);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        F(p + "attn_norm.weight", 1, H, 1.0f, 0.1f);
        F(p + "post_attention_norm.weight", 1, H, 1.0f, 0.1f);
        if (c.kind(l) == LayerKind::gated_delta_net) {
            W(p + "attn_qkv.weight", H, c.gdn_conv_channels(), 1.0f);
            W(p + "attn_gate.weight", H, c.gdn_v_width(), 1.0f);
            W(p + "ssm_beta.weight", H, c.gdn_value_heads, 1.0f);
            W(p + "ssm_alpha.weight", H, c.gdn_value_heads, 1.0f);
            F(p + "ssm_conv1d.weight", c.gdn_conv_channels(), c.gdn_conv_kernel, 0.0f, 0.5f);
            F(p + "ssm_a", 1, c.gdn_value_heads, -1.0f, 0.5f);
            F(p + "ssm_dt.bias", 1, c.gdn_value_heads, 0.0f, 0.5f);
            F(p + "ssm_norm.weight", 1, c.gdn_value_dim, 1.0f, 0.1f);
            W(p + "ssm_out.weight", c.gdn_v_width(), H, 0.5f);
        } else {
            attn_block(p);
        }
        ffn_block(p);
    }
    if (with_mtp && c.mtp_layers > 0) {
        const std::string p = "blk." + std::to_string(c.layers) + ".";
        F(p + "attn_norm.weight", 1, H, 1.0f, 0.1f);
        F(p + "post_attention_norm.weight", 1, H, 1.0f, 0.1f);
        F(p + "nextn.enorm.weight", 1, H, 1.0f, 0.1f);
        F(p + "nextn.hnorm.weight", 1, H, 1.0f, 0.1f);
        W(p + "nextn.eh_proj.weight", 2 * H, H, 1.0f);
        F(p + "nextn.shared_head_norm.weight", 1, H, 1.0f, 0.1f);
        attn_block(p);
        ffn_block(p);
    }
    return v;
}

struct LayerW {
    bool attn = false;
    cc::DenseWeight qkv, gate, beta, alpha, out;  // GDN (out = ssm_out)
    cc::DenseWeight q, k, v, o;                    // attention
    cc::DenseWeight ffn_gate, ffn_up, ffn_down;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    const float *conv = nullptr, *ssm_a = nullptr, *dt = nullptr, *ssm_norm = nullptr, *qn = nullptr, *kn = nullptr;
};

struct InputPack {
    std::uint32_t pos0;
    std::uint32_t ids[cc::kMaxRows];
};
__global__ void set_inputs_kernel(InputPack in, std::uint32_t* pos, std::uint32_t* ids) {
    if (threadIdx.x == 0) *pos = in.pos0;
    if (threadIdx.x < cc::kMaxRows) ids[threadIdx.x] = in.ids[threadIdx.x];
}
__global__ void zero_u32_kernel(std::uint32_t* p) { *p = 0; }

}  // namespace

// ================================================================ Impl
struct CudaEngine::Impl {
    Config c;
    EngineOptions opt;
    WeightStore W;
    std::vector<LayerW> L;
    LayerW M;  // MTP block
    bool mtp_ok = false;
    cc::DenseWeight head, embd;
    const float* out_norm = nullptr;
    const float *m_enorm = nullptr, *m_hnorm = nullptr, *m_shnorm = nullptr;
    cc::DenseWeight m_eh;

    static constexpr std::uint32_t RM = cc::kMaxRows;
    cudaStream_t s = nullptr;
    cc::GdnDims gd;
    cc::AttentionDims ad;

    // activations
    float *x = nullptr, *xn = nullptr, *z = nullptr, *gout = nullptr, *qg = nullptr, *q = nullptr, *k = nullptr, *v = nullptr,
          *ao = nullptr, *hid = nullptr, *logits = nullptr, *scratch = nullptr, *cos_t = nullptr, *sin_t = nullptr;
    float *mx = nullptr, *mxn = nullptr, *mcat = nullptr, *mtmp = nullptr, *mlogits = nullptr;
    std::uint32_t *d_ids = nullptr, *d_pos = nullptr, *d_top = nullptr, *d_parity = nullptr;
    void* d_argws = nullptr;
    std::uint32_t *d_mids = nullptr, *d_mpos = nullptr, *d_mtop = nullptr;
    // per-layer state / window buffers
    std::vector<float*> conv, delta, raw, qkvc, ar, br;
    std::vector<__nv_bfloat16*> kc, vc;
    __nv_bfloat16 *mkc = nullptr, *mvc = nullptr;
    std::vector<cc::GdnCommitLayer> commit_layers;

    std::uint32_t committed = 0, last_rows = 0;
    bool graphs_on = true;
    struct Graph {
        cudaGraphExec_t exec = nullptr;
        std::uint64_t nodes = 0;
    };
    std::unordered_map<std::uint32_t, Graph> graphs;
    std::uint64_t last_nodes = 0;

    template <class T>
    T* alloc(std::size_t n, bool zero = false) {
        T* p = nullptr;
        CK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
        if (zero) CK(cudaMemset(p, 0, n * sizeof(T)));
        return p;
    }
    [[nodiscard]] std::string nm(std::uint32_t l, const char* suffix) const { return "blk." + std::to_string(l) + "." + suffix; }

    LayerW resolve(std::uint32_t l, bool attn) const {
        LayerW w;
        w.attn = attn;
        w.attn_norm = W.f32(nm(l, "attn_norm.weight"));
        w.post_norm = W.f32(nm(l, "post_attention_norm.weight"));
        if (!attn) {
            w.qkv = W.dense(nm(l, "attn_qkv.weight"));
            w.gate = W.dense(nm(l, "attn_gate.weight"));
            w.beta = W.dense(nm(l, "ssm_beta.weight"));
            w.alpha = W.dense(nm(l, "ssm_alpha.weight"));
            w.out = W.dense(nm(l, "ssm_out.weight"));
            w.conv = W.f32(nm(l, "ssm_conv1d.weight"));
            w.ssm_a = W.f32(nm(l, "ssm_a"));
            w.dt = W.f32(nm(l, "ssm_dt.bias"));
            w.ssm_norm = W.f32(nm(l, "ssm_norm.weight"));
        } else {
            w.q = W.dense(nm(l, "attn_q.weight"));
            w.k = W.dense(nm(l, "attn_k.weight"));
            w.v = W.dense(nm(l, "attn_v.weight"));
            w.o = W.dense(nm(l, "attn_output.weight"));
            w.qn = W.f32(nm(l, "attn_q_norm.weight"));
            w.kn = W.f32(nm(l, "attn_k_norm.weight"));
        }
        w.ffn_gate = W.dense(nm(l, "ffn_gate.weight"));
        w.ffn_up = W.dense(nm(l, "ffn_up.weight"));
        w.ffn_down = W.dense(nm(l, "ffn_down.weight"));
        return w;
    }

    void init() {
        cc::set_programmatic_launch(opt.pdl);
        gd = {c.gdn_key_heads, c.gdn_key_dim, c.gdn_value_heads, c.gdn_value_dim, c.gdn_conv_kernel};
        ad.heads = c.heads;
        ad.kv_heads = c.kv_heads;
        ad.head_dim = c.head_dim;
        ad.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.head_dim)));
        ad.max_positions = opt.max_positions;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        {   // NVFP4 split-K workspace: grow-only, must exist before graph capture. Bound = rows 32, all jobs of the largest layer.
            std::size_t need = 0, layer_sum = 0;
            for (const auto& [name, t] : W.spec) {
                if (!t.fp4) continue;
                const std::size_t ks = cc::nvfp4_split_k(static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in));
                const std::size_t b = 2 * ks * cc::kNvfp4MaxRows * t.out * 4;
                layer_sum += b;
                need = std::max(need, b);
            }
            if (layer_sum) CK(cc::nvfp4_reserve_workspace(std::min<std::size_t>(layer_sum, std::max<std::size_t>(need * 4, 64u << 20))));
        }
        for (std::uint32_t l = 0; l < c.layers; ++l) L.push_back(resolve(l, c.kind(l) == LayerKind::full_attention));
        head = W.has("output.weight") ? W.dense("output.weight") : W.dense("token_embd.weight");
        embd = W.dense("token_embd.weight");
        out_norm = W.f32("output_norm.weight");
        mtp_ok = opt.load_mtp && c.mtp_layers > 0 && W.has(nm(c.layers, "nextn.eh_proj.weight"));
        if (mtp_ok) {
            M = resolve(c.layers, true);
            m_enorm = W.f32(nm(c.layers, "nextn.enorm.weight"));
            m_hnorm = W.f32(nm(c.layers, "nextn.hnorm.weight"));
            m_shnorm = W.f32(nm(c.layers, "nextn.shared_head_norm.weight"));
            m_eh = W.dense(nm(c.layers, "nextn.eh_proj.weight"));
        }
        const std::uint32_t H = c.hidden;
        x = alloc<float>(RM * H);
        xn = alloc<float>(RM * H);
        z = alloc<float>(RM * gd.v_width());
        gout = alloc<float>(RM * gd.v_width());
        qg = alloc<float>(static_cast<std::size_t>(RM) * c.q_gate_width());
        q = alloc<float>(static_cast<std::size_t>(RM) * c.heads * c.head_dim);
        k = alloc<float>(RM * c.kv_width());
        v = alloc<float>(RM * c.kv_width());
        ao = alloc<float>(static_cast<std::size_t>(RM) * c.heads * c.head_dim);
        hid = alloc<float>(static_cast<std::size_t>(RM) * c.intermediate);
        logits = alloc<float>(static_cast<std::size_t>(RM) * c.vocab);
        scratch = alloc<float>(cc::attention_scratch_floats(ad, RM));
        d_ids = alloc<std::uint32_t>(RM);
        d_pos = alloc<std::uint32_t>(1);
        d_top = alloc<std::uint32_t>(RM);
        d_parity = alloc<std::uint32_t>(1, true);
        d_argws = alloc<unsigned char>(cc::kArgmaxWorkspaceBytes);
        if (mtp_ok) {
            mx = alloc<float>(RM * H);
            mxn = alloc<float>(RM * H);
            mcat = alloc<float>(2 * RM * H);
            mtmp = alloc<float>(RM * H);
            mlogits = alloc<float>(static_cast<std::size_t>(RM) * c.vocab);
            d_mids = alloc<std::uint32_t>(RM);
            d_mpos = alloc<std::uint32_t>(1);
            d_mtop = alloc<std::uint32_t>(RM);
            const std::size_t n = static_cast<std::size_t>(c.kv_heads) * opt.max_positions * c.head_dim;
            mkc = alloc<__nv_bfloat16>(n, true);
            mvc = alloc<__nv_bfloat16>(n, true);
        }
        std::vector<float> ct, st;
        cc::rope_table_host(opt.max_positions, c.rotary_dim, c.rope_theta, ct, st);
        cos_t = alloc<float>(ct.size());
        sin_t = alloc<float>(st.size());
        CK(cudaMemcpy(cos_t, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sin_t, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
        const std::size_t kvn = static_cast<std::size_t>(c.kv_heads) * opt.max_positions * c.head_dim;
        const std::size_t dside = static_cast<std::size_t>(gd.value_heads) * gd.key_dim * gd.value_dim;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) {
                kc.push_back(alloc<__nv_bfloat16>(kvn, true));
                vc.push_back(alloc<__nv_bfloat16>(kvn, true));
                conv.push_back(nullptr);
                delta.push_back(nullptr);
                raw.push_back(nullptr);
                qkvc.push_back(nullptr);
                ar.push_back(nullptr);
                br.push_back(nullptr);
            } else {
                kc.push_back(nullptr);
                vc.push_back(nullptr);
                conv.push_back(alloc<float>(static_cast<std::size_t>(gd.conv_channels()) * 3, true));
                delta.push_back(alloc<float>(2 * dside, true));
                raw.push_back(alloc<float>(static_cast<std::size_t>(RM) * gd.conv_channels()));
                qkvc.push_back(alloc<float>(static_cast<std::size_t>(RM) * gd.conv_channels()));
                ar.push_back(alloc<float>(RM * gd.value_heads));
                br.push_back(alloc<float>(RM * gd.value_heads));
                commit_layers.push_back({{L[l].conv, L[l].ssm_a, L[l].dt, L[l].ssm_norm}, {conv[l], delta[l]},
                                         {raw[l], qkvc[l], ar[l], br[l]}});
            }
        }
        CK(cudaDeviceSynchronize());
    }

    void reset() {
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) continue;
            CK(cudaMemsetAsync(conv[l], 0, static_cast<std::size_t>(gd.conv_channels()) * 3 * 4, s));
            CK(cudaMemsetAsync(delta[l], 0, 2 * static_cast<std::size_t>(gd.value_heads) * gd.key_dim * gd.value_dim * 4, s));
        }
        CK(cudaMemsetAsync(d_parity, 0, 4, s));
        CK(cudaStreamSynchronize(s));
        committed = 0;
        last_rows = 0;
    }

    // ---- one attention layer body (shared by the trunk and the MTP block): mixer + MLP on rows of xa (residual stream).
    struct Bufs {
        float *x, *xn;
        const std::uint32_t* pos;
        __nv_bfloat16 *kc, *vc;
    };

    // GEMVs that share x: one launch when format and in agree (bitwise equal either way), else separate launches.
    void multi(const cc::DenseWeight* ws, float* const* ys, int n, const float* xin, std::uint32_t rows) {
        bool ok = true;
        for (int j = 1; j < n; ++j) ok = ok && ws[j].format == ws[0].format && ws[j].in == ws[0].in;
        if (ok && opt.multi_gemv) {
            CK(cc::dense_rows_multi(ws, ys, n, xin, rows, nullptr, s));
        } else {
            for (int j = 0; j < n; ++j) CK(cc::dense_rows(ws[j], xin, ys[j], rows, nullptr, s));
        }
    }

    void attn_mixer(const LayerW& w, const Bufs& b, std::uint32_t rows) {
        const std::uint32_t D = c.head_dim;
        const std::uint32_t qgw = c.heads * D * 2;
        {
            const cc::DenseWeight ws[3] = {w.q, w.k, w.v};
            float* const ys[3] = {qg, k, v};
            multi(ws, ys, 3, b.xn, rows);
        }
        const cc::GroupedView qsrc{rows, c.heads, D, qgw, 2 * D}, qdst{rows, c.heads, D, c.heads * D, D};
        const cc::GroupedView kvv{rows, c.kv_heads, D, c.kv_heads * D, D};
        CK(cc::rmsnorm(qg, qsrc, w.qn, q, qdst, c.rms_eps, s));
        CK(cc::rmsnorm(k, kvv, w.kn, k, kvv, c.rms_eps, s));
        CK(cc::rope_apply(q, qdst, c.rotary_dim, cos_t, sin_t, b.pos, s));
        CK(cc::rope_apply(k, kvv, c.rotary_dim, cos_t, sin_t, b.pos, s));
        CK(cc::kv_store(ad, k, v, b.kc, b.vc, b.pos, rows, s));
        CK(cc::attention_decode(ad, q, qdst, qg + D, qsrc, b.kc, b.vc, b.pos, rows, ao, scratch, s));
        CK(cc::dense_rows(w.o, ao, b.x, rows, b.x, s));
    }

    void mlp(const LayerW& w, const Bufs& b, std::uint32_t rows) {
        const cc::GroupedView hv{rows, 1, c.hidden, c.hidden, c.hidden};
        CK(cc::rmsnorm(b.x, hv, w.post_norm, b.xn, hv, c.rms_eps, s));
        CK(cc::dense_gate_up_silu_rows(w.ffn_gate, w.ffn_up, b.xn, hid, rows, s));
        CK(cc::dense_rows(w.ffn_down, hid, b.x, rows, b.x, s));
    }

    void trunk(std::uint32_t rows, LogitRows want) {
        const std::uint32_t H = c.hidden;
        const cc::GroupedView hv{rows, 1, H, H, H};
        CK(cc::embedding_rows(embd, d_ids, x, rows, s));
        std::uint32_t gi = 0;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            const LayerW& w = L[l];
            CK(cc::rmsnorm(x, hv, w.attn_norm, xn, hv, c.rms_eps, s));
            if (!w.attn) {
                {
                    const cc::DenseWeight ws[4] = {w.qkv, w.gate, w.beta, w.alpha};
                    float* const ys[4] = {raw[l], z, br[l], ar[l]};
                    multi(ws, ys, 4, xn, rows);
                }
                const auto& cl = commit_layers[gi++];
                CK(cc::gdn_window(gd, cl.layer, cl.state, cl.win, z, gout, rows, c.rms_eps, d_parity, s));
                CK(cc::dense_rows(w.out, gout, x, rows, x, s));
            } else {
                attn_mixer(w, Bufs{x, xn, d_pos, kc[l], vc[l]}, rows);
            }
            mlp(w, Bufs{x, xn, d_pos, nullptr, nullptr}, rows);
        }
        CK(cc::rmsnorm(x, hv, out_norm, xn, hv, c.rms_eps, s));
        if (want == LogitRows::all) {
            CK(cc::dense_rows(head, xn, logits, rows, nullptr, s));
            CK(cc::argmax_rows(logits, c.vocab, rows, d_top, d_argws, s));
        } else if (want == LogitRows::last) {
            CK(cc::dense_rows(head, xn + static_cast<std::size_t>(rows - 1) * H, logits, 1, nullptr, s));
            CK(cc::argmax_rows(logits, c.vocab, 1, d_top, d_argws, s));
        }
    }

    RowsOut forward(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want) {
        if (rows == 0 || rows > RM) throw std::runtime_error("forward_rows: rows must be 1..32");
        if (committed + rows > opt.max_positions) throw std::runtime_error("forward_rows: max_positions exceeded");
        InputPack in{};
        in.pos0 = committed;
        for (std::uint32_t i = 0; i < rows; ++i) in.ids[i] = tokens[i];
        set_inputs_kernel<<<1, 32, 0, s>>>(in, d_pos, d_ids);
        CK(cudaGetLastError());
        const std::uint32_t key = rows * 4 + static_cast<std::uint32_t>(want);
        if (graphs_on) {
            auto it = graphs.find(key);
            if (it == graphs.end()) {
                cudaGraph_t g = nullptr;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                try {
                    trunk(rows, want);
                } catch (...) {
                    cudaGraph_t dummy = nullptr;
                    cudaStreamEndCapture(s, &dummy);
                    if (dummy) cudaGraphDestroy(dummy);
                    throw;
                }
                CK(cudaStreamEndCapture(s, &g));
                Graph gr;
                CK(cudaGraphInstantiate(&gr.exec, g, 0));
                std::size_t n = 0;
                CK(cudaGraphGetNodes(g, nullptr, &n));
                gr.nodes = n;
                CK(cudaGraphDestroy(g));
                it = graphs.emplace(key, gr).first;
            }
            last_nodes = it->second.nodes;
            CK(cudaGraphLaunch(it->second.exec, s));
        } else {
            trunk(rows, want);
            last_nodes = 0;
        }
        last_rows = rows;
        RowsOut o;
        o.logits = want == LogitRows::none ? nullptr : logits;
        o.top = want == LogitRows::none ? nullptr : d_top;
        o.hidden_post = xn;
        o.hidden_pre = x;
        return o;
    }

    void commit_rows(std::uint32_t kk) {
        if (kk > last_rows) throw std::runtime_error("commit: k exceeds the last window");
        if (kk > 0) {
            CK(cc::gdn_commit(gd, commit_layers.data(), static_cast<std::uint32_t>(commit_layers.size()), last_rows, kk, d_parity, s));
        }
        committed += kk;
        last_rows = 0;
    }

    // ============================================================ large-M prefill (see cuda_engine.hpp prefill())
    // A SLOT is the buffer set + stream + GEMM scratch of one chunk in flight. With 2 slots chunk c runs on slot c % 2; chunk c layer l
    // only waits for chunk c-1 layer l (GDN state, conv state, KV rows), so the sequential GDN delta kernel of one chunk overlaps the
    // GEMMs of the other. Every state hand-off keeps the single-chunk order, so the result is bitwise the same for any slot count.
    struct Slot {
        cudaStream_t st = nullptr;
        cc::BigCtx* bc = nullptr;
        float *x = nullptr, *xn = nullptr, *z = nullptr, *gout = nullptr, *raw = nullptr, *qkvc = nullptr, *ar = nullptr, *br = nullptr,
              *qg = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *hid = nullptr, *scr = nullptr;
        std::uint32_t *ids = nullptr, *wpos = nullptr, *pos = nullptr, *cpar = nullptr;
        // activation split cache for the current x
        const float* act_x = nullptr;
        std::uint32_t act_in = 0, act_rows = 0;
        cc::BigAct act[2];
        bool have[2] = {false, false};
    };
    struct Pf {
        std::uint32_t C = 0;
        std::vector<Slot> sl;
        int terms = 3;
    } pf;

    [[nodiscard]] bool big_supported() const {
        if (!opt.prefill_big || cc::dense_arith() == cc::DenseArith::fp32) return false;
        if (opt.prefill_chunk == 0 || opt.prefill_chunk % 128 != 0 || opt.prefill_chunk > 8192) return false;
        for (const LayerW& w : L) {
            const cc::DenseWeight all[] = {w.qkv, w.gate, w.beta, w.alpha, w.out, w.q, w.k, w.v, w.o, w.ffn_gate, w.ffn_up, w.ffn_down};
            for (const auto& d : all)
                if (d.data != nullptr && !cc::dense_big_supported(d)) return false;
        }
        return c.head_dim == 256 && gd.key_dim == 128 && gd.value_dim == 128;
    }

    void pf_make_slot(Slot& sl, std::uint32_t C) {
        const std::uint32_t H = c.hidden;
        CK(cudaStreamCreateWithFlags(&sl.st, cudaStreamNonBlocking));
        sl.bc = cc::dense_big_ctx_create();
        sl.x = alloc<float>(static_cast<std::size_t>(C) * H);
        sl.xn = alloc<float>(static_cast<std::size_t>(C) * H);
        sl.z = alloc<float>(static_cast<std::size_t>(C) * gd.v_width());
        sl.gout = alloc<float>(static_cast<std::size_t>(C) * gd.v_width());
        sl.raw = alloc<float>(static_cast<std::size_t>(C) * gd.conv_channels());
        sl.qkvc = alloc<float>(static_cast<std::size_t>(C) * gd.conv_channels());
        sl.ar = alloc<float>(static_cast<std::size_t>(C) * gd.value_heads);
        sl.br = alloc<float>(static_cast<std::size_t>(C) * gd.value_heads);
        sl.qg = alloc<float>(static_cast<std::size_t>(C) * c.q_gate_width());
        sl.q = alloc<float>(static_cast<std::size_t>(C) * c.heads * c.head_dim);
        sl.k = alloc<float>(static_cast<std::size_t>(C) * c.kv_width());
        sl.v = alloc<float>(static_cast<std::size_t>(C) * c.kv_width());
        sl.ao = alloc<float>(static_cast<std::size_t>(C) * c.heads * c.head_dim);
        sl.hid = alloc<float>(static_cast<std::size_t>(C) * c.intermediate);
        sl.ids = alloc<std::uint32_t>(C);
        sl.wpos = alloc<std::uint32_t>(C / 32 + 1);
        sl.pos = alloc<std::uint32_t>(1);
        sl.cpar = alloc<std::uint32_t>(1);
        if (opt.prefill_arith == PrefillArith::exact) sl.scr = alloc<float>(cc::attention_scratch_floats(ad, RM));
        CK(cc::dense_big_reserve(C, std::max(c.intermediate, H), static_cast<std::uint64_t>(c.intermediate) * H, 3, sl.bc));
    }

    void pf_init(std::uint32_t nslots) {
        if (pf.C != opt.prefill_chunk && !pf.sl.empty()) throw std::runtime_error("prefill: prefill_chunk cannot change after the first prefill");
        pf.C = opt.prefill_chunk;
        while (pf.sl.size() < nslots) {
            pf.sl.emplace_back();
            pf_make_slot(pf.sl.back(), pf.C);
        }
    }

    [[nodiscard]] int pf_terms() const {
        if (opt.prefill_arith == PrefillArith::fast) return 1;
        if (opt.prefill_arith == PrefillArith::fast2) return 2;
        const auto a = cc::dense_arith();
        return a == cc::DenseArith::mma3 ? 3 : a == cc::DenseArith::mma2 ? 2 : 1;
    }

    // the activation planes of x for weight w (split lazily, cached until x changes)
    const cc::BigAct& pf_act(Slot& sl, const float* x, std::uint32_t rows, const cc::DenseWeight& w) {
        if (sl.act_x != x || sl.act_in != w.in || sl.act_rows != rows) {
            sl.have[0] = sl.have[1] = false;
            sl.act_x = x;
            sl.act_in = w.in;
            sl.act_rows = rows;
        }
        const int cls = w.format == cc::WeightFormat::nvfp4 ? 1 : 0;
        if (!sl.have[cls]) {
            CK(cc::dense_big_split(x, rows, w.in, w.format, pf.terms, &sl.act[cls], sl.st, sl.bc));
            sl.have[cls] = true;
        }
        return sl.act[cls];
    }
    static void pf_dirty(Slot& sl) {
        sl.act_x = nullptr;
        sl.have[0] = sl.have[1] = false;
    }

    void pf_gemm(Slot& sl, const cc::DenseWeight& w, const float* x, std::uint32_t rows, float* y, const float* aux, cc::BigEpi epi) {
        const cc::BigAct& a = pf_act(sl, x, rows, w);
        CK(cc::dense_big_gemm(w, a, y, aux, epi, sl.st, sl.bc));
    }

    void pf_mlp(Slot& sl, const LayerW& w, std::uint32_t rows) {
        const cc::GroupedView hv{rows, 1, c.hidden, c.hidden, c.hidden};
        CK(cc::rmsnorm(sl.x, hv, w.post_norm, sl.xn, hv, c.rms_eps, sl.st));
        pf_dirty(sl);
        pf_gemm(sl, w.ffn_gate, sl.xn, rows, sl.hid, nullptr, cc::BigEpi::none);
        pf_gemm(sl, w.ffn_up, sl.xn, rows, sl.hid, sl.hid, cc::BigEpi::silu_mul);  // y = silu(gate) * up, in place
        pf_dirty(sl);
        pf_gemm(sl, w.ffn_down, sl.hid, rows, sl.x, sl.x, cc::BigEpi::residual);
        pf_dirty(sl);
    }

    void pf_attn(Slot& sl, std::uint32_t l, const LayerW& w, std::uint32_t rows, std::uint32_t pos0) {
        const std::uint32_t D = c.head_dim, qgw = c.heads * D * 2;
        pf_gemm(sl, w.q, sl.xn, rows, sl.qg, nullptr, cc::BigEpi::none);
        pf_gemm(sl, w.k, sl.xn, rows, sl.k, nullptr, cc::BigEpi::none);
        pf_gemm(sl, w.v, sl.xn, rows, sl.v, nullptr, cc::BigEpi::none);
        const cc::GroupedView qsrc{rows, c.heads, D, qgw, 2 * D}, qdst{rows, c.heads, D, c.heads * D, D};
        const cc::GroupedView kvv{rows, c.kv_heads, D, c.kv_heads * D, D};
        CK(cc::rmsnorm(sl.qg, qsrc, w.qn, sl.q, qdst, c.rms_eps, sl.st));
        CK(cc::rmsnorm(sl.k, kvv, w.kn, sl.k, kvv, c.rms_eps, sl.st));
        CK(cc::rope_apply(sl.q, qdst, c.rotary_dim, cos_t, sin_t, sl.pos, sl.st));
        CK(cc::rope_apply(sl.k, kvv, c.rotary_dim, cos_t, sin_t, sl.pos, sl.st));
        CK(cc::kv_store(ad, sl.k, sl.v, kc[l], vc[l], sl.pos, rows, sl.st));
        if (opt.prefill_arith == PrefillArith::exact) {
            for (std::uint32_t r0 = 0, wi = 0; r0 < rows; r0 += RM, ++wi) {
                const std::uint32_t rw = std::min<std::uint32_t>(RM, rows - r0);
                const cc::GroupedView qs{rw, c.heads, D, qgw, 2 * D}, qd{rw, c.heads, D, c.heads * D, D};
                CK(cc::attention_decode(ad, sl.q + static_cast<std::size_t>(r0) * c.heads * D, qd,
                                        sl.qg + static_cast<std::size_t>(r0) * qgw + D, qs, kc[l], vc[l], sl.wpos + wi, rw,
                                        sl.ao + static_cast<std::size_t>(r0) * c.heads * D, sl.scr, sl.st));
            }
        } else {
            CK(cc::attention_prefill_flash(ad, sl.q, qdst, sl.qg + D, qsrc, kc[l], vc[l], pos0, rows, sl.ao, sl.st));
        }
        pf_dirty(sl);
        pf_gemm(sl, w.o, sl.ao, rows, sl.x, sl.x, cc::BigEpi::residual);
        pf_dirty(sl);
    }

    // one chunk of `rows` tokens at positions pos0.. through all layers on slot sl; x ends in sl.x. prev[l] = event of the previous chunk's
    // layer l (null for the first chunk); mine[l] is recorded after this chunk's layer l.
    void pf_chunk(Slot& sl, const std::uint32_t* tokens, std::uint32_t rows, std::uint32_t pos0, std::uint32_t read_side,
                  const cudaEvent_t* prev, cudaEvent_t* mine) {
        const std::uint32_t H = c.hidden;
        const cc::GroupedView hv{rows, 1, H, H, H};
        CK(cudaMemcpyAsync(sl.ids, tokens, static_cast<std::size_t>(rows) * 4, cudaMemcpyHostToDevice, sl.st));
        CK(cudaMemcpyAsync(sl.pos, &pos0, 4, cudaMemcpyHostToDevice, sl.st));
        CK(cudaMemcpyAsync(sl.cpar, &read_side, 4, cudaMemcpyHostToDevice, sl.st));
        CK(cc::iota_u32(sl.wpos, (rows + RM - 1) / RM, pos0, RM, sl.st));
        CK(cc::embedding_rows(embd, sl.ids, sl.x, rows, sl.st));
        std::uint32_t gi = 0;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            const LayerW& w = L[l];
            if (prev != nullptr) CK(cudaStreamWaitEvent(sl.st, prev[l], 0));
            CK(cc::rmsnorm(sl.x, hv, w.attn_norm, sl.xn, hv, c.rms_eps, sl.st));
            pf_dirty(sl);
            if (!w.attn) {
                pf_gemm(sl, w.qkv, sl.xn, rows, sl.raw, nullptr, cc::BigEpi::none);
                pf_gemm(sl, w.gate, sl.xn, rows, sl.z, nullptr, cc::BigEpi::none);
                pf_gemm(sl, w.beta, sl.xn, rows, sl.br, nullptr, cc::BigEpi::none);
                pf_gemm(sl, w.alpha, sl.xn, rows, sl.ar, nullptr, cc::BigEpi::none);
                const auto& cl = commit_layers[gi++];
                CK(cc::gdn_prefill_layer(gd, cl.layer, cl.state, cc::GdnWindow{sl.raw, sl.qkvc, sl.ar, sl.br}, sl.z, sl.gout, rows, c.rms_eps,
                                         sl.cpar, sl.st));
                pf_gemm(sl, w.out, sl.gout, rows, sl.x, sl.x, cc::BigEpi::residual);
                pf_dirty(sl);
            } else {
                pf_attn(sl, l, w, rows, pos0);
            }
            pf_mlp(sl, w, rows);
            CK(cudaEventRecord(mine[l], sl.st));
        }
    }

    RowsOut prefill(const std::uint32_t* tokens, std::uint32_t n, LogitRows want) {
        if (n == 0) throw std::runtime_error("prefill: n must be >= 1");
        if (committed + n > opt.max_positions) throw std::runtime_error("prefill: max_positions exceeded");
        last_rows = 0;
        if (!big_supported()) {  // windows + commit
            RowsOut o;
            for (std::uint32_t p = 0; p < n; p += RM) {
                const std::uint32_t r = std::min<std::uint32_t>(RM, n - p);
                o = forward(tokens + p, r, p + r == n ? want : LogitRows::none);
                commit_rows(r);
            }
            return o;
        }
        const std::uint32_t C = opt.prefill_chunk, nchunks = (n + C - 1) / C, H = c.hidden;
        pf_init(std::min<std::uint32_t>(std::max<std::uint32_t>(1, opt.prefill_slots), nchunks));
        pf.terms = pf_terms();
        const std::uint32_t nslots = static_cast<std::uint32_t>(pf.sl.size());
        CK(cudaStreamSynchronize(s));
        std::uint32_t par0 = 0;
        CK(cudaMemcpy(&par0, d_parity, 4, cudaMemcpyDeviceToHost));
        cudaEvent_t start = nullptr;
        CK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
        CK(cudaEventRecord(start, s));
        std::vector<cudaEvent_t> ev(static_cast<std::size_t>(nchunks) * c.layers);
        for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (std::uint32_t ci = 0; ci < nchunks; ++ci) {
            Slot& sl = pf.sl[ci % nslots];
            if (ci < nslots) CK(cudaStreamWaitEvent(sl.st, start, 0));
            const std::uint32_t p = ci * C, r = std::min<std::uint32_t>(C, n - p);
            pf_chunk(sl, tokens + p, r, committed + p, (par0 + ci) & 1u, ci ? &ev[static_cast<std::size_t>(ci - 1) * c.layers] : nullptr,
                     &ev[static_cast<std::size_t>(ci) * c.layers]);
        }
        Slot& last = pf.sl[(nchunks - 1) % nslots];
        CK(cudaStreamWaitEvent(s, ev[static_cast<std::size_t>(nchunks) * c.layers - 1], 0));
        const std::uint32_t rl = n - (nchunks - 1) * C;
        // last token: final norm + lm_head on its row, exported through the decode buffers (row 0)
        CK(cudaMemcpyAsync(x, last.x + static_cast<std::size_t>(rl - 1) * H, static_cast<std::size_t>(H) * 4, cudaMemcpyDeviceToDevice, s));
        const cc::GroupedView h1{1, 1, H, H, H};
        CK(cc::rmsnorm(x, h1, out_norm, xn, h1, c.rms_eps, s));
        if (want != LogitRows::none) {
            CK(cc::dense_rows(head, xn, logits, 1, nullptr, s));
            CK(cc::argmax_rows(logits, c.vocab, 1, d_top, d_argws, s));
        }
        const std::uint32_t par_end = (par0 + nchunks) & 1u;
        CK(cudaMemcpyAsync(d_parity, &par_end, 4, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s));  // the staging copy above and the event pool are released after this point
        for (auto e : ev) cudaEventDestroy(e);
        cudaEventDestroy(start);
        committed += n;
        RowsOut o;
        o.logits = want == LogitRows::none ? nullptr : logits;
        o.top = want == LogitRows::none ? nullptr : d_top;
        o.hidden_post = xn;
        o.hidden_pre = x;
        return o;
    }

    CudaEngine::MtpOut mtp(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0, bool want_top) {
        if (!mtp_ok) throw std::runtime_error("mtp_forward: no MTP tensors loaded");
        if (rows == 0 || rows > RM || pos0 + rows > opt.max_positions) throw std::runtime_error("mtp_forward: bad rows/pos0");
        const std::uint32_t H = c.hidden;
        InputPack in{};
        in.pos0 = pos0;
        for (std::uint32_t i = 0; i < rows; ++i) in.ids[i] = next_tokens[i];
        set_inputs_kernel<<<1, 32, 0, s>>>(in, d_mpos, d_mids);
        CK(cudaGetLastError());
        const cc::GroupedView src{rows, 1, H, H, H}, dst{rows, 1, H, 2 * H, H};
        CK(cc::embedding_rows(embd, d_mids, mtmp, rows, s));
        CK(cc::rmsnorm(mtmp, src, m_enorm, mcat, dst, c.rms_eps, s));
        CK(cc::rmsnorm(hidden_dev, src, m_hnorm, mcat + H, dst, c.rms_eps, s));
        CK(cc::dense_rows(m_eh, mcat, mx, rows, nullptr, s));
        const cc::GroupedView hv{rows, 1, H, H, H};
        CK(cc::rmsnorm(mx, hv, M.attn_norm, mxn, hv, c.rms_eps, s));
        const Bufs b{mx, mxn, d_mpos, mkc, mvc};
        attn_mixer(M, b, rows);
        mlp(M, b, rows);
        // mx holds the block output (pre mtp.norm); the hidden export needs it intact, so normalise into mxn.
        CK(cc::rmsnorm(mx, hv, m_shnorm, mxn, hv, c.rms_eps, s));
        CudaEngine::MtpOut o;
        if (want_top) {
            CK(cc::dense_rows(head, mxn, mlogits, rows, nullptr, s));
            CK(cc::argmax_rows(mlogits, c.vocab, rows, d_mtop, d_argws, s));
            o.logits = mlogits;
            o.top = d_mtop;
        }
        o.hidden_pre = mx;
        o.hidden_post = mxn;
        return o;
    }
};

// ================================================================ loading
CudaEngine::CudaEngine() : p_(new Impl) {}
CudaEngine::~CudaEngine() = default;

std::unique_ptr<CudaEngine> CudaEngine::from_gguf(const std::string& path, const EngineOptions& opt) {
    std::unique_ptr<CudaEngine> e(new CudaEngine);
    auto& P = *e->p_;
    P.opt = opt;
    const auto info = gguf::read_file(path);
    P.c = read_cfg(info);
    const auto t0 = std::chrono::steady_clock::now();
    auto specs = specs_from_gguf(info, P.c.layers, opt.load_mtp);
    mark_fp4(opt, specs);
    layout(P.W, specs, P.c.layers);
    std::uint64_t tmp_bytes = 0;
    for (const auto& t : specs) if (t.fp4) tmp_bytes = std::max(tmp_bytes, t.src_bytes);
    unsigned char* tmp = nullptr;  // BF16 staging for the on-the-fly conversion (largest eligible tensor)
    if (tmp_bytes) CK(cudaMalloc(reinterpret_cast<void**>(&tmp), tmp_bytes));
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    constexpr std::size_t kChunk = 128u << 20;
    unsigned char* pin = nullptr;
    CK(cudaMallocHost(reinterpret_cast<void**>(&pin), kChunk));
    for (auto& [name, t] : P.W.spec) {
        in.seekg(static_cast<std::streamoff>(t.file_off));
        std::uint64_t done = 0;
        const std::uint64_t total = t.fp4 ? t.src_bytes : t.bytes;
        unsigned char* dst = t.fp4 ? tmp : P.W.base + P.W.off[name];
        while (done < total) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, total - done));
            in.read(reinterpret_cast<char*>(pin), static_cast<std::streamsize>(n));
            if (!in) throw std::runtime_error("short read for " + name);
            CK(cudaMemcpy(dst + done, pin, n, cudaMemcpyHostToDevice));
            done += n;
        }
        if (t.fp4) {
            CK(cc::nvfp4_quantize_bf16(tmp, static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in),
                                       P.W.base + P.W.off[name], &t.gscale, nullptr));
            CK(cudaDeviceSynchronize());
        }
    }
    CK(cudaFreeHost(pin));
    if (tmp) CK(cudaFree(tmp));
    P.init();
    if (!opt.quiet)
        std::fprintf(stderr, "qwen35 engine: loaded %.2f GB in %.1f s, mtp %s\n", static_cast<double>(P.W.total) / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), P.mtp_ok ? "yes" : "no");
    return e;
}

std::unique_ptr<CudaEngine> CudaEngine::synthetic(const Config& cfg, PackFormat pack, const EngineOptions& opt, std::uint64_t seed) {
    std::unique_ptr<CudaEngine> e(new CudaEngine);
    auto& P = *e->p_;
    P.opt = opt;
    P.c = cfg;
    const auto t0 = std::chrono::steady_clock::now();
    if (opt.turbo_weights == TurboWeights::onfly) pack = PackFormat::bf16;  // onfly source is BF16
    const auto sp = synth_specs(cfg, pack, opt.load_mtp);
    std::vector<PackTensorSpec> specs;
    for (const auto& [t, f] : sp) specs.push_back(t);
    mark_fp4(opt, specs);
    layout(P.W, specs, cfg.layers);
    std::uint64_t tmp_bytes = 0;
    for (const auto& t : specs) if (t.fp4) tmp_bytes = std::max(tmp_bytes, t.src_bytes);
    unsigned char* tmp = nullptr;
    if (tmp_bytes) CK(cudaMalloc(reinterpret_cast<void**>(&tmp), tmp_bytes));
    for (const auto& [t0_, f] : sp) {
        auto& t = P.W.spec[t0_.name];
        unsigned char* dst = t.fp4 ? tmp : P.W.base + P.W.off[t.name];
        const std::uint64_t sd = seed ^ fnv(t.name);
        const std::uint64_t n = t.in * t.out;
        if (t.type == 0) {
            fill_f32_kernel<<<static_cast<unsigned>(std::min<std::uint64_t>((n + 255) / 256, 4096)), 256>>>(reinterpret_cast<float*>(dst), n, sd, f.base, f.amp);
        } else if (t.type == 30) {
            fill_bf16_kernel<<<4096, 256>>>(reinterpret_cast<std::uint16_t*>(dst), n, sd, f.base, f.amp);
        } else {
            fill_q8_kernel<<<4096, 256>>>(dst, n / 32, sd, f.amp);
        }
        CK(cudaGetLastError());
        if (t.fp4) {
            CK(cc::nvfp4_quantize_bf16(tmp, static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in),
                                       P.W.base + P.W.off[t.name], &t.gscale, nullptr));
            CK(cudaDeviceSynchronize());
        }
    }
    CK(cudaDeviceSynchronize());
    if (tmp) CK(cudaFree(tmp));
    P.init();
    if (!opt.quiet)
        std::fprintf(stderr, "qwen35 engine: synthetic %s %.2f GB generated in %.1f s, mtp %s\n", pack == PackFormat::bf16 ? "bf16" : "q8_0",
                     static_cast<double>(P.W.total) / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), P.mtp_ok ? "yes" : "no");
    return e;
}

static std::string engine_identity_base(const EngineOptions& o, const char* arith);
std::string engine_identity(const EngineOptions& o, const char* arith) {
    std::string base = engine_identity_base(o, arith);
    if (o.prefill_arith != PrefillArith::exact) base += std::string("-pf-") + prefill_arith_name(o.prefill_arith);
    return base;
}
static std::string engine_identity_base(const EngineOptions& o, const char* arith) {
    const std::string a = arith ? arith : "mma3";
    if (o.turbo_weights == TurboWeights::off) return "qwen35-cuda-exact-" + a + "-v1";
    if (o.turbo_weights == TurboWeights::native) return "qwen35-turbo-native-" + a + "-v1";
    return std::string("qwen35-turbo-nvfp4-onfly-rtn16-v1") + (o.turbo_lm_head ? "" : "-lmhead-bf16");
}

void print_effective_config(const EngineOptions& o, const char* arith, std::FILE* f) {
    const std::string id = engine_identity(o, arith);
    std::fprintf(f, "effective-config:\n  identity: %s\n  prefix-cache-key: %s (never shared across identities)\n", id.c_str(), id.c_str());
    std::fprintf(f, "  dense-arith: %s\n  turbo-weights: %s\n  turbo-bits: %u\n  turbo-lm-head: %s\n  turbo-multi-launch: %s (bitwise neutral)\n",
                 arith ? arith : "mma3", o.turbo_weights == TurboWeights::off ? "off (exact pack)" : o.turbo_weights == TurboWeights::native ? "native" : "onfly",
                 o.turbo_bits, o.turbo_weights == TurboWeights::onfly ? (o.turbo_lm_head ? "nvfp4" : "bf16") : "n/a", o.multi_gemv ? "on" : "off");
    if (o.turbo_weights == TurboWeights::onfly)
        std::fprintf(f, "  format: NVFP4 e2m1 + e4m3/16 + FP32 global; RTN; GEMV f16 mma, FP32 accumulate, activations RN f16, rows 1..32 bitwise row-invariant\n");
}

const Config& CudaEngine::config() const { return p_->c; }
const EngineOptions& CudaEngine::options() const { return p_->opt; }
std::string CudaEngine::identity() const {
    static const char* names[] = {"fp32", "mma3", "mma2", "mma1"};
    return engine_identity(p_->opt, names[static_cast<int>(cc::dense_arith())]);
}
std::uint32_t CudaEngine::nvfp4_tensors() const {
    std::uint32_t n = 0;
    for (const auto& [k, t] : p_->W.spec) n += t.fp4 ? 1u : 0u;
    return n;
}
bool CudaEngine::has_mtp() const { return p_->mtp_ok; }
cudaStream_t CudaEngine::stream() const { return p_->s; }
std::uint32_t CudaEngine::committed() const { return p_->committed; }
std::uint64_t CudaEngine::weight_bytes() const { return p_->W.trunk_bytes; }
std::uint64_t CudaEngine::state_bytes_per_pass() const {
    return static_cast<std::uint64_t>(p_->commit_layers.size()) * p_->gd.value_heads * p_->gd.key_dim * p_->gd.value_dim * 4 * 2;
}
void CudaEngine::reset() { p_->reset(); }
RowsOut CudaEngine::forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want) {
    return p_->forward(tokens, rows, want);
}
void CudaEngine::commit(std::uint32_t k) { p_->commit_rows(k); }
RowsOut CudaEngine::prefill(const std::uint32_t* tokens, std::uint32_t n, LogitRows want) { return p_->prefill(tokens, n, want); }
bool CudaEngine::prefill_is_big() const { return p_->big_supported(); }
const char* prefill_arith_name(PrefillArith a) {
    static const char* n[] = {"exact", "flash", "fast", "fast2"};
    return n[static_cast<int>(a)];
}
CudaEngine::MtpOut CudaEngine::mtp_forward(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows,
                                           std::uint32_t pos0, bool want_top) {
    return p_->mtp(next_tokens, hidden_dev, rows, pos0, want_top);
}
void CudaEngine::read_logits(const RowsOut& o, std::uint32_t row, float* host) const {
    CK(cudaMemcpyAsync(host, o.logits + static_cast<std::size_t>(row) * p_->c.vocab, static_cast<std::size_t>(p_->c.vocab) * 4,
                       cudaMemcpyDeviceToHost, p_->s));
    CK(cudaStreamSynchronize(p_->s));
}
void CudaEngine::read_top(const RowsOut& o, std::uint32_t rows, std::uint32_t* host) const {
    CK(cudaMemcpyAsync(host, o.top, rows * 4, cudaMemcpyDeviceToHost, p_->s));
    CK(cudaStreamSynchronize(p_->s));
}
void CudaEngine::read_hidden(const float* dev, std::uint32_t rows, float* host) const {
    CK(cudaMemcpyAsync(host, dev, static_cast<std::size_t>(rows) * p_->c.hidden * 4, cudaMemcpyDeviceToHost, p_->s));
    CK(cudaStreamSynchronize(p_->s));
}
void CudaEngine::sync() const { CK(cudaStreamSynchronize(p_->s)); }
void CudaEngine::set_graphs(bool on) { p_->graphs_on = on; }
std::uint64_t CudaEngine::launches_last_forward() const { return p_->last_nodes; }

}  // namespace strix::qwen35
