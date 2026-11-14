#pragma once
// Image-only admission and checkpoint policy for the shared H3 DiT.
#include <fstream>
#include <iterator>
#include <limits>
#include "h3/block/host_ref.hpp"
#include "h3/block/sampler.hpp"

namespace strix::h3::block {

inline fixture::Json read_json_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("image: missing JSON asset");
    const std::string s{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    return fixture::JsonParser(s).parse();
}

inline std::string image_operand_identity(const SafeTensors& st) {
    const std::string prefix="blocks.0.attn.qkv_proj";
    const auto& w=st.at(prefix+".weight");
    if (w.dtype==StDtype::bf16 || w.dtype==StDtype::f16 || w.dtype==StDtype::f32) return "dense-bf16";
    const auto& t=st.at(prefix+".comfy_quant");
    if (t.dtype!=StDtype::u8 || t.nbytes()>65536) throw std::invalid_argument("image: invalid operand metadata");
    std::string raw(std::size_t(t.nbytes()),'\0');st.read_raw(t,raw.data());
    const auto j=fixture::JsonParser(raw).parse();const auto fmt=j.at("format").str();
    if (fmt=="int8_tensorwise") return "int8-convrot";
    if (fmt=="asym_w4a8_int8") return "w4a8-convrot-iu8";
    throw std::invalid_argument("image: unsupported checkpoint operand");
}

struct ImageConfig {
    std::vector<float> sigmas;
    static ImageConfig parse(const fixture::Json& j, const fixture::Json& pdd) {
        auto eq = [&](const char* name, double value) {
            if (!j.at(name).is_number() || j.at(name).num() != value)
                throw std::invalid_argument(std::string("image: unsupported ") + name);
        };
        if (!std::holds_alternative<bool>(j.at("audio_enabled").v) || std::get<bool>(j.at("audio_enabled").v))
            throw std::invalid_argument("image: audio_enabled must be false");
        eq("hidden_size", kHidden); eq("num_layers", kNumBlocks);
        eq("num_attention_heads", kHeads); eq("attention_head_dim", kHeadDim);
        eq("ffn_hidden_size", kFfn); eq("latents_dim", 24); eq("text_dim", kTextDim);
        eq("token_refiner_num_layers", 2); eq("rope_inv_freq_len", 16);
        eq("timestep_input_dim", 256); eq("time_embed_hidden_size", 5376); eq("time_embed_dim", 2688);
        eq("adaln_out_features", 96768); eq("final_adaln_out_features", 10752);
        eq("norm_eps", 1e-5); eq("qk_norm_eps", 1e-5); eq("final_norm_eps", 1e-5);
        const auto& patch = j.at("patch_size").arr();
        if (patch.size() != 3 || patch[0].num() != 1 || patch[1].num() != 2 || patch[2].num() != 2)
            throw std::invalid_argument("image: patch_size must be [1,2,2]");
        ImageConfig c;
        if (pdd.at("num_inference_steps").num() != 6) throw std::invalid_argument("image: expected six PDD steps");
        for (const auto& s : pdd.at("sigmas").arr()) {
            const double v = s.num();
            if (!std::isfinite(v) || v < 0 || v > 1) throw std::invalid_argument("image: invalid sigma");
            c.sigmas.push_back(static_cast<float>(v));
        }
        if (c.sigmas.size() != 7 || c.sigmas.front() != 1 || c.sigmas.back() != 0)
            throw std::invalid_argument("image: invalid PDD endpoints/count");
        for (std::size_t i = 1; i < c.sigmas.size(); ++i)
            if (!(c.sigmas[i] < c.sigmas[i-1])) throw std::invalid_argument("image: sigmas must decrease strictly");
        return c;
    }
    static ImageConfig load(const std::string& model) {
        if (!std::filesystem::exists(model+"/transformer/config.json") && !std::filesystem::exists(model+"/pdd_config.json"))
            return {{1.0F,0.9675236940383911F,0.8875485062599182F,0.75F,0.5327739119529724F,0.23200887441635132F,0.0F}};
        return parse(read_json_file(model + "/transformer/config.json"), read_json_file(model + "/pdd_config.json"));
    }
    static LoadPolicy weights_policy() { return {false, false, true, true}; }
    static LoadPolicy weights_policy(const SafeTensors& st) {
        auto p=weights_policy(); p.image_operands=true; p.dense_bf16=false;
        p.direct_timestep=st.find("adaln_t_table")==nullptr;
        return p;
    }
    static ForwardPlanConfig plan_policy() { return {1.0F, 1.0F, kVisualCondT, kAudioCondT}; }
    static std::uint64_t conservative_bytes(std::uint32_t packed_rows) {
        // Resident BF16 projections, refiner, finite adaLN, host mirrors, reserve,
        // full driver workspace, split-attention partials and final projections.
        return (52ULL << 30) + std::uint64_t(packed_rows) * 320000ULL;
    }
    static PackedLayout layout(std::uint32_t text, std::uint32_t width, std::uint32_t height,
                               const std::vector<RefBlock>& refs = {}) {
        if (width < 32 || height < 32 || width % 32 || height % 32 || refs.size() > 16)
            throw std::invalid_argument("image: sides must be positive multiples of 32; at most 16 references");
        std::uint64_t rows = static_cast<std::uint64_t>(width / 32) * (height / 32) + text;
        for (const auto& r : refs) {
            if (r.kind != RefBlock::image || r.latent_h < 2 || r.latent_w < 2 || r.latent_h % 2 || r.latent_w % 2 || r.ref_audio_t)
                throw std::invalid_argument("image: invalid reference shape/kind");
            rows += static_cast<std::uint64_t>(r.latent_h / 2) * (r.latent_w / 2);
        }
        if (rows > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("image: packed sequence overflow");
        return PackedLayout::build(text, 1, height / 16, width / 16, 0, {}, refs);
    }
};

inline void load_image_pdd(GlobalWeights& g, const SafeTensors& st, const ImageConfig& c) {
    detail::expect_shape(st, "video_out.weight", {6, kVideoPatch, kHidden});
    detail::expect_shape(st, "video_out.bias", {6, kVideoPatch});
    if (c.sigmas.size() != 7 || g.audio_enabled) throw std::invalid_argument("image: PDD requires image policy");
    g.video_out = {6 * kVideoPatch, kHidden, detail::read_as_f32(st, "video_out.weight", false),
                  detail::read_as_f32(st, "video_out.bias", false)};
    g.pdd_steps = 6; g.pdd_fused = true;
}

// The image schedule has only six target times and one reference time. Evaluate raw
// adaLN once, then store the resulting FP32 columns. This is finite evaluation,
// never a curve fit, interpolation, quantization, or a change to the video loader.
class ImageBlockSource final : public BlockSource {
public:
    ImageBlockSource(const SafeTensors& st, GlobalWeights& g, const ImageConfig& c) : st_(st), raw_basis_(g.basis), policy_(ImageConfig::weights_policy(st)) {
        if (count_blocks(st) != kNumBlocks || g.audio_enabled)
            throw std::invalid_argument("image: expected complete image checkpoint");
        times_.push_back(static_cast<float>(kVisualCondT));
        for (std::size_t i = 0; i + 1 < c.sigmas.size(); ++i) times_.push_back(1.0F - model_sigma(c.sigmas[i]));
        std::sort(times_.begin(), times_.end());
        times_.erase(std::unique(times_.begin(), times_.end()), times_.end());
        embeds_ = curve_t_emb(g, times_);
        g.final_adaln = prepare(g.final_adaln);
        g.basis = static_cast<std::uint32_t>(times_.size());
        g.discrete_times = times_;
        g.direct_timestep = false;
        g.time_in = {}; g.time_out = {};
    }
    std::shared_ptr<const BlockWeights> block(std::uint32_t i) override {
        if (i >= kNumBlocks) throw std::out_of_range("image: block index");
        auto b = std::make_shared<BlockWeights>(load_block(st_, i, raw_basis_, policy_));
        b->adaln = prepare(b->adaln);
        return b;  // device driver owns residency; avoid a duplicate host model
    }
    std::uint32_t num_blocks() const override { return kNumBlocks; }
private:
    F32Linear prepare(const F32Linear& raw) const {
        const auto n = static_cast<std::uint32_t>(times_.size());
        const auto values = f32_linear(raw, embeds_.data(), n);
        F32Linear out; out.n = raw.n; out.k = n;
        out.w.resize(static_cast<std::size_t>(raw.n) * n); out.b.assign(raw.n, 0);
        for (std::uint32_t r = 0; r < n; ++r)
            for (std::uint32_t o = 0; o < raw.n; ++o) out.w[static_cast<std::size_t>(o)*n+r] = values[static_cast<std::size_t>(r)*raw.n+o];
        return out;
    }
    const SafeTensors& st_;
    std::uint32_t raw_basis_;
    LoadPolicy policy_;
    std::vector<float> times_, embeds_;
};

// Six no-CFG model evaluations; sampler carry and FP32 Euler operation order are shared with video.
inline AvState sample_image(const ImageConfig& c, const std::vector<float>& noise, const AvModelFn& model) {
    SamplerCarry carry;
    carry.x = initial_state(c.sigmas.front(), noise, {});
    sample_steps(model, carry, c.sigmas, SamplerKind::euler, c.sigmas.size() - 1);
    return carry.x;
}
}  // namespace strix::h3::block
