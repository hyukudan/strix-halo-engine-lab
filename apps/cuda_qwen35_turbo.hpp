#pragma once
// TURBO options of strix-cuda-qwen35-decode (Home: specific = CLI/identity of the qwen35 decode app; the NVFP4 format and
// kernels are shared in strix/cuda_core/nvfp4.hpp). Needs CK, cc:: and strix::gguf from the including translation unit.
//
// Levers (every one individually switchable; default = exact BF16, no TURBO lever on):
//   --turbo-weights native|onfly   weight source. native = current pack as is (no weight change); onfly = BF16 -> NVFP4 RTN
//                                  per 16-group at load (only the 4-bit copy stays in VRAM)
//   --turbo-bits 4                 bit width (4 = NVFP4; 3/2 later)
//   --turbo-lm-head nvfp4|bf16     lm_head follows the weight source (nvfp4) or stays BF16
//   --synthetic-27b                full-size random device weights, no disk (timing only)
//   --turbo-multi-launch 0|1       one launch for GEMVs sharing x (bitwise neutral)
//   --print-effective-config       print identity + levers and exit
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "strix/cuda_core/nvfp4.hpp"

namespace q35turbo {

struct Options {
    enum class Src { off, native, onfly } src = Src::off;
    unsigned bits = 4;
    bool lm_head_fp4 = true;
    bool synthetic = false;
    bool multi_launch = true;  // --turbo-multi-launch 0|1: one launch for GEMVs sharing x (bitwise equal to separate calls)
    bool print_cfg = false;

    [[nodiscard]] bool onfly() const { return src == Src::onfly; }
    [[nodiscard]] std::string identity() const {
        if (src == Src::off) return "qwen35-exact-bf16-v1";
        if (src == Src::native) return "qwen35-turbo-native-bf16-v1";
        return std::string("qwen35-turbo-nvfp4-onfly-rtn16-v1") + (lm_head_fp4 ? "" : "-lmhead-bf16");
    }
    bool parse(const std::string& a, const std::function<std::string()>& next) {
        if (a == "--turbo-weights") {
            const std::string v = next();
            if (v == "native") src = Src::native;
            else if (v == "onfly") src = Src::onfly;
            else if (v == "converted") throw std::runtime_error("--turbo-weights converted: not implemented yet (native|onfly)");
            else throw std::runtime_error("--turbo-weights expects native|onfly");
        } else if (a == "--turbo-bits") {
            bits = static_cast<unsigned>(std::stoul(next()));
            if (bits != 4) throw std::runtime_error("--turbo-bits: only 4 is implemented");
        } else if (a == "--turbo-lm-head") {
            const std::string v = next();
            if (v != "nvfp4" && v != "bf16") throw std::runtime_error("--turbo-lm-head expects nvfp4|bf16");
            lm_head_fp4 = v == "nvfp4";
        } else if (a == "--turbo-multi-launch") {
            multi_launch = next() != "0";
        } else if (a == "--synthetic-27b") {
            synthetic = true;
        } else if (a == "--print-effective-config") {
            print_cfg = true;
        } else {
            return false;
        }
        return true;
    }
    void print(std::FILE* f) const {
        std::fprintf(f, "effective-config:\n  identity: %s\n  prefix-cache-key: %s (never shared across identities)\n", identity().c_str(),
                     identity().c_str());
        std::fprintf(f, "  turbo-weights: %s\n  turbo-bits: %u\n  turbo-lm-head: %s\n  turbo-multi-launch: %s (bitwise neutral)\n", src == Src::off ? "off (exact BF16)" : src == Src::native ? "native" : "onfly",
                     bits, src == Src::onfly ? (lm_head_fp4 ? "nvfp4" : "bf16") : "n/a", multi_launch ? "on" : "off");
        if (src == Src::onfly)
            std::fprintf(f, "  format: NVFP4 e2m1 + e4m3/16 + FP32 global; RTN; GEMV f16 mma, FP32 accumulate, activations RN f16\n");
        if (synthetic) std::fprintf(f, "  weights: synthetic 27B shapes generated on device (no disk)\n");
    }
    // 2-D BF16 [out][in] weights that take the NVFP4 path
    [[nodiscard]] bool eligible(const strix::gguf::TensorInfo& t) const {
        if (!onfly() || t.ggml_type != 30 || t.dimensions.size() != 2) return false;
        if (t.name == "token_embd.weight") return false;
        if (t.name == "output.weight" && !lm_head_fp4) return false;
        if (t.name.size() < 7 || t.name.compare(t.name.size() - 7, 7, ".weight") != 0) return false;
        return strix::cuda_core::nvfp4_shape_ok(static_cast<std::uint32_t>(t.dimensions[1]), static_cast<std::uint32_t>(t.dimensions[0]));
    }
};

// ---- synthetic full-size weights (timing / bring-up; values are hashed noise) ----
__device__ __forceinline__ float hash_unit(std::uint64_t i, std::uint32_t seed) {
    std::uint64_t z = i * 0x9E3779B97F4A7C15ULL + seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return static_cast<float>(static_cast<std::uint32_t>(z >> 40)) * (1.0f / 16777216.0f) * 2.0f - 1.0f;  // [-1,1)
}
__global__ void synth_bf16(std::uint16_t* p, std::size_t n, std::uint32_t seed, float amp) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        const float v = (hash_unit(i, seed) + hash_unit(i, seed ^ 0x55555555u) + hash_unit(i, seed ^ 0x33333333u)) * amp;  // ~bell
        p[i] = static_cast<std::uint16_t>(__float_as_uint(v) >> 16);
    }
}
__global__ void synth_f32(float* p, std::size_t n, std::uint32_t seed, float base, float amp) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        p[i] = base + amp * hash_unit(i, seed);
}

inline void synth_fill(const strix::gguf::TensorInfo& t, void* dev) {
    std::size_t n = 1;
    for (auto d : t.dimensions) n *= d;
    std::uint32_t seed = 17;
    for (char ch : t.name) seed = seed * 131u + static_cast<unsigned char>(ch);
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 4096));
    if (t.ggml_type == 30) {
        synth_bf16<<<blocks, 256>>>(static_cast<std::uint16_t*>(dev), n, seed, 0.012f);
    } else {
        float base = 1.0f, amp = 0.05f;
        if (t.name.find("ssm_a") != std::string::npos) { base = -0.7f; amp = 0.5f; }
        else if (t.name.find("dt.bias") != std::string::npos) { base = 0.0f; amp = 0.5f; }
        else if (t.name.find("conv1d") != std::string::npos) { base = 0.0f; amp = 0.4f; }
        synth_f32<<<blocks, 256>>>(static_cast<float*>(dev), n, seed, base, amp);
    }
}

// Tensor table of the qwen35 dense family for config `c` (fields as in the app Cfg).
template <class C>
std::vector<strix::gguf::TensorInfo> synthetic_tensors(const C& c) {
    std::vector<strix::gguf::TensorInfo> v;
    auto add = [&](const std::string& n, std::uint64_t in, std::uint64_t out, bool f32) {
        strix::gguf::TensorInfo t;
        t.name = n;
        t.dimensions = out ? std::vector<std::uint64_t>{in, out} : std::vector<std::uint64_t>{in};
        t.ggml_type = f32 ? 0 : 30;
        t.byte_size = in * (out ? out : 1) * (f32 ? 4 : 2);
        v.push_back(t);
    };
    const std::uint64_t H = c.hidden, conv_ch = 2ull * c.key_heads * c.key_dim + std::uint64_t(c.value_heads) * c.value_dim,
                        vw = std::uint64_t(c.value_heads) * c.value_dim;
    add("token_embd.weight", H, c.vocab, false);
    add("output_norm.weight", H, 0, true);
    add("output.weight", H, c.vocab, false);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add(p + "attn_norm.weight", H, 0, true);
        add(p + "post_attention_norm.weight", H, 0, true);
        add(p + "ffn_gate.weight", H, c.inter, false);
        add(p + "ffn_up.weight", H, c.inter, false);
        add(p + "ffn_down.weight", c.inter, H, false);
        if (!c.is_attn(l)) {
            add(p + "attn_qkv.weight", H, conv_ch, false);
            add(p + "attn_gate.weight", H, vw, false);
            add(p + "ssm_beta.weight", H, c.value_heads, false);
            add(p + "ssm_alpha.weight", H, c.value_heads, false);
            add(p + "ssm_conv1d.weight", c.conv_k, conv_ch, true);
            add(p + "ssm_a", c.value_heads, 0, true);
            add(p + "ssm_dt.bias", c.value_heads, 0, true);
            add(p + "ssm_norm.weight", c.value_dim, 0, true);
            add(p + "ssm_out.weight", vw, H, false);
        } else {
            add(p + "attn_q.weight", H, std::uint64_t(c.heads) * c.head_dim * 2, false);
            add(p + "attn_k.weight", H, std::uint64_t(c.kv_heads) * c.head_dim, false);
            add(p + "attn_v.weight", H, std::uint64_t(c.kv_heads) * c.head_dim, false);
            add(p + "attn_output.weight", std::uint64_t(c.heads) * c.head_dim, H, false);
            add(p + "attn_q_norm.weight", c.head_dim, 0, true);
            add(p + "attn_k_norm.weight", c.head_dim, 0, true);
        }
    }
    return v;
}

}  // namespace q35turbo
