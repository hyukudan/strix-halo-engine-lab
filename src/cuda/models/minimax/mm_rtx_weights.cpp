// Per-layer weight blobs of the RTX PRO prefill. See mm_rtx_weights.hpp.
#include "mm_rtx_weights.hpp"
#include "mm_rtx_index.hpp"
#include "mm_rtx_prep.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::minimax::rtxpf::host {

namespace {

const std::string kP = "language_model.model.";
constexpr unsigned kHidden = 6144, kQLocal = 4096, kKVLocal = 256, kDenseLocal = 6144, kExpertLocal = 1536;

struct Builder {
    LayerBlob& b;
    std::size_t put(const void* p, std::size_t n) {
        const std::size_t off = (b.bytes.size() + 255) & ~std::size_t{255};
        b.bytes.resize(off + n);
        std::memcpy(b.bytes.data() + off, p, n);
        return off;
    }
    MatRef mat(const Exl3Host& e) {
        MatRef m;
        m.trellis = put(e.trellis.data(), e.trellis.size());
        m.suh = put(e.suh.data(), e.suh.size() * 2);
        m.svh = put(e.svh.data(), e.svh.size() * 2);
        m.input = e.input;
        m.output = e.output;
        m.bits = e.bits;
        return m;
    }
    std::size_t floats(const std::vector<float>& f) { return put(f.data(), f.size() * 4); }
    std::size_t gemma_gain(std::vector<float> weight) {
        for (float& gain : weight) gain = 1.0F + gain;
        return floats(weight);
    }
};

std::vector<float> transpose_rows(const std::vector<float>& a, std::size_t cols, std::size_t r0, std::size_t nr) {
    std::vector<float> t(cols * nr);  // rows [r0, r0 + nr) of a [*][cols] -> [cols][nr]
    for (std::size_t r = 0; r < nr; ++r)
        for (std::size_t c = 0; c < cols; ++c) t[c * nr + r] = a[(r0 + r) * cols + c];
    return t;
}

}  // namespace

LayerBlob build_layer(Pack& pack, unsigned L, bool residual_only) {
    LayerBlob blob;
    Builder bd{blob};
    const std::string lp = kP + "layers." + std::to_string(L) + ".", a = lp + "self_attn.", b = lp + "block_sparse_moe.";
    static const char* const kMlp[3] = {"gate_proj", "up_proj", "down_proj"};
    static const char* const kExp[3] = {"w1", "w3", "w2"};
    const std::string iq_name = a + kIndexQuery.tensor, ik_name = a + kIndexKey.tensor;
    if (!kIndexQuery.admits(pack.info(iq_name))) throw std::runtime_error("MMIDX refused " + iq_name);
    if (!kIndexKey.admits(pack.info(ik_name))) throw std::runtime_error("MMIDX refused " + ik_name);
    blob.bytes.reserve(std::size_t{2800} << 20);
    const auto wiq_all = pack.floats(iq_name);
    for (unsigned r = 0; r < 2; ++r) {
        RankRef& rr = blob.r[r];
        rr.q = bd.mat(pack.slice_out(a + "q_proj", r * kQLocal, kQLocal));
        rr.k = bd.mat(pack.slice_out(a + "k_proj", r * kKVLocal, kKVLocal));
        rr.v = bd.mat(pack.slice_out(a + "v_proj", r * kKVLocal, kKVLocal));
        rr.o = bd.mat(pack.slice_in(a + "o_proj", r * kQLocal, kQLocal));
        rr.wiq = bd.floats(transpose_rows(wiq_all, kIndexInput, r * kIndexQuery.outputs, kIndexQuery.outputs));
    }
    blob.wik = bd.floats(transpose_rows(pack.floats(ik_name), kIndexInput, 0, kIndexKey.outputs));
    const auto prep_gain = [&](const char* tensor) {
        const auto name = a + tensor;
        if (!pack.has(name)) throw std::runtime_error("MMPREP refused missing " + name);
        const auto& info = pack.info(name);
        if (!cuda::admits_msa_prep_gain(info.dtype, info.bytes)) throw std::runtime_error("MMPREP refused " + name);
        const auto gain = pack.floats(name);
        if (!std::ranges::all_of(gain, [](float entry) { return std::isfinite(entry); }))
            throw std::runtime_error("MMPREP refused nonfinite " + name);
        return bd.floats(gain);
    };
    blob.q_w = prep_gain("q_norm.weight");
    blob.k_w = prep_gain("k_norm.weight");
    blob.iq_w = prep_gain("index_q_norm.weight");
    blob.ik_w = prep_gain("index_k_norm.weight");
    blob.w_post = bd.gemma_gain(pack.floats(lp + "post_attention_layernorm.weight"));
    blob.w_next = residual_only ? bd.floats(std::vector<float>(kHidden, 1.0F))
                                : bd.gemma_gain(pack.floats(L + 1 < kLayers ? kP + "layers." + std::to_string(L + 1) +
                                                                                 ".input_layernorm.weight" : kP + "norm.weight"));
    blob.moe = L >= kFirstMoe;
    auto mlp = [&](unsigned r, const std::string& pre, const char* const* nm, unsigned inter) {
        MlpRef m;
        m.gate = bd.mat(pack.slice_out(pre + nm[0], r * inter, inter));
        m.up = bd.mat(pack.slice_out(pre + nm[1], r * inter, inter));
        m.down = bd.mat(pack.slice_in(pre + nm[2], r * inter, inter));
        return m;
    };
    if (!blob.moe) {
        for (unsigned r = 0; r < 2; ++r) blob.r[r].dense = mlp(r, lp + "mlp.", kMlp, kDenseLocal);
    } else {
        // expert-major so the two rank slices of one expert reuse the whole-tensor read of slice_out
        for (unsigned r = 0; r < 2; ++r) blob.r[r].experts.resize(kExperts);
        for (unsigned e = 0; e < kExperts; ++e)
            for (unsigned r = 0; r < 2; ++r) blob.r[r].experts[e] = mlp(r, b + "experts." + std::to_string(e) + ".", kExp, kExpertLocal);
        for (unsigned r = 0; r < 2; ++r) blob.r[r].shared = mlp(r, b + "shared_experts.", kMlp, kExpertLocal);
        blob.router = bd.floats(pack.floats(b + "gate.weight"));
        blob.bias = bd.floats(pack.floats(b + "e_score_correction_bias"));
    }
    return blob;
}

DevLayer bind(const LayerBlob& blob, const std::uint8_t* base) {
    auto m = [&](const MatRef& r) {
        DevMat d;
        d.trellis = reinterpret_cast<const std::uint32_t*>(base + r.trellis);
        d.suh = reinterpret_cast<const std::uint16_t*>(base + r.suh);
        d.svh = reinterpret_cast<const std::uint16_t*>(base + r.svh);
        d.input = r.input;
        d.output = r.output;
        d.bits = r.bits;
        return d;
    };
    auto ml = [&](const MlpRef& r) { return DevMlp{m(r.gate), m(r.up), m(r.down)}; };
    auto f = [&](std::size_t off) { return reinterpret_cast<const float*>(base + off); };
    DevLayer d;
    for (unsigned r = 0; r < 2; ++r) {
        const RankRef& rr = blob.r[r];
        DevRank& dr = d.r[r];
        dr.q = m(rr.q);
        dr.k = m(rr.k);
        dr.v = m(rr.v);
        dr.o = m(rr.o);
        dr.wiq = f(rr.wiq);
        if (!blob.moe) {
            dr.dense = ml(rr.dense);
        } else {
            dr.shared = ml(rr.shared);
            dr.experts.reserve(rr.experts.size());
            for (const auto& e : rr.experts) dr.experts.push_back(ml(e));
        }
    }
    d.wik = f(blob.wik);
    d.q_w = f(blob.q_w);
    d.k_w = f(blob.k_w);
    d.iq_w = f(blob.iq_w);
    d.ik_w = f(blob.ik_w);
    d.moe = blob.moe;
    if (blob.moe) {
        d.router = f(blob.router);
        d.bias = f(blob.bias);
    }
    return d;
}

}  // namespace strix::minimax::rtxpf::host
