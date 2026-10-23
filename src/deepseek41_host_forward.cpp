#include "strix/deepseek41_host_forward.hpp"

#include "strix/deepseek41_attention_host.hpp"

#include "strix/ds41_engram.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_weight_overlay.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::deepseek41 {

// ---- read-only mapping of the GGUF (tensor bytes are read in place) ----
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#if defined(_WIN32)
        // read-only view of the whole file; the view keeps the mapping alive once both handles are closed
        const HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("deepseek41 host forward: cannot open " + path.string());
        LARGE_INTEGER bytes{};
        const HANDLE mapping = ::GetFileSizeEx(file, &bytes) && bytes.QuadPart > 0
                                   ? ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr)
                                   : nullptr;
        void* p = mapping != nullptr ? ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
        if (mapping != nullptr) ::CloseHandle(mapping);
        ::CloseHandle(file);
        if (p == nullptr) throw std::runtime_error("deepseek41 host forward: cannot map " + path.string());
        size_ = static_cast<std::uint64_t>(bytes.QuadPart);
        base_ = static_cast<const std::uint8_t*>(p);
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 host forward: cannot open " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("deepseek41 host forward: cannot stat the model");
        size_ = static_cast<std::uint64_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("deepseek41 host forward: mmap failed");
        base_ = static_cast<const std::uint8_t*>(p);
#endif
    }
    ~MappedFile() {
#if defined(_WIN32)
        if (base_ != nullptr) ::UnmapViewOfFile(base_);
#else
        if (base_ != nullptr) ::munmap(const_cast<std::uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    [[nodiscard]] std::span<const std::uint8_t> bytes(std::uint64_t offset, std::uint64_t size) const {
        if (offset > size_ || size > size_ - offset) throw std::out_of_range("deepseek41 host forward: tensor outside the file");
        return {base_ + offset, static_cast<std::size_t>(size)};
    }

private:
    int fd_ = -1;
    std::uint64_t size_ = 0;
    const std::uint8_t* base_ = nullptr;
};

namespace {

constexpr float kNormEps = 1e-20F;

std::uint64_t fnv(std::span<const float> v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        auto b = std::bit_cast<std::uint32_t>(f);
        for (int i = 0; i < 4; ++i, b >>= 8U) {
            h ^= b & 0xffU;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

}  // namespace

std::vector<Op> HostForward::stand_in_ops() {
    return {};  // every piece has landed
}

HostForward::HostForward(const std::filesystem::path& gguf, const HostForwardOptions& options) : options_(options) {
    model_ = validate_target(read_file(gguf));
    plan_ = build_decode_plan(model_, 0, options.layers, true, options.head);
    file_ = std::make_unique<MappedFile>(gguf);
    if (!options.weights_overlay.empty()) overlay_ = std::make_unique<ds41::WeightOverlay>(options.weights_overlay, model_.manifest);
    for (const auto& t : model_.manifest.tensors) tensors_.emplace(t.name, &t);
    attention_ = std::make_unique<AttentionHost>(model_, options.layers, options.max_positions,
                                                 [this](const std::string& name) { return tensor(name); });
    hasher_ = std::make_unique<ds41::EngramHasher>(model_.engram);
    engram_rows_.assign(std::size_t{hasher_->layers()} * hasher_->columns(), 0U);
    for (const auto b : {Buffer::stream, Buffer::mix_attn, Buffer::mix_ffn, Buffer::pre_carry, Buffer::x, Buffer::qr, Buffer::q,
                         Buffer::kvrow, Buffer::o, Buffer::o_low, Buffer::y, Buffer::shared, Buffer::route_w, Buffer::raw})
        buf_[b].assign(plan_.buffer_elements(b), 0.0F);
}

HostForward::~HostForward() = default;

std::span<const std::uint8_t> HostForward::tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::invalid_argument("deepseek41 host forward: no tensor " + name);
    if (overlay_)
        if (const auto* e = overlay_->find(name)) return overlay_->bytes(*e);
    return file_->bytes(it->second->absolute_offset, it->second->byte_size);
}

std::string HostForward::overlay_description() const { return overlay_ ? overlay_->describe() : std::string{};
}

namespace {

// Every Q8_0 linear takes the A8 activation of its input (the layer piece's ds41-exact-v1 sites).
std::vector<float> q8_linear(std::span<const std::uint8_t> w, std::span<const float> x, std::size_t in, std::size_t out,
                             std::size_t per_group = 0) {
    std::vector<float> y(out);
    ds41::q8_0_a8_rows(w.data(), in, 0, out, per_group == 0 ? out : per_group, ds41::quantize_a8(x), y.data());
    return y;
}

std::vector<float> f16_linear(std::span<const std::uint8_t> w, std::span<const float> x, std::size_t in, std::size_t out) {
    std::vector<float> y(out);
    ds41::f16_rows(reinterpret_cast<const std::uint16_t*>(w.data()), in, 0, out, x.data(), y.data());
    return y;
}

std::span<const std::uint16_t> f16(std::span<const std::uint8_t> bytes) {
    return {reinterpret_cast<const std::uint16_t*>(bytes.data()), bytes.size() / 2U};
}

void norm(std::vector<float>& v, std::span<const float> w) {
    std::vector<float> out(v.size());
    ds41::rms_norm(v, w, 1e-20F, out);
    v.swap(out);
}

void put_mix(std::vector<float>& buf, const ds41::HcMix& m) {
    std::copy(m.pre.begin(), m.pre.end(), buf.begin());
    std::copy(m.post.begin(), m.post.end(), buf.begin() + 4);
    std::copy(m.comb.begin(), m.comb.end(), buf.begin() + 8);
}
ds41::HcMix get_mix(const std::vector<float>& buf) {
    ds41::HcMix m;
    std::copy(buf.begin(), buf.begin() + 4, m.pre.begin());
    std::copy(buf.begin() + 4, buf.begin() + 8, m.post.begin());
    std::copy(buf.begin() + 8, buf.begin() + 24, m.comb.begin());
    return m;
}

std::span<const float> f32(std::span<const std::uint8_t> bytes) {
    return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / 4U};
}

}  // namespace

const std::vector<float>& HostForward::step(std::uint32_t token) {
    const auto& c = model_.config;
    if (token >= c.vocab_size) throw std::invalid_argument("deepseek41 host forward: token outside the vocabulary");
    if (position_ >= options_.max_positions) throw std::out_of_range("deepseek41 host forward: context full");
    hasher_->push(token, engram_rows_);
    hashes_.clear();
    streams_.clear();
    for (const auto& s : plan_.steps) {
        run_step(s, token);
        if (s.op == Op::hc_post && s.site == "ffn") streams_.push_back(buf_[Buffer::stream]);
        if (!dump_label_.empty() && !s.outputs.empty() && s.outputs[0] != Buffer::logits && s.outputs[0] != Buffer::token &&
            dump_label_ == std::string(to_string(s.op)) + "@" + std::to_string(s.layer) + (s.site.empty() ? "" : "/" + s.site))
            dumped_ = buf_[s.outputs[0]];
        for (const auto b : s.outputs) {
            std::uint64_t h = 0;
            if (b == Buffer::logits) h = fnv(logits_);
            else if (b == Buffer::token) h = next_token_;
            else if (b == Buffer::route_ids) {
                for (const auto v : route_ids_) h = h * 1099511628211ULL + v;
            } else {
                h = fnv(buf_[b]);
                if (options_.buffer_tap)
                    options_.buffer_tap(std::string(to_string(s.op)) + (s.layer >= 0 ? "@" + std::to_string(s.layer) : "") +
                                            (s.site.empty() ? "" : "/" + s.site) + ":" + std::string(to_string(b)),
                                        buf_[b]);
            }
            hashes_.emplace_back(std::string(to_string(s.op)) + (s.layer >= 0 ? "@" + std::to_string(s.layer) : "") + (s.site.empty() ? "" : "/" + s.site) +
                                     ":" + std::string(to_string(b)),
                                 h);
        }
    }
    ++position_;
    return logits_;
}

void HostForward::run_step(const Step& s, std::uint32_t token) {
    const auto& c = model_.config;
    const std::size_t H = c.hidden_size, hc = c.hc_mult;
    const std::uint32_t pos = position_;
    auto& stream = buf_[Buffer::stream];
    auto& x = buf_[Buffer::x];
    switch (s.op) {
    case Op::embed: {
        ds41::embed_token(f16(tensor("token_embd.weight")).data(), H, token, stream);
        const auto pre = ds41::hc_identity_pre();
        std::copy(pre.begin(), pre.end(), buf_[Buffer::pre_carry].begin());
        return;
    }
    case Op::engram: {
        const auto& e = model_.engram;
        const auto layer = static_cast<std::uint32_t>(s.layer);
        const std::size_t li = static_cast<std::size_t>(std::find(e.layer_ids.begin(), e.layer_ids.end(), layer) - e.layer_ids.begin());
        const std::uint32_t cols = e.rows_per_token();
        const std::string p = "blk." + std::to_string(layer) + ".";
        const auto table = tensor(p + "engram_embd.weight");
        std::vector<std::uint8_t> rows(std::size_t{cols} * 264U);
        for (std::uint32_t k = 0; k < cols; ++k) {
            const auto r = engram_rows_[li * cols + k];
            std::copy_n(table.data() + r * 264U, 264, rows.data() + std::size_t{k} * 264U);
        }
        std::vector<float> emb(std::size_t{cols} * 256U);
        ds41::engram_rows_to_bf16(rows, emb);
        const auto kv = f16_linear(tensor(p + "engram_kv.weight"), emb, emb.size(), H * (hc + 1U));
        ds41::engram_inject(stream, kv, f32(tensor(p + "engram_q_norm.weight")), f32(tensor(p + "engram_k_norm.weight")), 1e-20F);
        return;
    }
    case Op::hc_mix: {
        const std::string p = "blk." + std::to_string(s.layer) + ".hc_" + s.site;
        const ds41::HcSiteWeights w{f16(tensor(p + "_fn.weight")), f32(tensor(p + "_base.weight")), f32(tensor(p + "_scale.weight"))};
        put_mix(buf_[s.site == "attn" ? Buffer::mix_attn : Buffer::mix_ffn],
                ds41::hc_mixes(stream, w, c.hc_sinkhorn_iters, static_cast<float>(c.hc_eps), 1e-20F));
        return;
    }
    case Op::hc_pre: {
        const auto& src = buf_[s.site == "attn" ? Buffer::pre_carry : Buffer::mix_attn];
        std::array<float, ds41::hc_copies> pre{};
        std::copy_n(src.begin(), ds41::hc_copies, pre.begin());
        ds41::hc_collapse(stream, pre, x);
        return;
    }
    case Op::rms_norm: {
        norm(buf_[s.site == "q" ? Buffer::qr : Buffer::x], f32(tensor(s.weights[0])));
        return;
    }
    case Op::q_a: buf_[Buffer::qr] = q8_linear(tensor(s.weights[0]), x, H, c.q_lora_rank); return;
    case Op::q_b:
        buf_[Buffer::q] = q8_linear(tensor(s.weights[0]), buf_[Buffer::qr], c.q_lora_rank, std::size_t{c.num_attention_heads} * c.head_dim);
        return;
    case Op::kv: buf_[Buffer::kvrow] = q8_linear(tensor(s.weights[0]), x, H, c.head_dim); return;
    case Op::window_write: attention_->window_write(static_cast<std::uint32_t>(s.layer), buf_[Buffer::kvrow], pos); return;
    case Op::compress: attention_->compress(static_cast<std::uint32_t>(s.layer), x, pos); return;
    case Op::index: attention_->index(static_cast<std::uint32_t>(s.layer), x, buf_[Buffer::qr], pos); return;
    case Op::attention: attention_->attention(static_cast<std::uint32_t>(s.layer), buf_[Buffer::q], pos, buf_[Buffer::o]); return;
    case Op::o_a: {
        const std::size_t in = std::size_t{c.num_attention_heads} * c.head_dim / c.o_groups;
        buf_[Buffer::o_low] = q8_linear(tensor(s.weights[0]), buf_[Buffer::o], in, std::size_t{c.o_groups} * c.o_lora_rank, c.o_lora_rank);
        return;
    }
    case Op::o_b:
    {  // two K-halves h0 + h1 (the declared order; tensor-parallel ranks each compute one half)
        auto& y = buf_[Buffer::y];
        y.assign(H, 0.0F);
        ds41::q8_0_a8_split_rows(tensor(s.weights[0]).data(), std::size_t{c.o_groups} * c.o_lora_rank, H, 2U, ds41::quantize_a8(buf_[Buffer::o_low]),
                                 y.data());
    }
        return;
    case Op::hc_post: {
        const auto& mixbuf = buf_[s.site == "attn" ? Buffer::mix_attn : Buffer::mix_ffn];
        std::vector<float> out(stream.size());
        ds41::hc_expand(buf_[Buffer::y], stream, get_mix(mixbuf), out);
        stream.swap(out);
        if (s.site == "ffn") std::copy(mixbuf.begin(), mixbuf.begin() + static_cast<std::ptrdiff_t>(hc), buf_[Buffer::pre_carry].begin());
        return;
    }
    case Op::router: {
        const auto r = ds41::reference_router(x, 1U, static_cast<std::uint32_t>(H), f32(tensor(s.weights[0])), c.n_routed_experts,
                                              f32(tensor(s.weights[1])), f32(tensor(s.weights[2])), std::vector<std::uint8_t>{0},
                                              c.num_experts_per_tok, static_cast<float>(c.routed_scaling_factor));
        route_ids_ = r.ids;
        buf_[Buffer::route_w] = r.weights;
        return;
    }
    case Op::shared_expert: {
        ds41::ExpertTensors t;
        t.gate_up_format = t.down_format = ds41::WeightFormat::q8_0;
        t.gate = tensor(s.weights[0]);
        t.up = tensor(s.weights[1]);
        t.down = tensor(s.weights[2]);
        t.hidden = static_cast<std::uint32_t>(H);
        t.inter = c.moe_intermediate_size;
        t.limit = static_cast<float>(c.swiglu_limit);
        const std::vector<ds41::A8Activation> xs{ds41::quantize_a8(x)};
        const std::vector<std::uint32_t> zero{0U};
        const std::vector<float> one{1.0F}, none(H, 0.0F);
        buf_[Buffer::shared] = ds41::reference_window(t, xs, ds41::build_window_routes(zero, 1U, 1U, 1U), one, none, 1U).raw;
        return;
    }
    case Op::routed_experts: {
        ds41::ExpertTensors t;
        t.gate = tensor(s.weights[0]);
        t.up = tensor(s.weights[1]);
        t.down = tensor(s.weights[2]);
        t.hidden = static_cast<std::uint32_t>(H);
        t.inter = c.moe_intermediate_size;
        t.limit = static_cast<float>(c.swiglu_limit);
        const std::vector<ds41::A8Activation> xs{ds41::quantize_a8(x)};
        const auto routes = ds41::build_window_routes(route_ids_, 1U, c.num_experts_per_tok, plan_.expert_split);
        if (ds41::expert_tp::enabled_from_env()) {
            // expert tensor parallelism (ds41-expert-tp2-i1152-v1): each rank folds its K-half parts of all six experts in
            // ascending id; y = (shared + sum0) + sum1. raw holds rank 0's parts (":raw" is rank-local, not compared).
            auto tp = ds41::expert_tp::reference_window(t, xs, routes, buf_[Buffer::route_w], buf_[Buffer::shared], c.num_experts_per_tok);
            buf_[Buffer::raw] = std::move(tp.partial[0]);
            buf_[Buffer::y] = std::move(tp.y);
            return;
        }
        const auto ref = ds41::reference_window(t, xs, routes, buf_[Buffer::route_w], buf_[Buffer::shared], c.num_experts_per_tok);
        buf_[Buffer::raw] = ref.raw;
        // two ranks: each owns one expert half and sends its weighted half sum; canonical = (shared + low) + high
        const auto low = ds41::reference_half_sum(routes, buf_[Buffer::route_w], ref.raw, 1U, c.num_experts_per_tok, static_cast<std::uint32_t>(H), 0U);
        const auto high = ds41::reference_half_sum(routes, buf_[Buffer::route_w], ref.raw, 1U, c.num_experts_per_tok, static_cast<std::uint32_t>(H), 1U);
        auto& y = buf_[Buffer::y];
        for (std::size_t i = 0; i < H; ++i) y[i] = (buf_[Buffer::shared][i] + low[i]) + high[i];
        if (fnv(y) != fnv(ref.y)) throw std::logic_error("deepseek41 host forward: two-half combine differs from the single-node window");
        return;
    }
    case Op::combine: return;  // folded into routed_experts above (the y of the two halves)
    case Op::head: {
        std::array<float, ds41::hc_copies> pre{};
        std::copy_n(buf_[Buffer::pre_carry].begin(), ds41::hc_copies, pre.begin());
        std::vector<float> h(H);
        ds41::hc_collapse(stream, pre, h);
        norm(h, f32(tensor("output_norm.weight")));
        logits_ = q8_linear(tensor("output.weight"), h, H, c.vocab_size);
        return;
    }
    case Op::argmax: {
        // two ranks: each takes the argmax of its vocabulary slice, then the candidates combine
        const auto r0 = ds41::vocab_slice(c.vocab_size, 2U, 0U), r1 = ds41::vocab_slice(c.vocab_size, 2U, 1U);
        const auto a0 = ds41::argmax_slice(std::span<const float>(logits_.data() + r0.first, r0.count), r0.first);
        const auto a1 = ds41::argmax_slice(std::span<const float>(logits_.data() + r1.first, r1.count), r1.first);
        const auto best = ds41::combine_argmax(a0, a1);
        if (best.finite == 0U) throw std::runtime_error("deepseek41 host forward: non-finite logits");
        next_token_ = best.index;
        return;
    }
    }
}

}  // namespace strix::deepseek41
