#include <hip/hip_runtime.h>
#include "device.hpp"
#include "gemma4/model/model.hpp"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace strix::gemma4::dflash {
namespace {
using gpu::Buffer;
void require(bool ok, const char* text) {
    if (!ok) throw std::runtime_error(std::string("device dflash: ") + text);
}
void checked(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(std::string("device dflash: ") + hipGetErrorString(e));
}
void launched() { checked(hipGetLastError()); }
__device__ void finite(float x, unsigned* failure) {
    if ((bit_cast<unsigned>(x) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
}
__device__ float sum_wave(float x) {
    for (unsigned d = 16; d; d >>= 1U) x = x + __shfl_down(x, d, 32);
    return __shfl(x, 0, 32);
}
__global__ void silu_kernel(float* gate, const float* up, unsigned n, unsigned* failure) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        finite(gate[i], failure); finite(up[i], failure);
        gate[i] = declared_silu(gate[i]) * up[i];
        finite(gate[i], failure);
    }
}
__global__ void policy_kernel(float* x, unsigned n, float scale, float cap, unsigned* failure) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = x[i] * scale;
        finite(value, failure);
        if (cap > 0.0F) {
            const float scaled = value / cap;
            finite(scaled, failure);
            value = declared_tanh(scaled) * cap;
        }
        x[i] = value;
        finite(value, failure);
    }
}
struct Read {
    const float* k;
    const float* v;
    const std::uint64_t* tags;
    const float* block_k;
    const float* block_v;
    std::uint64_t anchor, base;
    unsigned heads, dim, width, chunks;
};
__device__ std::uint64_t first(Read r, unsigned row) {
    return r.anchor + row >= window ? r.anchor + row - window + 1 : 0;
}
__device__ const float* at(Read r, std::uint64_t p, bool value, unsigned* failure) {
    if (p >= r.anchor) return (value ? r.block_v : r.block_k) + (p - r.anchor) * r.width;
    if (r.tags[p % window] != p) { atomicOr(failure, 2U); return nullptr; }
    return (value ? r.v : r.k) + (p % window) * r.width;
}
// Scores are packed relative to an absolute chunk boundary. Rows differ only
// in their left mask; the context/block boundary never splits a partial sum.
__global__ void score_kernel(const float* q, Read r, float* scores, float* maxima, unsigned* failure) {
    const unsigned rh = blockIdx.y, row = rh / r.heads, head = rh % r.heads;
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const auto begin = r.base + std::uint64_t(blockIdx.x) * attention_chunk;
    const auto left = first(r, row), end = r.anchor + block;
    auto* out = scores + std::size_t(rh) * r.chunks * attention_chunk;
    for (unsigned j = wave; j < attention_chunk; j += 4U) {
        const auto p = begin + j;
        if (p < left || p >= end) continue;
        const auto* k = at(r, p, false, failure);
        float sum = 0.0F;
        if (k)
            for (unsigned i = lane; i < r.dim; i += 32U)
                sum = sum + q[std::size_t(rh) * r.dim + i] * k[(head / 4U) * r.dim + i];
        sum = sum_wave(sum);
        if (!lane) { sum = sum * attention_scale; finite(sum, failure); out[p - r.base] = sum; }
    }
    __syncthreads();
    if (!threadIdx.x) {
        float m = bit_cast<float>(0xff800000U);
        for (unsigned j = 0; j < attention_chunk; ++j) {
            const auto p = begin + j;
            if (p >= left && p < end) m = out[p - r.base] > m ? out[p - r.base] : m;
        }
        maxima[std::size_t(rh) * r.chunks + blockIdx.x] = m;
    }
}
__global__ void max_kernel(Read r, float* maxima) {
    if (threadIdx.x) return;
    float m = bit_cast<float>(0xff800000U);
    for (unsigned c = unsigned((first(r, blockIdx.x / r.heads) - r.base) / attention_chunk); c < r.chunks; ++c) {
        const float value = maxima[std::size_t(blockIdx.x) * r.chunks + c];
        m = value > m ? value : m;
    }
    maxima[std::size_t(block * r.heads) * r.chunks + blockIdx.x] = m;
}
__global__ void partial_kernel(Read r, const float* scores, const float* maxima, float* partials, unsigned* failure) {
    const unsigned rh = blockIdx.y, row = rh / r.heads, head = rh % r.heads;
    const auto chunk_begin = r.base + std::uint64_t(blockIdx.x) * attention_chunk;
    const auto begin = max(chunk_begin, first(r, row)), end = min(chunk_begin + attention_chunk, r.anchor + block);
    __shared__ float e[attention_chunk];
    const float m = maxima[std::size_t(block * r.heads) * r.chunks + rh];
    for (auto p = begin + threadIdx.x; p < end; p += blockDim.x) {
        const float delta = scores[std::size_t(rh) * r.chunks * attention_chunk + p - r.base] - m;
        finite(delta, failure);
        e[p - chunk_begin] = declared_exp(delta);
    }
    __syncthreads();
    float* out = partials + (std::size_t(rh) * r.chunks + blockIdx.x) * (r.dim + 1U);
    if (!threadIdx.x) {
        float denominator = 0.0F;
        for (auto p = begin; p < end; ++p) denominator = denominator + e[p - chunk_begin];
        out[r.dim] = denominator; finite(denominator, failure);
    }
    for (unsigned d = threadIdx.x; d < r.dim; d += blockDim.x) {
        float numerator = 0.0F;
        for (auto p = begin; p < end; ++p) {
            const float* v = at(r, p, true, failure);
            if (v) numerator = numerator + e[p - chunk_begin] * v[(head / 4U) * r.dim + d];
        }
        out[d] = numerator; finite(numerator, failure);
    }
}
__global__ void merge_kernel(Read r, const float* partials, float* out, unsigned* failure) {
    const auto start = unsigned((first(r, blockIdx.x / r.heads) - r.base) / attention_chunk);
    float denominator = 0.0F;
    for (unsigned c = start; c < r.chunks; ++c)
        denominator = denominator + partials[(std::size_t(blockIdx.x) * r.chunks + c) * (r.dim + 1U) + r.dim];
    for (unsigned d = threadIdx.x; d < r.dim; d += blockDim.x) {
        float numerator = 0.0F;
        for (unsigned c = start; c < r.chunks; ++c)
            numerator = numerator + partials[(std::size_t(blockIdx.x) * r.chunks + c) * (r.dim + 1U) + d];
        const float value = numerator / denominator;
        out[std::size_t(blockIdx.x) * r.dim + d] = value; finite(value, failure);
    }
}
__global__ void publish_kernel(float* k, float* v, const float* staged_k, const float* staged_v,
    const std::uint64_t* positions, unsigned rows, unsigned width) {
    const unsigned layer = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const unsigned row = i / width, d = i % width;
    const auto dest = (std::size_t(layer) * window + positions[row] % window) * width + d;
    k[dest] = staged_k[std::size_t(layer) * rows * width + i];
    v[dest] = staged_v[std::size_t(layer) * rows * width + i];
}
struct Layer {
    LinearView q, k, v, o, gate, up, down, input_norm, ffn_norm, q_norm, k_norm;
};
}

struct DraftModel::Impl {
    Shape s;
    LinearView embedding, head, fc, context_norm, final_norm;
    float scale = 1, cap = 0;
    unsigned table_positions = 0;
    std::vector<Layer> layers;
    std::vector<Buffer> weights;
    Buffer rope;
    LinearView upload(const void* data, unsigned outputs, unsigned inputs, WeightFormat format, std::size_t stride) {
        weights.emplace_back(std::size_t(outputs) * stride);
        weights.back().upload(data, weights.back().size());
        return {weights.back().as<std::uint8_t>(), outputs, inputs, stride, format};
    }
    LinearView upload(const Bf16Matrix& m) { return upload(m.bits.data(), m.outputs, m.width, WeightFormat::bf16, m.width * 2ULL); }
    LinearView upload(const std::vector<float>& w) { return upload(w.data(), 1, unsigned(w.size()), WeightFormat::f32, w.size() * 4); }
};

struct DeviceDrafter::Impl {
    std::shared_ptr<const DraftModel> model;
    Shape s;
    LinearView embedding, head, fc, context_norm, final_norm;
    float scale = 1, cap = 0;
    bool shared = true, initialized = false, poisoned = false;
    std::uint64_t frontier = 0;
    unsigned table_positions = 0;
    std::vector<Layer> layers;
    const float* rope = nullptr;
    Buffer k, v, tags, staged_k, staged_v, taps, context, projected;
    Buffer residual, x, output, q, kb, vb, att, gate, up, normalized, logits, format_scratch;
    Buffer positions, tokens, scores, maxima, partials, failure, candidates, winners;
    std::array<std::uint64_t, window> host_tags;
    std::vector<std::uint64_t> pending;
    DeviceTimings times;
    hipEvent_t start = nullptr, stop = nullptr;
    ~Impl() { if (start) (void)hipEventDestroy(start); if (stop) (void)hipEventDestroy(stop); }
    void good() const { require(!poisoned, "instance poisoned by failed publication"); }
    void begin() { good(); failure.zero(); checked(hipEventRecord(start)); }
    double finish() {
        checked(hipEventRecord(stop)); checked(hipEventSynchronize(stop));
        unsigned failed = 0; failure.download(&failed, sizeof(failed));
        require(!failed, "nonfinite arithmetic or missing context tag");
        float ms = 0; checked(hipEventElapsedTime(&ms, start, stop)); return ms;
    }
    void finite(const float* values, std::size_t n) { gpu::check_finite(values, n, failure.as<unsigned>()); }
    void dot(LinearView w, const float* in, unsigned rows, float* out) {
        gpu::linear(w, in, rows, out, shared, nullptr, failure.as<unsigned>(), format_scratch.data());
    }
    void norm(const float* in, LinearView w, float* out, unsigned rows, unsigned width) {
        gpu::norm(in, w, out, rows, width); finite(out, std::size_t(rows) * width);
    }
    void rotate(float* values, LinearView w, unsigned rows, unsigned heads) {
        gpu::head_norm_rope(values, w, rows, heads, s.head_dim, positions.as<std::uint64_t>(), rope, false);
        finite(values, std::size_t(rows) * heads * s.head_dim);
    }
    void block_forward(std::uint64_t p) {
        require(p == frontier && p <= position_limit - block && p + block <= table_positions && pending.empty(), "invalid block frontier/reservation");
        const auto left = p >= window ? p - window + 1 : 0;
        for (auto j = left; j < p; ++j) require(host_tags[j % window] == j, "incomplete context tail");
        std::array<std::uint64_t, block> pos;
        std::iota(pos.begin(), pos.end(), p); positions.upload(pos.data(), sizeof(pos));
        const auto base = (left / attention_chunk) * attention_chunk;
        const unsigned chunks = unsigned((p + block - base + attention_chunk - 1) / attention_chunk);
        begin(); finite(residual.as<float>(), std::size_t(block) * s.hidden);
        for (unsigned l = 0; l < s.layers; ++l) {
            const auto& w = layers[l];
            norm(residual.as<float>(), w.input_norm, x.as<float>(), block, s.hidden);
            dot(w.q, x.as<float>(), block, q.as<float>());
            dot(w.k, x.as<float>(), block, kb.as<float>());
            dot(w.v, x.as<float>(), block, vb.as<float>());
            rotate(q.as<float>(), w.q_norm, block, s.heads); rotate(kb.as<float>(), w.k_norm, block, s.kv_heads);
            const Read read{k.as<float>() + std::size_t(l) * window * s.kv_width(),
                v.as<float>() + std::size_t(l) * window * s.kv_width(), tags.as<std::uint64_t>(),
                kb.as<float>(), vb.as<float>(), p, base, s.heads, s.head_dim, unsigned(s.kv_width()), chunks};
            hipLaunchKernelGGL(score_kernel, dim3(chunks, block * s.heads), dim3(128), 0, nullptr,
                q.as<float>(), read, scores.as<float>(), maxima.as<float>(), failure.as<unsigned>()); launched();
            hipLaunchKernelGGL(max_kernel, dim3(block * s.heads), dim3(32), 0, nullptr, read, maxima.as<float>()); launched();
            hipLaunchKernelGGL(partial_kernel, dim3(chunks, block * s.heads), dim3(128), 0, nullptr,
                read, scores.as<float>(), maxima.as<float>(), partials.as<float>(), failure.as<unsigned>()); launched();
            hipLaunchKernelGGL(merge_kernel, dim3(block * s.heads), dim3(128), 0, nullptr,
                read, partials.as<float>(), att.as<float>(), failure.as<unsigned>()); launched();
            dot(w.o, att.as<float>(), block, output.as<float>());
            gpu::residual(residual.as<float>(), output.as<float>(), block * s.hidden);
            finite(residual.as<float>(), std::size_t(block) * s.hidden);
            norm(residual.as<float>(), w.ffn_norm, x.as<float>(), block, s.hidden);
            dot(w.gate, x.as<float>(), block, gate.as<float>()); dot(w.up, x.as<float>(), block, up.as<float>());
            hipLaunchKernelGGL(silu_kernel, dim3((block * s.inter + 255U) / 256U), dim3(256), 0, nullptr,
                gate.as<float>(), up.as<float>(), block * s.inter, failure.as<unsigned>()); launched();
            dot(w.down, gate.as<float>(), block, output.as<float>());
            gpu::residual(residual.as<float>(), output.as<float>(), block * s.hidden);
            finite(residual.as<float>(), std::size_t(block) * s.hidden);
        }
        norm(residual.as<float>(), final_norm, normalized.as<float>(), block, s.hidden);
        times.block_ms = finish();
    }
    std::array<std::uint32_t, max_drafts> select() {
        begin(); finite(normalized.as<float>(), std::size_t(block) * s.hidden);
        dot(head, normalized.as<float>() + s.hidden, max_drafts, logits.as<float>());
        const unsigned n = max_drafts * head.outputs;
        hipLaunchKernelGGL(policy_kernel, dim3((n + 255U) / 256U), dim3(256), 0, nullptr,
            logits.as<float>(), n, scale, cap, failure.as<unsigned>()); launched();
        gpu::argmax(logits.as<float>(), max_drafts, head.outputs, candidates.as<gpu::Candidate>(), winners.as<gpu::Candidate>());
        times.head_ms = finish();
        std::array<gpu::Candidate, max_drafts> chosen;
        winners.download(chosen.data(), sizeof(chosen));
        std::array<std::uint32_t, max_drafts> result;
        for (unsigned d = 0; d < max_drafts; ++d) result[d] = chosen[d].id;
        return result;
    }
};

DraftModel::DraftModel(const Weights& w, RopeView table, LinearView embedding, LinearView head)
    : impl_(std::make_unique<Impl>()) {
    gpu::require_device(); w.validate(); table.validate();
    require(table.head_dim == w.shape.head_dim, "rotary shape");
    require(embedding.data && head.data && embedding.inputs == w.shape.hidden && head.inputs == w.shape.hidden &&
        embedding.outputs > mask_id && head.outputs > 0 && head.outputs <= gemma4::vocab &&
        embedding.row_bytes >= packed_row_bytes(embedding.format, embedding.inputs) &&
        (head.format == WeightFormat::exl3_mul1 ? exl3::geometry(head) : head.row_bytes >= packed_row_bytes(head.format, head.inputs)), "target asset geometry");
    auto& d = *impl_; d.s = w.shape; d.embedding = embedding; d.head = head; d.scale = w.logit_scale; d.cap = w.logit_cap;
    d.table_positions = table.positions;
    d.fc = d.upload(w.fc); d.context_norm = d.upload(w.context_norm); d.final_norm = d.upload(w.final_norm);
    for (const auto& l : w.layer)
        d.layers.push_back({d.upload(l.q), d.upload(l.k), d.upload(l.v), d.upload(l.o), d.upload(l.gate), d.upload(l.up),
            d.upload(l.down), d.upload(l.input_norm), d.upload(l.ffn_norm), d.upload(l.q_norm), d.upload(l.k_norm)});
    // Lossless layout adapter v1: B sliding stride=128 interleaved pairs; only
    // head_dim/2 pairs execute. Padding is immutable (1,0), never runtime trig.
    std::vector<float> packed(std::size_t(table.positions) * 256, 0.0F);
    for (unsigned p = 0; p < table.positions; ++p)
        for (unsigned i = 0; i < 128; ++i) {
            packed[std::size_t(p) * 256 + 2 * i] = i < table.head_dim / 2 ? table.cosine[std::size_t(p) * (table.head_dim / 2) + i] : 1.0F;
            if (i < table.head_dim / 2) packed[std::size_t(p) * 256 + 2 * i + 1] = table.sine[std::size_t(p) * (table.head_dim / 2) + i];
        }
    d.rope.resize(packed.size() * 4); d.rope.upload(packed.data(), d.rope.size());
}
DraftModel::~DraftModel() = default;
const Shape& DraftModel::shape() const { return impl_->s; }
std::size_t DraftModel::resident_bytes() const {
    std::size_t bytes = impl_->rope.size();
    for (const auto& w : impl_->weights) bytes += w.size();
    return bytes;
}
DeviceDrafter::DeviceDrafter(const Weights& w, RopeView table, LinearView embedding, LinearView head, bool shared)
    : DeviceDrafter(std::make_shared<DraftModel>(w, table, embedding, head), shared) {}
DeviceDrafter::DeviceDrafter(std::shared_ptr<const DraftModel> model, bool shared)
    : impl_(std::make_unique<Impl>()) {
    require(bool(model), "missing immutable draft model");
    auto& d = *impl_; d.model = std::move(model);
    const auto& m = *d.model->impl_;
    d.s = m.s; d.embedding = m.embedding; d.head = m.head;
    d.fc = m.fc; d.context_norm = m.context_norm; d.final_norm = m.final_norm;
    d.scale = m.scale; d.cap = m.cap; d.layers = m.layers;
    d.table_positions = m.table_positions; d.rope = m.rope.as<float>();
    const auto* off = std::getenv("STRIX_G4_SHARED_WEIGHTS");
    d.shared = shared && (!off || std::string_view(off) != "0");
    auto floats = [](Buffer& b, std::size_t count) { b.resize(count * sizeof(float)); };
    const auto& s = d.s;
    floats(d.k, std::size_t(s.layers) * window * s.kv_width()); floats(d.v, d.k.size() / 4);
    d.host_tags.fill(~std::uint64_t{0}); d.tags.resize(sizeof(d.host_tags)); d.tags.upload(d.host_tags.data(), sizeof(d.host_tags));
    floats(d.staged_k, std::size_t(s.layers) * block * s.kv_width()); floats(d.staged_v, d.staged_k.size() / 4);
    floats(d.taps, block * s.tap_width()); floats(d.context, block * s.hidden); floats(d.projected, block * s.hidden);
    for (auto* b : {&d.residual, &d.x, &d.output, &d.normalized}) floats(*b, block * s.hidden);
    for (auto* b : {&d.q, &d.att}) floats(*b, block * s.q_width());
    for (auto* b : {&d.kb, &d.vb}) floats(*b, block * s.kv_width());
    floats(d.gate, block * s.inter); floats(d.up, block * s.inter); floats(d.logits, max_drafts * d.head.outputs);
    d.format_scratch.resize(block * std::size_t(s.hidden) * sizeof(std::uint16_t));
    d.positions.resize(block * sizeof(std::uint64_t)); d.tokens.resize(block * sizeof(std::uint32_t));
    constexpr unsigned chunks = 18;
    floats(d.scores, std::size_t(block) * s.heads * chunks * attention_chunk);
    floats(d.maxima, std::size_t(block) * s.heads * (chunks + 1));
    floats(d.partials, std::size_t(block) * s.heads * chunks * (s.head_dim + 1));
    d.failure.resize(sizeof(unsigned)); d.candidates.resize(max_drafts * ((d.head.outputs + 255ULL) / 256) * sizeof(gpu::Candidate));
    d.winners.resize(max_drafts * sizeof(gpu::Candidate));
    checked(hipEventCreate(&d.start)); checked(hipEventCreate(&d.stop));
}
DeviceDrafter::~DeviceDrafter() = default;
const Shape& DeviceDrafter::shape() const { return impl_->s; }
std::uint64_t DeviceDrafter::frontier() const { return impl_->frontier; }
const DeviceTimings& DeviceDrafter::timings() const { return impl_->times; }
const DraftModel& DeviceDrafter::model() const { return *impl_->model; }
bool DeviceDrafter::reusable() const noexcept { return !impl_->poisoned; }
void DeviceDrafter::reset() {
    auto& d = *impl_; d.good();
    // No discarded tag can expose an old row. Clear payload as well so reuse
    // has bounded context-ring work independent of checkpoint/table size.
    d.pending.clear(); d.k.zero(); d.v.zero();
    d.host_tags.fill(~std::uint64_t{0});
    d.tags.upload(d.host_tags.data(), sizeof(d.host_tags));
    d.frontier = 0; d.initialized = false; d.times = {};
}
void DeviceDrafter::stage_append(std::span<const float> taps, std::span<const std::uint64_t> positions) {
    auto& d = *impl_; d.good();
    require(!positions.empty() && positions.size() <= block && d.pending.empty() &&
        taps.size() == positions.size() * d.s.tap_width(), "append geometry/pending state");
    require(!d.initialized || positions.front() == d.frontier, "duplicate, gap or rollback in append");
    for (std::size_t r = 0; r < positions.size(); ++r)
        require(positions[r] == positions.front() + r && positions[r] < position_limit && positions[r] < d.table_positions, "append positions");
    const auto rows = unsigned(positions.size());
    d.taps.upload(taps.data(), taps.size_bytes()); d.positions.upload(positions.data(), positions.size_bytes());
    d.begin(); d.finite(d.taps.as<float>(), taps.size());
    d.dot(d.fc, d.taps.as<float>(), rows, d.projected.as<float>());
    d.norm(d.projected.as<float>(), d.context_norm, d.context.as<float>(), rows, d.s.hidden);
    for (unsigned l = 0; l < d.s.layers; ++l) {
        auto* k = d.staged_k.as<float>() + std::size_t(l) * rows * d.s.kv_width();
        auto* v = d.staged_v.as<float>() + std::size_t(l) * rows * d.s.kv_width();
        d.dot(d.layers[l].k, d.context.as<float>(), rows, k); d.dot(d.layers[l].v, d.context.as<float>(), rows, v);
        d.rotate(k, d.layers[l].k_norm, rows, d.s.kv_heads);
    }
    d.times.append_ms = d.finish();
    d.pending.assign(positions.begin(), positions.end());
}
void DeviceDrafter::publish_append() {
    auto& d = *impl_; d.good(); require(!d.pending.empty(), "no staged append");
    try {
        const auto rows = unsigned(d.pending.size()), width = unsigned(d.s.kv_width());
        hipLaunchKernelGGL(publish_kernel, dim3((rows * width + 255U) / 256U, d.s.layers), dim3(256), 0, nullptr,
            d.k.as<float>(), d.v.as<float>(), d.staged_k.as<float>(), d.staged_v.as<float>(), d.positions.as<std::uint64_t>(), rows, width);
        launched(); gpu::synchronize();
        auto tags = d.host_tags;
        for (auto p : d.pending) tags[p % window] = p;
        d.tags.upload(tags.data(), sizeof(tags));
        d.host_tags = tags; d.frontier = d.pending.back() + 1; d.initialized = true; d.pending.clear();
    } catch (...) { d.poisoned = true; throw; }
}
void DeviceDrafter::discard_append() noexcept { impl_->pending.clear(); }
void DeviceDrafter::append(std::span<const float> taps, std::span<const std::uint64_t> positions) {
    if (positions.empty()) { require(taps.empty(), "empty append shape"); return; }
    stage_append(taps, positions); publish_append();
}
std::vector<float> DeviceDrafter::forward(std::span<const float> embedded, std::uint64_t position) {
    auto& d = *impl_; require(embedded.size() == std::size_t(block) * d.s.hidden, "embedded shape");
    d.residual.upload(embedded.data(), embedded.size_bytes()); d.block_forward(position);
    std::vector<float> result(embedded.size()); d.normalized.download(result.data(), result.size() * 4); return result;
}
std::array<std::uint32_t, max_drafts> DeviceDrafter::select(std::span<const float> normalized) {
    auto& d = *impl_; require(normalized.size() == std::size_t(block) * d.s.hidden, "normalized shape");
    d.normalized.upload(normalized.data(), normalized.size_bytes()); return d.select();
}
std::vector<float> DeviceDrafter::selected_logits() const {
    const auto& d = *impl_; d.good(); std::vector<float> result(std::size_t(max_drafts) * d.head.outputs);
    d.logits.download(result.data(), result.size() * sizeof(float)); return result;
}
std::array<std::uint32_t, max_drafts> DeviceDrafter::draft(std::uint32_t anchor, std::uint64_t position) {
    auto& d = *impl_; require(anchor < d.embedding.outputs, "anchor token");
    std::array<std::uint32_t, block> tokens; tokens.fill(mask_id); tokens[0] = anchor;
    d.tokens.upload(tokens.data(), sizeof(tokens));
    gpu::embedding(d.embedding, d.tokens.as<unsigned>(), block, d.residual.as<float>(), embedding_scale);
    d.block_forward(position); return d.select();
}
std::vector<float> DeviceDrafter::context_key(std::uint32_t layer, std::uint64_t position) const {
    const auto& d = *impl_; d.good(); require(layer < d.s.layers && position < position_limit &&
        d.host_tags[position % window] == position, "missing key");
    std::vector<float> result(d.s.kv_width());
    d.k.download(result.data(), result.size() * 4, (std::size_t(layer) * window + position % window) * result.size() * 4); return result;
}
std::vector<float> DeviceDrafter::context_value(std::uint32_t layer, std::uint64_t position) const {
    const auto& d = *impl_; d.good(); require(layer < d.s.layers && position < position_limit &&
        d.host_tags[position % window] == position, "missing value");
    std::vector<float> result(d.s.kv_width());
    d.v.download(result.data(), result.size() * 4, (std::size_t(layer) * window + position % window) * result.size() * 4); return result;
}
} // namespace strix::gemma4::dflash
