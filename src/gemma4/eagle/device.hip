#include <hip/hip_runtime.h>
#include "device.hpp"
#include "gemma4/kv/storage.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::gemma4::eagle {
namespace {
using gpu::Buffer;
void require(bool ok, const char* text) {
    if (!ok) throw std::runtime_error(std::string("device eagle: ") + text);
}
void checked(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(std::string("device eagle: ") + hipGetErrorString(e));
}
void launched() { checked(hipGetLastError()); }

struct DeviceResult {
    unsigned ids[8];
    unsigned tokens[8];
    float prob[8];
};

__device__ void flag(float x, unsigned* failure) {
    if ((bit_cast<unsigned>(x) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
}
// Lane-ordered partial sums followed by the g4-v1 butterfly (lane l += lane l+d, d = 16..1), as lane_dot.
__device__ float wave_reduce(float a) {
    for (unsigned d = 16; d; d >>= 1U) a = a + __shfl_down(a, d, 32);
    return __shfl(a, 0, 32);
}
// rms_norm factor: correctly rounded rsqrt(sum/width + eps); nonfinite -> NaN and a failure flag.
__device__ float norm_factor(float sum, unsigned width, unsigned* failure) {
    const float u = sum / float(width) + norm_eps;
    if ((bit_cast<unsigned>(u) & 0x7f800000U) == 0x7f800000U) {
        atomicOr(failure, 1U);
        return bit_cast<float>(0x7fc00000U);
    }
    return pow32_rsqrt(u);
}

// One wave per (row, tap): out = RMSNorm_j(tap_j). Input layout [row][tap][width].
__global__ void tap_norm_kernel(const float* taps, const float* weights, float* out, unsigned tap_count, unsigned width,
                                unsigned* failure) {
    const unsigned lane = threadIdx.x, j = blockIdx.x % tap_count;
    const float* x = taps + std::size_t(blockIdx.x) * width;
    const float* w = weights + std::size_t(j) * width;
    float s = 0.0F;
    for (unsigned i = lane; i < width; i += 32U) s = s + x[i] * x[i];
    const float r = norm_factor(wave_reduce(s), width, failure);
    for (unsigned i = lane; i < width; i += 32U) {
        const float y = (x[i] * r) * w[i];
        flag(y, failure);
        out[std::size_t(blockIdx.x) * width + i] = y;
    }
}

__device__ float embedded(LinearView table, unsigned token, unsigned i, float scale) {
    return embedding_value(gpu::coefficient(table, token, i), scale);
}
// concat: one wave per (row, part); part 0 -> RMSNorm_h(h) into out[row][0:w], part 1 -> RMSNorm_e(e) into out[row][w:2w].
// sum: one wave per row; out[row] = RMSNorm_h(h) + RMSNorm_e(e).
__global__ void input_kernel(const float* h, LinearView table, const unsigned* tokens, const float* h_norm,
                             const float* e_norm, float* out, unsigned width, unsigned target_vocab, bool concat,
                             float scale, unsigned* failure) {
    const unsigned lane = threadIdx.x;
    const unsigned row = concat ? blockIdx.x / 2U : blockIdx.x, part = concat ? blockIdx.x % 2U : 0U;
    const unsigned token = tokens[row];
    if (token >= target_vocab || token >= table.outputs) {
        if (lane == 0) atomicOr(failure, 4U);
        return;
    }
    const float* x = h + std::size_t(row) * width;
    if (concat) {
        float s = 0.0F;
        if (part == 0) {
            for (unsigned i = lane; i < width; i += 32U) s = s + x[i] * x[i];
        } else {
            for (unsigned i = lane; i < width; i += 32U) {
                const float e = embedded(table, token, i, scale);
                s = s + e * e;
            }
        }
        const float r = norm_factor(wave_reduce(s), width, failure);
        float* y = out + std::size_t(row) * 2U * width + (part ? width : 0U);
        for (unsigned i = lane; i < width; i += 32U) {
            const float value = part == 0 ? (x[i] * r) * h_norm[i] : (embedded(table, token, i, scale) * r) * e_norm[i];
            flag(value, failure);
            y[i] = value;
        }
        return;
    }
    float sh = 0.0F, se = 0.0F;
    for (unsigned i = lane; i < width; i += 32U) {
        sh = sh + x[i] * x[i];
        const float e = embedded(table, token, i, scale);
        se = se + e * e;
    }
    const float rh = norm_factor(wave_reduce(sh), width, failure), re = norm_factor(wave_reduce(se), width, failure);
    float* y = out + std::size_t(row) * width;
    for (unsigned i = lane; i < width; i += 32U) {
        const float a = (x[i] * rh) * h_norm[i];
        const float b = (embedded(table, token, i, scale) * re) * e_norm[i];
        const float value = a + b;
        flag(value, failure);
        y[i] = value;
    }
}

// One wave per (row, head slot) for head_dim 256. Slots: [0,q) Q heads in place (norm with the scaled Q weight,
// then RoPE); [q,q+k) K heads (norm, RoPE) -> k_dst; [q+k,q+k+v) V heads (unscaled norm) -> v_dst.
__global__ void heads_kernel(float* src, unsigned row_stride, unsigned q_heads, unsigned k_heads, unsigned v_heads,
                             unsigned k_offset, unsigned v_offset, const float* q_norm, const float* k_norm,
                             const std::uint64_t* positions, const float* table, float* k_dst, float* v_dst,
                             unsigned dst_stride, unsigned* failure) {
    constexpr unsigned dim = 256, per = dim / 32U;
    const unsigned slots = q_heads + k_heads + v_heads, row = blockIdx.x / slots, slot = blockIdx.x % slots;
    const unsigned lane = threadIdx.x;
    float* x;
    float* dst;
    const float* w;
    bool rotate = true;
    if (slot < q_heads) {
        x = src + std::size_t(row) * row_stride + std::size_t(slot) * dim;
        dst = x; w = q_norm;
    } else if (slot < q_heads + k_heads) {
        const auto head = slot - q_heads;
        x = src + std::size_t(row) * row_stride + k_offset + std::size_t(head) * dim;
        dst = k_dst + std::size_t(row) * dst_stride + std::size_t(head) * dim; w = k_norm;
    } else {
        const auto head = slot - q_heads - k_heads;
        x = src + std::size_t(row) * row_stride + v_offset + std::size_t(head) * dim;
        dst = v_dst + std::size_t(row) * dst_stride + std::size_t(head) * dim; w = nullptr; rotate = false;
    }
    float y[per];
    float s = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < per; ++j) {
        y[j] = x[lane + 32U * j];
        s = s + y[j] * y[j];
    }
    const float r = norm_factor(wave_reduce(s), dim, failure);
#pragma unroll
    for (unsigned j = 0; j < per; ++j) y[j] = w ? (y[j] * r) * w[lane + 32U * j] : y[j] * r;
    if (rotate) {
        const float* t = table + std::size_t(positions[row]) * 256U;
#pragma unroll
        for (unsigned j = 0; j < per / 2U; ++j) {
            const unsigned i = lane + 32U * j;
            const float a = y[j], b = y[j + per / 2U], cs = t[2U * i], sn = t[2U * i + 1U];
            y[j] = a * cs - b * sn;
            y[j + per / 2U] = b * cs + a * sn;
        }
    }
#pragma unroll
    for (unsigned j = 0; j < per; ++j) {
        flag(y[j], failure);
        dst[lane + 32U * j] = y[j];
    }
}

__global__ void publish_kernel(float* ring_k, float* ring_v, std::uint64_t* tags, const float* staged_k,
                               const float* staged_v, const std::uint64_t* positions, unsigned rows, unsigned width,
                               unsigned slots) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const unsigned row = i / width, d = i % width;
    const auto slot = std::size_t(positions[row] % slots);
    ring_k[slot * width + d] = staged_k[i];
    ring_v[slot * width + d] = staged_v[i];
    if (d == 0) tags[slot] = positions[row];
}

// Capped argmax over the draft vocabulary (low id on ties), top-1 probability with the fixed 1024-way strided
// partial sums and binary tree of eagle::select(), d2t mapping and the next chain token.
__global__ void __launch_bounds__(1024) select_kernel(const float* logits, unsigned vocabulary, float cap,
                                                      const unsigned* d2t, unsigned depth, unsigned* next_token,
                                                      DeviceResult* result, unsigned* failure) {
    __shared__ float best_value[select_threads];
    __shared__ unsigned best_id[select_threads];
    __shared__ float sums[select_threads];
    const unsigned t = threadIdx.x;
    float value = bit_cast<float>(0xff800000U);
    unsigned id = UINT32_MAX;
    for (unsigned i = t; i < vocabulary; i += select_threads) {
        const float raw = logits[i];
        flag(raw, failure);
        const float capped = declared_tanh(raw / cap) * cap;
        if (better(capped, i, value, id)) { value = capped; id = i; }
    }
    best_value[t] = value; best_id[t] = id;
    __syncthreads();
    for (unsigned s = select_threads / 2U; s; s >>= 1U) {
        if (t < s && better(best_value[t + s], best_id[t + s], best_value[t], best_id[t])) {
            best_value[t] = best_value[t + s]; best_id[t] = best_id[t + s];
        }
        __syncthreads();
    }
    const float maximum = best_value[0];
    float acc = 0.0F;
    for (unsigned i = t; i < vocabulary; i += select_threads)
        acc = acc + declared_exp(declared_tanh(logits[i] / cap) * cap - maximum);
    sums[t] = acc;
    __syncthreads();
    for (unsigned s = select_threads / 2U; s; s >>= 1U) {
        if (t < s) sums[t] = sums[t] + sums[t + s];
        __syncthreads();
    }
    if (t == 0) {
        const unsigned draft = best_id[0];
        const unsigned token = draft < vocabulary ? d2t[draft] : 0U;
        if (draft >= vocabulary) atomicOr(failure, 1U);
        result->ids[depth] = draft;
        result->tokens[depth] = token;
        result->prob[depth] = 1.0F / sums[0];
        *next_token = token;
    }
}
} // namespace

// ------------------------------------------------------------------ model
struct DeviceModel::Impl {
    Config c;
    LinearView embedding;
    std::vector<Buffer> store;
    LinearView fc, in_proj, qkv, q, kv, o, gate, up, down, head;
    LinearView input_norm, post_attn, pre_ffn, post_ffn, final_norm;
    const float *tap_norm = nullptr, *h_norm = nullptr, *e_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    const unsigned* d2t = nullptr;
    const float* rope = nullptr;
    std::uint32_t rope_positions = 0;
    std::size_t bytes = 0;

    const std::uint8_t* raw(const void* data, std::size_t size) {
        store.emplace_back(size);
        store.back().upload(data, size);
        bytes += size;
        return store.back().as<std::uint8_t>();
    }
    LinearView matrix(const Matrix& m) {
        if (m.bytes.empty()) return {};
        return {raw(m.bytes.data(), m.bytes.size()), m.outputs, m.inputs, m.row_bytes(), m.format};
    }
    LinearView norm_view(const std::vector<float>& w) {
        return {raw(w.data(), w.size() * 4), 1, std::uint32_t(w.size()), w.size() * 4, WeightFormat::f32};
    }
};

DeviceModel::DeviceModel(const Weights& w, RopeView rope, LinearView target_embedding) : impl_(std::make_unique<Impl>()) {
    gpu::require_device();
    w.validate();
    rope.validate();
    auto& d = *impl_;
    d.c = w.config;
    const auto& c = d.c;
    require(target_embedding.data && target_embedding.inputs == c.hidden && target_embedding.outputs >= c.target_vocab &&
            target_embedding.row_bytes >= row_bytes(target_embedding.format, target_embedding.inputs),
            "target embedding geometry");
    d.embedding = target_embedding;
    std::vector<float> taps;
    for (const auto& t : w.tap_norm) taps.insert(taps.end(), t.begin(), t.end());
    d.tap_norm = reinterpret_cast<const float*>(d.raw(taps.data(), taps.size() * 4));
    d.fc = d.matrix(w.fc);
    d.h_norm = reinterpret_cast<const float*>(d.raw(w.h_norm.data(), w.h_norm.size() * 4));
    d.e_norm = reinterpret_cast<const float*>(d.raw(w.e_norm.data(), w.e_norm.size() * 4));
    d.in_proj = d.matrix(w.in_proj);
    d.input_norm = d.norm_view(w.input_norm);
    // Fused QKV: row concatenation of the original packed rows (blocks never cross rows).
    std::vector<std::uint8_t> fused(w.q.bytes);
    fused.insert(fused.end(), w.k.bytes.begin(), w.k.bytes.end());
    fused.insert(fused.end(), w.v.bytes.begin(), w.v.bytes.end());
    const auto stride = w.q.row_bytes();
    const auto* qkv = d.raw(fused.data(), fused.size());
    d.qkv = {qkv, c.q_width() + 2 * c.kv_width(), c.hidden, stride, w.q.format};
    d.q = {qkv, c.q_width(), c.hidden, stride, w.q.format};
    d.kv = {qkv + std::size_t(c.q_width()) * stride, 2 * c.kv_width(), c.hidden, stride, w.q.format};
    const auto qn = w.q_norm_effective();
    d.q_norm = reinterpret_cast<const float*>(d.raw(qn.data(), qn.size() * 4));
    d.k_norm = reinterpret_cast<const float*>(d.raw(w.k_norm.data(), w.k_norm.size() * 4));
    d.o = d.matrix(w.o);
    d.post_attn = d.norm_view(w.post_attn_norm);
    d.pre_ffn = d.norm_view(w.pre_ffn_norm);
    d.gate = d.matrix(w.gate);
    d.up = d.matrix(w.up);
    d.down = d.matrix(w.down);
    d.post_ffn = d.norm_view(w.post_ffn_norm);
    d.final_norm = d.norm_view(w.final_norm);
    d.head = d.matrix(w.head);
    d.d2t = reinterpret_cast<const unsigned*>(d.raw(w.d2t.data(), w.d2t.size() * 4));
    d.rope = reinterpret_cast<const float*>(d.raw(rope.table.data(), rope.table.size() * 4));
    d.rope_positions = rope.positions;
}
DeviceModel::~DeviceModel() = default;
const Config& DeviceModel::config() const { return impl_->c; }
std::size_t DeviceModel::resident_bytes() const { return impl_->bytes; }
namespace {
std::size_t matrix_bytes(LinearView v) { return v.data ? std::size_t(v.outputs) * v.row_bytes : 0; }
}
std::size_t DeviceModel::step_weight_bytes() const {
    const auto& d = *impl_;
    return matrix_bytes(d.in_proj) + matrix_bytes(d.qkv) + matrix_bytes(d.o) + matrix_bytes(d.gate) + matrix_bytes(d.up) +
           matrix_bytes(d.down) + matrix_bytes(d.head) + d.embedding.row_bytes;
}
std::size_t DeviceModel::first_depth_weight_bytes() const {
    const auto& d = *impl_;
    return matrix_bytes(d.q) + matrix_bytes(d.o) + matrix_bytes(d.gate) + matrix_bytes(d.up) + matrix_bytes(d.down) +
           matrix_bytes(d.head);
}
std::size_t DeviceModel::stage_weight_bytes() const {
    const auto& d = *impl_;
    return matrix_bytes(d.fc) + matrix_bytes(d.in_proj) + matrix_bytes(d.kv);
}

// ------------------------------------------------------------------ drafter
struct DeviceDrafter::Impl {
    std::shared_ptr<const DeviceModel> model;
    const DeviceModel::Impl* m = nullptr;
    Config c;
    bool shared = true, poisoned = false, initialized = false, has_last = false, staged = false;
    std::uint64_t frontier = 0;
    std::uint32_t last_next = 0, staged_next = 0;
    std::vector<std::uint64_t> host_tags, staged_positions;
    Buffer ring_k, ring_v, tags;
    Buffer taps_in, tn, hbuf, cat, xbuf, abuf, kvout, sk, sv, spos, stokens, last_x, last_a;
    Buffer state, a1, qkv, att, out, y, gate, normed, logits, ok, ov, reads, cpos, next_token, result;
    Buffer scores, maxima, partials, failure;
    std::vector<float> host_taps;
    Timings times;
    hipEvent_t start = nullptr, stop = nullptr;
    bool one_chunk = false;

    ~Impl() {
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
    }
    void good() const { require(!poisoned, "instance poisoned by failed publication"); }
    unsigned* fail() const { return failure.as<unsigned>(); }
    void begin() { checked(hipMemsetAsync(failure.data(), 0, sizeof(unsigned), nullptr)); checked(hipEventRecord(start, nullptr)); }
    double finish(const char* what) {
        checked(hipEventRecord(stop, nullptr));
        checked(hipEventSynchronize(stop));
        unsigned failed = 0;
        failure.download(&failed, sizeof(failed));
        if (failed) throw std::runtime_error(std::string("device eagle: ") + what + " failed (nonfinite=1, missing KV tag=2, token=4): " +
                                             std::to_string(failed));
        float ms = 0;
        checked(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
    void dot(LinearView w, const float* x, unsigned rows, float* dst) {
        gpu::linear(w, x, rows, dst, shared, nullptr, fail());
        ++times.launches;
    }
    void heads(float* src, unsigned rows, unsigned row_stride, bool q_heads, bool kv_heads, unsigned k_offset, unsigned v_offset,
               const std::uint64_t* positions, float* k_dst, float* v_dst) {
        const unsigned qh = q_heads ? c.heads : 0U, kh = kv_heads ? c.kv_heads : 0U;
        hipLaunchKernelGGL(heads_kernel, dim3(rows * (qh + 2U * kh)), dim3(32), 0, nullptr, src, row_stride, qh, kh, kh,
                           k_offset, v_offset, m->q_norm, m->k_norm, positions, m->rope, k_dst, v_dst, c.kv_width(), fail());
        launched(); ++times.launches;
    }
    void input(const float* h, const unsigned* tokens, unsigned rows, float* dst) {
        hipLaunchKernelGGL(input_kernel, dim3(c.concat ? 2U * rows : rows), dim3(32), 0, nullptr, h, m->embedding, tokens,
                           m->h_norm, m->e_norm, dst, c.hidden, c.target_vocab, c.concat, c.embedding_scale, fail());
        launched(); ++times.launches;
    }
    // o projection, both residual epilogues, GeGLU, final norm, head and selection for one row at depth index.
    void tail(float* resid, unsigned depth) {
        dot(m->o, att.as<float>(), 1, out.as<float>());
        gpu::norm_residual(out.as<float>(), m->post_attn, resid, 1, c.hidden, {}, y.as<float>(), m->pre_ffn, fail());
        ++times.launches;
        gpu::gate_up(m->gate, m->up, y.as<float>(), 1, gate.as<float>(), shared, nullptr, fail());
        ++times.launches;
        dot(m->down, gate.as<float>(), 1, out.as<float>());
        gpu::norm_residual(out.as<float>(), m->post_ffn, resid, 1, c.hidden, {}, normed.as<float>(), m->final_norm, fail());
        ++times.launches;
        dot(m->head, normed.as<float>(), 1, logits.as<float>());
        hipLaunchKernelGGL(select_kernel, dim3(1), dim3(select_threads), 0, nullptr, logits.as<float>(), c.vocab, c.softcap,
                           m->d2t, depth, next_token.as<unsigned>(), result.as<DeviceResult>(), fail());
        launched(); ++times.launches;
    }
    void attend(const gpu::KvRead* read, unsigned stride) {
        gpu::attention(qkv.as<float>(), read, 1, c.heads, c.head_dim, stride, scores.as<float>(), maxima.as<float>(),
                       partials.as<float>(), att.as<float>(), fail(), nullptr, true);
        times.launches += one_chunk && stride <= attention_chunk ? 2U : 4U;
    }
};

DeviceDrafter::DeviceDrafter(std::shared_ptr<const DeviceModel> model, bool shared) : impl_(std::make_unique<Impl>()) {
    require(bool(model), "missing immutable drafter model");
    auto& d = *impl_;
    d.model = std::move(model);
    d.m = d.model->impl_.get();
    d.c = d.m->c;
    const auto& c = d.c;
    const auto* off = std::getenv("STRIX_G4_SHARED_WEIGHTS");
    d.shared = shared && (!off || std::string_view(off) != "0");
    const auto* oc = std::getenv("STRIX_G4_ONE_CHUNK_ATTN");
    const auto* fusion = std::getenv("STRIX_G4_FUSION");
    d.one_chunk = oc && std::string_view(oc) == "1" && (!fusion || std::string_view(fusion) != "0");
    auto floats = [](Buffer& b, std::size_t count) { b.resize(count * sizeof(float)); b.zero(); };
    const std::size_t H = c.hidden, kw = c.kv_width(), qw = c.q_width(), R = max_rows, T = c.tap_count();
    floats(d.ring_k, std::size_t(c.window) * kw); floats(d.ring_v, std::size_t(c.window) * kw);
    d.host_tags.assign(c.window, UINT64_MAX);
    d.tags.resize(c.window * sizeof(std::uint64_t));
    d.tags.upload(d.host_tags.data(), d.tags.size());
    floats(d.taps_in, R * T * H); floats(d.tn, R * T * H); floats(d.hbuf, R * H); floats(d.cat, R * 2 * H);
    floats(d.xbuf, R * H); floats(d.abuf, R * H); floats(d.kvout, R * 2 * kw); floats(d.sk, R * kw); floats(d.sv, R * kw);
    d.spos.resize(R * sizeof(std::uint64_t)); d.stokens.resize(R * sizeof(unsigned));
    floats(d.last_x, H); floats(d.last_a, H);
    floats(d.state, 2 * H); floats(d.a1, H); floats(d.qkv, qw + 2 * kw); floats(d.att, qw); floats(d.out, H); floats(d.y, H);
    floats(d.gate, c.ffn); floats(d.normed, H); floats(d.logits, c.vocab);
    floats(d.ok, std::size_t(max_depth) * kw); floats(d.ov, std::size_t(max_depth) * kw);
    d.reads.resize(max_depth * sizeof(gpu::KvRead)); d.cpos.resize(max_depth * sizeof(std::uint64_t));
    d.next_token.resize(sizeof(unsigned)); d.result.resize(sizeof(DeviceResult));
    constexpr std::size_t chunks = gemma4::max_positions / attention_chunk;
    floats(d.scores, std::size_t(c.heads) * gemma4::max_positions);
    floats(d.maxima, std::size_t(c.heads) * (chunks + 1));
    floats(d.partials, std::size_t(c.heads) * chunks * (c.head_dim + 1));
    d.failure.resize(sizeof(unsigned));
    checked(hipEventCreate(&d.start));
    checked(hipEventCreate(&d.stop));
}
DeviceDrafter::~DeviceDrafter() = default;
const Config& DeviceDrafter::config() const { return impl_->c; }
std::uint64_t DeviceDrafter::frontier() const { return impl_->frontier; }
const Timings& DeviceDrafter::timings() const { return impl_->times; }
bool DeviceDrafter::reusable() const noexcept { return !impl_->poisoned; }
const DeviceModel& DeviceDrafter::model() const { return *impl_->model; }
void DeviceDrafter::reset() {
    auto& d = *impl_;
    d.good();
    d.host_tags.assign(d.c.window, UINT64_MAX);
    d.tags.upload(d.host_tags.data(), d.tags.size());
    d.ring_k.zero(); d.ring_v.zero();
    d.frontier = 0; d.initialized = false; d.has_last = false; d.staged = false; d.times = {};
}
void DeviceDrafter::stage(const Rows& rows) {
    auto& d = *impl_;
    d.good();
    const auto& c = d.c;
    const auto n = rows.positions.size();
    const std::size_t H = c.hidden, T = c.tap_count(), kw = c.kv_width();
    require(!d.staged, "previous stage not published/discarded");
    require(n >= 1 && n <= max_rows && rows.taps.size() == n * engine_taps * H && rows.next_tokens.size() == n &&
            (!c.needs_final() || rows.hidden.size() == n * H), "stage geometry");
    require(!d.initialized || rows.positions[0] == d.frontier, "duplicate, gap or rollback in drafter stage");
    for (std::size_t r = 0; r < n; ++r)
        require(rows.positions[r] == rows.positions[0] + r && rows.positions[r] < d.m->rope_positions &&
                rows.next_tokens[r] < c.target_vocab, "stage positions/tokens");
    // Host-side tap selection: [row][tap][H] in fc order (engine order 5/11/17/23/35 + final hidden).
    d.host_taps.resize(n * T * H);
    for (std::size_t r = 0; r < n; ++r)
        for (std::uint32_t j = 0; j < T; ++j) {
            const auto slot = c.tap_slot(j);
            const float* src = slot < engine_taps ? rows.taps.data() + (r * engine_taps + slot) * H : rows.hidden.data() + r * H;
            std::copy_n(src, H, d.host_taps.data() + (r * T + j) * H);
        }
    d.taps_in.upload(d.host_taps.data(), d.host_taps.size() * 4);
    d.stokens.upload(rows.next_tokens.data(), n * sizeof(unsigned));
    d.spos.upload(rows.positions.data(), n * sizeof(std::uint64_t));
    const auto launches = d.times.launches;
    d.times.launches = 0;
    d.begin();
    hipLaunchKernelGGL(tap_norm_kernel, dim3(unsigned(n * T)), dim3(32), 0, nullptr, d.taps_in.as<float>(), d.m->tap_norm,
                       d.tn.as<float>(), unsigned(T), unsigned(H), d.fail());
    launched(); ++d.times.launches;
    d.dot(d.m->fc, d.tn.as<float>(), unsigned(n), d.hbuf.as<float>());
    d.input(d.hbuf.as<float>(), d.stokens.as<unsigned>(), unsigned(n), c.concat ? d.cat.as<float>() : d.xbuf.as<float>());
    if (c.concat) d.dot(d.m->in_proj, d.cat.as<float>(), unsigned(n), d.xbuf.as<float>());
    gpu::norm(d.xbuf.as<float>(), d.m->input_norm, d.abuf.as<float>(), unsigned(n), c.hidden, nullptr, d.fail());
    ++d.times.launches;
    d.dot(d.m->kv, d.abuf.as<float>(), unsigned(n), d.kvout.as<float>());
    d.heads(d.kvout.as<float>(), unsigned(n), unsigned(2 * kw), false, true, 0, unsigned(kw), d.spos.as<std::uint64_t>(),
            d.sk.as<float>(), d.sv.as<float>());
    d.times.stage_launches = d.times.launches;
    d.times.launches = launches;
    d.times.stage_ms = d.finish("stage");
    d.staged_positions.assign(rows.positions.begin(), rows.positions.end());
    d.staged_next = rows.next_tokens[n - 1];
    d.staged = true;
}
void DeviceDrafter::publish() {
    auto& d = *impl_;
    d.good();
    require(d.staged, "no staged rows");
    try {
        const auto n = unsigned(d.staged_positions.size()), kw = d.c.kv_width();
        const std::size_t H = d.c.hidden;
        d.begin();
        hipLaunchKernelGGL(publish_kernel, dim3((n * kw + 255U) / 256U), dim3(256), 0, nullptr, d.ring_k.as<float>(),
                           d.ring_v.as<float>(), d.tags.as<std::uint64_t>(), d.sk.as<float>(), d.sv.as<float>(),
                           d.spos.as<std::uint64_t>(), n, kw, d.c.window);
        launched();
        checked(hipMemcpyAsync(d.last_x.data(), d.xbuf.as<float>() + (n - 1) * H, H * 4, hipMemcpyDeviceToDevice, nullptr));
        checked(hipMemcpyAsync(d.last_a.data(), d.abuf.as<float>() + (n - 1) * H, H * 4, hipMemcpyDeviceToDevice, nullptr));
        d.times.publish_ms = d.finish("publish");
        for (const auto p : d.staged_positions) d.host_tags[p % d.c.window] = p;
        d.frontier = d.staged_positions.back() + 1;
        d.last_next = d.staged_next;
        d.initialized = true; d.has_last = true; d.staged = false;
    } catch (...) {
        d.poisoned = true;
        throw;
    }
}
void DeviceDrafter::discard() noexcept { impl_->staged = false; }
Draft DeviceDrafter::draft(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) {
    auto& d = *impl_;
    d.good();
    const auto& c = d.c;
    require(!d.staged && d.has_last && position == d.frontier && position > 0, "draft position must equal the committed frontier");
    require(anchor == d.last_next, "anchor differs from the last committed row's next token");
    require(k >= 1 && k <= max_depth && position + k - 1 < d.m->rope_positions && position + k <= gemma4::max_positions,
            "draft depth");
    const auto r = position - 1;
    const auto first = r < c.window ? 0 : r - c.window + 1;
    for (auto p = first; p <= r; ++p) require(d.host_tags[p % c.window] == p, "incomplete drafter context tail");
    const std::size_t H = c.hidden, kw = c.kv_width(), qw = c.q_width();
    gpu::KvRead reads[max_depth];
    std::uint64_t positions[max_depth];
    for (std::uint32_t depth = 1; depth <= k; ++depth) {
        positions[depth - 1] = r + depth - 1;
        reads[depth - 1] = {d.ring_k.as<float>(), d.ring_v.as<float>(), d.tags.as<std::uint64_t>(), d.ok.as<float>(),
                            d.ov.as<float>(), position, r + depth - 1, unsigned(kw), c.window, depth - 1, c.window, false};
    }
    d.reads.upload(reads, k * sizeof(gpu::KvRead));
    d.cpos.upload(positions, k * sizeof(std::uint64_t));
    const auto stride = unsigned((r + k + attention_chunk - 1) / attention_chunk * attention_chunk);
    const auto* read = d.reads.as<gpu::KvRead>();
    const auto* cpos = d.cpos.as<std::uint64_t>();
    float* states[2] = {d.state.as<float>(), d.state.as<float>() + H};
    d.times.launches = 0;
    d.begin();
    // depth 1: committed row r, cached x/a from its stage; K/V already in the ring.
    checked(hipMemcpyAsync(states[0], d.last_x.data(), H * 4, hipMemcpyDeviceToDevice, nullptr));
    d.dot(d.m->q, d.last_a.as<float>(), 1, d.qkv.as<float>());
    d.heads(d.qkv.as<float>(), 1, unsigned(qw), true, false, 0, 0, cpos, nullptr, nullptr);
    d.attend(read, stride);
    d.tail(states[0], 0);
    for (std::uint32_t depth = 2; depth <= k; ++depth) {
        float* prev = states[(depth - 2) % 2];
        float* cur = states[(depth - 1) % 2];
        d.input(prev, d.next_token.as<unsigned>(), 1, c.concat ? d.cat.as<float>() : cur);
        if (c.concat) d.dot(d.m->in_proj, d.cat.as<float>(), 1, cur);
        gpu::norm(cur, d.m->input_norm, d.a1.as<float>(), 1, c.hidden, nullptr, d.fail());
        ++d.times.launches;
        d.dot(d.m->qkv, d.a1.as<float>(), 1, d.qkv.as<float>());
        d.heads(d.qkv.as<float>(), 1, unsigned(qw + 2 * kw), true, true, unsigned(qw), unsigned(qw + kw), cpos + (depth - 1),
                d.ok.as<float>() + (depth - 2) * kw, d.ov.as<float>() + (depth - 2) * kw);
        d.attend(read + (depth - 1), stride);
        d.tail(cur, depth - 1);
    }
    d.times.draft_ms = d.finish("draft");
    DeviceResult host{};
    d.result.download(&host, sizeof(host));
    Draft out;
    out.depth = k;
    for (std::uint32_t i = 0; i < k; ++i) {
        out.draft_ids[i] = host.ids[i];
        out.tokens[i] = host.tokens[i];
        out.prob[i] = host.prob[i];
    }
    return out;
}
std::vector<float> DeviceDrafter::key(std::uint64_t position) const {
    const auto& d = *impl_;
    const auto slot = std::size_t(position % d.c.window);
    if (d.host_tags[slot] != position) return {};
    std::vector<float> out(d.c.kv_width());
    d.ring_k.download(out.data(), out.size() * 4, slot * out.size() * 4);
    return out;
}
std::vector<float> DeviceDrafter::value(std::uint64_t position) const {
    const auto& d = *impl_;
    const auto slot = std::size_t(position % d.c.window);
    if (d.host_tags[slot] != position) return {};
    std::vector<float> out(d.c.kv_width());
    d.ring_v.download(out.data(), out.size() * 4, slot * out.size() * 4);
    return out;
}

} // namespace strix::gemma4::eagle
