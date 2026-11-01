#include "device.hpp"
#include "validation.hpp"
#include "gemma4/kernels/rows_linear.hpp"
#include "gemma4/exl3/device.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <string_view>
#include <algorithm>
#include <cfenv>
#include <map>
#include <stdexcept>

namespace strix::gemma4::prefill {
namespace {
bool pack_reuse_enabled() {
    static const bool enabled = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_PACK_REUSE"); return !p || std::string_view(p) != "0"; }();
    return enabled;
}
using gpu::Buffer;
void checked(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
void launched() { checked(hipGetLastError()); }
__global__ void gather(LinearView table, const unsigned* tokens, unsigned rows, float scale, float* out) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(rows) * table.inputs)
        out[i] = embedding_value(gpu::coefficient(table, tokens[i / table.inputs], unsigned(i % table.inputs)), scale);
}
__global__ void scale_ple(float* x, unsigned count) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) x[i] = x[i] * model_ple_scale;
}
__global__ void combine_ple(float* x, LinearView table, const unsigned* tokens, unsigned count) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float v = embedding_value(gpu::coefficient(table, tokens[i / table.inputs], unsigned(i % table.inputs)), ple_embedding_scale);
        x[i] = (x[i] + v) * ple_combine_scale;
    }
}
__global__ void rotate(float* x, const std::uint64_t* positions, const float* table, unsigned rows, unsigned heads_count,
                        unsigned dim, bool global) {
    const unsigned active = min(dim / 2U, global ? 64U : 128U);
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(rows) * heads_count * active) return;
    const auto rh = i / active, pair = i % active;
    const auto index = rh * dim + pair, t = std::size_t(positions[rh / heads_count]) * (global ? 512U : 256U) + pair * 2;
    const float a = x[index], b = x[index + dim / 2], cs = table[t], sn = table[t + 1];
    x[index] = a * cs - b * sn;
    x[index + dim / 2] = b * cs + a * sn;
}
__global__ void ple_product(float* g, const float* ple, unsigned count, unsigned width, unsigned layers_count, unsigned layer) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) g[i] = gelu_tanh(g[i]) * ple[(i / width * layers_count + layer) * width + i % width];
}
} // namespace
struct DevicePrefill::Impl {
    HostWeights w;
    std::vector<Buffer> storage;
    std::map<const std::uint8_t*, LinearView> uploaded;
    std::map<const std::uint8_t*, std::unique_ptr<PreparedLinear>> prepared;
    Buffer sliding, global, residual, x, q, k, v, att, out, gate, up, ple, logits, taps;
    Buffer tokens, positions, reads, writes, scores, maxima, partials, failure, halves, maximum, attention_halves, global_halves, pair_halves, last_row;
    Buffer canonical_scores, canonical_denominators, canonical_packed, exl3_scratch;
    unsigned table_positions = 0, stride = 0;
    std::vector<Range> ranges;
    std::map<std::string, float> range_maxima;
    Stats stats;
    bool profile = false, fuse_ffn = false;
    hipEvent_t start = nullptr, stop = nullptr, converted = nullptr;
    hipStream_t queue = nullptr;
    ~Impl() {
        if (queue) (void)hipStreamDestroy(queue);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
        if (converted) (void)hipEventDestroy(converted);
    }
    template <class F> void timing(double& total, F body) {
        if (!profile) { body(); return; }
        checked(hipEventRecord(start, queue)); body(); checked(hipEventRecord(stop, queue));
        checked(hipEventSynchronize(stop));
        float ms = 0.0F; checked(hipEventElapsedTime(&ms, start, stop)); total += ms;
    }
    LinearView upload(LinearView view, Residency residency) {
        if (!view.data || residency == Residency::device) return view;
        if (const auto it = uploaded.find(view.data); it != uploaded.end()) {
            const auto& d = it->second;
            if (d.inputs != view.inputs || d.outputs != view.outputs || d.format != view.format || d.row_bytes != view.row_bytes)
                throw std::invalid_argument("ambiguous prefill tensor alias");
            return d;
        }
        validate_finite(view);
        storage.emplace_back(weight_storage_bytes(view));
        storage.back().upload(view.data, storage.back().size());
        const auto* original = view.data;
        view.data = storage.back().as<std::uint8_t>();
        uploaded.emplace(original, view);
        return view;
    }
    void finite(const Buffer& b, std::size_t count) { gpu::check_finite(b.as<float>(), count, failure.as<unsigned>(), queue); }
    // Fused gate/up/product/convert: leaves the down input panels in halves
    // and the three range maxima at the indices separate dots would use.
    bool fused(LinearView gate, LinearView up, LinearView down) const {
        if (!fuse_ffn || stats.direct < 3 || down.format == WeightFormat::exl3_mul1) return false;
        const auto g = prepared.find(gate.data), u = prepared.find(up.data), d = prepared.find(down.data);
        return g != prepared.end() && u != prepared.end() && d != prepared.end() && gelu_pair_supported(*g->second, *u->second);
    }
    void gelu_pair(const std::string& name, LinearView gate, LinearView up, unsigned rows) {
        const auto index = unsigned(ranges.size());
        ranges.push_back({name + "gate", 0.0F}); ranges.push_back({name + "up", 0.0F});
        timing(stats.linear_ms, [&] {
            prefill::gelu_pair(*prepared.at(gate.data), *prepared.at(up.data), x.as<float>(), rows, pair_halves.data(), halves.data(),
                               failure.as<unsigned>(), maximum.as<unsigned>() + index, queue);
        });
    }
    // EXL3 FFN on prepared panels (g4-exl3-wmma-pf-v1): down's input panels in halves.
    bool exl3_pair(const std::string& name, LinearView wg, LinearView wu, LinearView wd, unsigned rows) {
        if (!fuse_ffn || wg.format != WeightFormat::exl3_mul1) return false;
        const auto g = prepared.find(wg.data), u = prepared.find(wu.data);
        if (g == prepared.end() || u == prepared.end() || !prepared.contains(wd.data) ||
            !exl3_gelu_pair_supported(*g->second, *u->second, wd)) return false;
        const auto index = unsigned(ranges.size());
        ranges.push_back({name + "gate", 0.0F}); ranges.push_back({name + "up", 0.0F});
        record(name + "gate_up", [&] {
            prefill::exl3_gelu_pair(*g->second, *u->second, wd, x.as<float>(), rows, pair_halves.data(), gate.as<float>(),
                                    up.as<float>(), halves.data(), failure.as<unsigned>(), maximum.as<unsigned>() + index, queue);
        }, 4.0 * rows * wg.inputs * wg.outputs, true);
        return true;
    }
    // g4-v1 rows linear, or the EXL3 pack's own identity: many-row core for
    // chunks, its eight-row decode linear otherwise (same bytes either way).
    void exact_linear(LinearView matrix, const float* input, unsigned rows, float* output) {
        if (matrix.format != WeightFormat::exl3_mul1) {
            gpu::linear_rows(matrix, input, rows, output, queue, failure.as<unsigned>());
            return;
        }
        const auto bytes = std::max(exl3::device_scratch_bytes, exl3::rows_scratch_bytes(matrix, rows));
        if (exl3_scratch.size() < bytes) exl3_scratch.resize(bytes);
        if (exl3::device_linear_rows(matrix, input, rows, output, exl3_scratch.data(), exl3_scratch.size(), failure.as<unsigned>(), queue))
            return;
        for (unsigned r = 0; r < rows; r += 8)
            exl3::device_linear(matrix, input + std::size_t(r) * matrix.inputs, std::min(8U, rows - r),
                                output + std::size_t(r) * matrix.outputs, exl3_scratch.data(), failure.as<unsigned>(), true, queue);
    }
    // EXL3 many-row FFN (gate, up, fused middle stage, down) into out; false,
    // enqueueing nothing, when the pack, identity or switches select the separate linears.
    bool exl3_ffn(const std::string& name, LinearView wg, LinearView wu, LinearView wd, unsigned rows) {
        if (wg.format != WeightFormat::exl3_mul1 || wu.format != WeightFormat::exl3_mul1 || wd.format != WeightFormat::exl3_mul1)
            return false;
        const auto bytes = exl3::ffn_rows_scratch_bytes(wg, wu, wd, rows);
        if (!bytes) return false;
        if (exl3_scratch.size() < bytes) exl3_scratch.resize(bytes);
        bool done = false;
        record(name + "ffn", [&] {
            done = exl3::device_ffn_rows(wg, wu, wd, x.as<float>(), rows, gate.as<float>(), up.as<float>(), out.as<float>(),
                                         exl3_scratch.data(), exl3_scratch.size(), failure.as<unsigned>(), queue);
        }, 2.0 * rows * (2.0 * wg.inputs * wg.outputs + double(wd.inputs) * wd.outputs), true);
        if (done) { ranges.push_back({name + "gate", 0.0F}); ranges.push_back({name + "up", 0.0F}); ranges.push_back({name + "down", 0.0F}); }
        return done;
    }
    // Profiled stage outside dot(): its own kernel record (linear time when linear).
    template <class F> void record(const std::string& name, F body, double flops = 0.0, bool linear = false) {
        double elapsed = 0.0;
        timing(elapsed, body);
        (linear ? stats.linear_ms : stats.other_ms) += elapsed;
        if (profile) { auto& p = stats.kernels[name]; ++p.calls; p.ms += elapsed; p.matrix_ms += linear ? elapsed : 0.0; p.flops += flops; }
    }
    void dot(const std::string& name, LinearView matrix, const float* input, float* output, unsigned rows, bool preconverted = false) {
        const auto index = unsigned(ranges.size());
        ranges.push_back({name, 0.0F});
        const auto before = stats.linear_ms;
        if (stats.linears == Linears::exact) {
            timing(stats.linear_ms, [&] { exact_linear(matrix, input, rows, output); });
            if (profile) {
                auto& p = stats.kernels[name]; ++p.calls; p.ms += stats.linear_ms - before; p.matrix_ms += stats.linear_ms - before;
                p.flops += 2.0 * rows * matrix.inputs * matrix.outputs;
                p.bytes += double(rows) * (matrix.inputs + matrix.outputs) * sizeof(float) + double(matrix.row_bytes) * matrix.outputs;
            }
            return;
        }
        timing(stats.linear_ms, [&] {
            const auto it = prepared.find(matrix.data);
            prefill::linear(matrix, input, rows, output, halves.data(), failure.as<unsigned>(), maximum.as<unsigned>() + index, queue,
                            true, (rows > 8 || stats.direct >= 3) && it != prepared.end() ? it->second.get() : nullptr, converted, preconverted);
        });
        if (profile) {
            auto& p = stats.kernels[name]; ++p.calls; p.ms += stats.linear_ms - before;
            float ms = 0.0F;
            checked(hipEventElapsedTime(&ms, start, converted)); p.conversion_ms += ms;
            checked(hipEventElapsedTime(&ms, converted, stop)); p.matrix_ms += ms;
            p.flops += 2.0 * rows * matrix.inputs * matrix.outputs;
            // Algorithmic minimum traffic, not a hardware bandwidth counter.
            p.bytes += double(rows) * (matrix.inputs + matrix.outputs) * sizeof(float) +
                       (stats.prepared && (rows > 8 || stats.direct >= 3) ? 2.0 * matrix.inputs * matrix.outputs : double(matrix.row_bytes) * matrix.outputs);
        }
    }
    void norm(const Buffer& input, LinearView weight, Buffer& output, unsigned vectors, unsigned width) {
        gpu::norm(input.as<float>(), weight, output.as<float>(), vectors, width, queue, failure.as<unsigned>());
    }
    void head(Buffer& b, LinearView norm, unsigned rows, unsigned count, unsigned dim, bool is_global) {
        gpu::norm(b.as<float>(), norm, b.as<float>(), rows * count, dim, queue, failure.as<unsigned>());
        const auto total = rows * count * std::min(dim / 2, is_global ? 64U : 128U);
        hipLaunchKernelGGL(rotate, dim3((total + 255U) / 256U), dim3(256), 0, queue, b.as<float>(), positions.as<std::uint64_t>(),
                           is_global ? global.as<float>() : sliding.as<float>(), rows, count, dim, is_global);
        launched(); finite(b, std::size_t(rows) * count * dim);
    }
    void add(const Buffer& branch, LinearView norm, unsigned rows, LinearView scalar = {}) {
        this->norm(branch, norm, x, rows, w.config.hidden);
        gpu::residual(residual.as<float>(), x.as<float>(), rows * w.config.hidden, scalar, queue);
        finite(residual, std::size_t(rows) * w.config.hidden);
    }
};
DevicePrefill::DevicePrefill(HostWeights weights, RopeTables t, Residency residency, bool profile, AttentionMode attention,
                             Linears linears) : impl_(std::make_unique<Impl>()) {
    detail::validate_weights(weights);
    if (linears == Linears::exact && attention != AttentionMode::canonical)
        throw std::invalid_argument("exact prefill linears require canonical attention");
    gpu::require_device();
    if (std::fegetround() != FE_TONEAREST) throw std::invalid_argument("prefill requires RNE");
    detail::validate_tables(t);
    auto& b = *impl_;
    b.profile = profile;
    b.stats.attention = attention;
    b.stats.attention_packed = attention == AttentionMode::wmma && attention_pack_enabled();
    b.stats.canonical_kernel = attention == AttentionMode::canonical && canonical_kernel_enabled();
    b.stats.linears = linears;
    b.stats.tiled = linears == Linears::wmma && tiled_enabled();
    b.stats.prepared = prepared_enabled() && b.stats.tiled;
    b.stats.direct = b.stats.prepared ? direct_mode() : 0;
    if (profile) { checked(hipEventCreate(&b.start)); checked(hipEventCreate(&b.stop)); checked(hipEventCreate(&b.converted)); }
    b.w = weights;
    b.w.embedding = b.upload(weights.embedding, residency); b.w.ple_table = b.upload(weights.ple_table, residency);
    b.w.head = b.upload(weights.head, residency);
    b.w.ple_model = b.upload(weights.ple_model, residency); b.w.ple_model_norm = b.upload(weights.ple_model_norm, residency);
    b.w.final_norm = b.upload(weights.final_norm, residency);
    for (auto& y : b.w.layer)
        for (auto* view : {&y.input_norm, &y.post_attention_norm, &y.pre_ffn_norm, &y.post_ffn_norm, &y.post_ple_norm,
                          &y.q, &y.k, &y.v, &y.o, &y.q_norm, &y.k_norm, &y.gate, &y.up, &y.down, &y.ple_gate, &y.ple_projection, &y.scalar})
            *view = b.upload(*view, residency);
    b.table_positions = t.positions;
    b.sliding.resize(t.sliding.size_bytes()); b.sliding.upload(t.sliding.data(), b.sliding.size());
    b.global.resize(t.global.size_bytes()); b.global.upload(t.global.data(), b.global.size());
    checked(hipStreamCreateWithFlags(&b.queue, hipStreamNonBlocking));
    if (b.stats.prepared) {
        const auto prepare = [&](LinearView matrix) {
            if (matrix.data && !b.prepared.contains(matrix.data)) {
                auto p = std::make_unique<PreparedLinear>(matrix, b.queue);
                b.stats.prepared_bytes += p->panels().size();
                b.prepared.emplace(matrix.data, std::move(p));
            }
        };
        prepare(b.w.head.data ? b.w.head : b.w.embedding); prepare(b.w.ple_model);
        for (const auto& y : b.w.layer)
            for (auto matrix : {y.q, y.k, y.v, y.o, y.gate, y.up, y.down, y.ple_gate, y.ple_projection}) prepare(matrix);
    }
    const auto& c = b.w.config;
    const std::size_t hbytes = chunk_rows * std::size_t(c.hidden) * sizeof(float), abytes = chunk_rows * std::size_t(c.heads) * 512 * sizeof(float);
    for (auto* p : {&b.residual, &b.x, &b.out}) p->resize(hbytes);
    for (auto* p : {&b.q, &b.att}) p->resize(abytes);
    for (auto* p : {&b.k, &b.v}) p->resize(chunk_rows * std::size_t(c.kv_heads) * 512 * sizeof(float));
    b.gate.resize(chunk_rows * std::size_t(std::max(c.inter, c.ple_dim)) * sizeof(float)); b.up.resize(b.gate.size());
    b.ple.resize(chunk_rows * std::size_t(c.layers) * c.ple_dim * sizeof(float));
    b.logits.resize(std::size_t(c.vocab) * sizeof(float)); b.taps.resize(5 * hbytes);
    b.tokens.resize(chunk_rows * sizeof(unsigned)); b.positions.resize(chunk_rows * sizeof(std::uint64_t));
    b.reads.resize(chunk_rows * sizeof(gpu::KvRead)); b.writes.resize(chunk_rows * sizeof(gpu::KvWrite));
    b.failure.resize(sizeof(unsigned)); b.maximum.resize((2 + 11 * c.layers) * sizeof(unsigned));
    b.halves.resize(chunk_rows * std::size_t((std::max({c.hidden, c.inter, c.heads * 512, c.ple_dim}) + 15U) / 16U) * 16 * sizeof(std::uint16_t));
    b.last_row.resize(std::size_t(c.hidden) * sizeof(float));
    b.fuse_ffn = linears == Linears::wmma && fused_ffn_enabled();
    if (b.fuse_ffn) b.pair_halves.resize(chunk_rows * std::size_t((c.hidden + 15U) / 16U) * 16 * sizeof(std::uint16_t));
}
DevicePrefill::~DevicePrefill() = default;
const std::vector<Range>& DevicePrefill::ranges() const { return impl_->ranges; }
const Stats& DevicePrefill::stats() const { return impl_->stats; }
const HostWeights& DevicePrefill::borrow_device_weights() const { return impl_->w; }
void DevicePrefill::set_profile(bool enabled) {
    auto& b = *impl_;
    if (enabled && !b.start) { checked(hipEventCreate(&b.start)); checked(hipEventCreate(&b.stop)); checked(hipEventCreate(&b.converted)); }
    b.profile = enabled;
}
void DevicePrefill::use_canonical_kernel(bool enabled) {
    impl_->stats.canonical_kernel = impl_->stats.attention == AttentionMode::canonical && enabled;
}
ChunkResult DevicePrefill::forward(KvTxn& transaction, std::span<const unsigned> input, bool all_logits, bool logits, bool hidden) {
    auto& b = *impl_; const auto& w = b.w; const auto& c = w.config;
    auto* tx = dynamic_cast<gpu::DeviceStream::Impl::Txn*>(&transaction);
    const auto n = unsigned(input.size());
    if (!tx || tx->published || tx->failed || tx->evaluated || tx->rows || tx->stream->poisoned || !gpu::same_config(tx->stream->config, c))
        throw std::invalid_argument("invalid prefill transaction");
    if (!n || input.size() > chunk_rows || tx->start > tx->stream->admitted || input.size() > tx->stream->admitted - tx->start ||
        tx->start > b.table_positions || input.size() > b.table_positions - tx->start)
        throw std::invalid_argument("prefill row or position bounds");
    for (const auto t : input) if (t >= c.vocab) throw std::invalid_argument("prefill token out of range");
    ChunkResult result; result.rows = n; result.vocab = c.vocab; result.tap_width = 5 * c.hidden;
    try {
        tx->rows = n;
        for (unsigned l = 0; l < c.layers; ++l) {
            const auto bytes = std::size_t(n) * tx->stream->cache[l].width * sizeof(float);
            if (bytes > tx->overlay[l].k.size()) { tx->overlay[l].k.resize(bytes); tx->overlay[l].v.resize(bytes); }
        }
        b.ranges.clear(); b.maximum.zero(); b.failure.zero(); b.taps.zero();
        b.tokens.upload(input.data(), input.size_bytes());
        std::vector<std::uint64_t> positions(n);
        for (unsigned r = 0; r < n; ++r) positions[r] = tx->start + r;
        b.positions.upload(positions.data(), positions.size() * sizeof(std::uint64_t));
        const auto stride = unsigned(((tx->start + n + 63U) / 64U) * 64U), chunks = stride / 64;
        if (stride > b.stride) {
            b.stride = stride;
            b.scores.resize(8ULL * c.heads * stride * sizeof(float));
            b.maxima.resize(8ULL * c.heads * (chunks + 1ULL) * sizeof(float));
            b.partials.resize(8ULL * c.heads * chunks * 513 * sizeof(float));
            if (b.stats.attention_packed) {
                b.attention_halves.resize(attention_packed_bytes(stride, c.kv_heads * 512));
                b.global_halves.resize(attention_packed_bytes(stride, c.kv_heads * 512));
            }
        }
        if (b.stats.canonical_kernel) {
            // Global layers span every key from 0: the widest canonical score span.
            const auto bytes = std::size_t(n) * c.heads * stride * sizeof(float);
            if (bytes > b.canonical_scores.size()) b.canonical_scores.resize(bytes);
            if (b.canonical_denominators.size() < chunk_rows * std::size_t(c.heads) * sizeof(float))
                b.canonical_denominators.resize(chunk_rows * std::size_t(c.heads) * sizeof(float));
            const auto packed = canonical_packed_floats(chunk_rows, c.heads, 512, stride) * sizeof(float);
            if (canonical_pack_enabled() && packed > b.canonical_packed.size()) b.canonical_packed.resize(packed);
        }
        hipLaunchKernelGGL(gather, dim3((n * c.hidden + 255U) / 256U), dim3(256), 0, b.queue, w.embedding,
                           b.tokens.as<unsigned>(), n, embedding_scale, b.residual.as<float>());
        launched(); b.finite(b.residual, std::size_t(n) * c.hidden);
        b.dot("ple_model", w.ple_model, b.residual.as<float>(), b.ple.as<float>(), n);
        const auto pn = n * c.layers * c.ple_dim;
        hipLaunchKernelGGL(scale_ple, dim3((pn + 255U) / 256U), dim3(256), 0, b.queue, b.ple.as<float>(), pn);
        launched();
        gpu::norm(b.ple.as<float>(), w.ple_model_norm, b.ple.as<float>(), n * c.layers, c.ple_dim, b.queue, b.failure.as<unsigned>());
        hipLaunchKernelGGL(combine_ple, dim3((pn + 255U) / 256U), dim3(256), 0, b.queue, b.ple.as<float>(), w.ple_table, b.tokens.as<unsigned>(), pn);
        launched(); b.finite(b.ple, pn);
        // Descriptors are immutable for the chunk: upload every layer's reads
        // and the producers' writes once instead of draining the queue before a
        // synchronous upload in each layer. The canonical kernel also reads the
        // host copy.
        std::vector<gpu::KvRead> all_reads(std::size_t(c.layers) * n);
        {
            auto& reads = all_reads;
            std::vector<gpu::KvWrite> writes(std::size_t(c.layers) * n);
            for (unsigned l = 0; l < c.layers; ++l)
                for (unsigned r = 0; r < n; ++r) {
                    reads[std::size_t(l) * n + r] = tx->read(c.layer[l].producer, tx->start + r);
                    writes[std::size_t(l) * n + r] = {tx->overlay[l].k.as<float>(), tx->overlay[l].v.as<float>(), r};
                }
            if (b.reads.size() < reads.size() * sizeof(gpu::KvRead)) b.reads.resize(reads.size() * sizeof(gpu::KvRead));
            if (b.writes.size() < writes.size() * sizeof(gpu::KvWrite)) b.writes.resize(writes.size() * sizeof(gpu::KvWrite));
            b.reads.upload(reads.data(), reads.size() * sizeof(gpu::KvRead));
            b.writes.upload(writes.data(), writes.size() * sizeof(gpu::KvWrite));
        }
        // Sliding and global layers pack into separate buffers, so a layer that
        // reads its producer's KV (layers >= 24) reuses that producer's pack.
        unsigned packed_producer[2] = {~0U, ~0U};
        for (unsigned l = 0; l < c.layers; ++l) {
            const auto& y = w.layer[l]; const auto& a = c.layer[l]; const auto d = a.head_dim;
            auto* layer_reads = b.reads.as<gpu::KvRead>() + std::size_t(l) * n;
            const bool global = a.attention == AttentionKind::global;
            const auto name = "layer." + std::to_string(l) + ".";
            b.norm(b.residual, y.input_norm, b.x, n, c.hidden);
            b.dot(name + "q", y.q, b.x.as<float>(), b.q.as<float>(), n); b.head(b.q, y.q_norm, n, c.heads, d, global);
            if (a.producer == l) {
                b.dot(name + "k", y.k, b.x.as<float>(), b.k.as<float>(), n); b.dot(name + "v", y.v, b.x.as<float>(), b.v.as<float>(), n);
                b.head(b.k, y.k_norm, n, c.kv_heads, d, global);
                gpu::norm(b.v.as<float>(), {}, b.v.as<float>(), n * c.kv_heads, d, b.queue, b.failure.as<unsigned>());
                gpu::store_overlay(b.k.as<float>(), b.v.as<float>(), b.writes.as<gpu::KvWrite>() + std::size_t(l) * n, n, c.kv_heads * d, b.queue);
            }
            b.timing(b.stats.attention_ms, [&] {
            if (b.stats.attention == AttentionMode::wmma) {
                auto& owner = packed_producer[global ? 1 : 0];
                const bool pack = !b.stats.attention_packed || owner != a.producer || !pack_reuse_enabled();
                owner = a.producer;
                prefill::attention(b.q.as<float>(), layer_reads, n, c.heads, d, b.att.as<float>(), b.failure.as<unsigned>(), b.queue,
                                    b.stats.attention_packed ? (global ? b.global_halves : b.attention_halves).data() : nullptr,
                                    stride, global ? stride : std::min(stride, c.window + n - 1), pack);
            } else if (!b.stats.canonical_kernel ||
                       !canonical_attention(b.q.as<float>(), std::span<const gpu::KvRead>(all_reads).subspan(std::size_t(l) * n, n), c.heads, d, b.att.as<float>(), b.failure.as<unsigned>(), b.queue,
                                            {b.canonical_scores.as<float>(), b.canonical_scores.size() / sizeof(float),
                                             b.canonical_denominators.as<float>(), b.canonical_packed.as<float>(),
                                             b.canonical_packed.size() / sizeof(float)})) {
            for (unsigned r = 0; r < n; r += 8) {
                const auto count = std::min(8U, n - r);
                gpu::attention(b.q.as<float>() + std::size_t(r) * c.heads * d, layer_reads + r, count, c.heads, d,
                               stride, b.scores.as<float>(), b.maxima.as<float>(), b.partials.as<float>(),
                               b.att.as<float>() + std::size_t(r) * c.heads * d, b.failure.as<unsigned>(), b.queue);
            }
            }
            });
            b.dot(name + "o", y.o, b.att.as<float>(), b.out.as<float>(), n); b.add(b.out, y.post_attention_norm, n);
            b.norm(b.residual, y.pre_ffn_norm, b.x, n, c.hidden);
            if (b.stats.linears == Linears::exact && b.exl3_ffn(name, y.gate, y.up, y.down, n)) {
                // EXL3 many-row FFN, down included.
            } else if (b.stats.linears == Linears::exact) {
                // The decode target's gate_up: gelu_tanh(gate) * up stored per element.
                if (y.gate.format == WeightFormat::exl3_mul1 || y.up.format == WeightFormat::exl3_mul1) {
                    // EXL3 packs: separate linears (each profiled), then the decode target's gelu product.
                    b.dot(name + "gate", y.gate, b.x.as<float>(), b.gate.as<float>(), n);
                    b.dot(name + "up", y.up, b.x.as<float>(), b.up.as<float>(), n);
                    b.record(name + "gelu_product", [&] { gpu::gelu_product(b.gate.as<float>(), b.up.as<float>(), n * c.inter, b.queue); });
                } else {
                    b.ranges.push_back({name + "gate", 0.0F}); b.ranges.push_back({name + "up", 0.0F});
                    b.record(name + "gate_up", [&] {
                        gpu::gate_up_rows(y.gate, y.up, b.x.as<float>(), n, b.gate.as<float>(), b.queue, b.failure.as<unsigned>(),
                                          gpu::rows_linear_enabled(), -1, b.up.as<float>());
                    }, 4.0 * n * y.gate.inputs * y.gate.outputs, true);
                }
                b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n);
            } else if (b.exl3_pair(name, y.gate, y.up, y.down, n)) {
                b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n, true);
            } else if (b.fused(y.gate, y.up, y.down)) {
                b.gelu_pair(name, y.gate, y.up, n);
                b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n, true);
            } else {
                b.dot(name + "gate", y.gate, b.x.as<float>(), b.gate.as<float>(), n); b.dot(name + "up", y.up, b.x.as<float>(), b.up.as<float>(), n);
                gpu::gelu_product(b.gate.as<float>(), b.up.as<float>(), n * c.inter, b.queue); b.finite(b.gate, std::size_t(n) * c.inter);
                b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n);
            }
            b.add(b.out, y.post_ffn_norm, n);
            b.dot(name + "ple_gate", y.ple_gate, b.residual.as<float>(), b.gate.as<float>(), n);
            hipLaunchKernelGGL(ple_product, dim3((n * c.ple_dim + 255U) / 256U), dim3(256), 0, b.queue, b.gate.as<float>(),
                               b.ple.as<float>(), n * c.ple_dim, c.ple_dim, c.layers, l);
            launched(); b.finite(b.gate, std::size_t(n) * c.ple_dim);
            b.dot(name + "ple_projection", y.ple_projection, b.gate.as<float>(), b.out.as<float>(), n); b.add(b.out, y.post_ple_norm, n, y.scalar);
            for (unsigned t = 0; t < 5; ++t) if (tap_layers[t] == l)
                checked(hipMemcpy2DAsync(b.taps.as<float>() + t * c.hidden, 5ULL * c.hidden * sizeof(float), b.residual.data(),
                                        std::size_t(c.hidden) * sizeof(float), std::size_t(c.hidden) * sizeof(float), n, hipMemcpyDeviceToDevice, b.queue));
        }
        b.norm(b.residual, w.final_norm, b.x, n, c.hidden);
        // Last-row-only head uses the same WMMA tile and K order. Qualification
        // can request every row without changing any already computed state.
        const unsigned head_rows = all_logits ? n : 1;
        const bool head = logits || all_logits;
        if (head) {
            b.logits.resize(std::size_t(head_rows) * c.vocab * sizeof(float));
            b.dot("head", w.head.data ? w.head : w.embedding, b.x.as<float>() + (all_logits ? 0 : std::size_t(n - 1) * c.hidden), b.logits.as<float>(), head_rows);
            gpu::cap_logits(b.logits.as<float>(), head_rows * c.vocab, b.queue); b.finite(b.logits, std::size_t(head_rows) * c.vocab);
        } else {
            checked(hipMemcpyAsync(b.last_row.data(), b.x.as<float>() + std::size_t(n - 1) * c.hidden, std::size_t(c.hidden) * sizeof(float),
                                   hipMemcpyDeviceToDevice, b.queue));
        }
        gpu::synchronize(b.queue);
        unsigned failure = 0; b.failure.download(&failure, sizeof(failure));
        std::vector<unsigned> maxima(b.ranges.size()); b.maximum.download(maxima.data(), maxima.size() * sizeof(unsigned));
        std::string overflow;
        for (std::size_t i = 0; i < maxima.size(); ++i) {
            const float observed = bit_cast<float>(maxima[i]);
            auto& high = b.range_maxima[b.ranges[i].tensor];
            high = !std::isfinite(observed) ? observed : std::max(high, observed);
            b.ranges[i].input_maximum = high;
            if (overflow.empty() && (!std::isfinite(observed) || observed >= 65520.0F))
                overflow = "; first overflowing input " + b.ranges[i].tensor + " abs_max=" + std::to_string(observed);
        }
        if (failure) throw std::runtime_error("prefill failed: nonfinite arithmetic, FP16 overflow or missing KV (mask " + std::to_string(failure) + ")" + overflow);
        if (head) { result.last_logits.resize(c.vocab); b.logits.download(result.last_logits.data(), c.vocab * sizeof(float), std::size_t(head_rows - 1) * c.vocab * sizeof(float)); }
        result.taps.resize(std::size_t(n) * result.tap_width); b.taps.download(result.taps.data(), result.taps.size() * sizeof(float));
        if (all_logits) { result.all_logits.resize(std::size_t(n) * c.vocab); b.logits.download(result.all_logits.data(), result.all_logits.size() * sizeof(float)); }
        if (hidden) { result.hidden.resize(std::size_t(n) * c.hidden); b.x.download(result.hidden.data(), result.hidden.size() * sizeof(float)); }
        tx->evaluated = true;
    } catch (...) {
        (void)hipStreamSynchronize(b.queue); tx->failed = true; throw;
    }
    return result;
}
namespace {
bool defer_head_enabled() {
    static const bool enabled = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_DEFER_HEAD"); return !p || std::string_view(p) != "0"; }();
    return enabled;
}
} // namespace
std::vector<float> DevicePrefill::last_logits() {
    auto& b = *impl_; const auto& w = b.w; const auto& c = w.config;
    // Same head instruction sequence on the saved final-norm row; its range is
    // recorded after the deferring chunk's ranges, whose maxima slots are free.
    const auto index = b.ranges.size();
    b.failure.zero();
    if (b.logits.size() < std::size_t(c.vocab) * sizeof(float)) b.logits.resize(std::size_t(c.vocab) * sizeof(float));
    b.dot("head", w.head.data ? w.head : w.embedding, b.last_row.as<float>(), b.logits.as<float>(), 1);
    gpu::cap_logits(b.logits.as<float>(), c.vocab, b.queue); b.finite(b.logits, c.vocab);
    gpu::synchronize(b.queue);
    unsigned failure = 0, maximum = 0;
    b.failure.download(&failure, sizeof(failure)); b.maximum.download(&maximum, sizeof(maximum), index * sizeof(unsigned));
    const float observed = bit_cast<float>(maximum);
    auto& high = b.range_maxima[b.ranges[index].tensor];
    high = !std::isfinite(observed) ? observed : std::max(high, observed);
    b.ranges[index].input_maximum = high;
    if (failure) throw std::runtime_error("prefill head failed: nonfinite arithmetic or FP16 overflow (mask " + std::to_string(failure) + ")");
    std::vector<float> out(c.vocab); b.logits.download(out.data(), out.size() * sizeof(float));
    return out;
}
Result DevicePrefill::run(gpu::DeviceStream& stream, std::span<const unsigned> tokens, DraftContext* context) {
    Prompt prompt(*this, stream, context); prompt.append(tokens); return prompt.finish();
}
Prompt::Prompt(DevicePrefill& prefill, gpu::DeviceStream& stream, DraftContext* context)
    : prefill_(prefill), stream_(stream), context_(context) { result_.frontier = stream.frontier(); }
void Prompt::flush(bool final) {
    if (pending_.empty()) return;
    if (stream_.frontier() != result_.frontier) throw std::logic_error("prefill stream changed while buffering");
    auto tx = stream_.begin();
    const auto start = tx->frontier();
    auto out = prefill_.forward(*tx, pending_, false, final || !defer_head_enabled());
    // All values have passed the failure fence before any live cache is touched.
    tx->commit(pending_.size());
    Result chunk; chunk.frontier = stream_.frontier(); chunk.tail_taps = std::move(out.taps);
    for (unsigned r = 0; r < pending_.size(); ++r) chunk.tail_positions.push_back(start + r);
    if (context_) append_tail(*context_, chunk, out.tap_width / 5);
    result_.last_logits = std::move(out.last_logits);
    result_.tail_taps.insert(result_.tail_taps.end(), chunk.tail_taps.begin(), chunk.tail_taps.end());
    result_.tail_positions.insert(result_.tail_positions.end(), chunk.tail_positions.begin(), chunk.tail_positions.end());
    const auto excess = result_.tail_positions.size() > 1024 ? result_.tail_positions.size() - 1024 : 0;
    result_.tail_positions.erase(result_.tail_positions.begin(), result_.tail_positions.begin() + std::ptrdiff_t(excess));
    result_.tail_taps.erase(result_.tail_taps.begin(), result_.tail_taps.begin() + std::ptrdiff_t(excess * out.tap_width));
    result_.frontier = stream_.frontier(); pending_.clear();
}
void Prompt::append(std::span<const unsigned> tokens) {
    if (finished_ || failed_) throw std::logic_error("prefill prompt is closed");
    try {
        if (stream_.frontier() != result_.frontier) throw std::logic_error("prefill stream changed while buffering");
        (void)plan(stream_.frontier(), pending_.size() + tokens.size());
        while (!tokens.empty()) {
            const auto capacity = chunk_rows - unsigned(stream_.frontier() % chunk_rows);
            const auto count = std::min<std::size_t>(tokens.size(), capacity - pending_.size());
            pending_.insert(pending_.end(), tokens.begin(), tokens.begin() + std::ptrdiff_t(count)); tokens = tokens.subspan(count);
            if (pending_.size() == capacity) flush(false);
        }
    } catch (...) { failed_ = true; throw; }
}
Result Prompt::finish() {
    if (finished_ || failed_) throw std::logic_error("prefill prompt is closed");
    try {
        if (stream_.frontier() != result_.frontier) throw std::logic_error("prefill stream changed while buffering");
        flush(true);
        // A prompt ending exactly on a chunk boundary deferred its head.
        if (result_.last_logits.empty() && !result_.tail_positions.empty()) result_.last_logits = prefill_.last_logits();
        finished_ = true; return std::move(result_);
    }
    catch (...) { failed_ = true; throw; }
}
} // namespace strix::gemma4::prefill
