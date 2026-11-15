// strix-cuda-qwen35-decode: qwen35 GGUF (bf16 / q8_0) on one CUDA device through the shared cuda_core, exact greedy decode.
// v0: prefill-by-rows (rows <= 16 per pass), FP32 activations, BF16 KV. Identity first; see docs for the numeric order.
// Home: specific (model layer composition: which kernel follows which); all arithmetic lives in src/cuda_core.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/rope.hpp"
#include "strix/device/cuda_enumerator.hpp"
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

namespace {

// ---- config (local reader behind the strix::qwen35::Config field names; swap for config_from_gguf when it lands) ----
struct Cfg {
    std::uint32_t layers = 0, hidden = 0, inter = 0, vocab = 0, heads = 0, kv_heads = 0, head_dim = 0, rotary = 0, interval = 4;
    std::uint32_t key_heads = 0, key_dim = 0, value_heads = 0, value_dim = 0, conv_k = 4;
    float eps = 1e-6f;
    double theta = 1e7;
    [[nodiscard]] bool is_attn(std::uint32_t l) const { return (l + 1) % interval == 0; }
};

std::uint64_t meta_u(const strix::gguf::FileInfo& f, const std::string& k) {
    const auto* e = f.find_metadata(k);
    if (!e) throw std::runtime_error("missing metadata " + k);
    if (e->unsigned_value) return *e->unsigned_value;
    if (e->signed_value) return static_cast<std::uint64_t>(*e->signed_value);
    throw std::runtime_error("metadata not integer: " + k);
}
double meta_f(const strix::gguf::FileInfo& f, const std::string& k) {
    const auto* e = f.find_metadata(k);
    if (!e || !e->floating_value) throw std::runtime_error("missing float metadata " + k);
    return *e->floating_value;
}

Cfg read_cfg(const strix::gguf::FileInfo& f) {
    const auto* arch = f.find_metadata("general.architecture");
    if (!arch || !arch->string_value || *arch->string_value != "qwen35") throw std::runtime_error("architecture is not qwen35");
    Cfg c;
    c.layers = static_cast<std::uint32_t>(meta_u(f, "qwen35.block_count"));
    c.hidden = static_cast<std::uint32_t>(meta_u(f, "qwen35.embedding_length"));
    c.inter = static_cast<std::uint32_t>(meta_u(f, "qwen35.feed_forward_length"));
    c.heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.attention.head_count"));
    c.kv_heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.attention.head_count_kv"));
    c.head_dim = static_cast<std::uint32_t>(meta_u(f, "qwen35.attention.key_length"));
    c.eps = static_cast<float>(meta_f(f, "qwen35.attention.layer_norm_rms_epsilon"));
    c.theta = meta_f(f, "qwen35.rope.freq_base");
    c.rotary = static_cast<std::uint32_t>(meta_u(f, "qwen35.rope.dimension_count"));
    c.conv_k = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.conv_kernel"));
    c.key_dim = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.state_size"));
    c.key_heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.group_count"));
    c.value_heads = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.time_step_rank"));
    const std::uint32_t inner = static_cast<std::uint32_t>(meta_u(f, "qwen35.ssm.inner_size"));
    c.value_dim = inner / c.value_heads;
    c.interval = static_cast<std::uint32_t>(meta_u(f, "qwen35.full_attention_interval"));
    for (const auto& t : f.tensors)
        if (t.name == "token_embd.weight") c.vocab = static_cast<std::uint32_t>(t.dimensions.at(1));
    if (c.vocab == 0) throw std::runtime_error("token_embd.weight missing");
    return c;
}

// ---- device weights ----
struct Weights {
    unsigned char* base = nullptr;
    std::uint64_t total = 0, trunk_bytes = 0;
    std::map<std::string, strix::gguf::TensorInfo> info;
    std::map<std::string, std::uint64_t> off;
    [[nodiscard]] const void* ptr(const std::string& n) const { return base + off.at(n); }
    [[nodiscard]] bool has(const std::string& n) const { return off.count(n) != 0; }
    [[nodiscard]] cc::DenseWeight dense(const std::string& n) const {
        const auto& t = info.at(n);
        cc::DenseWeight w;
        w.data = ptr(n);
        w.in = static_cast<std::uint32_t>(t.dimensions.at(0));
        w.out = static_cast<std::uint32_t>(t.dimensions.size() > 1 ? t.dimensions[1] : 1);
        switch (t.ggml_type) {
            case 30: w.format = cc::WeightFormat::bf16; break;
            case 8: w.format = cc::WeightFormat::q8_0; break;
            case 0: w.format = cc::WeightFormat::f32; break;
            default: throw std::runtime_error("unsupported ggml type " + std::to_string(t.ggml_type) + " for " + n);
        }
        return w;
    }
    [[nodiscard]] const float* f32(const std::string& n) const {
        if (info.at(n).ggml_type != 0) throw std::runtime_error(n + " must be F32");
        return static_cast<const float*>(ptr(n));
    }
};

void load_weights(const std::string& path, const strix::gguf::FileInfo& f, Weights& W) {
    std::uint64_t total = 0;
    for (const auto& t : f.tensors) {
        if (t.name.rfind("blk.64.", 0) == 0 || t.name.rfind("mtp.", 0) == 0) continue;  // MTP trunk not used in v0
        W.info[t.name] = t;
        W.off[t.name] = total;
        total += (t.byte_size + 255) & ~255ULL;
        if (t.name != "token_embd.weight") W.trunk_bytes += t.byte_size;
    }
    W.total = total;
    CK(cudaMalloc(reinterpret_cast<void**>(&W.base), total));
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    constexpr std::size_t kChunk = 128u << 20;
    unsigned char* pin = nullptr;
    CK(cudaMallocHost(reinterpret_cast<void**>(&pin), kChunk));
    for (const auto& [name, t] : W.info) {
        in.seekg(static_cast<std::streamoff>(t.absolute_offset));
        std::uint64_t done = 0;
        while (done < t.byte_size) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, t.byte_size - done));
            in.read(reinterpret_cast<char*>(pin), static_cast<std::streamsize>(n));
            if (!in) throw std::runtime_error("short read for " + name);
            CK(cudaMemcpy(W.base + W.off[name] + done, pin, n, cudaMemcpyHostToDevice));
            done += n;
        }
    }
    CK(cudaFreeHost(pin));
}

// ---- engine ----
struct Engine {
    Cfg c;
    Weights W;
    std::uint32_t max_pos = 0;
    // activations (max 16 rows)
    float *x = nullptr, *xn = nullptr, *qkv = nullptr, *z = nullptr, *ar = nullptr, *br = nullptr, *gout = nullptr,
          *qg = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *hid = nullptr, *logits = nullptr,
          *scratch = nullptr, *cos_t = nullptr, *sin_t = nullptr;
    std::uint32_t* ids = nullptr;
    std::uint32_t* top = nullptr;
    std::vector<float*> conv_state, delta_state;
    std::vector<__nv_bfloat16*> kc, vc;
    std::uint64_t state_bytes = 0;
    cc::GdnDims gd;
    cc::AttentionDims ad;
    cudaStream_t s = nullptr;

    template <class T>
    T* alloc(std::size_t n, bool zero = false) {
        T* p = nullptr;
        CK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
        if (zero) CK(cudaMemset(p, 0, n * sizeof(T)));
        return p;
    }

    void init(std::uint32_t maxpos) {
        max_pos = maxpos;
        const std::uint32_t R = cc::kMaxRows;
        gd = {c.key_heads, c.key_dim, c.value_heads, c.value_dim, c.conv_k};
        ad.heads = c.heads;
        ad.kv_heads = c.kv_heads;
        ad.head_dim = c.head_dim;
        ad.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.head_dim)));
        ad.max_positions = max_pos;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        x = alloc<float>(R * c.hidden);
        xn = alloc<float>(R * c.hidden);
        qkv = alloc<float>(R * gd.conv_channels());
        z = alloc<float>(R * gd.v_width());
        ar = alloc<float>(R * gd.value_heads);
        br = alloc<float>(R * gd.value_heads);
        gout = alloc<float>(R * gd.v_width());
        qg = alloc<float>(R * c.heads * c.head_dim * 2);
        q = alloc<float>(R * c.heads * c.head_dim);
        k = alloc<float>(R * c.kv_heads * c.head_dim);
        v = alloc<float>(R * c.kv_heads * c.head_dim);
        ao = alloc<float>(R * c.heads * c.head_dim);
        hid = alloc<float>(R * c.inter);
        logits = alloc<float>(static_cast<std::size_t>(R) * c.vocab);
        scratch = alloc<float>(cc::attention_scratch_floats(ad, R));
        ids = alloc<std::uint32_t>(R);
        top = alloc<std::uint32_t>(R);
        std::vector<float> ct, st;
        cc::rope_table_host(max_pos, c.rotary, c.theta, ct, st);
        cos_t = alloc<float>(ct.size());
        sin_t = alloc<float>(st.size());
        CK(cudaMemcpy(cos_t, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sin_t, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (c.is_attn(l)) {
                const std::size_t n = static_cast<std::size_t>(c.kv_heads) * max_pos * c.head_dim;
                kc.push_back(alloc<__nv_bfloat16>(n, true));
                vc.push_back(alloc<__nv_bfloat16>(n, true));
                conv_state.push_back(nullptr);
                delta_state.push_back(nullptr);
            } else {
                kc.push_back(nullptr);
                vc.push_back(nullptr);
                conv_state.push_back(alloc<float>(static_cast<std::size_t>(gd.conv_channels()) * 3, true));
                delta_state.push_back(alloc<float>(static_cast<std::size_t>(gd.value_heads) * gd.key_dim * gd.value_dim, true));
                state_bytes += static_cast<std::uint64_t>(gd.value_heads) * gd.key_dim * gd.value_dim * 4 * 2;
            }
        }
    }

    void reset_state() {
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (c.is_attn(l)) continue;
            CK(cudaMemset(conv_state[l], 0, static_cast<std::size_t>(gd.conv_channels()) * 3 * 4));
            CK(cudaMemset(delta_state[l], 0, static_cast<std::size_t>(gd.value_heads) * gd.key_dim * gd.value_dim * 4));
        }
    }

    std::string nm(std::uint32_t l, const char* suffix) const { return "blk." + std::to_string(l) + "." + suffix; }

    // Runs `rows` tokens at positions pos0.. . If logits_rows: lm_head over all rows into `logits` ([rows][vocab]);
    // else lm_head on the last row only (into logits row 0).
    void forward(const std::uint32_t* host_ids, std::uint32_t rows, std::uint32_t pos0, bool all_rows_logits) {
        const std::uint32_t H = c.hidden;
        CK(cudaMemcpyAsync(ids, host_ids, rows * 4, cudaMemcpyHostToDevice, s));
        CK(cc::embedding_rows(W.dense("token_embd.weight"), ids, x, rows, s));
        const cc::GroupedView hv{rows, 1, H, H, H};
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            CK(cc::rmsnorm(x, hv, W.f32(nm(l, "attn_norm.weight")), xn, hv, c.eps, s));
            if (!c.is_attn(l)) {
                CK(cc::dense_rows(W.dense(nm(l, "attn_qkv.weight")), xn, qkv, rows, nullptr, s));
                CK(cc::dense_rows(W.dense(nm(l, "attn_gate.weight")), xn, z, rows, nullptr, s));
                CK(cc::dense_rows(W.dense(nm(l, "ssm_beta.weight")), xn, br, rows, nullptr, s));
                CK(cc::dense_rows(W.dense(nm(l, "ssm_alpha.weight")), xn, ar, rows, nullptr, s));
                cc::GdnLayer gl{W.f32(nm(l, "ssm_conv1d.weight")), W.f32(nm(l, "ssm_a")), W.f32(nm(l, "ssm_dt.bias")),
                                W.f32(nm(l, "ssm_norm.weight"))};
                CK(cc::gdn_decode(gd, gl, {conv_state[l], delta_state[l]}, qkv, z, ar, br, gout, rows, c.eps, s));
                CK(cc::dense_rows(W.dense(nm(l, "ssm_out.weight")), gout, x, rows, x, s));
            } else {
                const std::uint32_t D = c.head_dim, qgw = c.heads * D * 2;
                CK(cc::dense_rows(W.dense(nm(l, "attn_q.weight")), xn, qg, rows, nullptr, s));
                CK(cc::dense_rows(W.dense(nm(l, "attn_k.weight")), xn, k, rows, nullptr, s));
                CK(cc::dense_rows(W.dense(nm(l, "attn_v.weight")), xn, v, rows, nullptr, s));
                const cc::GroupedView qsrc{rows, c.heads, D, qgw, 2 * D}, qdst{rows, c.heads, D, c.heads * D, D};
                const cc::GroupedView kv{rows, c.kv_heads, D, c.kv_heads * D, D};
                CK(cc::rmsnorm(qg, qsrc, W.f32(nm(l, "attn_q_norm.weight")), q, qdst, c.eps, s));
                CK(cc::rmsnorm(k, kv, W.f32(nm(l, "attn_k_norm.weight")), k, kv, c.eps, s));
                CK(cc::rope_apply(q, qdst, c.rotary, cos_t, sin_t, pos0, s));
                CK(cc::rope_apply(k, kv, c.rotary, cos_t, sin_t, pos0, s));
                CK(cc::kv_store(ad, k, v, kc[l], vc[l], pos0, rows, s));
                CK(cc::attention_decode(ad, q, qdst, qg + D, qsrc, kc[l], vc[l], pos0, rows, ao, scratch, s));
                CK(cc::dense_rows(W.dense(nm(l, "attn_output.weight")), ao, x, rows, x, s));
            }
            CK(cc::rmsnorm(x, hv, W.f32(nm(l, "post_attention_norm.weight")), xn, hv, c.eps, s));
            CK(cc::dense_gate_up_silu_rows(W.dense(nm(l, "ffn_gate.weight")), W.dense(nm(l, "ffn_up.weight")), xn, hid, rows, s));
            CK(cc::dense_rows(W.dense(nm(l, "ffn_down.weight")), hid, x, rows, x, s));
        }
        CK(cc::rmsnorm(x, hv, W.f32("output_norm.weight"), xn, hv, c.eps, s));
        const auto head = W.has("output.weight") ? W.dense("output.weight") : W.dense("token_embd.weight");
        if (all_rows_logits) {
            CK(cc::dense_rows(head, xn, logits, rows, nullptr, s));
            CK(cc::argmax_rows(logits, c.vocab, rows, top, s));
        } else {
            CK(cc::dense_rows(head, xn + static_cast<std::size_t>(rows - 1) * H, logits, 1, nullptr, s));
            CK(cc::argmax_rows(logits, c.vocab, 1, top, s));
        }
    }
};

// ---- clock sampling (nvidia-smi, device 0) ----
struct ClockSampler {
    std::atomic<bool> stop{false};
    std::vector<int> mhz;
    std::thread th;
    void start() {
        th = std::thread([this] {
            while (!stop.load()) {
                FILE* p = _popen("nvidia-smi --query-gpu=clocks.sm -i 0 --format=csv,noheader,nounits", "r");
                if (p) {
                    char b[64] = {};
                    if (std::fgets(b, sizeof b, p)) mhz.push_back(std::atoi(b));
                    _pclose(p);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    void end() {
        stop = true;
        if (th.joinable()) th.join();
    }
};

}  // namespace

int main(int argc, char** argv) {
    strix::device::bind_cuda_main(argc, argv);
    try {
        std::string model, dump, ids_arg;
        std::uint32_t n_gen = 32, rows_cap = 16, max_pos = 4096;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--model") model = next();
            else if (a == "--prompt-ids") ids_arg = next();
            else if (a == "--n-gen") n_gen = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--rows") rows_cap = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--max-pos") max_pos = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--dump-logits") dump = next();
            else throw std::runtime_error("unknown argument " + a);
        }
        if (model.empty() || ids_arg.empty() || rows_cap == 0 || rows_cap > 16)
            throw std::runtime_error("usage: --model F.gguf --prompt-ids 1,2,3 [--n-gen N] [--rows 1..16] [--max-pos N] [--dump-logits F]");
        std::vector<std::uint32_t> prompt;
        {
            std::stringstream ss(ids_arg);
            std::string t;
            while (std::getline(ss, t, ',')) prompt.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        }
        if (prompt.size() + n_gen > max_pos) max_pos = static_cast<std::uint32_t>(prompt.size() + n_gen);

        const auto info = strix::gguf::read_file(model);
        Engine E;
        E.c = read_cfg(info);
        std::fprintf(stderr, "qwen35: layers %u hidden %u inter %u vocab %u heads %u/%u hd %u gdn %u/%u x %u\n", E.c.layers,
                     E.c.hidden, E.c.inter, E.c.vocab, E.c.heads, E.c.kv_heads, E.c.head_dim, E.c.key_heads, E.c.value_heads,
                     E.c.value_dim);
        const auto t0 = std::chrono::steady_clock::now();
        load_weights(model, info, E.W);
        E.init(max_pos);
        std::fprintf(stderr, "loaded %.2f GB in %.1f s\n", E.W.total / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

        // prompt by rows
        std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
        if (!dump.empty() && !df) throw std::runtime_error("cannot open " + dump);
        std::vector<float> host_logits(static_cast<std::size_t>(16) * E.c.vocab);
        std::vector<std::uint32_t> top1;
        std::uint32_t next_tok = 0, pos = 0;
        for (std::size_t p = 0; p < prompt.size(); p += rows_cap) {
            const std::uint32_t rows = static_cast<std::uint32_t>(std::min<std::size_t>(rows_cap, prompt.size() - p));
            const bool last = p + rows == prompt.size();
            E.forward(prompt.data() + p, rows, pos, df != nullptr);
            if (df) {
                CK(cudaMemcpyAsync(host_logits.data(), E.logits, static_cast<std::size_t>(rows) * E.c.vocab * 4,
                                   cudaMemcpyDeviceToHost, E.s));
                std::uint32_t tp[16];
                CK(cudaMemcpyAsync(tp, E.top, rows * 4, cudaMemcpyDeviceToHost, E.s));
                CK(cudaStreamSynchronize(E.s));
                std::fwrite(host_logits.data(), 4, static_cast<std::size_t>(rows) * E.c.vocab, df);
                for (std::uint32_t r = 0; r < rows; ++r) top1.push_back(tp[r]);
                next_tok = tp[rows - 1];
            } else if (last) {
                CK(cudaMemcpyAsync(&next_tok, E.top, 4, cudaMemcpyDeviceToHost, E.s));
                CK(cudaStreamSynchronize(E.s));
            }
            pos += rows;
        }
        CK(cudaStreamSynchronize(E.s));
        if (df) {
            std::fclose(df);
            std::printf("prompt top1:");
            for (auto t : top1) std::printf(" %u", t);
            std::printf("\n");
        }
        // greedy decode, 1 row per step
        std::vector<std::uint32_t> gen;
        std::vector<double> step_ms;
        ClockSampler clk;
        clk.start();
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (std::uint32_t i = 0; i < n_gen; ++i) {
            gen.push_back(next_tok);
            if (i + 1 == n_gen) break;
            CK(cudaEventRecord(e0, E.s));
            E.forward(&next_tok, 1, pos, false);
            CK(cudaEventRecord(e1, E.s));
            CK(cudaMemcpyAsync(&next_tok, E.top, 4, cudaMemcpyDeviceToHost, E.s));
            CK(cudaStreamSynchronize(E.s));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            step_ms.push_back(ms);
            ++pos;
        }
        clk.end();
        std::printf("generated:");
        for (auto t : gen) std::printf(" %u", t);
        std::printf("\n");
        if (!step_ms.empty()) {
            std::vector<double> s2(step_ms.begin() + (step_ms.size() > 2 ? 1 : 0), step_ms.end());
            double sum = 0;
            for (double m : s2) sum += m;
            const double mean = sum / s2.size();
            std::sort(s2.begin(), s2.end());
            const double bytes = static_cast<double>(E.W.trunk_bytes) + static_cast<double>(E.state_bytes);
            std::printf("decode: %zu steps, mean %.3f ms/token (median %.3f), %.2f tok/s, %.1f GB/s achieved (%.3f GB/pass) vs 1792 = %.1f%%\n",
                        s2.size(), mean, s2[s2.size() / 2], 1000.0 / mean, bytes / (mean * 1e-3) / 1e9, bytes / 1e9,
                        100.0 * bytes / (mean * 1e-3) / 1e9 / 1792.0);
        }
        if (!clk.mhz.empty()) {
            int lo = 1 << 30, hi = 0;
            double sm = 0;
            for (int m : clk.mhz) {
                lo = std::min(lo, m);
                hi = std::max(hi, m);
                sm += m;
            }
            std::printf("sm clock MHz: min %d avg %.0f max %d (%zu samples)\n", lo, sm / clk.mhz.size(), hi, clk.mhz.size());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-qwen35-decode: %s\n", e.what());
        return 1;
    }
}
