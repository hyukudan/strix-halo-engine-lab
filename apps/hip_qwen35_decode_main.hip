// strix-hip-qwen35-decode: Qwen3.8-27B (qwen35 dense) decode on one HIP device, exact-identity v0.
// Prompt goes through the decode rows (windows of up to --rows tokens), then greedy decode.
// Binds shared cores only: core/dense_linear (gemv_rows + weight_decode), attn_core/dense_decode,
// ops/norm/norm_device, hip_gdn_decode_step_dims (R1 GDN launcher path, swish gated norm).
// Home: specific = app glue (the layer loop); every kernel is shared.
#include <hip/hip_runtime.h>

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

#include "strix/gguf_reader.hpp"
#include "strix/hip_gdn_dims.hpp"
#include "strix/qwen35/config.hpp"
#include "../src/attn_core/dense_decode.hpp"
#include "../src/core/dense_linear.hpp"
#include "../src/ops/norm/norm_device.hpp"

namespace {
using namespace strix;
namespace dl = core::dense_linear;
namespace wd = core::weight_decode;

#define HK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); } while (0)

// RMSNorm through the shared ops/norm kernel: F32 stored gains (1+w); reciprocal = 1/sqrt (as the attention prep).
struct NormTraits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    static constexpr auto reduction_order = ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    template <unsigned L> __device__ __forceinline__ static float reduce(float* s, float part) {
        const unsigned l = threadIdx.x;
        s[l] = part;
        __syncthreads();
        for (unsigned h = L / 2; h > 0; h >>= 1U) {
            if (l < h) s[l] = s[l] + s[l + h];
            __syncthreads();
        }
        const float r = s[0];
        __syncthreads();
        return r;
    }
    __device__ __forceinline__ static float inv_sqrt(float x) { return 1.0F / sqrtf(x); }
    __device__ __forceinline__ static float load(const float* w, int, unsigned i) { return w[i]; }
};

__global__ void embed_kernel(const std::uint8_t* table, unsigned fmt, std::size_t row_bytes, unsigned hidden, const int* ids, float* out) {
    const unsigned r = blockIdx.x;
    const std::uint8_t* row = table + std::size_t(ids[r]) * row_bytes;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v;
        if (fmt == 30U) {
            std::uint16_t h;
            __builtin_memcpy(&h, row + 2ULL * i, 2);
            v = wd::bf16_to_float(h);
        } else if (fmt == 0U) {
            __builtin_memcpy(&v, row + 4ULL * i, 4);
        } else {
            const std::uint8_t* b = row + std::size_t(i / 32U) * 34U;
            std::uint16_t h;
            __builtin_memcpy(&h, b, 2);
            const int q = static_cast<signed char>(b[2 + i % 32U]);
            v = float(__builtin_bit_cast(_Float16, h)) * float(q);
        }
        out[std::size_t(r) * hidden + i] = v;
    }
}
__global__ void add_kernel(float* x, const float* y, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + y[i];
}
// Lowest index wins ties; fixed tree.
__global__ void argmax_kernel(const float* logits, unsigned vocab, int* out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* row = logits + std::size_t(blockIdx.x) * vocab;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (unsigned i = threadIdx.x; i < vocab; i += 1024U)
        if (row[i] > best) { best = row[i]; bidx = int(i); }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = bidx;
    __syncthreads();
    for (unsigned s = 512U; s > 0U; s >>= 1U) {
        if (threadIdx.x < s) {
            const float v = bv[threadIdx.x + s];
            const int i = bi[threadIdx.x + s];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && i < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = i; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = bi[0];
}

struct Tensor {
    std::uint8_t* data = nullptr;
    unsigned inputs = 0, outputs = 0, fmt = 0;  // fmt: ggml type 0 f32, 1 f16, 8 q8_0, 30 bf16
    std::size_t bytes = 0, row_bytes = 0;
};
constexpr unsigned kMaxRows = 16U;

class Loader {
public:
    explicit Loader(const std::string& path) : info_(gguf::read_file(path)), file_(path, std::ios::binary) {
        if (!file_) throw std::runtime_error("cannot open " + path);
        for (const auto& t : info_.tensors) by_name_[t.name] = &t;
    }
    const gguf::FileInfo& info() const { return info_; }
    const gguf::TensorInfo* find(const std::string& name) const {
        for (const char* suffix : {"", ".weight", ".bias"}) {
            auto it = by_name_.find(name + suffix);
            if (it != by_name_.end()) return it->second;
        }
        return nullptr;
    }
    Tensor load(const std::string& name) {
        const auto* t = find(name);
        if (!t) throw std::runtime_error("missing tensor " + name);
        Tensor r;
        r.fmt = t->ggml_type;
        r.bytes = t->byte_size;
        r.inputs = unsigned(t->dimensions.at(0));
        r.outputs = t->dimensions.size() > 1 ? unsigned(t->dimensions[1]) : 1U;
        const auto g = gguf::tensor_type_geometry(t->ggml_type);
        if (!g) throw std::runtime_error("unsupported ggml type for " + name);
        r.row_bytes = std::size_t(r.inputs / g->block_elements) * g->block_bytes;
        HK(hipMalloc(&r.data, r.bytes));
        file_.seekg(std::streamoff(t->absolute_offset));
        std::vector<char> stage(std::min<std::size_t>(r.bytes, 64ULL << 20));
        for (std::size_t done = 0; done < r.bytes;) {
            const auto n = std::min(stage.size(), r.bytes - done);
            file_.read(stage.data(), std::streamsize(n));
            if (!file_) throw std::runtime_error("short read " + name);
            HK(hipMemcpy(r.data + done, stage.data(), n, hipMemcpyHostToDevice));
            done += n;
        }
        total_ += r.bytes;
        return r;
    }
    std::size_t total() const { return total_; }

private:
    gguf::FileInfo info_;
    std::ifstream file_;
    std::map<std::string, const gguf::TensorInfo*> by_name_;
    std::size_t total_ = 0;
};

struct Layer {
    Tensor attn_norm, post_norm, gate, up, down;
    Tensor qkv, z, beta, alpha, conv, dt, a, snorm, out;  // gdn
    Tensor q, k, v, o, qnorm, knorm;                      // attention
    float* gdn_state = nullptr;
    std::uint16_t* conv_state = nullptr;
    std::uint16_t *k_cache = nullptr, *v_cache = nullptr;
};

dl::View view(const Tensor& t) { return {t.data, t.inputs, t.outputs, t.row_bytes}; }

template <unsigned Rows, wd::Format F, dl::Epilogue E>
void launch_dense(const Tensor& w, const Tensor& up, const float* x, float* out, hipStream_t s) {
    constexpr unsigned T = 2U;
    dl::kernel<Rows, T, F, E><<<(w.outputs + 4U * T - 1U) / (4U * T), 128, 0, s>>>(view(w), view(up), x, out);
}
// Rows are padded to the next of 1/2/4/8/16 over kMaxRows-sized scratch; every row is independent.
template <wd::Format F, dl::Epilogue E>
void dispatch_rows(unsigned rows, const Tensor& w, const Tensor& up, const float* x, float* out, hipStream_t s) {
    if (rows <= 1U) launch_dense<1, F, E>(w, up, x, out, s);
    else if (rows <= 2U) launch_dense<2, F, E>(w, up, x, out, s);
    else if (rows <= 4U) launch_dense<4, F, E>(w, up, x, out, s);
    else if (rows <= 8U) launch_dense<8, F, E>(w, up, x, out, s);
    else launch_dense<16, F, E>(w, up, x, out, s);
}
void linear(const Tensor& w, const float* x, float* out, unsigned rows, hipStream_t s) {
    if (w.fmt == 30U) dispatch_rows<wd::Format::bf16, dl::Epilogue::store>(rows, w, w, x, out, s);
    else if (w.fmt == 8U) dispatch_rows<wd::Format::q8_0, dl::Epilogue::store>(rows, w, w, x, out, s);
    else throw std::runtime_error("dense projection needs bf16 or q8_0 weights");
}
void glu(const Tensor& g, const Tensor& u, const float* x, float* out, unsigned rows, hipStream_t s) {
    if (g.fmt != u.fmt || g.outputs != u.outputs) throw std::runtime_error("gate/up mismatch");
    if (g.fmt == 30U) dispatch_rows<wd::Format::bf16, dl::Epilogue::silu_mul>(rows, g, u, x, out, s);
    else if (g.fmt == 8U) dispatch_rows<wd::Format::q8_0, dl::Epilogue::silu_mul>(rows, g, u, x, out, s);
    else throw std::runtime_error("gate/up need bf16 or q8_0");
}
void rmsnorm(const float* x, const Tensor& gain, unsigned n, float eps, float* y, unsigned rows, hipStream_t s) {
    ops::norm::rmsnorm_kernel<NormTraits, 256><<<rows, 256, 0, s>>>(x, reinterpret_cast<const float*>(gain.data), 0, n, eps, y);
}

std::string sysfs_clock() {
    for (int c = 0; c < 4; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/pp_dpm_sclk");
        std::string line, best;
        while (std::getline(f, line))
            if (line.find('*') != std::string::npos) best = line;
        if (!best.empty()) return best;
    }
    return "0: 0Mhz *";
}
int sysfs_busy() {
    for (int c = 0; c < 4; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/gpu_busy_percent");
        int v;
        if (f >> v) return v;
    }
    return -1;
}
std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> r;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty()) {
            const auto id = core::parse_u32(t);
            if (id > 0x7FFFFFFFU) throw std::invalid_argument("token ID exceeds signed index range");
            r.push_back(static_cast<int>(id));
        }
    return r;
}
}  // namespace

int main(int argc, char** argv) try {
    std::string model, dump, ids_arg = "1";
    unsigned gen = 16, rows = 1, maxpos = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--prompt-ids") ids_arg = next();
        else if (a == "--gen") gen = core::parse_u32(next());
        else if (a == "--rows") rows = core::parse_u32(next());
        else if (a == "--dump-logits") dump = next();
        else if (a == "--max-pos") maxpos = core::parse_u32(next());
        else throw std::invalid_argument("unknown flag " + a);
    }
    if (model.empty() || rows < 1U || rows > kMaxRows)
        throw std::invalid_argument("usage: --model GGUF [--prompt-ids 1,2,3] [--gen N] [--rows 1..16] [--dump-logits FILE] [--max-pos N]");
    const auto prompt = parse_ids(ids_arg);
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    core::require_capacity(prompt.size(), gen, 0xFFFFFFFFULL);
    const auto required = prompt.size() + gen;
    if (!maxpos) maxpos = static_cast<unsigned>(required);
    core::require_capacity(0, required, maxpos);

    Loader loader(model);
    const auto cfg = qwen35::validate(loader.info(), maxpos);
    qwen35::validate_backend(cfg, maxpos);
    core::validate_token_ids(prompt.data(), prompt.size(), cfg.vocab);
    const unsigned H = cfg.hidden;
    const unsigned V = unsigned(loader.find("token_embd")->dimensions.at(1));
    const unsigned D = cfg.head_dim, QW = cfg.heads * D, KW = cfg.kv_heads * D, I = cfg.intermediate;
    const unsigned GV = cfg.gdn_v_width(), GC = cfg.gdn_conv_channels(), GH = cfg.gdn_value_heads;
    if (D != 256U) throw std::runtime_error("only head_dim 256 is instantiated");
    hipStream_t stream;
    HK(hipStreamCreate(&stream));

    const auto t_load0 = std::chrono::steady_clock::now();
    Tensor embd = loader.load("token_embd"), onorm = loader.load("output_norm"), head = loader.load("output");
    std::vector<Layer> L(cfg.layers);
    std::size_t gdn_state_bytes = 0, kv_bytes_per_pos = 0;
    for (unsigned l = 0; l < cfg.layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        auto& y = L[l];
        y.attn_norm = loader.load(b + "attn_norm");
        y.post_norm = loader.load(b + "post_attention_norm");
        y.gate = loader.load(b + "ffn_gate");
        y.up = loader.load(b + "ffn_up");
        y.down = loader.load(b + "ffn_down");
        if (cfg.kind(l) == qwen35::LayerKind::gated_delta_net) {
            y.qkv = loader.load(b + "attn_qkv");
            y.z = loader.load(b + "attn_gate");
            y.beta = loader.load(b + "ssm_beta");
            y.alpha = loader.load(b + "ssm_alpha");
            y.conv = loader.load(b + "ssm_conv1d");
            y.dt = loader.load(b + "ssm_dt");
            y.a = loader.load(b + "ssm_a");
            y.snorm = loader.load(b + "ssm_norm");
            y.out = loader.load(b + "ssm_out");
            for (const Tensor* t : {&y.conv, &y.dt, &y.a, &y.snorm})
                if (t->fmt != 0U) throw std::runtime_error("GDN small tensors must be F32");
            const std::size_t sb = std::size_t(GH) * cfg.gdn_key_dim * cfg.gdn_value_dim * 4U;
            const std::size_t cb = std::size_t(GC) * cfg.gdn_conv_kernel * 2U;
            HK(hipMalloc(&y.gdn_state, sb));
            HK(hipMemset(y.gdn_state, 0, sb));
            HK(hipMalloc(&y.conv_state, cb));
            HK(hipMemset(y.conv_state, 0, cb));
            gdn_state_bytes += sb;
        } else {
            y.q = loader.load(b + "attn_q");
            y.k = loader.load(b + "attn_k");
            y.v = loader.load(b + "attn_v");
            y.o = loader.load(b + "attn_output");
            y.qnorm = loader.load(b + "attn_q_norm");
            y.knorm = loader.load(b + "attn_k_norm");
            const std::size_t cb = std::size_t(maxpos) * KW * 2U;
            HK(hipMalloc(&y.k_cache, cb));
            HK(hipMalloc(&y.v_cache, cb));
            HK(hipMemset(y.k_cache, 0, cb));
            HK(hipMemset(y.v_cache, 0, cb));
            kv_bytes_per_pos += 2U * KW * 2U;
        }
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    // bytes read per decode pass: all weights but the embedding table (one row), state read+write, KV read.
    const std::size_t pass_weight_bytes = loader.total() - embd.bytes;
    std::printf("loaded %.2f GiB in %.1f s (layers %u, hidden %u, vocab %u)\n", double(loader.total()) / double(1ULL << 30), load_s, cfg.layers, H, V);

    float *x, *xn, *qkv, *z, *beta, *gate, *core, *proj, *q_in, *kk, *vv, *qn, *ctx, *gu, *logits;
    auto fm = [&](float** p, std::size_t n) {
        HK(hipMalloc(p, n * kMaxRows * 4U));
        HK(hipMemset(*p, 0, n * kMaxRows * 4U));
    };
    fm(&x, H); fm(&xn, H); fm(&qkv, GC); fm(&z, GV); fm(&beta, GH); fm(&gate, GH); fm(&core, GV); fm(&proj, H);
    fm(&q_in, 2U * QW); fm(&kk, KW); fm(&vv, KW); fm(&qn, QW); fm(&ctx, QW); fm(&gu, I); fm(&logits, V);
    int *d_ids, *d_arg;
    HK(hipMalloc(&d_ids, kMaxRows * 4U));
    HK(hipMalloc(&d_arg, kMaxRows * 4U));
    std::vector<int> h_arg(kMaxRows);

    std::vector<float> dumped;
    unsigned pos = 0;
    auto forward = [&](unsigned n, const int* ids) {
        if (!n || n > kMaxRows) throw std::invalid_argument("forward rows outside 1..16");
        core::require_capacity(pos, n, maxpos);
        core::validate_token_ids(ids, n, cfg.vocab);
        HK(hipMemcpyAsync(d_ids, ids, n * 4U, hipMemcpyHostToDevice, stream));
        embed_kernel<<<n, 256, 0, stream>>>(embd.data, embd.fmt, embd.row_bytes, H, d_ids, x);
        const unsigned nb = (n * H + 255U) / 256U;
        for (unsigned l = 0; l < cfg.layers; ++l) {
            auto& y = L[l];
            rmsnorm(x, y.attn_norm, H, cfg.rms_eps, xn, n, stream);
            if (cfg.kind(l) == qwen35::LayerKind::gated_delta_net) {
                linear(y.qkv, xn, qkv, n, stream);
                linear(y.z, xn, z, n, stream);
                linear(y.beta, xn, beta, n, stream);
                linear(y.alpha, xn, gate, n, stream);
                HipGdnDimsStep g{};
                g.qkv = qkv; g.beta = beta; g.gate = gate; g.core = core; g.matrix = y.gdn_state; g.z = z;
                g.conv_state = y.conv_state; g.conv = y.conv.data; g.dt = y.dt.data; g.a = y.a.data; g.norm = y.snorm.data;
                g.small_type = 0; g.key_heads = cfg.gdn_key_heads; g.value_heads = GH; g.head_dim = cfg.gdn_key_dim;
                g.kernel = cfg.gdn_conv_kernel; g.rows = n; g.qk_epsilon = 1e-6F; g.rms_epsilon = cfg.rms_eps;
                g.norm_kind = HipGdnGatedNorm::swish;
                hip_gdn_decode_step_dims(g, stream);
                linear(y.out, core, proj, n, stream);
            } else {
                linear(y.q, xn, q_in, n, stream);
                linear(y.k, xn, kk, n, stream);
                linear(y.v, xn, vv, n, stream);
                attn_core::dense_decode::Params p{n, cfg.heads, cfg.kv_heads, 2U * D, cfg.rotary_dim, pos, cfg.rms_eps,
                                                  1.0F / std::sqrt(float(D)), cfg.rope_theta, true};
                attn_core::dense_decode::launch<256>(p, q_in, kk, vv, reinterpret_cast<const float*>(y.qnorm.data),
                                                     reinterpret_cast<const float*>(y.knorm.data), qn, y.k_cache, y.v_cache, ctx, stream);
                linear(y.o, ctx, proj, n, stream);
            }
            add_kernel<<<nb, 256, 0, stream>>>(x, proj, n * H);
            rmsnorm(x, y.post_norm, H, cfg.rms_eps, xn, n, stream);
            glu(y.gate, y.up, xn, gu, n, stream);
            linear(y.down, gu, proj, n, stream);
            add_kernel<<<nb, 256, 0, stream>>>(x, proj, n * H);
        }
        rmsnorm(x, onorm, H, cfg.rms_eps, xn, n, stream);
        linear(head, xn, logits, n, stream);
        argmax_kernel<<<n, 1024, 0, stream>>>(logits, V, d_arg);
        HK(hipMemcpyAsync(h_arg.data(), d_arg, n * 4U, hipMemcpyDeviceToHost, stream));
        if (!dump.empty()) {
            const auto at = dumped.size();
            dumped.resize(at + std::size_t(n) * V);
            HK(hipMemcpyAsync(dumped.data() + at, logits, std::size_t(n) * V * 4U, hipMemcpyDeviceToHost, stream));
        }
        HK(hipStreamSynchronize(stream));
        HK(hipGetLastError());
        pos += n;
    };

    const auto t_pf0 = std::chrono::steady_clock::now();
    unsigned last_n = 0;
    for (std::size_t at = 0; at < prompt.size();) {
        last_n = unsigned(std::min<std::size_t>(rows, prompt.size() - at));
        forward(last_n, prompt.data() + at);
        at += last_n;
    }
    const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
    std::printf("prompt %zu tokens: %.1f ms\n", prompt.size(), pf_ms);
    std::vector<int> generated{h_arg[last_n - 1U]};

    std::atomic<bool> run{true};
    std::atomic<long> clk_sum{0}, clk_n{0}, busy_sum{0};
    std::string last_clk;
    std::thread sampler([&] {
        while (run) {
            last_clk = sysfs_clock();
            clk_sum += std::atol(last_clk.c_str() + last_clk.find(' ') + 1);
            ++clk_n;
            busy_sum += std::max(0, sysfs_busy());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned s = 1; s < gen; ++s) {
        const int id = generated.back();
        forward(1, &id);
        generated.push_back(h_arg[0]);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    run = false;
    sampler.join();
    std::printf("generated:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");
    if (gen > 1U) {
        const double per = ms / double(gen - 1U);
        const double bytes = double(pass_weight_bytes) + double(gdn_state_bytes) * 2.0 + double(kv_bytes_per_pos) * double(pos);
        std::printf("decode: %.2f ms/token, %.2f tok/s, %.1f GB/s (%.0f %% of 243), bytes/pass %.2f GB\n", per, 1000.0 / per,
                    bytes / per / 1e6, bytes / per / 1e6 / 243.0 * 100.0, bytes / 1e9);
        std::printf("gpu: sclk avg %ld MHz (last: %s), busy avg %ld %%\n", clk_n ? clk_sum / clk_n : 0L, last_clk.c_str(),
                    clk_n ? busy_sum / clk_n : 0L);
    }
    if (!dump.empty()) {
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(dumped.data()), std::streamsize(dumped.size() * 4U));
        std::printf("dumped %zu positions x %u to %s\n", dumped.size() / V, V, dump.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
