#pragma once
// Shared pieces of the minimax-m3 layer gates (G3 one node, G4 two-rank TP): a minimal JSON reader, the safetensors
// pack (mmap), strix-trellis EXL3 linears (whole or H128-aligned slices), and small FP32 kernels. Included by one
// translation unit per gate program.
#include "strix/hip_minimax_msa.hpp"

#include <strix_trellis.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace msa = strix::minimax::msa;
namespace mh = strix::minimax::msa::hip;

[[maybe_unused]] constexpr unsigned kHidden = 6144, kQ = 64 * 128, kKV = 4 * 128, kIdxQ = 4 * 128, kDense = 12288, kExpertI = 3072;
[[maybe_unused]] constexpr unsigned kExperts = 128, kTopE = 4;
[[maybe_unused]] constexpr float kEps = 1e-6F, kAlpha = 1.702F, kLimit = 7.0F, kRouteScale = 2.0F;
[[maybe_unused]] const std::string kP = "language_model.model.";

[[maybe_unused]] void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
[[maybe_unused]] void lib_ok(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("trellis " + what + ": " + strix_status_string(s));
}

// ---------------------------------------------------------------- minimal JSON (safetensors headers, index)
struct Json;
using JsonObj = std::map<std::string, Json>;
using JsonArr = std::vector<Json>;
struct Json {
    std::variant<std::nullptr_t, double, std::string, JsonArr, JsonObj> v;
    const JsonObj& obj() const { return std::get<JsonObj>(v); }
    const JsonArr& arr() const { return std::get<JsonArr>(v); }
    const std::string& str() const { return std::get<std::string>(v); }
    double num() const { return std::get<double>(v); }
};
struct JsonParser {
    const char* p;
    const char* end;
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    Json parse() {
        ws();
        if (p >= end) throw std::runtime_error("json: eof");
        if (*p == '{') {
            ++p;
            JsonObj o;
            ws();
            if (*p == '}') { ++p; return {o}; }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                if (*p++ != ':') throw std::runtime_error("json: ':'");
                o.emplace(std::move(k), parse());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != '}') throw std::runtime_error("json: '}'");
                return {o};
            }
        }
        if (*p == '[') {
            ++p;
            JsonArr a;
            ws();
            if (*p == ']') { ++p; return {a}; }
            for (;;) {
                a.push_back(parse());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != ']') throw std::runtime_error("json: ']'");
                return {a};
            }
        }
        if (*p == '"') return {string()};
        if (std::strncmp(p, "null", 4) == 0) { p += 4; return {nullptr}; }
        if (std::strncmp(p, "true", 4) == 0) { p += 4; return {1.0}; }
        if (std::strncmp(p, "false", 5) == 0) { p += 5; return {0.0}; }
        char* e = nullptr;
        const double d = std::strtod(p, &e);
        if (e == p) throw std::runtime_error("json: value");
        p = e;
        return {d};
    }
    std::string string() {
        if (*p++ != '"') throw std::runtime_error("json: string");
        std::string s;
        while (*p != '"') {
            if (*p == '\\') { ++p; s += *p == 'n' ? '\n' : *p; ++p; continue; }
            s += *p++;
        }
        ++p;
        return s;
    }
};

// ---------------------------------------------------------------- safetensors pack (mmap)
struct Tensor {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    const std::uint8_t* data;
    std::uint64_t bytes;
};
class Pack {
public:
    explicit Pack(std::string dir) : dir_(std::move(dir)) {
        const std::string idx = read_file(dir_ + "/model.safetensors.index.json");
        JsonParser jp{idx.data(), idx.data() + idx.size()};
        const Json j = jp.parse();
        for (const auto& [k, v] : j.obj().at("weight_map").obj()) map_[k] = v.str();
    }
    ~Pack() {
        for (auto& [k, f] : files_) munmap(const_cast<std::uint8_t*>(f.base), f.size);
    }
    Tensor get(const std::string& name) {
        auto& f = file(map_.at(name));
        const auto& info = f.header.at(name).obj();
        Tensor t;
        t.dtype = info.at("dtype").str();
        for (const auto& s : info.at("shape").arr()) t.shape.push_back(static_cast<std::uint64_t>(s.num()));
        const auto& off = info.at("data_offsets").arr();
        const auto b = static_cast<std::uint64_t>(off[0].num()), e = static_cast<std::uint64_t>(off[1].num());
        t.data = f.base + 8 + f.header_bytes + b;
        t.bytes = e - b;
        return t;
    }
    std::vector<float> floats(const std::string& name) {
        const Tensor t = get(name);
        std::vector<float> out;
        if (t.dtype == "BF16") {
            out.resize(t.bytes / 2);
            for (std::size_t i = 0; i < out.size(); ++i) {
                std::uint16_t b;
                std::memcpy(&b, t.data + 2 * i, 2);
                out[i] = msa::bf16_to_float(b);
            }
        } else if (t.dtype == "F32") {
            out.resize(t.bytes / 4);
            std::memcpy(out.data(), t.data, t.bytes);
        } else if (t.dtype == "F16") {
            out.resize(t.bytes / 2);
            for (std::size_t i = 0; i < out.size(); ++i) {
                _Float16 h;
                std::memcpy(&h, t.data + 2 * i, 2);
                out[i] = static_cast<float>(h);
            }
        } else {
            throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
        }
        return out;
    }

private:
    struct File {
        const std::uint8_t* base;
        std::size_t size;
        std::uint64_t header_bytes;
        JsonObj header;
    };
    static std::string read_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }
    File& file(const std::string& fn) {
        auto it = files_.find(fn);
        if (it != files_.end()) return it->second;
        const std::string path = dir_ + "/" + fn;
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open " + path);
        struct stat st{};
        fstat(fd, &st);
        void* m = mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) throw std::runtime_error("mmap " + path);
        File f;
        f.base = static_cast<const std::uint8_t*>(m);
        f.size = static_cast<std::size_t>(st.st_size);
        std::memcpy(&f.header_bytes, f.base, 8);
        JsonParser jp{reinterpret_cast<const char*>(f.base + 8), reinterpret_cast<const char*>(f.base + 8 + f.header_bytes)};
        f.header = jp.parse().obj();
        return files_.emplace(fn, std::move(f)).first->second;
    }
    std::string dir_;
    std::map<std::string, std::string> map_;
    std::map<std::string, File> files_;
};

// ---------------------------------------------------------------- device helpers
template <class T> [[maybe_unused]] T* dalloc(std::size_t n) {
    T* p = nullptr;
    hip_ok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    return p;
}
template <class T> [[maybe_unused]] T* dupload(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    hip_ok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}

[[maybe_unused]] __global__ void rmsnorm_gemma_kernel(const float* x, const float* w, float* y, unsigned d) {
    __shared__ float red[256];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * d;
    float* yr = y + static_cast<std::size_t>(blockIdx.x) * d;
    float s = 0.0F;
    for (unsigned i = threadIdx.x; i < d; i += 256) s = __builtin_fmaf(xr[i], xr[i], s);
    red[threadIdx.x] = s;
    __syncthreads();
    for (unsigned k = 128; k > 0; k >>= 1U) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    const float r = 1.0F / __builtin_sqrtf(red[0] / static_cast<float>(d) + kEps);
    for (unsigned i = threadIdx.x; i < d; i += 256) yr[i] = (xr[i] * r) * (1.0F + w[i]);
}
// y[t][o] = sum_d x[t][d] * wt[d][o]; wt is the transposed weight [d][n] in FP32. Block: one token, 128 outputs.
[[maybe_unused]] __global__ void dense_t_kernel(const float* x, const float* wt, float* y, unsigned d, unsigned n) {
    __shared__ float xs[kHidden];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * d;
    for (unsigned i = threadIdx.x; i < d; i += 128) xs[i] = xr[i];
    __syncthreads();
    const unsigned o = blockIdx.y * 128 + threadIdx.x;
    if (o >= n) return;
    float acc = 0.0F;
    for (unsigned i = 0; i < d; ++i) acc = __builtin_fmaf(xs[i], wt[static_cast<std::size_t>(i) * n + o], acc);
    y[static_cast<std::size_t>(blockIdx.x) * n + o] = acc;
}
[[maybe_unused]] __global__ void add_kernel(float* h, const float* y, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) h[i] = h[i] + y[i];
}
[[maybe_unused]] __global__ void swiglu_kernel(const float* g, const float* u, float* a, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i >= n) return;
    const float gg = g[i] < kLimit ? g[i] : kLimit;
    const float uu = u[i] < -kLimit ? -kLimit : (u[i] > kLimit ? kLimit : u[i]);
    a[i] = gg * (1.0F / (1.0F + expf(-kAlpha * gg))) * (uu + 1.0F);
}
[[maybe_unused]] __global__ void gather_kernel(const float* x, const std::uint32_t* rows, float* out, unsigned d) {
    const float* src = x + static_cast<std::size_t>(rows[blockIdx.x]) * d;
    float* dst = out + static_cast<std::size_t>(blockIdx.x) * d;
    for (unsigned i = threadIdx.x; i < d; i += 256) dst[i] = src[i];
}
[[maybe_unused]] __global__ void scatter_add_kernel(const float* in, const std::uint32_t* rows, const float* w, float* y, unsigned d) {
    float* dst = y + static_cast<std::size_t>(rows[blockIdx.x]) * d;
    const float* src = in + static_cast<std::size_t>(blockIdx.x) * d;
    const float s = w[blockIdx.x];
    for (unsigned i = threadIdx.x; i < d; i += 256) dst[i] = __builtin_fmaf(s, src[i], dst[i]);
}
[[maybe_unused]] __global__ void sum_ranks_kernel(const float* part0, const float* part1, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) out[i] = part0[i] + part1[i];
}
[[maybe_unused]] unsigned grid(std::size_t n) { return static_cast<unsigned>((n + 255) / 256); }

// ---------------------------------------------------------------- trellis linears
class Trellis {
public:
    Trellis() {
        strix_context_desc d{};
        d.struct_size = sizeof(d);
        d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.device_ordinal = 0;
        lib_ok(strix_context_create(&d, &ctx_), "context");
        st_.struct_size = sizeof(st_);
        st_.abi_version = STRIX_TRELLIS_ABI_VERSION;
        st_.hip_stream = nullptr;
    }
    ~Trellis() {
        (void)hipDeviceSynchronize();
        if (ws_) (void)hipFree(ws_);
        (void)strix_context_destroy(ctx_);
    }
    struct Lin {
        strix_weights w = nullptr;
        std::uint64_t rows = 0, cols = 0;  // N outputs, I inputs
    };
    Lin prepare(Pack& pack, const std::string& prefix) {
        const Tensor tr = pack.get(prefix + ".trellis"), suh = pack.get(prefix + ".suh"), svh = pack.get(prefix + ".svh");
        strix_exl3_parts_desc p{};
        p.struct_size = sizeof(p);
        p.abi_version = STRIX_TRELLIS_ABI_VERSION;
        p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        p.bits_x2 = static_cast<std::uint32_t>(2 * (tr.shape[2] / 16));
        p.rows = svh.shape[0];
        p.columns = suh.shape[0];
        p.codebook = STRIX_CODEBOOK_MUL1;
        p.marker = STRIX_EXL3_MARKER_MUL1;
        p.trellis = {tr.data, tr.bytes};
        p.suh = {suh.data, suh.bytes};
        p.svh = {svh.data, svh.bytes};
        Lin l;
        lib_ok(strix_weights_prepare_parts(ctx_, &p, &st_, &l.w), "prepare " + prefix);
        l.rows = p.rows;
        l.cols = p.columns;
        return l;
    }
    // Tensor-parallel slices of one EXL3 linear (H128-aligned, so no requantisation): outputs [o0, o0 + n) keep the
    // tiles of those outputs of every input row (copied contiguously) and the matching svh; inputs [i0, i0 + n) keep
    // the contiguous trellis rows and the matching suh. An input slice yields a partial sum to reduce across ranks.
    Lin prepare_out_slice(Pack& pack, const std::string& prefix, std::uint64_t o0, std::uint64_t n) {
        const Tensor tr = pack.get(prefix + ".trellis"), suh = pack.get(prefix + ".suh"), svh = pack.get(prefix + ".svh");
        if (o0 % 128 != 0 || n % 128 != 0 || o0 + n > svh.shape[0]) throw std::invalid_argument("bad output slice " + prefix);
        const std::uint64_t in_tiles = tr.shape[0], out_tiles = tr.shape[1], w = tr.shape[2] * 2;  // bytes per tile
        std::vector<std::uint8_t> buf(in_tiles * (n / 16) * w);
        for (std::uint64_t i = 0; i < in_tiles; ++i)
            std::memcpy(buf.data() + i * (n / 16) * w, tr.data + (i * out_tiles + o0 / 16) * w, (n / 16) * w);
        return prepare_regions(prefix, buf.data(), buf.size(), suh.data, suh.bytes, svh.data + o0 * 2, n * 2, n,
                               suh.shape[0], tr.shape[2] / 16);
    }
    Lin prepare_in_slice(Pack& pack, const std::string& prefix, std::uint64_t i0, std::uint64_t n) {
        const Tensor tr = pack.get(prefix + ".trellis"), suh = pack.get(prefix + ".suh"), svh = pack.get(prefix + ".svh");
        if (i0 % 128 != 0 || n % 128 != 0 || i0 + n > suh.shape[0]) throw std::invalid_argument("bad input slice " + prefix);
        const std::uint64_t row_bytes = tr.shape[1] * tr.shape[2] * 2;  // one input tile row
        return prepare_regions(prefix, tr.data + (i0 / 16) * row_bytes, (n / 16) * row_bytes, suh.data + i0 * 2, n * 2,
                               svh.data, svh.bytes, svh.shape[0], n, tr.shape[2] / 16);
    }
    Lin prepare_regions(const std::string& what, const void* tr, std::uint64_t tr_bytes, const void* suh, std::uint64_t suh_bytes,
                        const void* svh, std::uint64_t svh_bytes, std::uint64_t rows, std::uint64_t cols, std::uint64_t bits) {
        strix_exl3_parts_desc p{};
        p.struct_size = sizeof(p);
        p.abi_version = STRIX_TRELLIS_ABI_VERSION;
        p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        p.bits_x2 = static_cast<std::uint32_t>(2 * bits);
        p.rows = rows;
        p.columns = cols;
        p.codebook = STRIX_CODEBOOK_MUL1;
        p.marker = STRIX_EXL3_MARKER_MUL1;
        p.trellis = {tr, tr_bytes};
        p.suh = {suh, suh_bytes};
        p.svh = {svh, svh_bytes};
        Lin l;
        lib_ok(strix_weights_prepare_parts(ctx_, &p, &st_, &l.w), "prepare slice " + what);
        l.rows = rows;
        l.cols = cols;
        return l;
    }
    void release(Lin& l) {
        hip_ok(hipDeviceSynchronize(), "sync before release");
        lib_ok(strix_weights_destroy(l.w), "destroy");
        l.w = nullptr;
    }
    // out[T][N] = in[T][I] @ W
    void gemm(const Lin& l, const float* in, float* out, std::uint64_t t) {
        if (t == 0) return;
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = l.w;
        set(d.input, const_cast<float*>(in), t, l.cols);
        set(d.output, out, t, l.rows);
        std::uint64_t bytes = 0;
        lib_ok(strix_matmul_workspace_bytes(ctx_, &d, &bytes), "workspace");
        if (bytes > ws_bytes_) {
            hip_ok(hipDeviceSynchronize(), "sync");
            if (ws_) hip_ok(hipFree(ws_), "free");
            hip_ok(hipMalloc(&ws_, bytes), "workspace");
            ws_bytes_ = bytes;
        }
        d.workspace = bytes ? ws_ : nullptr;
        d.workspace_bytes = bytes ? ws_bytes_ : 0;
        lib_ok(strix_gemm(ctx_, &d, &st_), "gemm");
    }

private:
    static void set(strix_tensor_desc& t, float* p, std::uint64_t rows, std::uint64_t width) {
        t.data = p;
        t.buffer_bytes = rows * width * sizeof(float);
        t.dtype = STRIX_DTYPE_F32;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = width;
        t.strides[0] = width;
        t.strides[1] = 1;
    }
    strix_context ctx_ = nullptr;
    strix_stream st_{};
    void* ws_ = nullptr;
    std::uint64_t ws_bytes_ = 0;
};

[[maybe_unused]] std::vector<float> read_f32(const std::string& path, std::size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    if (!f) throw std::runtime_error("read " + path);
    return v;
}

struct Metrics {
    double rel_l2, tok_p50, tok_p99, tok_max;
};
[[maybe_unused]] Metrics compare(const std::vector<float>& a, const std::vector<float>& ref, std::size_t t, std::size_t d) {
    double num = 0, den = 0;
    std::vector<double> tok(t);
    for (std::size_t i = 0; i < t; ++i) {
        double n2 = 0, d2 = 0;
        for (std::size_t j = 0; j < d; ++j) {
            const double x = a[i * d + j], r = ref[i * d + j];
            n2 += (x - r) * (x - r);
            d2 += r * r;
        }
        num += n2;
        den += d2;
        tok[i] = std::sqrt(n2 / std::max(d2, 1e-300));
    }
    std::sort(tok.begin(), tok.end());
    return {std::sqrt(num / den), tok[t / 2], tok[std::min(t - 1, t * 99 / 100)], tok[t - 1]};
}

[[maybe_unused]] std::vector<float> transpose_f(const std::vector<float>& w, std::size_t n, std::size_t d) {  // [n][d] -> [d][n]
    std::vector<float> t(n * d);
    for (std::size_t o = 0; o < n; ++o)
        for (std::size_t i = 0; i < d; ++i) t[i * n + o] = w[o * d + i];
    return t;
}


}  // namespace
