// GLM-5.3-Flash decode MoE kernels (src/glm/moe): bitwise tests against the Tier B reference chain, and timing.
//
// A (reference chain): router = dev::dense (R1, F16 288 x 4096) + dev::route (R14 / R5); per distinct routed expert in
//   ascending id with its rows (gathered): strix_gemv gate, strix_gemv up (strix-trellis 5acf090, the decode identity),
//   dev::swiglu (R15), strix_gemv down; the shared expert the same over all rows; dev::moe_combine (R16).
// B (new): strix::glm::moe::enqueue_moe (router + shared overlap, routed gate_up, down + combine) and enqueue_experts
//   (given ids / weights).
//
// usage: hip-glm-moe-bench [--test] [--bench] [--real DIR --rank R] [--router FILEPREFIX] [--iters N] [--layers L]
//   Without --real the weights are synthetic (random trellis words, random FP16 scales; experts K4, shared K6).
//   --real DIR: raw EXL3 tensors (tensor layout: layers.3.mlp.{experts.0,experts.287,shared_experts}
//               .{gate,up,down}_proj.{trellis,suh,svh}); the 288 expert slots alternate the real experts 0 / 287, rank
//               slice --rank (gate/up output halves, down input half).
//   --router P: P.weight.f16 [288 x 4096] and P.bias.f32 [288] (real L3 router); else synthetic.
//   --test: bitwise B == A (out, per-pair expert outputs, router logits / ids / weights) for R = 1..8 and several
//           routing patterns; pipeline/current intermediates x3; batch invariance; untouched padding rows; router ties.
//           Real qualification runs this suite separately with --rank 0 and --rank 1.
//   --bench: per cell (R x pattern) A B A B A B A (ABA x3), each the mean over --iters calls rotating over --layers
//            distinct weight copies (cold); GB/s of the weight bytes read; B parts per launch; read ceiling.
//            R1/R8 additionally time current/pipeline/current x3 on the same rotation.
#include "glm/common/glm_math.hpp"
#include "glm/kernels/glm_tier_a.hpp"
#include "glm/moe/glm_moe.hpp"
#include "strix_trellis.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace moe = strix::glm::moe;
namespace tier = strix::glm::dev;
constexpr unsigned H = moe::kHidden, I = moe::kInter, E = moe::kExperts, K = moe::kTopE;
constexpr double kCeiling = 246.7;  // GB/s, measured read ceiling per node (design notes)

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("OK   %s\n", what.c_str());
    }
    std::fflush(stdout);
}
void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
void lib_ok(strix_status s, const char* what) {
    if (s != STRIX_SUCCESS) {
        std::fprintf(stderr, "strix-trellis error %s: %d\n", what, static_cast<int>(s));
        std::exit(2);
    }
}
template <class T> T* dalloc(std::size_t n) {
    T* p = nullptr;
    hip_ok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    return p;
}
template <class T> T* dev(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    hip_ok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> std::vector<T> d2h(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip_ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
template <class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b, std::size_t* first = nullptr) {
    if (a.size() != b.size()) return false;
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0) return true;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            if (first) *first = i;
            return false;
        }
    return false;
}

std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
std::uint16_t f16_bits(float f) {
    const _Float16 h = static_cast<_Float16>(f);
    std::uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

// ---------------------------------------------------------------- weights
struct HostExl3 {
    std::vector<std::uint32_t> trellis;
    std::vector<std::uint16_t> suh, svh;
    unsigned input = 0, output = 0, bits = 0;
    std::uint64_t bytes() const { return trellis.size() * 4ULL + suh.size() * 2ULL + svh.size() * 2ULL; }
};
HostExl3 synthetic(unsigned input, unsigned output, unsigned bits, std::uint64_t seed) {
    HostExl3 h;
    h.input = input;
    h.output = output;
    h.bits = bits;
    h.trellis.resize(static_cast<std::size_t>(input / 16) * (output / 16) * bits * 8);
    std::uint64_t s = seed;
    for (std::size_t i = 0; i < h.trellis.size(); i += 2) {
        const std::uint64_t v = splitmix(s);
        h.trellis[i] = static_cast<std::uint32_t>(v);
        h.trellis[i + 1] = static_cast<std::uint32_t>(v >> 32);
    }
    auto scale = [&](std::vector<std::uint16_t>& v, unsigned n, float mag) {
        v.resize(n);
        for (unsigned i = 0; i < n; ++i) {
            const std::uint64_t r = splitmix(s);
            const float m = mag * (0.5F + static_cast<float>(r & 0xffff) / 65536.0F);
            v[i] = f16_bits((r >> 16) & 1 ? -m : m);
            if (i % 97 == 5) v[i] = static_cast<std::uint16_t>(((r >> 17) & 1 ? 0x8000U : 0U) | (1U + ((r >> 20) & 0x3feU)));  // subnormal
            if (i % 101 == 7) v[i] = (r >> 17) & 1 ? 0x8000U : 0U;                                                         // +-0
        }
    };
    scale(h.suh, input, 0.06F);
    scale(h.svh, output, 0.03F);
    return h;
}
std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(2);
    }
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
// Rank slice of a raw EXL3 tensor (trellis [in/16][out/16][16 bits] int16, suh [in], svh [out]):
// out_slice: output columns [first, first + n); else input rows [first, first + n).
HostExl3 real_slice(const std::string& prefix, unsigned in_full, unsigned out_full, bool out_slice, unsigned first, unsigned n) {
    const auto tr = read_file(prefix + ".trellis"), suh = read_file(prefix + ".suh"), svh = read_file(prefix + ".svh");
    const std::uint64_t in_tiles = in_full / 16, out_tiles = out_full / 16;
    const std::uint64_t tile = tr.size() / (in_tiles * out_tiles);  // bytes per 16 x 16 tile = 32 bits
    if (tile * in_tiles * out_tiles != tr.size() || tile % 32 || suh.size() != in_full * 2ULL || svh.size() != out_full * 2ULL) {
        std::fprintf(stderr, "bad tensor geometry %s\n", prefix.c_str());
        std::exit(2);
    }
    HostExl3 h;
    h.bits = static_cast<unsigned>(tile / 32);
    auto copy16 = [](std::vector<std::uint16_t>& v, const std::uint8_t* p, std::uint64_t count) {
        v.resize(count);
        std::memcpy(v.data(), p, count * 2);
    };
    if (out_slice) {
        h.input = in_full;
        h.output = n;
        h.trellis.resize(in_tiles * (n / 16) * tile / 4);
        auto* dst = reinterpret_cast<std::uint8_t*>(h.trellis.data());
        for (std::uint64_t i = 0; i < in_tiles; ++i)
            std::memcpy(dst + i * (n / 16) * tile, tr.data() + (i * out_tiles + first / 16) * tile, (n / 16) * tile);
        copy16(h.suh, suh.data(), in_full);
        copy16(h.svh, svh.data() + first * 2ULL, n);
    } else {
        h.input = n;
        h.output = out_full;
        const std::uint64_t row = out_tiles * tile;
        h.trellis.resize((n / 16) * row / 4);
        std::memcpy(h.trellis.data(), tr.data() + (first / 16) * row, (n / 16) * row);
        copy16(h.suh, suh.data() + first * 2ULL, n);
        copy16(h.svh, svh.data(), out_full);
    }
    return h;
}

class Lib {
public:
    Lib() {
        strix_context_desc d{};
        d.struct_size = sizeof(d);
        d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.device_ordinal = 0;
        lib_ok(strix_context_create(&d, &ctx_), "context");
        st_.struct_size = sizeof(st_);
        st_.abi_version = STRIX_TRELLIS_ABI_VERSION;
        st_.hip_stream = nullptr;
    }
    ~Lib() {
        (void)hipDeviceSynchronize();
        for (auto w : owned_) (void)strix_weights_destroy(w);
        if (ws_) (void)hipFree(ws_);
        (void)strix_context_destroy(ctx_);
    }
    strix_weights prepare(const HostExl3& h) {
        strix_exl3_parts_desc p{};
        p.struct_size = sizeof(p);
        p.abi_version = STRIX_TRELLIS_ABI_VERSION;
        p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        p.bits_x2 = 2 * h.bits;
        p.rows = h.output;
        p.columns = h.input;
        p.codebook = STRIX_CODEBOOK_MUL1;
        p.marker = STRIX_EXL3_MARKER_MUL1;
        p.trellis = {h.trellis.data(), h.trellis.size() * 4};
        p.suh = {h.suh.data(), h.suh.size() * 2};
        p.svh = {h.svh.data(), h.svh.size() * 2};
        strix_weights w = nullptr;
        lib_ok(strix_weights_prepare_parts(ctx_, &p, &st_, &w), "prepare");
        owned_.push_back(w);
        return w;
    }
    // strix_gemv (the decode identity), rows <= 8.
    void gemv(strix_weights w, unsigned input, unsigned output, const float* in, float* out, std::uint64_t rows) {
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = w;
        set(d.input, const_cast<float*>(in), rows, input);
        set(d.output, out, rows, output);
        std::uint64_t bytes = 0;
        lib_ok(strix_matmul_workspace_bytes(ctx_, &d, &bytes), "workspace");
        if (bytes > ws_bytes_) {
            hip_ok(hipDeviceSynchronize(), "sync");
            if (ws_) hip_ok(hipFree(ws_), "free");
            hip_ok(hipMalloc(&ws_, bytes), "workspace");
            ws_bytes_ = bytes;
        }
        d.workspace = ws_;
        d.workspace_bytes = ws_bytes_;
        lib_ok(strix_gemv(ctx_, &d, &st_), "gemv");
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
    std::vector<strix_weights> owned_;
};

struct DevExl3 {
    std::uint32_t* trellis = nullptr;
    std::uint16_t *suh = nullptr, *svh = nullptr;
    moe::Exl3 view(const HostExl3& h) const { return {trellis, suh, svh, h.input, h.output, h.bits}; }
};
DevExl3 upload(const HostExl3& h) { return {dev(h.trellis), dev(h.suh), dev(h.svh)}; }

// The model's MoE weights of one layer (rank slices): 288 routed experts (gate, up, down) + shared + router.
struct Model {
    std::vector<HostExl3> host;                     // distinct host matrices (3 per distinct expert content + 3 shared)
    std::vector<unsigned> expert_src;               // expert e -> index of its gate in host (gate, up, down consecutive)
    unsigned shared_src = 0;
    std::vector<moe::ExpertSlices*> experts_dev;    // [layer] device array [288]
    std::vector<moe::ExpertSlices> shared;          // [layer]
    std::vector<strix_weights> lib;                 // [(layer x 289 + slot) x 3 + matrix] library copies (lazy; slot 288 = shared)
    std::vector<std::uint16_t> router_w;            // 288 x 4096 F16
    std::vector<float> bias;
    std::uint16_t* router_w_dev = nullptr;
    float* bias_dev = nullptr;
    std::uint64_t expert_bytes = 0, shared_bytes = 0;  // per routed expert / shared (rank slices, trellis + scales)
};

void build_model(Model& m, Lib& lib, const std::string& real, unsigned rank, const std::string& router, unsigned layers,
                 unsigned layout) {
    if (real.empty()) {
        for (unsigned e = 0; e < E; ++e) {
            m.expert_src.push_back(static_cast<unsigned>(m.host.size()));
            m.host.push_back(synthetic(H, I, 4, 1000ULL + e * 3));
            m.host.push_back(synthetic(H, I, 4, 1001ULL + e * 3));
            m.host.push_back(synthetic(I, H, 4, 1002ULL + e * 3));
        }
        m.shared_src = static_cast<unsigned>(m.host.size());
        m.host.push_back(synthetic(H, I, 6, 7));
        m.host.push_back(synthetic(H, I, 6, 8));
        m.host.push_back(synthetic(I, H, 6, 9));
    } else {
        const std::string p = real + "/layers.3.mlp.";
        auto add = [&](const std::string& name) {
            const unsigned at = static_cast<unsigned>(m.host.size());
            m.host.push_back(real_slice(p + name + ".gate_proj", H, 2 * I, true, rank * I, I));
            m.host.push_back(real_slice(p + name + ".up_proj", H, 2 * I, true, rank * I, I));
            m.host.push_back(real_slice(p + name + ".down_proj", 2 * I, H, false, rank * I, I));
            return at;
        };
        const unsigned e0 = add("experts.0"), e287 = add("experts.287");
        m.shared_src = add("shared_experts");
        for (unsigned e = 0; e < E; ++e) m.expert_src.push_back(e % 2 ? e287 : e0);
        for (unsigned j = 0; j < 3; ++j)
            if (m.host[e0 + j].bits != 4 || m.host[m.shared_src + j].bits != 6) {
                std::fprintf(stderr, "unexpected real bits\n");
                std::exit(2);
            }
    }
    m.expert_bytes = m.host[m.expert_src[0]].bytes() + m.host[m.expert_src[0] + 1].bytes() + m.host[m.expert_src[0] + 2].bytes();
    m.shared_bytes = m.host[m.shared_src].bytes() + m.host[m.shared_src + 1].bytes() + m.host[m.shared_src + 2].bytes();
    m.lib.assign(static_cast<std::size_t>(layers) * (E + 1) * 3, nullptr);
    (void)lib;
    // Device copies per layer, distinct per layer and per expert slot (a rotation over layers reads cold weights; real
    // weights: 288 distinct copies of the two real experts). Layout (the loader question, --layout):
    //   0 one allocation per tensor (trellis 2 MiB / 3 MiB each);
    //   1 one region per layer: every matrix's trellis 2 MiB aligned, back to back (expert e at e x 6 MiB), all
    //     suh / svh in a tail block (the proposed slab layout);
    //   2 one region per layer in file order per expert ([gate trellis suh svh][up ..][down ..], 256-byte aligned
    //     tensors: trellis starts drift off the 2 MiB grid by 30 KiB per expert).
    for (unsigned l = 0; l < layers; ++l) {
        std::vector<moe::ExpertSlices> slots(E);
        unsigned char* base = nullptr;
        std::size_t tcur = 0, scur = 0;
        auto up16 = [](std::size_t v, std::size_t a) { return (v + a - 1) / a * a; };
        if (layout != 0) {
            std::size_t tr = 0, sc = 0;
            for (unsigned e = 0; e <= E; ++e)
                for (unsigned j = 0; j < 3; ++j) {
                    const HostExl3& h = m.host[(e < E ? m.expert_src[e] : m.shared_src) + j];
                    tr = up16(tr, layout == 1 ? 2097152 : 256) + h.trellis.size() * 4;
                    if (layout == 1) sc = up16(sc, 256) + up16(h.suh.size() * 2, 256) + up16(h.svh.size() * 2, 256);
                    else tr = up16(tr, 256) + up16(h.suh.size() * 2, 256) + up16(h.svh.size() * 2, 256);
                }
            scur = up16(tr, 2097152);
            // over-allocate and align the usable base (hipMalloc alone does not promise 2 MiB alignment)
            auto* raw = dalloc<unsigned char>(scur + sc + 2097152);
            base = reinterpret_cast<unsigned char*>((reinterpret_cast<std::uintptr_t>(raw) + 2097151) & ~std::uintptr_t{2097151});
        }
        auto put = [&](unsigned hi) -> moe::Exl3 {
            const HostExl3& h = m.host[hi];
            if (layout == 0) return upload(h).view(h);
            auto copy = [&](std::size_t& cur, std::size_t align, const void* src, std::size_t bytes) {
                cur = up16(cur, align);
                hip_ok(hipMemcpy(base + cur, src, bytes, hipMemcpyHostToDevice), "place");
                void* p = base + cur;
                cur += bytes;
                return p;
            };
            auto* t = static_cast<std::uint32_t*>(copy(tcur, layout == 1 ? 2097152 : 256, h.trellis.data(), h.trellis.size() * 4));
            std::size_t& sc = layout == 1 ? scur : tcur;
            auto* su = static_cast<std::uint16_t*>(copy(sc, 256, h.suh.data(), h.suh.size() * 2));
            auto* sv = static_cast<std::uint16_t*>(copy(sc, 256, h.svh.data(), h.svh.size() * 2));
            return {t, su, sv, h.input, h.output, h.bits};
        };
        for (unsigned e = 0; e < E; ++e) {
            const unsigned s = m.expert_src[e];
            slots[e].gate = put(s);
            slots[e].up = put(s + 1);
            slots[e].down = put(s + 2);
        }
        const unsigned s = m.shared_src;
        moe::ExpertSlices sh;
        sh.gate = put(s);
        sh.up = put(s + 1);
        sh.down = put(s + 2);
        m.shared.push_back(sh);
        m.experts_dev.push_back(dev(slots));
    }
    if (router.empty()) {
        std::uint64_t s = 77;
        m.router_w.resize(static_cast<std::size_t>(E) * H);
        for (auto& w : m.router_w) {
            const std::uint64_t r = splitmix(s);
            // mostly normal values around 0.02, some subnormals and exact zeros (conversion paths)
            if ((r & 0xff) == 0) w = static_cast<std::uint16_t>((r >> 8) & 0x83ffU);  // +-subnormal
            else if ((r & 0xff) == 1) w = (r >> 8) & 1 ? 0x8000U : 0;
            else w = f16_bits(0.03F * (static_cast<float>((r >> 8) & 0xffff) / 32768.0F - 1.0F));
        }
        m.bias.resize(E);
        for (auto& b : m.bias) b = 0.01F * static_cast<float>(splitmix(s) & 0xff) / 256.0F;
    } else {
        const auto w = read_file(router + ".weight.f16"), b = read_file(router + ".bias.f32");
        if (w.size() != static_cast<std::size_t>(E) * H * 2 || b.size() != E * 4ULL) {
            std::fprintf(stderr, "bad router files\n");
            std::exit(2);
        }
        m.router_w.resize(static_cast<std::size_t>(E) * H);
        std::memcpy(m.router_w.data(), w.data(), w.size());
        m.bias.resize(E);
        std::memcpy(m.bias.data(), b.data(), b.size());
    }
    m.router_w_dev = dev(m.router_w);
    m.bias_dev = dev(m.bias);
}
// The library's own copy of matrix j of expert slot e (E = shared) for layer l: distinct per layer and slot like the
// kernels' copies, so the reference chain reads cold weights in the bench rotation too.
strix_weights lib_of(Model& m, Lib& lib, unsigned layer, unsigned e, unsigned j) {
    const std::size_t at = (static_cast<std::size_t>(layer) * (E + 1) + e) * 3 + j;
    if (m.lib[at] == nullptr) m.lib[at] = lib.prepare(m.host[(e < E ? m.expert_src[e] : m.shared_src) + j]);
    return m.lib[at];
}

// ---------------------------------------------------------------- routing patterns (host ids, ascending per row)
std::vector<std::uint32_t> make_ids(unsigned rows, int pattern, std::uint64_t seed) {
    std::vector<std::uint32_t> ids(static_cast<std::size_t>(rows) * K);
    std::uint64_t s = seed;
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> pick;
        if (pattern == 1 || (pattern == 2 && r % 2 == 1)) {  // 1: all rows the same 8 experts; 2: odd rows copy r - 1
            if (r > 0) {
                std::copy(ids.begin() + (pattern == 1 ? 0 : (r - 1) * K), ids.begin() + (pattern == 1 ? K : r * K), ids.begin() + r * K);
                continue;
            }
        }
        if (pattern == 3) {  // extremes: rows alternate {0..7} and {280..287}
            for (unsigned k = 0; k < K; ++k) pick.push_back(r % 2 ? E - K + k : k);
        } else {
            while (pick.size() < K) {
                const auto e = static_cast<std::uint32_t>(splitmix(s) % E);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
        }
        std::sort(pick.begin(), pick.end());
        std::copy(pick.begin(), pick.end(), ids.begin() + r * K);
    }
    return ids;
}
std::vector<float> make_weights(const std::vector<std::uint32_t>& ids, std::uint64_t seed) {
    std::vector<float> w(ids.size());
    std::uint64_t s = seed;
    for (auto& v : w) v = 0.05F + 0.6F * static_cast<float>(splitmix(s) & 0xffff) / 65536.0F;
    return w;
}
unsigned distinct(const std::vector<std::uint32_t>& ids) {
    std::vector<std::uint32_t> v = ids;
    std::sort(v.begin(), v.end());
    return static_cast<unsigned>(std::unique(v.begin(), v.end()) - v.begin());
}
const char* pattern_name(int p) {
    static const char* n[] = {"random", "same", "pairs", "extremes", "disjoint", "maskedges"};
    return n[p];
}
constexpr int kPatterns = 6;
// Patterns 4 / 5 (make_ids handles 0..3): 4 = rows pairwise disjoint (row r: e = 36 k + r, k = 0..7); 5 = ids drawn
// from the mask-word edges {32 w - 1, 32 w} (incl. 0 and 287).
std::vector<std::uint32_t> make_ids_ext(unsigned rows, int pattern, std::uint64_t seed) {
    if (pattern < 4) return make_ids(rows, pattern, seed);
    std::vector<std::uint32_t> ids(static_cast<std::size_t>(rows) * K);
    std::vector<std::uint32_t> edges = {0};
    for (unsigned w = 1; w < 9; ++w) {
        edges.push_back(32 * w - 1);
        edges.push_back(32 * w);
    }
    edges.push_back(E - 1);
    std::uint64_t s = seed;
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> pick;
        if (pattern == 4) {
            for (unsigned k = 0; k < K; ++k) pick.push_back(36 * k + r);
        } else {
            while (pick.size() < K) {
                const auto e = edges[splitmix(s) % edges.size()];
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
        }
        std::sort(pick.begin(), pick.end());
        std::copy(pick.begin(), pick.end(), ids.begin() + r * K);
    }
    return ids;
}

// ---------------------------------------------------------------- reference chain (A)
struct Scratch {
    float *gy, *gg, *gu, *ga, *go, *eo, *sg, *su, *sa, *sd;
};
Scratch make_scratch() {
    Scratch s{};
    s.gy = dalloc<float>(8ULL * H);
    s.gg = dalloc<float>(8ULL * I);
    s.gu = dalloc<float>(8ULL * I);
    s.ga = dalloc<float>(8ULL * I);
    s.go = dalloc<float>(8ULL * H);
    s.eo = dalloc<float>(8ULL * K * H);
    s.sg = dalloc<float>(8ULL * I);
    s.su = dalloc<float>(8ULL * I);
    s.sa = dalloc<float>(8ULL * I);
    s.sd = dalloc<float>(8ULL * H);
    return s;
}
// Per-pair reference outputs (kernel pair order: shared R pairs, then routed (expert asc, row asc)).
struct PairRef {
    std::vector<std::vector<float>> down;     // strix_gemv(down) output, 4096
    std::vector<std::vector<float>> gate_up;  // strix_gemv(gate) | strix_gemv(up), 2 x 1024
    std::vector<unsigned> down_mat;           // host index of the pair's down matrix (for its suh)
};
// Per distinct expert ascending: gather its rows, gate/up/swiglu/down, scatter to eo[r][k]; shared; combine.
void chain_A(Model& m, Lib& lib, unsigned layer, const float* y, const std::vector<std::uint32_t>& ids, const float* w_dev,
             unsigned rows, float* out, Scratch& s, hipStream_t st, PairRef* ref = nullptr) {
    std::vector<std::uint32_t> experts = ids;
    std::sort(experts.begin(), experts.end());
    experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
    auto libw = [&](unsigned e, unsigned j) { return lib_of(m, lib, layer, e, j); };
    if (ref) {
        ref->down.assign(rows + ids.size(), {});
        ref->gate_up.assign(rows + ids.size(), {});
        ref->down_mat.assign(rows + ids.size(), 0);
    }
    auto keep_gu = [&](unsigned pair, const float* g, const float* u) {
        auto a = d2h(g, I), b = d2h(u, I);
        a.insert(a.end(), b.begin(), b.end());
        ref->gate_up[pair] = std::move(a);
    };
    unsigned pair = rows;
    for (const auto e : experts) {
        std::vector<unsigned> rws, ks;
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned k = 0; k < K; ++k)
                if (ids[r * K + k] == e) {
                    rws.push_back(r);
                    ks.push_back(k);
                }
        const auto n = static_cast<unsigned>(rws.size());
        for (unsigned j = 0; j < n; ++j)
            hip_ok(hipMemcpyAsync(s.gy + j * H, y + rws[j] * H, H * 4, hipMemcpyDeviceToDevice, st), "gather");
        lib.gemv(libw(e, 0), H, I, s.gy, s.gg, n);
        lib.gemv(libw(e, 1), H, I, s.gy, s.gu, n);
        tier::swiglu(s.gg, s.gu, s.ga, static_cast<std::size_t>(n) * I, st);
        lib.gemv(libw(e, 2), I, H, s.ga, s.go, n);
        for (unsigned j = 0; j < n; ++j) {
            hip_ok(hipMemcpyAsync(s.eo + (rws[j] * K + ks[j]) * H, s.go + j * H, H * 4, hipMemcpyDeviceToDevice, st), "scatter");
            if (ref) {
                ref->down[pair] = d2h(s.go + j * H, H);
                keep_gu(pair, s.gg + j * I, s.gu + j * I);
                ref->down_mat[pair] = m.expert_src[e] + 2;
            }
            ++pair;
        }
    }
    lib.gemv(libw(E, 0), H, I, y, s.sg, rows);
    lib.gemv(libw(E, 1), H, I, y, s.su, rows);
    tier::swiglu(s.sg, s.su, s.sa, static_cast<std::size_t>(rows) * I, st);
    lib.gemv(libw(E, 2), I, H, s.sa, s.sd, rows);
    if (ref)
        for (unsigned r = 0; r < rows; ++r) {
            ref->down[r] = d2h(s.sd + r * H, H);
            keep_gu(r, s.sg + r * I, s.su + r * I);
            ref->down_mat[r] = m.shared_src + 2;
        }
    tier::moe_combine(w_dev, s.eo, s.sd, out, rows, st);
}

// Host emulation of the library input transform of the down linear on the reference R15 output:
// a = swiglu(g, u) (glm_math silu, = dev::swiglu, gate K), t = RN16(H32(RN16(RN32(a * suh)))), H32 in the device order
// (hadamard.hpp: lane strides 1..16 with (lane & stride) ? peer - v : v + peer, register strides 32 and 64, * RN32(1/sqrt 128)).
std::vector<std::uint16_t> host_down_input(const std::vector<float>& gu, const std::vector<std::uint16_t>& suh) {
    std::vector<std::uint16_t> out(I);
    auto h2f = [](std::uint16_t b) { return strix::glm::f16_to_float(b); };
    for (unsigned blk = 0; blk < I / 128; ++blk) {
        float v[32][4];
        for (unsigned l = 0; l < 32; ++l)
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned c = blk * 128 + l + 32 * j;
                const float g = gu[c], u = gu[I + c];
                const float gg = g > strix::glm::kClamp ? strix::glm::kClamp : g;
                const float uu = u > strix::glm::kClamp ? strix::glm::kClamp : (u < -strix::glm::kClamp ? -strix::glm::kClamp : u);
                const float act = strix::glm::silu(gg) * uu;
                const float scaled = act * h2f(suh[c]);
                v[l][j] = static_cast<float>(static_cast<_Float16>(scaled));
            }
        for (unsigned st = 1; st < 32; st *= 2) {
            float nv[32][4];
            for (unsigned l = 0; l < 32; ++l)
                for (unsigned j = 0; j < 4; ++j) {
                    const float peer = v[l ^ st][j];
                    nv[l][j] = (l & st) ? peer - v[l][j] : v[l][j] + peer;
                }
            std::memcpy(v, nv, sizeof(v));
        }
        for (unsigned l = 0; l < 32; ++l) {
            for (unsigned st = 1; st < 4; st *= 2)
                for (unsigned j = 0; j < 4; ++j)
                    if (!(j & st)) {
                        const float a = v[l][j], b = v[l][j + st];
                        v[l][j] = a + b;
                        v[l][j + st] = a - b;
                    }
            for (unsigned j = 0; j < 4; ++j) {
                const _Float16 hv = static_cast<_Float16>(v[l][j] * 0x1.6a09e6p-4F);
                std::memcpy(&out[blk * 128 + l + 32 * j], &hv, 2);
            }
        }
    }
    return out;
}

// y kinds: 0 = random (uniform +-1, a few x8); 1 = edges: exact zeros, -0, FP32 subnormals, +-1e4 (swiglu clamps and
// sigmoid saturation downstream), FP16-subnormal-sized values.
std::vector<float> make_y(unsigned rows, std::uint64_t seed, int kind = 0) {
    std::vector<float> y(static_cast<std::size_t>(rows) * H);
    std::uint64_t s = seed;
    for (auto& v : y) {
        const std::uint64_t r = splitmix(s);
        v = 2.0F * (static_cast<float>(r & 0xffffff) / 16777216.0F - 0.5F);
        if ((r >> 40) % 97 == 0) v *= 8.0F;  // a few larger activations
        if (kind == 1) {
            switch ((r >> 32) % 8) {
                case 0: v = 0.0F; break;
                case 1: v = -0.0F; break;
                case 2: v = std::ldexp(v, -130); break;   // FP32 subnormal
                case 3: v = v > 0 ? 1e4F : -1e4F; break;   // saturation
                case 4: v = std::ldexp(v, -18); break;    // rounds to FP16 subnormals after the suh scale
                default: break;
            }
        }
    }
    return y;
}

// Streaming read of n uint4 words (4 independent loads in flight per thread per step), the size-matched read ceiling.
__global__ __launch_bounds__(256) void read_kernel(const uint4* p, std::size_t n, unsigned* out) {
    unsigned acc = 0;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * 256U;
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) {
        const uint4 a = p[i];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x9e3779b9U) out[0] = acc;
}

struct Bench {
    Model m;
    Lib lib;
    Scratch s{};
    moe::MoeWorkspace ws{};
    float *y = nullptr, *outA = nullptr, *outB = nullptr, *logitsA = nullptr, *logitsB = nullptr, *wA = nullptr, *wB = nullptr;
    std::uint32_t *idsA = nullptr, *idsB = nullptr;
    float* dbg_gu = nullptr;  // [kMaxPairs][2][1024] gate / up linear outputs (MoeView::debug_gate_up, tests only)
    hipStream_t st = nullptr;  // the library uses the null stream: everything below runs on the null stream

    void init(const std::string& real, unsigned rank, const std::string& router, unsigned layers, unsigned layout) {
        build_model(m, lib, real, rank, router, layers, layout);
        s = make_scratch();
        hip_ok(hipMalloc(&ws.base, moe::moe_workspace_bytes()), "ws");
        hip_ok(hipMemset(ws.base, 0, moe::moe_workspace_bytes()), "ws zero");
        y = dalloc<float>(16ULL * H);  // rows 8..15 = poison (never read)
        outA = dalloc<float>(16ULL * H);
        outB = dalloc<float>(16ULL * H);
        logitsA = dalloc<float>(8ULL * E);
        logitsB = dalloc<float>(8ULL * E);
        wA = dalloc<float>(8ULL * K);
        wB = dalloc<float>(8ULL * K);
        idsA = dalloc<std::uint32_t>(8ULL * K);
        idsB = dalloc<std::uint32_t>(8ULL * K);
        dbg_gu = dalloc<float>(static_cast<std::size_t>(moe::kMaxPairs) * 2 * I);
        hip_ok(hipDeviceSynchronize(), "init");
    }
    moe::MoeView view(unsigned rows, unsigned layer, bool router) {
        moe::MoeView v;
        v.rows = rows;
        v.y = y;
        v.router_w = m.router_w_dev;
        v.bias = m.bias_dev;
        v.ids = router ? idsB : idsA;
        v.weights = router ? wB : wA;
        v.logits = router ? logitsB : nullptr;
        v.experts = m.experts_dev[layer];
        v.shared = m.shared[layer];
        v.out = outB;
        v.ws = ws;
        return v;
    }
    void set_y(unsigned rows, std::uint64_t seed, int kind = 0) {
        auto h = make_y(rows, seed, kind);
        h.resize(16ULL * H, std::nanf(""));
        hip_ok(hipMemcpy(y, h.data(), h.size() * 4, hipMemcpyHostToDevice), "y");
    }
    // NaN in both outputs, the gate/up diagnostics and the workspace's pair outputs / down inputs (0xFFFF halves), so a
    // missing write can never leave an expected value behind.
    void poison_out() {
        std::vector<float> p(16ULL * H, std::nanf(""));
        hip_ok(hipMemcpy(outA, p.data(), p.size() * 4, hipMemcpyHostToDevice), "poison");
        hip_ok(hipMemcpy(outB, p.data(), p.size() * 4, hipMemcpyHostToDevice), "poison");
        hip_ok(hipMemset(dbg_gu, 0xff, static_cast<std::size_t>(moe::kMaxPairs) * 2 * I * 4), "poison dbg");
        for (unsigned q = 0; q < moe::kMaxPairs; ++q) {
            hip_ok(hipMemset(const_cast<float*>(moe::moe_pair_gate_partials(ws, q)), 0xff, 2 * 32 * I * 4ULL), "poison partials");
            hip_ok(hipMemset(const_cast<float*>(moe::moe_pair_output(ws, q)), 0xff, H * 4ULL), "poison y");
            hip_ok(hipMemset(const_cast<std::uint16_t*>(moe::moe_pair_down_input(ws, q)), 0xff, I * 2ULL), "poison at");
        }
    }
    // Every pair: the kernels' down output and gate / up linear outputs == the library's (each replaced linear).
    std::string compare_pairs(const PairRef& ref, bool& ok) {
        ok = true;
        std::string where;
        for (std::size_t p = 0; p < ref.down.size(); ++p) {
            std::size_t f = 0;
            const auto pb = d2h(moe::moe_pair_output(ws, static_cast<unsigned>(p)), H);
            if (!same_bits(ref.down[p], pb, &f)) {
                if (ok) where = " first bad down pair " + std::to_string(p) + " at " + std::to_string(f);
                ok = false;
            }
            const auto gb = d2h(dbg_gu + p * 2 * I, 2ULL * I);
            if (!same_bits(ref.gate_up[p], gb, &f)) {
                if (ok) where = " first bad gate/up pair " + std::to_string(p) + " at " + std::to_string(f);
                ok = false;
            }
            const auto at = d2h(moe::moe_pair_down_input(ws, static_cast<unsigned>(p)), I);
            if (!same_bits(host_down_input(ref.gate_up[p], m.host[ref.down_mat[p]].suh), at, &f)) {
                if (ok) where = " first bad down-input pair " + std::to_string(p) + " at " + std::to_string(f);
                ok = false;
            }
        }
        return where;
    }

    // Snapshot every materialized intermediate, including raw split partials. Counters
    // reset between calls; re-poison data only, so no stale correct value can hide a miss.
    std::vector<std::uint8_t> pipe_snapshot(unsigned rows, unsigned pairs, bool router) {
        std::vector<std::uint8_t> bytes;
        auto append = [&](const auto& h) {
            const auto* b = reinterpret_cast<const std::uint8_t*>(h.data());
            bytes.insert(bytes.end(), b, b + h.size() * sizeof(h[0]));
        };
        append(d2h(outB, 16ULL * H));
        if (router) {
            append(d2h(logitsB, rows * E)); append(d2h(idsB, rows * K)); append(d2h(wB, rows * K));
        }
        for (unsigned p = 0; p < pairs; ++p) {
            append(d2h(dbg_gu + static_cast<std::size_t>(p) * 2 * I, 2 * I));
            append(d2h(moe::moe_pair_gate_partials(ws, p), 2 * 32 * I));
            append(d2h(moe::moe_pair_down_input(ws, p), I));
            append(d2h(moe::moe_pair_output(ws, p), H));
        }
        return bytes;
    }
    void compare_pipe(moe::MoeView v, bool router, unsigned pairs, const std::string& tag) {
        const auto old = pipe_snapshot(v.rows, pairs, router);
        bool exact = true;
        for (unsigned rep = 0; rep < 3; ++rep) {
            const std::vector<float> poison(16ULL * H, std::nanf(""));
            hip_ok(hipMemcpy(outB, poison.data(), poison.size() * 4, hipMemcpyHostToDevice), "pipe poison out");
            hip_ok(hipMemset(dbg_gu, 0xff, static_cast<std::size_t>(moe::kMaxPairs) * 2 * I * 4), "pipe poison gu");
            for (unsigned p = 0; p < pairs; ++p) {
                hip_ok(hipMemset(const_cast<float*>(moe::moe_pair_gate_partials(ws, p)), 0xff, 2 * 32 * I * 4ULL), "pipe poison partials");
                hip_ok(hipMemset(const_cast<float*>(moe::moe_pair_output(ws, p)), 0xff, H * 4ULL), "pipe poison down");
                hip_ok(hipMemset(const_cast<std::uint16_t*>(moe::moe_pair_down_input(ws, p)), 0xff, I * 2ULL), "pipe poison at");
            }
            if (router) moe::enqueue_moe_pipe(v, st);
            else moe::enqueue_experts_pipe(v, st);
            hip_ok(hipDeviceSynchronize(), "pipe compare");
            exact = same_bits(old, pipe_snapshot(v.rows, pairs, router)) && exact;
        }
        check(exact, tag + " pipeline == current: all split/gate/up/down-input/down/R16 bits, padding, x3");
    }

    // Bitwise B == A for given ids (enqueue_experts): out, every pair's gate / up / down linear outputs.
    bool test_experts(unsigned rows, const std::vector<std::uint32_t>& ids, const std::string& tag, std::vector<float>* outB_h = nullptr) {
        const auto w = make_weights(ids, 99 + rows);
        hip_ok(hipMemcpy(idsA, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
        hip_ok(hipMemcpy(wA, w.data(), w.size() * 4, hipMemcpyHostToDevice), "w");
        poison_out();
        PairRef ref;
        chain_A(m, lib, 0, y, ids, wA, rows, outA, s, st, &ref);
        moe::MoeView v = view(rows, 0, false);
        v.debug_gate_up = dbg_gu;
        moe::enqueue_experts(v, st);
        hip_ok(hipDeviceSynchronize(), "B");
        compare_pipe(v, false, rows * (K + 1), tag);
        const auto a = d2h(outA, 16ULL * H), b = d2h(outB, 16ULL * H);
        std::size_t first = 0;
        const bool ok = same_bits(a, b, &first);  // includes the untouched (NaN-bit-identical) padding rows
        bool pairs_ok = true;
        const std::string where = compare_pairs(ref, pairs_ok);
        std::string msg = tag + ": R" + std::to_string(rows) + " distinct " + std::to_string(distinct(ids)) + " out bitwise";
        if (!ok)
            msg += " (first diff at " + std::to_string(first) + ": A " + std::to_string(a[first]) + " B " + std::to_string(b[first]) + ")";
        msg += ", pair outputs" + std::string(pairs_ok ? " bitwise" : " DIFFER" + where);
        check(ok && pairs_ok, msg);
        if (outB_h) *outB_h = b;
        return ok && pairs_ok;
    }
    // enqueue_moe (router + experts) vs dev::dense + dev::route + chain_A on the reference ids.
    bool test_router(unsigned rows, const std::string& tag, std::vector<float>* outB_h = nullptr) {
        poison_out();
        tier::dense(y, m.router_w_dev, tier::WType::F16, H, 0, E, logitsA, rows, st);
        tier::route(logitsA, m.bias_dev, idsA, wA, rows, st);
        hip_ok(hipDeviceSynchronize(), "route A");
        const auto ids = d2h(idsA, static_cast<std::size_t>(rows) * K);
        PairRef ref;
        chain_A(m, lib, 0, y, ids, wA, rows, outA, s, st, &ref);
        moe::MoeView v = view(rows, 0, true);
        v.debug_gate_up = dbg_gu;
        moe::enqueue_moe(v, st);
        hip_ok(hipDeviceSynchronize(), "B");
        compare_pipe(v, true, rows * (K + 1), tag);
        const bool lg = same_bits(d2h(logitsA, rows * E), d2h(logitsB, rows * E));
        const bool id = same_bits(ids, d2h(idsB, rows * K));
        const bool wt = same_bits(d2h(wA, rows * K), d2h(wB, rows * K));
        const auto b = d2h(outB, 16ULL * H);
        const bool out = same_bits(d2h(outA, 16ULL * H), b);
        bool pairs_ok = true;
        const std::string where = id ? compare_pairs(ref, pairs_ok) : std::string(" (ids differ)");
        if (!id) pairs_ok = false;
        check(lg && id && wt && out && pairs_ok,
              tag + ": R" + std::to_string(rows) + " router logits " + (lg ? "==" : "DIFFER") + ", ids " + (id ? "==" : "DIFFER") +
                  ", weights " + (wt ? "==" : "DIFFER") + ", out " + (out ? "bitwise" : "DIFFER") + ", pair outputs" +
                  (pairs_ok ? " bitwise" : " DIFFER" + where));
        if (outB_h) *outB_h = b;
        return lg && id && wt && out && pairs_ok;
    }
    // Canonical per-row ordering: shared, then that row's ascending expert ids.
    // Compare bytes (including all 32 split partials), rather than a hash.
    std::vector<std::vector<std::uint8_t>> row_intermediates(unsigned rows, const std::vector<std::uint32_t>& ids) {
        std::vector<std::vector<std::uint8_t>> result(rows);
        for (unsigned r = 0; r < rows; ++r) {
            auto append = [&](const auto& h) {
                const auto* b = reinterpret_cast<const std::uint8_t*>(h.data());
                result[r].insert(result[r].end(), b, b + h.size() * sizeof(h[0]));
            };
            for (unsigned k = 0; k <= K; ++k) {
                unsigned pair = r;
                if (k != 0) {
                    const unsigned e = ids[r * K + k - 1];
                    pair = rows;
                    for (unsigned j = 0; j < rows * K; ++j)
                        pair += ids[j] < e || (ids[j] == e && j / K < r) ? 1U : 0U;
                }
                append(d2h(dbg_gu + static_cast<std::size_t>(pair) * 2 * I, 2 * I));
                append(d2h(moe::moe_pair_gate_partials(ws, pair), 2 * 32 * I));
                append(d2h(moe::moe_pair_down_input(ws, pair), I));
                append(d2h(moe::moe_pair_output(ws, pair), H));
            }
        }
        return result;
    }
    // Rows of y8 (host, 8 rows) picked by `sel` into y (rest NaN).
    void load_rows(const std::vector<float>& y8, const std::vector<unsigned>& sel) {
        std::vector<float> h(16ULL * H, std::nanf(""));
        for (std::size_t i = 0; i < sel.size(); ++i) std::memcpy(h.data() + i * H, y8.data() + sel[i] * H, H * 4);
        hip_ok(hipMemcpy(y, h.data(), h.size() * 4, hipMemcpyHostToDevice), "rows");
    }
    // Invariance: a subset / permutation of the rows of an 8-row call, through enqueue_experts (ids, weights taken
    // along) or enqueue_moe (router), gives each row the same bits as inside the 8-row call.
    bool sub_call(const std::vector<float>& y8, const std::vector<std::uint32_t>& ids8, const std::vector<float>& w8,
                  const std::vector<float>& out8, const std::vector<std::vector<std::uint8_t>>& mids8,
                  const std::vector<unsigned>& sel, bool router) {
        load_rows(y8, sel);
        poison_out();
        const auto n = static_cast<unsigned>(sel.size());
        if (!router) {
            std::vector<std::uint32_t> ids;
            std::vector<float> w;
            for (unsigned r : sel) {
                ids.insert(ids.end(), ids8.begin() + r * K, ids8.begin() + (r + 1) * K);
                w.insert(w.end(), w8.begin() + r * K, w8.begin() + (r + 1) * K);
            }
            hip_ok(hipMemcpy(idsA, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
            hip_ok(hipMemcpy(wA, w.data(), w.size() * 4, hipMemcpyHostToDevice), "w");
            auto v = view(n, 0, false);
            v.debug_gate_up = dbg_gu;
            moe::enqueue_experts_pipe(v, st);
        } else {
            auto v = view(n, 0, true);
            v.debug_gate_up = dbg_gu;
            moe::enqueue_moe_pipe(v, st);
        }
        hip_ok(hipDeviceSynchronize(), "sub");
        const auto b = d2h(outB, static_cast<std::size_t>(n) * H);
        const auto mids = row_intermediates(n, d2h(router ? idsB : idsA, n * K));
        bool ok = true;
        for (unsigned i = 0; i < n; ++i)
            ok = same_bits(mids[i], mids8[sel[i]]) && std::memcmp(b.data() + i * H, out8.data() + sel[i] * H, H * 4) == 0 && ok;
        return ok;
    }
    void invariance(const std::string& tag, bool router, int pattern = 0) {
        const std::string kind = router ? "enqueue_moe" : "enqueue_experts";
        set_y(8, router ? 920 : 900);
        const auto y8 = d2h(y, 8ULL * H);
        std::vector<float> out8;
        std::vector<std::uint32_t> ids8;
        std::vector<float> w8;
        if (router) {
            test_router(8, tag + " invariance base " + kind, &out8);
            ids8 = d2h(idsB, 8ULL * K);
            w8 = d2h(wB, 8ULL * K);
        } else {
            ids8 = make_ids_ext(8, pattern, 901);
            test_experts(8, ids8, tag + " invariance base " + kind, &out8);
            w8 = make_weights(ids8, 99 + 8);
        }
        bool ok = true;
        const auto mids8 = row_intermediates(8, ids8);
        for (unsigned r = 0; r < 8; ++r) ok = sub_call(y8, ids8, w8, out8, mids8, {r}, router) && ok;
        ok = sub_call(y8, ids8, w8, out8, mids8, {2, 3, 4}, router) && ok;
        ok = sub_call(y8, ids8, w8, out8, mids8, {0, 1, 2, 3, 4}, router) && ok;
        ok = sub_call(y8, ids8, w8, out8, mids8, {7, 6, 5, 4, 3, 2, 1, 0}, router) && ok;
        ok = sub_call(y8, ids8, w8, out8, mids8, {5, 0, 7, 2}, router) && ok;
        for (unsigned n = 1; n <= 8; ++n) {
            std::vector<unsigned> sel;
            for (unsigned r = 0; r < n; ++r) sel.push_back((r + 3) % 8);
            ok = sub_call(y8, ids8, w8, out8, mids8, sel, router) && ok;
        }
        check(ok, tag + " batch invariance " + kind + ": R1 x8, R3, R5, reversed R8, permuted R1..8 rows: all intermediates and outputs == inside R8");
    }
    // Back-to-back: three different calls queued without host synchronisation and without clearing the workspace
    // (counters must reset themselves), each into its own output; each must equal the same call run alone.
    // Back-to-back: three different calls queued x3 without host synchronisation and without clearing the workspace
    // (counters must reset themselves); every launch writes its own NaN-poisoned output, and all nine must equal the
    // same call run alone.
    void back_to_back(const std::string& tag) {
        struct Call {
            unsigned rows;
            bool router;
            std::vector<float> y;
            std::vector<std::uint32_t> ids;
            std::vector<float> w, alone;
            float* yd;
            std::uint32_t* idsd[3];
            float* wd[3];
            float* out[3];
        };
        std::vector<Call> calls(3);
        const unsigned rows[3] = {8, 1, 5};
        const std::vector<float> nan16(16ULL * H, std::nanf(""));
        for (unsigned c = 0; c < 3; ++c) {
            Call& k = calls[c];
            k.rows = rows[c];
            k.router = c != 1;
            k.y = make_y(k.rows, 960 + c);
            k.y.resize(16ULL * H, std::nanf(""));
            k.ids = make_ids(k.rows, c == 2 ? 1 : 0, 970 + c);
            k.w = make_weights(k.ids, 980 + c);
            k.yd = dev(k.y);
            for (unsigned r = 0; r < 3; ++r) {  // own ids / weights / output per repetition
                k.idsd[r] = dev(k.ids);
                k.wd[r] = dev(k.w);
                k.out[r] = dev(nan16);
            }
        }
        auto issue = [&](Call& k, unsigned r) {
            moe::MoeView v = view(k.rows, 0, k.router);
            v.y = k.yd;
            v.ids = k.idsd[r];
            v.weights = k.wd[r];
            v.logits = nullptr;
            v.out = k.out[r];
            if (k.router) moe::enqueue_moe_pipe(v, st);
            else moe::enqueue_experts_pipe(v, st);
        };
        bool base_ok = true;
        for (auto& k : calls) {  // alone (repetition slot 0), validated against the reference chain A, then re-poisoned
            issue(k, 0);
            hip_ok(hipDeviceSynchronize(), "alone");
            k.alone = d2h(k.out[0], static_cast<std::size_t>(k.rows) * H);
            std::vector<std::uint32_t> ids = k.ids;
            const float* wref = k.wd[0];
            if (k.router) {  // the reference routing of this y
                tier::dense(k.yd, m.router_w_dev, tier::WType::F16, H, 0, E, logitsA, k.rows, st);
                tier::route(logitsA, m.bias_dev, idsA, wA, k.rows, st);
                hip_ok(hipDeviceSynchronize(), "route ref");
                ids = d2h(idsA, static_cast<std::size_t>(k.rows) * K);
                wref = wA;
            }
            hip_ok(hipMemcpy(outA, nan16.data(), nan16.size() * 4, hipMemcpyHostToDevice), "poison A");
            chain_A(m, lib, 0, k.yd, ids, wref, k.rows, outA, s, st);
            hip_ok(hipDeviceSynchronize(), "ref");
            const auto ref = d2h(outA, static_cast<std::size_t>(k.rows) * H);
            bool finite = true;
            for (float v : ref) finite = finite && std::isfinite(v);
            base_ok = base_ok && finite && same_bits(ref, k.alone);
            hip_ok(hipMemcpy(k.out[0], nan16.data(), nan16.size() * 4, hipMemcpyHostToDevice), "poison");
            hip_ok(hipMemcpy(k.idsd[0], k.ids.data(), k.ids.size() * 4, hipMemcpyHostToDevice), "ids");
            hip_ok(hipMemcpy(k.wd[0], k.w.data(), k.w.size() * 4, hipMemcpyHostToDevice), "w");
        }
        for (unsigned r = 0; r < 3; ++r)
            for (auto& k : calls) issue(k, r);
        hip_ok(hipDeviceSynchronize(), "b2b");
        bool ok = true;
        for (auto& k : calls)
            for (unsigned r = 0; r < 3; ++r) {
                const auto o = d2h(k.out[r], 16ULL * H);
                ok = ok && std::memcmp(o.data(), k.alone.data(), k.alone.size() * 4) == 0 &&
                     std::memcmp(o.data() + k.alone.size(), nan16.data(), (16ULL * H - k.alone.size()) * 4) == 0;
            }
        check(ok && base_ok, tag + " back-to-back: R8 moe, R1 experts, R5 moe queued x3 without sync == each alone");
        if (!base_ok) std::printf("     (back-to-back: a solo baseline differs from chain A or is non-finite)\n");
        for (auto& k : calls) {
            (void)hipFree(k.yd);
            for (unsigned r = 0; r < 3; ++r) {
                (void)hipFree(k.idsd[r]);
                (void)hipFree(k.wd[r]);
                (void)hipFree(k.out[r]);
            }
        }
    }

    void run_tests(const std::string& tag) {
        for (unsigned rows = 1; rows <= 8; ++rows) {
            set_y(rows, 500 + rows);
            for (int p = 0; p < kPatterns; ++p)
                test_experts(rows, make_ids_ext(rows, p, 40U + rows * 7 + static_cast<unsigned>(p)), tag + " experts " + pattern_name(p));
            test_router(rows, tag + " enqueue_moe");
        }
        // Edge inputs: zeros, -0, subnormals, +-1e4 (clamp / saturation), FP16-subnormal-sized activations.
        for (unsigned rows : {1U, 3U, 8U}) {
            set_y(rows, 600 + rows, 1);
            test_experts(rows, make_ids_ext(rows, 0, 610 + rows), tag + " edge inputs experts");
            test_router(rows, tag + " edge inputs enqueue_moe");
        }
        for (int p = 0; p < kPatterns; ++p) invariance(tag + " " + pattern_name(p), false, p);
        invariance(tag, true);
        back_to_back(tag);
        // Router ties and saturation.
        const auto keep_w = m.router_w;
        const auto keep_b = m.bias;
        std::vector<std::uint16_t> zero(m.router_w.size(), 0);
        hip_ok(hipMemcpy(m.router_w_dev, zero.data(), zero.size() * 2, hipMemcpyHostToDevice), "zero router");
        std::vector<float> tb(E, 0.25F);
        for (unsigned e = 0; e < E; e += 37) tb[e] = 0.5F;  // a few winners, the rest tie
        hip_ok(hipMemcpy(m.bias_dev, tb.data(), E * 4, hipMemcpyHostToDevice), "tie bias");
        set_y(8, 950);
        test_router(8, tag + " router ties (zero router, tied bias)");
        std::vector<float> eq(E, 0.0F);
        hip_ok(hipMemcpy(m.bias_dev, eq.data(), E * 4, hipMemcpyHostToDevice), "eq bias");
        test_router(3, tag + " router all tied (ids 0..7)");
        const auto ids_t = d2h(idsB, 3 * K);
        bool low = true;
        for (unsigned r = 0; r < 3; ++r)
            for (unsigned k = 0; k < K; ++k) low = low && ids_t[r * K + k] == k;
        check(low, tag + " router all tied -> ids 0..7 for every row");
        // Saturation: huge router weights -> |logit| >> 100, s saturates to exactly +0 or 1 (declared_exp thresholds):
        // keys tie inside each saturated group, the bias and then the lower id decide; bias -0 / +0 mixed (s + bias
        // never yields -0 since s >= +0, so -0 cannot reach a key here; noted, not testable through the router).
        std::vector<std::uint16_t> big(m.router_w.size());
        std::uint64_t sd = 4242;
        for (auto& w : big) w = static_cast<std::uint16_t>((splitmix(sd) & 1 ? 0xF800U : 0x7800U) | (splitmix(sd) & 0x3ffU));
        hip_ok(hipMemcpy(m.router_w_dev, big.data(), big.size() * 2, hipMemcpyHostToDevice), "big router");
        std::vector<float> zb(E, 0.0F);
        for (unsigned e = 0; e < E; e += 2) zb[e] = -0.0F;
        hip_ok(hipMemcpy(m.bias_dev, zb.data(), E * 4, hipMemcpyHostToDevice), "zero bias");
        set_y(5, 955);
        test_router(5, tag + " router saturation (|logit| huge, bias +-0)");
        hip_ok(hipMemcpy(m.router_w_dev, keep_w.data(), keep_w.size() * 2, hipMemcpyHostToDevice), "restore");
        hip_ok(hipMemcpy(m.bias_dev, keep_b.data(), E * 4, hipMemcpyHostToDevice), "restore");
    }

    // ------------------------------------------------------------ timing
    template <class F> double time_us(F&& fn, unsigned iters, unsigned layers) {
        fn(0);  // warm (code, workspace); weights stay cold through the rotation
        hipEvent_t a, b;
        hip_ok(hipEventCreate(&a), "ev");
        hip_ok(hipEventCreate(&b), "ev");
        hip_ok(hipEventRecord(a, st), "rec");
        for (unsigned i = 0; i < iters; ++i) fn((i + 1) % layers);
        hip_ok(hipEventRecord(b, st), "rec");
        hip_ok(hipEventSynchronize(b), "sync");
        float ms = 0;
        hip_ok(hipEventElapsedTime(&ms, a, b), "elapsed");
        (void)hipEventDestroy(a);
        (void)hipEventDestroy(b);
        return 1000.0 * ms / iters;
    }
    void run_bench(const std::string& tag, unsigned iters, unsigned layers, const std::vector<unsigned>& row_list) {
        const double router_bytes = static_cast<double>(E) * H * 2;
        for (int p : {0, 1}) {
            for (unsigned rows : row_list) {
                // y: random rows ("random"), or row 0 replicated ("same": the router then picks the same 8 experts
                // for every row). The routing is the one the router produces on this y (dev::dense + dev::route),
                // so A, B and the byte counts all cover the same experts.
                set_y(rows, 700 + rows);
                if (p == 1)
                    for (unsigned r = 1; r < rows; ++r) hip_ok(hipMemcpy(y + r * H, y, H * 4, hipMemcpyDeviceToDevice), "same y");
                tier::dense(y, m.router_w_dev, tier::WType::F16, H, 0, E, logitsA, rows, st);
                tier::route(logitsA, m.bias_dev, idsA, wA, rows, st);
                hip_ok(hipDeviceSynchronize(), "route");
                const auto ids = d2h(idsA, static_cast<std::size_t>(rows) * K);
                const unsigned nd = distinct(ids);
                const double exp_bytes = static_cast<double>(nd) * static_cast<double>(m.expert_bytes) + static_cast<double>(m.shared_bytes);
                // A: the reference chain incl. its router; B: enqueue_moe. Ax / Bx: experts only, on the same ids.
                auto A = [&](unsigned l) {
                    tier::dense(y, m.router_w_dev, tier::WType::F16, H, 0, E, logitsA, rows, st);
                    tier::route(logitsA, m.bias_dev, idsA, wA, rows, st);
                    chain_A_timed(l, ids, rows);
                };
                auto B = [&](unsigned l) { moe::enqueue_moe(view(rows, l, true), st); };
                auto Ax = [&](unsigned l) { chain_A_timed(l, ids, rows); };
                auto Bx = [&](unsigned l) { moe::enqueue_experts(view(rows, l, false), st); };
                if (rows == 1 || rows == 8) {
                    auto pipe = [&](unsigned l) { moe::enqueue_experts_pipe(view(rows, l, false), st); };
                    for (unsigned l = 0; l < layers; ++l) { Bx(l); pipe(l); }
                    hip_ok(hipDeviceSynchronize(), "pipe warm");
                    for (unsigned rep = 0; rep < 3; ++rep) {
                        const double old0 = time_us(Bx, iters, layers);
                        const double piped = time_us(pipe, iters, layers);
                        const double old1 = time_us(Bx, iters, layers);
                        std::printf("pipe COLD %s %s R%u rep %u old/new/old %.1f %.1f %.1f us\n",
                                    tag.c_str(), pattern_name(p), rows, rep, old0, piped, old1);
                    }
                }
                for (unsigned l = 0; l < layers; ++l) {  // outside timing: library copies of every layer, code warm
                    A(l);
                    B(l);
                }
                hip_ok(hipDeviceSynchronize(), "warm");
                const bool same_route = same_bits(ids, d2h(idsB, ids.size()));
                double ta[4], tb[3], tax[4], tbx[3];
                ta[0] = time_us(A, iters, layers);
                for (int t = 0; t < 3; ++t) {
                    tb[t] = time_us(B, iters, layers);
                    ta[t + 1] = time_us(A, iters, layers);
                }
                tax[0] = time_us(Ax, iters, layers);
                for (int t = 0; t < 3; ++t) {
                    tbx[t] = time_us(Bx, iters, layers);
                    tax[t + 1] = time_us(Ax, iters, layers);
                }
                std::printf("cell %s %-6s R%u distinct %2u%s | A", tag.c_str(), pattern_name(p), rows, nd, same_route ? "" : " ROUTE-MISMATCH");
                for (double v : ta) std::printf(" %8.1f", v);
                std::printf(" us | B(moe, effective weight GB/s)");
                for (double v : tb) std::printf(" %7.1f", v);
                const double bm = (tb[0] + tb[1] + tb[2]) / 3;
                std::printf(" us %6.1f GB/s %5.1f %% | B/A", (exp_bytes + router_bytes) / bm / 1e3,
                            100.0 * (exp_bytes + router_bytes) / bm / 1e3 / kCeiling);
                for (int t = 0; t < 3; ++t) std::printf(" %.3f", tb[t] / (0.5 * (ta[t] + ta[t + 1])));
                std::printf("\n     experts-only | Ax");
                for (double v : tax) std::printf(" %8.1f", v);
                std::printf(" us | Bx");
                for (double v : tbx) std::printf(" %7.1f", v);
                const double bx = (tbx[0] + tbx[1] + tbx[2]) / 3;
                std::printf(" us %6.1f GB/s %5.1f %% | Bx/Ax", exp_bytes / bx / 1e3, 100.0 * exp_bytes / bx / 1e3 / kCeiling);
                for (int t = 0; t < 3; ++t) std::printf(" %.3f", tbx[t] / (0.5 * (tax[t] + tax[t + 1])));
                std::printf("\n");
                parts(rows, ids, nd, layers);
                if (rows == 1 || rows == 8) prefetch_cell(rows, layers);
                std::fflush(stdout);
            }
        }
        ceiling();
    }
    void chain_A_timed(unsigned layer, const std::vector<std::uint32_t>& ids, unsigned rows) { chain_A(m, lib, layer, y, ids, wA, rows, outA, s, st); }
    // Per-launch parts of B (enqueue_moe marks), mean over iters cold calls.
    void parts(unsigned rows, const std::vector<std::uint32_t>& ids, unsigned nd, unsigned layers) {
        (void)ids;
        const unsigned n = 12;
        std::vector<hipEvent_t> ev(4 * n);
        for (auto& e : ev) hip_ok(hipEventCreate(&e), "ev");
        moe::enqueue_moe(view(rows, 0, true), st);
        for (unsigned i = 0; i < n; ++i) {
            hip_ok(hipEventRecord(ev[4 * i], st), "rec");
            moe::enqueue_moe(view(rows, (i + 1) % layers, true), st, &ev[4 * i + 1]);
            hip_ok(hipEventRecord(ev[4 * i + 3], st), "rec");
        }
        hip_ok(hipDeviceSynchronize(), "sync");
        double t[3] = {0, 0, 0};
        for (unsigned i = 0; i < n; ++i)
            for (unsigned k = 0; k < 3; ++k) {
                float ms = 0;
                hip_ok(hipEventElapsedTime(&ms, ev[4 * i + k], ev[4 * i + k + 1]), "el");
                t[k] += 1000.0 * ms / n;
            }
        for (auto& e : ev) (void)hipEventDestroy(e);
        auto b = [&](unsigned hi) { return static_cast<double>(m.host[hi].bytes()); };
        const double half_shared = b(m.shared_src) + b(m.shared_src + 1);
        const double gu = static_cast<double>(nd) * (b(m.expert_src[0]) + b(m.expert_src[0] + 1));
        const double dn = static_cast<double>(nd) * b(m.expert_src[0] + 2) + b(m.shared_src + 2);
        const double l1 = half_shared + static_cast<double>(E) * H * 2;  // effective weight bytes per launch
        std::printf("     parts: router+shared_gu %.1f us (%.1f GB/s) | routed gate_up %.1f us (%.1f GB/s, %.1f %%) | down+combine %.1f us "
                    "(%.1f GB/s, %.1f %%)\n",
                    t[0], l1 / t[0] / 1e3, t[1], gu / t[1] / 1e3, 100.0 * gu / t[1] / 1e3 / kCeiling, t[2], dn / t[2] / 1e3,
                    100.0 * dn / t[2] / 1e3 / kCeiling);
    }
    // MALL prefetch question: the next layer's router + shared expert are known before its routing, so they could be
    // read while the GPU waits on the TP exchange. Measured: enqueue_moe right after a read pass over that layer's
    // router + shared tensors (event between) vs the cold call.
    void prefetch_cell(unsigned rows, unsigned layers) {
        unsigned* guard = dalloc<unsigned>(1);
        auto touch = [&](unsigned l) {
            const moe::ExpertSlices& sh = m.shared[l];
            const moe::Exl3* mats[3] = {&sh.gate, &sh.up, &sh.down};
            for (const auto* x : mats) {
                read_kernel<<<256, 256, 0, st>>>(reinterpret_cast<const uint4*>(x->trellis),
                                                 static_cast<std::size_t>(x->input / 16) * (x->output / 16) * x->bits * 2, guard);
                read_kernel<<<4, 256, 0, st>>>(reinterpret_cast<const uint4*>(x->suh), x->input * 2 / 16, guard);
                read_kernel<<<4, 256, 0, st>>>(reinterpret_cast<const uint4*>(x->svh), x->output * 2 / 16, guard);
            }
            read_kernel<<<1, 256, 0, st>>>(reinterpret_cast<const uint4*>(m.bias_dev), E * 4 / 16, guard);
            read_kernel<<<256, 256, 0, st>>>(reinterpret_cast<const uint4*>(m.router_w_dev), static_cast<std::size_t>(E) * H * 2 / 16, guard);
        };
        const unsigned n = 12;
        std::vector<hipEvent_t> ev(3 * n);
        for (auto& e : ev) hip_ok(hipEventCreate(&e), "ev");
        moe::enqueue_moe(view(rows, 0, true), st);
        for (unsigned i = 0; i < n; ++i) {
            const unsigned l = (i + 1) % layers;
            hip_ok(hipEventRecord(ev[3 * i], st), "rec");
            touch(l);
            hip_ok(hipEventRecord(ev[3 * i + 1], st), "rec");
            moe::enqueue_moe(view(rows, l, true), st);
            hip_ok(hipEventRecord(ev[3 * i + 2], st), "rec");
        }
        hip_ok(hipDeviceSynchronize(), "sync");
        double tp = 0, tm = 0;
        for (unsigned i = 0; i < n; ++i) {
            float a = 0, b = 0;
            hip_ok(hipEventElapsedTime(&a, ev[3 * i], ev[3 * i + 1]), "el");
            hip_ok(hipEventElapsedTime(&b, ev[3 * i + 1], ev[3 * i + 2]), "el");
            tp += 1000.0 * a / n;
            tm += 1000.0 * b / n;
        }
        for (auto& e : ev) (void)hipEventDestroy(e);
        (void)hipFree(guard);
        std::printf("     prefetch R%u (immediate-warm upper bound, not an overlap measurement): read router+shared %.1f us, then "
                    "enqueue_moe %.1f us (vs the cold calls above)\n", rows, tp, tm);
    }
    // Size-matched streaming read of one routed expert set + shared (R1 random: 8 experts + shared) from a cold buffer.
    void ceiling() {
        const std::size_t bytes = static_cast<std::size_t>(8 * m.expert_bytes + m.shared_bytes) + static_cast<std::size_t>(E) * H * 2;
        const unsigned copies = 8;
        std::vector<uint4*> bufs(copies);
        for (auto& b : bufs) {
            hip_ok(hipMalloc(&b, bytes), "ceil");
            hip_ok(hipMemset(b, 1, bytes), "ceil");
        }
        unsigned* guard = dalloc<unsigned>(1);
        auto R = [&](unsigned l) { read_kernel<<<1024, 256, 0, st>>>(bufs[l % copies], bytes / 16, guard); };
        const double us = time_us(R, 24, copies);
        const double db = static_cast<double>(bytes);
        std::printf("ceiling: streaming read of %.1f MB (R1 layer bytes: 8 experts + shared + router) %.1f us = %.1f GB/s (%.1f %% of %.1f)\n", db / 1e6, us,
                    db / us / 1e3, 100.0 * db / us / 1e3 / kCeiling, kCeiling);
        for (auto b : bufs) (void)hipFree(b);
    }
};


}  // namespace

int main(int argc, char** argv) {
    bool test = false, bench = false;
    std::string real, router;
    unsigned rank = 0, iters = 20, layers = 3, layout = 0;
    std::vector<unsigned> row_list = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--test") test = true;
        else if (a == "--bench") bench = true;
        else if (a == "--real") real = next();
        else if (a == "--rank") rank = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--router") router = next();
        else if (a == "--iters") iters = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--layers") layers = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--layout") layout = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--rows") {  // comma list, e.g. 1,4,8
            row_list.clear();
            std::string v = next();
            for (std::size_t at = 0; at < v.size();) {
                const std::size_t c = v.find(',', at);
                row_list.push_back(static_cast<unsigned>(std::stoul(v.substr(at, c - at))));
                at = c == std::string::npos ? v.size() : c + 1;
            }
            for (unsigned r : row_list)
                if (r < 1 || r > 8) {
                    std::fprintf(stderr, "--rows values must be 1..8\n");
                    return 2;
                }
        }
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    const std::string tag = real.empty() ? "syn" : "real-r" + std::to_string(rank);
    size_t free_b = 0, total_b = 0;
    hip_ok(hipMemGetInfo(&free_b, &total_b), "meminfo");
    std::printf("device memory free %.1f GiB of %.1f GiB; workspace %.1f MiB; layers %u\n", static_cast<double>(free_b) / 1073741824.0,
                static_cast<double>(total_b) / 1073741824.0, static_cast<double>(moe::moe_workspace_bytes()) / 1048576.0, layers);
    Bench b;
    std::printf("layout %u\n", layout);
    b.init(real, rank, router, std::max(1U, layers), layout);
    std::printf("expert bytes %.3f MB, shared %.3f MB, router %.3f MB\n", static_cast<double>(b.m.expert_bytes) / 1e6,
                static_cast<double>(b.m.shared_bytes) / 1e6,
                static_cast<double>(E) * H * 2 / 1e6);
    if (test) b.run_tests(tag);
    if (bench) {
        b.run_bench(tag, iters, std::max(1U, layers), row_list);
        std::printf("bench done\n");
    }
    std::printf("checks failed: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
