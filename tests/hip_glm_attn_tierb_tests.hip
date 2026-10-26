// attention: gate + bench of the fused attention Tier B dispatches (src/glm/attn/glm_attn_tierb.hpp).
//   --test : (kernel-only, synthetic) per seed (3): every group, R = 1..8, TPW 4/2 x SG 1/2/4 x prime 0/1, two calls each,
//            partials poisoned and workspace guard zones checked around every call: each member's output
//            == strix_gemv (strix-trellis 5acf090) on the same member, bit for bit; pads and rows >= R untouched
//            (sentinels); inactive input rows poisoned with NaN; arrival counters zero after every call. Inputs rotate
//            over three modes: mixed (incl. -0, subnormal, x1e3), RN32 -> RN16 double-rounding ties against member 0's
//            suh, large alternating-sign values (split-boundary cancellation, finite). suh = random FP16 (not powers of
//            two). Then a 32-call no-sync burst per group (alternating inputs), a mixed reuse sequence (random group / R /
//            knobs, back-to-back groups on one workspace) and the DSA front end through both norms (q_a, norm, q_b, kv_a,
//            norm, wq_b vs fused a, norm, norm, fused b).
//   --bench: cold (>= 640 MB of weight sets per group, each set touched once per sweep), R in --rows (default 1..8):
//            stage 1 screens every spill-free knob combination (12) once; stage 2 confirms the fastest (S) against
//            A = the engine's library sequence (strix_gemv per member; KDA q|k|v + the three [n][3C] packs) and the
//            default B: A S B x3 + A, paired d per rep, sweep median [min, max]. F = size-matched streaming read (one
//            launch per set, every member array's bytes); A0 = qkv GEMVs without packs (the historical measurement boundary).
//            Then the DSA front end through both norms with the selected knobs: A, B, and the hybrids Ha / Hb.
// usage: hip_glm_attn_tierb_tests (--test | --bench) [--rows 1,2,...] [--seeds N]
#include "glm/attn/glm_attn_tierb.hpp"
#include "glm/kernels/glm_tier_a.hpp"

#include <strix_trellis.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace T = strix::glm::attn_tierb;
namespace D = strix::glm::dev;

namespace {

void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
void sok(strix_status s, const char* w) {
    if (s != STRIX_SUCCESS) {
        std::fprintf(stderr, "%s: %s\n", w, strix_status_string(s));
        std::exit(4);
    }
}
constexpr double kCeilGBs = 246.7;
constexpr std::uint32_t kSentinel = 0x7fbadbadU;
constexpr float kEps = 1e-5F;

template <class P> P* dalloc(std::size_t n) {
    P* p = nullptr;
    hok(hipMalloc(&p, n * sizeof(P)), "malloc");
    return p;
}
float f32(std::uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000U) << 16U, e = (h >> 10U) & 0x1fU, m = h & 0x3ffU;
    if (e == 0) return (sign ? -1.0F : 1.0F) * static_cast<float>(m) * 5.9604644775390625e-8F;
    return f32(sign | ((e + 112U) << 23U) | (m << 13U));
}

// One K6 matrix: the library's prepared weights, the raw device copy (file layout), host suh.
struct Matrix {
    unsigned in = 0, out = 0;
    strix_weights lib{};
    T::Mat raw{};
    std::uint64_t bytes = 0;  // trellis + suh + svh
    std::vector<std::uint16_t> hsuh;
};

Matrix make_matrix(strix_context ctx, const strix_stream* st, unsigned in, unsigned out, std::mt19937& g) {
    Matrix m;
    m.in = in;
    m.out = out;
    const std::uint64_t tb = static_cast<std::uint64_t>(in) * out * 6 / 8;
    std::vector<std::uint32_t> trellis(tb / 4);
    std::vector<std::uint16_t> svh(out);
    m.hsuh.resize(in);
    for (auto& w : trellis) w = static_cast<std::uint32_t>(g());
    for (auto& v : m.hsuh) v = static_cast<std::uint16_t>((g() & 0x8000U) | ((13U + g() % 5U) << 10U) | (g() & 0x3ffU));  // +-[0.25, 8)
    for (auto& v : svh) v = static_cast<std::uint16_t>(0x2000U + (g() & 0x7ffU));
    strix_exl3_parts_desc pd{};
    pd.struct_size = sizeof(pd);
    pd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    pd.transport_version = 1;
    pd.bits_x2 = 12;
    pd.rows = out;
    pd.columns = in;
    pd.codebook = STRIX_CODEBOOK_MUL1;
    pd.marker = STRIX_EXL3_MARKER_MUL1;
    pd.trellis = {trellis.data(), tb};
    pd.suh = {m.hsuh.data(), m.hsuh.size() * 2};
    pd.svh = {svh.data(), svh.size() * 2};
    sok(strix_weights_prepare_parts(ctx, &pd, st, &m.lib), "prepare");
    auto* t = dalloc<std::uint32_t>(tb / 4);
    auto* su = dalloc<std::uint16_t>(in);
    auto* sv = dalloc<std::uint16_t>(out);
    hok(hipMemcpy(t, trellis.data(), tb, hipMemcpyHostToDevice), "copy");
    hok(hipMemcpy(su, m.hsuh.data(), in * 2U, hipMemcpyHostToDevice), "copy");
    hok(hipMemcpy(sv, svh.data(), out * 2U, hipMemcpyHostToDevice), "copy");
    m.raw = T::Mat{t, su, sv, in, out, 6};
    m.bytes = tb + in * 2U + out * 2U;
    return m;
}
void free_matrix(Matrix& m) {
    if (m.lib) sok(strix_weights_destroy(m.lib), "destroy");
    for (const void* p : {static_cast<const void*>(m.raw.trellis), static_cast<const void*>(m.raw.suh), static_cast<const void*>(m.raw.svh)})
        hok(hipFree(const_cast<void*>(p)), "free");
    m = Matrix{};
}

strix_tensor_desc tensor(float* p, unsigned rows, unsigned cols) {
    strix_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.abi_version = STRIX_TRELLIS_ABI_VERSION;
    t.data = p;
    t.buffer_bytes = static_cast<std::uint64_t>(rows) * cols * 4;
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.strides[0] = cols;
    t.strides[1] = 1;
    return t;
}

// The engine's Tier B call (glm_lib.hpp Lib::run): contiguous [rows][in] -> [rows][out].
struct Lib {
    strix_context ctx{};
    void* ws = nullptr;
    std::uint64_t ws_bytes = 0;
    void gemv(const Matrix& m, const float* x, float* y, unsigned rows, hipStream_t s) {
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = m.lib;
        d.input = tensor(const_cast<float*>(x), rows, m.in);
        d.output = tensor(y, rows, m.out);
        std::uint64_t b = 0;
        sok(strix_matmul_workspace_bytes(ctx, &d, &b), "ws");
        if (b > ws_bytes) {
            hok(hipDeviceSynchronize(), "sync");
            if (ws) hok(hipFree(ws), "free");
            hok(hipMalloc(&ws, b), "ws");
            ws_bytes = b;
        }
        d.workspace = ws;
        d.workspace_bytes = ws_bytes;
        strix_stream st{sizeof(strix_stream), STRIX_TRELLIS_ABI_VERSION, s, {}};
        sok(strix_gemv(ctx, &d, &st), "gemv");
    }
};

struct GroupDef {
    const char* name;
    T::Group g;
    unsigned members;
    unsigned in[3], out[3];
    unsigned per_token;  // calls per token (layers): 34 KDA, 11 DSA
};
const GroupDef kGroups[] = {
    {"kda.qkv", T::Group::KdaQkv, 3, {4096, 4096, 4096}, {4096, 4096, 4096}, 34},
    {"kda.o", T::Group::KdaO, 1, {4096}, {4096}, 34},
    {"dsa.a", T::Group::DsaA, 2, {4096, 4096}, {1536, 512}, 11},
    {"dsa.b", T::Group::DsaB, 2, {1536, 1536}, {8192, 4096}, 11},
    {"dsa.o", T::Group::DsaO, 1, {8192}, {4096}, 11},
};
constexpr unsigned kNumGroups = 5;

// Output layout: KDA qkv packed [rows][3 * 4096 + pad] (the engine's [rows][3C] at pad 0); others one buffer per member
// [rows][out + pad]. The gate uses pad 64 (sentinel columns), the bench the engine's pad 0.
struct Outs {
    float* y[3] = {};
    unsigned stride[3] = {};
};
Outs out_layout(const GroupDef& gd, float* base, unsigned pad) {
    Outs o;
    if (gd.g == T::Group::KdaQkv) {
        for (unsigned p = 0; p < 3; ++p) {
            o.y[p] = base + p * 4096U;
            o.stride[p] = 3 * 4096 + pad;
        }
    } else {
        std::size_t off = 0;
        for (unsigned p = 0; p < gd.members; ++p) {
            o.y[p] = base + off;
            o.stride[p] = gd.out[p] + pad;
            off += static_cast<std::size_t>(T::kMaxRows) * o.stride[p];
        }
    }
    return o;
}
std::size_t out_floats(const GroupDef& gd, unsigned pad) {
    if (gd.g == T::Group::KdaQkv) return static_cast<std::size_t>(T::kMaxRows) * (3 * 4096 + pad);
    std::size_t n = 0;
    for (unsigned p = 0; p < gd.members; ++p) n += static_cast<std::size_t>(T::kMaxRows) * (gd.out[p] + pad);
    return n;
}

void fused(const GroupDef& gd, const Matrix* m, const float* x, unsigned rows, const Outs& o, void* ws, hipStream_t s, T::Knobs k) {
    const unsigned xs = gd.in[0];
    switch (gd.g) {
    case T::Group::KdaQkv: {
        const T::Mat mm[3] = {m[0].raw, m[1].raw, m[2].raw};
        T::kda_qkv(mm, x, xs, rows, o.y[0], o.stride[0], ws, s, k);
        break;
    }
    case T::Group::KdaO: T::kda_o(m[0].raw, x, xs, rows, o.y[0], o.stride[0], ws, s, k); break;
    case T::Group::DsaA: T::dsa_a(m[0].raw, m[1].raw, x, xs, rows, o.y[0], o.stride[0], o.y[1], o.stride[1], ws, s, k); break;
    case T::Group::DsaB: T::dsa_b(m[0].raw, m[1].raw, x, xs, rows, o.y[0], o.stride[0], o.y[1], o.stride[1], ws, s, k); break;
    case T::Group::DsaO: T::dsa_o(m[0].raw, x, xs, rows, o.y[0], o.stride[0], ws, s, k); break;
    }
}
// Arm A: the engine today (FastAttn::kda / dsa with Lib::run): strix_gemv per member; for KDA q|k|v into a [3][rows][C]
// scratch, then the three hipMemcpy2DAsync packs into [rows][3C]. packs = false: arm A0 (GEMVs only).
void library(Lib& lib, const GroupDef& gd, const Matrix* m, const float* x, unsigned rows, const Outs& o, float* scratch, hipStream_t s,
             bool packs = true) {
    if (gd.g == T::Group::KdaQkv) {
        for (unsigned p = 0; p < 3; ++p) {
            float* sp = scratch + static_cast<std::size_t>(p) * rows * 4096U;
            lib.gemv(m[p], x, sp, rows, s);
            if (packs)
                hok(hipMemcpy2DAsync(o.y[p], o.stride[p] * sizeof(float), sp, 4096 * sizeof(float), 4096 * sizeof(float), rows,
                                     hipMemcpyDeviceToDevice, s),
                    "pack");
        }
        return;
    }
    for (unsigned p = 0; p < gd.members; ++p) lib.gemv(m[p], x, o.y[p], rows, s);  // pad 0: contiguous
}

// Input rows [0, rows) by mode; rows [rows, 8) poisoned with NaN (never read by either side).
void fill_x(std::vector<float>& hx, unsigned in, unsigned rows, unsigned mode, const std::vector<std::uint16_t>& suh0, std::mt19937& g) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (unsigned r = 0; r < T::kMaxRows; ++r)
        for (unsigned c = 0; c < in; ++c) {
            float& v = hx[static_cast<std::size_t>(r) * in + c];
            if (r >= rows) {
                v = f32(0x7fc00001U);
                continue;
            }
            if (mode == 0) {
                const unsigned k = g() % 64U;
                v = k == 0 ? -0.0F : (k == 1 ? 1.0e-40F * ((g() & 1U) ? 1.0F : -1.0F) : (k == 2 ? nd(g) * 1.0e3F : nd(g)));
            } else if (mode == 1) {
                // x * suh rounds (RN32) onto or next to an FP16 midpoint: the RN32 -> RN16 double rounding boundary.
                const auto h = static_cast<std::uint16_t>((g() & 0x8000U) | ((9U + g() % 12U) << 10U) | (g() & 0x3ffU));
                const double lo = half_to_float(h), hi = half_to_float(static_cast<std::uint16_t>(h + 1U));
                const double mid = 0.5 * (lo + hi);
                v = static_cast<float>(mid / static_cast<double>(half_to_float(suh0[c])));
                const int nudge = static_cast<int>(g() % 5U) - 2;
                for (int i = 0; i < std::abs(nudge); ++i) v = std::nextafter(v, nudge > 0 ? INFINITY : -INFINITY);
            } else {
                v = ((c & 1U) ? -4.0e3F : 4.0e3F) * (1.0F + 1.0e-3F * nd(g));  // large, alternating: cancellation, finite
            }
        }
}

// The fused workspace sits between two guard zones (filled with kGuard, checked after every gate call); before every
// gate call the partial planes are poisoned with 0xFF bytes (NaN), so a missing or late partial write cannot be masked
// by a previous call's identical values.
constexpr std::size_t kGuardBytes = 64 * 1024;
constexpr std::uint8_t kGuard = 0xa5;
struct Ctx {
    strix_context ctx{};
    hipStream_t s{};
    strix_stream st{};
    Lib lib;
    unsigned char* alloc = nullptr;
    void* ws = nullptr;
};
Ctx make_ctx() {
    Ctx c;
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    sok(strix_context_create(&cd, &c.ctx), "context");
    hok(hipStreamCreate(&c.s), "stream");
    c.st = strix_stream{sizeof(strix_stream), STRIX_TRELLIS_ABI_VERSION, c.s, {}};
    c.lib.ctx = c.ctx;
    c.alloc = dalloc<unsigned char>(T::workspace_bytes() + 2 * kGuardBytes);
    hok(hipMemset(c.alloc, kGuard, T::workspace_bytes() + 2 * kGuardBytes), "guard");
    c.ws = c.alloc + kGuardBytes;
    hok(hipMemset(c.ws, 0, T::workspace_bytes()), "ws zero");
    return c;
}
void poison_partials(const Ctx& c) { hok(hipMemsetAsync(c.ws, 0xff, T::workspace_counters_offset(), c.s), "poison"); }
bool counters_zero(const Ctx& c) {
    std::vector<std::uint32_t> h(T::workspace_counters());
    hok(hipMemcpy(h.data(), static_cast<unsigned char*>(c.ws) + T::workspace_counters_offset(), h.size() * 4, hipMemcpyDeviceToHost), "cnt");
    return std::all_of(h.begin(), h.end(), [](std::uint32_t v) { return v == 0; });
}
bool guards_intact(const Ctx& c) {
    std::vector<std::uint8_t> a(kGuardBytes), b(kGuardBytes);
    hok(hipMemcpy(a.data(), c.alloc, kGuardBytes, hipMemcpyDeviceToHost), "g");
    hok(hipMemcpy(b.data(), c.alloc + kGuardBytes + T::workspace_bytes(), kGuardBytes, hipMemcpyDeviceToHost), "g");
    auto ok = [](const std::vector<std::uint8_t>& v) { return std::all_of(v.begin(), v.end(), [](std::uint8_t x) { return x == kGuard; }); };
    return ok(a) && ok(b);
}

// ---------------------------------------------------------------- gate
struct Checker {
    unsigned pass = 0, fail = 0;
    void record(bool ok, const std::string& what) {
        (ok ? pass : fail)++;
        if (!ok) std::printf("GATE FAIL %s\n", what.c_str());
    }
};

// Compare the fused padded layout against contiguous per-member references (host copies).
std::size_t compare(const GroupDef& gd, unsigned rows, const std::vector<std::uint32_t>& ref /* [member][8][out] */, const float* yf_dev,
                    const Outs& o_f, float* yf_base, std::size_t no, std::size_t& written, std::size_t& expect) {
    std::vector<std::uint32_t> hf(no);
    hok(hipMemcpy(hf.data(), yf_dev, no * 4, hipMemcpyDeviceToHost), "back");
    std::size_t bad = 0, roff = 0;
    expect = 0;
    for (unsigned p = 0; p < gd.members; ++p) {
        const auto fo = static_cast<std::size_t>(o_f.y[p] - yf_base);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned n = 0; n < gd.out[p]; ++n)
                bad += ref[roff + static_cast<std::size_t>(r) * gd.out[p] + n] != hf[fo + static_cast<std::size_t>(r) * o_f.stride[p] + n];
        roff += static_cast<std::size_t>(T::kMaxRows) * gd.out[p];
        expect += static_cast<std::size_t>(rows) * gd.out[p];
    }
    written = 0;
    for (std::size_t i = 0; i < no; ++i) written += hf[i] != kSentinel;
    return bad;
}

const T::Knobs kKnobs[] = {{1, 1, 4}, {1, 0, 4}, {2, 1, 4}, {2, 0, 4}, {4, 1, 4}, {4, 0, 4},
                           {1, 1, 2}, {1, 0, 2}, {2, 1, 2}, {2, 0, 2}, {4, 1, 2}, {4, 0, 2}};
constexpr unsigned kNumKnobs = sizeof(kKnobs) / sizeof(kKnobs[0]);
// Gate: the 12 combinations plus the default (Knobs{} = the per-(group, rows) table, attention).
const T::Knobs kGateKnobs[] = {{1, 1, 4}, {1, 0, 4}, {2, 1, 4}, {2, 0, 4}, {4, 1, 4}, {4, 0, 4}, {1, 1, 2},
                               {1, 0, 2}, {2, 1, 2}, {2, 0, 2}, {4, 1, 2}, {4, 0, 2}, {0, -1, 0}};
constexpr unsigned kNumGateKnobs = sizeof(kGateKnobs) / sizeof(kGateKnobs[0]);
constexpr unsigned kBurst = 32;

int run_test(unsigned seeds) {
    Ctx c = make_ctx();
    constexpr unsigned kPad = 64;
    float* x = dalloc<float>(static_cast<std::size_t>(T::kMaxRows) * 8192);
    std::size_t maxo = 0, maxr = 0;
    for (const auto& gd : kGroups) {
        maxo = std::max(maxo, out_floats(gd, kPad));
        std::size_t r = 0;
        for (unsigned p = 0; p < gd.members; ++p) r += static_cast<std::size_t>(T::kMaxRows) * gd.out[p];
        maxr = std::max(maxr, r);
    }
    float* yr = dalloc<float>(maxr);
    float* yf = dalloc<float>(maxo);
    std::vector<float> hx(static_cast<std::size_t>(T::kMaxRows) * 8192);
    const std::vector<std::uint32_t> sent(maxo, kSentinel);
    Checker ck;
    unsigned expected = 0;
    auto reference = [&](const GroupDef& gd, Matrix* m, const float* xin, unsigned rows, std::vector<std::uint32_t>& ref) {
        std::size_t off = 0;
        for (unsigned p = 0; p < gd.members; ++p) {
            c.lib.gemv(m[p], xin, yr + off, rows, c.s);
            off += static_cast<std::size_t>(T::kMaxRows) * gd.out[p];
        }
        hok(hipStreamSynchronize(c.s), "sync");
        ref.resize(off);
        hok(hipMemcpy(ref.data(), yr, off * 4, hipMemcpyDeviceToHost), "ref");
    };
    for (unsigned seed = 0; seed < seeds; ++seed) {
        std::mt19937 g(1000 + seed);
        std::vector<Matrix> mats[kNumGroups];
        for (unsigned gi = 0; gi < kNumGroups; ++gi)
            for (unsigned p = 0; p < kGroups[gi].members; ++p) mats[gi].push_back(make_matrix(c.ctx, &c.st, kGroups[gi].in[p], kGroups[gi].out[p], g));
        // (1) every group x R x knob combination, two calls each, partials poisoned before every call
        for (unsigned gi = 0; gi < kNumGroups; ++gi) {
            const GroupDef& gd = kGroups[gi];
            const Outs o_f = out_layout(gd, yf, kPad);
            const std::size_t no = out_floats(gd, kPad);
            const unsigned group_fail = ck.fail;
            for (unsigned rows = 1; rows <= T::kMaxRows; ++rows) {
                const unsigned mode = (rows + seed + gi) % 3U;
                fill_x(hx, gd.in[0], rows, mode, mats[gi][0].hsuh, g);
                hok(hipMemcpy(x, hx.data(), static_cast<std::size_t>(T::kMaxRows) * gd.in[0] * 4, hipMemcpyHostToDevice), "x");
                std::vector<std::uint32_t> ref;
                reference(gd, mats[gi].data(), x, rows, ref);
                for (const auto& k : kGateKnobs)
                    for (unsigned rep = 0; rep < 2; ++rep) {
                        hok(hipMemcpy(yf, sent.data(), no * 4, hipMemcpyHostToDevice), "sent");
                        poison_partials(c);
                        fused(gd, mats[gi].data(), x, rows, o_f, c.ws, c.s, k);
                        hok(hipStreamSynchronize(c.s), "sync");
                        std::size_t written = 0, expect = 0;
                        const std::size_t bad = compare(gd, rows, ref, yf, o_f, yf, no, written, expect);
                        const bool cz = counters_zero(c), gz = guards_intact(c);
                        char buf[256];
                        std::snprintf(buf, sizeof buf, "%s seed=%u R=%u mode=%u sg=%u prime=%d tpw=%u rep=%u: mismatch %zu, written %zu <= %zu, counters %s, guards %s",
                                      gd.name, seed, rows, mode, k.sg, k.prime, k.tpw, rep, bad, written, expect, cz ? "zero" : "NONZERO",
                                      gz ? "ok" : "TOUCHED");
                        ck.record(bad == 0 && written <= expect && cz && gz, buf);
                        ++expected;
                    }
            }
            std::printf("GATE %s seed=%u: R 1..8 x %u knobs x 2 calls (partials poisoned, guards): %s\n", gd.name, seed, kNumGateKnobs,
                        ck.fail == group_fail ? "PASS" : "FAIL");
        }
        // (2) burst: kBurst back-to-back calls per group without host synchronization, inputs alternating between two
        //     x (refs for both), random R and knobs, each call into its own output buffer; partials poisoned once first.
        {
            const unsigned burst_fail = ck.fail;
            std::vector<float*> ybuf(kBurst);
            for (auto& p : ybuf) p = dalloc<float>(maxo);
            float* x2 = dalloc<float>(static_cast<std::size_t>(T::kMaxRows) * 8192);
            for (unsigned gi = 0; gi < kNumGroups; ++gi) {
                const GroupDef& gd = kGroups[gi];
                const std::size_t no = out_floats(gd, kPad);
                float* xs2[2] = {x, x2};
                std::vector<std::vector<std::uint32_t>> refs(2 * (T::kMaxRows + 1));
                for (unsigned v = 0; v < 2; ++v) {
                    fill_x(hx, gd.in[0], T::kMaxRows, v, mats[gi][0].hsuh, g);
                    hok(hipMemcpy(xs2[v], hx.data(), static_cast<std::size_t>(T::kMaxRows) * gd.in[0] * 4, hipMemcpyHostToDevice), "x");
                    for (unsigned rows = 1; rows <= T::kMaxRows; ++rows) reference(gd, mats[gi].data(), xs2[v], rows, refs[v * (T::kMaxRows + 1) + rows]);
                }
                unsigned rr[kBurst];
                for (unsigned i = 0; i < kBurst; ++i) {
                    rr[i] = 1 + g() % T::kMaxRows;
                    hok(hipMemcpy(ybuf[i], sent.data(), no * 4, hipMemcpyHostToDevice), "sent");
                }
                for (unsigned i = 0; i < kBurst; ++i) {  // poison enqueued between calls (no host sync)
                    poison_partials(c);
                    fused(gd, mats[gi].data(), xs2[i & 1U], rr[i], out_layout(gd, ybuf[i], kPad), c.ws, c.s, kKnobs[g() % kNumKnobs]);
                }
                hok(hipStreamSynchronize(c.s), "sync");
                std::size_t bad = 0, over = 0;
                for (unsigned i = 0; i < kBurst; ++i) {
                    std::size_t w = 0, e = 0;
                    bad += compare(gd, rr[i], refs[(i & 1U) * (T::kMaxRows + 1) + rr[i]], ybuf[i], out_layout(gd, ybuf[i], kPad), ybuf[i], no, w, e);
                    over += w > e;
                }
                const bool cz = counters_zero(c), gz = guards_intact(c);
                ck.record(bad == 0 && over == 0 && cz && gz, std::string("burst ") + gd.name + " mismatch " + std::to_string(bad));
                ++expected;
            }
            std::printf("GATE burst seed=%u (%u back-to-back calls per group, alternating inputs, no host sync): %s\n", seed, kBurst,
                        ck.fail == burst_fail ? "PASS" : "FAIL");
            for (auto* p : ybuf) hok(hipFree(p), "free");
            hok(hipFree(x2), "free");
        }
        // (3) mixed reuse: random group / R / knobs, two different groups back to back before each check
        {
            std::vector<float*> xg(kNumGroups), yg(kNumGroups);
            std::vector<std::vector<std::uint32_t>> refs(kNumGroups * (T::kMaxRows + 1));
            for (unsigned gi = 0; gi < kNumGroups; ++gi) {
                const GroupDef& gd = kGroups[gi];
                xg[gi] = dalloc<float>(static_cast<std::size_t>(T::kMaxRows) * gd.in[0]);
                yg[gi] = dalloc<float>(out_floats(gd, kPad));
                fill_x(hx, gd.in[0], T::kMaxRows, 0, mats[gi][0].hsuh, g);  // all 8 rows valid; R rows used
                hok(hipMemcpy(xg[gi], hx.data(), static_cast<std::size_t>(T::kMaxRows) * gd.in[0] * 4, hipMemcpyHostToDevice), "x");
                for (unsigned rows = 1; rows <= T::kMaxRows; ++rows) reference(gd, mats[gi].data(), xg[gi], rows, refs[gi * (T::kMaxRows + 1) + rows]);
            }
            const unsigned seq_fail = ck.fail;
            for (unsigned it = 0; it < 150; ++it) {
                const unsigned g1 = g() % kNumGroups, g2 = (g1 + 1 + g() % (kNumGroups - 1)) % kNumGroups;
                const unsigned r1 = 1 + g() % T::kMaxRows, r2 = 1 + g() % T::kMaxRows;
                const T::Knobs k1 = kKnobs[g() % kNumKnobs], k2 = kKnobs[g() % kNumKnobs];
                const Outs o1 = out_layout(kGroups[g1], yg[g1], kPad), o2 = out_layout(kGroups[g2], yg[g2], kPad);
                const std::size_t n1 = out_floats(kGroups[g1], kPad), n2 = out_floats(kGroups[g2], kPad);
                hok(hipMemcpy(yg[g1], sent.data(), n1 * 4, hipMemcpyHostToDevice), "sent");
                hok(hipMemcpy(yg[g2], sent.data(), n2 * 4, hipMemcpyHostToDevice), "sent");
                poison_partials(c);
                fused(kGroups[g1], mats[g1].data(), xg[g1], r1, o1, c.ws, c.s, k1);
                poison_partials(c);
                fused(kGroups[g2], mats[g2].data(), xg[g2], r2, o2, c.ws, c.s, k2);
                hok(hipStreamSynchronize(c.s), "sync");
                std::size_t w1 = 0, e1 = 0, w2 = 0, e2 = 0;
                const std::size_t b1 = compare(kGroups[g1], r1, refs[g1 * (T::kMaxRows + 1) + r1], yg[g1], o1, yg[g1], n1, w1, e1);
                const std::size_t b2 = compare(kGroups[g2], r2, refs[g2 * (T::kMaxRows + 1) + r2], yg[g2], o2, yg[g2], n2, w2, e2);
                const bool cz = counters_zero(c) && guards_intact(c);
                char buf[200];
                std::snprintf(buf, sizeof buf, "reuse it=%u %s R=%u + %s R=%u: mismatch %zu + %zu, counters %s", it, kGroups[g1].name, r1,
                              kGroups[g2].name, r2, b1, b2, cz ? "zero" : "NONZERO");
                ck.record(b1 == 0 && b2 == 0 && w1 <= e1 && w2 <= e2 && cz, buf);
                ++expected;
            }
            std::printf("GATE reuse sequence seed=%u (150 pairs): %s\n", seed, ck.fail == seq_fail ? "PASS" : "FAIL");
            for (auto* p : xg) hok(hipFree(p), "free");
            for (auto* p : yg) hok(hipFree(p), "free");
        }
        // (4) DSA front end through both norms (BF16 norm weights), order A vs fused order B
        {
            Matrix& qa = mats[2][0];
            Matrix& kva = mats[2][1];
            Matrix& qb = mats[3][0];
            Matrix& wqb = mats[3][1];
            std::vector<std::uint16_t> hn(1536 + 512);
            for (auto& v : hn) {
                const float f = 1.0F + 0.25F * std::normal_distribution<float>(0.0F, 1.0F)(g);
                std::uint32_t b;
                std::memcpy(&b, &f, 4);
                v = static_cast<std::uint16_t>(b >> 16U);
            }
            auto* nw = dalloc<std::uint16_t>(hn.size());
            hok(hipMemcpy(nw, hn.data(), hn.size() * 2, hipMemcpyHostToDevice), "norm");
            const std::size_t R8 = T::kMaxRows;
            float* buf[2][6];  // qa, qr, q, kva, lat, qI
            const unsigned w6[6] = {1536, 1536, 8192, 512, 512, 4096};
            for (auto& arm : buf)
                for (unsigned i = 0; i < 6; ++i) arm[i] = dalloc<float>(R8 * w6[i]);
            const unsigned front_fail = ck.fail;
            for (unsigned rows = 1; rows <= T::kMaxRows; ++rows) {
                fill_x(hx, 4096, rows, rows % 3U, qa.hsuh, g);
                hok(hipMemcpy(x, hx.data(), R8 * 4096 * 4, hipMemcpyHostToDevice), "x");
                float** A = buf[0];
                float** B = buf[1];
                c.lib.gemv(qa, x, A[0], rows, c.s);
                D::rmsnorm(A[0], nw, D::WType::BF16, 1536, kEps, A[1], rows, c.s);
                c.lib.gemv(qb, A[1], A[2], rows, c.s);
                c.lib.gemv(kva, x, A[3], rows, c.s);
                D::rmsnorm(A[3], nw + 1536, D::WType::BF16, 512, kEps, A[4], rows, c.s);
                c.lib.gemv(wqb, A[1], A[5], rows, c.s);
                poison_partials(c);
                T::dsa_a(qa.raw, kva.raw, x, 4096, rows, B[0], 1536, B[3], 512, c.ws, c.s);
                D::rmsnorm(B[0], nw, D::WType::BF16, 1536, kEps, B[1], rows, c.s);
                D::rmsnorm(B[3], nw + 1536, D::WType::BF16, 512, kEps, B[4], rows, c.s);
                poison_partials(c);
                T::dsa_b(qb.raw, wqb.raw, B[1], 1536, rows, B[2], 8192, B[5], 4096, c.ws, c.s);
                hok(hipStreamSynchronize(c.s), "sync");
                std::size_t bad = 0;
                for (unsigned i = 0; i < 6; ++i) {
                    std::vector<std::uint32_t> ha(rows * w6[i]), hb(rows * w6[i]);
                    hok(hipMemcpy(ha.data(), A[i], ha.size() * 4, hipMemcpyDeviceToHost), "a");
                    hok(hipMemcpy(hb.data(), B[i], hb.size() * 4, hipMemcpyDeviceToHost), "b");
                    for (std::size_t j = 0; j < ha.size(); ++j) bad += ha[j] != hb[j];
                }
                ck.record(bad == 0 && counters_zero(c) && guards_intact(c), "dsa front R=" + std::to_string(rows) + " mismatch " + std::to_string(bad));
                ++expected;
            }
            std::printf("GATE dsa front end (q_a,norm,q_b,kv_a,norm,wq_b == fused a,norm,norm,fused b) seed=%u R 1..8: %s\n", seed,
                        ck.fail == front_fail ? "PASS" : "FAIL");
            for (auto& arm : buf)
                for (auto* p : arm) hok(hipFree(p), "free");
            hok(hipFree(nw), "free");
        }
        for (auto& v : mats)
            for (auto& m : v) free_matrix(m);
        std::fflush(stdout);
    }
    std::printf("GATE scope: kernel-only (synthetic K6 weights; real-slice + engine replay/commit/rollback qualification = the integrator's S-gate)\n");
    std::printf("GATE TOTAL pass=%u fail=%u expected=%u\n", ck.pass, ck.fail, expected);
    std::fflush(stdout);
    return ck.fail == 0 && ck.pass == expected ? 0 : 1;
}

// ---------------------------------------------------------------- bench
__global__ __launch_bounds__(256) void read_floor_kernel(const uint4* p, std::size_t n, std::uint32_t* sink) {
    std::uint32_t acc = 0;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) acc ^= p[i].x ^ p[i].y ^ p[i].z ^ p[i].w;
    if (acc == 0x9e3779b9U) sink[0] = acc;  // observable: the loads cannot be removed
}

std::vector<unsigned> parse_rows(const char* s) {
    std::vector<unsigned> r;
    std::string str(s);
    std::size_t p = 0;
    while (p < str.size()) {
        const std::size_t q = str.find(',', p);
        r.push_back(static_cast<unsigned>(std::stoul(str.substr(p, q == std::string::npos ? std::string::npos : q - p))));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return r;
}

struct Stat {
    double med = 0, lo = 0, hi = 0;
};
struct Timer {
    hipEvent_t e0{}, e1{};
    Timer() {
        hok(hipEventCreate(&e0), "ev");
        hok(hipEventCreate(&e1), "ev");
    }
    // us per set: one warm-up sweep, then 11 sweeps over every set (each set touched once per sweep): median, min, max
    template <class F> Stat run(hipStream_t s, unsigned sets, F&& body) {
        std::vector<double> t;
        for (int rep = -1; rep < 11; ++rep) {
            hok(hipEventRecord(e0, s), "rec");
            for (unsigned c = 0; c < sets; ++c) body(c);
            hok(hipEventRecord(e1, s), "rec");
            hok(hipEventSynchronize(e1), "sync");
            float ms = 0;
            hok(hipEventElapsedTime(&ms, e0, e1), "el");
            if (rep >= 0) t.push_back(ms * 1000.0 / sets);
        }
        std::sort(t.begin(), t.end());
        return Stat{t[t.size() / 2], t.front(), t.back()};
    }
};
std::string knob_name(const T::Knobs& k) {
    return "t" + std::to_string(k.tpw) + "s" + std::to_string(k.sg) + "p" + std::to_string(k.prime);
}
double median3(double a, double b, double c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }

int run_bench(const std::vector<unsigned>& rows_list) {
    Ctx c = make_ctx();
    float* x = dalloc<float>(static_cast<std::size_t>(T::kMaxRows) * 8192);
    std::vector<float> hx(static_cast<std::size_t>(T::kMaxRows) * 8192);
    std::mt19937 g(7);
    {
        const std::vector<std::uint16_t> one(8192, 0x3c00);
        fill_x(hx, 8192, T::kMaxRows, 0, one, g);
    }
    hok(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x");
    std::size_t maxo = 0;
    for (const auto& gd : kGroups) maxo = std::max(maxo, out_floats(gd, 0));
    float* y = dalloc<float>(maxo);
    float* scratch = dalloc<float>(static_cast<std::size_t>(3) * T::kMaxRows * 4096);
    std::uint32_t* sink = dalloc<std::uint32_t>(1);
    Timer tm;
    double tok[2][9] = {}, tok_sel[9] = {};  // per-token ms: [A, B default][rows], selected
    T::Knobs selected[kNumGroups][9] = {};
    std::vector<std::uint16_t> hn(2048, 0x3f80);  // BF16 1.0 norm weights
    auto* nw = dalloc<std::uint16_t>(hn.size());
    hok(hipMemcpy(nw, hn.data(), hn.size() * 2, hipMemcpyHostToDevice), "norm");
    // Order: DSA a, DSA b, then the DSA front end (needs their selections), then DSA o, KDA qkv, KDA o.
    const unsigned order[kNumGroups] = {2, 3, 4, 0, 1};
    for (unsigned oi = 0; oi < kNumGroups; ++oi) {
        const unsigned gi = order[oi];
        const GroupDef& gd = kGroups[gi];
        std::uint64_t set_bytes = 0;
        for (unsigned p = 0; p < gd.members; ++p) set_bytes += static_cast<std::uint64_t>(gd.in[p]) * gd.out[p] * 6 / 8 + gd.in[p] * 2U + gd.out[p] * 2U;
        const unsigned sets = static_cast<unsigned>(std::max<std::uint64_t>(4, ((640ULL << 20) + set_bytes - 1) / set_bytes));
        const auto sb = static_cast<double>(set_bytes);
        std::vector<std::vector<Matrix>> w(sets);
        for (auto& ws1 : w)
            for (unsigned p = 0; p < gd.members; ++p) ws1.push_back(make_matrix(c.ctx, &c.st, gd.in[p], gd.out[p], g));
        std::vector<uint4*> rb(sets);
        const std::size_t rb_n = set_bytes / 16;
        for (auto& p : rb) {
            p = dalloc<uint4>(rb_n);
            hok(hipMemset(p, 0x5a, rb_n * 16), "rb");
        }
        const Outs o = out_layout(gd, y, 0);
        std::printf("BENCH group %s: %u sets x %.3f MB, bytes floor %.2f us/call\n", gd.name, sets, sb / 1e6, sb / kCeilGBs / 1e3);
        for (unsigned rows : rows_list) {
            auto lib_arm = [&](bool packs) {
                return tm.run(c.s, sets, [&](unsigned k) { library(c.lib, gd, w[k].data(), x, rows, o, scratch, c.s, packs); });
            };
            auto fused_arm = [&](T::Knobs kn) { return tm.run(c.s, sets, [&](unsigned k) { fused(gd, w[k].data(), x, rows, o, c.ws, c.s, kn); }); };
            // stage 1: screen every spill-free knob combination once
            T::Knobs best{};
            double best_t = 1e30;
            std::printf("SCREEN %s R=%u", gd.name, rows);
            for (const auto& k : kKnobs) {
                if (T::scratch_bytes(gd.g, rows, k) != 0) {
                    std::printf(" %s=scratch", knob_name(k).c_str());
                    continue;
                }
                const Stat st = fused_arm(k);
                std::printf(" %s=%.2f", knob_name(k).c_str(), st.med);
                if (st.med < best_t) {
                    best_t = st.med;
                    best = k;
                }
            }
            std::printf("\n");
            selected[gi][rows] = best;
            // stage 2: A S B x3 + A, paired d for the selected S and the default B
            Stat a[4], sl[3], b[3];
            for (unsigned rep = 0; rep < 3; ++rep) {
                a[rep] = lib_arm(true);
                sl[rep] = fused_arm(best);
                b[rep] = fused_arm(T::Knobs{});
            }
            a[3] = lib_arm(true);
            const Stat f = tm.run(c.s, sets, [&](unsigned k) { read_floor_kernel<<<640, 256, 0, c.s>>>(rb[k], rb_n, sink); });
            double ds[3], db[3];
            for (unsigned rep = 0; rep < 3; ++rep) {
                ds[rep] = sl[rep].med - 0.5 * (a[rep].med + a[rep + 1].med);
                db[rep] = b[rep].med - 0.5 * (a[rep].med + a[rep + 1].med);
                std::printf("CONFIRM %s R=%u rep=%u A=%.2f[%.2f,%.2f] S(%s)=%.2f[%.2f,%.2f] B=%.2f[%.2f,%.2f]\n", gd.name, rows, rep, a[rep].med, a[rep].lo,
                            a[rep].hi, knob_name(best).c_str(), sl[rep].med, sl[rep].lo, sl[rep].hi, b[rep].med, b[rep].lo, b[rep].hi);
            }
            const double ma = median3(a[0].med, a[1].med, a[2].med), ms = median3(sl[0].med, sl[1].med, sl[2].med),
                         mb = median3(b[0].med, b[1].med, b[2].med);
            std::printf("SUMMARY %s R=%u A=%.2f S=%.2f (%s) B=%.2f F=%.2f us; d(S-A)=%+.2f/%+.2f/%+.2f d(B-A)=%+.2f/%+.2f/%+.2f; S %.1f GB/s = %.1f %% of bytes floor %.2f",
                        gd.name, rows, ma, ms, knob_name(best).c_str(), mb, f.med, ds[0], ds[1], ds[2], db[0], db[1], db[2], sb / ms / 1e3,
                        100.0 * sb / kCeilGBs / 1e3 / ms, sb / kCeilGBs / 1e3);
            if (gd.g == T::Group::KdaQkv) std::printf("; A0=%.2f", lib_arm(false).med);
            std::printf("\n");
            if (rows <= 8) {
                tok[0][rows] += gd.per_token * ma / 1e3;
                tok[1][rows] += gd.per_token * mb / 1e3;
                tok_sel[rows] += gd.per_token * ms / 1e3;
            }
            std::fflush(stdout);
        }
        for (auto& ws1 : w)
            for (auto& m : ws1) free_matrix(m);
        for (auto* p : rb) hok(hipFree(p), "free");
        if (gi != 3) continue;
        // DSA front end with both norms, selected knobs: A = q_a, norm, q_b, kv_a, norm, wq_b (strix_gemv); B = fused a,
        // norm, norm, fused b; Ha = fused a + library b; Hb = library a + fused b.
        const unsigned ins[4] = {4096, 4096, 1536, 1536}, outs[4] = {1536, 512, 8192, 4096};
        std::uint64_t fb = 0;
        for (unsigned p = 0; p < 4; ++p) fb += static_cast<std::uint64_t>(ins[p]) * outs[p] * 6 / 8 + ins[p] * 2U + outs[p] * 2U;
        const unsigned fsets = static_cast<unsigned>(((640ULL << 20) + fb - 1) / fb);
        std::vector<std::vector<Matrix>> fw(fsets);
        for (auto& ws1 : fw)
            for (unsigned p = 0; p < 4; ++p) ws1.push_back(make_matrix(c.ctx, &c.st, ins[p], outs[p], g));
        const unsigned w6[6] = {1536, 1536, 8192, 512, 512, 4096};
        float* f[6];
        for (unsigned i = 0; i < 6; ++i) f[i] = dalloc<float>(static_cast<std::size_t>(T::kMaxRows) * w6[i]);
        for (unsigned rows : rows_list) {
            const T::Knobs ka = selected[2][rows], kb = selected[3][rows];
            auto arm = [&](bool fa, bool fbz) {
                return tm.run(c.s, fsets, [&](unsigned k) {
                    if (fa) {
                        T::dsa_a(fw[k][0].raw, fw[k][1].raw, x, 4096, rows, f[0], 1536, f[3], 512, c.ws, c.s, ka);
                        D::rmsnorm(f[0], nw, D::WType::BF16, 1536, kEps, f[1], rows, c.s);
                        D::rmsnorm(f[3], nw + 1536, D::WType::BF16, 512, kEps, f[4], rows, c.s);
                    } else {
                        c.lib.gemv(fw[k][0], x, f[0], rows, c.s);
                        D::rmsnorm(f[0], nw, D::WType::BF16, 1536, kEps, f[1], rows, c.s);
                        if (!fbz) c.lib.gemv(fw[k][2], f[1], f[2], rows, c.s);  // the engine's order: q_b before kv_a
                        c.lib.gemv(fw[k][1], x, f[3], rows, c.s);
                        D::rmsnorm(f[3], nw + 1536, D::WType::BF16, 512, kEps, f[4], rows, c.s);
                    }
                    if (fbz) {
                        T::dsa_b(fw[k][2].raw, fw[k][3].raw, f[1], 1536, rows, f[2], 8192, f[5], 4096, c.ws, c.s, kb);
                    } else {
                        if (fa) c.lib.gemv(fw[k][2], f[1], f[2], rows, c.s);
                        c.lib.gemv(fw[k][3], f[1], f[5], rows, c.s);
                    }
                }).med;
            };
            double a[4], b[3], ha[3], hb[3];
            for (unsigned rep = 0; rep < 3; ++rep) {
                a[rep] = arm(false, false);
                b[rep] = arm(true, true);
                ha[rep] = arm(true, false);
                hb[rep] = arm(false, true);
            }
            a[3] = arm(false, false);
            auto d = [&](const double* v, unsigned r) { return v[r] - 0.5 * (a[r] + a[r + 1]); };
            std::printf("FRONT dsa R=%u a=%s b=%s: A=%.2f B=%.2f Ha=%.2f Hb=%.2f us; d(B)=%+.2f/%+.2f/%+.2f d(Ha)=%+.2f/%+.2f/%+.2f d(Hb)=%+.2f/%+.2f/%+.2f\n",
                        rows, knob_name(ka).c_str(), knob_name(kb).c_str(), median3(a[0], a[1], a[2]), median3(b[0], b[1], b[2]),
                        median3(ha[0], ha[1], ha[2]), median3(hb[0], hb[1], hb[2]), d(b, 0), d(b, 1), d(b, 2), d(ha, 0), d(ha, 1), d(ha, 2),
                        d(hb, 0), d(hb, 1), d(hb, 2));
            std::fflush(stdout);
        }
        for (auto& ws1 : fw)
            for (auto& m : ws1) free_matrix(m);
        for (auto* p : f) hok(hipFree(p), "free");
    }
    for (unsigned rows : rows_list)
        if (rows <= 8)
            std::printf("TOKEN R=%u tierB lib %.3f ms, fused default %.3f ms, fused selected %.3f ms per rank (34 x (qkv + o) + 11 x (a + b + o))\n", rows,
                        tok[0][rows], tok[1][rows], tok_sel[rows]);
    std::printf("BENCH DONE\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool test = false, bench = false;
    unsigned seeds = 3;
    std::vector<unsigned> rows = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--test") test = true;
        else if (a == "--bench") bench = true;
        else if (a == "--rows" && i + 1 < argc) rows = parse_rows(argv[++i]);
        else if (a == "--seeds" && i + 1 < argc) seeds = static_cast<unsigned>(std::stoul(argv[++i]));
        else {
            std::fprintf(stderr, "usage: %s (--test | --bench) [--rows 1,2,...] [--seeds N]\n", argv[0]);
            return 2;
        }
    }
    int rc = 0;
    if (test) rc = run_test(seeds);
    if (bench && rc == 0) rc = run_bench(rows);
    return rc;
}
