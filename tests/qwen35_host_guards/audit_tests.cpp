#include "strix/qwen35/cuda_engine.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/spec/engine_target.hpp"
#include "../../src/core/dense_linear.hpp"
#include "strix/env_flags.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace cc = strix::cuda_core;
namespace q = strix::qwen35;
namespace core = strix::core;
#define CK(expr) do { if ((expr) != cudaSuccess) throw std::runtime_error("mock CUDA failure"); } while (0)
namespace strix::cuda_core {
DenseArith process_policy = DenseArith::mma3, head_policy = DenseArith::mma3;
DenseArith dense_arith() { return process_policy; }
std::uint32_t nvfp4_split_k(std::uint32_t, std::uint32_t) { return 1; }
#include "audit_big.inc"
cudaError_t dense_big_reserve(std::uint32_t, std::uint32_t, std::uint64_t, int, BigCtx* ctx) {
    CK(bind_device(*ctx));
    return cudaMalloc(&ctx->planes[0], 32);
}
cudaError_t dense_rows(const DenseWeight&, const float*, float*, std::uint32_t, const float*, cudaStream_t, DenseArith a) {
    head_policy = a;
    return cudaSuccess;
}
std::size_t attention_scratch_floats(int, unsigned) { return 8; }
// attention.hpp (CUDA bf16 types) stays out of host builds; the engine Pf state holds e4m3 KV shadows by value.
struct KvShadowE4m3 { std::uint8_t *k8 = nullptr; float* ks = nullptr; std::uint8_t* v8t = nullptr; float* vs = nullptr; };
// gdn.hpp (CUDA __host__ __device__ dims) stays out too: the engine Slot holds the chunk16 workspace by value; the audit only
// sizes it. The resident operand cache is never created here (pf_cache_init below is a stand-in), so attach sees nullptr.
struct GdnChunkWork { float* ws = nullptr; std::size_t floats = 0; };
template <class Dims> std::size_t gdn_chunk_workspace_floats(const Dims&, std::uint32_t rows) { return rows; }
void dense_big_cache_destroy(BigOperandCache*) {}
void dense_big_ctx_attach_cache(BigCtx*, BigOperandCache*) {}
bool dense_big_fast4_supported() { return false; }  // FAST4 is not audited here (generic-build answer); fast8 is, extracted
}
struct Engine {
    using PrefillArith = q::PrefillArith;
    q::EngineOptions opt;
    struct Config {
        unsigned hidden = 16, intermediate = 32, vocab = 8, head_dim = 256, heads = 1, layers = 1;
        unsigned q_gate_width() const { return 32; }
        unsigned kv_width() const { return 16; }
    } c;
    struct Gdn {
        unsigned key_dim = 128, value_dim = 128, value_heads = 16;
        unsigned v_width() const { return 16; }
        unsigned conv_channels() const { return 16; }
    } gd;
    struct LayerW { cc::DenseWeight qkv, gate, beta, alpha, out, q, k, v, o, ffn_gate, ffn_up, ffn_down; };
    std::vector<LayerW> L;
    static constexpr unsigned RM = 32;
    unsigned committed = 0, last_rows = 7, work = 0;
    int ad = 0;
    cudaStream_t s = nullptr;
    cc::DenseWeight head;
    float *xn = nullptr, *logits = nullptr;
    using PrefillGdn = q::PrefillGdn;
    struct CacheDel {  // as engine.cu
        void operator()(cc::BigOperandCache* p) const { cc::dense_big_cache_destroy(p); }
    };
    // engine.cu pf_cache_init plans the resident operand cache on the device (operand_cache_plan); no cache in the audit:
    // dense_big_gemm then expands operands per call, the path the audited fallbacks and admission run on.
    void pf_cache_init() { pf.cache_tried = true; }
#include "audit_engine.inc"
};

// engine.hip WideScratch: the dense scratch plus the x16 tile buffer of the IU4 wide windows (_Float16 there; the audit
// never stores a tile, so a 16-bit stand-in keeps this host test free of _Float16).
using Half = std::uint16_t;
using hipStream_t = void*;
// engine.hip A4W4 class (core/iu4a4_dense.hpp): the activation codes of one window. The audited tensors carry no A4
// codes (a4_ok below is false), so its producers are unreachable here.
namespace strix::core::iu4a4_dense {
struct Act { std::uint8_t* codes = nullptr; float* u = nullptr; float* base = nullptr; unsigned rows = 0, in = 0; };
inline bool map2_supported(unsigned) { return false; }
template <class F> void enqueue_quantise_map2(const float*, const float*, unsigned, const Act&, hipStream_t) { runtime_require(false); }
}
struct WideScratch : core::DenseWideScratch { Half* xt = nullptr; core::iu4a4_dense::Act a4{}; };
struct Tensor { unsigned fmt = 30, inputs = 1, outputs = 1; };
unsigned limit = 0;
float* base_out = nullptr;
std::vector<unsigned> seen;
std::vector<const void*> scratch_seen;
// engine.hip Iu4Scratch.rows: the rows its FP16 scratch holds (16, or the EXACT wide window). The EXACT wide branch takes
// only IU4 weights (kIu4Fmt, type 200 as wide_iu4_ok below); the audited tensors are bf16 / q8_0, so it is not reached here.
struct Iu4Scratch { void *xh = nullptr, *g = nullptr, *u = nullptr; unsigned rows = 16U; };
constexpr unsigned kMaxRows = 16U;
constexpr unsigned kIu4Fmt = 200U;
void linear16(const Tensor&, const float*, float* out, unsigned rows, hipStream_t, const Iu4Scratch&) {
    // The dispatcher computes the 1/2/4/8/16 bucket but stores only `rows` rows (dense_linear::kernel and
    // dense_linear_iu4 `nrows` guards); pad rows are only read, inside the engine's 16-row scratch.
    runtime_require(rows >= 1U && rows <= 16U);
    runtime_require(out >= base_out && out + rows <= base_out + limit);
    seen.push_back(rows);
}
void glu16(const Tensor& g, const Tensor&, const float* x, float* out, unsigned rows, hipStream_t s, const Iu4Scratch& iu4) {
    linear16(g, x, out, rows, s, iu4);
}
void wide_quant(const float*, unsigned, unsigned, hipStream_t, const WideScratch& w) { scratch_seen.push_back(w.codes); }
void wide_dense(const Tensor&, float*, unsigned, hipStream_t, const WideScratch& w) { scratch_seen.push_back(w.codes); }
// The IU4 wide windows take only type-200 weights; the audited tensors are bf16 / q8_0, as in the real predicate.
bool wide_iu4_ok(const Tensor& t, const WideScratch& w, unsigned) { return w.codes && t.fmt == 200U; }
void wide_iu4(const Tensor&, const float*, float*, unsigned, hipStream_t, const Half* = nullptr) { runtime_require(false); }
// x16 tile form of the IU4 wide windows (core/f16_dense_tall.hpp producers); unreachable here (no type-200 weights).
bool wide_iu4_x16(const Tensor&, const WideScratch&, unsigned) { return false; }
struct SiluMul {};
bool a4_ok(const Tensor&, const WideScratch&) { return false; }
void a4_linear(const Tensor&, const float*, float*, unsigned, hipStream_t, const WideScratch&, bool = true) { runtime_require(false); }
namespace strix::core::f16_dense_tall {
struct TileRne {};
void enqueue_x16_tiles(const float*, Half*, unsigned, unsigned, hipStream_t) { runtime_require(false); }
template <class F, class Q = TileRne> void enqueue_x16_tiles_map2(const float*, const float*, Half*, unsigned, unsigned, hipStream_t) { runtime_require(false); }
}
// engine.hip tile quantiser policy of the x16 producers (STRIX_DENSE_FAST4_EMU gate instrument): the default cast only.
template <class Q> struct WithQ { using type = Q; };
template <class K> void with_tile_quantiser(K&& k) { k(WithQ<core::f16_dense_tall::TileRne>{}); }
void x16_tiles(const float*, Half*, unsigned, unsigned, hipStream_t) { runtime_require(false); }
void silu_mul_host(const float*, const float*, float*, std::size_t) {}
#include "audit_hip.inc"
#include "audit_cli.inc"
#include "audit_lengths.inc"
std::map<std::pair<int, int>, unsigned> attributes;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize = 1;
int cudaFuncSetAttribute(int kernel, int, int) { ++attributes[{current_device, kernel}]; return cudaSuccess; }
#include "audit_attributes.inc"
struct OtherEngine {
    enum class LogitRows { all };
    using Rows = int;  // forward_rows' output handle (EngineVerifyTarget observer contract)
    unsigned calls = 0;
    int forward_rows(const std::uint32_t*, std::uint32_t, LogitRows) { ++calls; return 0; }
    void read_top(int, std::uint32_t n, std::uint32_t* ids) { std::fill_n(ids, n, 7U); }
    void commit(std::uint32_t) {}
    void sync() {}
};
struct Bench {
    unsigned pf_bench = 8, R = 16;
    bool pf_twice = false, pf_sample = false, sampled = false;
    std::vector<std::uint32_t> ids = std::vector<std::uint32_t>(8);
    std::string label = "host";
    struct Runner { unsigned prefill_rows() { return 16; } } runner;
    Runner* E = &runner;
    static double now_ms() { static double clock = 0; return ++clock; }
    void prefill_ids(const std::vector<std::uint32_t>& v, std::uint32_t&) { runtime_require(v.size() <= ids.size()); }
#include "audit_bench.inc"
};
int failures = 0;
void check(bool ok, const char* name) { if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", name); } }
template<class F> bool rejects(F&& f) { try { f(); } catch (const std::exception&) { return true; } return false; }
int main() {
    Engine e;
    e.opt.quiet = true;
    std::vector<std::uint32_t> ids(160, 1);
    ids.back() = 8;
    check(rejects([&] { e.admit(ids.data(), 160); }) && e.last_rows == 7 && e.work == 0, "686-1 entire prompt admission before mutation");
    check(rejects([&] { e.admit(nullptr, 160); }), "686-1 null input");
    e.committed = 0xfffffff0U;
    check(rejects([&] { e.admit(ids.data(), 32); }), "686-1 capacity overflow");
    e.committed = 0;
    for (unsigned n = 1; n <= 65; ++n) {
        std::vector<float> v(n); limit = n; base_out = v.data(); seen.clear();
        linear(Tensor{}, v.data(), v.data(), n, nullptr, WideScratch{}, Iu4Scratch{});
        glu(Tensor{}, Tensor{}, v.data(), v.data(), n, nullptr, WideScratch{}, Iu4Scratch{});
        unsigned sum = 0; for (auto count : seen) sum += count;
        check(sum == 2 * n, "686-2 exact logical tails");
        // hip_dense_dec4: a window of <= 16 rows is ONE call (one weight pass), never a power-of-two split; wider
        // windows keep 16-row sweeps and a power-of-two tail (no call rounds a tail past the window).
        bool shape = n > 16U ? true : seen == std::vector<unsigned>{n, n};
        if (n > 16U)
            for (std::size_t i = 0; i < seen.size(); ++i) shape = shape && (seen[i] & (seen[i] - 1U)) == 0U;
        check(shape, "dec4 one call per <= 16-row window, power-of-two tails above");
    }
    std::int8_t ca = 0, cb = 0;
    WideScratch a, b;
    a.codes = &ca;
    b.codes = &cb;
    Tensor q8{8, 128, 16};
    linear(q8, nullptr, nullptr, 64, nullptr, a, Iu4Scratch{});
    linear(q8, nullptr, nullptr, 32, nullptr, b, Iu4Scratch{});
    linear(q8, nullptr, nullptr, 64, nullptr, a, Iu4Scratch{});
    check(scratch_seen == std::vector<const void*>{&ca, &ca, &cb, &cb, &ca, &ca}, "686-3 independent wide scratch");
    for (int device = 0; device < 2; ++device) {
        current_device = device;
        auto ctx = std::unique_ptr<cc::BigCtx>(cc::dense_big_ctx_create());
        CK(cc::bind_device(*ctx));
        const auto* table = cc::get_table(*ctx, cc::kChain, 16, 128, 1);
        check(table && live_allocations.at(table->kbs) == device, "686-4 device-local plan");
        check(cc::get_table(*ctx, cc::kChain, 16, 128, 1) == table, "686-4 plan reuse");
        current_device = 1 - device;
        check(cc::bind_device(*ctx) == cudaErrorInvalidValue, "686-4 reject cross-device context");
    }
    check(live_allocations.empty(), "686-4 table owner cleanup and device restore");
    for (int device = 0; device < 2; ++device) {
        current_device = device;
        CK(attr_big()); CK(attr_big8()); CK(attr_flash(0)); CK(attr_flash(1));
        for (int kernel = 1; kernel <= 3; ++kernel)
            check(attributes[{device, kernel}] != 0, "686-4 attributes initialized on each device");
    }
    current_device = 0;
    { cc::BigCtx ctx; CK(cc::bind_device(ctx)); fail_after = 1;
      check(cc::get_table(ctx, cc::kChain, 16, 128, 1) == nullptr && live_allocations.empty(), "686-4 partial plan allocation cleanup"); fail_after = -1; }
    auto worker = [] { auto& ctx = cc::scratch(); CK(cc::bind_device(ctx)); runtime_require(cc::get_table(ctx, cc::kChain, 16, 128, 1)); };
    std::thread t1(worker), t2(worker); t1.join(); t2.join();
    check(live_allocations.empty(), "686-4 concurrent implicit contexts released");
    for (auto policy : {cc::DenseArith::fp32, cc::DenseArith::mma1, cc::DenseArith::mma2, cc::DenseArith::mma3}) {
        e.opt.dense_arithmetic = policy;
        cc::process_policy = policy == cc::DenseArith::mma3 ? cc::DenseArith::fp32 : cc::DenseArith::mma3;
        check((e.big_fallback() != nullptr) == (policy == cc::DenseArith::fp32), "686-5 per-engine admission");
        check(e.pf_terms() == (policy == cc::DenseArith::mma3 ? 3 : policy == cc::DenseArith::mma2 ? 2 : 1), "686-5 per-engine terms");
        e.final_head(); check(cc::head_policy == policy, "686-5 per-engine final head");
    }
#if defined(STRIX_CUDA_FAST8_SM120A)
    check(cc::dense_big_fast8_supported(), "686-6 specific implementation available");
    device_prop.major = 8; check(!cc::dense_big_fast8_supported(), "686-6 unsupported device");
#else
    check(!cc::dense_big_fast8_supported(), "686-6 generic build unavailable");
#endif
    e.opt.prefill_arith = q::PrefillArith::fast8; ids.back() = 1;
    check(rejects([&] { e.admit(ids.data(), 160); }) && e.work == 0, "686-6 reject before state mutation");
    e.opt.prefill_arith = q::PrefillArith::exact;
    bool completed = false;
    for (int fail = 0; fail < 30; ++fail) {
        fail_after = fail;
        { Engine owned; lifetime.clear();
          if (rejects([&] { owned.pf_init(1); })) check(owned.pf.sl.empty(), "686-7 atomic slot publication");
          else completed = true; }
        check(live_allocations.empty() && streams.empty(), "686-7 partial/full slot cleanup");
        check(!lifetime.empty() && lifetime.front() == 'S' && lifetime.back() == 'D', "686-7 drain before release");
        if (completed) break;
    }
    fail_after = -1; check(completed, "686-7 all allocation failures exercised");
    { Engine owned; owned.pf_init(1); lifetime.clear();
      { Engine::Events pool{owned.pf, nullptr, nullptr, {}};
        pool.start = std::malloc(1); events.insert(pool.start);
        pool.ev.push_back(std::malloc(1)); events.insert(pool.ev.back()); }
      check(events.empty() && lifetime.front() == 'S' && lifetime.back() == 'E', "686-7 event cleanup drains work"); }
    Bench bench; bench.bench(); bench.pf_sample = true; bench.bench();
    check(!bench.sampled, "686-8 short sampled benchmark uses complete short workload");
    check(rejects([] { cli(0, 2); }), "686-9 reject zero repeats");
    check(rejects([] { cli(1, 0); }) && rejects([] { cli(1, 3); }), "686-design slot bounds");
    cli(1, 1); cli(2, 2);
    check(incomplete({1, 2}, {1, 2, 3}, 3) != 0 && incomplete({1, 2, 3}, {1, 2}, 3) != 0,
          "686-design reject incomplete output lengths");
    check(incomplete({1, 2, 3}, {1, 2, 3}, 3) == 0, "686-design complete output lengths");
    OtherEngine other;
    strix::spec::EngineVerifyTarget<OtherEngine> target(other);
    std::vector<std::uint32_t> greedy;
    target.verify(std::span<const std::uint32_t>(ids.data(), 2), greedy);
    check(other.calls == 1 && greedy == std::vector<std::uint32_t>{7, 7}, "686-design shared target accepts independent vocabulary");
    return failures ? 1 : 0;
}
