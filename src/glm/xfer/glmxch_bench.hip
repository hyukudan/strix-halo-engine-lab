// GLM TP2 exchange path: bitwise test and microbench (design notes).
// Adapted from mmxfer_bench (minimax/xfer 3851376c) to GLM's schedule and consumers.
//
// The GLM decode chain (all modes): per sublayer boundary a producer writes the rank partial from the normed input,
// the partial is reduced (R17), the real GLM consumers run (R7 expand, hc_pre = R6 + collapse, weighted RMSNorm; at
// MoE FFN sites the router R1/R5/R14), and each token ends with the vocabulary-split head: each rank's local argmax
// (first maximum) over its logits slice, the (value, id) exchange and the R18 pick. 91 exchanges per token. What is
// checked: the final 4 x 4096 state of every row, plus a hash folded on the device over every router id list and
// every picked token, against a CPU replay through the Tier A host emulation (src/glm/host), and between ranks.
//
//   glmxch-bench gates      one process: post_expand / post_expand_bracket == R17 + dev::hc_expand and hc_pre_v2 ==
//                           dev::hc_pre, rows 1..8 with poisoned inactive rows, subnormals, cancellation, signed zeros,
//                           R6 saturation / ties / exp underflow; C1 flag on a NaN partial.
//   glmxch-bench selftest   gates + two processes (fork before any HIP call) on the one GPU over socket pairs through the
//                           usb4-host transport code: the chain for every stream-wait arm vs the replay (rows 1 and 2,
//                           depth 4 and 2), the engine seam adapter (XferExchange), argmax split with ties/NaN, seal,
//                           corruption inline/deferred and size mismatch failing with the intended error on both ranks.
//   glmxch-bench boundary   one node, no exchange: GPU time of each post-exchange kernel and of whole boundaries.
//   glmxch-bench pair RANK OUT_DIR
//       (1) curve: two-sided Transport::exchange at the GLM sizes (8 B .. 32 MiB);
//       (2) dependent chain per arm at 8 / 16384 / 131072 B (a trivial consumer), 3 reps ABA;
//       (3) GLM token simulation, rows 1/2/4/8, 3 reps ABA, local baselines with the same values and consumer
//           schedule bracketing every group, every arm == replay (rows 1..8) and == the peer;
//       (4) transport calibration (eager threshold) with one module arm, == replay.
//       Any failure or mismatch ends the bench on this rank at once (the peer's transport fails with it): no re-pairing.
//       Reported cost = NET PATH OVERHEAD: token(arm) - token(its local baseline). The baseline computes both ranks'
//       partials and both head slices (90 extra producer launches + one head slice per token, timed in `boundary` as
//       "producer (1 rank)" and "head slice logits+argmax"); the report adds that work back to bound the exchange cost.
#include "glm_post_exchange.hpp"
#include "glm_post_fast.hpp"
#include "glm_xfer.hpp"
#include "glm_xfer_device.hpp"
#include "glm_xfer_exchange.hpp"

#include "glm/common/glm_math.hpp"
#include "glm/host/glm_host.hpp"
#include "glm/kernels/glm_tier_a.hpp"

#include "strix/dist/communicator.hpp"

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gx = strix::glm::xfer;
namespace glm = strix::glm;
namespace host = strix::glm::host;
namespace dev = strix::glm::dev;
namespace dist = strix::dist;
using Clock = std::chrono::steady_clock;

namespace {

void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
double us_since(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }

constexpr std::uint64_t kFnv0 = 14695981039346656037ULL, kFnvP = 1099511628211ULL;
std::uint64_t fnv(const void* p, std::size_t bytes, std::uint64_t h = kFnv0) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * kFnvP;
    return h;
}

constexpr unsigned kH = glm::kHidden, kS = glm::kHc * glm::kHidden, kMix = glm::kHcMix, kSub = gx::GlmSchedule::kLayers * 2;
constexpr unsigned kMaxRows = 8, kHeadV = kH;  // simulated head: each rank's vocabulary slice = 4096 ids
constexpr std::size_t kRow = gx::kRowF32;

// ---- synthetic GLM weights (deterministic; shapes and dtypes of the model)
float hashf(std::uint64_t k) {  // [-1, 1)
    k ^= k >> 33U;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33U;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33U;
    return static_cast<float>(k >> 40U) * (1.0F / 8388608.0F) - 1.0F;
}
std::uint16_t to_bf16(float f) {
    const std::uint32_t u = __builtin_bit_cast(std::uint32_t, f);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16U) & 1U)) >> 16U);
}
std::uint16_t to_f16(float f) {  // finite, normal range only (synthetic weights)
    const std::uint32_t u = __builtin_bit_cast(std::uint32_t, f);
    const std::uint32_t sign = (u >> 16U) & 0x8000U;
    const int e = static_cast<int>((u >> 23U) & 0xFFU) - 127 + 15;
    if (e <= 0) return static_cast<std::uint16_t>(sign);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10U) | ((u >> 13U) & 0x3FFU));
}

struct HostWeights {
    std::vector<std::vector<float>> fn, base, scale;   // per sublayer: [24][16384], [24], [3]
    std::vector<std::vector<std::uint16_t>> norm;      // per sublayer: [4096] BF16 (the sublayer's input norm)
    std::vector<std::vector<std::uint16_t>> router;    // per MoE layer: [288][4096] F16
    std::vector<std::vector<float>> bias;              // per MoE layer: [288]
    HostWeights() {
        fn.resize(kSub);
        base.resize(kSub);
        scale.resize(kSub);
        norm.resize(kSub);
        for (unsigned s = 0; s < kSub; ++s) {
            fn[s].resize(std::size_t{kMix} * kS);
            for (std::size_t i = 0; i < fn[s].size(); ++i) fn[s][i] = hashf((std::uint64_t{s} << 40U) + i) * 0.02F;
            base[s].resize(kMix);
            for (unsigned i = 0; i < kMix; ++i) base[s][i] = hashf((std::uint64_t{s} << 40U) + (1ULL << 39U) + i) * 0.5F;
            scale[s] = {0.5F + 0.01F * static_cast<float>(s % 7U), 0.75F, 0.25F + 0.01F * static_cast<float>(s % 5U)};
            norm[s].resize(kH);
            for (unsigned i = 0; i < kH; ++i) norm[s][i] = to_bf16(1.0F + 0.25F * hashf((std::uint64_t{s} << 40U) + (3ULL << 38U) + i));
        }
        router.resize(gx::GlmSchedule::kLayers);
        bias.resize(gx::GlmSchedule::kLayers);
        for (unsigned L = glm::kFirstMoe; L < gx::GlmSchedule::kLayers; ++L) {
            router[L].resize(std::size_t{glm::kExperts} * kH);
            for (std::size_t i = 0; i < router[L].size(); ++i) router[L][i] = to_f16(hashf((std::uint64_t{L} << 41U) + (1ULL << 40U) + i) * 0.05F);
            bias[L].resize(glm::kExperts);
            for (unsigned i = 0; i < glm::kExperts; ++i) bias[L][i] = hashf((std::uint64_t{L} << 41U) + (3ULL << 39U) + i) * 0.01F;
        }
    }
};
const HostWeights& host_weights() {
    static const HostWeights w;
    return w;
}

template <class T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    hip_ok(hipMalloc(reinterpret_cast<void**>(&p), v.size() * sizeof(T)), "upload");
    hip_ok(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return p;
}
struct DevWeights {
    std::vector<float*> fn, base, scale;
    std::vector<std::uint16_t*> norm, router;
    std::vector<float*> bias;
    DevWeights() {
        const HostWeights& h = host_weights();
        for (unsigned s = 0; s < kSub; ++s) {
            fn.push_back(upload(h.fn[s]));
            base.push_back(upload(h.base[s]));
            scale.push_back(upload(h.scale[s]));
            norm.push_back(upload(h.norm[s]));
        }
        router.assign(gx::GlmSchedule::kLayers, nullptr);
        bias.assign(gx::GlmSchedule::kLayers, nullptr);
        for (unsigned L = glm::kFirstMoe; L < gx::GlmSchedule::kLayers; ++L) {
            router[L] = upload(h.router[L]);
            bias[L] = upload(h.bias[L]);
        }
    }
};

// ---- the producer standing for a sublayer (attention / FFN on this rank's heads or I-slice): a partial from the
// normed input, different per rank, row and step. The same on device and host (-ffp-contract=off).
__host__ __device__ inline float prod_c(unsigned d, unsigned step, unsigned rank) {
    return static_cast<float>((d * 7U + step * 13U + rank * 5U) & 1023U) * (1.0F / 4096.0F) - 0.125F;
}
__host__ __device__ inline float prod_a(unsigned rank) { return rank == 0 ? 0.375F : -0.25F; }
__host__ __device__ inline float x0_of(unsigned i) { return 0.5F * static_cast<float>((i * 2654435761U) >> 20U) * (1.0F / 4096.0F) - 0.125F; }
// The simulated head: rank 0's logits = xin, rank 1's = -xin rotated by 1024 (exact; distinct maxima per rank).
__host__ __device__ inline float head_logit(const float* xin_row, unsigned j, unsigned rank) {
    return rank == 0 ? xin_row[j] : -xin_row[(j + 1024U) % kH];
}

constexpr unsigned kThreads = 256;
unsigned blocks_for(std::size_t n) { return static_cast<unsigned>(std::min<std::size_t>((n + kThreads - 1) / kThreads, 1024)); }

__global__ __launch_bounds__(kThreads) void produce_kernel(const float* x, float* out, unsigned n, unsigned step, unsigned rank,
                                                           gx::ProducerTarget target, int publish) {
    const float a = prod_a(rank);
    for (unsigned i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) out[i] = x[i] * a + prod_c(i % kH, step, rank);
    if (publish != 0) gx::glmx_publish(target, gridDim.x);
}
__global__ void x0_kernel(float* X, unsigned n) {
    for (unsigned i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) X[i] = x0_of(i);
}
__global__ void head_logits_kernel(const float* xin, float* out, unsigned rank, unsigned rows) {
    for (unsigned i = blockIdx.x * kThreads + threadIdx.x; i < rows * kH; i += gridDim.x * kThreads)
        out[i] = head_logit(xin + (i / kH) * kH, i % kH, rank);
}
__global__ void pack_kernel(const float* value, const std::uint32_t* id, std::uint64_t* out, unsigned rows) {
    for (unsigned r = threadIdx.x; r < rows; r += blockDim.x)
        out[r] = (static_cast<std::uint64_t>(id[r]) << 32U) | __builtin_bit_cast(std::uint32_t, value[r]);
}
// R18 from rank 0's candidate a and rank 1's b (the module's pick: rank 0 wins ties).
__global__ void pick_kernel(const std::uint64_t* a, const std::uint64_t* b, std::uint32_t* ids, unsigned rows) {
    for (unsigned r = threadIdx.x; r < rows; r += blockDim.x) {
        const float av = __builtin_bit_cast(float, static_cast<std::uint32_t>(a[r])), bv = __builtin_bit_cast(float, static_cast<std::uint32_t>(b[r]));
        const auto ai = static_cast<std::uint32_t>(a[r] >> 32U), bi = static_cast<std::uint32_t>(b[r] >> 32U);
        ids[r] = (bv > av || (bv == av && bi < ai)) ? bi : ai;
    }
}
// The device-side running hash of every router id list and every picked token (one lane, call order, row order).
__global__ void fold_kernel(std::uint64_t* h, const std::uint32_t* v, unsigned n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    std::uint64_t x = *h;
    for (unsigned i = 0; i < n; ++i) x = (x ^ v[i]) * kFnvP;
    *h = x;
}
// mmxfer's trivial chain consumer (h += p0 + p1), for the wire-level chain arms.
__global__ void consume_kernel(float* h, const float* p0, const float* p1, unsigned n) {
    for (unsigned i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) h[i] = h[i] + (p0[i] + p1[i]);
}
__global__ void consume_wait_kernel(float* h, const float* local, gx::RxView v, unsigned rank, unsigned n) {
    gx::glmx_wait(v);
    const auto* peer = static_cast<const float*>(v.payload);
    const float* p0 = rank == 0 ? local : peer;
    const float* p1 = rank == 0 ? peer : local;
    for (unsigned i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) h[i] = h[i] + (p0[i] + p1[i]);
}
// The 24 R1 chains of hc_pre alone (lanes 0..23 of one block per row, as in dev::hc_pre): the contract's latency part.
__global__ __launch_bounds__(256) void chains_only_kernel(const float* F, const float* fn, float* z) {
    const unsigned r = blockIdx.x, l = threadIdx.x;
    if (l >= kMix) return;
    const float* Fr = F + static_cast<std::size_t>(r) * kS;
    const float* row = fn + static_cast<std::size_t>(l) * kS;
    float acc = 0.0F;
    for (unsigned i = 0; i < kS; ++i) acc = __builtin_fmaf(Fr[i], row[i], acc);
    z[static_cast<std::size_t>(r) * kMix + l] = acc;
}
// The contract's latency floor of one R1 chain: 16,384 dependent fmaf with register operands (nothing to load), one
// wave; every lane runs its own chain, so the time is one chain's dependent-FMA latency plus the launch.
__global__ __launch_bounds__(32) void fma_chain_kernel(const float* a, float* out, unsigned n) {
    const unsigned l = threadIdx.x;
    const float x = a[l], c = a[32 + l];
    float acc = 0.0F;
    for (unsigned i = 0; i < n; ++i) acc = __builtin_fmaf(x, acc, c);
    out[l] = acc;
}
// Same chain with 16 dependent fmaf per loop step (straight-line inside, distinct register operands as in fma16), so
// the loop control is off the dependency 15 times in 16: the R1 chain rate hc_pre can at best reach.
__global__ __launch_bounds__(32) void fma_chain16_kernel(const float* a, float* out, unsigned n) {
    const unsigned l = threadIdx.x;
    float x[16];
    #pragma unroll
    for (unsigned k = 0; k < 16; ++k) x[k] = a[(l + k) % 32];
    const float c = a[32 + l];
    float acc = 0.0F;
    for (unsigned i = 0; i < n / 16; ++i) {
        #pragma unroll
        for (unsigned k = 0; k < 16; ++k) acc = __builtin_fmaf(x[k], c, acc);
    }
    out[l] = acc;
}
__global__ void sum_kernel(const float* a, const float* b, float* y, unsigned n) {  // R17 as its own op (gate reference)
    for (unsigned i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) y[i] = a[i] + b[i];
}

// A transport shared by several users in turn (the arms of one pairing).
class Borrowed final : public dist::Transport {
public:
    explicit Borrowed(dist::Transport& t) : t_(t) {}
    void send(std::span<const std::byte> m) override { t_.send(m); }
    void receive(std::span<std::byte> m) override { t_.receive(m); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); }
    [[nodiscard]] std::string name() const override { return t_.name(); }
    void abort(const std::string& why) override { t_.abort(why); }
    [[nodiscard]] std::string state() const override { return t_.state(); }

private:
    dist::Transport& t_;
};

// ---- arms
enum class Path { local, sync, glmx };
enum class Prod { fused, kernel, copy };  // self-published in place, in place + publish kernel, copy kernel
struct Arm {
    const char* name;
    Path path;
    Prod prod = Prod::copy;
    bool deferred = true;
    bool kwait = false;
    bool bracket = false;  // hc_bracket in the wire window + post_expand_bracket
    bool v2 = true;        // hc_pre_v2 (exact parallel R6) instead of dev::hc_pre
    gx::Memory tx = gx::Memory::host_coherent, rx = gx::Memory::host_coherent;
    const char* base = "";  // the local arm with the same consumer schedule (cost = arm - base)
    bool fast = false;      // hc_pre_v3 variant hcv + router_dense variant rv + route_v2 (exact, gated vs Tier A)
    unsigned tokens = 8;    // per run (the Tier A consumer arms are ~65 ms/token: 2)
    int hcv = 3, rv = 2;    // v5 (measurement); fast6 arms: 4 / 3 (LDS-only tile barriers, 36 router blocks)
};
Arm fast_arm(Arm a, const char* name, const char* base) {
    a.name = name;
    a.base = base;
    a.fast = true;
    return a;
}
Arm fast7_arm(Arm a, const char* name, int hcv) {  // design notes variants 5-7, router auto
    a = fast_arm(a, name, "");
    a.hcv = hcv;
    a.rv = -1;
    return a;
}
Arm fast6_arm(Arm a, const char* name, const char* base) {
    a = fast_arm(a, name, base);
    a.hcv = 4;
    a.rv = 3;
    return a;
}
const gx::Memory HM = gx::Memory::host_coherent, UM = gx::Memory::device_uncached;

std::vector<Arm> token_arms() {
    // Tier A consumers (as gated in P1; ~65 ms/token at R=1): 2 tokens per run, the reference schedule.
    Arm lv2{"local-v2", Path::local};
    Arm cv2{"def-copy", Path::glmx, Prod::copy, true, false, false, true, HM, HM, "local-v2"};
    for (Arm* a : {&lv2, &cv2}) a->tokens = 2;
    // fast consumers (exact, same bits), 8 tokens per run. The per-message costs of every module form come from the
    // wire-level chain phase (p1); here the token-level confirmation of the main forms. Local baselines bracket.
    const Arm lf{"local-fast", Path::local}, lfb{"local-fast-br", Path::local, Prod::copy, true, false, true};
    std::vector<Arm> all = {
        lv2,
        fast_arm(lf, "local-fast", ""),
        fast6_arm(lf, "local-fast6", ""),
        fast7_arm(lf, "local-fast7-5", 5),
        fast7_arm(lf, "local-fast7-6", 6),
        fast7_arm(lf, "local-fast7-7", 7),
        fast_arm(lfb, "local-fast-br", ""),
        cv2,
        fast_arm({"", Path::sync, Prod::copy, true, false, false, true}, "sync-fast", "local-fast"),
        fast_arm({"", Path::glmx, Prod::copy, true, false, false, true}, "def-copy-fast", "local-fast"),
        fast_arm({"", Path::glmx, Prod::fused, true, true, false, true}, "defkw-fused-fast", "local-fast"),
        fast_arm({"", Path::glmx, Prod::fused, true, true, true, true}, "defkw-fused-br-fast", "local-fast-br"),
        fast_arm(lf, "local-fast", ""),
        fast6_arm(lf, "local-fast6", ""),
        fast_arm(lfb, "local-fast-br", ""),
        lv2,
    };
    return all;
}

struct Rank {
    unsigned rank = 0, rows = 0;
    hipStream_t stream = nullptr;
    float *X = nullptr, *X2 = nullptr, *B = nullptr, *F = nullptr, *mix = nullptr, *xc = nullptr, *xin = nullptr;
    float *part = nullptr, *part1 = nullptr, *remote = nullptr, *logits = nullptr, *hl = nullptr, *hl1 = nullptr, *h = nullptr;
    float *amax_v = nullptr, *amax_v1 = nullptr;
    std::uint32_t *ids = nullptr, *amax_id = nullptr, *amax_id1 = nullptr, *next_ids = nullptr;
    std::uint64_t *pk = nullptr, *pk_r = nullptr, *fold = nullptr;
    float* rw = nullptr;
    unsigned* flag = nullptr;
    dist::Communicator* comm = nullptr;
    gx::Exchanger* xf = nullptr;
    const DevWeights* w = nullptr;
};
void rank_alloc(Rank& r, unsigned rank) {
    r.rank = rank;
    hip_ok(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking), "stream");
    auto f = [&](float*& p, std::size_t n) { hip_ok(hipMalloc(reinterpret_cast<void**>(&p), n * 4), "alloc"); };
    auto u = [&](std::uint32_t*& p, std::size_t n) { hip_ok(hipMalloc(reinterpret_cast<void**>(&p), n * 4), "alloc"); };
    auto q = [&](std::uint64_t*& p, std::size_t n) { hip_ok(hipMalloc(reinterpret_cast<void**>(&p), n * 8), "alloc"); };
    f(r.X, std::size_t{kMaxRows} * kS);
    f(r.X2, std::size_t{kMaxRows} * kS);
    f(r.B, std::size_t{kMaxRows} * kS);
    f(r.F, std::size_t{kMaxRows} * kS);
    f(r.mix, std::size_t{kMaxRows} * kMix);
    f(r.xc, std::size_t{kMaxRows} * kH);
    f(r.xin, std::size_t{kMaxRows} * kH);
    f(r.part, std::size_t{kMaxRows} * kH);
    f(r.part1, std::size_t{kMaxRows} * kH);
    f(r.remote, std::size_t{kMaxRows} * kH);
    f(r.logits, std::size_t{kMaxRows} * glm::kExperts);
    f(r.hl, std::size_t{kMaxRows} * kH);
    f(r.hl1, std::size_t{kMaxRows} * kH);
    f(r.rw, std::size_t{kMaxRows} * glm::kTopE);
    f(r.amax_v, kMaxRows);
    f(r.amax_v1, kMaxRows);
    f(r.h, std::size_t{kMaxRows} * 8 * kH);
    u(r.ids, std::size_t{kMaxRows} * glm::kTopE);
    u(r.amax_id, kMaxRows);
    u(r.amax_id1, kMaxRows);
    u(r.next_ids, kMaxRows);
    q(r.pk, kMaxRows);
    q(r.pk_r, kMaxRows);
    q(r.fold, 1);
    hip_ok(hipMalloc(reinterpret_cast<void**>(&r.flag), 4), "flag");
    hip_ok(hipMemset(r.flag, 0, 4), "flag");
}

gx::Options module_options(const Rank& r, const Arm& a, std::size_t cap) {
    gx::Options o;
    o.rank = r.rank;
    o.capacity = cap;
    o.stream = r.stream;
    o.crc_deferred = a.deferred;
    o.tx_memory = a.tx;
    o.rx_memory = a.rx;
    return o;
}

// Owns the Communicator / Exchanger of one arm over a shared transport.
struct ArmScope {
    std::unique_ptr<dist::Communicator> comm;
    std::unique_ptr<gx::Exchanger> xf;
    Rank& r;
    ArmScope(Rank& rr, dist::Transport* t, const Arm& a, std::size_t cap, unsigned depth = 4) : r(rr) {
        if (a.path == Path::glmx) {
            auto o = module_options(r, a, cap);
            o.depth = depth;
            xf = std::make_unique<gx::Exchanger>(std::make_unique<Borrowed>(*t), o);
        } else if (a.path == Path::sync) {
            dist::Group group;
            group.rank = r.rank;
            comm = std::make_unique<dist::Communicator>(group, std::make_unique<Borrowed>(*t), cap);
        }
        r.comm = comm.get();
        r.xf = xf.get();
    }
    ~ArmScope() {
        r.comm = nullptr;
        r.xf = nullptr;
    }
};

// ---- the GLM decode chain on the device
void hc_pre_any(Rank& r, const Arm& a, unsigned sub, const float* X) {
    const DevWeights& w = *r.w;
    if (a.fast) gx::hc_pre_v3(X, w.fn[sub], w.base[sub], w.scale[sub], r.F, r.mix, r.xc, r.rows, r.stream, a.hcv);
    else if (a.v2) gx::hc_pre_v2(X, w.fn[sub], w.base[sub], w.scale[sub], r.F, r.mix, r.xc, r.rows, r.stream);
    else dev::hc_pre(X, w.fn[sub], w.base[sub], w.scale[sub], r.F, r.mix, r.xc, r.rows, r.stream);
    dev::rmsnorm(r.xc, w.norm[sub], dev::WType::BF16, kH, glm::kEps, r.xin, r.rows, r.stream);
}
void chain_init_glm(Rank& r, const Arm& a) {
    hipLaunchKernelGGL(x0_kernel, dim3(blocks_for(std::size_t{r.rows} * kS)), dim3(kThreads), 0, r.stream, r.X, r.rows * kS);
    hip_ok(hipMemsetAsync(r.flag, 0, 4, r.stream), "flag");
    const std::uint64_t f0 = kFnv0;
    hip_ok(hipMemcpyAsync(r.fold, &f0, 8, hipMemcpyHostToDevice, r.stream), "fold");
    hc_pre_any(r, a, 0, r.X);
    hip_ok(hipStreamSynchronize(r.stream), "init");
}

// One sublayer boundary: producer -> reduction -> post_expand (R17 + R7) -> next site's hc_pre + norm (+ router).
// The local arm computes BOTH ranks' partials (the same values as the exchange arms) and reduces them without a wire.
void glm_boundary(Rank& r, const Arm& a, unsigned sub, unsigned step) {
    const unsigned n = r.rows * kH;
    const std::size_t bytes = std::size_t{r.rows} * kRow;
    const unsigned g = blocks_for(n);
    const gx::ProducerTarget none{};
    const gx::RxView noview{};
    const float* local = nullptr;
    const float* peer = nullptr;
    unsigned order = r.rank;
    gx::RxView view{};
    switch (a.path) {
        case Path::local:
            hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.part, n, step, 0U, none, 0);
            hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.part1, n, step, 1U, none, 0);
            local = r.part;
            peer = r.part1;
            order = 0;
            break;
        case Path::sync:
            hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.part, n, step, r.rank, none, 0);
            r.comm->exchange(r.stream, r.part, r.remote, bytes, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), sub, step});
            local = r.part;
            peer = r.remote;
            break;
        case Path::glmx: {
            const gx::Tag tag{sub % 2 == 0 ? gx::Kind::attn_sum : gx::Kind::ffn_sum, sub / 2, step};
            if (a.prod == Prod::copy) {
                hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.part, n, step, r.rank, none, 0);
                r.xf->send_copy(r.part, bytes, tag);
                local = r.part;
            } else {
                auto* tx = static_cast<float*>(r.xf->tx());
                const bool fused = a.prod == Prod::fused;
                hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, tx, n, step, r.rank,
                                   fused ? r.xf->producer_target() : none, fused ? 1 : 0);
                r.xf->send(bytes, tag, fused ? gx::Publish::producer : gx::Publish::kernel);
                local = tx;
            }
            break;
        }
    }
    if (a.bracket) gx::hc_bracket(r.X, r.mix, r.B, r.rows, r.stream);  // runs while the message travels
    const bool kw = a.kwait && a.path == Path::glmx;
    if (a.path == Path::glmx) {
        if (kw) view = r.xf->recv_view();
        else peer = static_cast<const float*>(r.xf->recv());
    }
    if (a.bracket) gx::post_expand_bracket(r.B, r.mix, local, kw ? nullptr : peer, kw ? view : noview, order, r.X2, r.flag, r.rows, r.stream);
    else gx::post_expand(r.X, r.mix, local, kw ? nullptr : peer, kw ? view : noview, order, r.X2, r.flag, r.rows, r.stream);
    std::swap(r.X, r.X2);
    const unsigned next = (sub + 1) % kSub;
    hc_pre_any(r, a, next, r.X);
    const unsigned L = next / 2;
    if (next % 2 == 1 && glm::is_moe_layer(L)) {  // FFN site: router (R1 288 x 4096 F16, R5, R14) on the normed input
        if (a.fast) {
            gx::router_dense(r.xin, r.w->router[L], r.logits, r.rows, r.stream, a.rv);
            gx::route_v2(r.logits, r.w->bias[L], r.ids, r.rw, r.rows, r.stream);
        } else {
            dev::dense(r.xin, r.w->router[L], dev::WType::F16, kH, 0, glm::kExperts, r.logits, r.rows, r.stream);
            dev::route(r.logits, r.w->bias[L], r.ids, r.rw, r.rows, r.stream);
        }
        hipLaunchKernelGGL(fold_kernel, dim3(1), dim3(32), 0, r.stream, r.fold, r.ids, r.rows * glm::kTopE);
    }
}

// The head: each rank's local first-max over its 4096-id slice, the (value, id) exchange, the R18 pick; the picked
// ids are folded into the hash. local: both slices on this rank, picked without a wire.
void glm_head(Rank& r, const Arm& a, unsigned step) {
    const unsigned g = blocks_for(std::size_t{r.rows} * kH);
    switch (a.path) {
        case Path::local:
            hipLaunchKernelGGL(head_logits_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.hl, 0U, r.rows);
            hipLaunchKernelGGL(head_logits_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.hl1, 1U, r.rows);
            dev::argmax(r.hl, kHeadV, 0, r.amax_v, r.amax_id, r.flag, r.rows, r.stream);
            dev::argmax(r.hl1, kHeadV, kHeadV, r.amax_v1, r.amax_id1, r.flag, r.rows, r.stream);
            hipLaunchKernelGGL(pack_kernel, dim3(1), dim3(64), 0, r.stream, r.amax_v, r.amax_id, r.pk, r.rows);
            hipLaunchKernelGGL(pack_kernel, dim3(1), dim3(64), 0, r.stream, r.amax_v1, r.amax_id1, r.pk_r, r.rows);
            hipLaunchKernelGGL(pick_kernel, dim3(1), dim3(64), 0, r.stream, r.pk, r.pk_r, r.next_ids, r.rows);
            break;
        case Path::sync:
            hipLaunchKernelGGL(head_logits_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.hl, r.rank, r.rows);
            dev::argmax(r.hl, kHeadV, r.rank * kHeadV, r.amax_v, r.amax_id, r.flag, r.rows, r.stream);
            hipLaunchKernelGGL(pack_kernel, dim3(1), dim3(64), 0, r.stream, r.amax_v, r.amax_id, r.pk, r.rows);
            r.comm->exchange(r.stream, r.pk, r.pk_r, std::size_t{r.rows} * 8, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::argmax), 99, step});
            hipLaunchKernelGGL(pick_kernel, dim3(1), dim3(64), 0, r.stream, r.rank == 0 ? r.pk : r.pk_r, r.rank == 0 ? r.pk_r : r.pk, r.next_ids, r.rows);
            break;
        case Path::glmx:
            hipLaunchKernelGGL(head_logits_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.xin, r.hl, r.rank, r.rows);
            gx::argmax_exchange(*r.xf, r.hl, r.rows, kHeadV, kH, r.rank * kHeadV, step, r.next_ids);
            break;
    }
    hipLaunchKernelGGL(fold_kernel, dim3(1), dim3(32), 0, r.stream, r.fold, r.next_ids, r.rows);
}

struct ChainResult {
    double us = 0;
    std::uint64_t state = 0, ids = 0;  // fnv of the final state rows; device fold of router ids + picked tokens
    unsigned flag = 0;
    [[nodiscard]] std::uint64_t all() const { return state ^ (ids * 3ULL) ^ (flag != 0 ? 0xF1A6ULL : 0ULL); }
};
ChainResult chain_result(Rank& r) {
    hip_ok(hipStreamSynchronize(r.stream), "sync");
    if (r.xf != nullptr) r.xf->drain();
    std::vector<float> X(std::size_t{r.rows} * kS);
    hip_ok(hipMemcpy(X.data(), r.X, X.size() * 4, hipMemcpyDeviceToHost), "X");
    ChainResult c;
    c.state = fnv(X.data(), X.size() * 4);
    hip_ok(hipMemcpy(&c.ids, r.fold, 8, hipMemcpyDeviceToHost), "fold");
    hip_ok(hipMemcpy(&c.flag, r.flag, 4, hipMemcpyDeviceToHost), "flag");
    return c;
}

ChainResult run_glm_tokens(Rank& r, dist::Transport* t, const Arm& a, unsigned rows, unsigned tokens, unsigned depth = 4) {
    std::unique_ptr<ArmScope> scope;
    if (a.path != Path::local) scope = std::make_unique<ArmScope>(r, t, a, std::size_t{kMaxRows} * kRow, depth);
    r.rows = rows;
    chain_init_glm(r, a);
    if (t != nullptr) {
        std::uint64_t z = 0, zz = 0;
        t->exchange(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&z), 8), std::span<std::byte>(reinterpret_cast<std::byte*>(&zz), 8));
    }
    const auto t0 = Clock::now();
    unsigned step = 0;
    for (unsigned k = 0; k < tokens; ++k) {
        for (unsigned sub = 0; sub < kSub; ++sub) glm_boundary(r, a, sub, step++);
        glm_head(r, a, step++);
        hip_ok(hipStreamSynchronize(r.stream), "token");
        if (r.xf != nullptr) r.xf->drain();  // the engine's once-per-step drain
    }
    const double us = us_since(t0) / tokens;
    ChainResult res = chain_result(r);
    res.us = us;
    return res;
}

// ---- CPU replay through the Tier A host emulation, one thread per row: both ranks' partials, R17, hc_expand, hc_mix
// + collapse + weighted rmsnorm, host::route at MoE FFN sites, host::argmax_local / argmax_merge for the head.
struct RowReplay {
    std::vector<float> X;                  // final state of the row
    std::vector<std::vector<std::uint32_t>> events;  // per device fold call, this row's values (route ids or the token)
};
RowReplay replay_row(unsigned row, unsigned tokens) {
    const HostWeights& w = host_weights();
    RowReplay out;
    std::vector<float> X(kS), X2(kS), xc(kH), xin(kH), y(kH), l0(kH), l1(kH);
    for (unsigned i = 0; i < kS; ++i) X[i] = x0_of(row * kS + i);
    host::HcMix m;
    auto pre = [&](unsigned sub) {
        const host::HcWeights hw{host::Mat{w.fn[sub].data(), host::WType::F32, kMix, kS}, w.base[sub].data(), w.scale[sub].data()};
        m = host::hc_mix(X, hw);
        host::hc_collapse(X, m, xc);
        const host::Vec nv{w.norm[sub].data(), host::WType::BF16, kH};
        host::rmsnorm(xc, &nv, glm::kEps, xin);
    };
    pre(0);
    unsigned step = 0;
    for (unsigned k = 0; k < tokens; ++k) {
        for (unsigned sub = 0; sub < kSub; ++sub) {
            for (unsigned d = 0; d < kH; ++d) {
                const float p0 = xin[d] * prod_a(0) + prod_c(d, step, 0);
                const float p1 = xin[d] * prod_a(1) + prod_c(d, step, 1);
                y[d] = p0 + p1;
            }
            host::hc_expand(X, m, y, X2);
            X.swap(X2);
            const unsigned next = (sub + 1) % kSub;
            pre(next);
            const unsigned L = next / 2;
            if (next % 2 == 1 && glm::is_moe_layer(L)) {
                const host::Route rt = host::route(xin, host::Mat{w.router[L].data(), host::WType::F16, glm::kExperts, kH}, w.bias[L].data());
                out.events.emplace_back(rt.id.begin(), rt.id.end());
            }
            ++step;
        }
        for (unsigned j = 0; j < kH; ++j) {
            l0[j] = head_logit(xin.data(), j, 0);
            l1[j] = head_logit(xin.data(), j, 1);
        }
        const host::ArgMax g = host::argmax_merge(host::argmax_local(l0, 0), host::argmax_local(l1, kHeadV));
        out.events.push_back({g.id});
        ++step;
    }
    out.X = std::move(X);
    return out;
}
struct Replay {
    std::vector<RowReplay> rows;
    explicit Replay(unsigned n, unsigned tokens) : rows(n) {
        std::vector<std::thread> t;
        for (unsigned r = 0; r < n; ++r) t.emplace_back([&, r] { rows[r] = replay_row(r, tokens); });
        for (auto& th : t) th.join();
    }
    // The hashes a run with the first `n` rows must give (device fold order: per call, rows in order).
    [[nodiscard]] ChainResult expect(unsigned n) const {
        ChainResult c;
        std::uint64_t s = kFnv0;
        for (unsigned r = 0; r < n; ++r) s = fnv(rows[r].X.data(), rows[r].X.size() * 4, s);
        c.state = s;
        std::uint64_t h = kFnv0;
        for (std::size_t e = 0; e < rows[0].events.size(); ++e)
            for (unsigned r = 0; r < n; ++r)
                for (std::uint32_t v : rows[r].events[e]) h = (h ^ v) * kFnvP;
        c.ids = h;
        return c;
    }
};

// ---- kernel gates (one process, no exchange)
int kernel_gates() {
    int failures = 0;
    hipStream_t s = nullptr;
    hip_ok(hipStreamCreate(&s), "stream");
    const DevWeights dw;
    const unsigned R = kMaxRows;
    const float qnan = __builtin_bit_cast(float, 0x7fc01234U);
    // Inputs: random X and partials, with edge values planted: subnormals, cancellation (p1 = -p0), signed zeros
    // reaching the output (X = -0 in all 4 streams and p0 = p1 = -0: bracket -0, post * -0 = -0, sum -0).
    std::vector<float> X(std::size_t{R} * kS), p0(std::size_t{R} * kH), p1(std::size_t{R} * kH);
    for (std::size_t i = 0; i < X.size(); ++i) X[i] = hashf(0x5eedULL + i) * 3.0F;
    for (std::size_t i = 0; i < p0.size(); ++i) {
        p0[i] = hashf(0x1111ULL + i) * 2.0F;
        p1[i] = hashf(0x2222ULL + i) * 2.0F;
    }
    for (unsigned r = 0; r < R; ++r) {
        for (unsigned d = 0; d < 64; ++d) p1[r * kH + d] = -p0[r * kH + d];                              // cancellation -> +0
        for (unsigned d = 64; d < 96; ++d) p0[r * kH + d] = 1e-40F * static_cast<float>(d), p1[r * kH + d] = -3e-41F;  // subnormal
        for (unsigned d = 96; d < 104; ++d) {
            p0[r * kH + d] = -0.0F;
            p1[r * kH + d] = -0.0F;
            for (unsigned j = 0; j < glm::kHc; ++j) X[std::size_t{r} * kS + j * kH + d] = -0.0F;
        }
        for (unsigned d = 104; d < 112; ++d)
            for (unsigned j = 0; j < glm::kHc; ++j) X[std::size_t{r} * kS + j * kH + d] = 1e-39F;     // subnormal state
    }
    // Saturation weights (sublayer slot "sat"): fn rows 0..3 large positive (pre -> sigmoid saturates to 1), rows 4..7
    // large negative (post -> 2 * 0), comb rows 8..23 in identical pairs (ties) with a large scale (exp underflow).
    std::vector<float> fsat(std::size_t{kMix} * kS), bsat(kMix), ssat = {50.0F, 50.0F, 400.0F};
    for (unsigned m = 0; m < kMix; ++m) {
        const float k = m < 4 ? 5.0F : m < 8 ? -5.0F : 0.5F;
        const unsigned src = m < 8 ? m : 8 + ((m - 8) / 2) * 2;  // comb ties: rows 2k and 2k+1 equal
        for (unsigned i = 0; i < kS; ++i) fsat[std::size_t{m} * kS + i] = k * hashf((std::uint64_t{src} << 32U) + 0xabcdULL + i);
        bsat[m] = m >= 8 && (m % 2) == 1 ? bsat[m - 1] : 0.0F;
    }
    float *dfsat = upload(fsat), *dbsat = upload(bsat), *dssat = upload(ssat);
    float *dX = upload(X), *dp0 = upload(p0), *dp1 = upload(p1);
    float *F, *mixA, *mixB, *xcA, *xcB, *y, *XoA, *XoB, *B;
    unsigned *fa, *fb;
    auto al = [&](float*& p, std::size_t n) { hip_ok(hipMalloc(reinterpret_cast<void**>(&p), n * 4), "alloc"); };
    al(F, std::size_t{R} * kS);
    al(mixA, std::size_t{R} * kMix);
    al(mixB, std::size_t{R} * kMix);
    al(xcA, std::size_t{R} * kH);
    al(xcB, std::size_t{R} * kH);
    al(y, std::size_t{R} * kH);
    al(XoA, std::size_t{R} * kS);
    al(XoB, std::size_t{R} * kS);
    al(B, std::size_t{R} * kS);
    hip_ok(hipMalloc(reinterpret_cast<void**>(&fa), 4), "f");
    hip_ok(hipMalloc(reinterpret_cast<void**>(&fb), 4), "f");
    auto fill = [&](float* p, std::size_t n) {  // poison (whole buffer): inactive rows must stay poisoned
        std::vector<float> v(n, qnan);
        hip_ok(hipMemcpy(p, v.data(), n * 4, hipMemcpyHostToDevice), "poison");
    };
    auto same = [&](const float* a, const float* b, std::size_t n, const std::string& what) {
        std::vector<float> ha(n), hb(n);
        hip_ok(hipMemcpy(ha.data(), a, n * 4, hipMemcpyDeviceToHost), "d2h");
        hip_ok(hipMemcpy(hb.data(), b, n * 4, hipMemcpyDeviceToHost), "d2h");
        if (std::memcmp(ha.data(), hb.data(), n * 4) != 0) {
            std::printf("GATE %s: FAIL\n", what.c_str());
            ++failures;
        }
    };
    auto poisoned = [&](const float* a, std::size_t from, std::size_t to, const std::string& what) {
        std::vector<float> h(to - from);
        hip_ok(hipMemcpy(h.data(), a + from, (to - from) * 4, hipMemcpyDeviceToHost), "d2h");
        for (float v : h)
            if (__builtin_bit_cast(std::uint32_t, v) != 0x7fc01234U) {
                std::printf("GATE %s: inactive row written  FAIL\n", what.c_str());
                ++failures;
                return;
            }
    };
    auto check_signed_zero = [&](const float* Xo, unsigned rows, const std::string& what) {
        std::vector<float> h(std::size_t{rows} * kS);
        hip_ok(hipMemcpy(h.data(), Xo, h.size() * 4, hipMemcpyDeviceToHost), "d2h");
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned j = 0; j < glm::kHc; ++j)
                for (unsigned d = 96; d < 104; ++d)
                    if (__builtin_bit_cast(std::uint32_t, h[std::size_t{r} * kS + j * kH + d]) != 0x80000000U) {
                        std::printf("GATE %s: -0 output expected  FAIL\n", what.c_str());
                        ++failures;
                        return;
                    }
    };
    struct Site {
        const char* name;
        const float *fn, *base, *scale;
    };
    const Site sites[] = {{"sub0", dw.fn[0], dw.base[0], dw.scale[0]}, {"sub7", dw.fn[7], dw.base[7], dw.scale[7]},
                          {"sub89", dw.fn[89], dw.base[89], dw.scale[89]}, {"sat", dfsat, dbsat, dssat}};
    for (unsigned rows = 1; rows <= R; ++rows)
        for (const Site& st : sites) {
            const std::string tag = std::string(st.name) + " rows " + std::to_string(rows);
            for (float* p : {mixA, mixB}) fill(p, std::size_t{R} * kMix);
            for (float* p : {xcA, xcB}) fill(p, std::size_t{R} * kH);
            dev::hc_pre(dX, st.fn, st.base, st.scale, F, mixA, xcA, rows, s);
            gx::hc_pre_v2(dX, st.fn, st.base, st.scale, F, mixB, xcB, rows, s);
            hip_ok(hipStreamSynchronize(s), "pre");
            same(mixA, mixB, std::size_t{rows} * kMix, "hc_pre_v2 mix " + tag);
            same(xcA, xcB, std::size_t{rows} * kH, "hc_pre_v2 collapse " + tag);
            poisoned(mixB, std::size_t{rows} * kMix, std::size_t{R} * kMix, "hc_pre_v2 mix " + tag);
            poisoned(xcB, std::size_t{rows} * kH, std::size_t{R} * kH, "hc_pre_v2 collapse " + tag);
            for (unsigned rank : {0U, 1U}) {
                const float* loc = rank == 0 ? dp0 : dp1;
                const float* per = rank == 0 ? dp1 : dp0;
                const std::string rt = tag + " rank " + std::to_string(rank);
                for (float* p : {XoA, XoB}) fill(p, std::size_t{R} * kS);
                hip_ok(hipMemset(fa, 0, 4), "f");
                hip_ok(hipMemset(fb, 0, 4), "f");
                hipLaunchKernelGGL(sum_kernel, dim3(blocks_for(rows * kH)), dim3(kThreads), 0, s, dp0, dp1, y, rows * kH);
                dev::hc_expand(dX, mixA, y, XoA, fa, rows, s);
                gx::post_expand(dX, mixA, loc, per, gx::RxView{}, rank, XoB, fb, rows, s);
                hip_ok(hipStreamSynchronize(s), "expand");
                same(XoA, XoB, std::size_t{rows} * kS, "post_expand " + rt);
                same(reinterpret_cast<float*>(fa), reinterpret_cast<float*>(fb), 1, "post_expand flag " + rt);
                poisoned(XoB, std::size_t{rows} * kS, std::size_t{R} * kS, "post_expand " + rt);
                if (std::string(st.name) == "sub0") check_signed_zero(XoB, rows, "post_expand " + rt);
                fill(XoB, std::size_t{R} * kS);
                hip_ok(hipMemset(fb, 0, 4), "f");
                gx::hc_bracket(dX, mixA, B, rows, s);
                gx::post_expand_bracket(B, mixA, loc, per, gx::RxView{}, rank, XoB, fb, rows, s);
                hip_ok(hipStreamSynchronize(s), "bracket");
                same(XoA, XoB, std::size_t{rows} * kS, "post_expand_bracket " + rt);
                same(reinterpret_cast<float*>(fa), reinterpret_cast<float*>(fb), 1, "post_expand_bracket flag " + rt);
                poisoned(XoB, std::size_t{rows} * kS, std::size_t{R} * kS, "post_expand_bracket " + rt);
            }
        }
    // hc_pre_v3 (both variants) == dev::hc_pre on every site incl. the saturation set, poisoned inactive rows
    for (int v : {0, 1, 2, 3, 4, 5, 6, 7})
        for (unsigned rows = 1; rows <= R; ++rows)
            for (const Site& st : sites) {
                const std::string tag = std::string("hc_pre_v3 variant ") + std::to_string(v) + " " + st.name + " rows " + std::to_string(rows);
                for (float* p : {mixA, mixB}) fill(p, std::size_t{R} * kMix);
                for (float* p : {xcA, xcB}) fill(p, std::size_t{R} * kH);
                dev::hc_pre(dX, st.fn, st.base, st.scale, F, mixA, xcA, rows, s);
                gx::hc_pre_v3(dX, st.fn, st.base, st.scale, F, mixB, xcB, rows, s, v);
                hip_ok(hipStreamSynchronize(s), "pre v3");
                same(mixA, mixB, std::size_t{rows} * kMix, tag + " mix");
                same(xcA, xcB, std::size_t{rows} * kH, tag + " collapse");
                poisoned(mixB, std::size_t{rows} * kMix, std::size_t{R} * kMix, tag);
                poisoned(xcB, std::size_t{rows} * kH, std::size_t{R} * kH, tag);
            }
    // router_dense == dev::dense (R1, F16, 288 x 4096) and route_v2 == dev::route: random inputs with -0 / subnormals,
    // then logits forcing ties: equal logits per group of 3, +-100 (sigmoid saturates to 1 / +0: massive key ties),
    // biases with ties; rows 1..8 with poisoned inactive rows.
    {
        float *lgA, *lgB, *wA, *wB;
        std::uint32_t *idA, *idB;
        al(lgA, std::size_t{R} * glm::kExperts);
        al(lgB, std::size_t{R} * glm::kExperts);
        al(wA, std::size_t{R} * glm::kTopE);
        al(wB, std::size_t{R} * glm::kTopE);
        hip_ok(hipMalloc(reinterpret_cast<void**>(&idA), std::size_t{R} * glm::kTopE * 4), "ids");
        hip_ok(hipMalloc(reinterpret_cast<void**>(&idB), std::size_t{R} * glm::kTopE * 4), "ids");
        std::vector<float> xr(std::size_t{R} * kH);
        for (std::size_t i = 0; i < xr.size(); ++i) xr[i] = hashf(0x7777ULL + i) * 2.0F;
        xr[1] = -0.0F;
        xr[2] = 1e-40F;
        float* dxr = upload(xr);
        for (unsigned L : {3U, 20U, 44U})
            for (unsigned rows = 1; rows <= R; ++rows) {
                const std::string tag = "router_dense L" + std::to_string(L) + " rows " + std::to_string(rows);
                for (float* p : {lgA, lgB}) fill(p, std::size_t{R} * glm::kExperts);
                dev::dense(dxr, dw.router[L], dev::WType::F16, kH, 0, glm::kExperts, lgA, rows, s);
                for (int v : {0, 1, 2, 3}) {
                    fill(lgB, std::size_t{R} * glm::kExperts);
                    gx::router_dense(dxr, dw.router[L], lgB, rows, s, v);
                    hip_ok(hipStreamSynchronize(s), "router");
                    same(lgA, lgB, std::size_t{rows} * glm::kExperts, tag + " variant " + std::to_string(v));
                    poisoned(lgB, std::size_t{rows} * glm::kExperts, std::size_t{R} * glm::kExperts, tag);
                }
            }
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<float> lg(std::size_t{R} * glm::kExperts), bias(glm::kExperts);
            for (std::size_t i = 0; i < lg.size(); ++i) {
                const unsigned e = static_cast<unsigned>(i % glm::kExperts), row = static_cast<unsigned>(i / glm::kExperts);
                switch (pattern) {
                    case 0: lg[i] = hashf(0x9000ULL + i) * 4.0F; break;                                  // random
                    case 1: lg[i] = hashf(0x9100ULL + row * 1000 + e / 3) * 4.0F; break;                // groups of 3 equal
                    case 2: lg[i] = (e % 2 == 0 ? 100.0F : -100.0F) * (row % 2 == 0 ? 1.0F : -1.0F); break;  // saturation
                    default: lg[i] = e % 5 == 0 ? -0.0F : 0.0F; break;                                   // all zero / -0
                }
            }
            for (unsigned e = 0; e < glm::kExperts; ++e) bias[e] = pattern == 3 ? 0.0F : hashf(0x9200ULL + e / 4) * 0.01F;  // ties in groups of 4
            float *dlg = upload(lg), *dbias = upload(bias);
            for (unsigned rows = 1; rows <= R; ++rows) {
                const std::string tag = "route_v2 pattern " + std::to_string(pattern) + " rows " + std::to_string(rows);
                for (float* p : {wA, wB}) fill(p, std::size_t{R} * glm::kTopE);
                fill(reinterpret_cast<float*>(idA), std::size_t{R} * glm::kTopE);
                fill(reinterpret_cast<float*>(idB), std::size_t{R} * glm::kTopE);
                dev::route(dlg, dbias, idA, wA, rows, s);
                gx::route_v2(dlg, dbias, idB, wB, rows, s);
                hip_ok(hipStreamSynchronize(s), "route");
                same(reinterpret_cast<float*>(idA), reinterpret_cast<float*>(idB), std::size_t{rows} * glm::kTopE, tag + " ids");
                same(wA, wB, std::size_t{rows} * glm::kTopE, tag + " weights");
                poisoned(wB, std::size_t{rows} * glm::kTopE, std::size_t{R} * glm::kTopE, tag);
            }
            hip_ok(hipFree(dlg), "free");
            hip_ok(hipFree(dbias), "free");
        }
        for (void* p : {static_cast<void*>(lgA), static_cast<void*>(lgB), static_cast<void*>(wA), static_cast<void*>(wB),
                        static_cast<void*>(idA), static_cast<void*>(idB), static_cast<void*>(dxr)})
            hip_ok(hipFree(p), "free");
    }
    {  // non-finite partial: both forms set the C1 flag (and agree with hc_expand's)
        std::vector<float> bad = p0;
        bad[3] = std::nanf("");
        float* dbad = upload(bad);
        for (unsigned* f : {fa, fb}) hip_ok(hipMemset(f, 0, 4), "f");
        gx::post_expand(dX, mixA, dbad, dp1, gx::RxView{}, 0, XoB, fa, 1, s);
        gx::hc_bracket(dX, mixA, B, 1, s);
        gx::post_expand_bracket(B, mixA, dp1, dbad, gx::RxView{}, 1, XoB, fb, 1, s);
        hip_ok(hipStreamSynchronize(s), "nan");
        unsigned a = 0, b = 0;
        hip_ok(hipMemcpy(&a, fa, 4, hipMemcpyDeviceToHost), "f");
        hip_ok(hipMemcpy(&b, fb, 4, hipMemcpyDeviceToHost), "f");
        if (a == 0 || b == 0) {
            std::printf("GATE non-finite flag: FAIL (%u %u)\n", a, b);
            ++failures;
        }
        hip_ok(hipFree(dbad), "free");
    }
    std::printf("kernel gates (hc_pre_v2 / hc_pre_v3 x2 == dev::hc_pre incl. R6 saturation / ties / underflow, router_dense == "
                "dev::dense, route_v2 == dev::route incl. key ties / saturation / -0, post_expand and "
                "post_expand_bracket == R17 + dev::hc_expand incl. cancellation / subnormals / -0, rows 1..8 with poisoned "
                "inactive rows, both rank orders, C1 flag): %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ---- greedy head split check (mmxfer's): 8 rows over a 2000-id vocabulary, ties across and inside ranks, a NaN row.
int argmax_check(unsigned rank, gx::Exchanger& xf, hipStream_t stream) {
    constexpr unsigned kRows = 8, kV = 2000, kV0 = 1000, kLd = 1024;
    std::vector<float> full(kRows * kV);
    for (unsigned r = 0; r < kRows; ++r)
        for (unsigned j = 0; j < kV; ++j) full[r * kV + j] = static_cast<float>(((j * 2654435761U + r * 40503U) >> 7) % 23U) * 0.5F - 4.0F;
    full[3 * kV + 17] = 100.0F;
    full[3 * kV + kV0 + 5] = 100.0F;
    full[5 * kV + kV0 + 300] = 200.0F;
    full[5 * kV + kV0 + 299] = 200.0F;
    full[7 * kV + kV0 + 9] = std::nanf("");
    std::vector<std::uint32_t> want(kRows);
    for (unsigned r = 0; r < kRows; ++r) {
        unsigned best = 0;
        bool finite = true;
        for (unsigned j = 0; j < kV; ++j) {
            finite &= std::isfinite(full[r * kV + j]);
            if (full[r * kV + j] > full[r * kV + best]) best = j;
        }
        want[r] = finite ? best : gx::kNonFinite;
    }
    if (want[7] != gx::kNonFinite) throw std::logic_error("argmax check setup");
    const unsigned v0 = rank == 0 ? 0 : kV0, vn = rank == 0 ? kV0 : kV - kV0;
    std::vector<float> mine(kRows * kLd, -1e30F);
    for (unsigned r = 0; r < kRows; ++r)
        for (unsigned j = 0; j < vn; ++j) mine[r * kLd + j] = full[r * kV + v0 + j];
    float* d = upload(mine);
    std::uint32_t* ids = nullptr;
    hip_ok(hipMalloc(reinterpret_cast<void**>(&ids), kRows * 4), "ids");
    int failures = 0;
    for (unsigned rows : {1U, kRows}) {
        gx::argmax_exchange(xf, d, rows, vn, kLd, v0, rows, ids);
        hip_ok(hipStreamSynchronize(stream), "argmax");
        xf.drain();
        std::vector<std::uint32_t> got(rows);
        hip_ok(hipMemcpy(got.data(), ids, rows * 4, hipMemcpyDeviceToHost), "ids");
        for (unsigned r = 0; r < rows; ++r)
            if (got[r] != want[r]) {  // stop at the first mismatch (no further exchange)
                std::printf("argmax rank %u rows %u row %u: got %u want %u  FAIL\n", rank, rows, r, got[r], want[r]);
                ++failures;
                break;
            }
        if (failures != 0) break;
    }
    hip_ok(hipFree(d), "free");
    hip_ok(hipFree(ids), "free");
    std::printf("argmax rank %u (%s): %s (row 7 non-finite -> kNonFinite checked)\n", rank, xf.name().c_str(), failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ---- wire-level chain (a trivial consumer) at the GLM sizes
std::vector<float> cpu_replay_chain(unsigned n, unsigned steps) {
    std::vector<float> h(n);
    for (unsigned i = 0; i < n; ++i) h[i] = x0_of(i);
    for (unsigned s = 0; s < steps; ++s)
        for (unsigned i = 0; i < n; ++i) {
            const float p0 = h[i] * prod_a(0) + prod_c(i % kH, s, 0);
            const float p1 = h[i] * prod_a(1) + prod_c(i % kH, s, 1);
            h[i] = h[i] + (p0 + p1);
        }
    return h;
}
ChainResult run_chain(Rank& r, dist::Transport& t, const Arm& a, std::size_t bytes, unsigned steps) {
    std::unique_ptr<ArmScope> scope;
    if (a.path != Path::local) scope = std::make_unique<ArmScope>(r, &t, a, std::size_t{kMaxRows} * kRow);
    const auto n = static_cast<unsigned>(bytes / 4);
    const unsigned g = blocks_for(n);
    hipLaunchKernelGGL(x0_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, n);
    hip_ok(hipStreamSynchronize(r.stream), "init");
    std::uint64_t z = 0, zz = 0;
    t.exchange(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&z), 8), std::span<std::byte>(reinterpret_cast<std::byte*>(&zz), 8));
    const gx::ProducerTarget none{};
    const auto t0 = Clock::now();
    for (unsigned s = 0; s < steps; ++s) {
        const gx::Tag tag{gx::Kind::test, s % 90U, s};
        if (a.path == Path::local) {  // both partials, no wire: the same values as the exchange arms
            hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, r.part, n, s, 0U, none, 0);
            hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, r.part1, n, s, 1U, none, 0);
            hipLaunchKernelGGL(consume_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, r.part, r.part1, n);
        } else if (a.path == Path::sync) {
            hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, r.part, n, s, r.rank, none, 0);
            r.comm->exchange(r.stream, r.part, r.remote, bytes, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), s % 90U, s});
            hipLaunchKernelGGL(consume_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, r.rank == 0 ? r.part : r.remote, r.rank == 0 ? r.remote : r.part, n);
        } else {
            const float* local = nullptr;
            if (a.prod == Prod::copy) {
                hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, r.part, n, s, r.rank, none, 0);
                r.xf->send_copy(r.part, bytes, tag);
                local = r.part;
            } else {
                auto* tx = static_cast<float*>(r.xf->tx());
                const bool fused = a.prod == Prod::fused;
                hipLaunchKernelGGL(produce_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, tx, n, s, r.rank, fused ? r.xf->producer_target() : none, fused ? 1 : 0);
                r.xf->send(bytes, tag, fused ? gx::Publish::producer : gx::Publish::kernel);
                local = tx;
            }
            if (a.kwait) {
                hipLaunchKernelGGL(consume_wait_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, local, r.xf->recv_view(), r.rank, n);
            } else {
                const auto* peer = static_cast<const float*>(r.xf->recv());
                hipLaunchKernelGGL(consume_kernel, dim3(g), dim3(kThreads), 0, r.stream, r.h, r.rank == 0 ? local : peer, r.rank == 0 ? peer : local, n);
            }
        }
    }
    hip_ok(hipStreamSynchronize(r.stream), "chain");
    if (r.xf != nullptr) r.xf->drain();
    ChainResult res;
    res.us = us_since(t0) / steps;
    std::vector<float> h(n);
    hip_ok(hipMemcpy(h.data(), r.h, bytes, hipMemcpyDeviceToHost), "h");
    res.state = fnv(h.data(), bytes);
    return res;
}

std::uint64_t peer_value(dist::Transport& t, std::uint64_t mine) {
    std::uint64_t theirs = 0;
    t.exchange(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&mine), 8), std::span<std::byte>(reinterpret_cast<std::byte*>(&theirs), 8));
    return theirs;
}

// Expect `body` to fail on this rank with a message matching `wanted` (ECMAScript regex; inline and deferred CRC
// failures are told apart by their exact ending), without hanging.
bool expect_failure(unsigned rank, const char* what, Rank& r, const std::string& wanted, const std::function<void()>& body) {
    try {
        body();
        hip_ok(hipStreamSynchronize(r.stream), "sync");
        if (r.xf != nullptr) r.xf->drain();
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (std::regex_search(msg, std::regex(wanted))) {
            std::printf("selftest rank %u %s: failed as required: %s\n", rank, what, e.what());
            return true;
        }
        std::printf("selftest rank %u %s: failed with an UNEXPECTED error: %s  FAIL\n", rank, what, e.what());
        return false;
    }
    std::printf("selftest rank %u %s: NO failure  FAIL\n", rank, what);
    return false;
}

// ---- the engine seam adapter (XferExchange) vs the engine's CommExchange rules: reduce = p0 + p1 in rank order for
// rows 1..8 over several steps; argmax merge R18 with ties, -0/+0 and a lower peer id, from both ranks' views.
int seam_check(unsigned rank, dist::Transport& t) {
    int failures = 0;
    hipStream_t s = nullptr;
    hip_ok(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
    gx::XferExchange xe(t, rank, kMaxRows);
    float* p = nullptr;
    float* v = nullptr;
    std::uint32_t* id = nullptr;
    hip_ok(hipMalloc(reinterpret_cast<void**>(&p), std::size_t{kMaxRows} * kRow), "p");
    hip_ok(hipMalloc(reinterpret_cast<void**>(&v), 32), "v");
    hip_ok(hipMalloc(reinterpret_cast<void**>(&id), 32), "id");
    for (unsigned step = 0; step < 3; ++step) {
        for (unsigned rows = 1; rows <= kMaxRows; ++rows) {
            std::vector<float> mine(std::size_t{rows} * kH), want(mine.size()), got(mine.size());
            for (std::size_t i = 0; i < mine.size(); ++i) {
                const float a = hashf((std::uint64_t{step} << 50U) + (std::uint64_t{rows} << 40U) + i) * 3.0F;
                const float b = i % 17 == 0 ? -a : hashf((std::uint64_t{step} << 50U) + (std::uint64_t{rows} << 40U) + (1ULL << 39U) + i) * 3.0F;
                mine[i] = rank == 0 ? a : b;
                want[i] = a + b;
            }
            hip_ok(hipMemcpyAsync(p, mine.data(), mine.size() * 4, hipMemcpyHostToDevice, s), "up");
            xe.reduce(p, rows, s);
            hip_ok(hipMemcpyAsync(got.data(), p, got.size() * 4, hipMemcpyDeviceToHost, s), "down");
            hip_ok(hipStreamSynchronize(s), "sync");
            if (std::memcmp(got.data(), want.data(), got.size() * 4) != 0) {
                std::printf("seam rank %u reduce rows %u step %u: FAIL\n", rank, rows, step);
                ++failures;
            }
        }
        const float v0[8] = {5.0F, -0.0F, 1.0F, 9.0F, 2.0F, -3.0F, 7.5F, 0.25F};
        const float v1[8] = {5.0F, 0.0F, 2.0F, 8.0F, 2.0F, -2.0F, 7.5F, 0.125F};
        const std::uint32_t i0[8] = {10, 3, 4, 5, 77439, 1, 2, 3}, i1[8] = {77441, 77442, 77443, 77444, 77450, 77441, 77442, 77443};
        std::vector<float> hv(8), wv(8);
        std::vector<std::uint32_t> hi(8), wi(8);
        for (unsigned r = 0; r < 8; ++r) {
            hv[r] = rank == 0 ? v0[r] : v1[r];
            hi[r] = rank == 0 ? i0[r] : i1[r];
            const bool peer_wins = v1[r] > v0[r] || (v1[r] == v0[r] && i1[r] < i0[r]);  // R18 (ids disjoint: symmetric)
            wv[r] = peer_wins ? v1[r] : v0[r];
            wi[r] = peer_wins ? i1[r] : i0[r];
        }
        hip_ok(hipMemcpyAsync(v, hv.data(), 32, hipMemcpyHostToDevice, s), "v");
        hip_ok(hipMemcpyAsync(id, hi.data(), 32, hipMemcpyHostToDevice, s), "id");
        xe.argmax(v, id, 8, s);
        hip_ok(hipMemcpyAsync(hv.data(), v, 32, hipMemcpyDeviceToHost, s), "v");
        hip_ok(hipMemcpyAsync(hi.data(), id, 32, hipMemcpyDeviceToHost, s), "id");
        hip_ok(hipStreamSynchronize(s), "sync");
        for (unsigned r = 0; r < 8; ++r)
            if (hi[r] != wi[r] || std::memcmp(&hv[r], &wv[r], 4) != 0) {
                std::printf("seam rank %u argmax row %u: got (%g, %u) want (%g, %u)  FAIL\n", rank, r, hv[r], hi[r], wv[r], wi[r]);
                ++failures;
            }
    }
    hip_ok(hipFree(p), "free");
    hip_ok(hipFree(v), "free");
    hip_ok(hipFree(id), "free");
    std::printf("seam rank %u (XferExchange: reduce rows 1..8 x 3 steps incl. cancellation, argmax R18 incl. -0/+0 ties): %s\n", rank,
                failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// =============================================================== selftest
int selftest() {
    std::atomic<int> failures{0};
    const unsigned tokens = 2;
    const Replay replay(2, tokens);  // CPU only, before the fork
    std::printf("cpu replay: rows 1 %016llx/%016llx, rows 2 %016llx/%016llx\n", static_cast<unsigned long long>(replay.expect(1).state),
                static_cast<unsigned long long>(replay.expect(1).ids), static_cast<unsigned long long>(replay.expect(2).state),
                static_cast<unsigned long long>(replay.expect(2).ids));
    constexpr int kGroups = 5;  // 0 main, 1 corruption inline, 2 corruption deferred, 3 size mismatch, 4 seam completion
    int sv[kGroups][2][2];
    for (auto& g : sv)
        for (auto& st : g)
            if (::socketpair(AF_UNIX, SOCK_STREAM, 0, st) != 0) throw std::runtime_error("socketpair");
    const pid_t child = ::fork();
    if (child < 0) throw std::runtime_error("fork");
    const unsigned me = child == 0 ? 1U : 0U;
    std::unique_ptr<dist::Transport> tg[kGroups];
    for (int g = 0; g < kGroups; ++g) {
        std::vector<int> fds;
        for (auto& st : sv[g]) {
            fds.push_back(st[me]);
            ::close(st[1 - me]);
        }
        tg[g] = dist::usb4_host_descriptors(fds, {"sp0", "sp1"}, {1, 1}, 0, std::chrono::milliseconds(30000));
    }
    auto rank_main = [&](unsigned rank) {
        if (rank == 0) failures += kernel_gates();  // HIP only after the fork (no HIP state may cross fork())
        Rank r;
        rank_alloc(r, rank);
        const DevWeights dw;
        r.w = &dw;
        std::uint64_t agree = kFnv0;
        for (const Arm& a : token_arms()) {
            if (a.kwait) continue;  // in-kernel waits: pair only (two ranks on one GPU, see mmxfer)
            for (unsigned rows : {1U, 2U})
                for (unsigned depth : {4U, 2U}) {
                    if (depth == 2U && std::string(a.name) != "def-copy") continue;
                    const auto res = run_glm_tokens(r, tg[0].get(), a, rows, tokens, depth);
                    const ChainResult want = replay.expect(rows);
                    agree = (agree ^ res.all()) * kFnvP;
                    if (res.state != want.state || res.ids != want.ids || res.flag != 0) {
                        std::printf("selftest rank %u %s rows %u depth %u: state %016llx ids %016llx flag %u != replay %016llx %016llx  FAIL\n", rank, a.name,
                                    rows, depth, static_cast<unsigned long long>(res.state), static_cast<unsigned long long>(res.ids), res.flag,
                                    static_cast<unsigned long long>(want.state), static_cast<unsigned long long>(want.ids));
                        ++failures;
                    }
                }
        }
        for (bool deferred : {false, true}) {
            const Arm a{"argmax", Path::glmx, Prod::fused, deferred};
            ArmScope scope(r, tg[0].get(), a, std::size_t{kMaxRows} * kRow);
            failures += argmax_check(rank, *r.xf, r.stream);
        }
        {  // seal: a large last message with deferred CRC (drain() seals it), checked against the replay
            const Arm a{"seal", Path::glmx, Prod::fused, true};
            const auto res = run_chain(r, *tg[0], a, std::size_t{kMaxRows} * kRow, 3);
            const auto ref = cpu_replay_chain(kMaxRows * kH, 3);
            if (res.state != fnv(ref.data(), std::size_t{kMaxRows} * kRow)) {
                std::printf("selftest rank %u seal: FAIL\n", rank);
                ++failures;
            }
        }
        failures += seam_check(rank, *tg[0]);
        if (peer_value(*tg[0], agree) != agree) {
            std::printf("selftest rank %u: ranks differ  FAIL\n", rank);
            ++failures;
        }
        for (bool deferred : {false, true}) {
            const Arm a{"corrupt", Path::glmx, Prod::fused, deferred};
            auto o = module_options(r, a, std::size_t{kMaxRows} * kRow);
            o.test_corrupt_sequence = 3;
            o.timeout = std::chrono::milliseconds(5000);
            gx::Exchanger xf(std::make_unique<Borrowed>(*tg[deferred ? 2 : 1]), o);
            r.xf = &xf;
            r.rows = kMaxRows;
            const std::string want = deferred ? R"(payload CRC mismatch at sequence 3 \(deferred check\)$)" : R"(payload CRC mismatch at sequence 3$)";
            if (!expect_failure(rank, deferred ? "corruption (deferred CRC)" : "corruption (inline CRC)", r, want, [&] {
                    chain_init_glm(r, a);
                    for (unsigned s = 0; s < 6; ++s) glm_boundary(r, a, s, s);
                }))
                ++failures;
            r.xf = nullptr;
            hip_ok(hipStreamSynchronize(r.stream), "after corruption");
        }
        {  // seam adapter, reduction-only call (no head): the corrupted LAST message must fail at complete()
            setenv("STRIX_GLMX_TEST_CORRUPT", "3", 1);
            setenv("STRIX_GLMX_TIMEOUT_MS", "5000", 1);
            float* p = nullptr;
            hip_ok(hipMalloc(reinterpret_cast<void**>(&p), kRow), "p");
            hip_ok(hipMemset(p, 0, kRow), "p");
            {
                gx::XferExchange xe(*tg[4], rank, 1);
                Rank probe;
                probe.stream = r.stream;
                if (!expect_failure(rank, "seam complete() after a reduction-only call", probe, R"(payload CRC mismatch at sequence 3 \(deferred check\)$)", [&] {
                        for (int k = 0; k < 3; ++k) xe.reduce(p, 1, r.stream);
                        hip_ok(hipStreamSynchronize(r.stream), "sync");
                        xe.complete();
                    }))
                    ++failures;
            }
            unsetenv("STRIX_GLMX_TEST_CORRUPT");
            unsetenv("STRIX_GLMX_TIMEOUT_MS");
            hip_ok(hipFree(p), "free");
        }
        {
            const Arm a{"mismatch", Path::glmx, Prod::kernel};
            auto o = module_options(r, a, std::size_t{kMaxRows} * kRow);
            o.timeout = std::chrono::milliseconds(5000);
            gx::Exchanger xf(std::make_unique<Borrowed>(*tg[3]), o);
            r.xf = &xf;
            if (!expect_failure(rank, "size mismatch", r, R"(peer message identity differs at sequence 1 \(|the peer closed the stream)", [&] {
                    (void)xf.tx();
                    xf.send(rank == 0 ? 4096 : 8192, gx::Tag{}, gx::Publish::kernel);
                    (void)xf.recv();
                }))
                ++failures;
            r.xf = nullptr;
        }
        hip_ok(hipStreamSynchronize(r.stream), "end");
    };
    try {
        rank_main(me);
    } catch (const std::exception& e) {
        std::printf("rank %u error: %s\n", me, e.what());
        ++failures;
    }
    for (auto& t : tg) t.reset();
    if (child == 0) {
        std::fflush(stdout);
        ::_exit(failures == 0 ? 0 : 1);
    }
    int status = 0;
    if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::printf("selftest: rank 1 process failed (status %d)\n", status);
        ++failures;
    }
    std::printf("selftest: kernel gates, GLM token chains (91 exchanges/token, state + router ids + picked tokens) vs CPU replay "
                "rows 1/2, seam adapter, argmax, seal, corruption x2, mismatch: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL",
                failures.load());
    return failures == 0 ? 0 : 1;
}

// =============================================================== boundary (one node, GPU time per kernel)
int boundary() {
    const DevWeights dw;
    Rank r;
    rank_alloc(r, 0);
    r.w = &dw;
    hipStream_t s = r.stream;
    hipEvent_t e0, e1;
    hip_ok(hipEventCreate(&e0), "event");
    hip_ok(hipEventCreate(&e1), "event");
    hipLaunchKernelGGL(x0_kernel, dim3(blocks_for(kMaxRows * kS)), dim3(kThreads), 0, s, r.X, kMaxRows * kS);
    hip_ok(hipMemcpyAsync(r.X2, r.X, std::size_t{kMaxRows} * kS * 4, hipMemcpyDeviceToDevice, s), "copy");
    hip_ok(hipMemsetAsync(r.part, 0, std::size_t{kMaxRows} * kH * 4, s), "zero");
    hip_ok(hipMemsetAsync(r.remote, 0, std::size_t{kMaxRows} * kH * 4, s), "zero");
    dev::hc_pre(r.X, dw.fn[0], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, kMaxRows, s);
    dev::rmsnorm(r.xc, dw.norm[0], dev::WType::BF16, kH, glm::kEps, r.xin, kMaxRows, s);  // every timed input is defined
    hip_ok(hipStreamSynchronize(s), "init");
    auto time = [&](const char* name, unsigned rows, const std::function<void(unsigned)>& op) {
        std::vector<float> v;
        for (unsigned it = 0; it < 220; ++it) {
            hip_ok(hipEventRecord(e0, s), "rec");
            op(it);
            hip_ok(hipEventRecord(e1, s), "rec");
            hip_ok(hipEventSynchronize(e1), "sync");
            float ms = 0;
            hip_ok(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            if (it >= 20) v.push_back(ms * 1000.0F);
        }
        std::sort(v.begin(), v.end());
        std::printf("BOUNDARY %-24s rows %u: p50 %8.2f us  p10 %8.2f  p90 %8.2f\n", name, rows, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10]);
    };
    for (unsigned rows : {1U, 2U, 4U, 8U}) {
        r.rows = rows;
        time("hc_pre (Tier A)", rows, [&](unsigned it) { dev::hc_pre(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s); });
        time("hc_pre_v2", rows, [&](unsigned it) { gx::hc_pre_v2(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s); });
        time("hc_pre (Tier A) warm fn", rows, [&](unsigned) { dev::hc_pre(r.X, dw.fn[0], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s); });
        time("hc_pre_v3 regs", rows, [&](unsigned it) { gx::hc_pre_v3(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, 0); });
        time("hc_pre_v3 lds", rows, [&](unsigned it) { gx::hc_pre_v3(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, 1); });
        time("hc_pre_v3 lds pipelined", rows, [&](unsigned it) { gx::hc_pre_v3(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, 2); });
        time("router_dense v0", rows, [&](unsigned it) { gx::router_dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], r.logits, rows, s, 0); });
        time("hc_pre_v5 deep", rows, [&](unsigned it) { gx::hc_pre_v3(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, 3); });
        time("router_dense deep", rows, [&](unsigned it) { gx::router_dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], r.logits, rows, s, 2); });
        time("hc_pre_v6 lds-barrier", rows, [&](unsigned it) { gx::hc_pre_v3(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, 4); });
        // design notes (review point 6): each candidate V in {5,6,7} bracketed by the baseline 4 (4,V,4), three reps
        // with the candidate order rotated; the x16 chain probe brackets the whole block (clock-state drift)
        auto chain_probe = [&](const char* tag) {
            const std::string a16 = std::string("fma chain x16 16384 ") + tag, a1 = std::string("fma chain x16 1024 ") + tag;
            time(a16.c_str(), rows, [&](unsigned) { hipLaunchKernelGGL(fma_chain16_kernel, dim3(1), dim3(32), 0, s, dw.fn[0], r.logits, 16384U); });
            time(a1.c_str(), rows, [&](unsigned) { hipLaunchKernelGGL(fma_chain16_kernel, dim3(1), dim3(32), 0, s, dw.fn[0], r.logits, 1024U); });
        };
        auto hc = [&](int v, const std::string& n) {
            time(n.c_str(), rows, [&](unsigned it) { gx::hc_pre_v3(r.X, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, v); });
        };
        chain_probe("pre");
        for (int rep = 0; rep < 3; ++rep)
            for (int i = 0; i < 3; ++i) {
                const int v = 5 + (i + rep) % 3;
                const std::string tag = " rep" + std::to_string(rep) + " v" + std::to_string(v);
                hc(4, "hc_pre A4" + tag + " a");
                hc(v, "hc_pre B" + std::to_string(v) + tag);
                hc(4, "hc_pre A4" + tag + " b");
            }
        chain_probe("post");
        for (int pf : {0, 1, 3}) {
            const std::string n = "chains lds-only pf" + std::to_string(pf);
            time(n.c_str(), rows, [&](unsigned it) { gx::hc_chains_lds_only(r.F, dw.fn[it % kSub], r.mix, s, pf); });
        }
        time("router_dense v4 36 blk", rows, [&](unsigned it) { gx::router_dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], r.logits, rows, s, 3); });
        time("fma chain floor 16384", rows, [&](unsigned) { hipLaunchKernelGGL(fma_chain_kernel, dim3(1), dim3(32), 0, s, dw.fn[0], r.logits, 16384U); });
        time("fma chain x16 16384", rows, [&](unsigned) { hipLaunchKernelGGL(fma_chain16_kernel, dim3(1), dim3(32), 0, s, dw.fn[0], r.logits, 16384U); });
        time("fma chain x16 1024", rows, [&](unsigned) { hipLaunchKernelGGL(fma_chain16_kernel, dim3(1), dim3(32), 0, s, dw.fn[0], r.logits, 1024U); });
        time("router_dense pipelined", rows, [&](unsigned it) { gx::router_dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], r.logits, rows, s, 1); });
        time("route_v2", rows, [&](unsigned it) { gx::route_v2(r.logits, dw.bias[glm::kFirstMoe + it % 42U], r.ids, r.rw, rows, s); });
        time("boundary attn->ffn (fast)", rows, [&](unsigned it) {
            gx::post_expand(r.X, r.mix, r.part, r.remote, gx::RxView{}, 0, r.X2, r.flag, rows, s);
            gx::hc_pre_v3(r.X2, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, 2);
            dev::rmsnorm(r.xc, dw.norm[it % kSub], dev::WType::BF16, kH, glm::kEps, r.xin, rows, s);
            gx::router_dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], r.logits, rows, s, 1);
            gx::route_v2(r.logits, dw.bias[glm::kFirstMoe + it % 42U], r.ids, r.rw, rows, s);
        });
        time("boundary ffn->attn (fast)", rows, [&](unsigned it) {
            gx::post_expand(r.X, r.mix, r.part, r.remote, gx::RxView{}, 0, r.X2, r.flag, rows, s);
            gx::hc_pre_v3(r.X2, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s, 2);
            dev::rmsnorm(r.xc, dw.norm[it % kSub], dev::WType::BF16, kH, glm::kEps, r.xin, rows, s);
        });
        time("hc_pre R1 chains only", rows, [&](unsigned it) {
            hipLaunchKernelGGL(chains_only_kernel, dim3(rows), dim3(256), 0, s, r.F, dw.fn[it % kSub], r.mix);
        });
        time("producer (1 rank)", rows, [&](unsigned it) {
            hipLaunchKernelGGL(produce_kernel, dim3(blocks_for(rows * kH)), dim3(kThreads), 0, s, r.xin, r.part, rows * kH, it, 0U, gx::ProducerTarget{}, 0);
        });
        time("head slice logits+argmax", rows, [&](unsigned) {
            hipLaunchKernelGGL(head_logits_kernel, dim3(blocks_for(rows * kH)), dim3(kThreads), 0, s, r.xin, r.hl, 1U, rows);
            dev::argmax(r.hl, kHeadV, kHeadV, r.amax_v, r.amax_id, r.flag, rows, s);
        });
        time("post_expand", rows, [&](unsigned) { gx::post_expand(r.X, r.mix, r.part, r.remote, gx::RxView{}, 0, r.X2, r.flag, rows, s); });
        time("hc_bracket", rows, [&](unsigned) { gx::hc_bracket(r.X, r.mix, r.B, rows, s); });
        time("post_expand_bracket", rows, [&](unsigned) { gx::post_expand_bracket(r.B, r.mix, r.part, r.remote, gx::RxView{}, 0, r.X2, r.flag, rows, s); });
        time("rmsnorm (weighted)", rows, [&](unsigned it) { dev::rmsnorm(r.xc, dw.norm[it % kSub], dev::WType::BF16, kH, glm::kEps, r.xin, rows, s); });
        time("router dense 288x4096", rows, [&](unsigned it) {
            dev::dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], dev::WType::F16, kH, 0, glm::kExperts, r.logits, rows, s);
        });
        time("route", rows, [&](unsigned it) { dev::route(r.logits, dw.bias[glm::kFirstMoe + it % 42U], r.ids, r.rw, rows, s); });
        time("boundary attn->ffn (v1)", rows, [&](unsigned it) {
            gx::post_expand(r.X, r.mix, r.part, r.remote, gx::RxView{}, 0, r.X2, r.flag, rows, s);
            dev::hc_pre(r.X2, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s);
            dev::rmsnorm(r.xc, dw.norm[it % kSub], dev::WType::BF16, kH, glm::kEps, r.xin, rows, s);
            dev::dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], dev::WType::F16, kH, 0, glm::kExperts, r.logits, rows, s);
            dev::route(r.logits, dw.bias[glm::kFirstMoe + it % 42U], r.ids, r.rw, rows, s);
        });
        time("boundary attn->ffn (v2)", rows, [&](unsigned it) {
            gx::post_expand(r.X, r.mix, r.part, r.remote, gx::RxView{}, 0, r.X2, r.flag, rows, s);
            gx::hc_pre_v2(r.X2, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s);
            dev::rmsnorm(r.xc, dw.norm[it % kSub], dev::WType::BF16, kH, glm::kEps, r.xin, rows, s);
            dev::dense(r.xin, dw.router[glm::kFirstMoe + it % 42U], dev::WType::F16, kH, 0, glm::kExperts, r.logits, rows, s);
            dev::route(r.logits, dw.bias[glm::kFirstMoe + it % 42U], r.ids, r.rw, rows, s);
        });
        time("boundary ffn->attn (v2)", rows, [&](unsigned it) {
            gx::post_expand(r.X, r.mix, r.part, r.remote, gx::RxView{}, 0, r.X2, r.flag, rows, s);
            gx::hc_pre_v2(r.X2, dw.fn[it % kSub], dw.base[0], dw.scale[0], r.F, r.mix, r.xc, rows, s);
            dev::rmsnorm(r.xc, dw.norm[it % kSub], dev::WType::BF16, kH, glm::kEps, r.xin, rows, s);
        });
    }
    for (unsigned rows : {1U, 8U}) {
        std::vector<std::string> done;  // each local baseline once
        for (const Arm& a : token_arms()) {
            if (a.path != Path::local || std::find(done.begin(), done.end(), std::string(a.name)) != done.end()) continue;
            done.emplace_back(a.name);
            std::vector<double> v;
            for (int rep = 0; rep < 5; ++rep) v.push_back(run_glm_tokens(r, nullptr, a, rows, a.tokens).us);
            std::sort(v.begin(), v.end());
            std::printf("BOUNDARY token %-14s rows %u: %8.1f us/token (median of 5 x %u tokens, 91 boundaries + head, no wire)\n", a.name, rows,
                        v[2], a.tokens);
        }
    }
    return 0;
}

// =============================================================== pair
struct Pct {
    double p10, p50, p90, p99;
};
Pct pct(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))]; };
    return {at(0.10), at(0.50), at(0.90), at(0.99)};
}
struct Cfg {
    const char* name;
    const char* eager;  // STRIX_DIST_USB4_EAGER
};
std::unique_ptr<dist::Transport> open_pair(unsigned rank, const Cfg& c) {
    setenv("STRIX_DIST_USB4_SPIN", "0", 1);
    setenv("STRIX_DIST_USB4_EAGER", c.eager, 1);
    dist::Usb4AutoOptions o;
    o.rank = rank;
    o.timeout = std::chrono::milliseconds(30000);
    dist::Usb4Topology topo;
    auto t = dist::usb4_host_auto(o, &topo);
    std::printf("PAIRING %s: %s\n", c.name, topo.receipt().c_str());
    return t;
}

void curve(unsigned rank, dist::Transport& t, const Cfg& c, std::FILE* out, const std::vector<std::size_t>& sizes) {
    const std::size_t maxb = *std::max_element(sizes.begin(), sizes.end()) + 64;
    std::vector<std::byte> a(maxb), b(maxb);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<std::byte>((i * 31 + rank) & 0xFF);
    for (const std::size_t payload : sizes) {
        const std::size_t m = payload + 64;
        const unsigned iters = payload <= 65536 ? 400 : payload <= (1U << 20U) ? 120 : 20;
        std::vector<double> ex;
        for (unsigned i = 0; i < iters + 20; ++i) {
            const auto t0 = Clock::now();
            t.exchange(std::span<const std::byte>(a.data(), m), std::span<std::byte>(b.data(), m));
            if (i >= 20) ex.push_back(us_since(t0));
        }
        const Pct e = pct(ex);
        std::fprintf(out, "{\"phase\":\"curve\",\"cfg\":\"%s\",\"rank\":%u,\"payload\":%zu,\"ex_p10\":%.2f,\"ex_p50\":%.2f,\"ex_p90\":%.2f,\"ex_p99\":%.2f}\n",
                     c.name, rank, payload, e.p10, e.p50, e.p90, e.p99);
        if (rank == 0)
            std::printf("CURVE %-12s %9zu B: exchange p50 %8.2f p90 %8.2f p99 %8.2f us (%.2f GB/s/dir)\n", c.name, payload, e.p50, e.p90, e.p99,
                        static_cast<double>(m) / e.p50 / 1000.0);
    }
}

int pair(unsigned rank, const std::string& out_dir) {
    const std::string path = out_dir + "/glmxch-r" + std::to_string(rank) + ".jsonl";
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (out == nullptr) throw std::runtime_error("cannot write " + path);
    setvbuf(out, nullptr, _IOLBF, 0);
    int failures = 0;
    const auto T0 = Clock::now();
    const unsigned tokens = 8;
    const Replay replay(kMaxRows, tokens), replay4(kMaxRows, 2);  // CPU, one thread per row, before pairing (8 and 2 tokens)
    std::printf("replay done %.1f s\n", us_since(T0) / 1e6);
    Rank r;
    rank_alloc(r, rank);
    const DevWeights dw;
    r.w = &dw;
    const Cfg base{"4s", "65536"};
    std::unique_ptr<dist::Transport> t = open_pair(rank, base);
    // ---- (1) curve at the GLM sizes
    curve(rank, *t, base, out, {8, 64, 512, kRow, 2 * kRow, 4 * kRow, 8 * kRow, 16 * kRow, std::size_t{1} << 20U, std::size_t{4} << 20U, std::size_t{2048} * kRow});
    std::printf("phase curve done %.1f s\n", us_since(T0) / 1e6);
    // ---- (2) wire-level chain, 3 reps ABA
    {
        const std::vector<Arm> arms = {{"local", Path::local},
                                       {"sync", Path::sync},
                                       {"def-copy", Path::glmx, Prod::copy},
                                       {"defkw-copy", Path::glmx, Prod::copy, true, true},
                                       {"defkw-fused", Path::glmx, Prod::fused, true, true},
                                       {"defkw-fused-txunc", Path::glmx, Prod::fused, true, true, false, true, UM, HM}};
        const unsigned steps = 600;
        for (int rep = 0; rep < 3; ++rep)
            for (const std::size_t b : {std::size_t{8}, kRow, std::size_t{kMaxRows} * kRow}) {
                const auto ref = cpu_replay_chain(static_cast<unsigned>(b / 4), steps);
                const std::uint64_t want = fnv(ref.data(), b);
                std::vector<Arm> order = arms;
                if (rep == 1) std::reverse(order.begin(), order.end());
                for (const Arm& a : order) {
                    const auto res = run_chain(r, *t, a, b, steps);
                    const std::uint64_t peer = peer_value(*t, res.state);
                    const bool ok = res.state == want && peer == res.state;  // local computes both partials: same bits
                    std::fprintf(out, "{\"phase\":\"chain\",\"rank\":%u,\"rep\":%d,\"arm\":\"%s\",\"bytes\":%zu,\"us_per_step\":%.2f,\"bitwise\":%s}\n", rank, rep,
                                 a.name, b, res.us, ok ? "true" : "false");
                    if (rank == 0) std::printf("CHAIN rep %d %-18s %7zu B: %8.2f us/step %s\n", rep, a.name, b, res.us, ok ? "bitwise ok" : "BITWISE FAIL");
                    if (!ok) throw std::runtime_error(std::string("chain mismatch: ") + a.name);
                }
            }
        std::printf("phase chain done %.1f s\n", us_since(T0) / 1e6);
    }
    // ---- (3) GLM token simulation: 91 exchanges per token with the real consumers
    {
        const std::vector<Arm> arms = token_arms();
        for (unsigned rows : {1U, 8U}) {
            for (int rep = 0; rep < 3; ++rep) {
                std::vector<Arm> order = arms;
                if (rep == 1) std::reverse(order.begin(), order.end());
                for (const Arm& a : order) {
                    const ChainResult want = a.tokens == tokens ? replay.expect(rows) : replay4.expect(rows);
                    if (a.tokens != tokens && a.tokens != 2) throw std::logic_error("replay token count");
                    const auto res = run_glm_tokens(r, t.get(), a, rows, a.tokens);
                    const std::uint64_t peer = peer_value(*t, res.all());
                    const bool ok = res.state == want.state && res.ids == want.ids && res.flag == 0 && peer == res.all();
                    std::fprintf(out, "{\"phase\":\"token\",\"cfg\":\"4s\",\"rank\":%u,\"rep\":%d,\"arm\":\"%s\",\"base\":\"%s\",\"rows\":%u,\"us_per_token\":%.1f,\"ok\":%s}\n",
                                 rank, rep, a.name, a.base, rows, res.us, ok ? "true" : "false");
                    if (rank == 0) std::printf("TOKEN N=%u rep %d %-18s: %8.1f us/token %s\n", rows, rep, a.name, res.us, ok ? "== replay" : "MISMATCH");
                    if (!ok) throw std::runtime_error(std::string("token mismatch: ") + a.name);
                }
            }
        }
        std::printf("phase token done %.1f s\n", us_since(T0) / 1e6);
    }
    {  // greedy head split on the pair
        const Arm a{"argmax", Path::glmx, Prod::fused, true};
        ArmScope scope(r, t.get(), a, std::size_t{kMaxRows} * kRow);
        const int f = argmax_check(rank, *r.xf, r.stream);
        std::fprintf(out, "{\"phase\":\"argmax\",\"rank\":%u,\"pass\":%s}\n", rank, f == 0 ? "true" : "false");
        if (f != 0) throw std::runtime_error("argmax mismatch on the pair");
    }
    // ---- (4) transport calibration (eager threshold), base first and last; each config re-pairs on both ranks
    {
        const Arm best = fast_arm({"", Path::glmx, Prod::fused, true, true, true, true}, "defkw-fused-br-fast", "local-fast-br");
        const Cfg cfgs[] = {base, {"4s-eager32k", "32768"}, {"4s-eager160k", "163840"}, base};
        for (std::size_t ci = 0; ci < std::size(cfgs); ++ci) {
            const Cfg& c = cfgs[ci];
            (void)peer_value(*t, ci);  // both ranks reach the re-pairing together
            t.reset();
            t = open_pair(rank, c);
            curve(rank, *t, c, out, {kRow, 2 * kRow, 4 * kRow, 8 * kRow});
            for (unsigned rows : {1U, 8U}) {
                const auto res = run_glm_tokens(r, t.get(), best, rows, tokens);
                const ChainResult want = replay.expect(rows);
                const std::uint64_t peer = peer_value(*t, res.all());
                const bool ok = res.state == want.state && res.ids == want.ids && res.flag == 0 && peer == res.all();
                std::fprintf(out, "{\"phase\":\"calib_token\",\"cfg\":\"%s\",\"pos\":%zu,\"rank\":%u,\"rows\":%u,\"us_per_token\":%.1f,\"ok\":%s}\n", c.name, ci, rank,
                             rows, res.us, ok ? "true" : "false");
                if (rank == 0) std::printf("CALIB %-14s N=%u: %8.1f us/token %s\n", c.name, rows, res.us, ok ? "== replay" : "MISMATCH");
                if (!ok) throw std::runtime_error("calibration mismatch");
            }
        }
    }
    std::fclose(out);
    std::printf("rank %u: pair bench %s, %d failures, %.1f s\n", rank, failures == 0 ? "PASS" : "FAIL", failures, us_since(T0) / 1e6);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "selftest") return selftest();
        if (mode == "boundary") return boundary();
        if (mode == "gates") return kernel_gates() == 0 ? 0 : 1;
        if (mode == "pair" && argc >= 4) return pair(static_cast<unsigned>(std::stoul(argv[2])), argv[3]);
        std::fprintf(stderr, "usage: %s selftest | gates | boundary | pair RANK OUT_DIR\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glmxch-bench error: %s\n", e.what());
        return 3;
    }
}
