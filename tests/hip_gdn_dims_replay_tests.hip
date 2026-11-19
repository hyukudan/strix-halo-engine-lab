// Device test of the rollback replay of the qwen35 HIP engine (src/qwen35/hip/engine.hip rollback): a verify window runs
// hip_gdn_decode_step_dims OUT OF PLACE (matrix_out = the spare buffer: rows_kernel for every row), and commit(k < rows)
// replays the first k rows IN PLACE on the committed buffer, where k >= the chunk of the shared cols core runs that core
// for the bulk and rows_kernel for the tail. The committed state must be the window's state after k rows, bit for bit.
// For k = 1..16 from one random state and raw inputs: out-of-place window of k rows vs in-place replay of k rows ->
// equal state, outputs and conv history (bitwise). Shape: Qwen 27B dense GDN (16 key / 48 value heads x 128, kernel 4).
// Run: strix-hip-gdn-dims-replay-tests (prints "gdn dims replay: PASS k 1..16"); exit 1 on any difference.
#include "strix/hip_gdn_dims.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::uint32_t kKeys = 16, kValues = 48, kDim = 128, kKernel = 4, kMaxRows = 16;
constexpr std::size_t kChannels = std::size_t(2U * kKeys + kValues) * kDim;
constexpr std::size_t kState = std::size_t(kValues) * kDim * kDim;

void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "gdn dims replay: %s failed: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

struct Rng {
    std::uint64_t s = 0x9e3779b97f4a7c15ULL;
    float uniform(float lo, float hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + (hi - lo) * float((s >> 40U) & 0xffffffULL) / 16777216.0F;
    }
};

std::vector<float> random(std::size_t n, float lo, float hi, Rng& r) {
    std::vector<float> v(n);
    for (auto& x : v) x = r.uniform(lo, hi);
    return v;
}

template <class T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    hip_ok(hipMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(T)), "hipMalloc");
    hip_ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

template <class T>
std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip_ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}

template <class T>
void reset(T* d, const std::vector<T>& h) {
    hip_ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "reset");
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

}  // namespace

int main() {
    Rng rng;
    // Raw window inputs (the engine's replay slot: qkv | beta | gate before the in-place conv / transform) and weights.
    const auto qkv0 = random(kChannels * kMaxRows, -1.0F, 1.0F, rng);
    const auto beta0 = random(std::size_t(kValues) * kMaxRows, -2.0F, 2.0F, rng);
    const auto gate0 = random(std::size_t(kValues) * kMaxRows, -2.0F, 2.0F, rng);
    const auto z = random(std::size_t(kValues) * kDim * kMaxRows, -1.0F, 1.0F, rng);
    const auto state0 = random(kState, -0.05F, 0.05F, rng);
    const auto conv_w = random(kChannels * kKernel, -0.5F, 0.5F, rng);
    const auto dt = random(kValues, -1.0F, 1.0F, rng);
    const auto a = random(kValues, -1.0F, 0.5F, rng);
    const auto norm = random(std::size_t(kValues) * kDim, 0.5F, 1.5F, rng);
    std::vector<std::uint16_t> conv0(kChannels * kKernel);
    for (auto& c : conv0) {
        const float f = rng.uniform(-1.0F, 1.0F);
        std::uint32_t bits;
        std::memcpy(&bits, &f, 4);
        c = std::uint16_t(bits >> 16U);
    }
    float *qkv = upload(qkv0), *beta = upload(beta0), *gate = upload(gate0), *zd = upload(z), *mat = upload(state0);
    float *out = upload(state0), *core = upload(std::vector<float>(std::size_t(kValues) * kDim * kMaxRows, 0.0F));
    float *cw = upload(conv_w), *dtd = upload(dt), *ad = upload(a), *nd = upload(norm);
    std::uint16_t* conv = upload(conv0);
    hipStream_t s = nullptr;
    hip_ok(hipStreamCreate(&s), "hipStreamCreate");

    auto run = [&](std::uint32_t k, bool in_place) {
        reset(qkv, qkv0); reset(beta, beta0); reset(gate, gate0); reset(mat, state0); reset(out, state0); reset(conv, conv0);
        strix::HipGdnDimsStep g{};
        g.qkv = qkv; g.beta = beta; g.gate = gate; g.core = core; g.matrix = mat; g.z = zd;
        g.conv_state = conv; g.conv = cw; g.dt = dtd; g.a = ad; g.norm = nd;
        g.small_type = 0; g.key_heads = kKeys; g.value_heads = kValues; g.head_dim = kDim; g.kernel = kKernel; g.rows = k;
        g.qk_epsilon = 1e-6F; g.rms_epsilon = 1e-6F; g.norm_kind = strix::HipGdnGatedNorm::swish;
        g.matrix_out = in_place ? nullptr : out;
        strix::hip_gdn_decode_step_dims(g, s);
        hip_ok(hipStreamSynchronize(s), "step");
        struct R { std::vector<float> state, core; std::vector<std::uint16_t> conv; } r;
        r.state = download(in_place ? mat : out, kState);
        r.core = download(core, std::size_t(kValues) * kDim * k);
        r.conv = download(conv, conv0.size());
        if (!in_place && !same_bits(download(mat, kState), state0)) {
            std::fprintf(stderr, "gdn dims replay: k %u out-of-place window modified the checkpoint\n", k);
            std::exit(1);
        }
        return r;
    };
    unsigned failures = 0;
    for (std::uint32_t k = 1; k <= kMaxRows; ++k) {
        const auto window = run(k, false);  // the verify window's arithmetic for k rows (rows_kernel, out of place)
        const auto replay = run(k, true);   // the rollback replay (chunked bulk + rows_kernel tail, in place)
        const bool st = same_bits(window.state, replay.state), co = same_bits(window.core, replay.core),
                   cv = same_bits(window.conv, replay.conv);
        std::printf("k %2u: state %s, outputs %s, conv history %s\n", k, st ? "bitwise" : "DIFFER", co ? "bitwise" : "DIFFER",
                    cv ? "bitwise" : "DIFFER");
        failures += (st && co && cv) ? 0U : 1U;
    }
    for (void* p : {static_cast<void*>(qkv), static_cast<void*>(beta), static_cast<void*>(gate), static_cast<void*>(zd),
                    static_cast<void*>(mat), static_cast<void*>(out), static_cast<void*>(core), static_cast<void*>(cw),
                    static_cast<void*>(dtd), static_cast<void*>(ad), static_cast<void*>(nd), static_cast<void*>(conv)})
        (void)hipFree(p);
    (void)hipStreamDestroy(s);
    if (failures) {
        std::printf("gdn dims replay: FAIL (%u of %u k)\n", failures, kMaxRows);
        return 1;
    }
    std::printf("gdn dims replay: PASS k 1..%u\n", kMaxRows);
    return 0;
}
