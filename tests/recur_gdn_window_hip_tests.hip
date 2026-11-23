// F9 gated delta window form, device fixture (src/ops/recur/gated_delta_window.hpp).
// Consecutive deferred-commit windows run through the real prepare / recurrence / norm kernels and the standalone commit
// kernel, with the saved-row slot alternating from window to window and distinct rows in every window, against the
// serial form (prefill_prepare / prefill_recurrent / prefill_norm: every row in order, no deferral, no saved slots).
// Compared byte for byte: the prepared qkv / beta / gate planes, the final gated core, the committed conv history
// (coded slots) and the committed recurrent state. Two routes: the single-session kernels (windows of 1..4 rows) and the
// cohort kernels (two sessions in one launch, blockIdx.y = segment, counts 1..8 per segment and window).
// A window that saves into the slot it still has to commit (`next = slot`) overwrites the pending rows before the
// recurrence reads them and the committed state and history differ from the serial run.
#include <hip/hip_runtime.h>

#include "ops/recur/gated_delta_window.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace win = ::strix::ops::recur::window;
using ::strix::GdnRecurrentStorage;

constexpr std::uint32_t kD = 128, kKeyHeads = 2, kValueHeads = 4, kTaps = 4, kMaxRows = 8;
constexpr std::uint32_t kChannels = (2U * kKeyHeads + kValueHeads) * kD;  // q | k | v
constexpr std::uint32_t kKeyWidth = kKeyHeads * kD, kValueWidth = kValueHeads * kD;
constexpr std::uint64_t kStride = kChannels;  // floats per row plane (qkv, z, beta, gate, core)
constexpr GdnRecurrentStorage kStorage = GdnRecurrentStorage::key_value;
constexpr float kEpsilon = 1.0e-6F;
int failures = 0;

// bf16-coded history slot (round to nearest even), decode / round as the binder's codec would.
struct Bf16Codec {
    using Slot = std::uint16_t;
    __host__ __device__ static Slot encode(float x) {
        std::uint32_t b;
        std::memcpy(&b, &x, sizeof b);
        return static_cast<Slot>((b + 0x7FFFU + ((b >> 16U) & 1U)) >> 16U);
    }
    __host__ __device__ static float decode(Slot s) {
        const std::uint32_t b = static_cast<std::uint32_t>(s) << 16U;
        float x;
        std::memcpy(&x, &b, sizeof x);
        return x;
    }
    __host__ __device__ static float round(float x) { return decode(encode(x)); }
};
struct F32Load {
    enum class Kind : std::uint32_t { f32 = 0 };
    __host__ __device__ static float load(const void* p, Kind, std::uint64_t index) {
        return static_cast<const float*>(p)[index];
    }
};
struct WinControl {
    std::uint32_t pending_rows;
    std::uint32_t pending_slot;
};
struct Binder {
    using Codec = Bf16Codec;
    using Load = F32Load;
    using Gate = ::strix::ops::recur::SoftplusGate;
    using Control = WinControl;
    static constexpr std::uint32_t max_rows = kMaxRows, taps = kTaps;
};
constexpr auto kKind = F32Load::Kind::f32;
using Slot = Bf16Codec::Slot;

struct Rng {
    std::uint64_t s;
    float next(float lo, float hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + (hi - lo) * static_cast<float>(static_cast<std::uint32_t>(s >> 40) & 0xFFFFFFU) / 16777216.0F;
    }
};

template <class T> struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    bool put(const std::vector<T>& h) {
        n = h.size();
        return hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)) == hipSuccess &&
               hipMemcpy(p, h.data(), n * sizeof(T), hipMemcpyHostToDevice) == hipSuccess;
    }
    bool get(std::vector<T>& h) const {
        h.resize(n);
        return hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess;
    }
    ~Dev() {
        if (p != nullptr) (void)hipFree(p);
    }
};
template <class T> bool bytes_equal(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// Saved row layout: conv input slots (one per channel), then k, v, beta, gate planes.
constexpr std::uint64_t kSavedK = (kChannels * sizeof(Slot) + 15U) / 16U * 16U;
constexpr std::uint64_t kSavedV = kSavedK + kKeyWidth * sizeof(float);
constexpr std::uint64_t kSavedBeta = kSavedV + kValueWidth * sizeof(float);
constexpr std::uint64_t kSavedGate = kSavedBeta + kValueHeads * sizeof(float);
constexpr std::uint64_t kSavedRowBytes = (kSavedGate + kValueHeads * sizeof(float) + 15U) / 16U * 16U;
constexpr std::uint64_t kSavedBytes = 2ULL * kMaxRows * kSavedRowBytes;

// Operands shared by every session (weights, dt, a, norm weights).
struct Weights {
    Dev<float> conv, dt, a, norm;
};
// Host-side initial data of one session and its device copies for one route.
struct Init {
    std::vector<float> qkv, z, beta, gate, state;
    std::vector<Slot> history;
};
struct Route {
    Dev<float> qkv, z, beta, gate, core, state;
    Dev<Slot> history;
    Dev<std::uint8_t> saved;
    Dev<WinControl> control;
    bool put(const Init& in, std::size_t rows) {
        return qkv.put(in.qkv) && z.put(in.z) && beta.put(in.beta) && gate.put(in.gate) &&
               core.put(std::vector<float>(rows * kStride, 0.0F)) && state.put(in.state) && history.put(in.history) &&
               saved.put(std::vector<std::uint8_t>(kSavedBytes, 0)) && control.put(std::vector<WinControl>(1));
    }
    win::SavedRows saved_rows() const {
        return win::SavedRows{reinterpret_cast<std::uintptr_t>(saved.p), kSavedRowBytes, kSavedK, kSavedV, kSavedBeta,
                              kSavedGate};
    }
    win::StridedRows strided() const { return win::StridedRows{qkv.p, z.p, beta.p, gate.p, core.p, kStride}; }
    template <unsigned N> win::RowSet<N> rowset(std::size_t begin) const {
        win::RowSet<N> r{};
        for (unsigned i = 0; i < N; ++i) {
            const std::size_t o = (begin + i) * kStride;
            r.qkv[i] = qkv.p + o; r.z[i] = z.p + o; r.beta[i] = beta.p + o; r.gate[i] = gate.p + o; r.core[i] = core.p + o;
        }
        return r;
    }
    bool set_control(std::uint32_t pending, std::uint32_t slot) const {
        const WinControl c{pending, slot};
        return hipMemcpy(control.p, &c, sizeof c, hipMemcpyHostToDevice) == hipSuccess;
    }
};

Init make_init(std::uint64_t seed, std::size_t rows) {
    Rng rng{seed};
    Init in;
    in.qkv.resize(rows * kStride); in.z.resize(rows * kStride); in.beta.resize(rows * kStride); in.gate.resize(rows * kStride);
    for (auto& x : in.qkv) x = rng.next(-1.5F, 1.5F);
    for (auto& x : in.z) x = rng.next(-2.0F, 2.0F);
    for (auto& x : in.beta) x = rng.next(-2.0F, 2.0F);
    for (auto& x : in.gate) x = rng.next(-1.0F, 1.0F);
    in.state.resize(static_cast<std::size_t>(kValueHeads) * kD * kD);
    for (auto& x : in.state) x = rng.next(-0.1F, 0.1F);
    in.history.resize(static_cast<std::size_t>(kChannels) * kTaps);
    for (auto& s : in.history) s = Bf16Codec::encode(rng.next(-1.0F, 1.0F));
    return in;
}

template <class F> void with_rows(unsigned count, F&& f) {
    switch (count) {
    case 1: f(std::integral_constant<unsigned, 1>{}); break;
    case 2: f(std::integral_constant<unsigned, 2>{}); break;
    case 3: f(std::integral_constant<unsigned, 3>{}); break;
    case 4: f(std::integral_constant<unsigned, 4>{}); break;
    case 5: f(std::integral_constant<unsigned, 5>{}); break;
    case 6: f(std::integral_constant<unsigned, 6>{}); break;
    case 7: f(std::integral_constant<unsigned, 7>{}); break;
    default: f(std::integral_constant<unsigned, 8>{}); break;
    }
}

constexpr std::uint32_t kPrepareBlocks = 2U * kKeyHeads + kValueHeads + 1U;
constexpr std::uint32_t kTiles = kValueHeads * (kD / win::kWindowColumns);
constexpr std::uint32_t kCommitBlocks = kTiles + kChannels / 32U;

void run_serial(const Route& r, std::uint32_t count, const Weights& w) {
    win::prefill_prepare_kernel<Binder><<<kPrepareBlocks, win::kWindowThreads>>>(
        r.strided(), count, r.history.p, w.conv.p, kKind, w.dt.p, kKind, w.a.p, kKind, kKeyHeads, kValueHeads, kD, kEpsilon);
    win::prefill_recurrent_kernel<kStorage><<<kTiles, 32>>>(r.strided(), count, r.state.p, kKeyHeads, kD);
    win::prefill_norm_kernel<Binder><<<dim3(kValueHeads, count), win::kWindowThreads>>>(r.strided(), w.norm.p, kKind, kD,
                                                                                         kEpsilon);
}

// Single-session kernels: windows of the given sizes, then the standalone commit of the last window's rows.
bool run_windows(const Route& r, const std::vector<unsigned>& sizes, const Weights& w) {
    std::size_t begin = 0;
    unsigned previous = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        // rows saved by window i-1 live in slot (i - 1) & 1; this window saves into the other one.
        if (!r.set_control(previous, static_cast<std::uint32_t>(i + 1U) & 1U)) return false;
        with_rows(sizes[i], [&](auto n) {
            constexpr unsigned N = decltype(n)::value;
            const auto rows = r.rowset<N>(begin);
            win::prepare_kernel<Binder, N><<<kPrepareBlocks, win::kWindowThreads>>>(
                rows, r.history.p, w.conv.p, kKind, w.dt.p, kKind, w.a.p, kKind, r.saved_rows(), r.control.p, kKeyHeads,
                kValueHeads, kD, kEpsilon);
            win::recurrent_kernel<Binder, N, kStorage><<<kTiles, 32>>>(rows, r.state.p, r.saved_rows(), r.control.p,
                                                                      kKeyHeads, kD);
            win::norm_kernel<Binder, N><<<dim3(kValueHeads, N), win::kWindowThreads>>>(rows, w.norm.p, kKind, kD, kEpsilon);
        });
        if (hipDeviceSynchronize() != hipSuccess) return false;
        begin += sizes[i];
        previous = sizes[i];
    }
    if (!r.set_control(previous, (static_cast<std::uint32_t>(sizes.size()) & 1U) ^ 1U)) return false;
    win::commit_kernel<Binder, kStorage><<<kCommitBlocks, 32>>>(r.state.p, r.history.p, r.saved_rows(), r.control.p,
                                                                kKeyHeads, kValueHeads, kD);
    return hipDeviceSynchronize() == hipSuccess;
}

// Cohort kernels: segment s runs counts[s][i] rows in window i (every segment has the same number of windows).
bool run_cohort(const std::vector<const Route*>& routes, const std::vector<std::vector<unsigned>>& counts,
                const Weights& w) {
    const std::size_t windows = counts[0].size();
    std::vector<std::size_t> begin(routes.size(), 0);
    std::vector<unsigned> previous(routes.size(), 0);
    for (std::size_t i = 0; i < windows; ++i) {
        win::Segments<Binder> seg{};
        for (std::size_t s = 0; s < routes.size(); ++s) {
            const Route& r = *routes[s];
            if (!r.set_control(previous[s], static_cast<std::uint32_t>(i + 1U) & 1U)) return false;
            win::Segment<Binder>& g = seg.segment[s];
            g.rows = r.rowset<kMaxRows>(begin[s]);
            g.count = counts[s][i];
            g.conv_state = r.history.p;
            g.matrix = r.state.p;
            g.saved = r.saved_rows();
            g.control = r.control.p;
        }
        const auto segments = static_cast<unsigned>(routes.size());
        win::prepare_anyseg_kernel<Binder, false><<<dim3(kPrepareBlocks, segments), win::kWindowThreads>>>(
            seg, w.conv.p, kKind, w.dt.p, kKind, w.a.p, kKind, kKeyHeads, kValueHeads, kD, kEpsilon);
        win::recurrent_anyseg_kernel<Binder, kStorage, win::kWindowColumns><<<dim3(kTiles, segments), 32>>>(seg, kKeyHeads, kD);
        for (std::size_t s = 0; s < routes.size(); ++s)
            with_rows(counts[s][i], [&](auto n) {
                constexpr unsigned N = decltype(n)::value;
                win::norm_kernel<Binder, N><<<dim3(kValueHeads, N), win::kWindowThreads>>>(
                    routes[s]->rowset<N>(begin[s]), w.norm.p, kKind, kD, kEpsilon);
            });
        if (hipDeviceSynchronize() != hipSuccess) return false;
        for (std::size_t s = 0; s < routes.size(); ++s) {
            begin[s] += counts[s][i];
            previous[s] = counts[s][i];
        }
    }
    for (std::size_t s = 0; s < routes.size(); ++s) {
        if (!routes[s]->set_control(previous[s], (static_cast<std::uint32_t>(windows) & 1U) ^ 1U)) return false;
        win::commit_kernel<Binder, kStorage><<<kCommitBlocks, 32>>>(routes[s]->state.p, routes[s]->history.p,
                                                                    routes[s]->saved_rows(), routes[s]->control.p,
                                                                    kKeyHeads, kValueHeads, kD);
    }
    return hipDeviceSynchronize() == hipSuccess;
}

// Compare a window-form run against the serial run of the same session.
bool compare(const Route& win_run, const Route& serial, const Init& init, const char* name) {
    std::vector<float> a, b;
    std::vector<Slot> ha, hb;
    std::vector<std::uint8_t> sa;
    bool ok = true;
    const auto same = [&](const Dev<float>& x, const Dev<float>& y, const char* what) {
        const bool e = x.get(a) && y.get(b) && bytes_equal(a, b);
        std::printf("  %s: %s\n", what, e ? "equal" : "DIFFER");
        ok = ok && e;
    };
    same(win_run.qkv, serial.qkv, "prepared qkv rows");
    same(win_run.beta, serial.beta, "beta rows");
    same(win_run.gate, serial.gate, "gate rows");
    same(win_run.core, serial.core, "outputs");
    same(win_run.state, serial.state, "committed state");
    const bool he = win_run.history.get(ha) && serial.history.get(hb) && bytes_equal(ha, hb);
    std::printf("  committed conv history: %s\n", he ? "equal" : "DIFFER");
    ok = ok && he;
    // not vacuous: state, history and outputs moved, both saved slots were written
    std::vector<float> core;
    ok = ok && win_run.core.get(core) && win_run.state.get(a) && !bytes_equal(a, init.state) && !bytes_equal(ha, init.history);
    bool any_core = false;
    for (float x : core) any_core = any_core || x != 0.0F;
    bool slot_used[2] = {false, false};
    if (win_run.saved.get(sa))
        for (unsigned slot = 0; slot < 2U; ++slot)
            for (std::uint64_t i = 0; i < kMaxRows * kSavedRowBytes; ++i)
                slot_used[slot] = slot_used[slot] || sa[slot * kMaxRows * kSavedRowBytes + i] != 0;
    ok = ok && any_core && slot_used[0] && slot_used[1];
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
    return true;
}

bool setup_weights(Weights& w) {
    Rng rng{7};
    std::vector<float> conv(static_cast<std::size_t>(kChannels) * kTaps), dt(kValueHeads), a(kValueHeads), norm(kD);
    for (auto& x : conv) x = rng.next(-0.8F, 0.8F);
    for (auto& x : dt) x = rng.next(-0.5F, 0.5F);
    for (auto& x : a) x = rng.next(-1.5F, -0.1F);
    for (auto& x : norm) x = rng.next(0.5F, 1.5F);
    return w.conv.put(conv) && w.dt.put(dt) && w.a.put(a) && w.norm.put(norm);
}

}  // namespace

int main() {
    Weights w;
    if (!setup_weights(w)) return 2;
    {  // single-session kernels, windows of 3, 1, 4, 2, 4, 1 rows (19 rows, six alternations of the slot)
        const std::vector<unsigned> sizes{3, 1, 4, 2, 4, 1};
        const std::size_t rows = 19;
        const Init init = make_init(31, rows);
        Route windowed, serial;
        if (!windowed.put(init, rows) || !serial.put(init, rows)) return 2;
        run_serial(serial, static_cast<std::uint32_t>(rows), w);
        if (hipDeviceSynchronize() != hipSuccess || !run_windows(windowed, sizes, w)) return 2;
        compare(windowed, serial, init, "windows == serial (single-session kernels, slots alternate)");
    }
    {  // cohort kernels: two sessions with their own data, segment counts differ within a launch
        const std::vector<std::vector<unsigned>> counts{{4, 2, 8, 1, 5}, {2, 7, 3, 6, 2}};  // 20 and 20 rows
        const std::size_t rows = 20;
        const Init init0 = make_init(41, rows), init1 = make_init(42, rows);
        Route w0, w1, s0, s1;
        if (!w0.put(init0, rows) || !w1.put(init1, rows) || !s0.put(init0, rows) || !s1.put(init1, rows)) return 2;
        run_serial(s0, static_cast<std::uint32_t>(rows), w);
        run_serial(s1, static_cast<std::uint32_t>(rows), w);
        if (hipDeviceSynchronize() != hipSuccess || !run_cohort({&w0, &w1}, counts, w)) return 2;
        compare(w0, s0, init0, "cohort segment 0 == serial");
        compare(w1, s1, init1, "cohort segment 1 == serial");
    }
    if (failures != 0) return 1;
    std::printf("recur gdn window hip tests: OK\n");
    return 0;
}
