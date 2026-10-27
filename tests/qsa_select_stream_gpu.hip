// Build: amdclang++ -O3 -std=gnu++20 --offload-arch=gfx1151 -Iinclude -x hip
//        tests/qsa_select_stream_gpu.hip -o build-c/qsa_select_stream_gpu
// No scorer arithmetic is under test: inject identical score bits into both
// production selectors and compare each against the shared host semantics and
// F11 streaming candidate. A GPU run is required to establish this gate.
#include "qsa_flash_prod.inc"
#include "qsa_select_stream_ref.hpp"

namespace ref = qsa_select_stream_ref;
using ref::U;
constexpr U Rows = 64, MaxBlocks = 65536, ScoreStride = MaxBlocks + 17;
constexpr U SelectedStride = ref::K * ref::R + 3 + 17;
constexpr U MaxStride = (MaxBlocks + strix::qsa_flash::kGroup - 1) / strix::qsa_flash::kGroup;

static void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t n) { check(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "hipMalloc"); }
    ~Device() { if (p) (void)hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

// Same epilogue statements as score_kernel and the L selector fixtures:
// lane = block, absent blocks = -inf, xor-1 then xor-2 numeric maxima.
__global__ __launch_bounds__(32) void group_max_kernel(
    strix::QsaRowsSelectViews<Rows> views, float* maxima) {
    const U lane = threadIdx.x, r = blockIdx.y, b = blockIdx.x * 32U + lane;
    const U blocks = views.visible[r] / ref::R;
    if (blockIdx.x * 32U >= blocks) return;
    const float score = b < blocks ? views.block_scores[r][b] : 0.0F;
    float m = b < blocks ? score : -INFINITY;
    m = fmaxf(m, __shfl_xor(m, 1, 32));
    m = fmaxf(m, __shfl_xor(m, 2, 32));
    if ((lane & 3U) == 0U && b < blocks)
        maxima[static_cast<std::uint64_t>(r) * MaxStride + b / strix::qsa_flash::kGroup] = m;
}

struct Case { std::vector<U> scores; U tail; };
struct Fixture {
    Device<float> a{Rows * ScoreStride}, b{Rows * ScoreStride}, maxima{Rows * MaxStride};
    Device<U> sa{Rows * SelectedStride}, sb{Rows * SelectedStride};

    void batch(const std::vector<Case>& cases, const std::string& label, U base) {
        std::vector<U> input(Rows * ScoreStride, ref::CANARY);
        strix::QsaRowsSelectViews<Rows> va{}, vb{};
        for (U r = 0; r < Rows; ++r) {
            const U n = r < cases.size() ? U(cases[r].scores.size()) : 0;
            if (r < cases.size())
                std::copy(cases[r].scores.begin(), cases[r].scores.end(), input.begin() + r * ScoreStride);
            va.block_scores[r] = a.p + r * ScoreStride;
            vb.block_scores[r] = b.p + r * ScoreStride;
            va.selected[r] = sa.p + r * SelectedStride;
            vb.selected[r] = sb.p + r * SelectedStride;
            va.visible[r] = vb.visible[r] = n * ref::R + (r < cases.size() ? cases[r].tail : 0);
        }
        check(hipMemcpy(a.p, input.data(), input.size() * sizeof(U), hipMemcpyHostToDevice), "h2d parked scores");
        check(hipMemcpy(b.p, input.data(), input.size() * sizeof(U), hipMemcpyHostToDevice), "h2d grouped scores");
        check(hipMemset(sa.p, 0xa5, Rows * SelectedStride * sizeof(U)), "canary parked selected");
        check(hipMemset(sb.p, 0xa5, Rows * SelectedStride * sizeof(U)), "canary grouped selected");
        check(hipMemset(maxima.p, 0xa5, Rows * MaxStride * sizeof(U)), "canary maxima");
        hipLaunchKernelGGL(group_max_kernel, dim3(MaxBlocks / 32U, Rows), dim3(32), 0, 0, vb, maxima.p);
        check(hipGetLastError(), "group maxima launch");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::select_rows_kernel<64>), dim3(Rows), dim3(strix::kSelectThreads), 0, 0,
            va, ref::R, ref::K);
        check(hipGetLastError(), "parked selection launch");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::qsa_flash::select_kernel<64>), dim3(Rows), dim3(strix::kSelectThreads), 0, 0,
            vb, maxima.p, MaxStride, ref::R, ref::K);
        check(hipGetLastError(), "grouped selection launch");
        check(hipDeviceSynchronize(), "selection sync");
        std::vector<U> pa(input.size()), pb(input.size()), xa(Rows * SelectedStride), xb(xa.size());
        check(hipMemcpy(pa.data(), a.p, pa.size() * sizeof(U), hipMemcpyDeviceToHost), "d2h parked scores");
        check(hipMemcpy(pb.data(), b.p, pb.size() * sizeof(U), hipMemcpyDeviceToHost), "d2h grouped scores");
        check(hipMemcpy(xa.data(), sa.p, xa.size() * sizeof(U), hipMemcpyDeviceToHost), "d2h parked selected");
        check(hipMemcpy(xb.data(), sb.p, xb.size() * sizeof(U), hipMemcpyDeviceToHost), "d2h grouped selected");
        // Check all 64 slots, including zero-visible padding rows and every guard.
        for (U r = 0; r < Rows; ++r) {
            const Case empty{{}, 0};
            const Case& c = r < cases.size() ? cases[r] : empty;
            const U n = U(c.scores.size());
            const std::string context = label + " row=" + std::to_string(base + r)
                + " blocks=" + std::to_string(n) + " tail=" + std::to_string(c.tail);
            auto host = ref::reference(c.scores, c.tail);
            auto parked_host = ref::reference(c.scores, c.tail, false);
            auto candidate = ref::emit(c.scores, c.tail,
                n > ref::K ? ref::stream(c.scores, 0, n) : std::vector<ref::Record>{});
            ref::same(host, parked_host, context + " host paths");
            ref::same(host, candidate, context + " host streaming");
            host.scores.resize(ScoreStride, ref::CANARY);
            candidate.scores.resize(ScoreStride, ref::CANARY);
            auto compare = [&](const std::vector<U>& scores, const std::vector<U>& selected, const char* kernel) {
                ref::Image actual(c.scores);
                actual.selected.assign(selected.begin() + r * SelectedStride, selected.begin() + (r + 1) * SelectedStride);
                actual.scores.assign(scores.begin() + r * ScoreStride, scores.begin() + (r + 1) * ScoreStride);
                ref::same(actual, host, context + kernel + " reference");
                ref::same(actual, candidate, context + kernel + " streaming");
            };
            compare(pa, xa, " parked");
            compare(pb, xb, " grouped");
        }
    }
};

int main() {
    try {
        const auto start = std::chrono::steady_clock::now();
        std::mt19937 rng(0x11c0512U);
        Fixture fixture;
        for (U kind = 0; kind < ref::labels.size(); ++kind) {
            std::vector<Case> cases;
            for (U n : ref::sizes) for (U tail = 0; tail < 4; ++tail)
                cases.push_back({ref::make_case(n, kind, rng), tail});
            fixture.batch(cases, ref::labels[kind], 0);
            std::cout << "PASS " << ref::labels[kind] << " rows=" << cases.size()
                      << " kernels=2 selected-capacity-and-score-canaries=bitwise\n";
        }
        // 576 seeded rows: every size/tail pair occurs at least ten times;
        // random class choices and score bits also exercise mixed-row views.
        constexpr U RandomRows = 576;
        for (U base = 0; base < RandomRows; base += Rows) {
            std::vector<Case> cases;
            for (U i = base; i < std::min(base + Rows, RandomRows); ++i) {
                const U n = ref::sizes[i % ref::sizes.size()];
                const U tail = U(i / ref::sizes.size()) % 4;
                const U kind = U(rng() % ref::labels.size());
                cases.push_back({ref::make_case(n, kind, rng), tail});
            }
            fixture.batch(cases, "seeded-random", base);
        }
        std::cout << "PASS seeded-random rows=" << RandomRows
                  << " kernels=2 selected-capacity-and-score-canaries=bitwise\n";
        if (!ref::paths.dense || !ref::paths.few || !ref::paths.capacity || !ref::paths.groups)
            throw std::runtime_error("missing reference branch coverage");
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "PASS total rows=" << ref::labels.size() * ref::sizes.size() * 4 + RandomRows
                  << " elapsed-seconds=" << seconds << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
