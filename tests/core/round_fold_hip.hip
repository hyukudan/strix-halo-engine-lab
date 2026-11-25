// Device identity of the shared round folds (no model): (1) the causal convolution with RoundInput on unrounded
// input and dynamic taps equals Bf16Round in place followed by the incumbent convolution (output and history bitwise,
// decode and prefill, static and dynamic taps, both output stores); (2) a BF16 tight-row projection with round_bf16
// equals the plain projection followed by Bf16Round (bitwise on finite values), and the tight rows refuse the separate
// store policy; (3) at prefill heights, through every store of the shared projection entries (the WMMA tile, the FMA
// rows on unaligned weights, the routed image entry, the K32 image tile and the K64 packed schedule), a projection
// with round_bf16 and round_separate equals the plain projection followed by Bf16Round for every bit, and a projection
// with round_bf16 alone keeps the incumbent function's bits, on inputs and weights holding quiet and signalling NaNs of
// both signs and payloads, +-Inf and Inf x 0 next to finite neighbours; every call reports the round store its launch
// used (and the routed entry whether it took the image routes, the image entry its K32 / K64 launch) and a fallback
// or a wrong store fails the case; (4) the image stores, and the function each reported store applies, fed raw NaN
// sums with low payload bits directly, store the incumbent / separate function's bits per policy. The FMA rows' case
// must also carry such a NaN in its outputs: otherwise it prints SKIPPED and the run does not pass.
#include "round_fold_support.hpp"
#include "core/row_transforms.hpp"
#include "core/short_convolution.hpp"

#include <string>

namespace {
namespace core = strix::core;

using namespace round_fold_support;

// One convolution case: arm A rounds input and dynamic in place then convolves (RoundInput off), arm B convolves the
// unrounded copies with RoundInput on. Both start from the same (already rounded) history.
template <bool RoundOutput>
bool conv_case(unsigned channels, unsigned taps, unsigned groups, unsigned rows, std::mt19937& rng) {
    const core::ShortConvolution plan{channels, taps, groups, 1.0F, groups ? 3.0F : 2.0F};
    const auto input = noise(std::size_t{rows} * channels, rng, 1.7F);
    const auto weights = noise(std::size_t{taps} * channels, rng, 0.6F);
    const auto dynamic = noise(std::size_t{rows} * (groups ? groups : 1U) * taps, rng, 0.3F);
    auto history = noise(std::size_t{taps - 1U} * channels, rng, 1.1F);
    for (auto& h : history) h = strix::weight::scalar_float::bf16_round(h);
    const std::vector<float> zeros(std::size_t{rows} * channels, 0.0F);

    float* in_a = upload(input);
    float* in_b = upload(input);
    float* dyn_a = groups ? upload(dynamic) : nullptr;
    float* dyn_b = groups ? upload(dynamic) : nullptr;
    float* w = upload(weights);
    float* hist_a = upload(history);
    float* hist_b = upload(history);
    float* out_a = upload(zeros);
    float* out_b = upload(zeros);
    namespace rt = core::row_transforms;
    ok(rt::elements(input.size(), rt::Bf16Round{in_a}, nullptr), "round input");
    if (groups) ok(rt::elements(dynamic.size(), rt::Bf16Round{dyn_a}, nullptr), "round taps");
    ok(core::causal_conv::launch_with_store<RoundOutput>(plan, {rows, in_a, w, dyn_a, hist_a, out_a}, nullptr), "A");
    ok(core::causal_conv::launch_with_store<RoundOutput, true>(plan, {rows, in_b, w, dyn_b, hist_b, out_b}, nullptr),
       "B");
    ok(hipDeviceSynchronize(), "sync");
    const auto oa = download(out_a, zeros.size()), ob = download(out_b, zeros.size());
    const auto ha = download(hist_a, history.size()), hb = download(hist_b, history.size());
    const bool same = std::memcmp(oa.data(), ob.data(), oa.size() * 4U) == 0 &&
                      std::memcmp(ha.data(), hb.data(), ha.size() * 4U) == 0;
    std::printf("%s round-fold-hip conv channels=%u taps=%u groups=%u rows=%u round_output=%d\n",
                same ? "PASS" : "FAIL", channels, taps, groups, rows, RoundOutput ? 1 : 0);
    for (void* p : {static_cast<void*>(in_a), static_cast<void*>(in_b), static_cast<void*>(dyn_a),
                    static_cast<void*>(dyn_b), static_cast<void*>(w), static_cast<void*>(hist_a),
                    static_cast<void*>(hist_b), static_cast<void*>(out_a), static_cast<void*>(out_b)})
        if (p) (void)hipFree(p);
    return same;
}

bool projection_case(unsigned width, unsigned outputs, unsigned rows, std::mt19937& rng) {
    const auto x = noise(std::size_t{rows} * width, rng, 1.0F);
    std::vector<std::uint16_t> wbits(std::size_t{outputs} * width);
    for (std::size_t i = 0; i < wbits.size(); ++i)
        wbits[i] = strix::weight::scalar_float::bf16_encode(noise(1, rng, 0.05F)[0]);
    const std::vector<float> zeros(std::size_t{rows} * outputs, 0.0F);
    float* dx = upload(x);
    std::uint16_t* dw = upload(wbits);
    float* ya = upload(zeros);
    float* yb = upload(zeros);
    const core::Bf16RowProjection plain{dw, outputs, ya};
    core::Bf16RowProjection stored{dw, outputs, yb};
    stored.round_bf16 = true;
    ok(core::fma_rows::enqueue_bf16_rows(dx, width, rows, {&plain, 1U}, outputs, nullptr), "plain projection");
    namespace rt = core::row_transforms;
    ok(rt::elements(zeros.size(), rt::Bf16Round{ya}, nullptr), "round projection");
    ok(core::fma_rows::enqueue_bf16_rows(dx, width, rows, {&stored, 1U}, outputs, nullptr), "stored projection");
    ok(hipDeviceSynchronize(), "sync");
    const auto a = download(ya, zeros.size()), b = download(yb, zeros.size());
    const bool same = std::memcmp(a.data(), b.data(), a.size() * 4U) == 0;
    std::printf("%s round-fold-hip projection width=%u outputs=%u rows=%u admitted=%d\n", same ? "PASS" : "FAIL", width,
                outputs, rows, core::fma_rows::rounds_in_store(rows) ? 1 : 0);
    (void)hipFree(dx);
    (void)hipFree(dw);
    (void)hipFree(ya);
    (void)hipFree(yb);
    return same;
}

// The store of one image launch fed raw sums directly: out[i] = Store::put(bits[i]) for the policy.
template <core::RoundStore R>
__global__ void inject_store_kernel(const std::uint32_t* bits, float* out, unsigned n, bool round) {
    using Binding = core::f16_tile_gemm::Bf16ImageProjections<1, R>;
    const typename Binding::Store store{out, n, round};
    const unsigned i = threadIdx.x;
    if (i < n) store.put(0U, 0U, i, __builtin_bit_cast(float, bits[i]));
}

// Raw NaN sums with payload bits below BF16 width of both signs (quiet, signalling, all-ones), canonical NaNs, rounding
// ties, overflow to Inf and signed zeros / Inf. The payload patterns tell the two functions apart by construction.
bool injected_store_case() {
    constexpr std::uint32_t patterns[] = {0x7FC12345U, 0xFFC12345U, 0x7F800001U, 0xFFBFFFFFU, 0x7FFFFFFFU, 0x7FC00001U,
                                          0x7FC00000U, 0xFFC00000U, 0x3F808000U, 0x3F818000U, 0x3F80C000U, 0x7F7FFFFFU,
                                          0x7F7F8000U, 0xFF800000U, 0x7F800000U, 0x80000000U, 0x00000000U, 0x00008001U};
    constexpr unsigned n = sizeof(patterns) / sizeof(patterns[0]);
    const std::vector<std::uint32_t> bits(patterns, patterns + n);
    std::uint32_t* d = upload(bits);
    float* inc = upload(std::vector<float>(n));
    float* sep = upload(std::vector<float>(n));
    float* off = upload(std::vector<float>(n));
    inject_store_kernel<core::RoundStore::incumbent><<<1, 64>>>(d, inc, n, true);
    inject_store_kernel<core::RoundStore::separate><<<1, 64>>>(d, sep, n, true);
    inject_store_kernel<core::RoundStore::separate><<<1, 64>>>(d, off, n, false);
    ok(hipGetLastError(), "inject launch");
    ok(hipDeviceSynchronize(), "sync");
    const auto gi = download(inc, n), gs = download(sep, n), go = download(off, n);
    std::size_t bad = 0, differ = 0;
    for (unsigned i = 0; i < n; ++i) {
        const float v = sf::u2f(patterns[i]);
        bad += sf::f2u(gi[i]) != incumbent_bits(v) || sf::f2u(gs[i]) != separate_bits(v) || sf::f2u(go[i]) != patterns[i];
        differ += sf::f2u(gi[i]) != sf::f2u(gs[i]);
    }
    // 0x7FC12345 stores 0x7FC12345 under the incumbent policy and 0x7FC10000 under the separate one.
    const bool payload = sf::f2u(gi[0]) == 0x7FC12345U && sf::f2u(gs[0]) == 0x7FC10000U && differ != 0U;
    const bool pass = bad == 0U && payload;
    std::printf("%s round-fold-hip injected-store patterns=%u mismatches=%zu policies_differ_on=%zu "
                "incumbent(7FC12345)=%08X separate(7FC12345)=%08X\n", pass ? "PASS" : "FAIL", n, bad, differ,
                sf::f2u(gi[0]), sf::f2u(gs[0]));
    (void)hipFree(d);
    (void)hipFree(inc);
    (void)hipFree(sep);
    (void)hipFree(off);
    return pass;
}

enum class Mode : unsigned { wmma, fma, routed, image_k32, image_packed };
const char* mode_name(Mode m) {
    constexpr const char* names[] = {"wmma", "fma-rows", "routed", "image-k32", "image-packed"};
    return names[static_cast<unsigned>(m)];
}

// What one public call reported: the round store its launch was instantiated with, whether a routed call took the
// image routes, and which image launch the image entry made.
struct Observed {
    core::RoundStore store{core::RoundStore::incumbent};
    bool image_used{false};
    core::ImageLaunch launched{core::ImageLaunch::k32_image};
};

// One call of `mode` on the projection p (its round flags decide the store). Every mode goes through the public entry
// and records what that call reports.
void run_projection(Mode mode, const float* dx, unsigned width, unsigned rows, const core::Bf16RowProjection& p,
                    unsigned outputs, std::uint16_t* image, std::span<std::uint16_t> image_span, Observed& seen) {
    switch (mode) {
    case Mode::wmma:
    case Mode::fma:
        ok(core::fma_rows::enqueue_bf16_rows(dx, width, rows, {&p, 1U}, outputs, nullptr, &seen.store), "rows");
        break;
    case Mode::routed:
        ok(core::fma_rows::enqueue_bf16_rows_routed(dx, width, rows, {&p, 1U}, outputs, image_span, nullptr,
                                                    &seen.image_used, &seen.store),
           "routed");
        break;
    default:
        ok(core::fma_rows::enqueue_bf16_operand_image(dx, width, rows, image, nullptr), "operand image");
        ok(core::fma_rows::enqueue_bf16_image_rows(image, width, rows, {&p, 1U}, outputs, nullptr, &seen.launched,
                                                   &seen.store),
           "image rows");
    }
}

// Raw sums through the function a tile epilogue applies for the bound store (round_stored<R>, the call the WMMA, K32
// and K64 epilogues make on the completed sum): out[i] = round_stored<R>(bits[i]).
template <core::RoundStore R>
__global__ void bound_round_kernel(const std::uint32_t* bits, float* out, unsigned n) {
    const unsigned i = threadIdx.x;
    if (i < n) out[i] = core::round_stored<R>(__builtin_bit_cast(float, bits[i]));
}

constexpr std::uint32_t kRawPatterns[] = {0x7FC12345U, 0xFFC12345U, 0x7F800001U, 0xFFBFFFFFU, 0x7FFFFFFFU, 0x7FC00001U,
                                          0x7FC00000U, 0x3F808000U, 0x3F818000U, 0x7F7F8000U, 0x7F800000U, 0x80000000U};

// The raw payload patterns through the round store a call reported: every pattern must store the bits of the expected
// policy's function (host); the discriminating patterns (0x7FC12345, ...) store the bits of that function only.
bool bound_payload_matches(core::RoundStore observed, core::RoundStore expected) {
    constexpr unsigned n = sizeof(kRawPatterns) / sizeof(kRawPatterns[0]);
    const std::vector<std::uint32_t> bits(kRawPatterns, kRawPatterns + n);
    std::uint32_t* d = upload(bits);
    float* out = upload(std::vector<float>(n));
    if (observed == core::RoundStore::separate) bound_round_kernel<core::RoundStore::separate><<<1, 64>>>(d, out, n);
    else bound_round_kernel<core::RoundStore::incumbent><<<1, 64>>>(d, out, n);
    ok(hipGetLastError(), "bound round launch");
    ok(hipDeviceSynchronize(), "sync");
    const auto got = download(out, n);
    bool same = true;
    for (unsigned i = 0; i < n; ++i) {
        const float v = sf::u2f(kRawPatterns[i]);
        same = same && sf::f2u(got[i]) == (expected == core::RoundStore::separate ? separate_bits(v) : incumbent_bits(v));
    }
    (void)hipFree(d);
    (void)hipFree(out);
    return same;
}

// One NaN case through `mode` (weights pitched at `pitch` elements when pitch > width): with round_bf16 +
// round_separate the store equals the separate Bf16Round pass over the unrounded sums; with round_bf16 alone it equals
// the incumbent function on them. Each arm's call must report its launch: the round store it was instantiated with
// (the discriminating raw payloads are then run through that function), for the routed entry that it took the image
// routes (a fallback to the FP32 rows fails the case instead of passing on the rows' own store), for the image entry
// the K32 / K64 launch the case names. The tile and image sums hold BF16-width NaNs only, so their payload is tested
// through the reported store; the FMA rows (their sums keep the operand payload) must also show it in the outputs.
bool store_case(Mode mode, unsigned width, unsigned outputs, unsigned rows, unsigned pitch, std::mt19937& rng,
                bool& skipped) {
    auto o = make_operands(width, outputs, rows, true, rng);
    const unsigned ld = pitch > width ? pitch : width;
    if (ld != width) {  // re-lay the weights [outputs][ld] with zero padding
        std::vector<std::uint16_t> wide(std::size_t{outputs} * ld + 1U, 0U);
        for (unsigned n = 0; n < outputs; ++n)
            std::copy_n(o.wall.begin() + 1 + std::ptrdiff_t{n} * width, width, wide.begin() + 1 + std::ptrdiff_t{n} * ld);
        o.wall = wide;
    }
    const std::vector<std::uint16_t> body(o.wall.begin() + 1, o.wall.end());
    float* dx = upload(o.x);
    std::uint16_t* dwall = upload(mode == Mode::fma ? o.wall : body);
    const std::uint16_t* dw = mode == Mode::fma ? dwall + 1 : dwall;
    const std::vector<float> zeros(std::size_t{rows} * outputs, 0.0F);
    float* yraw = upload(zeros);
    float* ysep = upload(zeros);
    float* yinc = upload(zeros);
    float* yref = upload(zeros);
    float* ydir = upload(zeros);
    std::uint16_t* image = nullptr;
    ok(hipMalloc(&image, std::size_t{rows} * width * sizeof(std::uint16_t) + 16U), "image");
    const std::span<std::uint16_t> image_span(image, std::size_t{rows} * width);
    core::Bf16RowProjection raw{dw, outputs, yraw}, separate{dw, outputs, ysep}, incumbent{dw, outputs, yinc};
    separate.round_bf16 = separate.round_separate = true;
    incumbent.round_bf16 = true;
    if (ld != width) raw.row_bytes = separate.row_bytes = incumbent.row_bytes = std::size_t{ld} * sizeof(std::uint16_t);

    const auto path = core::fma_rows::bf16_rows_path(width, rows, {&separate, 1U}, outputs);
    bool launch_ok = true;
    const char* launch = "";
    switch (mode) {
    case Mode::wmma:
        launch = "wmma";
        launch_ok = path == core::fma_rows::Bf16RowsPath::wmma;
        break;
    case Mode::fma:
        launch = "fma-rows";
        launch_ok = path == core::fma_rows::Bf16RowsPath::fma;
        break;
    case Mode::routed:
        launch = "routed";  // the routed call's own report (image_used) is checked below
        break;
    case Mode::image_k32:
        launch = "k32-image-tile";
        launch_ok = core::fma_rows::admits_bf16_image_projections(width, rows, {&separate, 1U}, outputs);
        break;
    case Mode::image_packed:
        launch = "k64-packed-image";
        launch_ok = core::fma_rows::admits_bf16_image_projections(width, rows, {&separate, 1U}, outputs);
        break;
    }
    if (!launch_ok) {
        std::printf("FAIL round-fold-hip store mode=%s width=%u outputs=%u rows=%u: expected launch not planned (%s)\n",
                    mode_name(mode), width, outputs, rows, launch);
        return false;
    }

    Observed raw_seen, sep_seen, inc_seen;
    run_projection(mode, dx, width, rows, raw, outputs, image, image_span, raw_seen);
    run_projection(mode, dx, width, rows, separate, outputs, image, image_span, sep_seen);
    run_projection(mode, dx, width, rows, incumbent, outputs, image, image_span, inc_seen);
    // Dispatch as the calls reported it.
    bool dispatch_ok = sep_seen.store == core::RoundStore::separate && inc_seen.store == core::RoundStore::incumbent;
    std::string reported;
    if (mode == Mode::routed) {
        dispatch_ok = dispatch_ok && raw_seen.image_used && sep_seen.image_used && inc_seen.image_used;
        reported = std::string("image_used=") + (raw_seen.image_used ? "1" : "0") + (sep_seen.image_used ? "1" : "0") +
                   (inc_seen.image_used ? "1" : "0");
        // The K64 packed schedule, decided by the image entry's own dispatch on the same operand image (the routed
        // call stored it): contiguous weights only (the image entry refuses a pitched row).
        if (ld == width) {
            core::ImageLaunch direct{};
            core::Bf16RowProjection probe = raw;
            probe.output = ydir;
            ok(core::fma_rows::enqueue_bf16_image_rows(image, width, rows, {&probe, 1U}, outputs, nullptr, &direct),
               "direct image rows");
            dispatch_ok = dispatch_ok && direct == core::ImageLaunch::k64_packed;
            reported += direct == core::ImageLaunch::k64_packed ? " direct=k64-packed" : " direct=k32-image";
        } else {
            reported += " pitched(packed-only)";
        }
    } else if (mode == Mode::image_k32 || mode == Mode::image_packed) {
        const auto want = mode == Mode::image_packed ? core::ImageLaunch::k64_packed : core::ImageLaunch::k32_image;
        dispatch_ok = dispatch_ok && sep_seen.launched == want && inc_seen.launched == want && raw_seen.launched == want;
        reported = sep_seen.launched == core::ImageLaunch::k64_packed ? "launched=k64-packed" : "launched=k32-image";
    } else {
        reported = "launched=tile/rows";
    }
    namespace rt = core::row_transforms;
    ok(hipMemcpy(yref, yraw, zeros.size() * sizeof(float), hipMemcpyDeviceToDevice), "copy");
    ok(rt::elements(zeros.size(), rt::Bf16Round{yref}, nullptr), "round projection");
    ok(hipDeviceSynchronize(), "sync");
    const auto a = download(yraw, zeros.size()), s = download(ysep, zeros.size()), i = download(yinc, zeros.size()),
               r = download(yref, zeros.size());
    std::size_t bad_sep = 0, bad_inc = 0, nan_outputs = 0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        bad_sep += sf::f2u(s[k]) != sf::f2u(r[k]);
        bad_inc += sf::f2u(i[k]) != incumbent_bits(a[k]);
        nan_outputs += is_nan_bits(sf::f2u(a[k]));
    }
    const std::size_t payload = discriminating(a);
    const bool bound_ok = bound_payload_matches(sep_seen.store, core::RoundStore::separate) &&
                          bound_payload_matches(inc_seen.store, core::RoundStore::incumbent);
    const bool no_payload = mode == Mode::fma && payload == 0U;
    if (no_payload) skipped = true;
    const bool pass = bad_sep == 0U && bad_inc == 0U && nan_outputs != 0U && dispatch_ok && bound_ok;
    std::printf("%s round-fold-hip store mode=%s launch=%s %s width=%u pitch=%u outputs=%u rows=%u nan_outputs=%zu "
                "separate_mismatches=%zu incumbent_mismatches=%zu raw_low_payload_nans=%zu reported_store=%s/%s "
                "dispatch=%d bound_payload=%d%s\n",
                !pass ? "FAIL" : no_payload ? "SKIPPED" : "PASS", mode_name(mode), launch, reported.c_str(), width, ld,
                outputs, rows, nan_outputs, bad_sep, bad_inc, payload,
                sep_seen.store == core::RoundStore::separate ? "separate" : "incumbent",
                inc_seen.store == core::RoundStore::separate ? "separate" : "incumbent", dispatch_ok ? 1 : 0, bound_ok ? 1 : 0,
                payload == 0U ? " (outputs carry no low-payload NaN: the payload is tested through the reported store)" : "");
    (void)hipFree(dx);
    (void)hipFree(dwall);
    (void)hipFree(yraw);
    (void)hipFree(ysep);
    (void)hipFree(yinc);
    (void)hipFree(yref);
    (void)hipFree(ydir);
    (void)hipFree(image);
    return pass;
}

// The policy is one per group: the tight rows (decode heights) have the incumbent epilogue only, mixed groups and a
// separate store without round_bf16 are refused by the entry rather than stored with one of the two.
bool policy_refusals() {
    std::mt19937 rng(7U);
    const auto o = make_operands(2048U, 256U, 300U, false, rng);
    const std::vector<std::uint16_t> body(o.wall.begin() + 1, o.wall.end());
    float* dx = upload(o.x);
    std::uint16_t* dw = upload(body);
    float* y0 = upload(std::vector<float>(std::size_t{300U} * 256U));
    float* y1 = upload(std::vector<float>(std::size_t{300U} * 256U));
    core::Bf16RowProjection separate{dw, 256U, y0}, incumbent{dw, 256U, y1};
    separate.round_bf16 = separate.round_separate = true;
    incumbent.round_bf16 = true;
    const core::Bf16RowProjection both[] = {separate, incumbent};
    core::Bf16RowProjection alone{dw, 256U, y0};
    alone.round_separate = true;  // round_separate without round_bf16
    const bool tight = core::fma_rows::enqueue_bf16_rows(dx, 2048U, 8U, {&separate, 1U}, 256U, nullptr) == hipErrorInvalidValue;
    const bool tight_incumbent = core::fma_rows::enqueue_bf16_rows(dx, 2048U, 8U, {&incumbent, 1U}, 256U, nullptr) == hipSuccess;
    const bool mixed = core::fma_rows::enqueue_bf16_rows(dx, 2048U, 300U, both, 256U, nullptr) == hipErrorInvalidValue;
    const bool orphan = core::fma_rows::enqueue_bf16_rows(dx, 2048U, 300U, {&alone, 1U}, 256U, nullptr) == hipErrorInvalidValue;
    const bool pass = tight && tight_incumbent && mixed && orphan;
    ok(hipDeviceSynchronize(), "sync");
    std::printf("%s round-fold-hip policy-refusals separate-at-tight-rows=%d incumbent-at-tight-rows=%d mixed-group=%d "
                "separate-without-round=%d\n", pass ? "PASS" : "FAIL", tight, tight_incumbent, mixed, orphan);
    (void)hipFree(dx);
    (void)hipFree(dw);
    (void)hipFree(y0);
    (void)hipFree(y1);
    return pass;
}
}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    bool all = true, skipped = false;
    for (const unsigned rows : {1U, 5U}) {
        all = conv_case<false>(512, 4, 0, rows, rng) && all;
        all = conv_case<true>(512, 4, 0, rows, rng) && all;
        all = conv_case<false>(512, 4, 8, rows, rng) && all;
        all = conv_case<true>(512, 4, 8, rows, rng) && all;
    }
    for (const unsigned rows : {1U, 4U, 16U}) all = projection_case(2304, 2304, rows, rng) && all;
    all = policy_refusals() && all;
    all = injected_store_case() && all;
    // Every store at the measured prefill heights. Each call reports the round store and the launch it made (the routed
    // entry: that it took the image routes); the raw payload patterns run through that reported store. The FMA rows keep
    // the operand's payload bits in their sums, so that case must also carry low-payload NaNs in its outputs (a SKIPPED
    // line fails the run); the tile and image sums hold BF16-width NaNs only, so their payload is tested through the
    // reported store.
    for (const unsigned rows : {4096U, 4111U}) {
        all = store_case(Mode::wmma, 2304, 2304, rows, 0U, rng, skipped) && all;
        all = store_case(Mode::fma, 2304, 2304, rows, 0U, rng, skipped) && all;
        all = store_case(Mode::routed, 2304, 2304, rows, 0U, rng, skipped) && all;
        all = store_case(Mode::image_k32, 2272, 2304, rows, 0U, rng, skipped) && all;
        all = store_case(Mode::image_packed, 2304, 2304, rows, 0U, rng, skipped) && all;
    }
    // The pitched store policy: weights K = 3,072 at a 3,200-element pitch, 2,304 outputs (the pitched WMMA tile and the
    // pitched packed route).
    all = store_case(Mode::wmma, 3072, 2304, 4096U, 3200U, rng, skipped) && all;
    all = store_case(Mode::routed, 3072, 2304, 4096U, 3200U, rng, skipped) && all;
    // A K32-aligned width that is not a multiple of 64 and a short height: the K32 image tile.
    all = store_case(Mode::wmma, 2272, 384, 300U, 0U, rng, skipped) && all;
    all = store_case(Mode::image_k32, 2272, 384, 300U, 0U, rng, skipped) && all;
    all = store_case(Mode::fma, 2272, 384, 300U, 0U, rng, skipped) && all;
    std::puts(!all ? "round_fold_hip: FAIL"
                   : skipped ? "round_fold_hip: SKIPPED (a required payload case had no low-payload NaN sum)"
                             : "round_fold_hip: PASS");
    return !all ? 1 : skipped ? 2 : 0;
}
