// Block-per-row controller: enqueue_low_rank_controller (and its tail form over a produced message) stores the same output
// bits whether the backend runs the six launches (readout, operand-norm pair, down projection, gate pair, pointwise
// controller, final norm) or the one block-per-row launch of core::controller_block, apart and in place. Rows 1, 2, 3, 8
// and 16 take the fused launch; 17 rows keep the launches. Value classes per row: random, exact and negative zeros,
// denormals, large magnitudes, and (a separate class) infinities and NaNs, compared with every NaN equal to every NaN.
// Prints "controller-fused rows R: identical N/N" per row count; exits non-zero on any mismatch.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/fma_bf16_rows.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#include "core/residual_mix.hpp"
#pragma clang diagnostic pop
#include "core/controller_block_rows.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace rows_ns = strix::core::fma_rows;
namespace mix = strix::core::residual_mix;
namespace block = strix::core::controller_block;
using strix::core::Bf16RowProjection;

constexpr unsigned kWidth = 2304U, kRank = 64U;

bool ok(hipError_t status) { return status == hipSuccess; }

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (!ok(hipMalloc(&device, host.size() * sizeof(T)))) return nullptr;
    if (!ok(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice))) return nullptr;
    return device;
}

float from_bits(std::uint32_t bits) {
    float x;
    std::memcpy(&x, &bits, sizeof x);
    return x;
}

enum class Kind { random, edge, nonfinite };

// Rows of `rows` x kWidth values. edge: row % 4 = 0 zeros and negative zeros, 1 denormals, 2 large magnitudes,
// 3 random with some negative zeros. nonfinite: random with infinities and NaNs sprinkled.
std::vector<float> make_rows(unsigned rows, Kind kind, float range, std::mt19937& random) {
    std::uniform_real_distribution<float> value(-range, range);
    std::vector<float> out(std::size_t{rows} * kWidth);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned i = 0; i < kWidth; ++i) {
            float x = value(random);
            if (kind == Kind::edge) {
                switch (r % 4U) {
                case 0U: x = (random() & 1U) != 0U ? -0.0F : 0.0F; break;
                case 1U: x = from_bits(static_cast<std::uint32_t>((random() % 0x007FFFFFU) | ((random() & 1U) << 31U))); break;
                case 2U: x = (random() & 1U) != 0U ? -1.0e17F * (1.0F + x / range) : 1.0e17F * (1.0F + x / range); break;
                default: if (i % 97U == 0U) x = -0.0F; break;
                }
            } else if (kind == Kind::nonfinite && i % 613U == 5U) {
                const std::uint32_t bits[] = {0x7F800000U, 0xFF800000U, 0x7FC00000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
                x = from_bits(bits[random() % 6U]);
            }
            out[std::size_t{r} * kWidth + i] = x;
        }
    }
    return out;
}

std::vector<std::uint16_t> bf16_weights(std::size_t count, std::mt19937& random, float range, bool edge) {
    std::vector<std::uint16_t> out(count);
    std::uniform_real_distribution<float> value(-range, range);
    for (std::size_t i = 0; i < count; ++i) {
        float x = value(random);
        if (edge && i % 211U == 0U) x = (random() & 1U) != 0U ? -0.0F : 0.0F;
        std::uint32_t bits;
        std::memcpy(&bits, &x, sizeof bits);
        out[i] = static_cast<std::uint16_t>(bits >> 16U);
    }
    return out;
}

// The incumbent backend operations: dense = shared BF16 rows, norm = the shared BF16-rows norm with a BF16 scale.
struct Operations {
    static constexpr float kEpsilon = 1e-6F;
    hipStream_t s = nullptr;  // the bench captures on its own stream
    hipStream_t stream() const { return s; }
    void check(hipError_t status) const {
        if (!ok(status)) throw std::runtime_error("controller-fused: launch failed");
    }
    void dense(const float* input, unsigned input_width, unsigned rows, std::span<const Bf16RowProjection> projections) {
        check(rows_ns::enqueue_bf16_rows(input, input_width, rows, projections, 1U << 20U, s));
    }
    void rms(const float* input, const std::uint16_t* scale, float* output, unsigned rows) {
        check(strix::ops::norm::enqueue_rms_bf16_rows({input, scale, output, rows, kWidth, kWidth, kWidth}, kEpsilon,
                                                      strix::ops::norm::RmsRowsReduction::parallel, s));
    }
};

// The served decode form (the HIP resident backend's operations): the gate pair with the controller as the shared tile's
// epilogue, then the norm (four launches per controller; the operand image is a prefill-only shape).
struct ServedOperations : Operations {
    bool dense_controller(const float* input, unsigned input_width, unsigned rows, std::span<const Bf16RowProjection> gates,
                          const mix::ControllerArgs& control) {
        const auto status = rows_ns::enqueue_bf16_controller_rows(input, input_width, rows, gates, control.current,
                                                                  control.message, control.output, s);
        if (status == hipErrorNotSupported) return false;
        check(status);
        return true;
    }
};

struct FusedOperations : Operations {
    unsigned taken = 0;
    bool fused_controller(block::Args args) {
        args.norm_epsilon = kEpsilon;
        if (!block::admits(args)) return false;
        check(block::enqueue(args, s));
        ++taken;
        return true;
    }
};

void release(std::initializer_list<void*> pointers) {
    for (void* p : pointers)
        if (p) (void)hipFree(p);
}

// Bitwise equal, every NaN equal to every NaN when nan_class. Counts the equal floats.
std::size_t identical(const std::vector<float>& a, const std::vector<float>& b, bool nan_class) {
    std::size_t same = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) == 0 || (nan_class && std::isnan(a[i]) && std::isnan(b[i]))) ++same;
    }
    return same;
}

// One case: both forms, apart and in place. Adds to same / total; returns false on a launch failure or a form not taken.
bool run(unsigned rows, Kind kind, std::mt19937& random, std::size_t& same, std::size_t& total) {
    const mix::LowRankController plan{kWidth, kRank, 2U, 1e-6F, true, true};
    const std::size_t elements = std::size_t{rows} * kWidth, scratch_elements = plan.scratch_elements(rows);
    const bool edge = kind != Kind::random;
    const auto raw_host = make_rows(rows, kind, 3.0F, random);
    const auto bank0_host = make_rows(rows, kind, 1.5F, random);
    const auto bank1_host = make_rows(rows, kind, 1.5F, random);
    auto* raw_ref = upload(raw_host);
    auto* raw_fused = upload(raw_host);
    auto* raw_tail_ref = upload(raw_host);
    auto* raw_tail_fused = upload(raw_host);
    auto* bank0 = upload(bank0_host);
    auto* bank1 = upload(bank1_host);
    auto* query = upload(make_rows(1U, Kind::random, 0.2F, random));
    auto* down = upload(bf16_weights(std::size_t{kRank} * 2U * kWidth, random, 0.05F, edge));
    auto* forget = upload(bf16_weights(std::size_t{kWidth} * kRank, random, 0.2F, edge));
    auto* gain = upload(bf16_weights(std::size_t{kWidth} * kRank, random, 0.2F, edge));
    auto* norm = upload(bf16_weights(kWidth, random, 1.0F, edge));
    float *out_ref = nullptr, *out_fused = nullptr, *out_tail_ref = nullptr, *out_tail_fused = nullptr;
    float *scratch_ref = nullptr, *scratch_fused = nullptr, *scratch_tail_ref = nullptr, *scratch_tail_fused = nullptr;
    float* message = nullptr;
    bool good = false;
    const std::size_t bytes = elements * sizeof(float), scratch_bytes = scratch_elements * sizeof(float);
    if (raw_ref && raw_fused && raw_tail_ref && raw_tail_fused && bank0 && bank1 && query && down && forget && gain && norm &&
        ok(hipMalloc(&out_ref, bytes)) && ok(hipMalloc(&out_fused, bytes)) && ok(hipMalloc(&out_tail_ref, bytes)) &&
        ok(hipMalloc(&out_tail_fused, bytes)) && ok(hipMalloc(&scratch_ref, scratch_bytes)) &&
        ok(hipMalloc(&scratch_fused, scratch_bytes)) && ok(hipMalloc(&scratch_tail_ref, scratch_bytes)) &&
        ok(hipMalloc(&scratch_tail_fused, scratch_bytes)) && ok(hipMalloc(&message, bytes)) &&
        ok(hipMemset(out_fused, 0x5A, bytes)) && ok(hipMemset(out_tail_fused, 0x5A, bytes)) &&
        ok(hipMemset(scratch_fused, 0x5A, scratch_bytes)) && ok(hipMemset(scratch_tail_fused, 0x5A, scratch_bytes))) {
        const mix::ConstRows banks[] = {{bank0, elements}, {bank1, elements}};
        const mix::ControllerWeights weights{{query, kWidth}, down, forget, gain, norm};
        Operations base;
        FusedOperations fused, fused_tail;
        // In place for the second pass of the larger cases: output = raw.
        const bool in_place = rows == 8U || rows == 3U;
        float* const target_ref = in_place ? raw_ref : out_ref;
        float* const target_fused = in_place ? raw_fused : out_fused;
        float* const target_tail_ref = in_place ? raw_tail_ref : out_tail_ref;
        float* const target_tail_fused = in_place ? raw_tail_fused : out_tail_fused;
        std::vector<float> ref(elements), got(elements), ref_tail(elements), got_tail(elements);
        bool launched = true;
        try {
            mix::enqueue_low_rank_controller(plan, weights, rows, {raw_ref, elements}, banks, {target_ref, elements},
                                             {scratch_ref, scratch_elements}, base);
            mix::enqueue_low_rank_controller(plan, weights, rows, {raw_fused, elements}, banks,
                                             {target_fused, elements}, {scratch_fused, scratch_elements}, fused);
            // The readout of the incumbent run is the message of the tail runs (its scratch holds it first).
            if (!ok(hipMemcpy(message, scratch_ref, bytes, hipMemcpyDeviceToDevice))) throw std::runtime_error("copy");
            mix::enqueue_low_rank_controller_tail(plan, weights, rows, {raw_tail_ref, elements}, {message, elements},
                                                  {target_tail_ref, elements}, {scratch_tail_ref, scratch_elements}, base);
            mix::enqueue_low_rank_controller_tail(plan, weights, rows, {raw_tail_fused, elements}, {message, elements},
                                                  {target_tail_fused, elements}, {scratch_tail_fused, scratch_elements},
                                                  fused_tail);
        } catch (const std::exception&) {
            launched = false;
        }
        const unsigned expected = rows <= block::kMaxRows ? 1U : 0U;
        good = launched && ok(hipDeviceSynchronize()) && fused.taken == expected && fused_tail.taken == expected &&
               ok(hipMemcpy(ref.data(), target_ref, bytes, hipMemcpyDeviceToHost)) &&
               ok(hipMemcpy(got.data(), target_fused, bytes, hipMemcpyDeviceToHost)) &&
               ok(hipMemcpy(ref_tail.data(), target_tail_ref, bytes, hipMemcpyDeviceToHost)) &&
               ok(hipMemcpy(got_tail.data(), target_tail_fused, bytes, hipMemcpyDeviceToHost));
        if (good) {
            const bool nan_class = kind == Kind::nonfinite;
            same += identical(ref, got, nan_class) + identical(ref_tail, got_tail, nan_class);
            total += 2U * elements;
        } else {
            std::printf("controller-fused rows %u: FAIL launch or route (taken %u/%u, expected %u)\n", rows, fused.taken,
                        fused_tail.taken, expected);
        }
    }
    release({raw_ref, raw_fused, raw_tail_ref, raw_tail_fused, bank0, bank1, query, down, forget, gain, norm, out_ref,
             out_fused, out_tail_ref, out_tail_fused, scratch_ref, scratch_fused, scratch_tail_ref, scratch_tail_fused,
             message});
    return good;
}

// --bench: one decode pass worth of controller tails (kBenchControllers, each with its own weights, so the weight bytes
// stream from memory as in a pass) captured as one graph per form; ABAB replays, the median per form. Prints
// "controller-fused bench rows R: launches X us fused Y us per controller (ratio)".
constexpr unsigned kBenchControllers = 96U, kBenchReps = 15U;

bool bench(unsigned rows, std::mt19937& random) {
    const mix::LowRankController plan{kWidth, kRank, 2U, 1e-6F, true, true};
    const std::size_t elements = std::size_t{rows} * kWidth, scratch_elements = plan.scratch_elements(rows);
    const std::size_t down_count = std::size_t{kRank} * 2U * kWidth, gate_count = std::size_t{kWidth} * kRank;
    auto* raw = upload(make_rows(rows, Kind::random, 3.0F, random));
    auto* message = upload(make_rows(rows, Kind::random, 1.5F, random));
    auto* query = upload(make_rows(1U, Kind::random, 0.2F, random));
    auto* down = upload(bf16_weights(down_count * kBenchControllers, random, 0.05F, false));
    auto* forget = upload(bf16_weights(gate_count * kBenchControllers, random, 0.2F, false));
    auto* gain = upload(bf16_weights(gate_count * kBenchControllers, random, 0.2F, false));
    auto* norm = upload(bf16_weights(std::size_t{kWidth} * kBenchControllers, random, 1.0F, false));
    constexpr unsigned kForms = 3U;  // 0 six launches, 1 served (gate pair + controller epilogue, norm), 2 fused
    float* out[kForms] = {nullptr, nullptr, nullptr};
    float* scratch = nullptr;
    hipStream_t stream = nullptr;
    hipGraphExec_t exec[kForms] = {nullptr, nullptr, nullptr};
    bool good = raw && message && query && down && forget && gain && norm &&
                ok(hipMalloc(&scratch, scratch_elements * sizeof(float))) &&
                ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    for (auto*& o : out) good = good && ok(hipMalloc(&o, elements * sizeof(float)));
    unsigned taken = 0;
    for (unsigned form = 0; good && form < kForms; ++form) {
        Operations six;
        ServedOperations served;
        FusedOperations fused;
        six.s = served.s = fused.s = stream;
        hipGraph_t graph = nullptr;
        good = ok(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        try {
            for (unsigned c = 0; good && c < kBenchControllers; ++c) {
                const mix::ControllerWeights weights{{query, kWidth}, down + down_count * c, forget + gate_count * c,
                                                     gain + gate_count * c, norm + std::size_t{kWidth} * c};
                if (form == 0U)
                    mix::enqueue_low_rank_controller_tail(plan, weights, rows, {raw, elements}, {message, elements},
                                                          {out[form], elements}, {scratch, scratch_elements}, six);
                else if (form == 1U)
                    mix::enqueue_low_rank_controller_tail(plan, weights, rows, {raw, elements}, {message, elements},
                                                          {out[form], elements}, {scratch, scratch_elements}, served);
                else
                    mix::enqueue_low_rank_controller_tail(plan, weights, rows, {raw, elements}, {message, elements},
                                                          {out[form], elements}, {scratch, scratch_elements}, fused);
            }
        } catch (const std::exception&) {
            good = false;
        }
        good = ok(hipStreamEndCapture(stream, &graph)) && good && ok(hipGraphInstantiate(&exec[form], graph, nullptr, nullptr, 0));
        if (graph) (void)hipGraphDestroy(graph);
        if (form == 2U) taken = fused.taken;
    }
    std::vector<float> ms[kForms];
    hipEvent_t start = nullptr, stop = nullptr;
    good = good && ok(hipEventCreate(&start)) && ok(hipEventCreate(&stop));
    for (unsigned rep = 0; good && rep < kBenchReps + 2U; ++rep) {
        for (unsigned form = 0; good && form < kForms; ++form) {
            float elapsed = 0.0F;
            good = ok(hipEventRecord(start, stream)) && ok(hipGraphLaunch(exec[form], stream)) &&
                   ok(hipEventRecord(stop, stream)) && ok(hipEventSynchronize(stop)) &&
                   ok(hipEventElapsedTime(&elapsed, start, stop));
            if (rep >= 2U) ms[form].push_back(elapsed);  // two warm-up replays per form
        }
    }
    // The last controller's rows of every form (same inputs): served and fused must hold the six launches' bits.
    std::vector<float> host[kForms];
    for (unsigned form = 0; good && form < kForms; ++form) {
        host[form].resize(elements);
        good = ok(hipMemcpy(host[form].data(), out[form], elements * sizeof(float), hipMemcpyDeviceToHost));
    }
    if (good) {
        double us[kForms];
        for (unsigned form = 0; form < kForms; ++form) {
            std::sort(ms[form].begin(), ms[form].end());
            us[form] = 1000.0 * ms[form][ms[form].size() / 2U] / kBenchControllers;
        }
        const std::size_t served_same = identical(host[0], host[1], false);
        const std::size_t fused_same = identical(host[0], host[2], false);
        std::printf("controller-fused bench rows %u: six %.2f us served %.2f us fused %.2f us per controller (fused/served "
                    "%.3f; fused taken %u/%u; bits served %zu/%zu fused %zu/%zu)\n",
                    rows, us[0], us[1], us[2], us[2] / us[1], taken, kBenchControllers, served_same, elements,
                    fused_same, elements);
        good = served_same == elements && fused_same == elements;
    } else {
        std::printf("controller-fused bench rows %u: FAIL (capture, launch or timing)\n", rows);
    }
    if (start) (void)hipEventDestroy(start);
    if (stop) (void)hipEventDestroy(stop);
    for (auto* e : exec)
        if (e) (void)hipGraphExecDestroy(e);
    if (stream) (void)hipStreamDestroy(stream);
    release({raw, message, query, down, forget, gain, norm, out[0], out[1], out[2], scratch});
    return good;
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 random(20261127U);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        bool good = true;
        for (const unsigned rows : {1U, 2U, 3U, 4U, 6U, 8U, 12U, 16U}) good = bench(rows, random) && good;
        return good ? 0 : 1;
    }
    int failures = 0;
    for (const unsigned rows : {1U, 2U, 3U, 8U, 16U, 17U}) {
        std::size_t same = 0, total = 0;
        bool good = true;
        for (const Kind kind : {Kind::random, Kind::random, Kind::edge, Kind::nonfinite})
            good = run(rows, kind, random, same, total) && good;
        std::printf("controller-fused rows %u: identical %zu/%zu\n", rows, same, total);
        if (!good || same != total) ++failures;
    }
    if (failures == 0) std::printf("controller-fused: PASS\n");
    else std::printf("controller-fused: FAIL (%d row counts)\n", failures);
    return failures == 0 ? 0 : 1;
}
