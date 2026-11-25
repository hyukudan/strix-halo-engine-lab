// Device fixture of the A4W4 projection-call trait (core/iu4a4_projections.hpp) against the shared A4W4 dense class it
// binds (core/iu4a4_dense.hpp, whose own fixture checks every bit against the contract dense-iu4a4-rot-v1):
//   image   the segmented BF16 source's codes + steps == iu4a4_rot::requantise_values of the same BF16 rows (host);
//   call    every segment's output (staged epilogue, column-blocks-first grid) == the plain GEMM's columns of the same
//           image + activations (incumbent RawStore epilogue and grid), bit for bit, and round_bf16 == bf16_round of
//           them (NaN rows NaN on both sides);
//   seal    a call whose rounding is not the image's sealed policy, an image sealed for the separate policy or for
//           another input width is refused; the seal carries the builder's pitch and rule;
//   gate    the gated call (x1 = gate logits) == the incumbent gate pass (row_transforms::Bf16SigmoidProduct) stored to
//           FP32 + the ungated call, bit for bit.
// Shapes: partial m CTAs, a single 16-column segment, an odd count of 128 blocks, pitched weight rows, and the Inkling 2
// Mini attention call (2,304 -> 3,072 + 384 + 3,072 + 1,024 + 1,024) at a partial height.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/iu4a4_projections.hpp"
#include "core/row_transforms.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace ip = strix::core::iu4a4_projections;
namespace ad = strix::core::iu4a4_dense;
namespace rot = strix::core::iu4a4_rot;
#define CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

std::uint32_t bits(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}
bool same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b); }
std::uint16_t bf16_bits(float v) { return static_cast<std::uint16_t>(bits(strix::attn::group_indexer::bf16_round(v)) >> 16U); }

template <class T> T* dev(std::size_t count) {
    void* p = nullptr;
    if (hipMalloc(&p, count * sizeof(T) + 16U) != hipSuccess) return nullptr;
    return static_cast<T*>(p);
}

int run_case(const char* name, unsigned rows, unsigned in, std::vector<unsigned> outs, unsigned pad, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const ip::Layout layout = ip::make_layout(in, outs);
    if (!layout.admitted()) {
        std::printf("FAIL %s: layout refused\n", name);
        return 1;
    }
    const unsigned n = layout.n, blocks = in / rot::kBlock, pitch = in + pad;
    // BF16 weights per segment (pitched rows; a few large columns), FP32 activations with outliers, a zero row and a
    // NaN row, gate logits.
    std::vector<std::vector<std::uint16_t>> w(outs.size());
    for (std::size_t s = 0; s < outs.size(); ++s) {
        w[s].resize(std::size_t(outs[s]) * pitch);
        for (std::size_t i = 0; i < w[s].size(); ++i) w[s][i] = bf16_bits(nd(g) * ((i % 97U) == 5U ? 8.0f : 0.05f));
    }
    std::vector<float> x(std::size_t(rows) * in), logit(x.size());
    for (auto& e : x) e = nd(g);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned k = 11; k < in; k += 173U) x[std::size_t(r) * in + k] *= 30.0f;
    if (rows > 2U) std::fill(x.begin() + 2 * std::ptrdiff_t(in), x.begin() + 3 * std::ptrdiff_t(in), 0.0f);
    if (rows > 6U) x[6U * std::size_t(in) + in / 3U] = std::nanf("");
    for (auto& e : logit) e = nd(g) * 2.0f;

    std::vector<std::uint16_t*> dw(outs.size());
    for (std::size_t s = 0; s < outs.size(); ++s) {
        dw[s] = dev<std::uint16_t>(w[s].size());
        if (!dw[s]) return 1;
        CHECK(hipMemcpy(dw[s], w[s].data(), w[s].size() * 2U, hipMemcpyHostToDevice));
    }
    auto* codes = dev<std::uint8_t>(layout.weight_code_bytes());
    auto* step = dev<float>(layout.weight_step_bytes() / 4U);
    auto* dx = dev<float>(x.size());
    auto* dl = dev<float>(x.size());
    auto* dgated = dev<float>(x.size());
    ip::Workspace ws{dev<std::uint8_t>(ip::workspace_code_bytes(rows, in)), dev<std::uint32_t>(ip::workspace_u_bytes(rows, in) / 4U),
                     dev<float>(ip::workspace_base_bytes(rows) / 4U), rows, in};
    auto* plain = dev<float>(std::size_t(rows) * n);
    std::vector<float*> ys(outs.size()), yr(outs.size()), yg(outs.size()), yu(outs.size());
    for (std::size_t s = 0; s < outs.size(); ++s)
        for (auto* v : {&ys, &yr, &yg, &yu}) (*v)[s] = dev<float>(std::size_t(rows) * outs[s]);
    CHECK(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dl, logit.data(), x.size() * 4U, hipMemcpyHostToDevice));

    ip::Bf16Segments src{};
    src.layout = layout;
    for (std::size_t s = 0; s < outs.size(); ++s) src.weights[s] = dw[s], src.pitch[s] = pitch;
    CHECK(ip::enqueue_build_image(src, codes, step, nullptr));
    // One image, sealed once per rounding policy of its calls (FP32 stores / incumbent BF16 rounding).
    const ip::Image image{codes, step, layout, ip::seal_of(src, ip::RoundPolicy::none)};
    const ip::Image image_round{codes, step, layout, ip::seal_of(src, ip::RoundPolicy::incumbent)};
    // Refusals: rows past the workspace, a missing output, a stride below the inputs, a rounding other than the sealed
    // one (both ways), an image sealed for the separate policy (no store of that policy here), a seal of another width.
    unsigned bad_refusal = 0;
    bad_refusal += ip::admits(image, rows + 1U, ws, false);
    std::vector<float*> missing = ys;
    missing.back() = nullptr;
    bad_refusal += ip::enqueue_call(dx, nullptr, in, rows, image, missing, false, ws, nullptr) != hipErrorInvalidValue;
    bad_refusal += ip::enqueue_call(dx, nullptr, in - 4U, rows, image, ys, false, ws, nullptr) != hipErrorInvalidValue;
    bad_refusal += ip::enqueue_call(dx, nullptr, in, rows, image, yr, true, ws, nullptr) != hipErrorInvalidValue;
    bad_refusal += ip::enqueue_call(dx, nullptr, in, rows, image_round, ys, false, ws, nullptr) != hipErrorInvalidValue;
    const ip::Image image_separate{codes, step, layout, ip::seal_of(src, ip::RoundPolicy::separate)};
    bad_refusal += ip::admits(image_separate, rows, ws, false);
    bad_refusal += ip::enqueue_call(dx, nullptr, in, rows, image_separate, yr, true, ws, nullptr) != hipErrorInvalidValue;
    ip::Image image_wide = image;
    image_wide.seal.in = in + rot::kBlock;
    bad_refusal += ip::admits(image_wide, rows, ws, false);
    // The seal carries the builder's pitch and rule.
    const ip::Seal sealed = image.seal;
    bad_refusal += sealed.in != in || sealed.rule != static_cast<unsigned>(rot::WeightRule::amax);
    for (std::size_t s = 0; s < outs.size(); ++s) bad_refusal += sealed.row_bytes[s] != std::size_t(pitch) * 2U;

    CHECK(ip::enqueue_call(dx, nullptr, in, rows, image, ys, false, ws, nullptr));
    CHECK(ip::enqueue_call(dx, nullptr, in, rows, image_round, yr, true, ws, nullptr));
    // The plain GEMM over the same image and the same activations (the shared class's own fixture checks it).
    const ad::Act act{ws.codes, ws.u, ws.base, rows, in};
    ad::enqueue_quantise(dx, in, act, nullptr);
    ad::enqueue_gemm(ad::Weights{codes, step, in, n}, act, ad::OutCols{plain, n, n / 16U}, nullptr);
    // Gate: fused call vs the incumbent gate pass + the ungated call.
    CHECK(ip::enqueue_call(dx, dl, in, rows, image, yg, false, ws, nullptr));
    CHECK(strix::core::row_transforms::elements(x.size(), strix::core::row_transforms::Bf16SigmoidProduct{dx, dl, dgated},
                                                static_cast<hipStream_t>(nullptr)));
    CHECK(ip::enqueue_call(dgated, nullptr, in, rows, image, yu, false, ws, nullptr));
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    std::vector<std::uint8_t> gc(layout.weight_code_bytes());
    std::vector<float> gs(layout.weight_step_bytes() / 4U), gp(std::size_t(rows) * n);
    CHECK(hipMemcpy(gc.data(), codes, gc.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gs.data(), step, gs.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gp.data(), plain, gp.size() * 4U, hipMemcpyDeviceToHost));
    unsigned bad_code = 0, bad_step = 0, bad_y = 0, bad_round = 0, bad_gate = 0, nan_rows = 0;
    // Image vs the contract rule on the host.
    std::vector<std::uint8_t> hc(gc.size(), 0);
    for (unsigned b = 0; b < blocks; ++b)
        for (unsigned c = 0; c < n; ++c) {
            const unsigned s = layout.segment(c);
            const std::uint16_t* row = w[s].data() + std::size_t(c - layout.first_col[s]) * pitch + std::size_t(b) * rot::kBlock;
            const auto value = [row](unsigned i) {
                const std::uint32_t u = std::uint32_t{row[i]} << 16U;
                float f;
                std::memcpy(&f, &u, 4);
                return f;
            };
            const float hs = rot::requantise_values(value, b, c, n, hc.data());
            bad_step += bits(gs[ad::step_index(b, c, n)]) != bits(hs);
        }
    for (std::size_t i = 0; i < hc.size(); ++i) bad_code += hc[i] != gc[i];
    for (std::size_t s = 0; s < outs.size(); ++s) {
        const unsigned o = outs[s];
        std::vector<float> a(std::size_t(rows) * o), r(a.size()), gt(a.size()), un(a.size());
        CHECK(hipMemcpy(a.data(), ys[s], a.size() * 4U, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(r.data(), yr[s], a.size() * 4U, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(gt.data(), yg[s], a.size() * 4U, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(un.data(), yu[s], a.size() * 4U, hipMemcpyDeviceToHost));
        for (unsigned m = 0; m < rows; ++m)
            for (unsigned j = 0; j < o; ++j) {
                const std::size_t i = std::size_t(m) * o + j;
                const float ref = gp[std::size_t(m) * n + layout.first_col[s] + j];
                bad_y += !same(a[i], ref);
                bad_round += !same(r[i], strix::attn::group_indexer::bf16_round(ref));
                bad_gate += !same(gt[i], un[i]);
                nan_rows += std::isnan(a[i]) && j == 0U;
            }
    }
    for (auto* p : dw) CHECK(hipFree(p));
    for (void* p : {static_cast<void*>(codes), static_cast<void*>(step), static_cast<void*>(dx), static_cast<void*>(dl), static_cast<void*>(dgated),
                    static_cast<void*>(ws.codes), static_cast<void*>(ws.u), static_cast<void*>(ws.base), static_cast<void*>(plain)})
        CHECK(hipFree(p));
    for (auto* v : {&ys, &yr, &yg, &yu})
        for (float* p : *v) CHECK(hipFree(p));
    const unsigned bad = bad_refusal + bad_code + bad_step + bad_y + bad_round + bad_gate;
    std::printf("%s %s rows %u in %u n %u segments %zu: refusal %u code %u step %u y %u round %u gate %u (nan rows %u)\n",
                bad ? "FAIL" : "PASS", name, rows, in, n, outs.size(), bad_refusal, bad_code, bad_step, bad_y, bad_round, bad_gate,
                nan_rows);
    return bad ? 1 : 0;
}
}  // namespace

int main() {
    int fails = 0;
    fails += run_case("small", 40, 256, {32, 48, 16}, 0, 1U);
    fails += run_case("pitched-odd-blocks", 133, 384, {16, 64}, 64, 2U);
    fails += run_case("single", 17, 128, {16}, 0, 3U);
    fails += run_case("attention-call", 300, 2304, {3072, 384, 3072, 1024, 1024}, 0, 4U);
    fails += run_case("output-call", 129, 3072, {2304}, 128, 5U);
    std::printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
