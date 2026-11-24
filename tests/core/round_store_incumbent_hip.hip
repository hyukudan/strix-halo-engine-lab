// Bits of the incumbent rounded-store policy (core::RoundStore::incumbent: a projection with round_bf16 and nothing
// else), at the shapes and entries of the production bindings that request rounded projections today: the low-rank
// controller's down projection (width 2 x hidden, rank outputs; the FP32 rows and the BF16 operand image entries) and
// the DSpark context K / V projections (hidden width, up to 256 rows per chunk). Each case checks the stored bits
// against the incumbent function on the unrounded sums and prints the digest of its output: the fixture builds on
// both sides of a change to the shared stores, and the two digest lists must be identical line by line. A case on FMA
// rows (unaligned weights) carries NaN sums with low payload bits, the input that tells the incumbent function from
// the separate pass's; without such a sum the line says SKIPPED and the run does not pass.
#include "round_fold_support.hpp"
#include "inkling2/inkling2_mini_config.hpp"

namespace {
namespace core = strix::core;
using namespace round_fold_support;

// mode 0: enqueue_bf16_rows on aligned weights; 1: the same call on 2-byte offset weights (FMA rows); 2: the BF16
// operand image of the rows and enqueue_bf16_image_rows. Returns false on a bit mismatch with the incumbent function.
bool incumbent_case(const char* binding, unsigned mode, unsigned width, unsigned outputs, unsigned rows, bool nan,
                    bool require_payload, std::mt19937& rng, bool& skipped) {
    const auto o = make_operands(width, outputs, rows, nan, rng);
    const std::vector<std::uint16_t> body(o.wall.begin() + 1, o.wall.end());
    float* dx = upload(o.x);
    std::uint16_t* dwall = upload(mode == 1U ? o.wall : body);
    const std::uint16_t* dw = mode == 1U ? dwall + 1 : dwall;
    const std::vector<float> zeros(std::size_t{rows} * outputs, 0.0F);
    float* yraw = upload(zeros);
    float* ystored = upload(zeros);
    std::uint16_t* image = nullptr;
    ok(hipMalloc(&image, std::size_t{rows} * width * sizeof(std::uint16_t) + 16U), "image");
    const core::Bf16RowProjection raw{dw, outputs, yraw};
    core::Bf16RowProjection stored{dw, outputs, ystored};
    stored.round_bf16 = true;  // the production bindings' request: nothing else is set
    const auto project = [&](const core::Bf16RowProjection& p) {
        if (mode == 2U) {
            ok(core::fma_rows::enqueue_bf16_operand_image(dx, width, rows, image, nullptr), "operand image");
            ok(core::fma_rows::enqueue_bf16_image_rows(image, width, rows, {&p, 1U}, outputs, nullptr), "image rows");
        } else {
            ok(core::fma_rows::enqueue_bf16_rows(dx, width, rows, {&p, 1U}, outputs, nullptr), "rows");
        }
    };
    project(raw);
    project(stored);
    ok(hipDeviceSynchronize(), "sync");
    const auto a = download(yraw, zeros.size()), b = download(ystored, zeros.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) bad += incumbent_bits(a[i]) != sf::f2u(b[i]);
    const std::size_t payload = discriminating(a);
    const bool no_payload = require_payload && payload == 0U;
    if (no_payload) skipped = true;
    const bool pass = bad == 0U;
    std::printf("%s round-store-incumbent %s mode=%u width=%u outputs=%u rows=%u mismatches=%zu raw_low_payload_nans=%zu "
                "%sdigest=%016llx\n", pass ? (no_payload ? "SKIPPED" : "PASS") : "FAIL", binding, mode, width, outputs,
                rows, bad, payload, no_payload ? "no low-payload NaN reached the store; " : "",
                static_cast<unsigned long long>(fnv1a(b.data(), b.size() * sizeof(float))));
    (void)hipFree(dx);
    (void)hipFree(dwall);
    (void)hipFree(yraw);
    (void)hipFree(ystored);
    (void)hipFree(image);
    return pass;
}
}  // namespace

int main() {
    namespace ink = strix::inkling2;
    std::mt19937 rng(20261124U);
    bool all = true, skipped = false;
    // The low-rank controller's down projection: [rows][2 x hidden] operands, rank outputs, round_latent_bf16.
    for (const unsigned rows : {17U, 300U, 4096U, 8209U}) {
        all = incumbent_case("controller-rows", 0U, 2U * ink::kHiddenWidth, ink::kControllerRank, rows, true, false, rng, skipped) && all;
        all = incumbent_case("controller-image", 2U, 2U * ink::kHiddenWidth, ink::kControllerRank, rows, true, false, rng, skipped) && all;
    }
    // The DSpark context K / V projections: hidden-width mixed rows, kv-width outputs, chunks up to 256 rows.
    for (const unsigned rows : {17U, 200U, 256U})
        all = incumbent_case("dspark-kv", 0U, ink::kHiddenWidth, ink::kKvWidth, rows, true, false, rng, skipped) && all;
    // FMA rows (weights without 16-byte alignment): the input whose NaN sums carry low payload bits.
    all = incumbent_case("fma-rows", 1U, ink::kHiddenWidth, 384U, 300U, true, true, rng, skipped) && all;
    all = incumbent_case("fma-rows", 1U, ink::kHiddenWidth, 384U, 4111U, true, true, rng, skipped) && all;
    std::puts(!all ? "round_store_incumbent: FAIL" : skipped ? "round_store_incumbent: SKIPPED (no discriminating payload)"
                                                              : "round_store_incumbent: PASS");
    return !all ? 1 : skipped ? 2 : 0;
}
