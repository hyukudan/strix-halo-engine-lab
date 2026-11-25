// Residual transition, HIP body: residual_mix::launch_transition stores, in one launch, every image of the leaves it
// replaces with their bits: launch_add (plain or BF16-rounded, in place or not) into the current bank, launch_mix (the
// paired-bank one-row readout) into each query row, and enqueue_rms_bf16_rows (parallel) of each normed query row.
// Cases cover the decode shapes of the bound engine (width 2304, 2..5 banks, 1 or 3 queries, the attention-site
// AddRound and the MLP-site in-place add, norms on some queries only, a missing gain), normalized banks, the register
// width limit (4096), an odd width, non-finite values and admission refusals (one bank, width above the limit, an
// a8g32 image). No model. Prints "residual transition hip: PASS" when every case matches.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
// The fixture binds the add, readout and transition leaves; the unused kernels would warn here.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#include "core/residual_transition_hip.hpp"
#pragma clang diagnostic pop

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace mix = strix::core::residual_mix;
namespace norm = strix::ops::norm;

bool ok(hipError_t status) { return status == hipSuccess; }

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (!ok(hipMalloc(&device, host.size() * sizeof(T)))) return nullptr;
    if (!ok(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice))) return nullptr;
    return device;
}

std::vector<float> download(const float* device, std::size_t count) {
    std::vector<float> host(count);
    if (!ok(hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost))) host.assign(count, 0.0F);
    return host;
}

std::vector<float> values(std::size_t count, std::mt19937& random, float range, bool specials) {
    std::uniform_real_distribution<float> value(-range, range);
    std::vector<float> out(count);
    for (auto& x : out) x = value(random);
    if (specials) {
        const std::uint32_t bits[] = {0x7F800000U, 0xFF800000U, 0x7FC00000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
        for (std::size_t i = 5; i < out.size(); i += 613U) std::memcpy(&out[i], &bits[random() % 6U], sizeof(float));
    }
    return out;
}

std::vector<std::uint16_t> gains(std::size_t count, std::mt19937& random) {
    std::uniform_real_distribution<float> value(0.5F, 1.5F);
    std::vector<std::uint16_t> out(count);
    for (auto& g : out) {
        const float x = value(random);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &x, sizeof bits);
        g = static_cast<std::uint16_t>(bits >> 16U);
    }
    return out;
}

// Bitwise comparison; NaN payloads included.
std::size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint32_t x = 0, y = 0;
        std::memcpy(&x, &a[i], sizeof x);
        std::memcpy(&y, &b[i], sizeof y);
        bad += x != y ? 1U : 0U;
    }
    return bad;
}

enum class Prologue { none, add, add_round_in_place };

struct Case {
    const char* name;
    unsigned width;
    unsigned banks;
    unsigned queries;
    Prologue prologue;
    unsigned norm_mask;  // bit q: query q is normed
    bool gain_on_first;  // false: query 0's norm runs without a gain
    bool normalized;
    bool specials;
};

// One arm's device images, uploaded from the same host data.
struct Arm {
    std::vector<float*> banks;
    float* rhs{};
    std::vector<float*> outputs;
    std::vector<float*> normed;
};

Arm make_arm(const std::vector<std::vector<float>>& banks, const std::vector<float>& rhs, unsigned queries,
             std::size_t width) {
    Arm arm;
    for (const auto& bank : banks) arm.banks.push_back(upload(bank));
    arm.rhs = upload(rhs);
    const std::vector<float> poison(width, -7.0F);
    for (unsigned q = 0; q < queries; ++q) {
        arm.outputs.push_back(upload(poison));
        arm.normed.push_back(upload(poison));
    }
    return arm;
}

bool valid(const Arm& arm) {
    if (!arm.rhs) return false;
    for (auto* p : arm.banks) if (!p) return false;
    for (std::size_t q = 0; q < arm.outputs.size(); ++q) if (!arm.outputs[q] || !arm.normed[q]) return false;
    return true;
}

mix::TransitionArgs bind(const Case& test, Arm& arm, const float* queries, const std::uint16_t* gain) {
    mix::TransitionArgs args{};
    auto& m = args.mix;
    for (unsigned b = 0; b < test.banks; ++b) m.banks[b] = arm.banks[b];
    m.bank_count = test.banks;
    m.queries = queries;
    m.rows = 1;
    m.width = test.width;
    m.query_count = test.queries;
    m.epsilon = 1e-6F;
    m.normalized = test.normalized;
    for (unsigned q = 0; q < test.queries; ++q) m.output[q] = arm.outputs[q];
    float* current = arm.banks[test.banks - 1U];
    if (test.prologue == Prologue::add) args.add = {arm.rhs, arm.banks[0], current, test.width, false};
    // Attention site: hidden = bf16(residual + readout), in place over the current bank image.
    if (test.prologue == Prologue::add_round_in_place) args.add = {current, arm.rhs, current, test.width, true};
    for (unsigned q = 0; q < test.queries; ++q) {
        if (((test.norm_mask >> q) & 1U) == 0U) continue;
        args.norm[q].output = arm.normed[q];
        args.norm[q].gain = (q == 0U && !test.gain_on_first) ? nullptr : gain + std::size_t{q} * test.width;
    }
    args.norm_epsilon = 1e-6F;
    return args;
}

// Incumbent leaves in program order.
bool run_leaves(const mix::TransitionArgs& args, hipStream_t stream) {
    if (mix::has_prologue(args) && !ok(mix::launch_add(args.add, stream))) return false;
    if (!ok(mix::launch_mix(args.mix, stream))) return false;
    for (unsigned q = 0; q < args.mix.query_count; ++q) {
        if (!args.norm[q].output) continue;
        const norm::Bf16RmsRows rows{args.mix.output[q], args.norm[q].gain, args.norm[q].output, 1U, args.mix.width,
                                     args.mix.width, args.mix.width};
        if (!ok(norm::enqueue_rms_bf16_rows(rows, args.norm_epsilon, norm::RmsRowsReduction::parallel, stream)))
            return false;
    }
    return true;
}

bool run_case(const Case& test, std::uint32_t seed, hipStream_t stream) {
    std::mt19937 random(seed);
    const std::size_t width = test.width;
    std::vector<std::vector<float>> bank_rows;
    for (unsigned b = 0; b < test.banks; ++b) bank_rows.push_back(values(width, random, 2.0F, test.specials));
    const auto rhs = values(width, random, 2.0F, test.specials);
    auto* queries = upload(values(std::size_t{test.queries} * width, random, 0.25F, test.specials));
    auto* gain = upload(gains(std::size_t{test.queries} * width, random));
    Arm leaves = make_arm(bank_rows, rhs, test.queries, width);
    Arm fused = make_arm(bank_rows, rhs, test.queries, width);
    if (!queries || !gain || !valid(leaves) || !valid(fused)) {
        std::printf("residual transition hip %s: FAIL (allocation)\n", test.name);
        return false;
    }
    const auto leaf_args = bind(test, leaves, queries, gain);
    const auto fused_args = bind(test, fused, queries, gain);
    if (!mix::hip_transition_admits(fused_args)) {
        std::printf("residual transition hip %s: FAIL (admission)\n", test.name);
        return false;
    }
    if (!run_leaves(leaf_args, stream) || !ok(mix::launch_transition(fused_args, stream)) ||
        !ok(hipStreamSynchronize(stream))) {
        std::printf("residual transition hip %s: FAIL (launch)\n", test.name);
        return false;
    }
    std::size_t bad = 0;
    for (unsigned b = 0; b < test.banks; ++b)
        bad += mismatches(download(leaves.banks[b], width), download(fused.banks[b], width));
    for (unsigned q = 0; q < test.queries; ++q) {
        bad += mismatches(download(leaves.outputs[q], width), download(fused.outputs[q], width));
        bad += mismatches(download(leaves.normed[q], width), download(fused.normed[q], width));
    }
    std::printf("residual transition hip %s: %s (%zu mismatching values)\n", test.name, bad == 0 ? "ok" : "FAIL", bad);
    return bad == 0;
}

// Review regression: the add operands may alias a shifted window of the current bank (lhs = output +- shift in one
// allocation). Every operand must be read before any store, so the bank equals lhs + rhs of the operand snapshot, and
// the readout and norms equal the leaves run on that expected bank without a prologue.
bool shifted_alias(unsigned width, int shift, bool rhs_aliases, std::uint32_t seed, hipStream_t stream) {
    std::mt19937 random(seed);
    const auto span = static_cast<std::size_t>(shift < 0 ? -shift : shift);
    const std::size_t room = width + span;
    const auto bank0 = values(width, random, 2.0F, false);
    const auto buffer = values(room, random, 2.0F, false);
    const auto other = values(width, random, 2.0F, false);
    auto* queries = upload(values(width, random, 0.25F, false));
    auto* gain = upload(gains(width, random));
    float* all = upload(buffer);
    float* rhs_dev = upload(other);
    // shift > 0: lhs above the output; shift < 0: lhs below it. rhs_aliases: rhs is the other end of the window.
    float* output = all + (shift > 0 ? 0 : span);
    const float* lhs = all + (shift > 0 ? span : 0);
    const float* rhs = rhs_aliases ? output : rhs_dev;
    const float* lhs_host = buffer.data() + (shift > 0 ? span : 0);
    const float* rhs_host = rhs_aliases ? buffer.data() + (shift > 0 ? 0 : span) : other.data();
    std::vector<float> expected(width);
    for (std::size_t i = 0; i < width; ++i) expected[i] = lhs_host[i] + rhs_host[i];
    Arm leaves = make_arm({bank0, expected}, other, 1, width);
    Arm fused = make_arm({bank0}, other, 1, width);
    if (!queries || !gain || !all || !rhs_dev || !valid(leaves) || !valid(fused)) {
        std::printf("residual transition hip shifted alias: FAIL (allocation)\n");
        return false;
    }
    fused.banks.push_back(output);
    const Case test{"shifted alias", width, 2, 1, Prologue::none, 0x1, true, false, false};
    const auto leaf_args = bind(test, leaves, queries, gain);
    auto fused_args = bind(test, fused, queries, gain);
    fused_args.add = {lhs, rhs, output, width, false};
    if (!mix::hip_transition_admits(fused_args)) {
        std::printf("residual transition hip shifted alias: FAIL (admission)\n");
        return false;
    }
    if (!run_leaves(leaf_args, stream) || !ok(mix::launch_transition(fused_args, stream)) ||
        !ok(hipStreamSynchronize(stream))) {
        std::printf("residual transition hip shifted alias: FAIL (launch)\n");
        return false;
    }
    std::size_t bad = mismatches(expected, download(output, width));
    bad += mismatches(download(leaves.outputs[0], width), download(fused.outputs[0], width));
    bad += mismatches(download(leaves.normed[0], width), download(fused.normed[0], width));
    std::printf("residual transition hip shifted alias width %u shift %d%s: %s (%zu mismatching values)\n", width, shift,
                rhs_aliases ? " rhs aliased" : "", bad == 0 ? "ok" : "FAIL", bad);
    return bad == 0;
}

// Shapes the HIP body declines: the caller keeps the separate leaves.
bool refusals() {
    float dummy[8]{};
    std::int8_t codes[8]{};
    mix::TransitionArgs args{};
    args.mix.banks[0] = dummy;
    args.mix.bank_count = 1;
    args.mix.queries = dummy;
    args.mix.rows = 1;
    args.mix.width = 2304;
    args.mix.query_count = 1;
    args.mix.epsilon = 1e-6F;
    std::vector<float> out(2304), bank(2304), q(2304);
    args.mix.banks[0] = bank.data();
    args.mix.queries = q.data();
    args.mix.output[0] = out.data();
    const bool one_bank = !mix::hip_transition_admits(args);
    std::vector<float> bank2(5000), out2(5000), q2(5000), bank3(5000);
    auto wide = args;
    wide.mix.banks[0] = bank2.data();
    wide.mix.banks[1] = bank3.data();
    wide.mix.bank_count = 2;
    wide.mix.queries = q2.data();
    wide.mix.output[0] = out2.data();
    wide.mix.width = 5000;
    const bool too_wide = !mix::hip_transition_admits(wide);
    std::vector<float> bank4(2304), normed(2304), deltas(72);
    auto image = args;
    image.mix.banks[1] = bank4.data();
    image.mix.bank_count = 2;
    image.norm[0].output = normed.data();
    image.norm[0].codes = codes;
    image.norm[0].deltas = deltas.data();
    image.norm_epsilon = 1e-6F;
    const bool with_image = !mix::hip_transition_admits(image);
    auto plain = image;
    plain.norm[0].codes = nullptr;
    plain.norm[0].deltas = nullptr;
    const bool admitted = mix::hip_transition_admits(plain);
    const bool pass = one_bank && too_wide && with_image && admitted;
    std::printf("residual transition hip refusals: %s (one bank %d, width 5000 %d, image %d, plain admitted %d)\n",
                pass ? "ok" : "FAIL", one_bank, too_wide, with_image, admitted);
    return pass;
}

}  // namespace

int main() {
    hipStream_t stream{};
    if (!ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking))) {
        std::printf("residual transition hip: FAIL (stream)\n");
        return 1;
    }
    const Case cases[] = {
        {"attn-site sliding 2 banks", 2304, 2, 1, Prologue::add_round_in_place, 0x1, true, false, false},
        {"attn-site full 3 queries", 2304, 4, 3, Prologue::add_round_in_place, 0x7, true, false, false},
        {"mlp-site 5 banks", 2304, 5, 1, Prologue::add, 0x1, true, false, false},
        {"block start no prologue", 2304, 3, 1, Prologue::none, 0x1, true, false, false},
        {"readout only (controller pass)", 2304, 4, 1, Prologue::add_round_in_place, 0x0, true, false, false},
        {"some queries normed, no gain", 2304, 3, 3, Prologue::add, 0x5, false, false, false},
        {"normalized banks", 2304, 4, 2, Prologue::none, 0x3, true, true, false},
        {"register limit 4096, 8 banks", 4096, 8, 2, Prologue::add, 0x3, true, false, false},
        {"odd width", 1001, 3, 1, Prologue::add_round_in_place, 0x1, true, false, false},
        {"non-finite values", 2304, 4, 3, Prologue::add_round_in_place, 0x7, true, false, true},
    };
    bool pass = refusals();
    std::uint32_t seed = 1234;
    for (const auto& test : cases) pass = run_case(test, seed++, stream) && pass;
    for (const unsigned width : {512U, 2304U})
        for (const int shift : {128, -128, 1, -1})
            for (const bool rhs_aliases : {false, true}) pass = shifted_alias(width, shift, rhs_aliases, seed++, stream) && pass;
    std::printf("residual transition hip: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
