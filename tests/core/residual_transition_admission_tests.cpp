// Admission of the HIP residual transition body (residual_mix::hip_transition_admits), host only: the decode shapes
// the bound engine fuses are admitted (attention-site in-place BF16 add, MLP-site add, no prologue, readout only),
// everything the HIP body does not reproduce is declined so the caller keeps the separate leaves (one bank: fewer
// than 256 norm lanes; width above the register readout; an a8g32 image; a prologue that does not store the current
// bank; an output that overlaps a bank).
#include "core/residual_mix.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace mix = strix::core::residual_mix;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("residual transition admission: FAIL %s\n", what);
        ++failures;
    }
}

struct Images {
    explicit Images(std::size_t width, unsigned banks, unsigned queries)
        : bank(banks, std::vector<float>(width)), query(std::size_t{queries} * width), out(queries, std::vector<float>(width)),
          normed(queries, std::vector<float>(width)), rhs(width), gain(std::size_t{queries} * width) {}
    std::vector<std::vector<float>> bank;
    std::vector<float> query;
    std::vector<std::vector<float>> out;
    std::vector<std::vector<float>> normed;
    std::vector<float> rhs;
    std::vector<std::uint16_t> gain;
};

mix::TransitionArgs decode_args(Images& images, unsigned width, bool prologue, bool round) {
    mix::TransitionArgs args{};
    auto& m = args.mix;
    m.bank_count = static_cast<unsigned>(images.bank.size());
    for (unsigned b = 0; b < m.bank_count; ++b) m.banks[b] = images.bank[b].data();
    m.queries = images.query.data();
    m.rows = 1;
    m.width = width;
    m.query_count = static_cast<unsigned>(images.out.size());
    m.epsilon = 1e-6F;
    for (unsigned q = 0; q < m.query_count; ++q) {
        m.output[q] = images.out[q].data();
        args.norm[q].output = images.normed[q].data();
        args.norm[q].gain = images.gain.data() + std::size_t{q} * width;
    }
    args.norm_epsilon = 1e-6F;
    float* current = images.bank.back().data();
    if (prologue) args.add = {current, images.rhs.data(), current, width, round};
    return args;
}

}  // namespace

int main() {
    Images sliding(2304, 2, 1);
    check(mix::hip_transition_admits(decode_args(sliding, 2304, true, true)), "attention site, in-place AddRound");
    check(mix::hip_transition_admits(decode_args(sliding, 2304, false, false)), "block start, no prologue");
    Images full(2304, 4, 3);
    check(mix::hip_transition_admits(decode_args(full, 2304, true, false)), "full attention site, three norms");
    auto readout_only = decode_args(full, 2304, true, true);
    for (auto& norm : readout_only.norm) norm = {};
    check(mix::hip_transition_admits(readout_only), "readout only (the controller consumes the row)");

    Images one(2304, 1, 1);
    check(!mix::hip_transition_admits(decode_args(one, 2304, false, false)), "one bank declined");
    Images wide(4097, 2, 1);
    check(!mix::hip_transition_admits(decode_args(wide, 4097, false, false)), "width above the register readout declined");
    Images limit(4096, 8, 2);
    check(mix::hip_transition_admits(decode_args(limit, 4096, true, false)), "register width limit admitted");

    std::vector<std::int8_t> codes(2304);
    std::vector<float> deltas(72);
    auto image = decode_args(sliding, 2304, false, false);
    image.norm[0].codes = codes.data();
    image.norm[0].deltas = deltas.data();
    check(mix::fits(image) && !mix::hip_transition_admits(image), "a8g32 image is CUDA-only: declined on HIP");

    auto stray = decode_args(sliding, 2304, true, false);
    stray.add.output = sliding.rhs.data();
    check(!mix::hip_transition_admits(stray), "prologue that does not store the current bank declined");
    auto overlap = decode_args(sliding, 2304, false, false);
    overlap.mix.output[0] = sliding.bank[0].data();
    check(!mix::hip_transition_admits(overlap), "readout output over a bank declined");

    std::printf("residual transition admission: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
