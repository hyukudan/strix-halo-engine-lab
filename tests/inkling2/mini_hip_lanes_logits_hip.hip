#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "inkling2/inkling2_mini_resident_decoder.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// HIP eager lane mode (STRIX_INKLING_HIP_LANES, read when the decoder is built) against the per-lane launches:
// a cohort decode step with the lanes on must give every lane the full score row (read_slot_logits, the pitched row
// the greedy pick saw), the pick and the continuation of the same step with the lanes off, bit for bit.
// Widths 2, 9 and 32 (exec::kStepMaxRows), distinct prompt lengths per slot (distinct positions), slots listed in a
// nonsequential order, 4 greedy steps per width. The same process builds the decoder twice (off first, then on).
namespace strix::inkling2::test {

inline constexpr unsigned kLanesSteps = 4U;
inline constexpr unsigned kLanesMax = 32U;
inline constexpr std::array<std::array<unsigned, 2>, 3> kLanesWidths{{{2U, 1U}, {9U, 4U}, {32U, 7U}}};

inline std::vector<unsigned> lanes_prompt(unsigned slot) {
    std::vector<unsigned> ids(7U + 3U * slot);
    for (std::size_t i = 0; i < ids.size(); ++i)
        ids[i] = 1000U + static_cast<unsigned>((slot * 389U + i * 173U + i / 5U) % 20000U);
    return ids;
}

struct LanesRun {
    std::vector<float> scores;   // [width][step][row] x vocabulary, in call row order
    std::vector<unsigned> picks;  // [width][step][row]
};

inline LanesRun lanes_run(const char* checkpoint, bool packed) {
    MiniResidentDecoder decoder(checkpoint, 160U, 64U, packed, MiniResidentBackend::default_step_graphs,
                                weight::kBf16WeightStorage, 0U, false, kLanesMax);
    decoder.allocate_slots(kLanesMax);
    LanesRun run;
    std::vector<float> row(kOutputVocabulary);
    for (const auto& [lanes, stride] : kLanesWidths) {
        for (unsigned slot = 0; slot < lanes; ++slot) decoder.prefill_slot(slot, lanes_prompt(slot), true);
        // Row r of every call is slot (r * stride + 1) % lanes: a permutation (stride coprime to lanes).
        std::vector<unsigned> order(lanes), feeds(lanes), top(lanes);
        for (unsigned r = 0; r < lanes; ++r) {
            order[r] = (r * stride + 1U) % lanes;
            feeds[r] = 2000U + 97U * order[r];
        }
        for (unsigned step = 0; step < kLanesSteps; ++step) {
            decoder.decode_slots(order, feeds, top);
            for (unsigned r = 0; r < lanes; ++r) {
                decoder.read_slot_logits(r, row);
                run.scores.insert(run.scores.end(), row.begin(), row.end());
                run.picks.push_back(top[r]);
                feeds[r] = top[r];
            }
        }
    }
    return run;
}

inline void set_lanes(bool on) {
    if (on) ::unsetenv("STRIX_INKLING_HIP_LANES");
    else ::setenv("STRIX_INKLING_HIP_LANES", "0", 1);
}

inline void lanes_logits(const char* checkpoint, bool packed) {
    const auto moe = packed ? "packed" : "grouped";
    set_lanes(false);
    const auto off = lanes_run(checkpoint, packed);
    set_lanes(true);
    const auto on = lanes_run(checkpoint, packed);
    if (off.scores.size() != on.scores.size() || off.picks.size() != on.picks.size())
        throw std::runtime_error("lanes on/off runs differ in size");
    std::size_t base = 0U;
    for (const auto& [lanes, stride] : kLanesWidths) {
        (void)stride;
        for (unsigned step = 0; step < kLanesSteps; ++step) {
            for (unsigned r = 0; r < lanes; ++r, ++base) {
                const auto at = base * kOutputVocabulary;
                if (std::memcmp(off.scores.data() + at, on.scores.data() + at, kOutputVocabulary * sizeof(float)) != 0 ||
                    off.picks[base] != on.picks[base])
                    throw std::runtime_error("lanes on differs from off: C" + std::to_string(lanes) + " step " +
                                             std::to_string(step) + " row " + std::to_string(r));
            }
        }
        std::printf("PASS Inkling Mini HIP lanes logits %s C%u lanes on == off (%u rows x %u steps, scores + picks)\n",
                    moe, lanes, lanes, kLanesSteps);
    }
}

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    using namespace strix;
    try {
        if (argc < 2) throw std::invalid_argument("expected MODEL.safetensors [--packed-decode]");
        bool packed = false;
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--packed-decode") packed = true;
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        inkling2::test::lanes_logits(argv[1], packed);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini HIP lanes logits: " << error.what() << '\n';
        return 1;
    }
}
