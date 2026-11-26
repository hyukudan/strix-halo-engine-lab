#include "inkling2/inkling2_mini_weight_options.hpp"

#include <iostream>

namespace mini = strix::inkling2;

int main() {
    // A bound calibrated decode image serves every prompt emits-first: its final prompt token is a BF16 prefill row,
    // never a compact decode row on the held-back route, whatever the served route lever says.
    if (!mini::mini_served_prompt_emits_first(false, false, true)) {
        std::cerr << "a calibrated decode image met the held-back served prompt route\n";
        return 1;
    }
    if (mini::mini_served_prompt_emits_first(false, false, false) ||
        !mini::mini_served_prompt_emits_first(true, false, false) ||
        !mini::mini_served_prompt_emits_first(false, true, false)) {
        std::cerr << "the incumbent served prompt route changed\n";
        return 1;
    }
}
