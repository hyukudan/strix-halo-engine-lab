#include "inkling2/inkling2_mini_weight_options.hpp"

#include <iostream>

namespace mini = strix::inkling2;

int main() {
    if (mini::mini_server_calibrated_row_allowed(true, false)) {
        std::cerr << "held-back HTTP prompt anchor selected the calibrated decode row\n";
        return 1;
    }
    if (!mini::mini_server_calibrated_row_allowed(true, true) ||
        mini::mini_server_calibrated_row_allowed(false, true)) {
        std::cerr << "emits-first or unset calibration admission changed\n";
        return 1;
    }
}
