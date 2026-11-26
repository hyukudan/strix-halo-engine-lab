#include "strix/attn/head_rms_reference.hpp"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    // epsilon=1: the two heads have inverse RMS 1/3 and 1/sqrt(19).
    std::array heads{4.0F, 0.0F, -6.0F, 0.0F, 0.0F, 0.0F};
    strix::attn::head_rms_reference(heads, 2U, 1.0F);
    if (std::abs(heads[0] - 4.0F / 3.0F) > 1e-6F ||
        std::abs(heads[2] + 6.0F / std::sqrt(19.0F)) > 1e-6F ||
        heads[1] != 0.0F || heads[3] != 0.0F || heads[4] != 0.0F || heads[5] != 0.0F) {
        std::cerr << "incorrect independent head RMS reductions\n";
        return 1;
    }
    try { strix::attn::head_rms_reference(heads, 4U, 1.0F); }
    catch (const std::invalid_argument&) { return 0; }
    std::cerr << "partial head was accepted\n";
    return 1;
}
