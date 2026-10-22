#include "strix/ds41_prepared_load.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 8) throw std::invalid_argument("usage: strix-ds41-prepare-load MODEL DEST RANK ep|tp [OVERLAY|-] [LAYERS=40] [packed|dup]");
        const std::string mode = argv[4], layout = argc > 7 ? argv[7] : "packed";
        if ((mode != "ep" && mode != "tp") || (layout != "packed" && layout != "dup")) throw std::invalid_argument("invalid mode/layout");
        strix::ds41::prepared::prepare(argv[1], argc > 5 && std::string(argv[5]) != "-" ? argv[5] : "", argv[2],
                                     static_cast<unsigned>(std::stoul(argv[3])), 2, mode == "tp", layout == "packed",
                                     argc > 6 ? static_cast<unsigned>(std::stoul(argv[6])) : 40);
        std::puts("prepared artifact committed"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
