#include "strix/ds41_prepared_load.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
int main(int argc, char** argv) {
    try {
        const bool omitted = argc >= 4 && (std::string(argv[3]) == "ep" || std::string(argv[3]) == "tp");
        const int shift = omitted ? 1 : 0;
        if (argc < 5 - shift || argc > 8 - shift) throw std::invalid_argument("usage: strix-ds41-prepare-load MODEL [NEW_DIRECTORY|-] RANK ep|tp [OVERLAY|-] [LAYERS=40] [packed|dup]");
        const std::string mode = argv[4 - shift], layout = argc > 7 - shift ? argv[7 - shift] : "packed";
        if ((mode != "ep" && mode != "tp") || (layout != "packed" && layout != "dup")) throw std::invalid_argument("invalid mode/layout");
        strix::ds41::prepared::prepare(argv[1], argc > 5 - shift && std::string(argv[5 - shift]) != "-" ? argv[5 - shift] : "", omitted ? "" : argv[2],
                                     static_cast<unsigned>(std::stoul(argv[3 - shift])), 2, mode == "tp", layout == "packed",
                                     argc > 6 - shift ? static_cast<unsigned>(std::stoul(argv[6 - shift])) : 40);
        std::puts("prepared artifact committed"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
