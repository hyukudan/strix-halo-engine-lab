#include "gemma4/exl3/format.hpp"

#include <iostream>
#include <sstream>
#include <string_view>

int gemma4_decode_entry(int argc, char** argv);
int gemma4_serve_entry(int argc, char** argv);

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    const auto entry = std::string_view(argv[1]) == "decode" ? gemma4_decode_entry : gemma4_serve_entry;
    const auto expected = unsigned(argv[2][0] - '0');
    argv[2] = argv[0];
    // The manifest has an arm and deliberately omits the schema. Runtime must
    // select the core before the load fails, with no fallback or model allocation.
    std::ostringstream diagnostic;
    auto* const stderr_buffer = std::cerr.rdbuf(diagnostic.rdbuf());
    const int result = entry(argc - 2, argv + 2);
    std::cerr.rdbuf(stderr_buffer);
    const auto selected = strix::gemma4::exl3::identity_slot().load();
    if (result != 1 || diagnostic.str().find("missing asset field: schema") == std::string::npos || selected != expected) {
        std::cerr << diagnostic.str();
        std::cerr << "startup status " << result << ", selected v" << selected << ", expected v" << expected << '\n';
        return 1;
    }
    std::cout << "startup selected v" << selected << " before pack-load refusal\n";
    return 0;
}
