#include "gemma4/prefill/emulator.hpp"
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace pf = strix::gemma4::prefill;
using namespace strix::gemma4;
unsigned load32(const std::uint8_t* b) {
    return unsigned(b[0]) | (unsigned(b[1]) << 8U) | (unsigned(b[2]) << 16U) | (unsigned(b[3]) << 24U);
}
int main(int argc, char** argv) try {
    if (argc > 2) throw std::invalid_argument("usage: emulator-tests [TRACE.bin]");
    const auto path = argc == 2 ? argv[1] : G4_PREFILL_FIXTURE;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto size = static_cast<std::streamoff>(f.tellg());
    if (!f || size <= 0 || size > 64 * 1024 * 1024 || size % 72) throw std::runtime_error("NO-REFERENCE: invalid instruction fixture");
    f.seekg(0);
    std::array<std::uint8_t, 72> record{};
    std::size_t count = 0;
    while (f.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()))) {
        std::array<std::uint16_t, 16> a{}, b{};
        for (unsigned i = 0; i < 16; ++i) { a[i] = load16(record.data() + 2 * i); b[i] = load16(record.data() + 32 + 2 * i); }
        const auto result = bit_cast<unsigned>(pf::instruction(a, b, bit_cast<float>(load32(record.data() + 64))));
        if (result != load32(record.data() + 68)) throw std::runtime_error("instruction byte mismatch at case " + std::to_string(count));
        ++count;
    }
    std::printf("WMMA emulator instruction bytes EXACT cases=%zu\n", count);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
