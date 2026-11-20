// Host tests of the shared state image codec (include/strix/session/state_image.hpp): a round trip keeps every field
// bit for bit; another layout, another version, a flipped byte, a truncated file and trailing bytes are refused.
#include "strix/session/state_image.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace si = strix::session::state_image;

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
bool refused(std::vector<std::byte> bytes, std::string_view layout, const char* needle) {
    try {
        (void)si::decode(bytes, layout);
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}
si::Image sample() {
    si::Image im;
    im.layout = "hy4 first=0 last=42 context=33280 rows=8";
    im.position = 5;
    im.tokens = {1U, 70000U, 3U, 0xFFFFFFFFU, 120831U};
    for (int r = 0; r < 3; ++r) {
        si::Region reg;
        reg.name = "latent." + std::to_string(r);
        reg.bytes.resize(static_cast<std::size_t>(17 * (r + 1)));
        for (std::size_t i = 0; i < reg.bytes.size(); ++i) reg.bytes[i] = static_cast<std::byte>((i * 31U + static_cast<std::size_t>(r)) & 0xFFU);
        im.regions.push_back(reg);
    }
    im.regions.push_back({"empty", {}});
    return im;
}
}  // namespace

int main() {
    const auto im = sample();
    const auto bytes = si::encode(im);
    const auto back = si::decode(bytes, im.layout);
    expect(back.layout == im.layout && back.position == im.position && back.tokens == im.tokens, "round trip header");
    bool same = back.regions.size() == im.regions.size();
    for (std::size_t i = 0; same && i < im.regions.size(); ++i)
        same = back.regions[i].name == im.regions[i].name && back.regions[i].bytes == im.regions[i].bytes;
    expect(same, "round trip regions bit for bit");
    expect(si::encode(back) == bytes, "re-encode is byte identical");

    expect(refused(bytes, "hy4 first=0 last=42 context=4096 rows=8", "layout"), "another layout refused");
    auto flipped = bytes;
    flipped[bytes.size() / 2] ^= std::byte{1};
    expect(refused(flipped, im.layout, "checksum"), "flipped byte refused");
    auto cut = bytes;
    cut.resize(bytes.size() - 9);
    expect(refused(cut, im.layout, "checksum"), "truncated file refused");
    auto version = bytes;
    version[8] = std::byte{2};
    // re-seal the trailer so the version check itself is reached
    version.resize(version.size() - 8);
    {
        const auto h = si::fnv64(version);
        for (unsigned i = 0; i < 8U; ++i) version.push_back(static_cast<std::byte>((h >> (8U * i)) & 0xFFU));
    }
    expect(refused(version, im.layout, "version"), "other version refused");
    expect(refused(std::vector<std::byte>(10), im.layout, "shorter"), "tiny file refused");

    if (failures == 0) std::printf("session_state_image_tests: PASS\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
