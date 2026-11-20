// Host test of the shared MoE route tap (include/strix/moe/route_file.hpp), the engine side of the FAST gate v2:
// record -> replay round trip, local flip bookkeeping, mismatch refusals, exhaustion, the one-ulp null.
#include "strix/moe/route_file.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
template <typename F> bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}
std::string temp_file(const char* name) {
    return (std::filesystem::temp_directory_path() / (std::string("strix-route-") + name)).string();
}
}  // namespace

int main() {
    using strix::moe::RouteTap;
    using strix::moe::RouteTapConfig;
    const auto path = temp_file("ref.rte");
    // Reference arm: two calls (layer 0 two rows, layer 1 one row), top-2.
    std::vector<std::uint32_t> ids0 = {3, 7, 1, 2}, ids1 = {9, 4};
    std::vector<float> w0 = {0.75F, 0.25F, 0.5F, 0.5F}, w1 = {0.6F, 0.4F};
    {
        RouteTapConfig c;
        c.out_path = path;
        RouteTap tap(c);
        check(!tap.on_call(0, 2, 2, ids0.data(), w0.data()), "record only does not change the routes");
        check(!tap.on_call(1, 1, 2, ids1.data(), w1.data()), "record only does not change the routes (2)");
        check(tap.receipt().find("\"records\":2") != std::string::npos, "out receipt counts records");
    }
    // Replay arm: its own routes differ in one id of row 1 (layer 0) and in order only (layer 1).
    {
        RouteTapConfig c;
        c.in_path = path;
        RouteTap tap(c);
        std::vector<std::uint32_t> a0 = {7, 3, 1, 5}, a1 = {4, 9};
        std::vector<float> v0 = {0.1F, 0.2F, 0.3F, 0.4F}, v1 = {0.9F, 0.1F};
        check(tap.on_call(0, 2, 2, a0.data(), v0.data()), "replay changes the routes");
        check(a0 == ids0 && v0 == w0, "replay restores the recorded ids and weights");
        const auto mid = tap.receipt();
        check(mid.find("\"complete\":false") != std::string::npos, "receipt flags an incomplete replay");
        check(tap.on_call(1, 1, 2, a1.data(), v1.data()), "replay changes the routes (2)");
        const auto r = tap.receipt();
        check(r.find("\"consumed\":2") != std::string::npos && r.find("\"complete\":true") != std::string::npos,
              "receipt: all records consumed");
        // Order-only differences are not flips; one replaced id in layer 0 row 1 is one flip in 6 slots.
        check(r.find("\"local_flips\":{\"slots\":6,\"ids\":1,\"rows_changed\":1") != std::string::npos, "local flip count");
        check(r.find("[0,4,1]") != std::string::npos && r.find("[1,2,0]") != std::string::npos, "per-layer flips");
        std::vector<std::uint32_t> extra = {1, 2};
        std::vector<float> ew = {0.5F, 0.5F};
        check(throws([&] { tap.on_call(2, 1, 2, extra.data(), ew.data()); }), "replay past the end refuses");
    }
    // Shape mismatch refuses (layer, rows, top-k all checked).
    {
        RouteTapConfig c;
        c.in_path = path;
        RouteTap tap(c);
        std::vector<std::uint32_t> a = {1, 2, 3};
        std::vector<float> v = {1, 1, 1};
        check(throws([&] { tap.on_call(0, 1, 3, a.data(), v.data()); }), "replay with another top-k refuses");
    }
    {
        RouteTapConfig c;
        c.in_path = path;
        RouteTap tap(c);
        std::vector<std::uint32_t> a = {1, 2, 3, 4};
        std::vector<float> v = {1, 1, 1, 1};
        check(throws([&] { tap.on_call(5, 2, 2, a.data(), v.data()); }), "replay at another layer refuses");
    }
    // One-ulp null: only the named layer moves, every weight by exactly one ulp, deterministic per seed.
    {
        RouteTapConfig c;
        strix::moe::parse_route_null("1:7", c);
        check(c.null_on && c.null_layer == 1 && c.null_seed == 7, "null parse");
        RouteTap tap(c);
        std::vector<std::uint32_t> a = {3, 7, 1, 2};
        std::vector<float> v = w0, before = w0;
        check(!tap.on_call(0, 2, 2, a.data(), v.data()) && v == before, "null leaves other layers untouched");
        std::vector<std::uint32_t> b = {9, 4};
        std::vector<float> u = w1;
        check(tap.on_call(1, 1, 2, b.data(), u.data()), "null changes the named layer");
        bool one_ulp = true;
        for (std::size_t i = 0; i < u.size(); ++i)
            one_ulp = one_ulp && u[i] != w1[i] &&
                      (u[i] == std::nextafter(w1[i], 2.0F) || u[i] == std::nextafter(w1[i], -2.0F));
        check(one_ulp, "null moves every selected weight by one ulp");
        check(b == std::vector<std::uint32_t>({9, 4}), "null never changes ids");
        check(tap.receipt().find("\"nudged\":2") != std::string::npos, "null receipt counts nudges");
        RouteTap again(c);
        std::vector<float> u2 = w1;
        std::vector<std::uint32_t> b2 = {9, 4};
        again.on_call(1, 1, 2, b2.data(), u2.data());
        check(u2 == u, "null is deterministic per seed");
    }
    check(throws([] { RouteTapConfig c; strix::moe::parse_route_null("x:1", c); }), "bad null text refuses");
    check(strix::moe::route_set_diff(ids0.data(), ids0.data(), 2) == 0, "set diff of equal sets");
    {
        RouteTapConfig c;
        check(!c.active(), "no switch, inactive");
        RouteTap tap(c);
        std::vector<std::uint32_t> a = {1, 2};
        std::vector<float> v = {1, 1};
        check(!tap.on_call(0, 1, 2, a.data(), v.data()), "inactive tap is a no-op");
    }
    // Replaying a malformed route file must fail at open, not only when a later router call reads its body.
    // In particular, seekg() past EOF is permitted by iostreams, so a truncated payload needs a size check.
    const auto corrupt = temp_file("truncated.rte");
    std::filesystem::copy_file(path, corrupt, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(corrupt, std::filesystem::file_size(corrupt) - 1);
    check(throws([&] { RouteTapConfig c; c.in_path = corrupt; RouteTap tap(c); }),
          "truncated final route payload refused at open");
    std::filesystem::remove(corrupt);
    const auto trailing = temp_file("partial-header.rte");
    std::filesystem::copy_file(path, trailing, std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream f(trailing, std::ios::binary | std::ios::app);
        f.put('X');
    }
    check(throws([&] { RouteTapConfig c; c.in_path = trailing; RouteTap tap(c); }),
          "partial trailing route header refused at open");
    std::filesystem::remove(trailing);
    std::filesystem::remove(path);
    if (failures) return 1;
    std::printf("moe_route_file_tests: all passed\n");
    return 0;
}
