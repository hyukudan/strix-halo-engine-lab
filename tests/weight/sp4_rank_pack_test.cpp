// A rank's share of a pinned SP4 pack (strix/weight/sp4_rank_pack.hpp), CPU only: cuts split evenly or are refused,
// the materialized bytes are the slice functions' bytes, the decline rule keeps its order, the pin is never trusted
// when empty, malformed or zero, and the payload is hashed only once the identity equals the pin.
#include "strix/weight/sp4_rank_pack.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace sr = strix::weight::sp4_rank;
namespace sl = strix::weight::sp4_slice;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}
template <class F>
bool throws(F&& f) {
    try {
        (void)f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

struct Format {
    int id = 0;
    friend bool operator==(const Format&, const Format&) = default;
};
struct Entry {
    std::uint64_t rows = 0, columns = 0, bytes = 0;
    Format storage{};
    std::optional<int> sp_header{};
};
struct FakePack {
    strix::weight::Digest256 id{};
    bool payload = true;
    mutable int verified = 0;
    std::map<std::string, Entry, std::less<>> entries;
    [[nodiscard]] const strix::weight::Digest256& identity() const { return id; }
    [[nodiscard]] bool verify_identity() const {
        ++verified;
        return payload;
    }
    [[nodiscard]] const Entry* entry(std::string_view n) const {
        const auto it = entries.find(n);
        return it == entries.end() ? nullptr : &it->second;
    }
};

std::vector<std::uint8_t> synthetic(std::uint64_t rows, std::uint64_t columns) {
    std::vector<std::uint8_t> m(rows * sl::row_bytes(columns));
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = static_cast<std::uint8_t>((i * 2654435761ULL) >> 13);
    return m;
}

void cuts() {
    const auto o = sr::output_share(512, 256, 1, 2);
    check(!o.by_input && o.first == 256 && o.count == 256 && o.slice_rows() == 256 && o.slice_columns() == 256, "output share");
    const auto i = sr::input_share(64, 512, 1, 2);
    check(i.by_input && i.first == 256 && i.count == 256 && i.slice_rows() == 64 && i.slice_columns() == 256, "input share");
    const auto w = sr::whole(64, 256);
    check(w.slice_rows() == 64 && w.slice_columns() == 256 && w.slice_bytes() == 64 * sl::row_bytes(256), "whole");
    check(throws([] { return sr::output_share(511, 256, 0, 2); }), "odd outputs refused");
    check(throws([] { return sr::input_share(64, 384, 0, 2); }), "half superblock input share refused");
    check(throws([] { return sr::output_share(512, 256, 2, 2); }), "rank outside the world refused");

    const auto m = synthetic(64, 512);
    std::vector<std::uint8_t> scratch;
    const auto vo = sr::materialize(sr::output_share(64, 512, 1, 2), m, scratch);
    const auto ro = sl::output_slice(m, 64, 512, 32, 32);
    check(vo.data() == ro.data() && vo.size() == ro.size() && scratch.empty(), "output cut is the slice view");
    const auto vi = sr::materialize(i, m, scratch);
    const auto ri = sl::input_slice(m, 64, 512, 256, 256);
    check(std::vector<std::uint8_t>(vi.begin(), vi.end()) == ri && vi.data() == scratch.data(), "input cut is the gathered slice");
    const auto vw = sr::materialize(sr::whole(64, 512), m, scratch);
    check(vw.size() == m.size() && vw.data() == m.data(), "whole cut is the matrix");
}

void rule() {
    sr::PackFacts all{true, true, true, true, true, true, true, true, true};
    check(sr::decide(false, false, {}) == sr::Outcome::off, "nothing requested -> off before anything else");
    check(sr::decide(true, false, all) == sr::Outcome::prerequisite, "prerequisite before the pack");
    check(sr::decide(true, true, all) == sr::Outcome::on, "all facts -> on");
    // Each fact, cleared alone, yields its own outcome (order of the rule).
    const sr::Outcome expect[] = {sr::Outcome::no_pack, sr::Outcome::open_failed, sr::Outcome::identity_unpinned,
                                  sr::Outcome::identity_malformed, sr::Outcome::identity_mismatch, sr::Outcome::payload_mismatch,
                                  sr::Outcome::entry_missing, sr::Outcome::entry_shape, sr::Outcome::not_sp4t};
    bool sr::PackFacts::*fields[] = {&sr::PackFacts::exists, &sr::PackFacts::opened, &sr::PackFacts::pin_set,
                                     &sr::PackFacts::pin_wellformed, &sr::PackFacts::identity_equal, &sr::PackFacts::payload_ok,
                                     &sr::PackFacts::entries_present, &sr::PackFacts::entries_shaped, &sr::PackFacts::entries_sp4t};
    for (std::size_t k = 0; k < 9; ++k) {
        auto f = all;
        f.*fields[k] = false;
        check(sr::decide(true, true, f) == expect[k], std::string("fact ") + std::to_string(k) + " -> its outcome");
        check(std::string(sr::outcome_text(expect[k])).rfind("declined", 0) == 0, "decline text");
    }
}

void facts() {
    FakePack pk;
    for (std::size_t b = 0; b < pk.id.bytes.size(); ++b) pk.id.bytes[b] = static_cast<std::uint8_t>(b + 1);
    const std::string pin = sr::hex(pk.id);
    check(pin.size() == 64 && pin.substr(0, 4) == "0102", "hex digest");
    const Format sp4t{7};
    pk.entries["a"] = {512, 256, 512 * sl::row_bytes(256), sp4t, 1};
    pk.entries["b"] = {64, 512, 64 * sl::row_bytes(512), sp4t, 1};
    const std::vector<sr::Need> needs{{"a", sr::output_share(512, 256, 0, 2)}, {"b", sr::input_share(64, 512, 0, 2)}};
    auto f = sr::pack_facts(true, true, pk, pin, needs, sp4t);
    check(sr::decide(true, true, f) == sr::Outcome::on && pk.verified == 1, "pinned pack with every entry -> on");
    check(sr::arena_bytes(needs) == ((256 * sl::row_bytes(256) + 255) & ~255ULL) + ((64 * sl::row_bytes(256) + 255) & ~255ULL),
          "arena bytes = aligned slices");

    pk.verified = 0;
    check(sr::decide(true, true, sr::pack_facts(true, true, pk, "", needs, sp4t)) == sr::Outcome::identity_unpinned &&
              pk.verified == 0,
          "empty pin declines without hashing");
    check(sr::decide(true, true, sr::pack_facts(true, true, pk, std::string(64, '0'), needs, sp4t)) == sr::Outcome::identity_malformed,
          "zero pin is malformed");
    check(sr::decide(true, true, sr::pack_facts(true, true, pk, pin.substr(1), needs, sp4t)) == sr::Outcome::identity_malformed,
          "short pin is malformed");
    std::string other = pin;
    other[63] = other[63] == 'f' ? 'e' : 'f';
    check(sr::decide(true, true, sr::pack_facts(true, true, pk, other, needs, sp4t)) == sr::Outcome::identity_mismatch &&
              pk.verified == 0,
          "other pin -> mismatch without hashing");
    pk.payload = false;
    check(sr::decide(true, true, sr::pack_facts(true, true, pk, pin, needs, sp4t)) == sr::Outcome::payload_mismatch, "bad payload");
    pk.payload = true;
    check(sr::decide(true, true, sr::pack_facts(false, true, pk, pin, needs, sp4t)) == sr::Outcome::no_pack, "absent pack");
    check(sr::decide(true, true, sr::pack_facts(true, false, pk, pin, needs, sp4t)) == sr::Outcome::open_failed, "unreadable pack");

    auto bad = pk;
    bad.entries.erase("b");
    check(sr::decide(true, true, sr::pack_facts(true, true, bad, pin, needs, sp4t)) == sr::Outcome::entry_missing, "missing entry");
    bad = pk;
    bad.entries["b"].bytes -= 1;
    check(sr::decide(true, true, sr::pack_facts(true, true, bad, pin, needs, sp4t)) == sr::Outcome::entry_shape, "short entry");
    bad = pk;
    bad.entries["a"].columns = 128;
    check(sr::decide(true, true, sr::pack_facts(true, true, bad, pin, needs, sp4t)) == sr::Outcome::entry_shape, "wrong columns");
    bad = pk;
    bad.entries["a"].storage = Format{3};
    check(sr::decide(true, true, sr::pack_facts(true, true, bad, pin, needs, sp4t)) == sr::Outcome::not_sp4t, "other storage");
    bad = pk;
    bad.entries["a"].sp_header.reset();
    check(sr::decide(true, true, sr::pack_facts(true, true, bad, pin, needs, sp4t)) == sr::Outcome::not_sp4t, "no table header");
}

}  // namespace

int main() {
    cuts();
    rule();
    facts();
    if (g_failures != 0) {
        std::fprintf(stderr, "sp4 rank pack: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("sp4 rank pack: all checks passed\n");
    return 0;
}
