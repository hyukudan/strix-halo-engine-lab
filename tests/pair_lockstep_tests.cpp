#include "strix/pair/lockstep.hpp"
#include "strix/pair/handshake.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

template <class F>
std::string thrown(F&& f) {
    try {
        f();
    } catch (const strix::pair::ProtocolError& e) {
        return e.what();
    }
    return {};
}

void test_rounds_advance_in_step() {
    strix::pair::Lockstep a(0), b(1);
    for (int k = 0; k < 4; ++k) {
        const auto ha = a.header(), hb = b.header();
        a.accept(hb);
        b.accept(ha);
    }
    expect(a.round() == 4U && b.round() == 4U, "both ranks advance one round per exchange");
    expect(a.header().ack_round == 3U, "ack_round is the last completed round");
    expect(strix::pair::Lockstep(0).header().ack_round == strix::pair::kNoRound, "first round acknowledges none");
}

void test_duplicate_or_stale_round() {
    strix::pair::Lockstep a(0), b(1);
    a.accept(b.header());  // b missed this round (dropped message): a is one ahead
    const auto e = thrown([&] { a.accept(b.header()); });
    expect(e.find("stale round 0, expected 1") != std::string::npos, "stale round names both rounds");
    expect(a.round() == 1U, "a rejected round does not advance");
    expect(!thrown([] { strix::pair::check_round(7, 8); }).empty(), "check_round rejects a mismatch");
    expect(thrown([] { strix::pair::check_round(7, 7); }).empty(), "check_round accepts an equal round");
}

void test_ack_mismatch() {
    strix::pair::LockstepHeader mine{3, 2, 0}, peer{3, 1, 1};
    expect(thrown([&] { strix::pair::check_pair(mine, peer); }).find("not acknowledged") != std::string::npos,
           "ack mismatch rejected");
}

void test_same_rank_and_range() {
    strix::pair::Lockstep a(0), twin(0);
    expect(thrown([&] { a.check(twin.header()); }).find("same rank") != std::string::npos, "same rank rejected");
    strix::pair::LockstepHeader mine{0, strix::pair::kNoRound, 0}, peer{0, strix::pair::kNoRound, 2};
    expect(thrown([&] { strix::pair::check_pair(mine, peer); }).find("out of range") != std::string::npos,
           "rank > 1 rejected");
}

void test_startup_hello() {
    using strix::pair::PairHello;
    PairHello a, b;
    a.protocol = b.protocol = 23;
    b.rank = 1;
    a.executable_sha256[31] = b.executable_sha256[31] = 1;
    a.model_sha256[31] = b.model_sha256[31] = 2;
    strix::pair::set_gpu_arch(a, "gfx1151");
    strix::pair::set_gpu_arch(b, "gfx1151");
    a.route = b.route = 0x8000000080000000ULL;
    a.offers[0] = 8;
    b.offers[0] = 4;
    std::size_t exchanges = 0;
    const auto exchange = [&](auto out, auto in) {
        ++exchanges;
        expect(out.size() == 304U && in.size() == 304U, "one fixed-size startup frame per rank");
        expect(std::memcmp(out.data(), &a, sizeof a) == 0, "all startup fields are sent");
        std::memcpy(in.data(), &b, sizeof b);
    };
    const auto peer = strix::pair::exchange_hello(a, exchange);
    expect(exchanges == 1U && peer.offers[0] == 4U, "one exchange carries independent memory offers");
    const auto valid = b;
    const auto reject = [&](std::string_view field) {
        const auto error = thrown([&] { (void)strix::pair::exchange_hello(a, exchange); });
        expect(error.find(field) != std::string::npos, "pair hello reports the mismatched field");
        b = valid;
    };
    b.version = 0;
    reject("wire version");
    b.rank = a.rank;
    reject("complementary ranks");
    b.executable_sha256[31] = 0;
    reject("executable SHA-256");
    b.model_sha256[31] = 0;
    reject("model identity");
    strix::pair::set_gpu_arch(b, "gfx1100");
    reject("GPU architecture");
    b.route = 0;
    reject("route fingerprint");  // the old GLM xor-fold would have matched
    b.placement_reference = 1;
    reject("placement and receipt");
    b.transport_capabilities = 1;
    reject("transport capabilities");
    b.contract[0] = 2;
    constexpr std::array<std::string_view, 1> fields{"wire and split policy"};
    expect(thrown([&] { (void)strix::pair::exchange_hello(a, exchange, fields); }).find(fields[0]) != std::string::npos,
           "bindings can name a wire or split mismatch");
}

}  // namespace

int main() {
    test_rounds_advance_in_step();
    test_duplicate_or_stale_round();
    test_ack_mismatch();
    test_same_rank_and_range();
    test_startup_hello();
    if (failures) return EXIT_FAILURE;
    std::printf("PASS pair lockstep\n");
    return EXIT_SUCCESS;
}
