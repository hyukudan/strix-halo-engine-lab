// pair::EmbedFrame: round trip, once-per-digest ledger, corruption, regime rule.
#include "strix/pair/embed_frame.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

// Test checksum (CRC-32C, bitwise): the pair passes dist::crc32c.
std::uint32_t crc(std::span<const std::byte> bytes) noexcept {
    std::uint32_t c = 0xFFFFFFFFU;
    for (const auto b : bytes) {
        c ^= static_cast<std::uint8_t>(b);
        for (int k = 0; k < 8; ++k) c = (c >> 1U) ^ (0x82F63B78U & (0U - (c & 1U)));
    }
    return ~c;
}

using namespace strix;

session::EmbedRows span(std::uint8_t tag, std::uint32_t rows, std::uint32_t width) {
    session::EmbedRows r;
    r.digest.fill(tag);
    r.rows = rows;
    r.width = width;
    r.real_token_id = 129264;
    r.data.resize(std::size_t{rows} * width);
    for (std::size_t i = 0; i < r.data.size(); ++i) r.data[i] = static_cast<std::uint16_t>(tag * 1000U + i);
    r.role.assign(rows, session::EmbedRole::content);
    r.role.front() = session::EmbedRole::begin;
    r.role.back() = session::EmbedRole::end;
    return r;
}

template <class F>
bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

}  // namespace

int main() {
    // Transfer boundaries: the pair gate's 22 MiB cap, including a partial final message.
    constexpr std::size_t cap = std::size_t{22} << 20U;
    for (const auto n : {std::size_t{0}, std::size_t{1}, cap - 1U, cap, cap + 1U}) {
        std::vector<std::byte> source(n), received(n);
        for (std::size_t i = 0; i < n; ++i) source[i] = static_cast<std::byte>(i % 251U);
        std::size_t sent = 0, calls = 0;
        pair::transfer_embed_frame_bytes(source, {}, cap, [&](auto out, auto in) {
            CHECK(!out.empty() && out.size() <= cap && out.size() == in.size());
            CHECK(std::equal(out.begin(), out.end(), source.begin() + static_cast<std::ptrdiff_t>(sent)));
            sent += out.size();
            ++calls;
            std::fill(in.begin(), in.end(), std::byte{0xff});
        });
        CHECK(sent == n && calls == (n + cap - 1U) / cap);
        std::size_t at = 0;
        pair::transfer_embed_frame_bytes({}, received, cap, [&](auto out, auto in) {
            CHECK(!out.empty() && out.size() <= cap && out.size() == in.size());
            CHECK(std::all_of(out.begin(), out.end(), [](auto b) { return b == std::byte{0}; }));
            std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(at), in.size(), in.begin());
            at += in.size();
        });
        CHECK(at == n && received == source);
    }
    std::array<std::byte, 1> byte{};
    const auto no_exchange = [&](auto, auto) { CHECK(false); };
    CHECK(throws([&] { pair::transfer_embed_frame_bytes(byte, {}, 0U, no_exchange); }));
    CHECK(throws([&] { pair::transfer_embed_frame_bytes(byte, byte, cap, no_exchange); }));

    session::EmbedInterner sender;
    // the sender burns a few handles first, so its handles differ from the receiver's
    auto warm = sender.intern(span(9, 2, 4));
    auto a = sender.intern(span(1, 5, 4)), b = sender.intern(span(2, 3, 4));
    auto table = std::make_shared<const session::EmbedTable>(std::vector<session::EmbedLease>{a, b});
    std::vector<std::uint32_t> prompt = {10, 11};
    for (const auto id : a.ids()) prompt.push_back(id);
    prompt.push_back(12);
    for (const auto id : b.ids()) prompt.push_back(id);

    pair::EmbedFrameLedger peer(2);
    pair::EmbedFrameReceiver rx(2);
    // text only: no frame
    const std::vector<std::uint32_t> text = {1, 2, 3};
    CHECK(pair::encode_embed_frame(*table, text, peer, crc).empty());

    const auto f1 = pair::encode_embed_frame(*table, prompt, peer, crc);
    const auto r1 = rx.receive(f1, prompt, crc);
    CHECK(r1.spans == 2U && r1.sent == 2U);
    CHECK(r1.ids.size() == prompt.size());
    // same rows at the same positions; text ids untouched
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        CHECK(session::is_embed_id(prompt[i]) == session::is_embed_id(r1.ids[i]));
        if (!session::is_embed_id(prompt[i])) { CHECK(prompt[i] == r1.ids[i]); continue; }
        const auto* s = table->find(session::embed_handle(prompt[i]));
        const auto* d = r1.table->find(session::embed_handle(r1.ids[i]));
        CHECK(s && d && session::embed_row(prompt[i]) == session::embed_row(r1.ids[i]));
        if (s && d) {
            const auto row = session::embed_row(prompt[i]);
            CHECK(std::memcmp(s->row(row).data(), d->row(row).data(), s->width * 2U) == 0);
            CHECK(s->real_token_id == d->real_token_id && s->role[row] == d->role[row]);
        }
    }
    // the same prompt again: only the manifest travels
    const auto f2 = pair::encode_embed_frame(*table, prompt, peer, crc);
    CHECK(f2.size() == pair::kEmbedFrameHeaderBytes + 2U * pair::kEmbedFrameEntryBytes);
    const auto r2 = rx.receive(f2, prompt, crc);
    CHECK(r2.sent == 0U && r2.ids == r1.ids);
    // a third span evicts the oldest on both ends (capacity 2): a later frame naming it carries its rows again
    auto c = sender.intern(span(3, 2, 4));
    auto table3 = std::make_shared<const session::EmbedTable>(std::vector<session::EmbedLease>{c});
    const auto pc = c.ids();
    CHECK(rx.receive(pair::encode_embed_frame(*table3, pc, peer, crc), pc, crc).sent == 1U);
    CHECK(rx.held() == 2U);
    const auto f4 = pair::encode_embed_frame(*table, prompt, peer, crc);
    const auto r4 = rx.receive(f4, prompt, crc);
    // a was evicted by c; touching a evicts b before b's entry is written, so b travels again too (both ends agree)
    CHECK(r4.sent == 2U && rx.held() == 2U);
    // corruption: a payload byte, the manifest, truncation, a span the receiver does not hold
    pair::EmbedFrameLedger fresh(2);
    auto bad = pair::encode_embed_frame(*table, prompt, fresh, crc);
    bad.back() = std::byte{static_cast<std::uint8_t>(static_cast<std::uint8_t>(bad.back()) ^ 1U)};
    pair::EmbedFrameReceiver rx2(2);
    CHECK(throws([&] { (void)rx2.receive(bad, prompt, crc); }));
    auto bad2 = f1;
    bad2[pair::kEmbedFrameHeaderBytes + 4U] = std::byte{7};
    pair::EmbedFrameReceiver rx_manifest(2), rx_short(2);  // frame errors are terminal; never retry rx2
    CHECK(throws([&] { (void)rx_manifest.receive(bad2, prompt, crc); }));
    CHECK(throws([&] { (void)rx_short.receive(std::span<const std::byte>(f1.data(), f1.size() - 1U), prompt, crc); }));
    pair::EmbedFrameReceiver rx3(2);
    CHECK(throws([&] { (void)rx3.receive(f2, prompt, crc); }));  // manifest only, nothing held
    // deep rows are refused by version 1
    auto deep = span(4, 2, 4);
    deep.deep.push_back({3U, deep.data});
    auto dl = sender.intern(std::move(deep));
    const session::EmbedTable dt(std::vector<session::EmbedLease>{dl});
    pair::EmbedFrameLedger dpeer;
    const auto dids = dl.ids();
    CHECK(throws([&] { (void)pair::encode_embed_frame(dt, dids, dpeer, crc); }));

    {  // byte budget: both ledgers drop the oldest spans past it (same rule, from the manifest), so a span that fell out
       // travels again and the receiver never holds more than the budget; a span above the budget is held alone
        session::EmbedInterner bi;
        const auto ba = bi.intern(span(11, 4, 8)), bb = bi.intern(span(12, 4, 8)), bc = bi.intern(span(13, 4, 8));  // 64 bytes each
        const session::EmbedTable ball(std::vector<session::EmbedLease>{ba, bb, bc});
        std::vector<std::uint32_t> bids;
        for (const auto* l : {&ba, &bb, &bc}) {
            const auto one = l->ids();
            bids.insert(bids.end(), one.begin(), one.end());
        }
        pair::EmbedFrameLedger btx(64, 150);
        pair::EmbedFrameReceiver brx(64, 150);
        const auto bf1 = pair::encode_embed_frame(ball, bids, btx, crc);
        CHECK(btx.size() == 2 && btx.bytes() == 128 && !btx.holds(ba.rows().digest) && btx.holds(bc.rows().digest));
        CHECK(brx.receive(bf1, bids, crc).sent == 3 && brx.held() == 2);
        const session::EmbedTable only_a(std::vector<session::EmbedLease>{ba});
        const auto ids_a = ba.ids();
        const auto bf2 = pair::encode_embed_frame(only_a, ids_a, btx, crc);  // a fell out: its rows travel again
        CHECK(brx.receive(bf2, ids_a, crc).sent == 1 && brx.held() == 2 && btx.bytes() == 128);
        const session::EmbedTable only_c(std::vector<session::EmbedLease>{bc});
        const auto ids_c = bc.ids();
        const auto bf3 = pair::encode_embed_frame(only_c, ids_c, btx, crc);  // c is still held: manifest only
        CHECK(brx.receive(bf3, ids_c, crc).sent == 0);
        pair::EmbedFrameLedger tiny(64, 10);
        std::vector<session::EmbedDigest> gone;
        tiny.touch(ba.rows().digest, 64, &gone);
        tiny.touch(bb.rows().digest, 64, &gone);
        CHECK(tiny.size() == 1 && tiny.bytes() == 64 && gone.size() == 1 && gone[0] == ba.rows().digest);
        CHECK(throws([] { pair::EmbedFrameLedger zero_budget(4, 0); (void)zero_budget; }));
    }

    // regime rule: an empty table (no receipts) is always rank0
    using M = pair::EmbedEncodeMode;
    CHECK(pair::select_embed_encode_mode({4U, 20000U}, {}) == M::rank0);
    const pair::EmbedRegimeRow rows[] = {{M::peer, 0.1, 500.0}, {M::split, 0.0, 0.0}};
    CHECK(pair::select_embed_encode_mode({1U, 100U}, rows) == M::rank0);   // 10 ms of prefix < 500 ms encode
    CHECK(pair::select_embed_encode_mode({1U, 8000U}, rows) == M::peer);   // 800 ms >= 500 ms
    CHECK(pair::select_embed_encode_mode({2U, 0U}, rows) == M::split);
    const pair::EmbedRegimeRow peer_only[] = {{M::peer, 0.1, 500.0}};
    CHECK(pair::select_embed_encode_mode({3U, 100U}, peer_only) == M::rank0);

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("pair-embed-frame: ok");
    return 0;
}
