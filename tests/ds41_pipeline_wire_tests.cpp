#include "strix/ds41_pipeline_front.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include "deepseek41_real_config.inc"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <thread>

#include <unistd.h>

// strix/ds41-pipeline-v1: the cut of the released model (which side sections cross at each K), encode/decode of every
// message bitwise, rejection of malformed input (magic, version, header check, truncation, trailing bytes, sizes, a
// boundary that does not match its cut), the session token file and a loopback TCP exchange.
namespace {

namespace pipe = strix::ds41::pipe;
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}
void expect_throw(const std::function<void()>& f, const std::string& what, const std::string& needle = "") {
    try {
        f();
        expect(false, what + " (no exception)");
    } catch (const std::exception& e) {
        expect(needle.empty() || std::string(e.what()).find(needle) != std::string::npos, what + " (wrong reason: " + e.what() + ")");
    }
}

std::vector<std::uint32_t> layers_of(const std::vector<pipe::SourceLayer>& v) {
    std::vector<std::uint32_t> out;
    for (const auto& s : v) out.push_back(s.layer);
    return out;
}

// A boundary for `plan` with random contents, as a FRONT would produce it.
pipe::Boundary make_boundary(const pipe::BoundaryPlan& plan, const pipe::BoundaryGeometry& g, std::uint32_t position, std::uint32_t rows,
                             std::mt19937& rng) {
    pipe::Boundary b;
    b.position = position;
    b.rows = rows;
    for (std::uint32_t r = 0; r < rows; ++r) b.tokens.push_back(static_cast<std::uint32_t>(rng() % g.vocab));
    std::uniform_real_distribution<float> u(-4.0F, 4.0F);
    b.stream.resize(std::size_t{rows} * g.hc * g.hidden);
    for (auto& f : b.stream) f = u(rng);
    const auto random_bytes = [&](std::size_t n) {
        std::vector<std::uint8_t> v(n);
        for (auto& x : v) x = static_cast<std::uint8_t>(rng());
        return v;
    };
    b.sections.push_back({pipe::SectionKind::mix_ffn, plan.cut - 1U, 0U, rows, random_bytes(std::size_t{rows} * g.mix_floats() * 4U)});
    for (const auto& s : plan.kv_sources) {
        const std::uint32_t first = position / s.ratio, end = (position + rows) / s.ratio;
        b.sections.push_back({pipe::SectionKind::kv_entries, s.layer, first, end - first,
                              random_bytes(std::size_t{end - first} * (g.kv_entry_bytes + g.idx_entry_bytes))});
    }
    for (const auto& s : plan.topk_sources) {
        std::vector<std::uint8_t> v;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::uint32_t n = static_cast<std::uint32_t>(rng() % (g.topk + 1U));
            const auto put = [&](std::uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i))); };
            put(n);
            for (std::uint32_t i = 0; i < n; ++i) put(i * 3U);
        }
        b.sections.push_back({pipe::SectionKind::topk, s.layer, 0U, rows, v});
    }
    if (plan.candidate_source >= 0) {
        const std::uint32_t count = (position + rows) / plan.candidate_ratio;
        const std::uint32_t stride = (count + g.candidate_block - 1U) / g.candidate_block;
        b.sections.push_back({pipe::SectionKind::candidates, static_cast<std::uint32_t>(plan.candidate_source), stride, rows,
                              random_bytes(std::size_t{rows} * stride)});
    }
    return b;
}

bool same(const pipe::Boundary& a, const pipe::Boundary& b) {
    if (a.position != b.position || a.rows != b.rows || a.tokens != b.tokens || a.stream.size() != b.stream.size()) return false;
    if (std::memcmp(a.stream.data(), b.stream.data(), a.stream.size() * 4U) != 0) return false;
    if (a.sections.size() != b.sections.size()) return false;
    for (std::size_t i = 0; i < a.sections.size(); ++i) {
        const auto &x = a.sections[i], &y = b.sections[i];
        if (x.kind != y.kind || x.layer != y.layer || x.first != y.first || x.count != y.count || x.bytes != y.bytes) return false;
    }
    return true;
}

}  // namespace

int main() try {
    const auto config = strix::deepseek41::parse_config_json(real_config_json);
    const auto attrs = strix::deepseek41::layer_attributes(config);
    const auto g = pipe::boundary_geometry(config);
    expect(g.stream_bytes_per_row() == 81920U, "stream 4 x 5120 FP32 per row");
    expect(g.kv_entry_bytes == 288U && g.idx_entry_bytes == 68U, "compressed entry sizes");
    expect(g.mix_floats() == 24U, "mix floats");

    // ---- the cut of the released model ----
    {
        const auto p1 = pipe::plan_boundary(attrs, 1);  // layers 1..: computes every source itself
        expect(p1.kv_sources.empty() && p1.topk_sources.empty() && p1.candidate_source < 0, "cut 1 carries no source");
        const auto p3 = pipe::plan_boundary(attrs, 3);  // layers 3..7 read layer 2's KV and top-k
        expect(layers_of(p3.kv_sources) == std::vector<std::uint32_t>{2} && layers_of(p3.topk_sources) == std::vector<std::uint32_t>{2} &&
                   p3.kv_sources[0].ratio == 2U,
               "cut 3: kv@2 (ratio 2) and topk@2");
        const auto p8 = pipe::plan_boundary(attrs, 8);
        expect(p8.kv_sources.empty() && p8.topk_sources.empty(), "cut 8: layer 8 is a new source");
        const auto p20 = pipe::plan_boundary(attrs, 20);
        expect(p20.kv_sources.empty() && p20.topk_sources.empty() && p20.candidate_source < 0, "cut 20: the TAIL computes layer 20");
        const auto p21 = pipe::plan_boundary(attrs, 21);
        expect(layers_of(p21.kv_sources) == std::vector<std::uint32_t>{20} && layers_of(p21.topk_sources) == std::vector<std::uint32_t>{20} &&
                   p21.candidate_source == 20 && p21.candidate_ratio == 1U,
               "cut 21: kv@20, topk@20, cand@20");
        const auto p23 = pipe::plan_boundary(attrs, 23);
        expect(layers_of(p23.kv_sources) == std::vector<std::uint32_t>{20} && layers_of(p23.topk_sources) == std::vector<std::uint32_t>{20} &&
                   p23.candidate_source == 20,
               "cut 23 (the default pipeline): kv@20, topk@20 (layer 23), cand@20 (users 24..36)");
        const auto p25 = pipe::plan_boundary(attrs, 25);
        expect(layers_of(p25.kv_sources) == std::vector<std::uint32_t>{20} && layers_of(p25.topk_sources) == std::vector<std::uint32_t>{24} &&
                   p25.candidate_source == 20,
               "cut 25: kv@20, topk@24, cand@20");
        const auto p37 = pipe::plan_boundary(attrs, 37);
        expect(layers_of(p37.topk_sources) == std::vector<std::uint32_t>{36} && p37.candidate_source < 0, "cut 37: topk@36, no candidate user left");
        expect_throw([&] { (void)pipe::plan_boundary(attrs, 0); }, "cut 0 refused");
        expect_throw([&] { (void)pipe::plan_boundary(attrs, 40); }, "cut 40 refused");
        const auto b = pipe::step_bytes(p23, g, 8191, 1);
        expect(b.stream == 81920U && b.mix == 96U && b.kv == 356U && b.topk_max == 2052U && b.candidates == 1024U, "cut 23 bytes of one row at 8K");
        std::printf("cut 23, one row at position 8191: stream %llu + side %llu (mix %llu, kv %llu, top-k <= %llu, candidates %llu) + framing %llu = %llu B\n",
                    static_cast<unsigned long long>(b.stream), static_cast<unsigned long long>(b.side()), static_cast<unsigned long long>(b.mix),
                    static_cast<unsigned long long>(b.kv), static_cast<unsigned long long>(b.topk_max), static_cast<unsigned long long>(b.candidates),
                    static_cast<unsigned long long>(b.framing), static_cast<unsigned long long>(b.total()));
    }

    // ---- messages round trip ----
    std::mt19937 rng(41);
    for (const std::uint32_t cut : {1U, 3U, 20U, 21U, 23U, 37U})
        for (const std::uint32_t rows : {1U, 3U, 8U})
            for (const std::uint32_t position : {0U, 1U, 7U, 4096U}) {
                const auto plan = pipe::plan_boundary(attrs, cut);
                pipe::StepMsg m;
                m.flags = pipe::kStepWantLogits;
                m.boundary = make_boundary(plan, g, position, rows, rng);
                pipe::validate_boundary(m.boundary, plan, g);
                const auto bytes = pipe::encode(m);
                const auto back = pipe::decode_step(bytes, g, 8);
                const std::string at = "cut " + std::to_string(cut) + " R " + std::to_string(rows) + " p " + std::to_string(position);
                expect(back.flags == m.flags && same(back.boundary, m.boundary), "step round trip " + at);
                pipe::validate_boundary(back.boundary, plan, g);
                // rejections
                expect_throw([&] { (void)pipe::decode_step(std::span(bytes).first(bytes.size() - 1U), g, 8); }, "truncated step " + at, "truncated");
                auto longer = bytes;
                longer.push_back(0);
                expect_throw([&] { (void)pipe::decode_step(longer, g, 8); }, "trailing byte " + at, "trailing");
                if (rows > 1U) expect_throw([&] { (void)pipe::decode_step(bytes, g, rows - 1U); }, "too many rows " + at, "row count");
                auto bad = m.boundary;
                bad.sections.pop_back();
                expect_throw([&] { pipe::validate_boundary(bad, plan, g); }, "missing section " + at, "sections");
                if (!plan.kv_sources.empty() && rows >= 2U) {
                    auto shifted = m.boundary;
                    for (auto& s : shifted.sections)
                        if (s.kind == pipe::SectionKind::kv_entries) ++s.first;
                    expect_throw([&] { pipe::validate_boundary(shifted, plan, g); }, "kv entries out of range " + at, "entries");
                }
                auto small = m.boundary;
                small.stream.pop_back();
                expect_throw([&] { pipe::validate_boundary(small, plan, g); }, "stream size " + at, "stream");
            }
    {
        pipe::Result r;
        r.position = 17;
        r.tokens = {5, 129279};
        r.logits_fnv = {0x0123456789abcdefULL, 0};
        const auto back = pipe::decode_result(pipe::encode(r), g.vocab, 8);
        expect(back.position == 17U && back.tokens == r.tokens && back.logits_fnv == r.logits_fnv && back.logits.empty(), "result round trip");
        pipe::Result one;
        one.tokens = {7};
        one.logits_fnv = {1};
        one.logits.assign(g.vocab, 0.5F);
        expect(pipe::decode_result(pipe::encode(one), g.vocab, 8).logits.size() == g.vocab, "result with logits");
        r.tokens[1] = g.vocab;
        expect_throw([&] { (void)pipe::decode_result(pipe::encode(r), g.vocab, 8); }, "result token outside the vocabulary", "token");
    }
    {
        pipe::Hello h;
        for (std::size_t i = 0; i < 32U; ++i) h.token[i] = static_cast<std::uint8_t>(i * 7U);
        h.cut = 23;
        h.layers = 40;
        h.hidden = 5120;
        h.hc = 4;
        h.vocab = 129280;
        h.max_positions = 65536;
        h.model_identity = 0xfeedULL;
        h.backend = "hip";
        h.numeric_identity = "ds41-exact-v1";
        const auto back = pipe::decode_hello(pipe::encode(h));
        expect(back.token == h.token && back.cut == 23U && back.model_identity == 0xfeedULL && back.backend == "hip" &&
                   back.numeric_identity == "ds41-exact-v1" && back.max_positions == 65536U,
               "hello round trip");
        pipe::HelloAck a;
        a.status = 1;
        a.first_layer = 23;
        a.end_layer = 40;
        a.world = 2;
        a.message = "bad token";
        const auto ab = pipe::decode_hello_ack(pipe::encode(a));
        expect(ab.status == 1U && ab.first_layer == 23U && ab.world == 2U && ab.message == "bad token", "hello ack round trip");
        pipe::StateMsg s;
        s.manifest = R"({"schema":"strix/ds41-state-manifest-v1"})";
        s.state = {1, 2, 3, 4, 5};
        const auto sb = pipe::decode_state(pipe::encode(s));
        expect(sb.manifest == s.manifest && sb.state == s.state, "state message round trip");
        expect(pipe::decode_u32(pipe::encode_u32(5)) == 5U && pipe::decode_text(pipe::encode_text("x")) == "x", "small payloads");
    }
    // ---- header ----
    {
        pipe::Header h;
        h.type = pipe::MsgType::step;
        h.request = 99;
        h.bytes = 12345;
        h.flags = 3;
        auto b = pipe::encode_header(h);
        const auto back = pipe::decode_header(b);
        expect(back.type == pipe::MsgType::step && back.request == 99U && back.bytes == 12345U && back.flags == 3U, "header round trip");
        auto bad = b;
        bad[0] ^= 1U;
        expect_throw([&] { (void)pipe::decode_header(bad); }, "bad magic", "magic");
        bad = b;
        bad[4] = 2;
        expect_throw([&] { (void)pipe::decode_header(bad); }, "other version", "version");
        bad = b;
        bad[10] ^= 1U;
        expect_throw([&] { (void)pipe::decode_header(bad); }, "corrupt header", "check");
        expect_throw([&] { (void)pipe::decode_header(b, 100); }, "payload above the bound", "bound");
    }
    // ---- token file ----
    const auto dir = std::filesystem::temp_directory_path() / ("ds41-pipe-test-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    {
        std::ofstream(dir / "tok") << "00112233445566778899aabbccddeeff\n00112233445566778899AABBCCDDEEFF\n";
        const auto t = pipe::read_token_file(dir / "tok");
        expect(t[0] == 0x00U && t[15] == 0xffU && t[31] == 0xffU, "token file");
        auto other = t;
        other[31] ^= 1U;
        expect(pipe::same_token(t, t) && !pipe::same_token(t, other), "token comparison");
        std::ofstream(dir / "short") << "0011";
        expect_throw([&] { (void)pipe::read_token_file(dir / "short"); }, "short token", "64 hex");
        std::ofstream(dir / "nothex") << std::string(64, 'g');
        expect_throw([&] { (void)pipe::read_token_file(dir / "nothex"); }, "non-hex token", "hexadecimal");
    }
    std::filesystem::remove_all(dir);
    // ---- loopback TCP ----
    {
        pipe::Listener listener("127.0.0.1", 0);
        const auto port = listener.port();
        const auto plan = pipe::plan_boundary(attrs, 23);
        pipe::StepMsg m;
        m.boundary = make_boundary(plan, g, 100, 8, rng);
        const auto payload = pipe::encode(m);
        std::thread server([&] {
            auto c = listener.accept(std::chrono::milliseconds(10000), std::chrono::milliseconds(10000));
            if (!c) return;
            for (int i = 0; i < 3; ++i) {
                const auto msg = c->receive();
                c->send(pipe::MsgType::ack, msg.header.request, msg.payload);  // echo
            }
        });
        auto c = pipe::connect("127.0.0.1:" + std::to_string(port), std::chrono::milliseconds(5000), std::chrono::milliseconds(10000));
        for (std::uint64_t i = 0; i < 3U; ++i) {
            c->send(pipe::MsgType::step, 40 + i, payload);
            const auto back = c->receive();
            expect(back.header.type == pipe::MsgType::ack && back.header.request == 40 + i && back.payload == payload, "loopback echo");
        }
        server.join();
        expect(c->bytes_sent() == 3U * (pipe::kHeaderBytes + payload.size()), "bytes sent");
        std::printf("loopback: 3 steps of 8 rows (%zu B each) echoed\n", payload.size());
        expect_throw([&] { (void)c->receive(); }, "peer closed", "closed");
    }
    // ---- the FRONT's client against a fake TAIL: the session token, a step, an ERROR answer ----
    {
        pipe::Listener listener("127.0.0.1", 0);
        std::array<std::uint8_t, 32> secret{};
        for (std::size_t i = 0; i < 32U; ++i) secret[i] = static_cast<std::uint8_t>(200U - i);
        std::thread tail([&] {
            for (int session = 0; session < 2; ++session) {
                auto c = listener.accept(std::chrono::milliseconds(10000), std::chrono::milliseconds(10000));
                if (!c) return;
                const auto hello = c->receive();
                const auto h = pipe::decode_hello(hello.payload);
                pipe::HelloAck ack;
                ack.first_layer = h.cut;
                ack.end_layer = h.layers;
                if (!pipe::same_token(h.token, secret)) ack.status = 1, ack.message = "wrong session token";
                c->send(pipe::MsgType::hello_ack, hello.header.request, pipe::encode(ack));
                if (ack.status != 0U) continue;
                const auto step = c->receive();
                const auto sm = pipe::decode_step(step.payload, g, 8);
                pipe::Result r;
                r.position = sm.boundary.position;
                r.tokens.assign(sm.boundary.rows, 42U);
                r.logits_fnv.assign(sm.boundary.rows, 7U);
                c->send(pipe::MsgType::result, step.header.request, pipe::encode(r));
                const auto bad = c->receive();
                c->send(pipe::MsgType::error, bad.header.request, pipe::encode_text("step at position 9, the TAIL is at 1"));
            }
        });
        pipe::Hello h;
        h.cut = 23;
        h.layers = 40;
        h.vocab = g.vocab;
        pipe::ClientOptions co;
        co.connect = "127.0.0.1:" + std::to_string(listener.port());
        co.connect_wait = std::chrono::milliseconds(5000);
        co.io_timeout = std::chrono::milliseconds(10000);
        co.token = secret;
        co.token[0] ^= 1U;
        expect_throw([&] { pipe::StageClient refused(co, h); }, "a wrong token is refused", "wrong session token");
        co.token = secret;
        pipe::StageClient client(co, h);
        expect(client.ack().first_layer == 23U, "hello accepted");
        const auto plan = pipe::plan_boundary(attrs, 23);
        const auto r = client.step(make_boundary(plan, g, 0, 3, rng));
        expect(r.tokens == std::vector<std::uint32_t>{42U, 42U, 42U} && r.logits_fnv[0] == 7U, "step through the client");
        expect_throw([&] { (void)client.step(make_boundary(plan, g, 9, 1, rng)); }, "an ERROR answer throws", "the TAIL is at 1");
        tail.join();
    }
    expect_throw([] { (void)pipe::split_host_port("nohost"); }, "host:port parse");
    expect(pipe::split_host_port("192.0.2.1:7000") == std::pair<std::string, std::uint16_t>{"192.0.2.1", 7000}, "host:port");
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("ds41 pipeline wire tests pass\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
