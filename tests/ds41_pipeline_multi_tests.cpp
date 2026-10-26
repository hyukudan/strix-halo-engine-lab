#include "strix/ds41_pipeline_multi.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include "deepseek41_real_config.inc"

#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <thread>

// strix/ds41-pipeline-multi-v1: MULTI_HELLO / SEQ_STEP / SEQ_IMPORT round trips and rejections, and a loopback session
// where the TAIL sends the requests of several slots ahead (two groups in flight) and a FRONT with one sequence per
// slot answers each in order with that slot's boundary: every boundary arrives byte-identical, at its slot's position.
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

// A deterministic boundary of `rows` rows of slot `slot` at `position` (a FRONT's answer; the TAIL recomputes it to check).
pipe::Boundary boundary_of(const pipe::BoundaryPlan& plan, const pipe::BoundaryGeometry& g, std::uint32_t slot, std::uint32_t position,
                           const std::vector<std::uint32_t>& tokens) {
    std::mt19937 rng(slot * 1000003U + position);
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    pipe::Boundary b;
    b.position = position;
    b.rows = rows;
    b.tokens = tokens;
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
        b.sections.push_back({pipe::SectionKind::candidates, static_cast<std::uint32_t>(plan.candidate_source), stride, rows, random_bytes(std::size_t{rows} * stride)});
    }
    return b;
}

}  // namespace

int main() try {
    const auto config = strix::deepseek41::parse_config_json(real_config_json);
    const auto attrs = strix::deepseek41::layer_attributes(config);
    const auto g = pipe::boundary_geometry(config);
    const std::uint32_t K = 22;
    const auto plan = pipe::plan_boundary(attrs, K);

    // ---- round trips and rejections ----
    {
        pipe::MultiHello mh;
        for (std::size_t i = 0; i < 32; ++i) mh.hello.token[i] = static_cast<std::uint8_t>(i * 7U);
        mh.hello.cut = K;
        mh.hello.layers = 40;
        mh.hello.hidden = g.hidden;
        mh.hello.hc = g.hc;
        mh.hello.vocab = g.vocab;
        mh.hello.max_positions = 2048;
        mh.hello.model_identity = 0x1234567890ABCDEFULL;
        mh.hello.backend = "cuda";
        mh.hello.numeric_identity = "ds41-expert-tp2-i1152-v1";
        mh.slots = 8;
        mh.max_rows = 8;
        const auto back = pipe::decode_multi_hello(pipe::encode(mh));
        expect(back.hello.token == mh.hello.token && back.hello.cut == K && back.hello.model_identity == mh.hello.model_identity &&
                   back.hello.numeric_identity == mh.hello.numeric_identity && back.slots == 8U && back.max_rows == 8U,
               "MULTI_HELLO round trip");
        auto bad = pipe::encode(mh);
        bad.push_back(0);
        expect_throw([&] { (void)pipe::decode_multi_hello(bad); }, "MULTI_HELLO trailing byte", "trailing");
        mh.slots = 0;
        expect_throw([&] { (void)pipe::decode_multi_hello(pipe::encode(mh)); }, "MULTI_HELLO without slots", "no slots");

        pipe::SeqStep q{3, 517, 2, {11, 12, 13, 14}};
        const auto qb = pipe::decode_seq_step(pipe::encode(q), 8);
        expect(qb.slot == 3U && qb.position == 517U && qb.commit == 2U && qb.tokens == q.tokens, "SEQ_STEP round trip");
        expect_throw([&] { (void)pipe::decode_seq_step(pipe::encode(q), 2); }, "SEQ_STEP above max rows", "rows");
        pipe::SeqStep empty{1, 0, 0, {}};
        expect_throw([&] { (void)pipe::decode_seq_step(pipe::encode(empty), 8); }, "SEQ_STEP without rows", "rows");
        auto cut = pipe::encode(q);
        cut.pop_back();
        expect_throw([&] { (void)pipe::decode_seq_step(cut, 8); }, "SEQ_STEP truncated", "truncated");

        pipe::StateMsg sm{"{\"layers\":[0,40],\"position\":9}", {1, 2, 3, 4, 5}};
        const auto [slot, sb] = pipe::decode_seq_import(pipe::encode_seq_import(6, sm));
        expect(slot == 6U && sb.manifest == sm.manifest && sb.state == sm.state, "SEQ_IMPORT round trip");

        pipe::Header h;
        h.type = pipe::MsgType::seq_boundary;
        h.request = 77;
        h.bytes = 10;
        h.flags = 5;
        const auto hb = pipe::decode_header(pipe::encode_header(h));
        expect(hb.type == pipe::MsgType::seq_boundary && hb.flags == 5U, "SEQ_BOUNDARY header (slot in the flags)");
        expect(std::string(pipe::to_string(pipe::MsgType::seq_import)) == "SEQ_IMPORT", "message names");
    }

    // ---- loopback session: two groups of slots in flight, FIFO answers ----
    {
        pipe::Listener listener("127.0.0.1", 0);
        const auto port = listener.port();
        const std::uint32_t slots = 4, steps = 12;
        std::thread front([&] {  // a FRONT with one sequence per slot (positions only; boundaries from boundary_of)
            try {
                auto c = pipe::connect("127.0.0.1:" + std::to_string(port), std::chrono::seconds(10), std::chrono::seconds(10));
                pipe::MultiHello mh;
                mh.hello.cut = K;
                mh.hello.layers = 40;
                mh.slots = slots;
                mh.max_rows = 8;
                c->send(pipe::MsgType::multi_hello, 1, pipe::encode(mh));
                const auto ack = c->receive();
                expect(ack.header.type == pipe::MsgType::hello_ack, "FRONT: HELLO_ACK");
                std::vector<std::uint32_t> pos(slots, 0U), last(slots, 0U);
                for (;;) {
                    const auto m = c->receive();
                    if (m.header.type == pipe::MsgType::close) {
                        c->send(pipe::MsgType::ack, m.header.request, {});
                        return;
                    }
                    if (m.header.type == pipe::MsgType::seq_reset) {
                        const auto s = pipe::decode_u32(m.payload);
                        pos.at(s) = 0;
                        last.at(s) = 0;
                        c->send(pipe::MsgType::ack, m.header.request, pipe::encode_u32(s));
                        continue;
                    }
                    const auto q = pipe::decode_seq_step(m.payload, 8);
                    if (q.commit != 0U) pos.at(q.slot) -= last.at(q.slot) - q.commit;
                    if (q.position != pos.at(q.slot)) {
                        c->send(pipe::MsgType::error, m.header.request, pipe::encode_text("position"));
                        return;
                    }
                    pipe::StepMsg sm;
                    sm.boundary = boundary_of(plan, g, q.slot, q.position, q.tokens);
                    c->send(pipe::MsgType::seq_boundary, m.header.request, pipe::encode(sm), q.slot);
                    pos[q.slot] += static_cast<std::uint32_t>(q.tokens.size());
                    last[q.slot] = static_cast<std::uint32_t>(q.tokens.size());
                }
            } catch (const std::exception& e) {
                expect(false, std::string("FRONT thread: ") + e.what());
            }
        });
        auto conn = listener.accept(std::chrono::seconds(10), std::chrono::seconds(10));
        const auto first = conn->receive();
        const auto mh = pipe::decode_multi_hello(first.payload);
        expect(mh.slots == slots, "TAIL: the FRONT's slots");
        pipe::HelloAck ack;
        conn->send(pipe::MsgType::hello_ack, first.header.request, pipe::encode(ack));
        // two groups {0, 1}, {2, 3}: group A's next step is sent before group B's answers are read
        struct Want {
            std::uint64_t request;
            std::uint32_t slot, position;
            std::vector<std::uint32_t> tokens;
            bool reset;
        };
        std::deque<Want> pending;
        std::uint64_t req = 2;
        std::vector<std::uint32_t> pos(slots, 0U);
        std::mt19937 rng(7);
        const auto send_reset = [&](std::uint32_t s) {
            conn->send(pipe::MsgType::seq_reset, req, pipe::encode_u32(s));
            pending.push_back({req++, s, 0, {}, true});
            pos[s] = 0;
        };
        const auto send_group = [&](std::uint32_t g0, std::uint32_t kept_prev) {
            for (std::uint32_t s = g0; s < g0 + 2U; ++s) {
                const std::uint32_t rows = 1U + static_cast<std::uint32_t>(rng() % 4U);
                pipe::SeqStep q;
                q.slot = s;
                q.position = pos[s];
                q.commit = kept_prev;
                for (std::uint32_t r = 0; r < rows; ++r) q.tokens.push_back(static_cast<std::uint32_t>(rng() % g.vocab));
                conn->send(pipe::MsgType::seq_step, req, pipe::encode(q));
                pending.push_back({req++, s, q.position, q.tokens, false});
            }
        };
        std::uint64_t checked = 0, bytes = 0;
        std::map<std::uint32_t, std::uint32_t> rows_of;  // the rows of each slot's last step (kept whole: commit 0)
        const auto receive_group = [&] {
            for (int k = 0; k < 2;) {
                const auto want = pending.front();
                pending.pop_front();
                const auto m = conn->receive();
                expect(m.header.request == want.request, "answers in request order");
                if (want.reset) {
                    expect(m.header.type == pipe::MsgType::ack, "ACK of a reset");
                    continue;
                }
                expect(m.header.type == pipe::MsgType::seq_boundary && m.header.flags == want.slot, "SEQ_BOUNDARY of the slot");
                const auto sm = pipe::decode_step(m.payload, g, 8, true);
                const auto ref = boundary_of(plan, g, want.slot, want.position, want.tokens);
                pipe::validate_boundary(sm.boundary, plan, g);
                bool same = sm.boundary.position == ref.position && sm.boundary.tokens == ref.tokens &&
                            std::memcmp(sm.boundary.stream_data(), ref.stream.data(), ref.stream.size() * 4U) == 0 &&
                            sm.boundary.sections.size() == ref.sections.size();
                for (std::size_t i = 0; same && i < ref.sections.size(); ++i) same = sm.boundary.sections[i].bytes == ref.sections[i].bytes;
                expect(same, "boundary byte-identical");
                bytes += m.payload.size();
                pos[want.slot] += static_cast<std::uint32_t>(want.tokens.size());
                ++checked;
                ++k;
            }
        };
        for (std::uint32_t s = 0; s < slots; ++s) send_reset(s);
        send_group(0, 0);
        send_group(2, 0);
        for (std::uint32_t t = 0; t < steps; ++t) {
            receive_group();  // group A
            send_group(0, 0);  // A's next rows go out before B is read (A's positions moved by its answers)
            receive_group();  // group B
            send_group(2, 0);
        }
        receive_group();
        receive_group();
        conn->send(pipe::MsgType::close, req++, {});
        expect(conn->receive().header.type == pipe::MsgType::ack, "CLOSE acknowledged");
        front.join();
        expect(checked == 2U * 2U * (steps + 1U), "every boundary checked");
        std::printf("loopback multi session: %llu boundaries of 4 slots, 2 groups in flight, %.1f KB, all byte-identical\n",
                    static_cast<unsigned long long>(checked), static_cast<double>(bytes) / 1e3);
    }
    if (failures) {
        std::fprintf(stderr, "%d FAILED\n", failures);
        return 1;
    }
    std::printf("strix-ds41-pipeline-multi-tests: PASS\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
