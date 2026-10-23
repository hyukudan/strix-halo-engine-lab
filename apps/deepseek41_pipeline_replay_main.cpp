#include "strix/ds41_pipeline_wire.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// strix-deepseek41-pipeline-replay: a FRONT without layers. It sends the STEP payloads of a record file (written by
// strix-deepseek41-rank --role pipeline-record) to a pipeline TAIL in order, one at a time, and prints each RESULT
// with the round trip seen here (send start .. RESULT received). The TAIL's compute per token and its exactness are
// measured without a FRONT's compute in the loop.
//   strix-deepseek41-pipeline-replay --record FILE --stage-connect HOST:PORT --stage-token-file PATH [--steps N]
//       [--timeline OUT.csv] [--busy-poll] [--tail-layers N] [--close] [--refuse-at I,..] [--lose-at I,..]
// Fault injection (tests of a TAIL's handling of refused and lost STEPs; each ends the session, the replay connects
// again and sends STEP I itself next): --refuse-at I sends STEP I with its position moved by 1000 first (the TAIL
// refuses it), --lose-at I sends the header and half of STEP I's payload and closes the connection.
// --reset: the session starts with a RESET (the TAIL back to position 0: several records against one TAIL);
// --row-fnv: every STEP of R > 1 rows asks for each row's full-logits FNV (kStepWantRowFnv).
//   strix-deepseek41-pipeline-replay --msg-log FILE --stage-connect HOST:PORT --stage-token-file PATH [--tail-layers N]
//       sends a FRONT's whole session verbatim (a message log written by the CUDA FRONT's --msg-log: "DS4M" u32 version 1,
//       then per message u32 type + u64 length + payload: HELLO (its token replaced by the token file's), IMPORT_STATE,
//       STEPs, COMMITs, CLOSE) and prints every RESULT row: a TAIL replay of recorded FRONT boundaries from any state.
// Output: "pos P fed T argmax A logits_fnv F" per row (the rank app's format; windows of R > 1 rows are committed whole
// after the RESULT), "teacher-forced: ..." (whether each argmax equals the next recorded token) and a summary line.
namespace {

namespace pipe = strix::ds41::pipe;

struct Args {
    std::string record, connect, token_file, timeline, msg_log;
    std::size_t steps = 0;  // 0: all
    std::uint32_t tail_layers = 0;  // --tail-layers N: the TAIL's end layer (trimmed-model tests; default the record's)
    bool close = false, busy_poll = false;
    std::vector<std::size_t> refuse_at, lose_at;
    bool reset = false, row_fnv = false;
};

std::uint32_t u32_at(const std::vector<std::uint8_t>& p, std::size_t at) {
    if (at + 4U > p.size()) throw std::runtime_error("record: truncated step");
    std::uint32_t v;
    std::memcpy(&v, p.data() + at, 4);
    return v;
}

int run_log(const Args& a) {
    std::ifstream in(a.msg_log, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + a.msg_log);
    std::uint32_t head[2] = {};
    in.read(reinterpret_cast<char*>(head), 8);
    if (!in || head[0] != 0x4D345344U || head[1] != 1U) throw std::runtime_error(a.msg_log + ": not a DS4M v1 message log");
    const auto conn = pipe::connect(a.connect, std::chrono::seconds(900), std::chrono::seconds(600));
    conn->set_busy_poll(a.busy_poll);
    std::uint64_t id = 0;
    std::uint32_t vocab = 0;
    std::size_t rows_total = 0;
    for (;;) {
        std::uint32_t type = 0;
        std::uint64_t n = 0;
        if (!in.read(reinterpret_cast<char*>(&type), 4)) break;
        if (!in.read(reinterpret_cast<char*>(&n), 8) || n > (std::uint64_t{1} << 30U)) throw std::runtime_error("message log: bad entry");
        std::vector<std::uint8_t> p(static_cast<std::size_t>(n));
        if (!in.read(reinterpret_cast<char*>(p.data()), static_cast<std::streamsize>(n))) throw std::runtime_error("message log: truncated");
        const auto t = static_cast<pipe::MsgType>(type);
        ++id;
        if (t == pipe::MsgType::hello) {
            auto h = pipe::decode_hello(p);
            h.token = pipe::read_token_file(a.token_file);
            if (a.tail_layers != 0U) h.layers = a.tail_layers;
            vocab = h.vocab;
            conn->send(t, id, pipe::encode(h));
            const auto ack = pipe::decode_hello_ack(conn->receive(1U << 20U).payload);
            if (ack.status != 0U) throw std::runtime_error("TAIL refused the session: " + ack.message);
            std::printf("TAIL %s: layers [%u, %u), world %u, %s (backend of the log: %s)\n", a.connect.c_str(), ack.first_layer, ack.end_layer, ack.world,
                        ack.numeric_identity.c_str(), h.backend.c_str());
            continue;
        }
        conn->send(t, id, p);
        const auto m = conn->receive(std::uint64_t{64} << 20U);
        if (m.header.type == pipe::MsgType::error) throw std::runtime_error("TAIL error: " + pipe::decode_text(m.payload));
        if (t == pipe::MsgType::step) {
            if (m.header.type != pipe::MsgType::result) throw std::runtime_error("expected a RESULT");
            const auto r = pipe::decode_result(m.payload, vocab, 8U);
            const std::uint32_t rows = u32_at(p, 8);
            for (std::uint32_t row = 0; row < rows && row < r.tokens.size(); ++row)
                std::printf("pos %u fed %u argmax %u logits_fnv %016llx%s\n", u32_at(p, 4) + row, u32_at(p, 24 + 4U * row), r.tokens[row],
                            static_cast<unsigned long long>(row < r.logits_fnv.size() ? r.logits_fnv[row] : 0ULL), rows > 1U ? " (window)" : "");
            rows_total += rows;
        } else if (t == pipe::MsgType::close) {
            break;
        }
    }
    std::printf("message log replayed: %llu messages, %zu rows\n", static_cast<unsigned long long>(id), rows_total);
    return 0;
}

int run(const Args& a) {
    if (!a.msg_log.empty()) return run_log(a);
    std::ifstream in(a.record, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + a.record);
    std::uint32_t head[2] = {};
    in.read(reinterpret_cast<char*>(head), 8);
    if (!in || head[0] != 0x52345344U || head[1] != 1U) throw std::runtime_error(a.record + ": not a DS4R v1 record");
    const auto next = [&](std::vector<std::uint8_t>& out) {
        std::uint64_t n = 0;
        if (!in.read(reinterpret_cast<char*>(&n), 8)) return false;
        if (n > (std::uint64_t{1} << 30U)) throw std::runtime_error("record: entry too large");
        out.resize(static_cast<std::size_t>(n));
        if (!in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n))) throw std::runtime_error("record: truncated");
        return true;
    };
    std::vector<std::uint8_t> hello_bytes;
    if (!next(hello_bytes)) throw std::runtime_error("record: no HELLO");
    auto hello = pipe::decode_hello(hello_bytes);
    hello.token = pipe::read_token_file(a.token_file);
    if (a.tail_layers != 0U) hello.layers = a.tail_layers;
    std::vector<std::vector<std::uint8_t>> steps;
    for (std::vector<std::uint8_t> s; next(s);) steps.push_back(std::move(s));
    if (a.steps != 0U && a.steps < steps.size()) steps.resize(a.steps);
    std::printf("record %s: cut %u, %zu steps, backend %s\n", a.record.c_str(), hello.cut, steps.size(), hello.backend.c_str());

    std::unique_ptr<pipe::Connection> conn;
    std::uint64_t id = 1;
    const auto open_session = [&] {
        conn = pipe::connect(a.connect, std::chrono::seconds(900), std::chrono::seconds(600));
        conn->set_busy_poll(a.busy_poll);
        conn->send(pipe::MsgType::hello, ++id, pipe::encode(hello));
        const auto ackm = conn->receive(1U << 20U);
        if (ackm.header.type != pipe::MsgType::hello_ack) throw std::runtime_error("expected HELLO_ACK");
        const auto ack = pipe::decode_hello_ack(ackm.payload);
        if (ack.status != 0U) throw std::runtime_error("TAIL refused the session: " + ack.message);
        std::printf("TAIL %s: layers [%u, %u), world %u, %s\n", a.connect.c_str(), ack.first_layer, ack.end_layer, ack.world, ack.numeric_identity.c_str());
        std::fflush(stdout);
    };
    open_session();
    if (a.reset) {
        conn->send(pipe::MsgType::reset, ++id, {});
        if (conn->receive(1U << 20U).header.type != pipe::MsgType::ack) throw std::runtime_error("expected the ACK of the reset");
    }
    if (a.row_fnv)
        for (auto& st : steps)
            if (u32_at(st, 8) > 1U) {
                const std::uint32_t f = u32_at(st, 0) | pipe::kStepWantRowFnv;
                std::memcpy(st.data(), &f, 4);
            }
    std::printf("READY {\"role\":\"pipeline-replay\",\"cut\":%u,\"stage\":\"%s\"}\n", hello.cut, a.connect.c_str());
    std::fflush(stdout);
    const auto has = [](const std::vector<std::size_t>& v, std::size_t i) { return std::find(v.begin(), v.end(), i) != v.end(); };
    std::size_t injected = 0;

    std::vector<double> rtt;      // per step (RESULT received; a window's COMMIT follows outside the round trip)
    std::vector<std::uint32_t> argmax, fed, positions;  // per row
    std::vector<std::uint64_t> fnvs;
    std::vector<std::size_t> step_rows;
    for (std::size_t si = 0; si < steps.size(); ++si) {
        const auto& s = steps[si];
        if (has(a.refuse_at, si)) {  // the same STEP at a wrong position: the TAIL refuses it and ends the session
            auto bad = s;
            const std::uint32_t pos = u32_at(s, 4) + 1000U;
            std::memcpy(bad.data() + 4, &pos, 4);
            conn->send(pipe::MsgType::step, ++id, bad);
            const auto m = conn->receive(std::uint64_t{64} << 20U);
            if (m.header.type != pipe::MsgType::error) throw std::runtime_error("refuse-at: the TAIL did not refuse the moved STEP");
            std::printf("injected: STEP %zu refused (%s)\n", si, pipe::decode_text(m.payload).c_str());
            ++injected;
            conn.reset();
            open_session();
        }
        if (has(a.lose_at, si)) {  // the FRONT goes away in the middle of the STEP
            conn->send_truncated(pipe::MsgType::step, ++id, s, s.size() / 2U);
            std::printf("injected: STEP %zu lost after %zu of %zu bytes\n", si, s.size() / 2U, s.size());
            ++injected;
            conn.reset();
            open_session();
        }
        ++id;
        const auto t0 = std::chrono::steady_clock::now();
        conn->send(pipe::MsgType::step, id, s);
        const auto m = conn->receive(std::uint64_t{64} << 20U);
        const auto t1 = std::chrono::steady_clock::now();
        if (m.header.type == pipe::MsgType::error) throw std::runtime_error("TAIL error: " + pipe::decode_text(m.payload));
        if (m.header.type != pipe::MsgType::result || m.header.request != id) throw std::runtime_error("expected the RESULT of the step");
        const auto r = pipe::decode_result(m.payload, hello.vocab, 8U);
        rtt.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        const std::uint32_t rows = u32_at(s, 8);
        if (r.tokens.size() != rows) throw std::runtime_error("RESULT row count");
        step_rows.push_back(rows);
        for (std::uint32_t row = 0; row < rows; ++row) {
            positions.push_back(u32_at(s, 4) + row);
            fed.push_back(u32_at(s, 24 + 4U * row));
            argmax.push_back(r.tokens[row]);
            fnvs.push_back(row < r.logits_fnv.size() ? r.logits_fnv[row] : 0ULL);
        }
        if (rows > 1U) {  // a verify window: keep every row
            conn->send(pipe::MsgType::commit, ++id, pipe::encode_u32(rows));
            const auto done = conn->receive(1U << 20U);
            if (done.header.type != pipe::MsgType::ack) throw std::runtime_error("expected the ACK of the commit");
        }
    }
    for (std::size_t i = 0; i < positions.size(); ++i)
        std::printf("pos %u fed %u argmax %u logits_fnv %016llx\n", positions[i], fed[i], argmax[i], static_cast<unsigned long long>(fnvs[i]));
    std::size_t follow = 0, checked = 0;
    for (std::size_t i = 0; i + 1U < fed.size(); ++i) {
        ++checked;
        if (argmax[i] == fed[i + 1U]) ++follow;
    }
    std::printf("teacher-forced: %zu of %zu argmax equal the next recorded token\n", follow, checked);
    if (injected) std::printf("injected faults: %zu (each followed by a new session)\n", injected);
    if (a.close) {
        conn->send(pipe::MsgType::close, ++id, {});
        (void)conn->receive(1U << 20U);
    }
    if (!a.timeline.empty()) {
        std::ofstream f(a.timeline);
        f << "step,rows,rtt_us\n";
        for (std::size_t i = 0; i < rtt.size(); ++i) f << i << ',' << step_rows[i] << ',' << rtt[i] << '\n';
    }
    if (rtt.size() > 2U) {  // the first two steps pay one-time costs
        std::vector<double> v(rtt.begin() + 2, rtt.end());
        double mean = 0.0;
        for (const double x : v) mean += x;
        mean /= static_cast<double>(v.size());
        std::sort(v.begin(), v.end());
        std::printf("replay: %zu steps (%zu rows), round trip median %.1f us, mean %.1f us, p90 %.1f us (steps after the second) -> %.2f steps/s at the TAIL\n",
                    rtt.size(), positions.size(), v[v.size() / 2U], mean, v[v.size() * 9U / 10U], 1e6 / mean);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--record") a.record = next();
        else if (s == "--msg-log") a.msg_log = next();
        else if (s == "--stage-connect") a.connect = next();
        else if (s == "--stage-token-file") a.token_file = next();
        else if (s == "--steps") a.steps = std::stoul(next());
        else if (s == "--timeline") a.timeline = next();
        else if (s == "--busy-poll") a.busy_poll = true;
        else if (s == "--tail-layers") a.tail_layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--close") a.close = true;
        else if (s == "--reset") a.reset = true;
        else if (s == "--row-fnv") a.row_fnv = true;
        else if (s == "--refuse-at" || s == "--lose-at") {
            auto& v = s == "--refuse-at" ? a.refuse_at : a.lose_at;
            std::stringstream ss(next());
            for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stoul(t));
        }
        else throw std::invalid_argument("unknown argument " + s);
    }
    if ((a.record.empty() && a.msg_log.empty()) || a.connect.empty() || a.token_file.empty())
        throw std::invalid_argument("usage: strix-deepseek41-pipeline-replay --record FILE --stage-connect HOST:PORT --stage-token-file PATH");
    return run(a);
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
