// Host tests of the shared raw-socket HTTP reader (include/strix/serve/http_wire.hpp) and of the Qwen prefill
// service's reader built on it (src/cuda/models/qwen/qwen_http.hpp): malformed or ambiguous heads give a clean
// error, a slow peer is cut off by one total deadline, and the body buffer follows the bytes received. The Qwen
// service's request body reader (qwen_prefill_request.hpp) refuses malformed bodies with an exception.
#include "strix/serve/http_wire.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>
#include "qwen_http.hpp"
#endif
#include "qwen_prefill_request.hpp"

namespace {

int failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

using strix::serve::HttpWireLimits;
using strix::serve::HttpWireMessage;
using namespace std::chrono_literals;

std::string head_of(const std::string& fields) { return "POST /v1/prefill HTTP/1.1\r\nHost: x" + fields; }

void test_close_response_head() {
    const struct { int status; std::string_view reason; } cases[] = {
        {200, "OK"}, {400, "Bad Request"}, {401, "Unauthorized"}, {404, "Not Found"},
        {405, "Method Not Allowed"}, {408, "Request Timeout"}, {409, "Conflict"},
        {413, "Payload Too Large"}, {431, "Request Header Fields Too Large"},
        {500, "Internal Server Error"}, {502, "Bad Gateway"}, {503, "Service Unavailable"},
        {599, "Status"},
    };
    const std::string body("\x01\x00\x7f\x80\xff", 5);
    for (const auto& [status, reason] : cases) {
        const std::string old_head = "HTTP/1.1 " + std::to_string(status) + ' ' + std::string(reason) +
            "\r\nContent-Type: application/octet-stream\r\nContent-Length: 5\r\nConnection: close\r\n\r\n";
        const auto head = strix::serve::format_http_close_response_head(
            status, strix::serve::http_prefill_reason(status), "application/octet-stream", body.size());
        CHECK(head + body == old_head + body);
    }
    CHECK(strix::serve::format_http_close_response_head(200, "OK", "application/json", 0) ==
          "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
}

void test_head_table() {
    const HttpWireLimits limits;
    struct Row { std::string fields; bool ok; const char* name; };
    const std::vector<Row> rows = {
        {"\r\nContent-Length: 5", true, "plain length"},
        {"\r\nContent-Length: 0005", true, "leading zeros are digits"},
        {"\r\nContent-Length:\t 5 \t", true, "optional whitespace around the value"},
        {"\r\nContent-Length: 5\r\nContent-Length: 5", true, "repeated, agreeing"},
        {"\r\nContent-Length: 5\r\nContent-Length: 6", false, "repeated, conflicting"},
        {"\r\nContent-Length: 6\r\ncontent-length: 5", false, "repeated, conflicting, other case"},
        {"\r\nContent-Length: +5", false, "plus sign"},
        {"\r\nContent-Length: -1", false, "minus sign"},
        {"\r\nContent-Length: 10abc", false, "trailing garbage"},
        {"\r\nContent-Length: 1 2", false, "inner whitespace"},
        {"\r\nContent-Length: 0x10", false, "hex"},
        {"\r\nContent-Length: ", false, "empty"},
        {"\r\nContent-Length: 18446744073709551616", false, "64-bit overflow"},
        {"\r\nContent-Length: 99999999999999999999999999", false, "far overflow"},
        {"\r\nContent-Length : 5", false, "space before the colon"},
        {"\r\n Content-Length: 5", false, "obsolete line folding"},
        {"\r\nNoColonHere", false, "field without colon"},
        {"\r\n: empty-name", false, "empty field name"},
        {"\r\nX-A: a\x01" "b", true, "control byte in a value is not framing"},
        {"\r\nTransfer-Encoding: chunked", false, "chunked"},
        {"\r\nTransfer-Encoding: identity", true, "identity"},
    };
    for (const auto& row : rows) {
        HttpWireMessage m;
        const std::string err = strix::serve::parse_http_head(head_of(row.fields), m, limits);
        if ((err.empty()) != row.ok) std::fprintf(stderr, "row '%s': error text '%s'\n", row.name, err.c_str());
        CHECK(err.empty() == row.ok);
    }
    HttpWireLimits strict;
    strict.reject_repeated_length = true;
    HttpWireMessage m;
    CHECK(!strix::serve::parse_http_head(head_of("\r\nContent-Length: 5\r\nContent-Length: 5"), m, strict).empty());
    // A NUL inside a value or the start line is refused.
    CHECK(!strix::serve::parse_http_head(std::string("POST / HTTP/1.1\r\nX: a\0b", 23), m, limits).empty());
    CHECK(!strix::serve::parse_http_head(std::string("POST /\0 HTTP/1.1", 16), m, limits).empty());
    std::uint64_t v = 0;
    CHECK(strix::serve::parse_content_length_value("18446744073709551615", v) && v == 18446744073709551615ULL);
    CHECK(!strix::serve::parse_content_length_value("", v));
}

// A scripted peer: every call returns the next piece (or -2 to model a wait that timed out).
struct Script {
    std::vector<std::string> pieces;
    std::size_t at = 0;
    long operator()(char* p, std::size_t cap, int) {
        if (at >= pieces.size()) return 0;
        const std::string& s = pieces[at];
        if (s == "<timeout>") { ++at; return -2; }
        const auto n = std::min(cap, s.size());
        std::copy_n(s.data(), n, p);
        if (n == s.size()) ++at; else pieces[at] = s.substr(n);
        return static_cast<long>(n);
    }
};

std::string read_with(Script script, const HttpWireLimits& limits, HttpWireMessage& m, std::string* carry_out = nullptr) {
    std::string carry;
    auto err = strix::serve::read_http_message(script, m, limits, carry);
    if (carry_out) *carry_out = carry;
    return err;
}

void test_reader() {
    HttpWireLimits limits;
    limits.max_header_bytes = 1024;
    limits.max_body_bytes = 100;
    HttpWireMessage m;
    CHECK(read_with({{"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhel", "lo"}}, limits, m).empty() && m.body == "hello");
    std::string carry;
    CHECK(read_with({{"POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\nabNEXT"}}, limits, m, &carry).empty() && m.body == "ab" && carry == "NEXT");
    CHECK(read_with({{"GET /health HTTP/1.1\r\n\r\n"}}, limits, m).empty() && m.body.empty());
    CHECK(read_with({{"POST / HTTP/1.1\r\nContent-Length: 101\r\n\r\n"}}, limits, m) == "body too large");
    CHECK(read_with({{"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhe"}}, limits, m) == "truncated body");
    CHECK(read_with({{"POST / HTTP/1.1\r\nContent-Le"}}, limits, m) == "truncated headers");  // no CRLFCRLF, then EOF
    CHECK(read_with({{}}, limits, m) == "connection closed");
    CHECK(read_with({{std::string(5000, 'a')}}, limits, m) == "headers too large");  // no terminator
    CHECK(read_with({{"GET / HTTP/1.1\r\nX: " + std::string(2000, 'a') + "\r\n\r\n"}}, limits, m) == "headers too large");  // terminator beyond the limit
    CHECK(read_with({{"POST / HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 6\r\n\r\nhello!"}}, limits, m) == "conflicting Content-Length");
    CHECK(read_with({{"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhe", "<timeout>"}}, limits, m) == "request timed out");
    // Without Content-Length a request has no body; a reply body runs to the end of the connection.
    CHECK(read_with({{"POST / HTTP/1.1\r\n\r\nxyz"}}, limits, m, &carry).empty() && m.body.empty() && carry == "xyz");
    limits.body_until_close = true;
    CHECK(read_with({{"HTTP/1.1 200 OK\r\n\r\nxy", "z"}}, limits, m).empty() && m.body == "xyz");
    limits.max_body_bytes = 2;
    CHECK(read_with({{"HTTP/1.1 200 OK\r\n\r\nxy", "z"}}, limits, m) == "body too large");
}

// The deadline covers the whole request: a peer that never stops sending small pieces is still cut off.
void test_deadline() {
    HttpWireLimits limits;
    limits.request_timeout_ms = 200;
    HttpWireMessage m;
    std::string carry;
    const auto begin = std::chrono::steady_clock::now();
    int calls = 0;
    const auto err = strix::serve::read_http_message(
        [&](char* p, std::size_t, int ms) -> long {
            ++calls;
            CHECK(ms >= 0 && ms <= 200);
            std::this_thread::sleep_for(40ms);
            p[0] = 'a';  // never completes the head
            return 1;
        }, m, limits, carry);
    const auto took = std::chrono::steady_clock::now() - begin;
    CHECK(err == "request timed out");
    CHECK(took < 1500ms);
    CHECK(calls >= 2 && calls < 20);
}

void test_spawn() {
    std::atomic<int> live{0};
    std::atomic<int> ran{0};
    std::atomic<bool> release{false};
    CHECK(strix::serve::spawn_counted_connection(live, 2, [&] { ++ran; while (!release.load()) std::this_thread::sleep_for(1ms); }));
    CHECK(strix::serve::spawn_counted_connection(live, 2, [&] { ++ran; while (!release.load()) std::this_thread::sleep_for(1ms); }));
    CHECK(live.load() == 2);
    CHECK(!strix::serve::spawn_counted_connection(live, 2, [&] { ++ran; }));  // at the cap: refused, count unchanged
    CHECK(live.load() == 2);
    release = true;
    for (int i = 0; i < 2000 && live.load() > 0; ++i) std::this_thread::sleep_for(1ms);
    CHECK(live.load() == 0 && ran.load() == 2);
    // A worker that throws still gives its slot back.
    CHECK(strix::serve::spawn_counted_connection(live, 0, [] { throw 1; }));
    for (int i = 0; i < 2000 && live.load() > 0; ++i) std::this_thread::sleep_for(1ms);
    CHECK(live.load() == 0);
}

#ifndef _WIN32
struct Pair {
    int a = -1, b = -1;
    Pair() { int fds[2]; if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0) { a = fds[0]; b = fds[1]; } }
    ~Pair() { if (a >= 0) ::close(a); if (b >= 0) ::close(b); }
};

// poll() is used, so a descriptor number above FD_SETSIZE works (select() would write outside its fd_set).
void test_high_descriptor() {
    Pair pair;
    CHECK(pair.a >= 0);
    rlimit lim{};
    if (::getrlimit(RLIMIT_NOFILE, &lim) != 0 || lim.rlim_cur < 1500) { std::puts("skip high descriptor: RLIMIT_NOFILE"); return; }
    const int high = ::fcntl(pair.a, F_DUPFD, 1200);
    CHECK(high >= 1200);
    CHECK(!strix::serve::wire_wait_readable(high, 10));
    CHECK(::send(pair.b, "x", 1, 0) == 1);
    CHECK(strix::serve::wire_wait_readable(high, 1000));
    char c = 0;
    CHECK(strix::serve::wire_recv(high, &c, 1, 100) == 1 && c == 'x');
    CHECK(strix::serve::wire_recv(high, &c, 1, 20) == -2);
    ::close(high);
}

// Qwen service reader over a real socket pair. `BASE_QWEN` builds against the unhardened header.
std::string qwen_read(const std::string& wire, std::size_t max_body, strix::qwen_http::Message& m, bool close_after = true,
                      int deadline_ms = 60000) {
    Pair pair;
    std::thread writer([&] {
        std::size_t off = 0;
        while (off < wire.size()) {
            const auto k = ::send(pair.b, wire.data() + off, std::min<std::size_t>(wire.size() - off, 1U << 16U), MSG_NOSIGNAL);
            if (k <= 0) break;
            off += static_cast<std::size_t>(k);
        }
        if (close_after) ::shutdown(pair.b, SHUT_WR);
    });
    std::string carry;
#ifdef BASE_QWEN
    (void)deadline_ms;
    const std::string err = strix::qwen_http::read_message(pair.a, m, max_body, false, carry);
#else
    const std::string err = strix::qwen_http::read_message(pair.a, m, max_body, false, carry, deadline_ms);
#endif
    ::shutdown(pair.a, SHUT_RDWR);
    ::shutdown(pair.b, SHUT_RDWR);
    writer.join();
    return err;
}

void test_qwen_reader() {
    using strix::qwen_http::Message;
    const std::size_t cap = std::size_t{64} << 20;
    const auto post = [](const std::string& fields, const std::string& body) {
        return "POST /v1/prefill HTTP/1.1\r\nHost: x" + fields + "\r\n\r\n" + body;
    };
    {
        Message m;
        CHECK(qwen_read(post("\r\nContent-Length: 5", "hello"), cap, m).empty() && m.body == "hello");
    }
    struct Bad { std::string fields; const char* name; };
    const std::vector<Bad> bad = {
        {"\r\nContent-Length: 6\r\nContent-Length: 5", "duplicate conflicting"},
        {"\r\nContent-Length: +5", "plus"},
        {"\r\nContent-Length: 5abc", "trailing garbage"},
        {"\r\nContent-Length: 5 5", "inner whitespace"},
        {"\r\nContent-Length: -1", "minus"},
        {"\r\nContent-Length: 99999999999999999999999", "overflow"},
        {"\r\nContent-Length : 5", "space before colon"},
        {"\r\nTransfer-Encoding: chunked", "chunked"},
    };
    for (const auto& b : bad) {
        Message m;
        const std::string err = qwen_read(post(b.fields, "hello"), cap, m);
        if (err.empty()) std::fprintf(stderr, "qwen reader accepted: %s\n", b.name);
        CHECK(!err.empty());
    }
    {
        Message m;  // header too large and missing terminator
        CHECK(!qwen_read("POST / HTTP/1.1\r\nX: " + std::string(70000, 'a') + "\r\n\r\n", cap, m).empty());
        Message m2;
        CHECK(qwen_read("POST / HTTP/1.1\r\nContent-Le", cap, m2) == "truncated headers");
    }
    {
        // The announced length is not reserved before the bytes arrive.
        Message m;
        const std::string err = qwen_read(post("\r\nContent-Length: 67108864", "abc"), cap, m);
        CHECK(err == "truncated body");
        CHECK(m.body.capacity() < (std::size_t{1} << 20));
    }
#ifndef BASE_QWEN
    {
        // A peer that trickles one byte at a time is cut off by the total deadline, not held until it stops.
        Message m;
        const auto begin = std::chrono::steady_clock::now();
        std::string trickle = "POST / HTTP/1.1\r\nX-Slow: ";
        Pair pair;
        std::atomic<bool> stop{false};
        std::thread writer([&] {
            for (const char ch : trickle) if (::send(pair.b, &ch, 1, MSG_NOSIGNAL) != 1) return;
            while (!stop.load()) { if (::send(pair.b, "a", 1, MSG_NOSIGNAL) != 1) return; std::this_thread::sleep_for(20ms); }
        });
        std::string carry;
        const std::string err = strix::qwen_http::read_message(pair.a, m, cap, false, carry, 300);
        const auto took = std::chrono::steady_clock::now() - begin;
        stop = true;
        ::shutdown(pair.a, SHUT_RDWR);
        writer.join();
        CHECK(err == "request timed out");
        CHECK(took < 2000ms);
    }
#endif
}
#endif

}  // namespace

// The Qwen prefill service's body reader refuses every malformed body with an exception (answered 400) and never
// dereferences an absent key; a valid body reads every field as before.
void test_qwen_prefill_request() {
    const auto refused = [](const char* body) {
        try {
            strix::qwen_http::PrefillRequest r;
            strix::qwen_http::parse_prefill_request(strix::serve::parse_json(body), 65520U, r);
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    CHECK(refused("{}"));
    CHECK(refused("[]"));
    CHECK(refused("7"));
    CHECK(refused(R"({"handoff":"h:1"})"));
    CHECK(refused(R"({"tokens":[1,2]})"));
    CHECK(refused(R"({"tokens":"12","handoff":"h:1"})"));
    CHECK(refused(R"({"tokens":[1,2],"handoff":3})"));
    CHECK(refused(R"({"tokens":[1,2],"handoff":"h:1","block_schedule":"x"})"));
    CHECK(refused(R"({"tokens":[1,2],"handoff":"h:1","final_block_rows":65521})"));
    CHECK(refused(R"({"tokens":[1,2],"handoff":"h:1","final_block_rows":-1})"));
    CHECK(refused(R"({"tokens":[1,-2],"handoff":"h:1"})"));
    CHECK(refused(R"({"tokens":[1,4294967296],"handoff":"h:1"})"));
    CHECK(refused(R"({"tokens":[1,2],"handoff":"h:1","local_rows":-1})"));
    CHECK(!refused(R"({"tokens":[1,4294967295],"handoff":"h:1","local_rows":0})"));
    strix::qwen_http::PrefillRequest r;
    strix::qwen_http::parse_prefill_request(
        strix::serve::parse_json(R"({"tokens":[5,6,7],"handoff":"h:1","session":{"a":1},"handoff_api_key":"k",)"
                                 R"("handoff_codec":"c","local_rows":9,"trace":true,"block_schedule":"tail-large",)"
                                 R"("final_block_rows":65520,"node_completions":"n"})"),
        65520U, r);
    CHECK((r.tokens == std::vector<std::uint32_t>{5, 6, 7}) && r.handoff == "h:1" && r.session.is_object());
    CHECK(r.api_key == "k" && r.has_codec && r.codec == "c" && r.local_rows == 9 && r.trace);
    CHECK(r.has_schedule && r.tail_large && r.has_final_rows && r.final_rows == 65520U && r.node_completions == "n");
}

// A slow request is answered 408, an oversized body 413 and an oversized header 431; every other error 400.
void test_error_status() {
    using strix::serve::http_wire_error_status;
    CHECK(http_wire_error_status("request timed out") == 408);
    CHECK(http_wire_error_status("body too large") == 413);
    CHECK(http_wire_error_status("headers too large") == 431);
    for (const char* e : {"bad Content-Length", "conflicting Content-Length", "malformed start line", "truncated body",
                          "malformed header field", "chunked bodies are not supported"})
        CHECK(http_wire_error_status(e) == 400);
    // The reader's own errors map to the same statuses (a rewording of the reader cannot fall back to 400).
    HttpWireLimits limits;
    limits.max_header_bytes = 1024;
    limits.max_body_bytes = 100;
    HttpWireMessage m;
    CHECK(http_wire_error_status(read_with({{"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhe", "<timeout>"}}, limits, m)) == 408);
    CHECK(http_wire_error_status(read_with({{"POST / HTTP/1.1\r\nContent-Length: 101\r\n\r\n"}}, limits, m)) == 413);
    CHECK(http_wire_error_status(read_with({{std::string(5000, 'a')}}, limits, m)) == 431);
}

int main() {
    test_qwen_prefill_request();
    test_error_status();
    test_head_table();
    test_close_response_head();
    test_reader();
    test_deadline();
    test_spawn();
#ifndef _WIN32
    test_high_descriptor();
    test_qwen_reader();
#endif
    if (failures != 0) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::puts("serve_http_wire_tests: ok");
    return 0;
}
