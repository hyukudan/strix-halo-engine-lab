#pragma once

// Shared, header-only HTTP/1.1 message reader for the small raw-socket services that do not run on
// strix::serve::HttpServer (CUDA prefill services). It applies the same rules as HttpServer's request reader:
// header field names are tokens, Content-Length is plain decimal digits and repeated fields must agree, the header
// and body sizes are bounded, and one total deadline covers the header and the body. The body buffer grows with the
// bytes that arrive; it is never reserved from the announced length.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#endif

namespace strix::serve {

#ifdef _WIN32
using WireSocket = SOCKET;
#else
using WireSocket = int;
#endif

// Waits until `s` is readable (data, end of stream or an error: recv() reports which). poll() is used, never
// select(), so a descriptor number above FD_SETSIZE is fine. timeout_ms < 0 waits without limit.
inline bool wire_wait_readable(WireSocket s, int timeout_ms) {
#ifdef _WIN32
    WSAPOLLFD p{};
    p.fd = s;
    p.events = POLLRDNORM;
    const int rc = WSAPoll(&p, 1, timeout_ms);
#else
    pollfd p{};
    p.fd = s;
    p.events = POLLIN;
    const int rc = ::poll(&p, 1, timeout_ms);
#endif
    return rc > 0;
}

// One receive with a limit. Returns n > 0 bytes, 0 when the peer closed, -1 on an error and -2 when `timeout_ms`
// passed without data. timeout_ms < 0 does a plain recv() (the socket's own receive timeout applies).
inline long wire_recv(WireSocket s, char* p, std::size_t n, int timeout_ms) {
    if (timeout_ms >= 0 && !wire_wait_readable(s, timeout_ms)) return -2;
    const std::size_t piece = std::min<std::size_t>(n, std::size_t{1} << 20U);
#ifdef _WIN32
    const int k = ::recv(s, p, static_cast<int>(piece), 0);
#else
    const auto k = ::recv(s, p, piece, 0);
#endif
    return static_cast<long>(k);
}

struct HttpWireLimits {
    std::size_t max_header_bytes = 64U << 10U;
    std::uint64_t max_body_bytes = 64ULL << 20U;
    // One deadline from the first read: the header must complete within it, and the body within it plus one second
    // per started MiB. 0 = none (replies of a peer that computes before answering).
    int request_timeout_ms = 60'000;
    // Without Content-Length: false = no body (a request), true = the body runs to the end of the connection.
    bool body_until_close = false;
    // false: repeated Content-Length fields must agree; true: any repetition is refused.
    bool reject_repeated_length = false;
};

struct HttpWireMessage {
    std::string start;                                            // request line or status line
    std::vector<std::pair<std::string, std::string>> headers;     // names lower-cased, values trimmed
    std::string body;
    std::optional<std::uint64_t> content_length;
    // First value of a header (name in lower case), empty when absent.
    [[nodiscard]] std::string header(std::string_view name) const {
        for (const auto& [k, v] : headers)
            if (k == name) return v;
        return {};
    }
};

// The status a server answers for an error of read_http_message: 408 when the request missed its deadline, 413 for
// a body over its limit, 431 for a header over its limit, 400 for every malformed request. A client can then tell a
// slow link from a malformed request.
// read_http_message returns exactly these texts for the three cases, so the status cannot drift from the reader.
inline constexpr std::string_view kHttpWireTimedOut = "request timed out";
inline constexpr std::string_view kHttpWireBodyTooLarge = "body too large";
inline constexpr std::string_view kHttpWireHeadersTooLarge = "headers too large";
[[nodiscard]] inline int http_wire_error_status(std::string_view error) {
    if (error == kHttpWireTimedOut) return 408;
    if (error == kHttpWireBodyTooLarge) return 413;
    if (error == kHttpWireHeadersTooLarge) return 431;
    return 400;
}

// The prefill response reason table is part of the wire format of the Qwen and
// DeepSeek services. A caller with another protocol can supply its own reason.
[[nodiscard]] inline constexpr std::string_view http_prefill_reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

[[nodiscard]] inline std::string format_http_close_response_head(
    int status, std::string_view reason, std::string_view content_type, std::size_t body_bytes) {
    std::string head = "HTTP/1.1 " + std::to_string(status) + " ";
    head.append(reason);
    head.append("\r\nContent-Type: ").append(content_type);
    head.append("\r\nContent-Length: ").append(std::to_string(body_bytes));
    head.append("\r\nConnection: close\r\n\r\n");
    return head;
}

namespace wire_detail {

inline bool is_tchar(char c) {
    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    return std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}
inline bool is_token(std::string_view s) { return !s.empty() && std::all_of(s.begin(), s.end(), is_tchar); }
inline std::string_view trim_ows(std::string_view v) {
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.remove_prefix(1);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.remove_suffix(1);
    return v;
}
inline std::string lower_ascii(std::string_view v) {
    std::string r(v);
    for (char& c : r)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return r;
}

}  // namespace wire_detail

// Content-Length value: one or more decimal digits and nothing else, representable in 64 bits.
[[nodiscard]] inline bool parse_content_length_value(std::string_view v, std::uint64_t& out) {
    if (v.empty() || v.find_first_not_of("0123456789") != std::string_view::npos) return false;
    std::uint64_t parsed = 0;
    const auto r = std::from_chars(v.data(), v.data() + v.size(), parsed);
    if (r.ec != std::errc{} || r.ptr != v.data() + v.size()) return false;
    out = parsed;
    return true;
}

// Parses a complete head (without the final CRLFCRLF) into `m`. Returns an error text, empty on success.
[[nodiscard]] inline std::string parse_http_head(std::string_view head, HttpWireMessage& m, const HttpWireLimits& limits) {
    using namespace wire_detail;
    m.headers.clear();
    m.content_length.reset();
    const auto line_end = head.find("\r\n");
    const auto first = head.substr(0, line_end);
    if (first.empty() || first.find_first_of(std::string_view("\0\r\n", 3)) != std::string_view::npos) return "malformed start line";
    m.start = std::string(first);
    for (std::size_t at = line_end == std::string_view::npos ? head.size() : line_end + 2U; at < head.size();) {
        auto next = head.find("\r\n", at);
        if (next == std::string_view::npos) next = head.size();
        const auto line = head.substr(at, next - at);
        const auto colon = line.find(':');
        // A field name is a token: no whitespace or control bytes (this also refuses obsolete line folding).
        if (colon == std::string_view::npos || !is_token(line.substr(0, colon))) return "malformed header field";
        const auto value = trim_ows(line.substr(colon + 1U));
        if (value.find_first_of(std::string_view("\0\r\n", 3)) != std::string_view::npos) return "malformed header field";
        const auto name = lower_ascii(line.substr(0, colon));
        if (name == "transfer-encoding" && lower_ascii(value) != "identity") return "chunked bodies are not supported";
        if (name == "content-length") {
            std::uint64_t parsed = 0;
            if (!parse_content_length_value(value, parsed)) return "bad Content-Length";
            if (m.content_length && (limits.reject_repeated_length || *m.content_length != parsed)) return "conflicting Content-Length";
            m.content_length = parsed;
        }
        m.headers.emplace_back(name, std::string(value));
        at = next + 2U;
    }
    return {};
}

// Reads one message through `recv(char* buffer, std::size_t capacity, int timeout_ms) -> long` (the contract of
// wire_recv). Starts with the bytes already in `carry` and leaves there what follows the message. Returns an error
// text, empty on success.
template <class Recv>
[[nodiscard]] std::string read_http_message(Recv&& recv, HttpWireMessage& m, const HttpWireLimits& limits, std::string& carry) {
    using Clock = std::chrono::steady_clock;
    std::string buf = std::move(carry);
    carry.clear();
    std::optional<Clock::time_point> deadline;
    if (limits.request_timeout_ms > 0) deadline = Clock::now() + std::chrono::milliseconds(limits.request_timeout_ms);
    enum class Got { data, closed, late };
    char tmp[65536];
    const auto receive = [&](std::size_t want) {
        int wait_ms = -1;
        if (deadline) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - Clock::now()).count();
            if (left <= 0) return Got::late;
            wait_ms = static_cast<int>(std::min<long long>(left, 1'000'000));
        }
        const long k = recv(tmp, std::min(sizeof(tmp), want), wait_ms);
        if (k == -2) return Got::late;
        if (k <= 0) return Got::closed;
        buf.append(tmp, static_cast<std::size_t>(k));
        return Got::data;
    };
    std::size_t head_end = std::string::npos;
    while ((head_end = buf.find("\r\n\r\n")) == std::string::npos) {
        // Without a terminator the head is at least size - 3 bytes long.
        if (buf.size() > limits.max_header_bytes + 3U) return std::string(kHttpWireHeadersTooLarge);
        const auto got = receive(sizeof(tmp));
        if (got == Got::late) return std::string(kHttpWireTimedOut);
        if (got == Got::closed) return buf.empty() ? "connection closed" : "truncated headers";
    }
    if (head_end > limits.max_header_bytes) return std::string(kHttpWireHeadersTooLarge);
    if (const auto err = parse_http_head(std::string_view(buf).substr(0, head_end), m, limits); !err.empty()) return err;
    m.body = buf.substr(head_end + 4U);
    buf.clear();
    buf.shrink_to_fit();
    if (!m.content_length) {
        if (!limits.body_until_close) { carry = std::move(m.body); m.body.clear(); return {}; }
        for (;;) {
            const auto got = receive(sizeof(tmp));
            if (got == Got::late) return std::string(kHttpWireTimedOut);
            if (got == Got::closed) break;
            m.body.append(buf);
            buf.clear();
            if (m.body.size() > limits.max_body_bytes) return std::string(kHttpWireBodyTooLarge);
        }
        return {};
    }
    const std::uint64_t want_u = *m.content_length;
    if (want_u > limits.max_body_bytes) return std::string(kHttpWireBodyTooLarge);
    const auto want = static_cast<std::size_t>(want_u);
    if (m.body.size() > want) { carry = m.body.substr(want); m.body.resize(want); }
    if (deadline) *deadline += std::chrono::seconds(static_cast<long long>((want_u + (1U << 20U) - 1U) >> 20U));
    while (m.body.size() < want) {
        const auto got = receive(want - m.body.size());
        if (got == Got::late) return std::string(kHttpWireTimedOut);
        if (got == Got::closed) return "truncated body";
        m.body.append(buf);
        buf.clear();
    }
    return {};
}

// Admission and an exception-safe detached worker for one connection. `live` counts running workers; with
// limit > 0 a worker is refused (false) while `live` has reached it. A failed thread start (std::system_error)
// restores the count and returns false: the caller still owns, and must close, its connection. The worker's own exit
// decrements `live`, also when `fn` throws.
template <class F>
[[nodiscard]] bool spawn_counted_connection(std::atomic<int>& live, int limit, F&& fn) {
    if (limit > 0) {
        int cur = live.load();
        do {
            if (cur >= limit) return false;
        } while (!live.compare_exchange_weak(cur, cur + 1));
    } else {
        ++live;
    }
    try {
        std::thread([&live, f = std::forward<F>(fn)]() mutable {
            struct Done {
                std::atomic<int>& n;
                ~Done() { --n; }
            } done{live};
            try {
                f();
            } catch (...) {
            }
        }).detach();
    } catch (...) {
        --live;
        return false;
    }
    return true;
}

}  // namespace strix::serve
