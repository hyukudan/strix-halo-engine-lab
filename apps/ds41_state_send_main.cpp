// Point-to-point transfer of a deepseek41 attention state file (strix/ds41-attention-state-v1, ~890 B per position plus
// the 40 window rings): the prefill host sends it to the decode group's rank 0. One TCP connection (TCP_NODELAY, 4 MB
// socket buffers), each message length-prefixed with its SHA-256; the receiver acknowledges the last byte at once (the
// sender's timing: first byte sent .. acknowledgement received, one round trip included), then hashes the payload and
// returns its own SHA-256 so both sides report the check.
//
//   strix-ds41-state-send --send FILE HOST:PORT [--repeat N]   N timed sends of the same file (default 3)
//   strix-ds41-state-send --recv PORT OUT_FILE [--count N]     accepts one connection, receives N messages, keeps the last
//   strix-ds41-state-send --synth POSITIONS OUT_FILE           (built with the state library) a synthetic 40-layer state
//
// Self-contained without --synth: `c++ -O2 -std=c++20 apps/ds41_state_send_main.cpp` on POSIX, MSVC with ws2_32 on
// Windows.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
constexpr socket_t kBadSocket = INVALID_SOCKET;
inline void close_socket(socket_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t kBadSocket = -1;
inline void close_socket(socket_t s) { ::close(s); }
#endif

#ifdef STRIX_DS41_STATE_SYNTH
#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_attention.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---- SHA-256 (FIPS 180-4) --------------------------------------------------------------------------------------------
using Digest = std::array<std::uint8_t, 32>;
Digest sha256(const std::uint8_t* data, std::size_t n) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto rotr = [](std::uint32_t x, unsigned r) { return (x >> r) | (x << (32U - r)); };
    const auto block = [&](const std::uint8_t* b) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t{b[4 * i]} << 24U) | (std::uint32_t{b[4 * i + 1]} << 16U) | (std::uint32_t{b[4 * i + 2]} << 8U) | b[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    };
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) block(data + i);
    std::uint8_t tail[128] = {};
    const std::size_t rest = n - i;
    if (rest != 0U) std::memcpy(tail, data + i, rest);
    tail[rest] = 0x80;
    const std::size_t tail_len = rest + 9 <= 64 ? 64 : 128;
    const std::uint64_t bits = std::uint64_t{n} * 8U;
    for (int j = 0; j < 8; ++j) tail[tail_len - 1 - j] = static_cast<std::uint8_t>(bits >> (8U * j));
    for (std::size_t j = 0; j < tail_len; j += 64) block(tail + j);
    Digest out{};
    for (int j = 0; j < 8; ++j)
        for (int b = 0; b < 4; ++b) out[4 * j + b] = static_cast<std::uint8_t>(h[j] >> (24U - 8U * b));
    return out;
}
std::string hex(const Digest& d) {
    static const char* x = "0123456789abcdef";
    std::string s;
    for (const auto b : d) { s += x[b >> 4U]; s += x[b & 15U]; }
    return s;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---- sockets ---------------------------------------------------------------------------------------------------------
struct Net {
    Net() {
#ifdef _WIN32
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w) != 0) throw std::runtime_error("WSAStartup failed");
#endif
    }
    ~Net() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

void tune(socket_t s) {
    int one = 1, buf = 4 << 20;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof buf);
}

void send_all(socket_t s, const void* p, std::size_t n) {
    const char* b = static_cast<const char*>(p);
    while (n != 0U) {
        const int chunk = static_cast<int>(std::min<std::size_t>(n, 1U << 24U));
        const auto k = ::send(s, b, chunk, 0);
        if (k <= 0) throw std::runtime_error("send failed");
        b += k;
        n -= static_cast<std::size_t>(k);
    }
}
void recv_all(socket_t s, void* p, std::size_t n) {
    char* b = static_cast<char*>(p);
    while (n != 0U) {
        const int chunk = static_cast<int>(std::min<std::size_t>(n, 1U << 24U));
        const auto k = ::recv(s, b, chunk, 0);
        if (k <= 0) throw std::runtime_error("connection closed or recv failed");
        b += k;
        n -= static_cast<std::size_t>(k);
    }
}

constexpr char kMagic[8] = {'D', 'S', '4', '1', 'S', 'N', 'D', '1'};
struct Header {
    char magic[8];
    std::uint64_t bytes;
    std::uint8_t sha[32];
};

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

int do_send(const std::string& file, const std::string& target, int repeat) {
    const auto colon = target.rfind(':');
    if (colon == std::string::npos) throw std::runtime_error("target must be HOST:PORT");
    const std::string host = target.substr(0, colon), port = target.substr(colon + 1);
    const auto data = read_file(file);
    auto t0 = std::chrono::steady_clock::now();
    const Digest d = sha256(data.data(), data.size());
    std::printf("file %s: %zu bytes (%.2f MB), sha256 %s (%.1f ms)\n", file.c_str(), data.size(), static_cast<double>(data.size()) / 1e6, hex(d).c_str(),
                ms_since(t0));
    Net net;
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) throw std::runtime_error("cannot resolve " + target);
    const socket_t s = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == kBadSocket) throw std::runtime_error("socket failed");
    tune(s);
    t0 = std::chrono::steady_clock::now();
    if (::connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0) throw std::runtime_error("cannot connect to " + target);
    freeaddrinfo(res);
    std::printf("connected to %s in %.2f ms\n", target.c_str(), ms_since(t0));
    std::vector<double> times;
    bool all_ok = true;
    for (int i = 0; i < repeat; ++i) {
        Header h{};
        std::memcpy(h.magic, kMagic, 8);
        h.bytes = data.size();
        std::memcpy(h.sha, d.data(), 32);
        t0 = std::chrono::steady_clock::now();
        send_all(s, &h, sizeof h);
        send_all(s, data.data(), data.size());
        char ack = 0;
        recv_all(s, &ack, 1);
        const double ms = ms_since(t0);
        Digest theirs{};
        recv_all(s, theirs.data(), 32);
        const bool ok = ack == 'A' && theirs == d;
        all_ok = all_ok && ok;
        times.push_back(ms);
        std::printf("send %d: %.2f ms, %.1f MB/s, receiver sha256 %s\n", i, ms, static_cast<double>(data.size()) / 1e3 / ms, ok ? "match" : "MISMATCH");
    }
    close_socket(s);
    std::sort(times.begin(), times.end());
    std::printf("RESULT bytes %zu min_ms %.2f median_ms %.2f min_MBps %.1f check %s\n", data.size(), times.front(), times[times.size() / 2],
                static_cast<double>(data.size()) / 1e3 / times.front(), all_ok ? "ok" : "FAILED");
    return all_ok ? 0 : 1;
}

int do_recv(const std::string& port, const std::string& out, int count) {
    Net net;
    const socket_t ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls == kBadSocket) throw std::runtime_error("socket failed");
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(static_cast<std::uint16_t>(std::stoi(port)));
    if (::bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || ::listen(ls, 1) != 0) throw std::runtime_error("cannot listen on " + port);
    std::printf("listening on %s\n", port.c_str());
    std::fflush(stdout);
    const socket_t s = ::accept(ls, nullptr, nullptr);
    if (s == kBadSocket) throw std::runtime_error("accept failed");
    tune(s);
    std::vector<std::uint8_t> data;
    bool all_ok = true;
    for (int i = 0; i < count; ++i) {
        Header h{};
        recv_all(s, &h, sizeof h);
        const auto t0 = std::chrono::steady_clock::now();  // header in: the payload follows
        if (std::memcmp(h.magic, kMagic, 8) != 0 || h.bytes > (std::uint64_t{1} << 36U)) throw std::runtime_error("bad header");
        data.resize(h.bytes);
        recv_all(s, data.data(), data.size());
        const double ms = ms_since(t0);
        const char ack = 'A';
        send_all(s, &ack, 1);
        const auto t1 = std::chrono::steady_clock::now();
        const Digest d = sha256(data.data(), data.size());
        const double hash_ms = ms_since(t1);
        send_all(s, d.data(), 32);
        const bool ok = std::memcmp(d.data(), h.sha, 32) == 0;
        all_ok = all_ok && ok;
        std::printf("recv %d: %llu bytes in %.2f ms after the header (%.1f MB/s), sha256 %s %s (%.1f ms)\n", i, static_cast<unsigned long long>(h.bytes),
                    ms, static_cast<double>(h.bytes) / 1e3 / ms, hex(d).c_str(), ok ? "ok" : "MISMATCH", hash_ms);
    }
    close_socket(s);
    close_socket(ls);
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!f) throw std::runtime_error("cannot write " + out);
    return all_ok ? 0 : 1;
}

#ifdef STRIX_DS41_STATE_SYNTH
// A released-plan 40-layer state of `positions` positions with random contents (the sizes of a real handoff).
int do_synth(std::uint32_t positions, const std::string& out) {
    namespace sd = strix::deepseek41;
    namespace attn = strix::ds41::attn;
    std::mt19937_64 rng(positions);
    const auto fill = [&](auto& v, std::size_t n) {
        v.resize(n);
        auto* b = reinterpret_cast<std::uint8_t*>(v.data());
        for (std::size_t i = 0; i < n * sizeof(v[0]); i += 8) {
            const std::uint64_t r = rng();
            std::memcpy(b + i, &r, std::min<std::size_t>(8, n * sizeof(v[0]) - i));
        }
    };
    const auto plan = attn::released_plan();
    sd::AttentionState s;
    s.position = positions;
    s.layers.resize(plan.size());
    for (std::size_t l = 0; l < plan.size(); ++l) {
        const auto& r = plan[l];
        auto& L = s.layers[l];
        L.owns_kv = r.owns_kv; L.owns_index = r.owns_index; L.ratio = r.ratio;
        fill(L.ring, std::size_t{sd::AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
        if (r.owns_kv) {
            const std::size_t entries = positions / r.ratio;
            fill(L.kv, entries * attn::kKvEntryBytes);
            fill(L.idx_keys, entries * attn::kIdxEntryBytes);
            if (r.ratio == 2U) { std::vector<std::uint32_t> c; fill(c, 1024); L.carry.assign(1024, 0.5F); L.carry_valid = true; }
        }
        if (r.owns_index) { L.topk.resize(std::min<std::uint32_t>(attn::kTopK, positions / 2)); for (std::size_t i = 0; i < L.topk.size(); ++i) L.topk[i] = static_cast<std::uint32_t>(i); }
    }
    s.tokens.resize(positions);
    for (auto& t : s.tokens) t = static_cast<std::uint32_t>(rng() % 129280U);
    sd::save_attention_state(out, s);
    std::printf("synthetic state %u positions: bytes() %.3f MB, file %s\n", positions, static_cast<double>(s.bytes()) / 1e6, out.c_str());
    return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        const auto opt = [&](const char* name, int def) {
            for (int i = 1; i + 1 < argc; ++i)
                if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
            return def;
        };
        if (mode == "--send" && argc >= 4) return do_send(argv[2], argv[3], std::max(1, opt("--repeat", 3)));
        if (mode == "--recv" && argc >= 4) return do_recv(argv[2], argv[3], std::max(1, opt("--count", 3)));
#ifdef STRIX_DS41_STATE_SYNTH
        if (mode == "--synth" && argc >= 4) return do_synth(static_cast<std::uint32_t>(std::stoul(argv[2])), argv[3]);
#endif
        std::fprintf(stderr, "usage: %s --send FILE HOST:PORT [--repeat N] | --recv PORT OUT_FILE [--count N]"
#ifdef STRIX_DS41_STATE_SYNTH
                             " | --synth POSITIONS OUT_FILE"
#endif
                             "\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
