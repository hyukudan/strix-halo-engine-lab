#include "strix/device/hip_gate.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/dist/transport.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// strix-dist-transport-soak: the pair transport alone, no model. Both ranks move tagged messages for a fixed time:
// every 4 KiB block starts with (magic, sender rank, message number, block number, message size) followed by bytes
// that are a function of the same. The receiver checks every block, so damage that reaches the caller is classified:
//   lost      a later block of the same message where block b was expected (b..found-1 never arrived)
//   repeated  an earlier block (a block arrived twice, or out of order)
//   foreign   a block of another message (the byte stream shifted by a loss in an earlier message)
//   damaged   no valid tag, or a valid tag with wrong bytes
// With the transport integrity on (default) nothing should reach the caller: the close line "usb4 integrity: ..." counts
// what the transport repaired. With STRIX_DIST_USB4_INTEGRITY=0 (raw stream) the first damage is printed and the soak
// stops (the stream is unusable after a loss); a stall shows as the transport's no-progress error with its state.
//   strix-dist-transport-soak --transport usb4auto --rank R [--seconds 120] [--sizes 4096,65536,...,67108864]
//       [--mode exchange|oneway|mixed] [--gpu-load]
// Output (rank 0 and rank 1 each): "soak: ..." summary line, "SOAK-DAMAGE ..." lines, rc 0 clean / 3 damage / 2 error.
namespace {

constexpr std::uint32_t soak_magic = 0x4b414f53U;  // "SOAK"
constexpr std::size_t block = 4096;

struct Tag {
    std::uint32_t magic, rank;
    std::uint64_t message, block, size;
};
static_assert(sizeof(Tag) == 32);

std::uint8_t fill(std::uint32_t rank, std::uint64_t message, std::uint64_t index) {
    std::uint64_t x = (message * 0x9e3779b97f4a7c15ULL) ^ (index * 0xbf58476d1ce4e5b9ULL) ^ rank;
    x ^= x >> 29U;
    return static_cast<std::uint8_t>(x & 0xffU);
}

void make(std::vector<std::byte>& out, std::uint32_t rank, std::uint64_t message) {
    for (std::size_t off = 0; off < out.size(); off += block) {
        const std::size_t len = std::min(block, out.size() - off);
        const Tag t{soak_magic, rank, message, off / block, out.size()};
        std::memcpy(out.data() + off, &t, std::min(len, sizeof t));
        for (std::size_t i = sizeof t; i < len; ++i) out[off + i] = static_cast<std::byte>(fill(rank, message, off + i));
    }
}

struct Stats {
    std::uint64_t messages = 0, bytes = 0, damage = 0;
};

// Checks a received message; prints the first damaged block of it (classified) and returns whether it was intact.
bool check(const std::vector<std::byte>& in, std::uint32_t from, std::uint64_t message, const char* how) {
    for (std::size_t off = 0; off < in.size(); off += block) {
        const std::size_t len = std::min(block, in.size() - off);
        const std::uint64_t index = off / block;
        bool ok = true;
        Tag t{};
        if (len >= sizeof t) {
            std::memcpy(&t, in.data() + off, sizeof t);
            ok = t.magic == soak_magic && t.rank == from && t.message == message && t.block == index && t.size == in.size();
        }
        for (std::size_t i = sizeof t; ok && i < len; ++i) ok = in[off + i] == static_cast<std::byte>(fill(from, message, off + i));
        if (len < sizeof t)
            for (std::size_t i = 0; ok && i < len; ++i) ok = in[off + i] == static_cast<std::byte>(fill(from, message, off + i));
        if (ok) continue;
        std::string kind = "damaged";
        std::string detail;
        if (t.magic == soak_magic && t.rank == from) {
            if (t.message == message && t.block > index) {
                kind = "lost";
                detail = " (blocks " + std::to_string(index) + ".." + std::to_string(t.block - 1U) + " missing, " +
                         std::to_string((t.block - index) * block) + " B)";
            } else if (t.message == message && t.block < index) {
                kind = "repeated";
                detail = " (block " + std::to_string(t.block) + " again)";
            } else if (t.message != message) {
                kind = "foreign";
                detail = " (block " + std::to_string(t.block) + " of message " + std::to_string(t.message) + ")";
            }
        }
        std::printf("SOAK-DAMAGE %s message %llu (%s, %zu B) at offset %zu (block %llu)%s\n", kind.c_str(),
                    static_cast<unsigned long long>(message), how, in.size(), off, static_cast<unsigned long long>(index), detail.c_str());
        std::fflush(stdout);
        return false;
    }
    return true;
}

// Device memory traffic on the side (a copy loop), like an engine keeping the GPU busy during the transfers.
void gpu_load(std::atomic<bool>& stop, std::atomic<std::uint64_t>& copies) {
    void* a = nullptr;
    void* b = nullptr;
    const std::size_t bytes = std::size_t{512} << 20U;
    if (hipMalloc(&a, bytes) != hipSuccess || hipMalloc(&b, bytes) != hipSuccess) {
        std::fprintf(stderr, "gpu load: hipMalloc failed\n");
        return;
    }
    while (!stop.load()) {
        if (hipMemcpy(b, a, bytes, hipMemcpyDeviceToDevice) != hipSuccess) break;
        copies.fetch_add(1);
    }
    (void)hipFree(a);
    (void)hipFree(b);
}

}  // namespace

int main(int argc, char** argv) try {
    strix::device::bind_hip_main(argc, argv);
    std::string transport = "usb4auto", mode = "mixed";
    std::uint32_t rank = 0;
    double seconds = 120.0;
    bool load = false;
    std::vector<std::size_t> sizes{4096, 65536, 1U << 20U, 8U << 20U, 16U << 20U, 64U << 20U};
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--transport") transport = next();
        else if (s == "--rank") rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--seconds") seconds = std::stod(next());
        else if (s == "--mode") mode = next();
        else if (s == "--gpu-load") load = true;
        else if (s == "--sizes") {
            sizes.clear();
            std::stringstream ss(next());
            for (std::string t; std::getline(ss, t, ',');) sizes.push_back(std::stoull(t));
        } else throw std::invalid_argument("unknown option " + s);
    }
    if (rank > 1U || sizes.empty() || (mode != "exchange" && mode != "oneway" && mode != "mixed"))
        throw std::invalid_argument("--rank 0|1, --sizes non-empty, --mode exchange|oneway|mixed");
    for (const auto n : sizes)
        if (n == 0U || n > (std::size_t{1} << 30U)) throw std::invalid_argument("--sizes: 1 .. 1 GiB");
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> copies{0};
    std::thread loader;
    if (load) loader = std::thread([&] { gpu_load(stop, copies); });
    auto t = strix::dist::open_pair(transport, rank, {std::chrono::seconds(60)}).host;
    std::printf("rank %u transport %s\n", rank, t->name().c_str());
    std::fflush(stdout);
    const std::uint32_t peer = 1U - rank;
    Stats st;
    int rc = 0;
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t message = 0;
    std::vector<std::byte> out, in;
    // Both ranks run the same schedule; rank 0 decides when to stop (a 1-byte go/stop word each round).
    try {
        for (std::uint64_t round = 0;; ++round) {
            std::byte go{1}, peer_go{1};
            if (rank == 0U) go = std::byte{static_cast<unsigned char>(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds)};
            t->exchange(std::span<const std::byte>(&go, 1), std::span<std::byte>(&peer_go, 1));
            if ((rank == 0U ? go : peer_go) == std::byte{0}) break;
            const auto size = sizes[round % sizes.size()];
            out.resize(size);
            in.resize(size);
            const bool exchange = mode == "exchange" || (mode == "mixed" && round % 2U == 0U);
            if (exchange) {
                make(out, rank, message);
                t->exchange(out, in);
                if (!check(in, peer, message, "exchange")) { ++st.damage; rc = 3; }
                st.bytes += 2U * size;
                ++message;
            } else {
                for (std::uint32_t sender = 0; sender < 2U; ++sender, ++message) {
                    if (sender == rank) {
                        make(out, rank, message);
                        t->send(out);
                    } else {
                        t->receive(in);
                        if (!check(in, peer, message, sender == 0U ? "one-way 0>1" : "one-way 1>0")) { ++st.damage; rc = 3; }
                    }
                    st.bytes += size;
                }
            }
            ++st.messages;
            if (rc == 3) break;  // a raw stream is unusable after damage; with integrity on this never happens
        }
    } catch (const std::exception& e) {
        std::printf("SOAK-ERROR %s\n", e.what());
        std::printf("SOAK-STATE %s\n", t->state().c_str());
        rc = 2;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    stop = true;
    if (loader.joinable()) loader.join();
    std::printf("soak: rank %u mode %s gpu-load %s (%llu copies) rounds %llu bytes %.3f GB in %.1f s (%.2f Gb/s both ways) damage %llu rc %d\n",
                rank, mode.c_str(), load ? "on" : "off", static_cast<unsigned long long>(copies.load()),
                static_cast<unsigned long long>(st.messages), static_cast<double>(st.bytes) / 1e9, s,
                static_cast<double>(st.bytes) * 8.0 / s / 1e9, static_cast<unsigned long long>(st.damage), rc);
    std::fflush(stdout);
    t.reset();  // the close line "usb4 integrity: ..." goes to stderr
    return rc;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
