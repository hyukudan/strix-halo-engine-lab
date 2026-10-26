// Host pipeline ordering and integrity on real HIP streams, with small socket
// buffers to force partial I/O in both directions. No USB4 device is needed.
#include "strix/dist/communicator.hpp"

#include <sys/socket.h>
#include <signal.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>

namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Mutate final : strix::dist::Transport {
    std::unique_ptr<strix::dist::Transport> inner;
    int mode = 0;
    unsigned calls = 0;
    Mutate(std::unique_ptr<strix::dist::Transport> t, int m) : inner(std::move(t)), mode(m) {}
    void send(std::span<const std::byte> b) override { inner->send(b); }
    void receive(std::span<std::byte> b) override { inner->receive(b); }
    std::string name() const override { return inner->name(); }
    void abort(const std::string& s) noexcept override { inner->abort(s); }
    void exchange(std::span<const std::byte> a, std::span<std::byte> b) override {
        inner->exchange(a, b);
        if (++calls == 2U && mode) {
            if (mode == 1) b[17] ^= std::byte{1};
            else {
                strix::dist::Trailer trailer;
                std::memcpy(&trailer, b.data() + b.size() - sizeof(trailer), sizeof(trailer));
                ++trailer.reserved[1];
                std::memcpy(b.data() + b.size() - sizeof(trailer), &trailer, sizeof(trailer));
            }
        }
    }
};
void run(int fault) {
    constexpr std::size_t capacity = 21U << 20U;
    std::vector<int> fds[2];
    for (unsigned k = 0; k < 4; ++k) {
        int pair[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) throw std::runtime_error("socketpair");
        for (unsigned r = 0; r < 2; ++r) {
            int size = 4096;
            ::setsockopt(pair[r], SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
            fds[r].push_back(pair[r]);
        }
    }
    std::exception_ptr errors[2];
    bool integrity_error[2] = {false, false};
    const auto rank = [&](unsigned r) {
        try {
            auto t = strix::dist::usb4_host_descriptors(fds[r], {"a", "b", "c", "d"}, {3, 3, 2, 2}, 4096,
                                                       std::chrono::milliseconds(2000));
            strix::dist::Communicator c({r, 2, 17, 29}, std::make_unique<Mutate>(std::move(t), fault), capacity);
            hipStream_t stream{};
            check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            void *local = nullptr, *remote = nullptr;
            check(hipMalloc(&local, capacity)); check(hipMalloc(&remote, capacity));
            std::vector<std::byte> expected(capacity), got(capacity);
            bool rejected = false;
            try {
                const std::vector<std::size_t> sizes = fault ? std::vector<std::size_t>{capacity} :
                    std::vector<std::size_t>{64, 122880, 163840, 1048577, 13U << 20U, capacity, 1048577, capacity};
                for (const auto n : sizes) {
                    for (std::size_t i = 0; i < n; ++i) expected[i] = static_cast<std::byte>((i * 131U + r * 17U) >> 3U);
                    check(hipMemcpyAsync(local, expected.data(), n, hipMemcpyHostToDevice, stream));
                    c.start(stream, local, n, {7, 3, n});
                    // Must be ordered AFTER every off-stream D2H read of local.
                    check(hipMemsetAsync(local, 0, n, stream));
                    c.finish(stream, remote);
                    check(hipMemcpyAsync(got.data(), remote, n, hipMemcpyDeviceToHost, stream));
                    check(hipStreamSynchronize(stream));
                    for (std::size_t i = 0; i < n; ++i)
                        if (got[i] != static_cast<std::byte>((i * 131U + (1U - r) * 17U) >> 3U))
                            throw std::runtime_error("hostpath bytes differ");
                }
            } catch (const std::exception& e) {
                const std::string s = e.what();
                integrity_error[r] = s.find("CRC mismatch") != std::string::npos || s.find("identity/offset") != std::string::npos;
                // Either rank may abort the sockets before the other finishes
                // reading. At least one rank must name the injected corruption.
                if (!fault || (!integrity_error[r] && s.find("all streams closed") == std::string::npos)) throw;
                rejected = true;
                try { c.exchange(stream, local, remote, 64, {}); throw std::runtime_error("poison not retained"); }
                catch (const std::exception& later) {
                    if (std::string(later.what()).find("stopped earlier") == std::string::npos) throw;
                }
            }
            if (fault && !rejected) throw std::runtime_error("fault not detected");
            check(hipStreamSynchronize(stream));
            check(hipFree(local)); check(hipFree(remote)); check(hipStreamDestroy(stream));
        } catch (...) { errors[r] = std::current_exception(); }
    };
    std::thread peer([&] { rank(1); }); rank(0); peer.join();
    for (auto e : errors) if (e) std::rethrow_exception(e);
    if (fault && !integrity_error[0] && !integrity_error[1]) throw std::runtime_error("integrity failure not reported");
}

// A failed transport must wake a CRC worker sleeping on its publication counter;
// destruction without finish must also cancel those waits before freeing memory.
void stalled(bool abandon) {
    int pair[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) throw std::runtime_error("socketpair");
    int size = 4096;
    ::setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
    hipStream_t stream{};
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    void *local = nullptr, *remote = nullptr;
    constexpr std::size_t bytes = 2U << 20U;
    check(hipMalloc(&local, bytes)); check(hipMalloc(&remote, bytes));
    check(hipMemsetAsync(local, 91, bytes, stream));
    ::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "1000", 1);
    const auto begin = std::chrono::steady_clock::now();
    {
        auto t = strix::dist::usb4_host_descriptors({pair[0]}, {"silent"}, {1}, 0, std::chrono::milliseconds(150));
        strix::dist::Communicator c({0, 2, 17, 29}, std::move(t), bytes);
        c.start(stream, local, bytes, {7, 3, 1});
        if (!abandon) {
            bool failed = false;
            try { c.finish(stream, remote); }
            catch (const std::exception&) { failed = true; }
            if (!failed) throw std::runtime_error("silent peer was accepted");
        }
    }
    ::unsetenv("STRIX_DIST_WAIT_TIMEOUT_MS");
    if (std::chrono::steady_clock::now() - begin > std::chrono::seconds(3))
        throw std::runtime_error("pipeline cancellation exceeded bound");
    check(hipStreamSynchronize(stream));
    check(hipFree(local)); check(hipFree(remote)); check(hipStreamDestroy(stream));
    ::close(pair[1]);
}
}
int main() try {
    ::signal(SIGPIPE, SIG_IGN); // socket test double, like dist_transport_tests
    ::setenv("GPU_MAX_HW_QUEUES", "16", 0);
    ::setenv("STRIX_DIST_HOST_CHUNK_BYTES", "1048576", 1);
    ::setenv("STRIX_DIST_CRC_PARALLEL", "1", 1);
    run(0); run(1); run(2);
    ::setenv("STRIX_DIST_HOST_CRC_OVERLAP", "1", 1);
    run(0); run(1); run(2);
    stalled(false); stalled(true);
    std::cout << "hostpath bytes, source lifetime, reuse, CRC/offset corruption and poison PASS\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
