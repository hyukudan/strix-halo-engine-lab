// STRIX_DIST_CRC_DEFER=1: ready-view exchanges send their CRC in the next trailer and check the received one after the
// consumer view is handed out. Checks: the bytes are the same as with inline CRCs over many generations mixed with
// ordinary exchanges; a corrupted payload is delivered but the NEXT exchange (ready view or ordinary) fails; without
// deferral the same corruption fails at once; ranks that disagree on the mode fail at their first exchange.
#include "strix/dist/communicator.hpp"
#include "strix/dist/staged_ready_kernels.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void expect(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
std::uint32_t pattern(unsigned rank, unsigned i) { return 0x9e3779b1U * (i + 1U) ^ (rank ? 0x35a5a55aU : 0x735a5aa5U); }

// Flips one payload bit of the received message of exchange number `at` (counted from 0).
class Corrupt final : public strix::dist::Transport {
    std::unique_ptr<strix::dist::Transport> inner_;
    unsigned at_, seen_ = 0;
public:
    Corrupt(std::unique_ptr<strix::dist::Transport> t, unsigned at) : inner_(std::move(t)), at_(at) {}
    void send(std::span<const std::byte> b) override { inner_->send(b); }
    void receive(std::span<std::byte> b) override { inner_->receive(b); }
    void exchange(std::span<const std::byte> a, std::span<std::byte> b) override {
        inner_->exchange(a, b);
        if (seen_++ == at_) b[b.size() / 3U] ^= std::byte{0x10};  // inside the payload, before the 128-byte trailer
    }
    std::string name() const override { return "memory-corrupt-one"; }
};

struct Pair {
    hipStream_t s[2]{};
    void* input[2]{};
    void* output[2]{};
    static constexpr std::size_t stride = 262144;
    Pair() {
        for (unsigned r = 0; r < 2; ++r) {
            check(hipStreamCreateWithFlags(&s[r], hipStreamNonBlocking));
            check(hipMalloc(&input[r], stride));
            check(hipMalloc(&output[r], stride * 32U));
        }
    }
    ~Pair() {
        for (unsigned r = 0; r < 2; ++r) {
            (void)hipFree(input[r]); (void)hipFree(output[r]); (void)hipStreamDestroy(s[r]);
        }
    }
};

// One exchange of generation i on both communicators: a ready view (consumer copies to output slot) or an ordinary
// exchange. Returns per rank whether its finish threw (and the message).
std::array<std::string, 2> step(std::array<strix::dist::Communicator*, 2> c, Pair& p, unsigned i, std::size_t size, bool ready) {
    const strix::dist::Tag tag{1, i, 41};
    for (unsigned r = 0; r < 2; ++r) {
        if (ready) {
            const auto v = c[r]->prepare_staged_ready(p.s[r], size);
            expect(v.data != nullptr, "ready producer unavailable");
            strix::dist::enqueue_staged_pattern(v, size, pattern(r, i), p.s[r]);
            c[r]->start_staged_ready(p.s[r], size, tag);
        } else {
            const auto value = static_cast<unsigned char>(pattern(r, i));
            check(hipMemsetAsync(p.input[r], value, size, p.s[r]));
            c[r]->start(p.s[r], p.input[r], size, tag);
        }
    }
    std::array<std::string, 2> failed{};
    for (unsigned r = 0; r < 2; ++r) {
        auto* dst = static_cast<unsigned char*>(p.output[r]) + (i % 32U) * Pair::stride;
        try {
            if (ready) {
                const auto v = c[r]->finish_staged_ready();
                strix::dist::enqueue_staged_consume(v, dst, size, p.s[r]);
                c[r]->release_staged_ready(p.s[r]);
            } else {
                c[r]->finish(p.s[r], dst);
            }
        } catch (const std::exception& e) {
            failed[r] = e.what();
        }
    }
    return failed;
}

void exact_many() {
    ::setenv("STRIX_DIST_CRC_DEFER", "1", 1);
    auto pair = strix::dist::memory_pair();
    Pair p;
    {
        strix::dist::Communicator a({0, 2, 7, 1}, std::move(pair.first), Pair::stride);
        strix::dist::Communicator b({1, 2, 7, 1}, std::move(pair.second), Pair::stride);
        const std::array<std::size_t, 8> grid{16, 76, 4096, 20480, 65536, 122880, 163840, Pair::stride};
        std::array<std::size_t, 32> sizes{};
        std::array<bool, 32> was_ready{};
        for (unsigned i = 0; i < 512; ++i) {
            const auto size = grid[i % grid.size()];
            const bool ready = i % 5U != 4U;  // every 5th one ordinary: it carries and checks the previous deferred CRC
            sizes[i % 32U] = size;
            was_ready[i % 32U] = ready;
            const auto failed = step({&a, &b}, p, i, size, ready);
            expect(failed[0].empty() && failed[1].empty(), "unexpected failure at " + std::to_string(i) + ": " + failed[0] + failed[1]);
            if (i % 32U == 31U) {
                std::vector<unsigned char> got(Pair::stride * 32U);
                for (unsigned r = 0; r < 2; ++r) {
                    check(hipStreamSynchronize(p.s[r]));
                    check(hipMemcpy(got.data(), p.output[r], got.size(), hipMemcpyDeviceToHost));
                    for (unsigned j = 0; j < 32U; ++j) {
                        const auto value = pattern(1U - r, i - 31U + j);
                        for (std::size_t k = 0; k < sizes[j]; ++k) {
                            const auto want = was_ready[j] ? static_cast<unsigned char>(value >> ((k & 3U) * 8U)) : static_cast<unsigned char>(value);
                            expect(got[j * Pair::stride + k] == want, "deferred-CRC data differs at generation " + std::to_string(i - 31U + j));
                        }
                    }
                }
            }
        }
        // ends with an ordinary exchange: every deferred CRC was checked
        expect(step({&a, &b}, p, 512, 64, false)[1].empty(), "final ordinary exchange failed");
        a.quiesce(p.s[0]); b.quiesce(p.s[1]);
        expect(a.stats().exchanges == 513 && b.stats().exchanges == 513, "lost exchange");
    }
    ::unsetenv("STRIX_DIST_CRC_DEFER");
    std::cout << "crc defer: 512 ready/ordinary generations 16..262144 B, bytes identical PASS\n";
}

// Corrupt rank 1's received payload of exchange `at`; `next_ready`: whether the following exchange is a ready view.
void injection(bool defer, bool next_ready) {
    if (defer) ::setenv("STRIX_DIST_CRC_DEFER", "1", 1);
    else ::setenv("STRIX_DIST_CRC_DEFER", "0", 1);
    constexpr unsigned at = 6;
    auto pair = strix::dist::memory_pair();
    Pair p;
    std::string where;
    {
        strix::dist::Communicator a({0, 2, 5, 1}, std::move(pair.first), Pair::stride);
        strix::dist::Communicator b({1, 2, 5, 1}, std::make_unique<Corrupt>(std::move(pair.second), at), Pair::stride);
        for (unsigned i = 0; i <= at + 1U && where.empty(); ++i) {
            const auto failed = step({&a, &b}, p, i, 20480, i == at + 1U ? next_ready : true);
            expect(failed[0].empty(), "rank 0 failed: " + failed[0]);
            if (!failed[1].empty()) {
                where = std::to_string(i) + ": " + failed[1];
                expect(failed[1].find("CRC mismatch") != std::string::npos, "rank 1 failed for another reason: " + failed[1]);
                expect(i == (defer ? at + 1U : at), "corruption caught at exchange " + where);
            }
        }
        expect(!where.empty(), "corrupted payload never rejected");
        bool poisoned = false;
        try { (void)b.prepare_staged_ready(p.s[1], 64); } catch (const std::runtime_error&) { poisoned = true; }
        expect(poisoned, "failed exchange did not stop the communicator");
        check(hipStreamSynchronize(p.s[0]));
        check(hipStreamSynchronize(p.s[1]));
    }
    ::unsetenv("STRIX_DIST_CRC_DEFER");
    std::cout << "crc defer: injection " << (defer ? "deferred" : "inline") << ", next " << (next_ready ? "ready" : "ordinary")
              << " -> rejected at exchange " << where << " PASS\n";
}

void mode_mismatch() {
    auto pair = strix::dist::memory_pair();
    Pair p;
    {
        ::setenv("STRIX_DIST_CRC_DEFER", "1", 1);
        strix::dist::Communicator a({0, 2, 3, 1}, std::move(pair.first), Pair::stride);
        ::setenv("STRIX_DIST_CRC_DEFER", "0", 1);
        strix::dist::Communicator b({1, 2, 3, 1}, std::move(pair.second), Pair::stride);
        const auto failed = step({&a, &b}, p, 0, 4096, true);
        expect(!failed[0].empty() || !failed[1].empty(), "ranks with different CRC modes were not rejected");
        expect((failed[0] + failed[1]).find("STRIX_DIST_CRC_DEFER") != std::string::npos, "mode mismatch not named: " + failed[0] + failed[1]);
        check(hipStreamSynchronize(p.s[0]));
        check(hipStreamSynchronize(p.s[1]));
    }
    std::cout << "crc defer: ranks with different modes fail at the first exchange PASS\n";
}
} // namespace

int main() try {
    ::setenv("STRIX_DIST_STAGED_READY_BYTES", "262144", 1);
    exact_many();
    injection(true, true);
    injection(true, false);
    injection(false, true);
    mode_mismatch();
    std::cout << "crc defer tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "crc defer tests: " << e.what() << '\n';
    return 1;
}
