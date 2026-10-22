#include "strix/dist/communicator.hpp"
#include "strix/dist/staged_ready_kernels.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void expect(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
std::uint32_t pattern(unsigned rank, unsigned i) { return 0x9e3779b1U * (i + 1U) ^ (rank ? 0x35a5a55aU : 0x735a5aa5U); }

void stress() {
    constexpr std::size_t stride = 262144, batch = 32;
    auto pair = strix::dist::memory_pair();
    hipStream_t s[2]{};
    void* output[2]{};
    for (unsigned r = 0; r < 2; ++r) {
        check(hipStreamCreateWithFlags(&s[r], hipStreamNonBlocking));
        check(hipMalloc(&output[r], stride * batch));
    }
    {
        strix::dist::Communicator a({0, 2, 7, 1}, std::move(pair.first), stride + 1);
        strix::dist::Communicator b({1, 2, 7, 1}, std::move(pair.second), stride + 1);
        std::array<strix::dist::Communicator*, 2> c{&a, &b};
        expect(!a.prepare_staged_ready(s[0], stride + 1).data, "oversize must fall back");
        std::array<std::size_t, batch> sizes{};
        const std::array<std::size_t, 12> grid{1, 3, 15, 16, 17, 76, 4096, 20480, 65536, 122880, 163840, stride};
        for (unsigned i = 0; i < 512; ++i) {
            const auto size = grid[i % grid.size()];
            const auto slot = i % batch;
            sizes[slot] = size;
            for (unsigned r = 0; r < 2; ++r) {
                const auto v = c[r]->prepare_staged_ready(s[r], size);
                expect(v.data && v.flag && v.token, "ready producer unavailable");
                strix::dist::enqueue_staged_pattern(v, size, pattern(r, i), s[r]);
            }
            // No producer event or stream synchronization before either start.
            const strix::dist::Tag tag{1, i, 37};
            a.start_staged_ready(s[0], size, tag);
            b.start_staged_ready(s[1], size, tag);
            for (unsigned r = 0; r < 2; ++r) {
                const auto v = c[r]->finish_staged_ready();
                bool rejected = false;
                try { (void)c[r]->prepare_staged_ready(s[r], size); } catch (const std::logic_error&) { rejected = true; }
                expect(rejected, "unreleased consumer must block a new view");
                strix::dist::enqueue_staged_consume(v, static_cast<unsigned char*>(output[r]) + slot * stride, size, s[r]);
                c[r]->release_staged_ready(s[r]);
            }
            if (slot + 1U == batch) {
                std::vector<unsigned char> got(stride * batch);
                for (unsigned r = 0; r < 2; ++r) {
                    check(hipStreamSynchronize(s[r]));
                    check(hipMemcpy(got.data(), output[r], got.size(), hipMemcpyDeviceToHost));
                    for (unsigned j = 0; j < batch; ++j) {
                        const auto value = pattern(1U - r, i + 1U - static_cast<unsigned>(batch) + j);
                        for (std::size_t k = 0; k < sizes[j]; ++k)
                            expect(got[j * stride + k] == static_cast<unsigned char>(value >> ((k & 3U) * 8U)),
                                   "staged data corrupted or overwritten before consumer release");
                    }
                }
            }
        }
        a.quiesce(s[0]); b.quiesce(s[1]);
        expect(a.stats().exchanges == 512 && b.stats().exchanges == 512, "lost staged exchange");
    }
    for (unsigned r = 0; r < 2; ++r) { check(hipFree(output[r])); check(hipStreamDestroy(s[r])); }
    std::cout << "staged readiness: 512 generations, 1..262144 B, async producer/consumer and reuse PASS\n";
}

void mixed_fallback() {
    // Alternate ready views, event-based fused fallback and ordinary copies on
    // the same communicator, without draining consumers between generations.
    ::setenv("STRIX_DIST_STAGING", "ready", 1);
    constexpr std::size_t stride = 262145, batch = 32;
    auto pair = strix::dist::memory_pair();
    hipStream_t s[2]{};
    void* input[2]{};
    void* output[2]{};
    for (unsigned r = 0; r < 2; ++r) {
        check(hipStreamCreateWithFlags(&s[r], hipStreamNonBlocking));
        check(hipMalloc(&input[r], stride));
        check(hipMalloc(&output[r], stride * batch));
    }
    {
        strix::dist::Communicator a({0, 2, 9, 1}, std::move(pair.first), stride);
        strix::dist::Communicator b({1, 2, 9, 1}, std::move(pair.second), stride);
        std::array<strix::dist::Communicator*, 2> c{&a, &b};
        std::array<std::size_t, batch> sizes{};
        for (unsigned i = 0; i < 128; ++i) {
            const unsigned mode = i % 3U;
            const auto size = mode == 1U ? stride : std::size_t{20480 + 143360 * (i & 1U)};
            const auto slot = i % batch;
            sizes[slot] = size;
            for (unsigned r = 0; r < 2; ++r) {
                const auto value = static_cast<unsigned char>(17U + 3U * i + r);
                const strix::dist::Tag tag{2, i, 38};
                if (mode == 0U) {
                    const auto v = c[r]->prepare_staged_ready(s[r], size);
                    expect(v.data, "mixed ready producer unavailable");
                    strix::dist::enqueue_staged_pattern(v, size, 0x01010101U * value, s[r]);
                    c[r]->start_staged_ready(s[r], size, tag);
                } else if (mode == 1U) {
                    expect(!c[r]->prepare_staged_ready(s[r], size).data, "oversize must select fused fallback");
                    auto* v = c[r]->send_staging(size);
                    expect(v, "ready mode must retain event-based fused fallback");
                    check(hipMemsetAsync(v, value, size, s[r]));
                    c[r]->start_staged(s[r], size, tag);
                } else {
                    check(hipMemsetAsync(input[r], value, size, s[r]));
                    c[r]->start(s[r], input[r], size, tag);
                }
            }
            for (unsigned r = 0; r < 2; ++r) {
                auto* dst = static_cast<unsigned char*>(output[r]) + slot * stride;
                if (mode == 0U) {
                    const auto v = c[r]->finish_staged_ready();
                    expect(!c[r]->send_staging(size), "held ready view admitted fused producer");
                    strix::dist::enqueue_staged_consume(v, dst, size, s[r]);
                    c[r]->release_staged_ready(s[r]);
                } else if (mode == 1U) {
                    const auto* v = c[r]->finish_staged(s[r]);
                    bool rejected = false;
                    try { (void)c[r]->prepare_staged_ready(s[r], 76); }
                    catch (const std::logic_error&) { rejected = true; }
                    expect(rejected, "held fused view admitted ready producer");
                    check(hipMemcpyAsync(dst, v, size, hipMemcpyDeviceToDevice, s[r]));
                    c[r]->release_staged(s[r]);
                } else {
                    c[r]->finish(s[r], dst);
                }
            }
            if (slot + 1U == batch) {
                std::vector<unsigned char> got(stride * batch);
                for (unsigned r = 0; r < 2; ++r) {
                    check(hipStreamSynchronize(s[r]));
                    check(hipMemcpy(got.data(), output[r], got.size(), hipMemcpyDeviceToHost));
                    for (unsigned j = 0; j < batch; ++j) {
                        const auto gen = i + 1U - static_cast<unsigned>(batch) + j;
                        const auto value = static_cast<unsigned char>(17U + 3U * gen + 1U - r);
                        expect(std::all_of(got.begin() + j * stride, got.begin() + j * stride + sizes[j],
                                           [value](unsigned char v) { return v == value; }),
                               "mixed fallback data corrupted or RX reused before consumer");
                    }
                }
            }
        }
        a.quiesce(s[0]); b.quiesce(s[1]);
    }
    for (unsigned r = 0; r < 2; ++r) {
        check(hipFree(input[r])); check(hipFree(output[r])); check(hipStreamDestroy(s[r]));
    }
    ::unsetenv("STRIX_DIST_STAGING");
    std::cout << "staged readiness: 128 mixed ready/fused/copy generations, oversized fallback and RX lifetime PASS\n";
}

void producer_failures() {
    ::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "30", 1);
    for (const bool wrong : {false, true}) {
        auto pair = strix::dist::memory_pair();
        hipStream_t stream{};
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        {
            strix::dist::Communicator a({0, 2, 3, 1}, std::move(pair.first), 4096);
            const auto v = a.prepare_staged_ready(stream, 76);
            expect(v.data, "failure test view unavailable");
            if (wrong) {
                const auto bad = v.token + 99U;
                check(hipMemcpy(v.flag, &bad, sizeof(bad), hipMemcpyHostToDevice));
            }
            a.start_staged_ready(stream, 76, {1, 0, 0});
            bool failed = false;
            try { (void)a.finish_staged_ready(); } catch (const std::runtime_error&) { failed = true; }
            expect(failed, "missing/wrong producer flag accepted");
            bool poisoned = false;
            try { (void)a.prepare_staged_ready(stream, 76); } catch (const std::runtime_error&) { poisoned = true; }
            expect(poisoned, "failed readiness did not poison communicator");
        }
        check(hipStreamDestroy(stream));
    }
    ::unsetenv("STRIX_DIST_WAIT_TIMEOUT_MS");
    std::cout << "staged readiness: silent/wrong-generation producer fails closed PASS\n";
}

class Corrupt final : public strix::dist::Transport {
    std::unique_ptr<strix::dist::Transport> inner_;
public:
    explicit Corrupt(std::unique_ptr<strix::dist::Transport> t) : inner_(std::move(t)) {}
    void send(std::span<const std::byte> b) override { inner_->send(b); }
    void receive(std::span<std::byte> b) override { inner_->receive(b); b[0] ^= std::byte{1}; }
    void exchange(std::span<const std::byte> a, std::span<std::byte> b) override { inner_->exchange(a, b); b[0] ^= std::byte{1}; }
    std::string name() const override { return "memory-corrupt-test"; }
};
void crc_failure() {
    auto pair = strix::dist::memory_pair();
    hipStream_t s[2]{};
    for (auto& stream : s) check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    {
        strix::dist::Communicator a({0, 2, 3, 1}, std::make_unique<Corrupt>(std::move(pair.first)), 4096);
        strix::dist::Communicator b({1, 2, 3, 1}, std::move(pair.second), 4096);
        auto va = a.prepare_staged_ready(s[0], 76), vb = b.prepare_staged_ready(s[1], 76);
        strix::dist::enqueue_staged_pattern(va, 76, 0xabcdef12U, s[0]);
        strix::dist::enqueue_staged_pattern(vb, 76, 0x735af123U, s[1]);
        a.start_staged_ready(s[0], 76, {1, 0, 0}); b.start_staged_ready(s[1], 76, {1, 0, 0});
        bool failed = false;
        try { (void)a.finish_staged_ready(); } catch (const std::runtime_error&) { failed = true; }
        expect(failed, "CRC corruption exposed to a staged consumer");
        (void)b.finish_staged_ready(); // no consumer launch: destructor must still drain safely
    }
    for (auto stream : s) check(hipStreamDestroy(stream));
    std::cout << "staged readiness: payload CRC corruption rejected before consumer PASS\n";
}
}
int main() try {
    ::setenv("STRIX_DIST_STAGED_READY_BYTES", "262144", 1);
    stress(); mixed_fallback(); producer_failures(); crc_failure();
    std::cout << "staged readiness tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "staged readiness tests: " << e.what() << '\n';
    return 1;
}
