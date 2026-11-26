#include "strix/serve/cohort_decode.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
thread_local std::size_t* allocation_count = nullptr;

#if defined(__GNUC__)
__attribute__((noinline))
#endif
void free_test_allocation(void* ptr) noexcept { std::free(ptr); }
}

void* operator new(std::size_t size) {
    if (allocation_count != nullptr) ++*allocation_count;
    if (void* ptr = std::malloc(size == 0 ? 1 : size)) return ptr;
    throw std::bad_alloc{};
}

void operator delete(void* ptr) noexcept { free_test_allocation(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { ::operator delete(ptr); }

namespace {
using namespace strix::serve;

struct CountAllocations {
    std::size_t count = 0;
    CountAllocations() { allocation_count = &count; }
    ~CountAllocations() { allocation_count = nullptr; }
};

constexpr std::uint64_t offset = 14695981039346656037ULL;
constexpr std::uint64_t append(std::uint64_t digest, std::uint32_t id) {
    return (digest ^ id) * 1099511628211ULL;
}

constexpr std::uint32_t predict(std::uint64_t digest) {
    return static_cast<std::uint32_t>(digest >> 31U) % 251U;
}

class Backend : public CohortBackend {
public:
    std::uint32_t lanes() const override { return 2; }
    std::uint32_t rows_per_forward() const override { return 2; }
    std::uint32_t max_positions() const override { return 512; }

    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) digest.at(lane) = offset;
        for (auto id : tokens) digest.at(lane) = append(digest.at(lane), id);
        prompt_rows += tokens.size();
    }

    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        if (groups.size() != tokens.size() || groups.size() != top.size())
            throw std::logic_error("expected one row per lane");
        pending_lanes = groups.size();
        for (std::size_t i = 0; i < pending_lanes; ++i) {
            if (groups[i].rows != 1 || groups[i].reads_scores) throw std::logic_error("expected a greedy row");
            const auto lane = groups[i].lane;
            pending[i] = {lane, append(digest.at(lane), tokens[i])};
            top[i] = predict(pending[i].second);
        }
    }

    void commit(std::span<const std::uint32_t> keep) override {
        if (keep.size() != pending_lanes) throw std::logic_error("wrong commit width");
        for (std::size_t i = 0; i < pending_lanes; ++i) {
            if (keep[i] != 1) throw std::logic_error("expected one committed row");
            const auto [lane, value] = pending[i];
            digest.at(lane) = value;
        }
        decoded_rows += pending_lanes;
    }

    void release(std::uint32_t) override {}

    std::array<std::uint64_t, 2> digest{offset, offset};
    std::size_t prompt_rows = 0, decoded_rows = 0;

private:
    std::array<std::pair<std::uint32_t, std::uint64_t>, 2> pending{};
    std::size_t pending_lanes = 0;
};

std::pair<std::size_t, std::size_t> measure(CohortDecoder& decoder, const std::size_t& total_rows, std::size_t steps) {
    const auto start = total_rows;
    CountAllocations counter;
    for (std::size_t i = 0; i < steps; ++i) decoder.step();
    return {counter.count, total_rows - start};
}

}  // namespace

int main() {
    Backend backend;
    CohortConfig config;
    config.prompt_slice = 4;
    config.prompt_share = 0;
    config.context_draft = false;
    config.max_draft = 0;
    CohortDecoder decoder(backend, config);
    decoder.set_fake_clock(1);

    std::array<CohortRequest, 2> requests{};
    for (std::size_t lane = 0; lane < requests.size(); ++lane) {
        auto& request = requests[lane];
        request.id = lane;
        request.max_tokens = 128;
        request.reuse = request.retain = false;
        for (std::uint32_t i = 0; i < 81; ++i)
            request.prompt.push_back(i + static_cast<std::uint32_t>(lane));
        decoder.submit(request);
    }
    for (unsigned i = 0; i < 4; ++i) decoder.step();  // admits both lanes and warms prompt scratch
    const auto [prefill_allocations, prompt_rows] = measure(decoder, backend.prompt_rows, 12);
    while (backend.decoded_rows < 190 && !decoder.idle()) decoder.step();
    // Both lanes are decoding and their result vectors have room for another eight IDs.
    const auto [decode_allocations, decoded_rows] = measure(decoder, backend.decoded_rows, 8);
    decoder.run();
    const auto results = decoder.take_finished();
    if (results.size() != requests.size()) return 1;
    for (const auto& result : results) {
        const auto& request = requests.at(static_cast<std::size_t>(result.id));
        auto state = offset;
        for (auto id : request.prompt) state = append(state, id);
        std::vector<std::uint32_t> expected;
        for (std::uint32_t i = 0; i < request.max_tokens; ++i) {
            const auto id = predict(state);
            expected.push_back(id);
            if (i + 1 < request.max_tokens) state = append(state, id);
        }
        if (result.ids != expected || result.finish != "length" || !result.error.empty() ||
            backend.digest.at(static_cast<std::size_t>(result.id)) != state) {
            std::cerr << "greedy IDs or committed digest differ for request " << result.id << '\n';
            return 1;
        }
    }
    std::cout << "prefill " << prefill_allocations << " allocations / " << prompt_rows << " tokens\n"
              << "decode " << decode_allocations << " allocations / " << decoded_rows << " tokens\n";
    return prompt_rows == 48 && decoded_rows == 16 && prefill_allocations == 0 && decode_allocations == 0 ? 0 : 1;
}
