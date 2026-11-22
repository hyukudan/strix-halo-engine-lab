#pragma once

#include "strix/device/runtime.hpp"
#include "strix/head_argmax.hpp"
#if defined(__HIPCC__)
#include "strix/hip_generation_output.hpp"
#else
#include "cuda/core/head_argmax.cuh"
#endif

#include <algorithm>
#include <array>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix::core {

#if defined(__HIPCC__)

// Greedy over a contiguous logits matrix in one stream. The shared selector
// handles rows in bounded launch groups and returns mapped replies. Read a
// reply only after the producer and all groups have completed on that stream.
class ResidentGreedy {
    struct FreePinned {
        void operator()(HipGreedyRowsReply* p) const noexcept { (void)hipHostFree(p); }
    };
public:
    explicit ResidentGreedy(unsigned capacity, unsigned vocabulary)
        : capacity_(capacity), vocabulary_(vocabulary), partials_(static_cast<std::size_t>(
              hip_greedy_rows_scratch_bytes(std::min(capacity, hip_greedy_rows_max), vocabulary_))) {
        if (!capacity || !vocabulary) throw std::invalid_argument("invalid resident greedy capacity");
        replies_.reserve((capacity - 1U) / hip_greedy_rows_max + 1U);
        for (unsigned base = 0; base < capacity; base += hip_greedy_rows_max) {
            const auto count = std::min(capacity - base, hip_greedy_rows_max);
            void* address{};
            device::native::check(hipHostMalloc(&address, std::size_t{count} * sizeof(HipGreedyRowsReply),
                                         hipHostMallocMapped | hipHostMallocCoherent), "allocate resident HIP greedy replies");
            replies_.emplace_back(static_cast<HipGreedyRowsReply*>(address));
        }
    }

    void enqueue(std::span<const float> logits, device::native::Stream stream) {
        submitted_rows_ = 0;
        const auto rows = logits.size() / vocabulary_;
        if (!rows || rows > capacity_ || logits.size() % vocabulary_ != 0)
            throw std::invalid_argument("invalid resident HIP greedy logits shape");
        const ExecutionBufferAddress scratch{{}, reinterpret_cast<std::uintptr_t>(partials_.data()),
                                             partials_.span().size_bytes(), 0U};
        const auto row_bytes = std::size_t{vocabulary_} * sizeof(float);
        std::array<ExecutionBufferAddress, hip_greedy_rows_max> addresses{};
        for (std::size_t base = 0; base < rows; base += addresses.size()) {
            const auto count = std::min(addresses.size(), rows - base);
            for (std::size_t i = 0; i < count; ++i) {
                const auto* row = logits.data() + (base + i) * vocabulary_;
                addresses[i] = {{}, reinterpret_cast<std::uintptr_t>(row), row_bytes, 0U};
            }
            enqueue_hip_greedy_rows(std::span(addresses).first(count), vocabulary_, scratch,
                                    replies_[base / addresses.size()].get(), stream);
        }
        submitted_rows_ = rows;
    }
    [[nodiscard]] unsigned token(std::size_t row = 0) const {
        if (row >= submitted_rows_) throw std::out_of_range("invalid resident HIP greedy reply row");
        const auto reply = replies_[row / hip_greedy_rows_max].get()[row % hip_greedy_rows_max];
        return reply.status != 0U ? head_argmax::kNone : reply.token;
    }

private:
    unsigned capacity_, vocabulary_;
    std::size_t submitted_rows_ = 0;
    device::native::Buffer<std::uint8_t> partials_;
    std::vector<std::unique_ptr<HipGreedyRowsReply, FreePinned>> replies_;
};


#else

// The caller reads the pinned index rows after the same stream fence as the
// forward status. The native CUDA picker supplies first-max and finite checks.
class ResidentGreedy {
    struct FreePinned {
        void operator()(std::uint32_t* p) const noexcept { (void)cudaFreeHost(p); }
    };

public:
    explicit ResidentGreedy(unsigned capacity, unsigned vocabulary)
        : capacity_(capacity), vocabulary_(vocabulary), chunks_(head_argmax::default_chunks(vocabulary)),
          partials_(std::size_t{capacity} * chunks_), device_tokens_(capacity) {
        if (!capacity || !vocabulary) throw std::invalid_argument("invalid resident greedy capacity");
        void* address{};
        device::native::check(cudaMallocHost(&address, std::size_t{capacity} * sizeof(std::uint32_t)),
                              "allocate resident CUDA greedy replies");
        replies_.reset(static_cast<std::uint32_t*>(address));
    }

    void enqueue(std::span<const float> logits, device::native::Stream stream) {
        namespace argmax = cuda::core::head_argmax;
        submitted_rows_ = 0;
        const auto rows = logits.size() / vocabulary_;
        if (!rows || rows > capacity_ || logits.size() % vocabulary_ != 0)
            throw std::invalid_argument("invalid resident CUDA greedy logits shape");
        device::native::check(argmax::pick_rows(
            {logits.data(), 0U, vocabulary_, static_cast<unsigned>(rows), 0U, chunks_, partials_.data()},
            argmax::EmitIndexOrNone{device_tokens_.data()}, stream), "enqueue resident CUDA greedy pick");
        device::native::copy(replies_.get(), device_tokens_.data(), rows * sizeof(std::uint32_t),
                             device::native::device_to_host, stream);
        submitted_rows_ = rows;
    }

    [[nodiscard]] unsigned token(std::size_t row = 0) const {
        if (row >= submitted_rows_) throw std::out_of_range("invalid resident CUDA greedy reply row");
        return replies_.get()[row];
    }

    // Raw buffers for the optional CUDA step-program route (row 0).
    [[nodiscard]] auto* step_program_partials() noexcept { return partials_.data(); }
    [[nodiscard]] auto* step_program_device_token() noexcept { return device_tokens_.data(); }
    [[nodiscard]] auto* step_program_reply() noexcept { return replies_.get(); }

private:
    unsigned capacity_, vocabulary_, chunks_;
    std::size_t submitted_rows_ = 0;
    device::native::Buffer<head_argmax::Pick> partials_;
    device::native::Buffer<std::uint32_t> device_tokens_;
    std::unique_ptr<std::uint32_t, FreePinned> replies_;
};

#endif

} // namespace strix::core
