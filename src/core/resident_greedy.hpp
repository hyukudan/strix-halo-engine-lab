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
#include <type_traits>
#include <utility>
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
        void* mapped{};
        device::native::check(hipHostGetDevicePointer(&mapped, replies_.front().get(), 0),
                              "map resident HIP greedy reply");
        device_reply_ = static_cast<HipGreedyRowsReply*>(mapped);
    }

    void enqueue(std::span<const float> logits, device::native::Stream stream) {
        submitted_rows_ = 0;
        const auto rows = logits.size() / vocabulary_;
        if (!rows || rows > capacity_ || logits.size() % vocabulary_ != 0)
            throw std::invalid_argument("invalid resident HIP greedy logits shape");
        enqueue_rows(logits, static_cast<unsigned>(rows), vocabulary_, stream);
    }

    // Rows start every row_stride floats (row_stride >= vocabulary), e.g. a padded pitch that keeps each
    // row 16-byte aligned for the dense head that writes it. Picks equal those of the contiguous rows.
    void enqueue_rows(std::span<const float> logits, unsigned rows, std::size_t row_stride,
                      device::native::Stream stream) {
        submitted_rows_ = 0;
        if (rows > capacity_ || !::strix::head_argmax::strided_rows_fit(logits.size(), rows, row_stride, vocabulary_))
            throw std::invalid_argument("invalid resident HIP greedy logits shape");
        const ExecutionBufferAddress scratch{{}, reinterpret_cast<std::uintptr_t>(partials_.data()),
                                             partials_.span().size_bytes(), 0U};
        const auto row_bytes = std::size_t{vocabulary_} * sizeof(float);
        std::array<ExecutionBufferAddress, hip_greedy_rows_max> addresses{};
        for (std::size_t base = 0; base < rows; base += addresses.size()) {
            const auto count = std::min(addresses.size(), rows - base);
            for (std::size_t i = 0; i < count; ++i) {
                const auto* row = logits.data() + (base + i) * row_stride;
                addresses[i] = {{}, reinterpret_cast<std::uintptr_t>(row), row_bytes, 0U};
            }
            enqueue_hip_greedy_rows(std::span(addresses).first(count), vocabulary_, scratch,
                                    replies_[base / addresses.size()].get(), stream);
        }
        submitted_rows_ = rows;
    }
    // Row 0 picked by another device producer that writes the same token and status as enqueue over the full row
    // (e.g. the EXACT screened head, core/hip_screened_head.hpp): `produce` enqueues it on the stream with the
    // device-visible reply of row 0; token(0) then reads it after the same stream fence.
    template <class Producer>
    void enqueue_produced(Producer&& produce) {
        submitted_rows_ = 0;
        produce_checked(std::forward<Producer>(produce), device_reply_, "enqueue produced HIP greedy reply");
        submitted_rows_ = 1;
    }
    // Rows 0..rows-1 picked by one device producer (e.g. the lanes form of the EXACT screened head): `produce`
    // enqueues them with the device-visible replies of the first launch group (rows <= hip_greedy_rows_max), row r
    // at replies[r]; token(r) then reads them after the same stream fence.
    template <class Producer>
    void enqueue_produced_rows(unsigned rows, Producer&& produce) {
        submitted_rows_ = 0;
        if (rows == 0U || rows > capacity_ || rows > hip_greedy_rows_max)
            throw std::invalid_argument("invalid resident HIP greedy produced rows");
        produce_checked(std::forward<Producer>(produce), device_reply_, "enqueue produced HIP greedy replies");
        submitted_rows_ = rows;
    }

    // A producer may return its launch status (cudaError_t / hipError_t): a rejected launch must not publish the
    // reply of an earlier submission, so the status is checked (throws) before the row is marked submitted.
    template <class Producer, class Target>
    static void produce_checked(Producer&& produce, Target target, const char* operation) {
        if constexpr (std::is_void_v<std::invoke_result_t<Producer&&, Target>>)
            std::forward<Producer>(produce)(target);
        else
            device::native::check(std::forward<Producer>(produce)(target), operation);
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
    HipGreedyRowsReply* device_reply_{};
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
        submitted_rows_ = 0;
        const auto rows = logits.size() / vocabulary_;
        if (!rows || rows > capacity_ || logits.size() % vocabulary_ != 0)
            throw std::invalid_argument("invalid resident CUDA greedy logits shape");
        enqueue_rows(logits, static_cast<unsigned>(rows), vocabulary_, stream);
    }

    // Rows start every row_stride floats (row_stride >= vocabulary), e.g. a padded pitch that keeps each
    // row 16-byte aligned for the dense head that writes it. Picks equal those of the contiguous rows.
    void enqueue_rows(std::span<const float> logits, unsigned rows, std::size_t row_stride,
                      device::native::Stream stream) {
        namespace argmax = cuda::core::head_argmax;
        submitted_rows_ = 0;
        if (rows > capacity_ || !::strix::head_argmax::strided_rows_fit(logits.size(), rows, row_stride, vocabulary_))
            throw std::invalid_argument("invalid resident CUDA greedy logits shape");
        device::native::check(argmax::pick_rows(
            {logits.data(), row_stride, vocabulary_, rows, 0U, chunks_, partials_.data()},
            argmax::EmitIndexOrNone{device_tokens_.data()}, stream), "enqueue resident CUDA greedy pick");
        device::native::copy(replies_.get(), device_tokens_.data(), rows * sizeof(std::uint32_t),
                             device::native::device_to_host, stream);
        submitted_rows_ = rows;
    }

    // Row 0 picked by another device producer that writes the same index-or-none word as enqueue over the full row
    // (e.g. the EXACT screened head, cuda/core/head_screened.cuh): `produce` enqueues it on the stream into the device
    // token of row 0; the pinned reply copy follows on the same stream and token(0) reads it after the stream fence.
    template <class Producer>
    void enqueue_produced(Producer&& produce, device::native::Stream stream) {
        submitted_rows_ = 0;
        produce_checked(std::forward<Producer>(produce), device_tokens_.data(), "enqueue produced CUDA greedy pick");
        device::native::copy(replies_.get(), device_tokens_.data(), sizeof(std::uint32_t),
                             device::native::device_to_host, stream);
        submitted_rows_ = 1;
    }

    // Rows 0..rows-1 picked by another device producer that writes the index-or-none word of row r at device_tokens[r]
    // (e.g. the lanes form of the EXACT screened head); one pinned reply copy follows on the same stream and token(r)
    // reads row r after the stream fence.
    template <class Producer>
    void enqueue_produced_rows(unsigned rows, Producer&& produce, device::native::Stream stream) {
        submitted_rows_ = 0;
        if (!rows || rows > capacity_) throw std::invalid_argument("invalid resident CUDA greedy produced rows");
        produce_checked(std::forward<Producer>(produce), device_tokens_.data(), "enqueue produced CUDA greedy rows");
        device::native::copy(replies_.get(), device_tokens_.data(), rows * sizeof(std::uint32_t),
                             device::native::device_to_host, stream);
        submitted_rows_ = rows;
    }

    // A producer may return its launch status (cudaError_t / hipError_t): a rejected launch must not publish the
    // reply of an earlier submission, so the status is checked (throws) before the row is marked submitted.
    template <class Producer, class Target>
    static void produce_checked(Producer&& produce, Target target, const char* operation) {
        if constexpr (std::is_void_v<std::invoke_result_t<Producer&&, Target>>)
            std::forward<Producer>(produce)(target);
        else
            device::native::check(std::forward<Producer>(produce)(target), operation);
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
