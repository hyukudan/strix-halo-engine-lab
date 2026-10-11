#pragma once

#include "strix/execution_memory.hpp"
#include "strix/runtime_allocator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix {

using HostUploadHandle = std::uint64_t;

enum class HostUploadStatus {
    pending,
    complete,
    canceled,
    failed_retained,
};

class HostUploadSource {
public:
    virtual ~HostUploadSource() = default;
    // Storage extent, address and allocation generation are immutable for this
    // object's lifetime. Only its pool writes bytes, before preparing a copy.
    [[nodiscard]] virtual std::span<std::byte> writable_bytes() noexcept = 0;
    [[nodiscard]] virtual std::span<const std::byte> bytes() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t allocation_generation() const noexcept = 0;
};

class HostUploadBackend {
public:
    virtual ~HostUploadBackend() = default;

    [[nodiscard]] virtual std::shared_ptr<HostUploadSource> allocate_source(
        std::uint64_t bytes, std::uint64_t allocation_generation) = 0;

    // Prepare must not enqueue a copy. It retains the source before submit,
    // including when submit later throws after work may have been queued.
    [[nodiscard]] virtual HostUploadHandle prepare(
        const ExecutionBufferAddress& destination,
        std::shared_ptr<const HostUploadSource> source,
        std::uint64_t bytes) = 0;
    virtual void submit(HostUploadHandle handle) = 0;
    // Only complete/canceled prove that no operation can access the source.
    // An error or return from cancel does not provide that proof.
    [[nodiscard]] virtual HostUploadStatus poll(HostUploadHandle handle) = 0;
    virtual void cancel(HostUploadHandle handle) = 0;
    virtual void retire(HostUploadHandle handle) = 0;

    // The backend outlives its pools and retains unretired sources on pool
    // destruction. Backend teardown must drain or quarantine pending sources.
    // The caller retains the destination allocation until quiescence too.
};

struct HostUploadPolicy {
    std::uint32_t maximum_slots = 2;
    std::uint64_t bytes_per_slot = 64ULL * 1024ULL;
    std::uint64_t maximum_total_bytes = 128ULL * 1024ULL;
    std::uint64_t maximum_polls = 4096;
};

struct HostUploadTicket {
    std::uint64_t pool_generation = 0;
    std::uint32_t slot = 0;
    std::uint64_t source_allocation_generation = 0;
    std::uint64_t sequence = 0;
};

// One serialized owner governs a pool/backend. Tickets have a process-unique,
// nonwrapping pool namespace, even with independent allocation counters. Source
// generations must remain unique within a pool lifetime; its counter outlives it.
class HostUploadPool {
public:
    HostUploadPool(HostUploadBackend& backend,
                   AllocationGenerationSource& generations,
                   HostUploadPolicy policy = {});
    HostUploadPool(const HostUploadPool&) = delete;
    HostUploadPool& operator=(const HostUploadPool&) = delete;
    ~HostUploadPool();

    [[nodiscard]] HostUploadTicket start(
        std::span<const std::byte> bytes,
        const ExecutionBufferAddress& destination);
    [[nodiscard]] HostUploadStatus poll(
        const HostUploadTicket& ticket,
        const ExecutionBufferAddress& current_destination);
    void wait(const HostUploadTicket& ticket,
              const ExecutionBufferAddress& current_destination);
    void cancel(const HostUploadTicket& ticket);
    // Cleanup remains possible after failure. True means a released-source
    // proof was observed and the backend operation was retired.
    [[nodiscard]] bool drain(const HostUploadTicket& ticket,
                             const ExecutionBufferAddress& current_destination);

    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] std::uint64_t allocated_bytes() const noexcept;
    [[nodiscard]] std::uint32_t active_uploads() const noexcept;

private:
    struct Slot {
        std::shared_ptr<HostUploadSource> source;
        ExecutionBufferAddress destination;
        HostUploadHandle handle = 0;
        std::uint64_t source_generation = 0;
        std::uint64_t sequence = 0;
        bool cancel_requested = false;
    };
    Slot& require_ticket(const HostUploadTicket& ticket);
    void cancel_noexcept(Slot& slot) noexcept;

    HostUploadBackend* backend_;
    AllocationGenerationSource* generations_;
    HostUploadPolicy policy_;
    std::uint64_t pool_generation_ = 0;
    bool failed_ = false;
    std::vector<Slot> slots_;
};

}  // namespace strix
