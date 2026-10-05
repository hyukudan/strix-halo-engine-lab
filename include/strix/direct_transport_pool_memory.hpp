#pragma once

#include "strix/direct_transport_pool.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/transport.hpp"

#include <cstdint>
#include <vector>

namespace strix::transport {

struct DirectTransportLinkPoolBinding {
    TransportBufferView tx;
    TransportBufferView rx;
};

struct DirectTransportPoolMemoryBinding {
    std::vector<DirectTransportLinkPoolBinding> links;
    std::uint64_t total_gpu_bytes = 0;
    std::uint64_t binding_identity = 0;
};

void validate_direct_transport_pool_memory_binding(
    const MultiLinkTransportPoolPlan& plan,
    const DirectTransportPoolMemoryBinding& binding);

class DirectTransportPoolMemoryOwner {
public:
    DirectTransportPoolMemoryOwner() = default;
    DirectTransportPoolMemoryOwner(
        const DirectTransportPoolMemoryOwner&) = delete;
    DirectTransportPoolMemoryOwner& operator=(
        const DirectTransportPoolMemoryOwner&) = delete;

    DirectTransportPoolMemoryOwner(
        DirectTransportPoolMemoryOwner&& other) noexcept;
    DirectTransportPoolMemoryOwner& operator=(
        DirectTransportPoolMemoryOwner&& other) noexcept;

    ~DirectTransportPoolMemoryOwner();

    [[nodiscard]] static DirectTransportPoolMemoryOwner allocate(
        const MultiLinkTransportPoolPlan& plan,
        RuntimeMemoryAllocator& allocator,
        AllocationGenerationSource& generations);

    [[nodiscard]] const DirectTransportPoolMemoryBinding&
    binding() const noexcept {
        return binding_;
    }

    [[nodiscard]] bool valid() const noexcept {
        return allocator_ != nullptr;
    }

private:
    void release() noexcept;

    RuntimeMemoryAllocator* allocator_ = nullptr;
    DirectTransportPoolMemoryBinding binding_;
};

}  // namespace strix::transport
