#include "strix/direct_transport_pool_memory.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix::transport {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "direct transport pool memory byte overflow");
    }
    return left + right;
}

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t build_binding_identity(
    const DirectTransportPoolMemoryBinding& binding) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, binding.total_gpu_bytes);
    hash_u64(hash, static_cast<std::uint64_t>(binding.links.size()));
    for (const auto& link : binding.links) {
        for (const auto* view : {&link.tx, &link.rx}) {
            hash_u64(hash, static_cast<std::uint64_t>(view->address));
            hash_u64(hash, view->bytes);
            hash_u64(hash, view->allocation_generation);
            hash_u64(hash, static_cast<std::uint64_t>(view->domain));
            hash_u64(hash, static_cast<std::uint64_t>(view->access));
        }
    }
    return hash;
}

TransportBufferView make_view(
    std::uintptr_t address,
    std::uint64_t bytes,
    std::uint64_t generation) {
    TransportBufferView view;
    view.address = address;
    view.bytes = bytes;
    view.allocation_generation = generation;
    view.domain = TransportMemoryDomain::gpu_transport_pool;
    view.access = TransportBufferAccess::read_write;
    return view;
}

}  // namespace

void validate_direct_transport_pool_memory_binding(
    const MultiLinkTransportPoolPlan& plan,
    const DirectTransportPoolMemoryBinding& binding) {
    if (plan.links.empty() ||
        binding.links.size() != plan.links.size() ||
        binding.total_gpu_bytes != plan.total_gpu_bytes ||
        binding.binding_identity == 0) {
        throw std::invalid_argument(
            "direct transport pool memory binding does not match plan");
    }

    std::uint64_t total = 0;
    for (std::size_t index = 0; index < plan.links.size(); ++index) {
        const auto& expected = plan.links[index];
        const auto& actual = binding.links[index];

        if (expected.pool_alignment == 0 ||
            actual.tx.address == 0 ||
            actual.rx.address == 0 ||
            actual.tx.address == actual.rx.address ||
            actual.tx.bytes != expected.tx_pool_bytes ||
            actual.rx.bytes != expected.rx_pool_bytes ||
            actual.tx.allocation_generation == 0 ||
            actual.rx.allocation_generation == 0 ||
            actual.tx.allocation_generation == actual.rx.allocation_generation ||
            actual.tx.domain != TransportMemoryDomain::gpu_transport_pool ||
            actual.rx.domain != TransportMemoryDomain::gpu_transport_pool ||
            actual.tx.access != TransportBufferAccess::read_write ||
            actual.rx.access != TransportBufferAccess::read_write ||
            actual.tx.address % expected.pool_alignment != 0 ||
            actual.rx.address % expected.pool_alignment != 0) {
            throw std::invalid_argument(
                "direct transport link pool binding is incompatible");
        }

        total = checked_add(
            total,
            checked_add(actual.tx.bytes, actual.rx.bytes));
    }

    if (total != binding.total_gpu_bytes ||
        binding.binding_identity != build_binding_identity(binding)) {
        throw std::invalid_argument(
            "direct transport pool memory identity/byte accounting is invalid");
    }
}

DirectTransportPoolMemoryOwner::DirectTransportPoolMemoryOwner(
    DirectTransportPoolMemoryOwner&& other) noexcept
    : allocator_(other.allocator_),
      binding_(std::move(other.binding_)) {
    other.allocator_ = nullptr;
    other.binding_ = {};
}

DirectTransportPoolMemoryOwner&
DirectTransportPoolMemoryOwner::operator=(
    DirectTransportPoolMemoryOwner&& other) noexcept {
    if (this != &other) {
        release();
        allocator_ = other.allocator_;
        binding_ = std::move(other.binding_);
        other.allocator_ = nullptr;
        other.binding_ = {};
    }
    return *this;
}

DirectTransportPoolMemoryOwner::~DirectTransportPoolMemoryOwner() {
    release();
}

DirectTransportPoolMemoryOwner
DirectTransportPoolMemoryOwner::allocate(
    const MultiLinkTransportPoolPlan& plan,
    RuntimeMemoryAllocator& allocator,
    AllocationGenerationSource& generations) {
    if (plan.links.empty() || plan.total_gpu_bytes == 0) {
        throw std::invalid_argument(
            "direct transport pool allocation requires non-empty plan");
    }

    DirectTransportPoolMemoryOwner owner;
    owner.allocator_ = &allocator;
    owner.binding_.links.reserve(plan.links.size());
    owner.binding_.total_gpu_bytes = plan.total_gpu_bytes;

    try {
        for (const auto& link : plan.links) {
            if (link.tx_pool_bytes == 0 ||
                link.rx_pool_bytes == 0 ||
                link.pool_alignment == 0) {
                throw std::invalid_argument(
                    "direct transport pool plan contains invalid link geometry");
            }

            const auto tx_generation = generations.next();
            const auto tx_address = allocator.allocate_device(
                link.tx_pool_bytes,
                link.pool_alignment);

            DirectTransportLinkPoolBinding current;
            current.tx = make_view(
                tx_address,
                link.tx_pool_bytes,
                tx_generation);
            owner.binding_.links.push_back(current);

            const auto rx_generation = generations.next();
            const auto rx_address = allocator.allocate_device(
                link.rx_pool_bytes,
                link.pool_alignment);
            owner.binding_.links.back().rx = make_view(
                rx_address,
                link.rx_pool_bytes,
                rx_generation);
        }

        owner.binding_.binding_identity =
            build_binding_identity(owner.binding_);
        validate_direct_transport_pool_memory_binding(
            plan,
            owner.binding_);
    } catch (...) {
        owner.release();
        throw;
    }

    return owner;
}

void DirectTransportPoolMemoryOwner::release() noexcept {
    if (allocator_ == nullptr) {
        return;
    }

    for (auto iterator = binding_.links.rbegin();
         iterator != binding_.links.rend();
         ++iterator) {
        if (iterator->rx.address != 0) {
            allocator_->free_device(
                iterator->rx.address,
                iterator->rx.bytes);
        }
        if (iterator->tx.address != 0) {
            allocator_->free_device(
                iterator->tx.address,
                iterator->tx.bytes);
        }
    }

    allocator_ = nullptr;
    binding_ = {};
}

}  // namespace strix::transport
