#pragma once

#include "strix/direct_gpu_stream_gpu.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <stdexcept>

// Entirely synthetic memory/export state. Byte vectors exercise exact copies
// and retention; synthetic addresses/descriptors never denote GPU/DMA memory.
namespace direct_gpu_stream_fixture {
using namespace strix;
using namespace strix::transport;
inline bool same_view(const TransportBufferView& a, const TransportBufferView& b) {
    return a.address == b.address && a.bytes == b.bytes && a.allocation_generation == b.allocation_generation && a.domain == b.domain && a.access == b.access;
}
struct State {
    DirectGpuExportDevice device{3, {71}, 720, 720};
    std::map<std::uintptr_t, std::vector<std::byte>> blocks;
    std::map<int, DirectGpuExportFdIdentity> fds;
    std::uintptr_t next = 0x100000;
    int next_fd = 70, freed = 0;
    bool failed = false;
    DirectTransportLinkPoolBinding bound;
    std::weak_ptr<DirectGpuPoolUseLease> use;
    std::pair<std::uintptr_t, std::vector<std::byte>*> containing(std::uintptr_t address, std::size_t bytes) {
        for (auto& [base, block] : blocks) if (address >= base && address - base <= block.size() && bytes <= block.size() - (address - base)) return {base, &block};
        throw std::runtime_error("synthetic address outside owned block");
    }
    std::byte* pointer(std::uintptr_t address, std::size_t bytes) {
        auto [base, block] = containing(address, bytes); return block->data() + (address - base);
    }
};
class Memory final : public RuntimeMemoryAllocator, public DirectGpuPoolReleaseWitness {
public:
    explicit Memory(std::shared_ptr<State> state) : state_(std::move(state)) {}
    std::uintptr_t allocate_device(std::uint64_t bytes, std::uint64_t) override {
        const auto address = state_->next; state_->next += 0x100000; state_->blocks.emplace(address, std::vector<std::byte>(static_cast<std::size_t>(bytes))); return address;
    }
    std::uintptr_t allocate_host(std::uint64_t, std::uint64_t) override { throw std::logic_error("GPU fixture only"); }
    void free_device(std::uintptr_t address, std::uint64_t bytes) noexcept override {
        if (!state_->use.expired()) { state_->failed = true; return; }
        const auto found = state_->blocks.find(address);
        if (found == state_->blocks.end() || found->second.size() != bytes) { state_->failed = true; return; }
        state_->blocks.erase(found); ++state_->freed;
    }
    void free_host(std::uintptr_t, std::uint64_t) noexcept override {}
    void bind_owned_pool(const DirectTransportLinkPoolBinding& pool) override { state_->bound = pool; }
    DirectPoolReleaseStatus release_owned_pool(const DirectTransportLinkPoolBinding& pool) const noexcept override {
        if (!state_->use.expired()) return DirectPoolReleaseStatus::pending;
        return !state_->blocks.contains(pool.tx.address) && !state_->blocks.contains(pool.rx.address) ? DirectPoolReleaseStatus::released : DirectPoolReleaseStatus::failed_retained;
    }
    std::shared_ptr<DirectGpuPoolUseLease> retain_owned_pool(const DirectTransportLinkPoolBinding& pool) override {
        if (state_->failed || !state_->use.expired() || !same_view(pool.tx, state_->bound.tx) || !same_view(pool.rx, state_->bound.rx)) throw std::logic_error("foreign synthetic use lease");
        class Lease final : public DirectGpuPoolUseLease {
        public:
            explicit Lease(std::shared_ptr<State> state) : state_(std::move(state)) {}
            bool current() const noexcept override { return !state_->failed && state_->blocks.contains(state_->bound.tx.address) && state_->blocks.contains(state_->bound.rx.address); }
        private:
            std::shared_ptr<State> state_;
        };
        auto lease = std::make_shared<Lease>(state_); state_->use = lease; return lease;
    }
private:
    std::shared_ptr<State> state_;
};
class ExportIo final : public DirectGpuExportIo {
public:
    explicit ExportIo(std::shared_ptr<State> state) : state_(std::move(state)) {}
    DirectGpuExportDevice inspect_device() override { return state_->device; }
    DirectGpuExportAllocation inspect_allocation(const TransportBufferView& view) override {
        auto [base, block] = state_->containing(view.address, 1); return {base, block->size(), state_->device.index, true, false};
    }
    void export_dma_buf(const DirectGpuExportAllocation&, int& fd) override {
        fd = state_->next_fd++; state_->fds.emplace(fd, DirectGpuExportFdIdentity{11, static_cast<std::uint64_t>(fd), true});
    }
    DirectGpuExportFdIdentity inspect_descriptor(int fd) override { return state_->fds.at(fd); }
    bool close_descriptor(int fd) noexcept override { return state_->fds.erase(fd) == 1; }
private:
    std::shared_ptr<State> state_;
};
struct Pool {
    std::shared_ptr<State> state = std::make_shared<State>();
    std::shared_ptr<Memory> memory = std::make_shared<Memory>(state);
    std::shared_ptr<ExportIo> export_io = std::make_shared<ExportIo>(state);
    AllocationGenerationSource generations;
    std::shared_ptr<DirectGpuExportOwner> exported;
    std::shared_ptr<DirectGpuExportImportLease> lease;
    Pool() {
        const auto plan = plan_multi_link_transport_pools(std::array{DirectFrameGeometry{4096, 32, true}});
        exported = DirectGpuExportOwner::allocate_and_export(export_io, memory, plan, generations, state->device); lease = exported->retain_for_import();
    }
    TransportBufferView external(std::uint64_t bytes) {
        return {memory->allocate_device(bytes, 4096), bytes, generations.next(), TransportMemoryDomain::gpu_device, TransportBufferAccess::read_write};
    }
};
class Gpu final : public DirectGpuStreamGpu {
public:
    explicit Gpu(std::shared_ptr<DirectGpuExportImportLease> lease) : lease_(std::move(lease)), exported_(lease_->binding()) {}
    const DirectGpuExportBinding& exported_binding() const noexcept override { return exported_; }
    std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept override { return lease_; }
    void require_current() override { if (failed || stopped) throw std::runtime_error("synthetic GPU owner unavailable"); lease_->require_current(); }
    void bind_import(const DirectDriverImportBinding& value) override { imported = value; }
    void stage_outbound_segments(const TransferDescriptor&, std::span<const TransportBufferView>, const TransportBufferView&) override {}
    void prepare(const DirectDriverOperation& operation) override { require_current(); operations[operation.handle] = operation; }
    DirectDriverVisibilityProof publish_terminal(const DirectDriverOperation& operation, const TransportBufferView&, const GpuCommitRecord& value) override {
        require_current(); ++published; terminal = value; return {operation, producer_complete ? DirectDriverVisibility::producer_release_to_external_dma : DirectDriverVisibility::unavailable};
    }
    DirectDriverCommitRead read_terminal(const DirectDriverOperation& operation, const TransportBufferView&) override {
        require_current(); ++acquired; return {terminal, {operation, acquire_complete ? DirectDriverVisibility::consumer_acquire_from_external_dma : DirectDriverVisibility::unavailable}};
    }
    TransportBufferView acquired_receive_view(const DirectDriverOperation& operation) override { auto view = operation.buffer; view.domain = TransportMemoryDomain::gpu_device; return view; }
    void consume_received_segments(const DirectDriverOperation&, std::span<const TransportBufferView>) override { ++unpacked; }
    DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation& operation) override {
        require_current(); ++consumers; return {operation, consumer_complete ? DirectDriverVisibility::consumer_release_complete : DirectDriverVisibility::unavailable};
    }
    void retire(const DirectDriverOperation& operation) override { operations.erase(operation.handle); ++retired; }
    void request_stop() noexcept override { stopped = true; }
    DirectDriverQuiescenceStatus poll_quiescence() noexcept override { return gpu_idle ? DirectDriverQuiescenceStatus::quiescent : DirectDriverQuiescenceStatus::pending; }
    DirectPoolReleaseStatus close() noexcept override { stopped = true; if (!gpu_idle) return DirectPoolReleaseStatus::failed_retained; lease_.reset(); return DirectPoolReleaseStatus::released; }
    DirectDriverImportBinding imported;
    GpuCommitRecord terminal;
    std::map<TransferHandle, DirectDriverOperation> operations;
    bool failed = false, stopped = false, gpu_idle = true, producer_complete = true, acquire_complete = true, consumer_complete = true;
    int published = 0, acquired = 0, consumers = 0, unpacked = 0, retired = 0;
private:
    std::shared_ptr<DirectGpuExportImportLease> lease_;
    DirectGpuExportBinding exported_;
};
} // namespace direct_gpu_stream_fixture
