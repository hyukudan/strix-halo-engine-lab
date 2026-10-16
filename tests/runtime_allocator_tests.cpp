#include "strix/runtime_allocator.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeAllocator final
    : public strix::RuntimeMemoryAllocator {
public:
    int fail_after = -1;
    int allocation_calls = 0;
    int device_frees = 0;
    int host_frees = 0;

    std::uintptr_t next_device =
        static_cast<std::uintptr_t>(
            0x90000000ULL);
    std::uintptr_t next_host =
        static_cast<std::uintptr_t>(
            0xa0000000ULL);

    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        maybe_fail();
        const auto address =
            align(next_device, alignment);
        next_device =
            address +
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return address;
    }

    std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        maybe_fail();
        const auto address =
            align(next_host, alignment);
        next_host =
            address +
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return address;
    }

    void free_device(
        std::uintptr_t,
        std::uint64_t) noexcept override {
        ++device_frees;
    }

    void free_host(
        std::uintptr_t,
        std::uint64_t) noexcept override {
        ++host_frees;
    }

private:
    void maybe_fail() {
        if (fail_after >= 0 &&
            allocation_calls == fail_after) {
            ++allocation_calls;
            throw std::runtime_error(
                "injected allocation failure");
        }
        ++allocation_calls;
    }

    static std::uintptr_t align(
        std::uintptr_t value,
        std::uint64_t alignment) {
        const auto remainder =
            value %
            static_cast<std::uintptr_t>(
                alignment);
        return remainder == 0
            ? value
            : value +
                  static_cast<std::uintptr_t>(
                      alignment - remainder);
    }
};

strix::CompiledSerialExecution fake_compiled() {
    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x1234ULL;
    compiled.arena.alignment = 256;
    compiled.arena.total_bytes = 4096;
    return compiled;
}

strix::SerialSessionStateLayout fake_state_layout() {
    strix::SerialSessionStateLayout layout;
    layout.layout_identity =
        0x5678ULL;
    layout.device_alignment = 256;
    layout.host_alignment = 64;
    layout.device_total_bytes = 8192;
    layout.host_total_bytes = 128;
    return layout;
}

struct LeaseCounters {int device_frees=0,host_frees=0,destroyed=0;};
class LeaseAllocator final : public strix::RuntimeMemoryAllocator {
public:
    explicit LeaseAllocator(std::shared_ptr<LeaseCounters> counts):counts_(std::move(counts)) {}
    ~LeaseAllocator() override {++counts_->destroyed;}
    int throw_lease_after=-1;
    mutable int lease_calls=0;
    bool opaque=false;
    std::uintptr_t allocate_device(std::uint64_t bytes,std::uint64_t alignment) override {
        const auto address=allocator_.allocate_device(bytes,alignment);
        device_[address]={bytes,next_++};return address;
    }
    std::uintptr_t allocate_host(std::uint64_t bytes,std::uint64_t alignment) override {
        const auto address=allocator_.allocate_host(bytes,alignment);
        host_[address]={bytes,next_++};return address;
    }
    void free_device(std::uintptr_t address,std::uint64_t) noexcept override {
        if(device_.erase(address)) ++counts_->device_frees;
    }
    void free_host(std::uintptr_t address,std::uint64_t) noexcept override {
        if(host_.erase(address)) ++counts_->host_frees;
    }
    std::uint64_t allocation_lease_identity(std::uintptr_t address,std::uint64_t bytes,bool host) const override {
        if(throw_lease_after>=0 && lease_calls++==throw_lease_after) throw std::runtime_error("lease lookup failed");
        if(opaque) return 0;
        const auto& map=host?host_:device_;const auto found=map.find(address);
        if(found==map.end() || found->second.first!=bytes) throw std::invalid_argument("no live exact lease");
        return found->second.second;
    }
    std::uint64_t replace_device_lease(std::uintptr_t address,std::uint64_t lease) {
        auto& current=device_.at(address).second;const auto previous=current;current=lease;return previous;
    }
private:
    FakeAllocator allocator_;
    std::shared_ptr<LeaseCounters> counts_;
    std::uint64_t next_=1;
    std::map<std::uintptr_t,std::pair<std::uint64_t,std::uint64_t>> device_,host_;
};
template<class F> bool inventory_rejects(F&& action) {
    try {action();} catch(const std::invalid_argument&) {return true;}
    return false;
}
void retained_allocator_contract() {
    using namespace strix;
    const auto compiled=fake_compiled();const auto layout=fake_state_layout();
    auto counts=std::make_shared<LeaseCounters>();
    auto allocator=std::make_shared<LeaseAllocator>(counts);
    std::weak_ptr<LeaseAllocator> weak_allocator=allocator;
    AllocationGenerationSource generations;
    auto value=RuntimeExecutionMemoryOwner::allocate(compiled,layout,allocator,generations);
    value.validate_inventory_ownership();
    auto moved=std::move(value);
    expect(!value.valid() && !value.retained_allocator() && moved.retained_allocator().get()==allocator.get(),
        "moving retained memory transfers the actual allocator pin and clears the source");
    expect(inventory_rejects([&] {value.validate_inventory_ownership();}),"moved-from memory cannot produce inventory");
    auto owner=std::make_shared<RuntimeExecutionMemoryOwner>(std::move(moved));
    expect(owner->weak_from_this().lock().get()==owner.get(),"shared memory has its actual control-block identity after value moves");
    auto& scratch=const_cast<ExecutionMemoryBinding&>(owner->scratch());
    auto& state=const_cast<SessionStateMemoryBinding&>(owner->state());
    const auto saved_scratch=scratch;const auto saved_state=state;
    for(unsigned fault=0;fault<13U;++fault) {
        switch(fault) {
        case 0:++scratch.execution_identity;break;case 1:++scratch.arena_base;break;
        case 2:++scratch.arena_bytes;break;case 3:++scratch.allocation_generation;break;
        case 4:++state.layout_identity;break;case 5:++state.device_base;break;
        case 6:++state.device_bytes;break;case 7:++state.device_allocation_generation;break;
        case 8:++state.host_base;break;case 9:++state.host_bytes;break;
        case 10:++state.host_allocation_generation;break;
        case 11:state.stage_mapping=reinterpret_cast<const DistributedStageStateMapping*>(std::uintptr_t{1});break;
        case 12:allocator->replace_device_lease(scratch.arena_base,9001U);break;
        }
        expect(inventory_rejects([&] {owner->validate_inventory_ownership();}),
            "retained inventory rejects every binding field and same-address replacement lease");
        if(fault==12U) allocator->replace_device_lease(saved_scratch.arena_base,1U);
        scratch=saved_scratch;state=saved_state;
    }
    owner->validate_inventory_ownership();
    allocator.reset();
    expect(!weak_allocator.expired(),"actual allocator remains alive after the caller drops its pin");
    owner.reset();
    expect(weak_allocator.expired() && counts->device_frees==2 && counts->host_frees==1 && counts->destroyed==1,
        "retained pools free exactly once before the last actual allocator is destroyed");

    auto assign_counts=std::make_shared<LeaseCounters>();auto assign_allocator=std::make_shared<LeaseAllocator>(assign_counts);
    auto first=RuntimeExecutionMemoryOwner::allocate(compiled,layout,assign_allocator,generations);
    auto second=RuntimeExecutionMemoryOwner::allocate(compiled,layout,assign_allocator,generations);
    second=std::move(first);second.validate_inventory_ownership();
    expect(assign_counts->device_frees==2 && assign_counts->host_frees==1 && !first.valid(),
        "move assignment releases the old pools and transfers retained ownership");

    auto failure_counts=std::make_shared<LeaseCounters>();auto failing=std::make_shared<LeaseAllocator>(failure_counts);
    failing->throw_lease_after=1;bool failed=false;
    try {(void)RuntimeExecutionMemoryOwner::allocate(compiled,layout,failing,generations);} catch(const std::runtime_error&) {failed=true;}
    expect(failed && failure_counts->device_frees==2 && failure_counts->host_frees==1,
        "setup lease failure releases all allocated pools while retaining the allocator");
    failing->throw_lease_after=0;failing->lease_calls=0;
    {auto borrowed=RuntimeExecutionMemoryOwner::allocate(compiled,layout,*failing,generations);
     expect(!borrowed.retained_allocator() && failing->lease_calls==0,
        "legacy borrowed allocation neither retains allocator nor queries new lease hooks");}
    auto opaque=std::make_shared<LeaseAllocator>(std::make_shared<LeaseCounters>());opaque->opaque=true;
    auto opaque_owner=RuntimeExecutionMemoryOwner::allocate(compiled,layout,opaque,generations);
    expect(opaque_owner.valid() && inventory_rejects([&] {opaque_owner.validate_inventory_ownership();}),
        "opaque allocators support retained lifetime without fabricated inventory evidence");
    expect(inventory_rejects([&] {(void)RuntimeExecutionMemoryOwner::allocate(compiled,layout,
        std::shared_ptr<RuntimeMemoryAllocator>{},generations);}),"null retained allocator rejects before allocation");
}

}  // namespace

int main() {
    retained_allocator_contract();
    auto compiled = fake_compiled();
    auto layout = fake_state_layout();

    FakeAllocator allocator;
    strix::AllocationGenerationSource generations;

    {
        auto owner =
            strix::RuntimeExecutionMemoryOwner::allocate(
                compiled,
                layout,
                allocator,
                generations);

        expect(owner.valid(),
               "runtime memory owner becomes valid after allocation");
        expect(owner.scratch().arena_bytes ==
                   compiled.arena.total_bytes &&
                   owner.scratch().allocation_generation == 1,
               "scratch allocation receives first generation");
        expect(owner.state().device_allocation_generation == 2 &&
                   owner.state().host_allocation_generation == 3,
               "state pools receive independent allocation generations");

        auto moved = std::move(owner);
        expect(!owner.valid() && moved.valid(),
               "runtime memory owner transfers allocation ownership by move");
    }

    expect(allocator.device_frees == 2 &&
               allocator.host_frees == 1,
           "runtime owner releases scratch and state pools exactly once");

    FakeAllocator failing;
    failing.fail_after = 1;
    strix::AllocationGenerationSource failure_generations;
    bool failed = false;
    try {
        static_cast<void>(
            strix::RuntimeExecutionMemoryOwner::allocate(
                compiled,
                layout,
                failing,
                failure_generations));
    } catch (const std::runtime_error&) {
        failed = true;
    }
    expect(failed,
           "allocation failure propagates");
    expect(failing.device_frees == 1 &&
               failing.host_frees == 0,
           "partial allocation failure releases already-owned scratch");

    bool zero_generation_rejected = false;
    try {
        strix::AllocationGenerationSource invalid(0);
        static_cast<void>(invalid);
    } catch (const std::invalid_argument&) {
        zero_generation_rejected = true;
    }
    expect(zero_generation_rejected,
           "allocation generation source rejects zero start");

    if (failures == 0) {
        std::cout << "runtime allocator tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
