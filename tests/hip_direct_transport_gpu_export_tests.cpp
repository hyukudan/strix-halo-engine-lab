#include "strix/hip_direct_transport_gpu_export.hpp"

#include <array>
#include <cerrno>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

// HIP calls are injected. Real /dev/null and /dev/zero FDs exercise POSIX
// ownership only; they are NOT DMA-BUFs or GPU/driver qualification evidence.
#if defined(__linux__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool value, const char* message) { if (!value) { std::cerr << message << '\n'; ++failures; } }
template<class F> bool rejected(F function) { try { function(); } catch (const std::exception&) { return true; } return false; }
struct Counts { int allocated = 0, freed = 0; };
class Allocator final : public RuntimeMemoryAllocator {
public:
    explicit Allocator(std::shared_ptr<Counts> counts) : counts_(std::move(counts)) {}
    std::uintptr_t allocate_device(std::uint64_t bytes, std::uint64_t alignment) override {
        ++counts_->allocated; const auto address = next_; next_ += bytes + alignment; return address;
    }
    std::uintptr_t allocate_host(std::uint64_t, std::uint64_t) override { throw std::logic_error("GPU pool fixture only"); }
    void free_device(std::uintptr_t, std::uint64_t) noexcept override { ++counts_->freed; }
    void free_host(std::uintptr_t, std::uint64_t) noexcept override {}
private:
    std::uintptr_t next_ = 0x100000;
    std::shared_ptr<Counts> counts_;
};
struct Fixture {
    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
    std::shared_ptr<Allocator> allocator = std::make_shared<Allocator>(counts);
    HipDirectGpuExportCalls calls = HipDirectGpuExportCalls::native();
    AllocationGenerationSource generations;
    MultiLinkTransportPoolPlan plan = plan_multi_link_transport_pools(std::array{DirectFrameGeometry{4096, 4, true}});
    std::vector<int> fds;
    int active_device = 3, native_device = 3, exports = 0, native_error_at = -1;
    bool managed = false, output_before_error = false;
    Fixture() {
        calls.get_device = [&](int* value) { *value = active_device; return hipSuccess; };
        calls.device_uuid = [](hipUUID* value, hipDevice_t device) {
            if (device != 3) return hipErrorInvalidDevice;
            *value = {}; value->bytes[0] = 71; return hipSuccess;
        };
        calls.runtime_version = [](int* value) { *value = 720; return hipSuccess; };
        calls.driver_version = [](int* value) { *value = 720; return hipSuccess; };
        calls.pointer_attributes = [&](hipPointerAttribute_t* value, const void*) {
            *value = {}; value->type = hipMemoryTypeDevice; value->device = native_device; value->isManaged = managed ? 1 : 0; return hipSuccess;
        };
        calls.address_range = [&](hipDeviceptr_t* base, std::size_t* bytes, hipDeviceptr_t address) {
            *base = address; *bytes = static_cast<std::size_t>(plan.links[0].tx_pool_bytes); return hipSuccess;
        };
        calls.dma_buf_range_export = [&](int* descriptor, hipDeviceptr_t address, std::size_t bytes) {
            ++exports;
            expect(reinterpret_cast<std::uintptr_t>(address) >= 0x100000 && bytes == plan.links[0].tx_pool_bytes && *descriptor == -1,
                   "native range-export call receives exact full base/extent and adopted empty output slot");
            if (exports == native_error_at && !output_before_error) return hipErrorNotSupported;
            *descriptor = ::open("/dev/null", O_RDONLY);
            if (*descriptor < 0) throw std::runtime_error("open fixture FD failed");
            fds.push_back(*descriptor);
            return exports == native_error_at ? hipErrorNotSupported : hipSuccess;
        };
    }
    std::shared_ptr<DirectGpuExportOwner> prepare() {
        auto io = std::make_shared<HipDirectGpuExportIo>(calls);
        return DirectGpuExportOwner::allocate_and_export(io, allocator, plan, generations, io->inspect_device());
    }
    bool all_closed() const {
        for (int fd : fds) if (::fcntl(fd, F_GETFD) >= 0 || errno != EBADF) return false;
        return true;
    }
};
}
#endif

int main() {
#if defined(__linux__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
    {
        Fixture fixture; auto owner = fixture.prepare(); const auto& binding = owner->binding();
        expect(fixture.exports == 2 && binding.device.index == 3 && binding.device.uuid[0] == 71 &&
               (::fcntl(binding.descriptors[0], F_GETFD) & FD_CLOEXEC) != 0 &&
               (::fcntl(binding.descriptors[1], F_GETFD) & FD_CLOEXEC) != 0,
               "injected HIP calls integrate with actual fstat/fcntl FD setup");
        owner.reset(); expect(fixture.all_closed() && fixture.counts->freed == 2,
                              "real local descriptors close before shared pool and allocator retire");
    }
    for (bool output_first : {false, true}) {
        Fixture fixture; fixture.native_error_at = 2; fixture.output_before_error = output_first;
        expect(rejected([&] { static_cast<void>(fixture.prepare()); }) && fixture.all_closed() && fixture.counts->freed == 2,
               "unsupported/partial HIP export error preserves then closes already acquired real local FDs");
    }
    {
        Fixture fixture; fixture.calls.set_fd_flags = [](int, int) { return -1; };
        expect(rejected([&] { static_cast<void>(fixture.prepare()); }) && fixture.all_closed() && fixture.counts->freed == 2,
               "failed CLOEXEC setup after HIP output adoption does not lose FD ownership");
    }
    for (bool managed : {false, true}) {
        Fixture fixture; fixture.managed = managed; if (!managed) fixture.native_device = 0;
        expect(rejected([&] { static_cast<void>(fixture.prepare()); }) && fixture.exports == 0 && fixture.counts->freed == 2,
               "real provider assembly rejects foreign native device and managed memory before export");
    }
    {
        Fixture fixture; auto owner = fixture.prepare(); const int fd = owner->binding().descriptors[0];
        const int saved = ::dup(fd), foreign = ::open("/dev/zero", O_RDONLY | O_CLOEXEC);
        if (saved < 0 || foreign < 0 || ::dup2(foreign, fd) != fd || ::fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {
            throw std::runtime_error("replace fixture FD failed");
        }
        expect(rejected([&] { owner->require_current(); }) && owner->release() == DirectPoolReleaseStatus::failed_retained &&
               ::fcntl(fd, F_GETFD) >= 0 && fixture.counts->freed == 0,
               "actual FD number replacement preserves foreign descriptor and original pool ownership");
        if (::dup2(saved, fd) != fd || ::fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) throw std::runtime_error("restore fixture FD failed");
        static_cast<void>(::close(saved)); static_cast<void>(::close(foreign));
        expect(owner->release() == DirectPoolReleaseStatus::released && fixture.all_closed() && fixture.counts->freed == 2,
               "exact restored FD cleanup releases retained resources without healing failure state");
    }
    {
        Fixture fixture; fixture.calls.address_range = {};
        expect(rejected([&] { HipDirectGpuExportIo absent(fixture.calls); }), "incomplete syscall/HIP table cannot publish a provider");
    }
    if (failures == 0) std::cout << "injected HIP + real POSIX export ownership fixtures passed; no GPU/driver DMA executed\n";
    return failures == 0 ? 0 : 1;
#else
    std::cout << "HIP DMA-BUF export is unavailable on this build platform\n";
    return 0;
#endif
}
