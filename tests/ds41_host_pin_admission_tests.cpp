#include "strix/ds41_host_pin_admission.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace strix::ds41;
namespace {
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
constexpr std::uint64_t GiB = 1ULL << 30;
struct Room { std::uint64_t free_commit = 100 * GiB, free_physical = 100 * GiB; bool known = true; };
struct Faults {
    Room room;
    int allocations = 0, grows = 0, sleeps = 0, queries = 0;
    bool allocation_ok = true, grow_works = false;
    std::chrono::steady_clock::time_point clock{};
    HostPinReason run(bool implicit, double wait = 2, double floor = 20, bool allow_first = false) {
        const auto deadline = clock + std::chrono::milliseconds(static_cast<int>(wait * 1000));
        std::uint32_t waits = 0;
        auto result = allocate_host_pin(GiB, floor, implicit, allow_first, deadline, waits,
            [&] { ++queries; return room; }, [&] { ++allocations; return allocation_ok; },
            [&](double) { ++grows; if (grow_works) room.free_commit = 100 * GiB; },
            [&] { ++sleeps; clock += std::chrono::seconds(1); }, [&] { return clock; });
        check(waits == static_cast<std::uint32_t>(sleeps), "wait accounting");
        return result;
    }
};
}
int main() {
    try {
        check(host_pin_deadline(1e300) == std::chrono::steady_clock::time_point::max(), "huge finite wait saturation");
        check(valid_host_pin_policy(0,0) && valid_host_pin_policy(20,90), "valid policy");
        for (double bad : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            check(!valid_host_pin_policy(bad,90), "invalid floor");
            check(!valid_host_pin_policy(20,bad), "invalid wait");
        }
        check(host_pin_admission(true,25*GiB,22*GiB,GiB,20*GiB) == HostPinReason::none, "inclusive boundaries");
        check(host_pin_admission(true,25*GiB-1,22*GiB,GiB,20*GiB) == HostPinReason::physical, "physical boundary");
        check(host_pin_admission(true,25*GiB,22*GiB-1,GiB,20*GiB) == HostPinReason::commit, "commit boundary");
        for (bool implicit : {true,false}) {
            Faults f; f.room.free_physical = 24 * GiB;
            check(f.run(implicit) == HostPinReason::physical && !f.allocations && !f.grows && !f.sleeps, "physical reject");
            f = {}; f.room.known = false;
            check(f.run(implicit) == HostPinReason::telemetry && !f.allocations && !f.grows && !f.sleeps, "unknown telemetry");
            f = {};
            check(f.run(implicit) == HostPinReason::none && f.allocations == 1 && !f.sleeps, "success");
        }
        Faults f; f.room.free_commit = 21*GiB;
        check(f.run(true) == HostPinReason::commit && !f.allocations && !f.grows && !f.sleeps, "implicit commit immediate");
        f = {}; f.allocation_ok = false;
        check(f.run(true) == HostPinReason::alloc && f.allocations == 1 && !f.grows && !f.sleeps, "implicit alloc immediate");
        f = {}; f.room.free_commit = 21*GiB; f.grow_works = true;
        check(f.run(false) == HostPinReason::none && f.grows == 1 && f.sleeps == 1, "explicit growth retry");
        f = {}; f.room.free_commit = 21*GiB;
        check(f.run(false) == HostPinReason::commit && f.grows == 2 && f.sleeps == 2 && !f.allocations, "explicit deadline");
        f = {}; f.allocation_ok = false;
        check(f.run(false) == HostPinReason::alloc && f.allocations == 2 && f.sleeps == 2, "explicit alloc deadline");
        f = {};
        check(f.run(false,0,20,false) == HostPinReason::commit && !f.queries, "expired shared deadline skips layer");
        f = {};
        check(f.run(false,0,20,true) == HostPinReason::none && f.allocations == 1, "explicit zero wait one attempt");
        f = {}; f.room.free_physical = 4*GiB; // mandatory buffers already charged; zero floor still needs n+4GiB
        check(f.run(true,90,0) == HostPinReason::physical && !f.allocations, "implicit zero floor preserves margin");
        f = {}; f.allocation_ok = false;
        check(f.run(true,90,0) == HostPinReason::alloc && f.allocations == 1, "zero floor allocation fallback");
        std::cout << "host pin admission: all CPU/fault-injection checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
