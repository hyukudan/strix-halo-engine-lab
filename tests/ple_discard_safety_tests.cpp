#include "strix/ple_discard_safety.hpp"
#include "strix/registered_backing_lifetime.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::ple_reclaim;
namespace {
void expect(bool ok) { if (!ok) throw std::runtime_error("discard safety assertion failed"); }
template<class F> void refuses(F f) {
    bool refused = false;
    try { f(); } catch (const std::exception&) { refused = true; }
    expect(refused);
}
template<class F> void safety_refuses(F f) {
    bool refused = false;
    try { f(); } catch (const DiscardSafetyError&) { refused = true; }
    expect(refused); // Generic errors could be swallowed by automatic fallback.
}
void disjointness() {
    const std::array drop{DiscardRange{0x10000,8192,7,19,0x4000}};
    std::vector<RegisteredBacking> regs{{"weight",0x80000,4096,7,19,0x6000}};
    const auto validate = [&] { validate_discard_set(4096,drop,drop,regs); };
    validate(); // Exact touching endpoints, same file, different virtual mappings.
    regs[0].file_offset = 0x5fff; // One byte on the final discarded page.
    safety_refuses(validate);
    regs[0].inode = 20;
    validate(); // Same offsets in another inode are independent.
    regs[0].inode = 19; regs[0].device = 8;
    validate(); // Same inode number on another device is independent.
    regs[0].address = 0x11fff;
    safety_refuses(validate); // Even different files may not overlap in virtual pages.
    regs[0] = {"alias",0x90000,1,7,19,0x3fff};
    validate();
    regs[0].bytes = 2;
    safety_refuses(validate); // Registration rounds both ends outward.
    regs.clear();
    auto advice = drop;
    advice[0].file_offset = 0x4001;
    safety_refuses([&] { validate_discard_set(4096,{},advice,regs); });
    advice = drop; advice[0].bytes = 0;
    safety_refuses([&] { validate_discard_set(4096,drop,advice,regs); }); // No whole-file fadvise.
    advice = drop; advice[0].file_offset = UINT64_MAX - 4095;
    safety_refuses([&] { validate_discard_set(4096,{},advice,regs); });
    safety_refuses([&] { validate_discard_set(0,drop,drop,regs); });
    safety_refuses([&] { validate_discard_set(3000,drop,drop,regs); });
    // A disjoint madvise never licenses an overlapping file-cache discard.
    advice = drop; advice[0].file_offset = 0;
    regs = {{"other-mapping",0x90000,4096,7,19,0}};
    safety_refuses([&] { validate_discard_set(4096,drop,advice,regs); });
    advice[0].file_offset = 0x8000;
    regs[0].file_offset = 0x4000;
    safety_refuses([&] { validate_discard_set(4096,drop,advice,regs); });
    // Failure is unconditional: diagnostic/strict-off paths cannot continue.
    bool discarded = false;
    safety_refuses([&] { validate_discard_set(4096,drop,drop,regs); discarded = true; });
    expect(!discarded);
}
void lock_failure_policy() {
    // Keep main's automatic/off parsing; only the exact explicit =1 is mandatory.
    for (const char* setting : {static_cast<const char*>(nullptr), "", "1", "0", "01", "true", "10"}) {
        const strix::RegisteredPageLockPolicy policy(setting);
        expect(policy.enabled == !(setting && setting[0] == '0'));
        expect(policy.required == (setting && std::string(setting) == "1"));
    }
    for (const auto* failure : {"unreadable RLIMIT_MEMLOCK", "insufficient RLIMIT_MEMLOCK", "mlock2 failed"}) {
        const bool limit_failure = std::string(failure) != "mlock2 failed";
        for (const char* setting : {static_cast<const char*>(nullptr), "", "0", "1"}) {
            strix::RegisteredPageLockPolicy policy(setting);
            unsigned warnings = 0, lock_attempts = 0;
            bool refused = false;
            const auto fail = [&] { policy.failed(failure, [&] { ++warnings; }); };
            try {
                if (policy.enabled && limit_failure) fail();
                for (unsigned range = 0; range < 3U; ++range) {
                    if (policy.enabled) { ++lock_attempts; fail(); }
                }
            } catch (const strix::RegisteredPageLockError& error) {
                refused = true;
                expect(std::string(error.what()) == failure);
            }
            const bool off = setting && setting[0] == '0';
            const bool explicit_request = setting && std::string(setting) == "1";
            expect(refused == explicit_request);
            expect(warnings == (off || explicit_request ? 0U : 1U));
            expect(lock_attempts == (off || limit_failure ? 0U : 1U));
            if (!explicit_request) expect(!policy.enabled); // No later lock attempts.
        }
    }
}
void lifetime() {
    std::vector<std::string> events;
    const auto acquire = [&](bool alias_fails) {
        strix::acquire_registered_backing(
            [&] { events.emplace_back("register"); },
            [&] { events.emplace_back("track"); },
            [&] { events.emplace_back("fresh-alias"); if (alias_fails) throw std::runtime_error("alias"); });
    };
    const auto retire = [&](bool fail, bool drain_fails) {
        return strix::retire_registered_backings(std::array{1},
            [&] { events.emplace_back("drain"); if (drain_fails) throw std::runtime_error("drain"); },
            [&](int) { events.emplace_back("unregister"); return !fail; },
            [&](int) { events.emplace_back("munlock"); });
    };
    acquire(false);
    expect(retire(false,false)); events.emplace_back("discard");
    acquire(false); // Reuse gets a fresh alias, never the old alias.
    expect(events == std::vector<std::string>{"register","track","fresh-alias","drain","unregister","munlock","discard",
        "register","track","fresh-alias"});
    events.clear();
    expect(!retire(true,false));
    expect(events == std::vector<std::string>{"drain","unregister"}); // Owner must retain its mapping.
    events.clear();
    refuses([&] { (void)retire(false,true); });
    expect(events == std::vector<std::string>{"drain"});
    events.clear();
    refuses([&] { acquire(true); });
    expect(retire(false,false));
    expect(events == std::vector<std::string>{"register","track","fresh-alias","drain","unregister","munlock"});
    events.clear();
    expect(!strix::retire_registered_backings(std::array{1,2,3},
        [&] { events.emplace_back("drain"); },
        [&](int range) { events.emplace_back("unregister" + std::to_string(range)); return range != 2; },
        [&](int range) { events.emplace_back("munlock" + std::to_string(range)); }));
    // Every unregister precedes the first munlock (no VMA split next to a live registration); the failed range stays
    // locked even when its neighbors retire successfully.
    expect(events == std::vector<std::string>{"drain","unregister1","unregister2","unregister3","munlock1","munlock3"});
    // Acquire: every range is locked before the first registration; the owner gets the lock flag with the range.
    events.clear();
    const auto acquire_all = [&](const std::vector<int>& ranges, int lock_fails, int lock_throws, int register_fails,
                                 int alias_fails) {
        strix::acquire_locked_backings(ranges,
            [&](int range) {
                if (range == lock_throws) throw std::runtime_error("required lock");
                events.emplace_back("mlock" + std::to_string(range));
                return range != lock_fails;
            },
            [&](int range) { events.emplace_back("munlock" + std::to_string(range)); },
            [&](int range, bool locked, const auto& handed) {
                strix::acquire_registered_backing(
                    [&] { if (range == register_fails) throw std::runtime_error("register");
                          events.emplace_back("register" + std::to_string(range)); },
                    [&] { events.emplace_back(std::string(locked ? "track-locked" : "track") + std::to_string(range)); handed(); },
                    [&] { if (range == alias_fails) throw std::runtime_error("alias"); });
            });
    };
    acquire_all({1,2,3}, 2, 0, 0, 0);
    expect(events == std::vector<std::string>{"mlock1","mlock2","mlock3","register1","track-locked1","register2","track2",
        "register3","track-locked3"});
    events.clear();  // Registration failure: the untracked locked ranges are unlocked here, the tracked one is the owner's.
    refuses([&] { acquire_all({1,2,3}, 0, 0, 2, 0); });
    expect(events == std::vector<std::string>{"mlock1","mlock2","mlock3","register1","track-locked1","munlock2","munlock3"});
    events.clear();  // Alias failure after tracking: the owner already holds range 2 and unlocks it on release.
    refuses([&] { acquire_all({1,2,3}, 0, 0, 0, 2); });
    expect(events == std::vector<std::string>{"mlock1","mlock2","mlock3","register1","track-locked1","register2",
        "track-locked2","munlock3"});
    events.clear();  // A required lock that fails registers nothing and unlocks what it locked.
    refuses([&] { acquire_all({1,2,3}, 0, 3, 0, 0); });
    expect(events == std::vector<std::string>{"mlock1","mlock2","munlock1","munlock2"});
    // Teardown unlock: one call per contiguous run, a gap (failed unregister / other merged range) starts a new run.
    events.clear();
    {
        strix::CoalescedUnlock runs([&](std::uintptr_t base, std::uint64_t bytes) {
            events.emplace_back("munlock" + std::to_string(base) + "+" + std::to_string(bytes));
        });
        runs.add(0x1000, 0x1000); runs.add(0x2000, 0x1000); runs.add(0x3000, 0);  // contiguous, empty ignored
        runs.add(0x5000, 0x1000); runs.add(0x6000, 0x2000);                     // gap at 0x3000..0x5000
        runs.add(0x1000, 0x1000);                                                // out of order: own call
        runs.flush(); runs.flush();                                              // flush is idempotent
    }
    expect(events == std::vector<std::string>{"munlock4096+8192","munlock20480+12288","munlock4096+4096"});
    events.clear();
    expect(strix::retire_registered_backings(std::vector<int>{},
        [&] { events.emplace_back("drain"); },
        [&](int) { events.emplace_back("unregister"); return true; },
        [&](int) { events.emplace_back("munlock"); }));
    expect(events.empty());
}
}
int main() {
    try { disjointness(); lock_failure_policy(); lifetime(); std::cout << "PLE discard safety and registered backing lifetime tests passed\n"; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
