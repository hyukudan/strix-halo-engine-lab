#include "strix/deepseek41_serve_cache_policy.hpp"
#include "strix/deepseek41_cache_lease.hpp"

#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace ds = strix::deepseek41;

void expect(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

ds::ServeCachePolicy policy(std::initializer_list<std::string_view> args = {}, bool device = true, bool hetero = false, bool dspark = false) {
    ds::ServeCachePolicy p;
    for (auto at = args.begin(); at != args.end(); ++at) {
        const auto key = *at;
        std::string_view value;
        if (p.takes_value(key)) {
            if (++at == args.end()) throw std::runtime_error("missing test argument");
            value = *at;
        }
        expect(p.parse(key, value), "unrecognized cache option");
    }
    p.resolve(device, hetero, dspark, "cache-env", "home");
    return p;
}

int main() try {
    const auto a = policy(), b = policy({"--cache-profile", "agent"});
    expect(a.prefix_cache_mb == 512 && b.prefix_cache_mb == 512 && a.prefix_reserve_mb == 8192, "bounded default");
    expect(a.prefix_inrun && a.prefix_inrun_reach == 0 && a.prefix_session && !a.prefix_before_tail, "default snapshots");
    expect(a.prefix_dir.empty() && a.prefix_disk_mb == 0, "environment alone cannot enable disk");
    expect(policy({}, true, false, true).prefix_inrun && !policy({}, true, false, true).prefix_session, "implicit sessions exclude DSpark");
    expect(policy({"--prefix-session-snapshots"}, true, false, true).prefix_session, "explicit DSpark/session conflict reaches validation");
    expect(policy({}, false).prefix_cache_mb == 0 && policy({}, true, true).prefix_cache_mb == 0, "rows and hetero receive no bundle");
    expect(policy({"--prefix-cache-mb", "32"}, false).prefix_cache_mb == 32, "explicit rows conflict reaches validation");
    const auto off = policy({"--cache-profile", "off"});
    expect(off.prefix_cache_mb == 0 && !off.prefix_inrun && !off.prefix_session, "off alone is baseline");
    const auto explicit_off = policy({"--cache-profile", "off", "--prefix-cache-mb", "64", "--prefix-inrun-snapshots"});
    expect(explicit_off.prefix_cache_mb == 64 && explicit_off.prefix_inrun, "profile off preserves explicit features");
    for (const bool reverse : {false, true}) {
        const auto p = reverse ? policy({"--no-prefix-cache", "--prefix-cache-mb", "64", "--prefix-session-snapshots"})
                               : policy({"--prefix-cache-mb", "64", "--prefix-session-snapshots", "--no-prefix-cache"});
        expect(p.prefix_cache_mb == 0 && !p.prefix_session && !p.prefix_inrun, "host veto dominates either order");
        const auto q = reverse ? policy({"--prefix-cache-mb", "0", "--prefix-cache-mb", "64"})
                               : policy({"--prefix-cache-mb", "64", "--prefix-cache-mb", "0"});
        expect(q.prefix_cache_mb == 0 && !q.prefix_session, "numeric zero is a hard veto");
        const auto r = reverse ? policy({"--no-prefix-disk", "--no-prefix-before-tail", "--no-prefix-session-snapshots", "--no-prefix-inrun-snapshots",
                                        "--prefix-cache-dir", "disk", "--prefix-before-tail", "--prefix-session-snapshots", "--prefix-inrun-snapshots"})
                               : policy({"--prefix-cache-dir", "disk", "--prefix-before-tail", "--prefix-session-snapshots", "--prefix-inrun-snapshots",
                                        "--no-prefix-disk", "--no-prefix-before-tail", "--no-prefix-session-snapshots", "--no-prefix-inrun-snapshots"});
        expect(r.prefix_dir.empty() && !r.prefix_before_tail && !r.prefix_session && !r.prefix_inrun, "component vetoes dominate either order");
    }
    const auto custom = policy({"--prefix-cache-mb", "128", "--prefix-reserve-mb", "17", "--prefix-inrun-reach", "512"});
    expect(custom.prefix_cache_mb == 128 && custom.prefix_reserve_mb == 17 && custom.prefix_inrun_reach == 512, "explicit budgets and reach survive fills");
    const auto disk = policy({"--cache-profile", "off", "--prefix-cache-mb", "64", "--prefix-cache-disk-mb", "0"});
    expect(disk.prefix_dir == "cache-env" && disk.prefix_disk_mb == 0, "explicit disk zero retains unlimited semantics");
    expect(policy({"--prefix-cache-dir", "disk"}).prefix_disk_mb == 4096, "explicit directory gets finite default");
    for (const auto bad : {"-1", "12x", "18446744073709551615", ""}) {
        bool threw = false;
        try { ds::ServeCachePolicy p; p.parse("--prefix-cache-mb", bad); }
        catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "malformed/overflow budget rejected");
    }
    expect(!ds::prefix_memory_admitted(0, 0, 0), "unknown memory fails closed");
    expect(!ds::prefix_memory_admitted(8192, 8192, 1), "reserve is after predicted allocations");
    expect(ds::prefix_memory_admitted(8193, 8192, 1), "round predicted bytes upward");
    expect(!ds::prefix_memory_admitted(1, UINT64_MAX, UINT64_MAX), "admission arithmetic cannot wrap");
#if defined(__unix__)
    const auto root = std::filesystem::temp_directory_path() / ("ds41-host-lease-test-" + std::to_string(::getpid()));
    {
        ds::PrefixHostLease first(512, root), second(512, root), denied(64, root);
        expect(first.budget_mb() == 512 && second.budget_mb() == 512 && denied.budget_mb() == 0, "aggregate host ceiling");
    }
    {
        ds::PrefixHostLease large(8192, root);
        expect(large.budget_mb() == 1024, "released leases are reusable; explicit budget is capped");
    }
    std::filesystem::remove_all(root);
#else
    expect(ds::PrefixHostLease(512).budget_mb() == 0, "unsupported host reservation fails closed");
#endif
    std::puts("deepseek41 serve cache policy tests passed");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
