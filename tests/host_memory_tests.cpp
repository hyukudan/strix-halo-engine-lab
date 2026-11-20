// Host test of the shared host memory home (strix/host_memory.hpp): the counter parser on recorded /proc text, the PSI
// parser, the bounded recovery wait on a scripted node, and one live read of this machine's counters (Linux only).
#include "strix/host_memory.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace {
namespace hm = strix::host_memory;
int failures = 0;
void require(bool ok, const char* what) {
    if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}

// Recorded from a 128 GiB unified-memory node (kernel 7.4), plus malformed lines the parser must skip.
const char* kMeminfo =
    "MemTotal:       131008292 kB\n"
    "MemFree:        89635928 kB\n"
    "MemAvailable:   89664076 kB\n"
    "Cached:         32640480 kB\n"
    "HugePages_Total:       3\n"
    "Broken line without colon\n"
    ":   12 kB\n"
    "NoNumber:  kB\n"
    "Huge:  99999999999999999999 kB\n"
    "Wrap:  18014398509481984 kB\n";
const char* kPsi =
    "some avg10=12.50 avg60=0.24 avg300=3.23 total=5003199078\n"
    "full avg10=3.00 avg60=0.20 avg300=3.21 total=4797346273\n";

void test_parse() {
    const auto f = hm::parse_kb_fields(kMeminfo);
    require(f.at("MemTotal") == 131008292ULL * 1024U, "MemTotal in bytes");
    require(f.at("Cached") == 32640480ULL * 1024U, "Cached in bytes");
    require(f.at("HugePages_Total") == 3U, "a value without unit is kept as is");
    require(f.count("Broken line without colon") == 0 && f.count("") == 0 && f.count("NoNumber") == 0,
            "malformed lines are skipped");
    require(f.count("Huge") == 0 && f.count("Wrap") == 0, "overflowing values are skipped, never wrapped");
    const auto r = hm::reading_from(f);
    require(r.ok && r.free == 89635928ULL * 1024U && r.available == 89664076ULL * 1024U, "reading");
    require(!hm::reading_from(hm::parse_kb_fields("MemFree: 1 kB\n")).ok, "a reading needs MemAvailable too");
    require(!hm::read_meminfo("/nonexistent/meminfo").ok, "unreadable file: not ok");

    const auto p = hm::parse_psi(kPsi);
    require(p.ok && p.some10 == 12.5 && p.full10 == 3.0 && p.full60 == 0.2, "psi");
    require(!hm::parse_psi("garbage\n").ok && hm::parse_psi("").full10 == 0.0, "psi unreadable: zero, not ok");
}

hm::Reading reading(std::uint64_t available_gib, std::uint64_t total_gib = 125) {
    hm::Reading r;
    r.total = total_gib * hm::kGiB;
    r.free = available_gib * hm::kGiB;
    r.available = available_gib * hm::kGiB;
    r.ok = true;
    return r;
}

// A scripted node: MemAvailable per read (GiB). The wait advances a fake clock only.
struct Script {
    std::vector<std::uint64_t> gib;
    std::size_t at = 0;
    double slept = 0;
    hm::Reading read() {
        const auto g = gib[std::min(at, gib.size() - 1)];
        ++at;
        return reading(g);
    }
};

hm::WaitResult run(Script& s, std::uint64_t need_gib, hm::WaitPolicy p = {}) {
    hm::Reading last;
    return hm::await_memory(
        p, [&] { return s.read(); }, [](const hm::Reading& r) { return r.available; },
        [&](const hm::Reading& r) { return r.available >= (need_gib + 4U) * hm::kGiB; },  // a 4 GiB floor
        [&](double sec) { s.slept += sec; }, last);
}

void test_wait() {
    {  // fits at once: one read, no sleep
        Script s{{110}};
        const auto w = run(s, 100);
        require(w.fits && w.reads == 1 && s.slept == 0, "fits at once");
    }
    {  // an exiting process gives back 6 GiB per poll: wait until it fits
        Script s{{80, 86, 92, 98, 104, 110}};
        const auto w = run(s, 100);
        require(w.fits && w.reads == 5 && w.why == "fits" && s.slept == 2.0, "recovering node is waited for");
    }
    {  // nothing is coming back: give up after stall_s, not max_wait_s
        Script s{{80}};
        const auto w = run(s, 100);
        require(!w.fits && w.why == "not recovering" && s.slept == 3.0, "flat node: stall exit");
    }
    {  // rising forever but slowly enough never to fit: bounded by max_wait_s
        Script s;
        for (std::uint64_t g = 10; g < 200; ++g) s.gib.push_back(g / 2U);  // +0.5 GiB per poll
        hm::WaitPolicy p;
        p.max_wait_s = 5;
        const auto w = run(s, 100, p);
        require(!w.fits && w.why == "timeout" && s.slept == 5.0, "bounded wait");
    }
    {  // waiting switched off
        Script s{{80, 110}};
        hm::WaitPolicy p;
        p.max_wait_s = 0;
        const auto w = run(s, 100, p);
        require(!w.fits && w.why == "no wait" && w.reads == 1, "no wait when off");
        require(hm::describe(w).find("host-memory wait: no wait after 1 reads") == 0, "describe");
    }
}

void test_live() {
#if defined(__linux__)
    const auto r = hm::read_meminfo();
    require(r.ok && r.total > 0 && r.available <= r.total && r.free <= r.total, "live /proc/meminfo");
    const auto p = hm::read_psi();
    std::cout << "live: MemTotal " << hm::gib_str(r.total) << " GiB MemAvailable " << hm::gib_str(r.available)
              << " GiB psi "
              << (p.ok ? "some10=" + std::to_string(p.some10) + " full10=" + std::to_string(p.full10) : std::string("n/a"))
              << "\n";
#endif
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    test_parse();
    test_wait();
    test_live();
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "host_memory tests: PASS\n";
    return 0;
}
