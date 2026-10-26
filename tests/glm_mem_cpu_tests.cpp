// GLM load memory accounting, CPU tests (design notes): /proc + sysfs probe parsing, the admission check (fails
// closed with every term named), the hipMalloc cost model against the load measurement numbers, the monitor's trace + watchdog, and
// PackReader streaming a synthetic multi-GiB safetensors set through a bounded staging buffer: page cache of the shards
// stays empty (O_DIRECT, or buffered + DONTNEED), RSS stays bounded, and a capped budget refuses before any byte is read.
//
//   strix-glm5-mem-cpu-tests [DIR]   DIR: where the synthetic pack goes (default $TMPDIR or /var/tmp; NOT tmpfs);
//   GLM_MEM_TEST_GIB=N (default 4) total pack size, GLM_MEM_TEST_KEEP=1 keeps the files.
#include "glm/engine/glm_mem.hpp"
#include "glm/engine/glm_pack.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace mem = strix::glm::engine::mem;
using strix::glm::engine::detail::PackReader;

namespace {

int failures = 0, checks = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        ++checks;                                                                 \
        if (!(c)) {                                                               \
            ++failures;                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
        }                                                                         \
    } while (0)

void write_file(const std::string& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << s;
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string meminfo(std::uint64_t avail_kb, std::uint64_t cma_free_kb, std::uint64_t swap_free_kb) {
    return "MemTotal:       128657860 kB\nMemFree:          2000000 kB\nMemAvailable:   " + std::to_string(avail_kb) +
           " kB\nCached:            123456 kB\nActive(file):     1048576 kB\nInactive(file):   1048576 kB\nShmem:              65536 kB\n"
           "SwapTotal:        8388604 kB\nSwapFree:       " +
           std::to_string(swap_free_kb) + " kB\nCmaTotal:              0 kB\nCmaFree:        " + std::to_string(cma_free_kb) + " kB\nGPUActive:      109347940 kB\n";
}

// ---------------------------------------------------------------- probe, admission, cost model
void test_probe(const fs::path& tmp) {
    const fs::path d = tmp / "probe";
    fs::create_directories(d / "drm");
    write_file((d / "meminfo").string(), meminfo(20 * 1048576ULL, 3 * 1048576ULL, 8388604));
    write_file((d / "status").string(), "Name:\tx\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\nRssAnon:\t   512 kB\nRssFile:\t   256 kB\nRssShmem:\t   256 kB\n");
    write_file((d / "drm/mem_info_gtt_total").string(), "125829120000\n");
    write_file((d / "drm/mem_info_gtt_used").string(), "19345408\n");
    write_file((d / "drm/mem_info_vram_total").string(), "536870912\n");
    write_file((d / "drm/mem_info_vram_used").string(), "1000\n");
    // zoneinfo as on the nodes: KHO scratch = MIGRATE_CMA pageblocks, 183,296 + 3,267,584 pages = 13.16 GiB
    write_file((d / "zoneinfo").string(), "Node 0, zone      DMA\n        cma      0\n      nr_free_cma  0\nNode 0, zone    DMA32\n"
                                          "        cma      183296\n      nr_free_cma  183279\nNode 0, zone   Normal\n"
                                          "        cma      3267584\n      nr_free_cma  2912503\n");
    write_file((d / "psi").string(), "some avg10=31.50 avg60=10.00 avg300=2.00 total=123\nfull avg10=27.25 avg60=9.00 avg300=1.00 total=99\n");
    mem::Probe p({(d / "meminfo").string(), (d / "status").string(), (d / "drm").string(), (d / "zoneinfo").string(), (d / "psi").string()});
    const mem::Snapshot s = p.read();
    CHECK(s.mem_available == 20ULL * mem::kGiB);
    CHECK(s.cma_free == 3ULL * mem::kGiB);
    CHECK(s.cma_total == 3450880ULL * 4096);
    CHECK(s.file_lru == 2ULL * mem::kGiB);
    CHECK(s.gpu_active == 109347940ULL * 1024);
    CHECK(s.psi_full10 == 27.25);
    // CMA cannot back GTT pages: 20 - 3 (CmaFree) - min(13.16 - 3 used in CMA, 2 file pages) = 15 GiB
    CHECK(s.ram_headroom() == 15ULL * mem::kGiB);
    CHECK(s.swap_used() == 0);
    CHECK(s.rss == 1024 * 1024 && s.rss_anon == 512 * 1024 && s.rss_shmem == 256 * 1024 && s.hwm == 2048 * 1024);
    CHECK(s.gpu && s.gtt_total == 125829120000ULL && s.gtt_used == 19345408ULL && s.vram_total == 536870912ULL);
    CHECK(s.gtt_headroom() == 125829120000ULL - 19345408ULL);
    CHECK(s.dev_used() == 19345408ULL + 1000ULL && s.dev_counters());
    fs::create_directories(d / "drm2");  // GTT counters present, VRAM used missing: not a usable device measurement
    write_file((d / "drm2/mem_info_gtt_total").string(), "125829120000\n");
    write_file((d / "drm2/mem_info_gtt_used").string(), "19345408\n");
    CHECK(!mem::Probe({(d / "meminfo").string(), (d / "status").string(), (d / "drm2").string()}).read().gpu);
    CHECK(s.meminfo_ok && s.cma_free_ok && s.zoneinfo_ok);
    mem::Probe nogpu({(d / "meminfo").string(), (d / "status").string(), (d / "nonexistent").string(), (d / "nonexistent").string(),
                      (d / "nonexistent").string()});
    const mem::Snapshot ng = nogpu.read();
    CHECK(!ng.gpu && ng.dev_counters() && ng.dev_used() == 109347940ULL * 1024);  // GPUActive fallback
    // the real files parse on any Linux
    const mem::Snapshot r = mem::Probe({"/proc/meminfo", "/proc/self/status", "/nonexistent"}).read();
    CHECK(r.mem_total > 0 && r.mem_available > 0 && r.rss > 0);
}

void test_admit() {
    mem::Snapshot s;
    s.meminfo_ok = s.cma_free_ok = s.zoneinfo_ok = true;
    s.mem_available = 20 * mem::kGiB;
    s.cma_free = 2 * mem::kGiB;  // RAM headroom 18
    s.gpu = true;
    s.gtt_total = 117 * mem::kGiB;
    s.gtt_used = 100 * mem::kGiB;  // GTT headroom 17
    const mem::Margins m{8 * mem::kGiB, 2 * mem::kGiB};
    bool threw = false;
    try {
        mem::admit("ok", {{"a", 8 * mem::kGiB, 1 * mem::kGiB}, {"b", 1 * mem::kGiB, 0}}, s, m);  // gtt 9+2<=17, ram 10+8<=18
    } catch (const mem::BudgetError&) {
        threw = true;
    }
    CHECK(!threw);
    std::string msg;
    try {
        mem::admit("too much RAM", {{"weights", 9 * mem::kGiB, 0}, {"host staging", 0, 2 * mem::kGiB}}, s, m);  // ram 11+8 > 18
    } catch (const mem::BudgetError& e) {
        msg = e.what();
    }
    CHECK(msg.find("REFUSED too much RAM") != std::string::npos);
    CHECK(msg.find("RAM: need") != std::string::npos && msg.find("GTT:") == std::string::npos);
    CHECK(msg.find("weights") != std::string::npos && msg.find("host staging") != std::string::npos);
    msg.clear();
    s.gtt_used = 110 * mem::kGiB;  // GTT headroom 7
    try {
        mem::admit("too much GTT", {{"weights", 6 * mem::kGiB, 0}}, s, m);
    } catch (const mem::BudgetError& e) {
        msg = e.what();
    }
    CHECK(msg.find("GTT: need 6.00") != std::string::npos);
    s.gpu = false;  // no counters: only the RAM check
    threw = false;
    try {
        mem::admit("no gpu", {{"weights", 6 * mem::kGiB, 0}}, s, m);
    } catch (const mem::BudgetError&) {
        threw = true;
    }
    CHECK(!threw);
    // no /proc/meminfo: refused, whatever the numbers
    s.meminfo_ok = false;
    msg.clear();
    try {
        mem::admit("blind", {{"weights", 1, 0}}, s, m);
    } catch (const mem::BudgetError& e) {
        msg = e.what();
    }
    CHECK(msg.find("REFUSED blind: /proc/meminfo") != std::string::npos);
    CHECK(s.ram_headroom() == 0);
    // zone layout unknown: every file page counted as CMA-resident
    mem::Snapshot z;
    z.meminfo_ok = z.cma_free_ok = true;
    z.mem_available = 20 * mem::kGiB;
    z.cma_free = 1 * mem::kGiB;
    z.file_lru = 5 * mem::kGiB;
    CHECK(z.ram_headroom() == 14 * mem::kGiB);
    z.zoneinfo_ok = true;  // known: no CMA in use -> only the free CMA is subtracted
    z.cma_total = 1 * mem::kGiB;
    CHECK(z.ram_headroom() == 19 * mem::kGiB);
    // CmaFree missing while the zone layout shows CMA: the whole CMA area is subtracted
    mem::Snapshot c;
    c.meminfo_ok = c.zoneinfo_ok = true;
    c.mem_available = 20 * mem::kGiB;
    c.cma_free = 5 * mem::kGiB;  // value present but not trusted (flag off)
    c.cma_total = 13 * mem::kGiB;
    CHECK(c.ram_headroom() == 7 * mem::kGiB);
    c.zoneinfo_ok = false;  // neither CmaFree nor the zone layout: free CMA unknowable -> nothing trusted
    CHECK(c.ram_headroom() == 0);
}

void test_cost_model() {
    // The GLM expert linear rank slice: trellis 4096 x 1024 x 4 bit + suh + svh (out slice: suh 8 KiB, svh 2 KiB; in slice:
    // suh 2 KiB, svh 8 KiB), one strix-trellis allocation of 256 B-aligned regions.
    CHECK(mem::prepared_bytes(2097152, 8192, 2048) == 2107392);
    CHECK(mem::prepared_bytes(2097152, 2048, 8192) == 2107392);
    CHECK(mem::prepared_bytes(100, 10, 10) == 768);  // the empty bias region is aligned too (as prepare_regions)
    const std::uint64_t ex = 2107392;
    const mem::AllocModel g2 = mem::infer_alloc_model(ex, 4194304.0);
    CHECK(g2.calibrated && g2.ratio == 0 && g2.granule == 2 * mem::kMiB && g2.physical(ex) == 4194304);
    const mem::AllocModel g4 = mem::infer_alloc_model(ex, 2109440.0);
    CHECK(g4.calibrated && g4.granule == 4096 && g4.physical(ex) == 2109440);
    const mem::AllocModel g64 = mem::infer_alloc_model(ex, 2162688.0);
    CHECK(g64.granule == 64 * mem::kKiB);
    const mem::AllocModel gr = mem::infer_alloc_model(ex, 6.0e6);
    CHECK(gr.ratio > 2.8 && gr.ratio < 2.9 && gr.physical(ex) >= 6000000);
    const mem::AllocModel bad = mem::infer_alloc_model(ex, 1.0e6);  // < 0.9 x size: untrusted -> worst plausible
    CHECK(!bad.calibrated && bad.granule == 2 * mem::kMiB);
    // retained calibration batches
    bool ok = false;
    const mem::AllocModel b2 = mem::model_from_batches(ex, {4194304.0, 4194304.0, 4190208.0, 4194304.0}, ok);
    CHECK(ok && b2.granule == 2 * mem::kMiB && b2.calibrated);
    const mem::AllocModel b4 = mem::model_from_batches(ex, {2109440.0, 2109440.0, 2109440.0, 2109440.0}, ok);
    CHECK(ok && b4.granule == 4096);
    // one batch reused freed fragments (looks cheap): the largest batch decides
    const mem::AllocModel bm = mem::model_from_batches(ex, {4194304.0, 4194304.0, 3800000.0, 4194304.0}, ok);
    CHECK(ok && bm.granule == 2 * mem::kMiB);
    // disagreeing batches (another process allocating / freeing): conservative, not marked calibrated
    const mem::AllocModel bx = mem::model_from_batches(ex, {2109440.0, 0.0, 2109440.0, 2109440.0}, ok);
    CHECK(!ok && !bx.calibrated && bx.physical(ex) >= 4194304);
    const mem::AllocModel by = mem::model_from_batches(ex, {9.0e6, 2109440.0, 2109440.0, 2109440.0}, ok);
    CHECK(!ok && by.physical(ex) >= 9000000);
    // load measurement, rank slice (design notes): 36,288 expert linears = 71.29 GiB at 4 KiB, 141.75 GiB at 2 MiB
    mem::LoadPlan plan;
    for (int i = 0; i < 36288; ++i) plan.add(ex);
    plan.step("experts");
    plan.add(1268776960);  // the embedding arena chunk
    plan.step("embed");
    CHECK(plan.dominant_size() == ex);
    CHECK(std::fabs(mem::gib(plan.physical(g4, plan.mark[0])) - 71.290) < 0.001);
    CHECK(std::fabs(mem::gib(plan.physical(g2, plan.mark[0])) - 141.750) < 0.001);
    CHECK(plan.physical(g2) == plan.physical(g2, plan.mark[0]) + mem::round_up(1268776960, 2 * mem::kMiB));
    CHECK(plan.summary(g2).find("36289 device allocations") != std::string::npos);
}

// The load measurement rank slice (TP2 rank 0, 45 layers, mtp 0, slots 8, capacity 2048): every device allocation of the e04c4ef6 load
// as (bytes, count), from design notes on the real headers (36,782 allocations, 77.04 GiB logical;
// the 33 constructor buffers, 23 MiB, are the pre-load baseline, not the plan).
// Admitted against node 1 at the start of the load measurement: the 2 MiB granule (148 GiB physical) is refused before anything is loaded,
// the page granule (77.1 GiB) fits.
void test_e1_plan() {
    static const std::pair<std::uint64_t, int> hist[] = {
        {4ULL, 3}, {32ULL, 4}, {40ULL, 1}, {64ULL, 1}, {128ULL, 3}, {256ULL, 2}, {512ULL, 4}, {768ULL, 1}, {2048ULL, 4},
        {4096ULL, 3}, {6144ULL, 2}, {8208ULL, 1}, {9216ULL, 1}, {16384ULL, 5}, {32768ULL, 2}, {49152ULL, 1}, {65536ULL, 2},
        {131072ULL, 6}, {196608ULL, 3}, {524288ULL, 3}, {1179648ULL, 34}, {1582080ULL, 11}, {2097152ULL, 11},
        {2107392ULL, 36288}, {2478080ULL, 1}, {3155968ULL, 126}, {4194304ULL, 22}, {4729856ULL, 22}, {9456640ULL, 11},
        {12599296ULL, 136}, {15749120ULL, 9}, {16777216ULL, 45}, {25190400ULL, 11}, {238058752ULL, 1}, {1073741824ULL, 1},
        {1268776960ULL, 1}};
    mem::LoadPlan plan;
    for (const auto& [b, n] : hist)
        for (int i = 0; i < n; ++i) plan.add(b);
    CHECK(plan.device.size() == 36782);
    CHECK(plan.logical == 82725329028ULL);
    CHECK(plan.dominant_size() == 2107392);
    const mem::AllocModel page = mem::infer_alloc_model(2107392, 2109440.0), g2 = mem::infer_alloc_model(2107392, 4194304.0);
    std::fprintf(stderr, "e1 plan: logical %.3f GiB, physical %.3f GiB at 4 KiB, %.3f GiB at 2 MiB\n", mem::gib(plan.logical),
                 mem::gib(plan.physical(page)), mem::gib(plan.physical(g2)));
    CHECK(std::fabs(mem::gib(plan.physical(g2)) - 148.166) < 0.005);  // memmodel.py "2MiB all" minus the ctor baseline
    mem::Snapshot c2;  // node 1 idle (read-only check at 10:5x: MemAvailable 120.5 GiB, gtt_total 117.19 GiB, gtt_used 18 MiB)
    c2.meminfo_ok = c2.cma_free_ok = c2.zoneinfo_ok = true;
    c2.mem_available = 126318640ULL * 1024;
    c2.gpu = true;
    c2.gtt_total = 125829120000ULL;
    c2.gtt_used = 19345408ULL;
    const std::vector<mem::Term> host{{"pinned staging", 1 * mem::kGiB, 0}, {"pageable whole + bounce", 0, 476 * mem::kMiB + 64 * mem::kMiB}};
    std::string msg;
    try {
        std::vector<mem::Term> t = host;
        t.push_back({"device plan (physical)", plan.physical(g2), 0});
        mem::admit("e1 rank 0 load", t, c2, mem::Margins{});
    } catch (const mem::BudgetError& e) {
        msg = e.what();
    }
    std::fprintf(stderr, "e1 plan at 2 MiB: %s\n", msg.substr(0, msg.find('\n')).c_str());
    CHECK(msg.find("REFUSED e1 rank 0 load") != std::string::npos && msg.find("GTT: need 149.17") != std::string::npos);
    bool threw = false;
    try {
        std::vector<mem::Term> t = host;
        t.push_back({"device plan (physical)", plan.physical(page), 0});
        mem::admit("e1 rank 0 load", t, c2, mem::Margins{});
    } catch (const mem::BudgetError&) {
        threw = true;
    }
    CHECK(!threw);
}

// ---------------------------------------------------------------- monitor: trace + watchdog
void test_monitor(const fs::path& tmp) {
    const fs::path d = tmp / "mon";
    fs::create_directories(d);
    const std::string mi = (d / "meminfo").string(), trace = (d / "trace.tsv").string();
    write_file(mi, meminfo(40 * 1048576ULL, 0, 8388604));
    std::atomic<int> aborted{0};
    std::string reason;
    std::mutex mu;
    {
        mem::Watchdog wd;
        wd.min_ram_headroom = 4 * mem::kGiB;
        wd.max_swap_growth = 1 * mem::kGiB;
        wd.max_rss = 0;
        mem::Monitor m(mem::Probe({mi, "/proc/self/status", "/nonexistent"}), trace, wd,
                       [&](const std::string& why) {
                           std::lock_guard<std::mutex> g(mu);
                           reason = why;
                           aborted.fetch_add(1);
                       },
                       20, 100);
        m.phase("load");
        m.counters(3 * mem::kGiB, 5 * mem::kGiB);
        std::this_thread::sleep_for(std::chrono::milliseconds(450));
        CHECK(aborted.load() == 0 && !m.tripped());
        CHECK(m.lines() >= 3);  // one line per 100 ms
        write_file(mi + ".new", meminfo(3 * 1048576ULL, 0, 8388604));  // RAM headroom 3 GiB < 4: trip
        fs::rename(mi + ".new", mi);
        for (int i = 0; i < 50 && aborted.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(aborted.load() == 1 && m.tripped());
    }
    {
        std::lock_guard<std::mutex> g(mu);
        CHECK(reason.find("RAM headroom 3.00 GiB < 4.00 GiB") != std::string::npos);
    }
    const std::string t = slurp(trace);
    CHECK(t.rfind("# t_s\tphase", 0) == 0);
    CHECK(t.find("\tload\t") != std::string::npos);
    CHECK(t.find("\t3072\t5120\n") != std::string::npos);  // dev_logical_mib, read_mib
    CHECK(t.find("# WATCHDOG ABORT (glm memory): RAM headroom") != std::string::npos);
    // swap growth trip
    write_file(mi, meminfo(40 * 1048576ULL, 0, 8388604));
    aborted = 0;
    {
        mem::Watchdog wd;
        wd.max_rss = 0;
        wd.max_swap_growth = 1 * mem::kGiB;
        mem::Monitor m(mem::Probe({mi, "/proc/self/status", "/nonexistent"}), "", wd, [&](const std::string& why) {
            std::lock_guard<std::mutex> g(mu);
            reason = why;
            aborted.fetch_add(1);
        }, 20, 100);
        write_file(mi + ".new", meminfo(40 * 1048576ULL, 0, 8388604 - 2 * 1048576ULL));  // +2 GiB swap
        fs::rename(mi + ".new", mi);
        for (int i = 0; i < 50 && aborted.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(aborted.load() == 1);
    }
    CHECK(reason.find("swap use grew 2.00 GiB") != std::string::npos);
    // RSS trip on the real process (the threshold is below this test's own RSS)
    aborted = 0;
    {
        mem::Watchdog wd;
        wd.min_ram_headroom = 0;
        wd.max_swap_growth = 0;
        wd.max_rss = 1 * mem::kMiB;
        mem::Monitor m(mem::Probe({mi, "/proc/self/status", "/nonexistent"}), "", wd, [&](const std::string&) { aborted.fetch_add(1); },
                       20, 100);
        for (int i = 0; i < 50 && aborted.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(aborted.load() == 1);
    }
    // PSI full trip (thrash)
    aborted = 0;
    {
        const std::string psi = (d / "psi").string();
        write_file(psi, "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");
        mem::Watchdog wd;
        wd.max_rss = 0;
        mem::Monitor m(mem::Probe({mi, "/proc/self/status", "/nonexistent", "/nonexistent", psi}), "", wd,
                       [&](const std::string& why) {
                           std::lock_guard<std::mutex> g(mu);
                           reason = why;
                           aborted.fetch_add(1);
                       },
                       20, 100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(aborted.load() == 0);
        write_file(psi + ".new", "some avg10=60.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=40.00 avg60=0.00 avg300=0.00 total=0\n");
        fs::rename(psi + ".new", psi);
        for (int i = 0; i < 50 && aborted.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(aborted.load() == 1);
    }
    CHECK(reason.find("memory PSI full avg10 40.0 %") != std::string::npos);
    // telemetry loss: meminfo disappears -> abort
    aborted = 0;
    {
        mem::Watchdog wd;
        wd.max_rss = 0;
        const std::string mi2 = (d / "meminfo2").string();
        write_file(mi2, meminfo(40 * 1048576ULL, 0, 8388604));
        mem::Monitor m(mem::Probe({mi2, "/proc/self/status", "/nonexistent"}), "", wd, [&](const std::string& why) {
            std::lock_guard<std::mutex> g(mu);
            reason = why;
            aborted.fetch_add(1);
        }, 20, 100);
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        CHECK(aborted.load() == 0);
        fs::remove(mi2);
        for (int i = 0; i < 50 && aborted.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(aborted.load() == 1);
    }
    CHECK(reason.find("telemetry lost: /proc/meminfo") != std::string::npos);
    // blocked trace sink: a full pipe nobody reads; the writer thread blocks, the abort still fires within the grace
    aborted = 0;
    {
        int pfd[2];
        CHECK(::pipe(pfd) == 0);
        ::fcntl(pfd[1], F_SETFL, O_NONBLOCK);
        std::vector<char> junk(4096, 'x');
        while (::write(pfd[1], junk.data(), junk.size()) > 0) {
        }
        const std::string sink = "/proc/self/fd/" + std::to_string(pfd[1]);  // reopened blocking
        mem::Watchdog wd;
        wd.min_ram_headroom = 4 * mem::kGiB;
        wd.max_rss = 0;
        write_file(mi, meminfo(40 * 1048576ULL, 0, 8388604));
        const auto t0 = std::chrono::steady_clock::now();
        double fired = -1;
        {
            mem::Monitor m(mem::Probe({mi, "/proc/self/status", "/nonexistent"}), sink, wd, [&](const std::string&) {
                fired = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                aborted.fetch_add(1);
            }, 20, 20, 300);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            write_file(mi + ".new", meminfo(1 * 1048576ULL, 0, 8388604));
            fs::rename(mi + ".new", mi);
            for (int i = 0; i < 100 && aborted.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            CHECK(aborted.load() == 1);
            CHECK(fired > 0 && fired < 0.9);  // 100 ms + sample + 300 ms grace, never the blocked write
        }  // destructor: writer still blocked -> detached, returns within ~2 s
        ::close(pfd[0]);  // unblocks the detached writer (EPIPE -> counted)
        ::close(pfd[1]);
    }
    // disabled watchdog never trips
    aborted = 0;
    {
        mem::Watchdog wd;
        wd.enabled = false;
        wd.max_rss = 1;
        mem::Monitor m(mem::Probe({mi, "/proc/self/status", "/nonexistent"}), "", wd, [&](const std::string&) { aborted.fetch_add(1); },
                       20, 100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    CHECK(aborted.load() == 0);
}

// ---------------------------------------------------------------- synthetic large pack
struct Synth {
    std::vector<std::string> files;
    std::vector<std::string> names;
    std::uint64_t bytes = 0, tensor_bytes = 0;
};

std::uint64_t pattern(std::uint64_t tensor, std::uint64_t word) { return (tensor * 0x9E3779B97F4A7C15ULL) ^ (word * 0xBF58476D1CE4E5B9ULL); }

void drop_file_cache(const std::string& p) {
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) return;
    (void)::fdatasync(fd);
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
}

// Resident page-cache bytes of a file (mincore over a PROT_READ mapping that is never touched).
std::uint64_t resident(const std::string& p) {
    const int fd = ::open(p.c_str(), O_RDONLY);
    struct stat st {};
    ::fstat(fd, &st);
    const auto n = static_cast<std::size_t>(st.st_size);
    void* a = ::mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (a == MAP_FAILED) return ~0ULL;
    const long pg = ::sysconf(_SC_PAGESIZE);
    std::vector<unsigned char> v((n + static_cast<std::size_t>(pg) - 1) / static_cast<std::size_t>(pg));
    ::mincore(a, n, v.data());
    ::munmap(a, n);
    std::uint64_t r = 0;
    for (unsigned char c : v) r += (c & 1) ? static_cast<std::uint64_t>(pg) : 0;
    return r;
}

// N shards of T tensors (U8 [tensor_bytes]) + model.safetensors.index.json, deterministic contents.
Synth make_pack(const fs::path& dir, std::uint64_t total_gib) {
    Synth s;
    fs::create_directories(dir);
    const std::uint64_t per_file = std::uint64_t{1} << 30, tb = 16 * mem::kMiB + 4096 + 512;  // odd size: unaligned offsets
    const std::uint64_t nfiles = std::max<std::uint64_t>(total_gib, 1), per = per_file / tb;
    s.tensor_bytes = tb;
    std::string wm;
    std::vector<std::uint64_t> buf(tb / 8 + 1);
    for (std::uint64_t f = 0; f < nfiles; ++f) {
        char fn[64];
        std::snprintf(fn, sizeof fn, "model-%05llu-of-%05llu.safetensors", static_cast<unsigned long long>(f + 1),
                      static_cast<unsigned long long>(nfiles));
        std::string head = "{";
        for (std::uint64_t t = 0; t < per; ++t) {
            const std::string name = "t." + std::to_string(f) + "." + std::to_string(t);
            head += (t ? "," : "") + std::string("\"") + name + "\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(tb) +
                    "],\"data_offsets\":[" + std::to_string(t * tb) + "," + std::to_string((t + 1) * tb) + "]}";
            wm += (wm.empty() ? "" : ",") + std::string("\"") + name + "\":\"" + fn + "\"";
            s.names.push_back(name);
        }
        head += "}";
        const std::string path = (dir / fn).string();
        std::ofstream o(path, std::ios::binary | std::ios::trunc);
        const std::uint64_t hb = head.size();
        o.write(reinterpret_cast<const char*>(&hb), 8);
        o << head;
        for (std::uint64_t t = 0; t < per; ++t) {
            const std::uint64_t id = f * per + t;
            for (std::uint64_t w = 0; w < buf.size(); ++w) buf[w] = pattern(id, w);
            o.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(tb));
        }
        o.close();
        s.files.push_back(path);
        s.bytes += 8 + hb + per * tb;
        drop_file_cache(path);
    }
    write_file((dir / "model.safetensors.index.json").string(), "{\"metadata\":{},\"weight_map\":{" + wm + "}}");
    return s;
}

std::uint64_t resident_all(const Synth& s) {
    std::uint64_t r = 0;
    for (const auto& f : s.files) r += resident(f);
    return r;
}

// Stream every tensor through a bounded staging buffer (chunks of `stage` bytes) and verify the contents.
bool stream_all(PackReader& pr, const Synth& s, std::uint64_t stage, mem::Monitor* mon) {
    std::vector<std::uint8_t> st(stage);
    bool ok = true;
    std::uint64_t id = 0;
    for (const auto& n : s.names) {
        const auto& t = pr.info(n);
        for (std::uint64_t o = 0; o < t.bytes; o += stage) {
            const std::uint64_t m = std::min(stage, t.bytes - o);
            pr.read(t, o, m, st.data());
            // spot-check the first and last word of the chunk (word-aligned chunks: stage is a multiple of 8)
            std::uint64_t w0 = 0, w1 = 0;
            std::memcpy(&w0, st.data(), 8);
            std::memcpy(&w1, st.data() + ((m - 8) & ~std::uint64_t{7}), 8);
            ok = ok && w0 == pattern(id, o / 8) && w1 == pattern(id, (o + ((m - 8) & ~std::uint64_t{7})) / 8);
        }
        ++id;
        if (mon) mon->counters(0, pr.bytes_read());
    }
    return ok;
}

void test_pack(const fs::path& tmp) {
    const char* g = std::getenv("GLM_MEM_TEST_GIB");
    const std::uint64_t gib_total = g ? std::strtoull(g, nullptr, 10) : 4;
    const fs::path dir = tmp / "pack";
    const auto t0 = std::chrono::steady_clock::now();
    const Synth s = make_pack(dir, gib_total);
    std::fprintf(stderr, "pack: %zu files, %zu tensors, %.2f GiB written in %.1f s\n", s.files.size(), s.names.size(), mem::gib(s.bytes),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const std::uint64_t r0 = resident_all(s);
    std::fprintf(stderr, "pack: resident before reads %.1f MiB\n", static_cast<double>(r0) / mem::kMiB);
    CHECK(r0 < s.bytes / 100);  // the generator dropped its cache

    // (a) capped budget: the plan (all tensors as device weights) does not fit -> refused before a single byte is read
    {
        PackReader pr(dir.string());
        mem::LoadPlan plan;
        for (const auto& n : s.names) plan.add(pr.info(n).bytes);
        mem::Snapshot capped;
        capped.meminfo_ok = capped.cma_free_ok = capped.zoneinfo_ok = true;
        capped.mem_available = plan.logical / 2 + 8 * mem::kGiB;  // half the pack + the margin
        std::string msg;
        try {
            mem::admit("synthetic load", {{"device plan (physical)", plan.physical(mem::AllocModel{}), 0}}, capped, mem::Margins{});
        } catch (const mem::BudgetError& e) {
            msg = e.what();
        }
        CHECK(msg.find("REFUSED synthetic load") != std::string::npos);
        CHECK(pr.bytes_read() == 0);
        // headers only: the index/header reads left no file pages behind
        CHECK(resident_all(s) < 64 * mem::kKiB * s.files.size() + r0);
    }

    // (b) O_DIRECT streaming with a 64 MiB staging buffer, memory trace on: shards' page cache stays empty, RSS bounded
    const std::string trace = (tmp / "trace.tsv").string();
    fs::remove(trace);
    const mem::Snapshot before = mem::Probe({"/proc/meminfo", "/proc/self/status", "/nonexistent"}).read();
    {
        mem::Watchdog wd;
        wd.min_ram_headroom = 0;  // the test runs on whatever box; the watchdog itself is tested above
        wd.max_swap_growth = 0;
        wd.max_rss = before.rss + 512 * mem::kMiB;  // trips if the reader ever holds whole files
        std::atomic<int> tripped{0};
        mem::Monitor mon(mem::Probe({"/proc/meminfo", "/proc/self/status", "/nonexistent"}), trace, wd,
                         [&](const std::string&) { tripped.fetch_add(1); }, 50, 1000);
        mon.phase("stream-direct");
        PackReader pr(dir.string());
        CHECK(pr.direct_files() == s.files.size());
        const auto t1 = std::chrono::steady_clock::now();
        CHECK(stream_all(pr, s, 64 * mem::kMiB, &mon));
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        CHECK(pr.bytes_read() == s.names.size() * s.tensor_bytes);
        const std::uint64_t r1 = resident_all(s);
        const mem::Snapshot after = mem::Probe({"/proc/meminfo", "/proc/self/status", "/nonexistent"}).read();
        std::fprintf(stderr, "pack: O_DIRECT %.2f GiB in %.1f s (%.2f GB/s), resident after %.1f MiB, VmHWM %.0f -> %.0f MiB\n",
                     mem::gib(pr.bytes_read()), secs, static_cast<double>(pr.bytes_read()) / secs / 1e9,
                     static_cast<double>(r1) / mem::kMiB, static_cast<double>(before.hwm) / mem::kMiB, static_cast<double>(after.hwm) / mem::kMiB);
        CHECK(r1 < s.bytes / 100);
        CHECK(after.hwm < before.hwm + 64 * mem::kMiB + PackReader::bounce_bytes() + 128 * mem::kMiB);
        CHECK(tripped.load() == 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));
        const std::string t = slurp(trace);
        std::size_t lines = 0;
        for (char c : t) lines += c == '\n';
        CHECK(lines >= 2 + static_cast<std::size_t>(secs));  // header + >= 1 line per second
        CHECK(t.find("\tstream-direct\t") != std::string::npos);
    }

    // (c) buffered fallback (STRIX_GLM_LOAD_DIRECT=0): DONTNEED after every read keeps the cache empty too
    ::setenv("STRIX_GLM_LOAD_DIRECT", "0", 1);
    {
        PackReader pr(dir.string());
        CHECK(pr.direct_files() == 0);
        CHECK(stream_all(pr, s, 64 * mem::kMiB, nullptr));
        const std::uint64_t r2 = resident_all(s);
        std::fprintf(stderr, "pack: buffered + DONTNEED resident after %.1f MiB\n", static_cast<double>(r2) / mem::kMiB);
        CHECK(r2 < s.bytes / 100);
    }
    ::unsetenv("STRIX_GLM_LOAD_DIRECT");

    // (d) control: plain buffered reads without DONTNEED DO fill the cache (proves the residency check sees it)
    {
        std::vector<std::uint8_t> st(64 * mem::kMiB);
        for (const auto& f : s.files) {
            const int fd = ::open(f.c_str(), O_RDONLY);
            while (::read(fd, st.data(), st.size()) > 0) {
            }
            ::close(fd);
        }
        const std::uint64_t r3 = resident_all(s);
        std::fprintf(stderr, "pack: control (plain reads) resident %.1f MiB of %.1f MiB\n", static_cast<double>(r3) / mem::kMiB,
                     static_cast<double>(s.bytes) / mem::kMiB);
        CHECK(r3 > s.bytes / 2);
        for (const auto& f : s.files) drop_file_cache(f);
    }

    // (e) tmpfs (O_DIRECT only on recent kernels; otherwise the open falls back with a warning): reads stay correct
    if (fs::exists("/dev/shm")) {
        const fs::path shm = fs::path("/dev/shm") / ("glm-mem-test-" + std::to_string(::getpid()));
        const Synth small = make_pack(shm, 1);
        bool refused = false;
        try {  // strict default: an O_DIRECT open failure refuses (older kernels' tmpfs), never falls back silently
            PackReader strict(shm.string());
            CHECK(stream_all(strict, small, 8 * mem::kMiB, nullptr));
        } catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("O_DIRECT open failed") != std::string::npos;
            CHECK(refused);
        }
        std::fprintf(stderr, "pack: tmpfs O_DIRECT %s\n", refused ? "refused (strict)" : "supported");
        ::setenv("STRIX_GLM_LOAD_DIRECT", "0", 1);
        PackReader pr(shm.string());
        ::unsetenv("STRIX_GLM_LOAD_DIRECT");
        CHECK(stream_all(pr, small, 8 * mem::kMiB, nullptr));
        fs::remove_all(shm);
    }
    if (std::getenv("GLM_MEM_TEST_KEEP") == nullptr) fs::remove_all(dir);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* td = std::getenv("TMPDIR");
    const fs::path base = argc > 1 ? fs::path(argv[1]) : fs::path(td && *td ? td : "/var/tmp");
    const fs::path tmp = base / ("glm-mem-tests-" + std::to_string(::getpid()));
    fs::create_directories(tmp);
    test_probe(tmp);
    test_admit();
    test_cost_model();
    test_e1_plan();
    test_monitor(tmp);
    test_pack(tmp);
    fs::remove_all(tmp);
    std::printf("glm mem cpu tests: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
