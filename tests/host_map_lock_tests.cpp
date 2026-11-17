// host_map_lock.hpp: which /proc/self/maps lines are the runtime's shmem host memory, and that a pass locks only the
// maps that are new (a replaced or refused map is handled once). On Linux also a live pass over this process.
#include "strix/host_map_lock.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace {
namespace hml = strix::host_map_lock;

void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("host map lock check failed: ") + what);
}

bool takes(const char* line) {
    hml::Range r;
    return hml::runtime_shared_line(line, r);
}

void parser() {
    hml::Range r;
    check(hml::runtime_shared_line("7f1234000000-7f1234200000 rw-s 00000000 00:01 4097                       /dev/zero (deleted)", r),
          "ROCr shmem host allocation");
    check(r.begin == 0x7f1234000000ULL && r.end == 0x7f1234200000ULL, "range bounds");
    check(takes("7f0000001000-7f0000002000 rw-s 00000000 00:01 12 /memfd:hsa (deleted)\n"), "memfd, trailing newline");
    check(takes("7f0000001000-7f0000002000 rw-s 00000000 00:01 12 /SYSV00000000 (deleted)\r\n"), "SysV segment, CRLF");
    check(takes("7f0000001000-7f0000002000 rw-s 00000000 00:01 12 [anon_shmem:queue]"), "named anonymous shmem");
    check(takes("ffffffffff600000-ffffffffff601000 rw-s 00000000 00:01 12 /dev/zero (deleted)"), "16 hex digits");
    // Not the runtime's shmem: private, read-only, device files (TTM BOs never migrate), file maps, plain anonymous.
    check(!takes("7f1234000000-7f1234200000 rw-p 00000000 00:01 4097 /dev/zero (deleted)"), "private /dev/zero");
    check(!takes("7f1234000000-7f1234200000 r--s 00000000 00:01 4097 /dev/zero (deleted)"), "read-only shared");
    check(!takes("7f1234000000-7f1234200000 rw-s 1a000000 00:06 512 /dev/kfd"), "kfd doorbells");
    check(!takes("7f1234000000-7f1234200000 rw-s 10f000000 00:06 600 /dev/dri/renderD128"), "render node BO");
    check(!takes("7f1234000000-7f1234200000 rw-s 00000000 103:02 77 /home/user/model.gguf"), "file mapping");
    check(!takes("7f1234000000-7f1234200000 rw-s 00000000 00:00 0"), "no path");
    check(!takes("7f1234000000-7f1234200000 rw-p 00000000 00:00 0 [heap]"), "heap");
    // Malformed lines are skipped, never misread.
    check(!takes(""), "empty");
    check(!takes("garbage"), "one field");
    check(!takes("7f1234200000-7f1234000000 rw-s 00000000 00:01 4097 /dev/zero (deleted)"), "inverted range");
    check(!takes("7f12340g0000-7f1234200000 rw-s 00000000 00:01 4097 /dev/zero (deleted)"), "bad hex");
    check(!takes("117f1234000000000-127f1234200000000 rw-s 00000000 00:01 4097 /dev/zero (deleted)"), "too many digits");

    const std::string maps =
        "55d0c0000000-55d0c0021000 r-xp 00000000 103:02 11 /usr/bin/strix\n"
        "7f0000000000-7f0000200000 rw-s 00000000 00:01 4097 /dev/zero (deleted)\n"
        "7f0000200000-7f0000400000 rw-p 00000000 00:00 0\n"
        "7f0000400000-7f0000401000 rw-s 00000000 00:01 4098 /dev/zero (deleted)\n"
        "7f0000401000-7f0000402000 rw-s 00000000 00:06 512 /dev/kfd";  // last line without a newline
    const std::vector<hml::Range> got = hml::runtime_shared_ranges(maps);
    check(got.size() == 2, "two runtime maps in the text");
    check(got[0].begin == 0x7f0000000000ULL && got[1].end == 0x7f0000401000ULL, "file order and bounds");
}

void tracker() {
    hml::Tracker t;
    std::vector<hml::Range> locked;
    const auto lock = [&](const hml::Range& r) {
        locked.push_back(r);
        return r.begin != 0x3000;  // this one is refused
    };
    const std::vector<hml::Range> a{{0x1000, 0x2000}, {0x3000, 0x4000}};
    hml::Pass p = t.apply(a, lock);
    check(p.maps == 2 && p.bytes == 0x2000 && p.fresh == 1 && p.refused == 1 && locked.size() == 2, "first pass locks all");
    p = t.apply(a, lock);
    check(p.maps == 2 && p.fresh == 0 && p.refused == 0 && locked.size() == 2, "unchanged maps: no lock call (refused not retried)");
    const std::vector<hml::Range> b{{0x1000, 0x2800}, {0x5000, 0x6000}};  // replaced at the same address + a new one
    p = t.apply(b, lock);
    check(p.fresh == 2 && locked.size() == 4 && locked[2].end == 0x2800 && locked[3].begin == 0x5000, "replaced and new maps locked");
    p = t.apply({{0x3000, 0x4000}}, lock);
    check(p.refused == 1 && locked.size() == 5, "a map that left and came back is tried again");
}

#if defined(__linux__)
void live() {
    const std::size_t bytes = std::size_t{1} << 16;
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    check(p != MAP_FAILED, "shared anonymous mmap");
    static_cast<char*>(p)[0] = 1;
    const auto begin = reinterpret_cast<std::uintptr_t>(p);
    bool seen = false;
    for (const hml::Range& r : hml::runtime_shared_ranges(hml::read_self_maps())) seen |= r.begin <= begin && begin < r.end;
    check(seen, "this process's shared anonymous map is listed as /dev/zero (deleted) rw-s");
    hml::Tracker t;
    const hml::Pass first = hml::lock_runtime_shared_maps(t);
    check(first.maps >= 1 && first.fresh + first.refused == first.maps, "every map tried once");
    const hml::Pass second = hml::lock_runtime_shared_maps(t);
    check(second.fresh == 0 && second.refused == 0, "second pass makes no lock call");
    ::munmap(p, bytes);
    std::printf("live: %zu runtime shmem maps, %zu locked, %zu refused\n", first.maps, first.fresh, first.refused);
}
#endif
}  // namespace

int main() try {
    parser();
    tracker();
#if defined(__linux__)
    live();
#endif
    std::puts("host map lock tests: ok");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
