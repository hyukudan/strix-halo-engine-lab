// Q6_K down decode read discipline (src/moe_core/kquant_block_reads.hpp, used by moe_core/hip_moe_f16x_kquant_down.inc):
// every byte read for a block lies inside the block (or the leading bytes of its own aligned dword), so the last block of
// a tensor reads nothing past the tensor. Checked twice: span arithmetic for every (parity, step, half), and by replaying
// the reads byte for byte on a tensor whose last byte is the last byte before an inaccessible page.
#include "../src/moe_core/kquant_block_reads.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
namespace q6 = strix::moe_core::q6k_reads;

void check(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

void span_tests() {
    static_assert(q6::all_reads_in_block());
    // The previous d read (the dword at block - odd + 208) leaves the block for odd = 0: the rule must reject it.
    const q6::Span old_d_even{208, 212};
    check(!q6::span_ok(old_d_even, 0U), "old d dword read accepted");
    check(q6::span_ok(q6::d_span(), 0U) && q6::d_span().end == 210, "d span");
    // The reads reach the block's last ql / qh / scale bytes (the spans are not trivially short).
    int ql_max = 0, qh_max = 0, sc_max = 0;
    for (unsigned odd = 0U; odd <= 2U; odd += 2U)
        for (unsigned step = 0U; step < 4U; ++step)
            for (unsigned half = 0U; half < 2U; ++half) {
                const auto ql = q6::ql_span(odd, step, half), qh = q6::qh_span(odd, step), sc = q6::sc_span(odd, step, half);
                ql_max = ql.end > ql_max ? ql.end : ql_max;
                qh_max = qh.end > qh_max ? qh.end : qh_max;
                sc_max = sc.end > sc_max ? sc.end : sc_max;
                // the two scale bytes read are the ones the decoder uses: 192 + 8n + 2j and the next
                const unsigned at = q6::sc_at(odd, step, half) - odd;
                check(at == 192U + 8U * q6::sub_n(step) + 2U * q6::sub_j(step, half), "scale byte offset");
                check(static_cast<int>(at) + 2 <= sc.end && static_cast<int>(at) >= sc.begin, "scale bytes outside word");
            }
    check(ql_max >= 128 && ql_max <= 132 && qh_max >= 192 && qh_max <= 196 && sc_max >= 208 && sc_max <= 210,
          "span coverage");
}

struct Guarded {
    unsigned char* page = nullptr;  // readable page; the next page is inaccessible
    std::size_t size = 0;
    Guarded() {
#if defined(_WIN32)
        SYSTEM_INFO si{};
        GetSystemInfo(&si);
        size = si.dwPageSize;
        page = static_cast<unsigned char*>(VirtualAlloc(nullptr, 2U * size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        DWORD old = 0;
        if (page == nullptr || VirtualProtect(page + size, size, PAGE_NOACCESS, &old) == 0) throw std::runtime_error("guard");
#else
        size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        void* p = mmap(nullptr, 2U * size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap");
        page = static_cast<unsigned char*>(p);
        if (mprotect(page + size, size, PROT_NONE) != 0) throw std::runtime_error("mprotect");
#endif
    }
    ~Guarded() {
#if defined(_WIN32)
        VirtualFree(page, 0, MEM_RELEASE);
#else
        munmap(page, 2U * size);
#endif
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
};

// Replays the decoder's reads of one block byte for byte (memcpy of each span) and folds them into a checksum so the
// reads are not elided. A read past the tensor end touches the guard page and faults.
unsigned replay_block(const unsigned char* blk) {
    const unsigned odd = static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(blk)) & 2U;
    unsigned sum = 0U;
    unsigned char buf[64];
    auto take = [&](q6::Span s) {
        const auto n = static_cast<std::size_t>(s.end - s.begin);
        std::memcpy(buf, blk + s.begin, n);
        for (std::size_t i = 0; i < n; ++i) sum = sum * 31U + static_cast<unsigned>(buf[i]);
    };
    for (unsigned step = 0U; step < 4U; ++step)
        for (unsigned half = 0U; half < 2U; ++half) {
            take(q6::ql_span(odd, step, half));
            take(q6::qh_span(odd, step));
            take(q6::sc_span(odd, step, half));
            take(q6::d_span());
        }
    return sum;
}

void guarded_page_tests() {
    const Guarded g;
    std::memset(g.page, 0x5a, g.size);
    unsigned total = 0U;
    // Tensors of 1..7 blocks ending exactly at the guard page and 2 bytes before it (both parities of the last block).
    for (std::size_t gap = 0U; gap <= 2U; gap += 2U)
        for (std::size_t blocks = 1U; blocks <= 7U; ++blocks) {
            const std::size_t bytes = blocks * q6::block_bytes;
            const unsigned char* tensor = g.page + g.size - gap - bytes;
            for (std::size_t b = 0U; b < blocks; ++b) total += replay_block(tensor + b * q6::block_bytes);
        }
    check(total != 0U, "replay");
}
}  // namespace

int main() {
#if defined(_WIN32)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    try {
        span_tests();
        guarded_page_tests();
        std::puts("Q6_K down block reads PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
