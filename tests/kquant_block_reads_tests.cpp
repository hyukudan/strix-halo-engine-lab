// Q6_K down decode read discipline (src/moe_core/kquant_block_reads.hpp, used by moe_core/hip_moe_f16x_kquant_down.inc):
// every byte read for a block lies inside the block (or the leading bytes of its own aligned dword), so the last block of
// a tensor reads nothing past the tensor. Checked twice: span arithmetic for every (parity, step pair, halves), and by replaying
// the reads byte for byte on a tensor whose last byte is the last byte before an inaccessible page.
// Q4_K header reuse (q4k_reads): the K loop's two register stages are replayed on a row of random superblocks; every step
// must decode its own superblock's header and qs bytes, with the header requested only on steps s%4 < 2.
#include "../src/moe_core/kquant_block_reads.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    // A scale tail read at base + 200 + 8n also for odd = 0 would leave the block (n = 1): the rule must reject it.
    check(!q6::span_ok(q6::from_base(0U, 208U, 4U), 0U), "even-parity scale tail past the block accepted");
    // The reads reach the block's last ql / qh / scale bytes (the spans are not trivially short).
    int ql_max = 0, qh_max = 0, sc_max = 0;
    for (unsigned odd = 0U; odd <= 2U; odd += 2U)
        for (unsigned step = 0U; step < 4U; step += 2U) {
            const unsigned n = q6::sub_n(step);
            check(q6::sub_n(step + 1U) == n, "a pair shares its superblock quarter");
            for (unsigned halves = 1U; halves <= 2U; ++halves)
                for (unsigned first = 0U; first + halves <= 2U; ++first) {
                    const auto ql = q6::ql_span(odd, step, first, halves);
                    // realigned word 0 starts at the first weight byte of the first half: block byte 64n + 32 first
                    check(ql.begin + static_cast<int>(odd) == static_cast<int>(64U * n + 32U * first), "ql pair offset");
                    check(ql.end - ql.begin == static_cast<int>(4U * (8U * halves + 1U)), "ql pair length");
                    ql_max = ql.end > ql_max ? ql.end : ql_max;
                }
            const auto qh = q6::qh_span(odd, step), sp = q6::sc_pair_span(odd, step), st = q6::sc_tail_span(odd, step);
            check(qh.begin + static_cast<int>(odd) == static_cast<int>(128U + 32U * n), "qh offset");
            qh_max = qh.end > qh_max ? qh.end : qh_max;
            sc_max = sp.end > sc_max ? sp.end : sc_max;
            sc_max = st.end > sc_max ? st.end : sc_max;
            // the scale bytes the decoder picks for (step parity, half) are 192 + 8n + 2j and the next
            for (unsigned p = 0U; p < 2U; ++p)
                for (unsigned half = 0U; half < 2U; ++half)
                    check(192U + 8U * n + q6::sc_byte(step + p, half) == 192U + 8U * n + 2U * q6::sub_j(step + p, half) &&
                              q6::sc_byte(step + p, half) + 1U < 8U,
                          "scale byte offset");
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
    unsigned char buf[128];
    auto take = [&](q6::Span s) {
        const auto n = static_cast<std::size_t>(s.end - s.begin);
        std::memcpy(buf, blk + s.begin, n);
        for (std::size_t i = 0; i < n; ++i) sum = sum * 31U + static_cast<unsigned>(buf[i]);
    };
    for (unsigned step = 0U; step < 4U; step += 2U) {
        take(q6::ql_span(odd, step, 0U, 2U));
        take(q6::ql_span(odd, step, 0U, 1U));
        take(q6::ql_span(odd, step, 1U, 1U));
        take(q6::qh_span(odd, step));
        take(q6::sc_pair_span(odd, step));
        take(q6::sc_tail_span(odd, step));
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

// One register stage of the Q4_K down trait: the bytes F16xKQuantDown<12>::load leaves in Raw {head, lo, hi}.
struct Q4Stage { unsigned char head[16], qs[32]; };
void q4k_replay_tests() {
    namespace q4 = strix::moe_core::q4k_reads;
    static_assert(q4::headers_held(4U) && q4::headers_held(24U) && !q4::headers_held(6U));
    std::mt19937 rng(7U);
    for (unsigned blocks = 1U; blocks <= 9U; ++blocks) {
        std::string row(static_cast<std::size_t>(blocks) * q4::block_bytes, '\0');
        for (auto& c : row) c = static_cast<char>(rng() & 255U);
        const auto* base = reinterpret_cast<const unsigned char*>(row.data());
        Q4Stage stage[2];
        std::memset(stage, 0xA5, sizeof(stage));  // never-written header bytes must not reach a decode
        unsigned header_loads = 0U, qs_bytes = 0U;
        const auto fetch = [&](unsigned s) {
            const unsigned char* blk = base + static_cast<std::size_t>(s / 4U) * q4::block_bytes;
            Q4Stage& r = stage[s % 2U];
            if (q4::header_load(s)) { std::memcpy(r.head, blk, q4::header_bytes); ++header_loads; }
            std::memcpy(r.qs, blk + q4::qs_from_block(s), 32U);
            qs_bytes += 32U;
        };
        const auto decode = [&](unsigned s) {
            const unsigned char* blk = base + static_cast<std::size_t>(s / 4U) * q4::block_bytes;
            const Q4Stage& r = stage[s % 2U];
            check(std::memcmp(r.head, blk, q4::header_bytes) == 0, "Q4_K stage holds another superblock's header");
            check(std::memcmp(r.qs, blk + 16U + 32U * (s % 4U), 32U) == 0, "Q4_K stage holds the wrong qs bytes");
        };
        const unsigned steps = 4U * blocks;
        fetch(0U);
        fetch(1U);
        for (unsigned s = 0U; s < steps; s += 2U) {
            decode(s);
            if (s + 2U < steps) fetch(s + 2U);
            decode(s + 1U);
            if (s + 3U < steps) fetch(s + 3U);
        }
        check(header_loads == 2U * blocks, "Q4_K header requested other than twice per superblock");
        check(qs_bytes == 128U * blocks, "Q4_K qs bytes not read exactly once");
    }
}
}  // namespace

int main() {
#if defined(_WIN32)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    try {
        span_tests();
        guarded_page_tests();
        q4k_replay_tests();
        std::puts("Q6_K down block reads + Q4_K header reuse replay PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
