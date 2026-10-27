// Attend union fixtures (standalone; build with the engine TU's flags: amdclang++ -O3 -std=gnu++20
// --offload-arch=gfx1151 -Iinclude -x hip). For 64-row groups of 4-row tiles, qsa_fast::union_multi_kernel in the
// merge form (STRIX_QSA_UNION_MERGE default) and in the range form must both equal a host oracle of the union
// (ascending blocks below the tile's visible block count, row masks, token 0..3 counts, zero padding to a multiple of
// four, untouched words around it), byte for byte over the whole canary-filled scratch. Cases cover dense rows,
// sparse selections up to 1,048,576 visible tokens, shared and disjoint blocks, repeated tokens, the selection's
// unordered zero fill and over-capacity lists (both take the range form), and empty rows. Exit 0 only if all match.
// Timing (after the exact checks): both forms over 128 copies of a case's 64-row group (one T8192 layer-chunk of
// union work), ABBA rounds of hipEvent-timed launches, median per form.
#include "qsa_flash_prod.inc"
#include <map>

namespace strix {
namespace {
static void check(hipError_t e, const char* what) { if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e)); std::exit(2); } }
template <class T> static T* dalloc(std::size_t n) { void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "hipMalloc"); return static_cast<T*>(p); }
struct Rng { std::uint64_t x; std::uint32_t next() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return static_cast<std::uint32_t>(x >> 16); } };

constexpr unsigned Rows = 64U, TR = 4U, kCap = 2100U;

// Host oracle of one tile's union words (header [padded, -, -, -, packed counts], then (block, mask) pairs).
static std::vector<std::uint32_t> oracle(const std::vector<std::vector<std::uint32_t>>& sel, const std::vector<std::uint32_t>& visible,
    unsigned r0, std::vector<std::uint32_t> words) {
    std::uint32_t blocks = 0;
    for (unsigned r = 0; r < TR; ++r) blocks = std::max(blocks, (visible[r0 + r] + 3U) / 4U);
    if (blocks == 0U) return words; // the kernel writes padding 0 and the counts; see below
    std::map<std::uint32_t, std::uint32_t> mask;
    std::uint32_t weight[TR * 4U] = {};
    for (unsigned r = 0; r < TR; ++r)
        for (const auto token : sel[r0 + r]) {
            if (token < 4U) ++weight[r * 4U + token];
            if (token / 4U < blocks) mask[token / 4U] |= 1U << r;
        }
    std::uint32_t at = 4U + TR, written = 0;
    for (const auto& [b, m] : mask) { words[at++] = b; words[at++] = m; ++written; }
    const std::uint32_t padded = (written + 3U) / 4U * 4U;
    for (; written < padded; ++written) { words[at++] = 0U; words[at++] = 0U; }
    words[0] = padded;
    for (unsigned r = 0; r < TR; ++r) {
        std::uint32_t packed = 0;
        for (unsigned k = 0; k < 4U; ++k) packed |= std::min(weight[r * 4U + k], 255U) << (8U * k);
        words[4U + r] = packed;
    }
    return words;
}

struct Case { const char* name; unsigned mode; };
static bool run_case(const Case& c, hipStream_t s, std::map<std::string, unsigned>& merged_tiles) {
    Rng rng{0x5E1EC7ULL * (c.mode + 1U)};
    std::vector<std::uint32_t> visible(Rows);
    std::vector<std::vector<std::uint32_t>> sel(Rows);
    for (unsigned r = 0; r < Rows; ++r) {
        auto& list = sel[r];
        const unsigned tile = r / TR;
        std::uint32_t v = 0;
        switch (c.mode) {
        case 0: v = 1U + 31U * r; for (std::uint32_t t = 0; t < v && list.size() < 2051U; ++t) list.push_back(t); break; // dense
        case 1: case 2: case 3: { // sparse: 512 blocks of a large range + tail; mode 2 shares blocks in a tile, 3 = 1M
            v = c.mode == 3U ? 1048576U - 977U * r : 300000U + 4099U * r + (r % 4U);
            const std::uint32_t complete = v / 4U;
            std::vector<std::uint32_t> blocks;
            Rng tr{0xB10CULL + (c.mode == 2U ? tile : r)};
            while (blocks.size() < 512U) {
                const std::uint32_t b = (c.mode == 2U && blocks.size() % 3U == 0U ? tr.next() : rng.next()) % complete;
                if (std::find(blocks.begin(), blocks.end(), b) == blocks.end()) blocks.push_back(b);
            }
            std::sort(blocks.begin(), blocks.end());
            for (const auto b : blocks) for (unsigned k = 0; k < 4U; ++k) list.push_back(4U * b + k);
            for (std::uint32_t t = complete * 4U; t < v; ++t) list.push_back(t);
            break;
        }
        case 4: // repeated tokens (non-decreasing), some rows empty
            v = 220000U + 13U * r; // above the merge threshold
            if (r % 5U != 4U) for (std::uint32_t t = 0; t < v && list.size() < 2040U; t += 1U + rng.next() % 9U) { list.push_back(t); if (rng.next() % 4U == 0U) list.push_back(t); }
            break;
        case 5: // the selection's zero fill (tokens 0..3 repeated before the ascending part) in one row per tile
            v = 250000U + 7U * r;
            if (r % TR == 1U) for (unsigned z = 0; z < 8U; ++z) list.push_back(z % 4U);
            for (std::uint32_t t = 64U; t < v && list.size() < 2051U; ++t) if (rng.next() % 3U == 0U) list.push_back(t);
            break;
        case 6: // over the merge capacity in one row of every other tile
            v = 230000U + r;
            for (std::uint32_t t = 0; t < v && list.size() < (r % 8U == 2U ? 2080U : 2051U); ++t) if (rng.next() % 2U == 0U) list.push_back(t);
            break;
        default: // tiles whose rows all see no complete block, and tiles of tiny rows
            v = r % 8U < 4U ? 0U : 1U + r % 3U;
            for (std::uint32_t t = 0; t < v; ++t) list.push_back(t);
            break;
        }
        visible[r] = v;
    }
    const std::size_t per_row = 16384U; // words per row (production rows hold heads x 2051 floats)
    std::vector<std::uint32_t> host_sel(Rows * kCap, 0xFFFFFFFFU);
    for (unsigned r = 0; r < Rows; ++r) std::copy(sel[r].begin(), sel[r].end(), host_sel.begin() + static_cast<std::ptrdiff_t>(r * kCap));
    auto* d_sel = dalloc<std::uint32_t>(host_sel.size());
    check(hipMemcpy(d_sel, host_sel.data(), host_sel.size() * 4, hipMemcpyHostToDevice), "h2d sel");
    std::vector<std::uint32_t> canary(Rows * per_row);
    for (std::size_t i = 0; i < canary.size(); ++i) canary[i] = 0xC0DE0000U ^ static_cast<std::uint32_t>(i * 2654435761U);
    auto* d_a = dalloc<std::uint32_t>(canary.size()); auto* d_b = dalloc<std::uint32_t>(canary.size());
    check(hipMemcpy(d_a, canary.data(), canary.size() * 4, hipMemcpyHostToDevice), "h2d a");
    check(hipMemcpy(d_b, canary.data(), canary.size() * 4, hipMemcpyHostToDevice), "h2d b");
    const std::uint32_t union_at = 16U;
    qsa_fast::Views<Rows> va{}, vb{};
    for (unsigned r = 0; r < Rows; ++r) {
        va.selected[r] = vb.selected[r] = d_sel + r * kCap;
        va.scratch[r] = reinterpret_cast<float*>(d_a + r * per_row);
        vb.scratch[r] = reinterpret_cast<float*>(d_b + r * per_row);
        va.count[r] = vb.count[r] = static_cast<std::uint32_t>(sel[r].size());
        va.visible[r] = vb.visible[r] = visible[r];
    }
    auto* d_va = dalloc<qsa_fast::Views<Rows>>(1); auto* d_vb = dalloc<qsa_fast::Views<Rows>>(1);
    check(hipMemcpy(d_va, &va, sizeof(va), hipMemcpyHostToDevice), "h2d va");
    check(hipMemcpy(d_vb, &vb, sizeof(vb), hipMemcpyHostToDevice), "h2d vb");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, true>), dim3(Rows / TR), dim3(256), 0, s, d_va, union_at);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, false>), dim3(Rows / TR), dim3(256), 0, s, d_vb, union_at);
    check(hipGetLastError(), "launch");
    check(hipStreamSynchronize(s), "sync");
    std::vector<std::uint32_t> a(canary.size()), b(canary.size());
    check(hipMemcpy(a.data(), d_a, a.size() * 4, hipMemcpyDeviceToHost), "d2h a");
    check(hipMemcpy(b.data(), d_b, b.size() * 4, hipMemcpyDeviceToHost), "d2h b");
    bool ok = true;
    unsigned merge_eligible = 0;
    for (unsigned tile = 0; tile < Rows / TR; ++tile) {
        const unsigned r0 = tile * TR;
        std::vector<std::uint32_t> words(canary.begin() + static_cast<std::ptrdiff_t>(r0 * per_row) + union_at,
            canary.begin() + static_cast<std::ptrdiff_t>((r0 + 1U) * per_row));
        auto expect = oracle(sel, visible, r0, words);
        std::uint32_t blocks = 0;
        for (unsigned r = 0; r < TR; ++r) blocks = std::max(blocks, (visible[r0 + r] + 3U) / 4U);
        if (blocks == 0U) { expect[0] = 0U; for (unsigned r = 0; r < TR; ++r) expect[4U + r] = 0U; }
        bool eligible = blocks > qsa_fast::kMergeMinBlocks;
        for (unsigned r = 0; r < TR; ++r) {
            eligible = eligible && sel[r0 + r].size() <= qsa_fast::kMergeCap && std::is_sorted(sel[r0 + r].begin(), sel[r0 + r].end());
        }
        merge_eligible += eligible ? 1U : 0U;
        for (std::size_t i = 0; i < expect.size(); ++i) {
            const std::size_t at = r0 * per_row + union_at + i;
            if (a[at] != expect[i] || b[at] != expect[i]) {
                std::fprintf(stderr, "FAIL %s tile %u word %zu: merge %08x range %08x expect %08x\n", c.name, tile, i, a[at], b[at], expect[i]);
                ok = false;
                break;
            }
        }
        for (unsigned r = 1; r < TR && ok; ++r) // rows other than the tile's first are never written
            for (std::size_t i = 0; i < per_row; ++i)
                if (a[(r0 + r) * per_row + i] != canary[(r0 + r) * per_row + i]) { std::fprintf(stderr, "FAIL %s row %u written\n", c.name, r0 + r); ok = false; break; }
        for (std::size_t i = 0; i < union_at && ok; ++i)
            if (a[r0 * per_row + i] != canary[r0 * per_row + i]) { std::fprintf(stderr, "FAIL %s tile %u below union_at\n", c.name, tile); ok = false; }
    }
    merged_tiles[c.name] = merge_eligible;
    if (ok && (c.mode == 0U || c.mode == 1U || c.mode == 3U)) {
        constexpr unsigned kGroups = 128U, kRounds = 5U;
        std::vector<qsa_fast::Views<Rows>> many(kGroups, va);
        auto* d_many = dalloc<qsa_fast::Views<Rows>>(kGroups);
        check(hipMemcpy(d_many, many.data(), kGroups * sizeof(va), hipMemcpyHostToDevice), "h2d many");
        hipEvent_t e0, e1;
        check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
        const auto timed = [&](bool merge) {
            check(hipEventRecord(e0, s), "record");
            if (merge) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, true>), dim3(kGroups * (Rows / TR)), dim3(256), 0, s, d_many, union_at);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, false>), dim3(kGroups * (Rows / TR)), dim3(256), 0, s, d_many, union_at);
            check(hipEventRecord(e1, s), "record"); check(hipEventSynchronize(e1), "sync");
            float ms = 0.0F; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            return ms;
        };
        timed(true); timed(false);
        std::vector<float> tm, tr;
        for (unsigned i = 0; i < kRounds; ++i) { tm.push_back(timed(true)); tr.push_back(timed(false)); tr.push_back(timed(false)); tm.push_back(timed(true)); }
        std::sort(tm.begin(), tm.end()); std::sort(tr.begin(), tr.end());
        std::printf("TIME %s 128 groups: merge %.3f ms, range %.3f ms (%+.1f %%)\n", c.name, tm[tm.size() / 2], tr[tr.size() / 2],
            100.0 * (tm[tm.size() / 2] - tr[tr.size() / 2]) / tr[tr.size() / 2]);
        check(hipEventDestroy(e0), "event"); check(hipEventDestroy(e1), "event");
        check(hipFree(d_many), "free");
    }
    check(hipFree(d_sel), "free"); check(hipFree(d_a), "free"); check(hipFree(d_b), "free");
    check(hipFree(d_va), "free"); check(hipFree(d_vb), "free");
    std::printf("%s %s merge-eligible tiles %u/%u\n", ok ? "PASS" : "FAIL", c.name, merge_eligible, Rows / TR);
    return ok;
}
} // namespace
} // namespace strix

int main() {
    using namespace strix;
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) { std::puts("SKIP no HIP device"); return 77; }
    hipStream_t s;
    check(hipStreamCreate(&s), "stream");
    const Case cases[] = {{"dense", 0}, {"sparse-300k", 1}, {"shared-blocks", 2}, {"sparse-1m", 3}, {"repeats-empty", 4},
        {"zero-fill", 5}, {"over-capacity", 6}, {"no-complete-block", 7}};
    std::map<std::string, unsigned> merged;
    bool ok = true;
    for (const auto& c : cases) ok = run_case(c, s, merged) && ok;
    // The merge form must actually run for the sorted cases and be declined for the unsorted / over-capacity tiles.
    ok = ok && merged["sparse-1m"] == Rows / TR && merged["sparse-300k"] == Rows / TR && merged["dense"] == 0U && merged["zero-fill"] == 0U &&
        merged["repeats-empty"] == Rows / TR && merged["over-capacity"] == Rows / TR / 2U;
    check(hipStreamDestroy(s), "stream");
    std::puts(ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
