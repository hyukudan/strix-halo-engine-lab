// Attend union fixtures (standalone; build with the engine TU's flags: amdclang++ -O3 -std=gnu++20
// --offload-arch=gfx1151 -Iinclude -x hip). For 64-row groups of 4-row tiles, qsa_fast::union_multi_kernel in the
// merge form (STRIX_QSA_UNION_MERGE default) and in the range form must both equal a host oracle of the union
// (ascending blocks below the tile's visible block count, row masks, token 0..3 counts, zero padding to a multiple of
// four, untouched words around it), byte for byte over the whole canary-filled scratch. Cases cover dense rows,
// sparse selections up to 1,048,576 visible tokens, shared and disjoint blocks, repeated tokens, the selection's
// unordered zero fill and over-capacity lists (both take the range form), and empty rows. Exit 0 only if all match.
// Timing (after the exact checks): both forms over 128 copies of a case's 64-row group (one T8192 layer-chunk of
// union work), ABBA rounds of hipEvent-timed launches, median per form.
// Also checks diagnostic device reductions against a host recount, nonbinding cap restoration, and binding caps
// against a comparison-sort oracle (16/32/64 rows, tails, ties, nonfinites, empty rows, multiplicities, both builders).
// qsa_flash_fixtures.hip's default smoke / --attend-occ-only checks complete attention bytes for a nonbinding cap.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
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

static unsigned host_score_key(float value) {
    if (std::isnan(value)) value = -INFINITY;
    unsigned bits = 0U; std::memcpy(&bits, &value, sizeof(bits));
    if (!(bits & 0x7fffffffU)) bits = 0U;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
static unsigned host_memberships(unsigned mask) {
    unsigned count = 0U;
    for (unsigned r = 0; r < TR; ++r) count += (mask >> r) & 1U;
    return count;
}
// Independent comparison sort oracle; production uses radix thresholding and ascending compaction.
static void cap_oracle(std::vector<std::uint32_t>& words, const std::vector<float>& scores,
    const std::vector<unsigned>& score_blocks, const std::vector<unsigned>& visible,
    unsigned stride, unsigned r0, unsigned cap) {
    struct Entry { unsigned block, mask, best; bool keep; };
    std::vector<Entry> entries;
    const auto old_padded = words[0];
    for (unsigned i = 0; i < old_padded; ++i) {
        const auto at = 4U + TR + 2U * i;
        if (words[at + 1U]) entries.push_back({words[at], words[at + 1U], host_score_key(-INFINITY), false});
    }
    if (!cap || entries.size() <= cap) return;
    for (auto& entry : entries) for (unsigned r = 0; r < TR; ++r) {
        if ((entry.mask & (1U << r)) && entry.block < score_blocks[r0 + r])
            entry.best = std::max(entry.best, host_score_key(scores[(r0 + r) * stride + entry.block]));
        if (((entry.mask & (1U << r)) && visible[r0 + r] && entry.block == (visible[r0 + r] - 1U) / 4U) ||
            (!entry.block && words[4U + r])) entry.keep = true;
    }
    for (unsigned r = 0; r < TR; ++r) {
        std::size_t chosen = entries.size();
        unsigned best = 0U;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& entry = entries[i];
            if (!(entry.mask & (1U << r)) || 4U * entry.block >= visible[r0 + r]) continue;
            const auto key = entry.block < score_blocks[r0 + r] ?
                host_score_key(scores[(r0 + r) * stride + entry.block]) : host_score_key(-INFINITY);
            if (chosen == entries.size() || key > best) { best = key; chosen = i; }
        }
        if (chosen < entries.size()) entries[chosen].keep = true;
    }
    std::vector<std::size_t> ranked;
    unsigned kept = 0U;
    for (std::size_t i = 0; i < entries.size(); ++i) { if (entries[i].keep) ++kept; else ranked.push_back(i); }
    std::sort(ranked.begin(), ranked.end(), [&](std::size_t a, std::size_t b) {
        const auto& x = entries[a]; const auto& y = entries[b];
        const auto xm = host_memberships(x.mask), ym = host_memberships(y.mask);
        if (xm != ym) return xm > ym;
        if (x.best != y.best) return x.best > y.best;
        return x.block < y.block;
    });
    for (const auto i : ranked) if (kept < cap) { entries[i].keep = true; ++kept; }
    std::fill(words.begin() + 4U + TR, words.begin() + 4U + TR + 2U * old_padded, 0U);
    unsigned at = 4U + TR;
    for (const auto& entry : entries) if (entry.keep) { words[at++] = entry.block; words[at++] = entry.mask; }
    words[0] = (kept + 3U) / 4U * 4U;
}

template <unsigned TestRows>
static bool capped_cases(hipStream_t stream, unsigned mode) {
    constexpr unsigned Stride = 96U, PerRow = 2048U, UnionAt = 16U;
    std::vector<unsigned> visible(TestRows), score_blocks(TestRows);
    std::vector<std::vector<unsigned>> selected(TestRows);
    std::vector<float> scores(TestRows * Stride);
    for (unsigned r = 0; r < TestRows; ++r) {
        visible[r] = 241U + 3U * (r % TR); // distinct partial tails and an aligned causal boundary
        if (mode == 3U) visible[r] = 220001U + 3U * (r % TR); // exercise the merge builder before the same cap
        score_blocks[r] = mode == 2U ? 0U : std::min(Stride, visible[r] / 4U);
        const unsigned candidates[] = {0U, 1U + r % TR, 5U + r % 2U, 9U, 11U, 15U + r % TR,
            23U + r % 3U, 29U + r % TR, 40U + r % TR};
        for (const auto b : candidates) for (unsigned k = 0; k < 4U; ++k) selected[r].push_back(b * 4U + k);
        for (unsigned t = visible[r] / 4U * 4U; t < visible[r]; ++t) selected[r].push_back(t);
        if (mode == 1U && r % 3U == 1U) { selected[r].push_back(0U); selected[r].push_back(0U); }
        if (mode == 2U && r % 7U == 0U) selected[r].clear(); // no invented selection for an empty row
        if (mode != 1U) std::sort(selected[r].begin(), selected[r].end());
        for (unsigned b = 0; b < Stride; ++b) {
            float value = mode == 0U ? 1.0F : static_cast<float>(static_cast<int>((b * 17U + r * 11U) % 23U) - 11);
            if (mode == 1U) {
                if (b % 11U == 0U) value = std::numeric_limits<float>::quiet_NaN();
                else if (b % 13U == 0U) value = INFINITY;
                else if (b % 7U == 0U) value = -INFINITY;
                else if (b % 5U == 0U) value = r % 2U ? -0.0F : 0.0F;
            }
            scores[r * Stride + b] = value;
        }
    }
    std::vector<unsigned> flat(TestRows * 64U, UINT32_MAX), canary(TestRows * PerRow, 0xcede1234U);
    for (unsigned r = 0; r < TestRows; ++r) std::copy(selected[r].begin(), selected[r].end(), flat.begin() + r * 64U);
    auto* d_selected = dalloc<unsigned>(flat.size()); auto* d_scratch = dalloc<unsigned>(canary.size());
    auto* d_scores = dalloc<float>(scores.size());
    check(hipMemcpy(d_selected, flat.data(), flat.size() * sizeof(unsigned), hipMemcpyHostToDevice), "cap selections");
    check(hipMemcpy(d_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice), "cap scores");
    qsa_fast::Views<TestRows> views{};
    for (unsigned r = 0; r < TestRows; ++r) {
        views.selected[r] = d_selected + r * 64U; views.count[r] = static_cast<unsigned>(selected[r].size());
        views.scratch[r] = reinterpret_cast<float*>(d_scratch + r * PerRow); views.visible[r] = visible[r];
        views.block_scores[r] = d_scores + r * Stride; views.score_blocks[r] = score_blocks[r];
    }
    std::vector<unsigned> original = canary;
    unsigned largest = 0U;
    for (unsigned tile = 0; tile < TestRows / TR; ++tile) {
        const unsigned r0 = tile * TR, at = r0 * PerRow + UnionAt;
        std::vector<unsigned> words(original.begin() + at, original.begin() + (r0 + 1U) * PerRow);
        words = oracle(selected, visible, r0, words);
        unsigned n = words[0];
        while (n && !words[4U + TR + 2U * (n - 1U) + 1U]) --n;
        largest = std::max(largest, n);
        std::copy(words.begin(), words.end(), original.begin() + at);
    }
    bool ok = true;
    const unsigned caps[] = {0U, 1U, 4U, 9U, 13U, largest, largest + 1U};
    for (const auto cap : caps) {
        auto expected = original;
        for (unsigned tile = 0; tile < TestRows / TR; ++tile) {
            const unsigned r0 = tile * TR, at = r0 * PerRow + UnionAt;
            std::vector<unsigned> words(expected.begin() + at, expected.begin() + (r0 + 1U) * PerRow);
            cap_oracle(words, scores, score_blocks, visible, Stride, r0, cap);
            std::copy(words.begin(), words.end(), expected.begin() + at);
        }
        for (unsigned repeat = 0; repeat < 4U; ++repeat) {
            check(hipMemcpyAsync(d_scratch, canary.data(), canary.size() * sizeof(unsigned), hipMemcpyHostToDevice, stream), "cap reset");
            if (repeat % 2U)
                STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_kernel<TestRows, TR, true, true>), dim3(TestRows / TR), dim3(256), 0, stream,
                    views, UnionAt, cap);
            else
                STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_kernel<TestRows, TR, false, true>), dim3(TestRows / TR), dim3(256), 0, stream,
                    views, UnionAt, cap);
            check(hipStreamSynchronize(stream), "cap sync");
            std::vector<unsigned> got(canary.size());
            check(hipMemcpy(got.data(), d_scratch, got.size() * sizeof(unsigned), hipMemcpyDeviceToHost), "cap result");
            if (got != expected) { std::fprintf(stderr, "FAIL cap rows %u mode %u cap %u repeat %u oracle\n", TestRows, mode, cap, repeat); ok = false; }
            for (unsigned tile = 0; tile < TestRows / TR; ++tile) {
                const auto at = tile * TR * PerRow + UnionAt;
                unsigned survivors = 0U;
                for (unsigned i = 0; i < got[at]; ++i) survivors |= got[at + 4U + TR + 2U * i + 1U];
                for (unsigned r = 0; r < TR; ++r) if (!selected[tile * TR + r].empty() && !(survivors & (1U << r))) {
                    std::fprintf(stderr, "FAIL cap erased row %u\n", tile * TR + r); ok = false;
                }
            }
        }
    }
    std::vector<float> after(scores.size());
    check(hipMemcpy(after.data(), d_scores, after.size() * sizeof(float), hipMemcpyDeviceToHost), "cap score reads");
    ok = ok && std::memcmp(after.data(), scores.data(), scores.size() * sizeof(float)) == 0;
    check(hipFree(d_selected), "free"); check(hipFree(d_scratch), "free"); check(hipFree(d_scores), "free");
    std::printf("%s cap rows %u mode %u ties/tails/multiplicity/survival/host-sort\n", ok ? "PASS" : "FAIL", TestRows, mode);
    return ok;
}

static bool preserved_selection_cases(hipStream_t stream) {
    constexpr unsigned TestRows = 16U, Blocks = 64U, TopK = 4U, SelectedPitch = 32U, MaximaPitch = Blocks / 4U;
    std::vector<float> scores(TestRows * Blocks), maxima(TestRows * MaximaPitch);
    std::vector<unsigned> visible(TestRows), poison(TestRows * SelectedPitch, 0xcede1234U);
    for (unsigned r = 0; r < TestRows; ++r) {
        visible[r] = r % 4U == 0U ? 19U : 259U;
        for (unsigned b = 0; b < Blocks; ++b) {
            float value = r % 5U == 0U ? -INFINITY : r % 5U == 1U ? 1.0F : static_cast<float>((b * 13U + r) % 17U);
            if (r % 5U == 3U && b % 7U == 0U) value = std::numeric_limits<float>::quiet_NaN();
            scores[r * Blocks + b] = value;
        }
        for (unsigned g = 0; g < MaximaPitch; ++g) {
            const auto* row = scores.data() + r * Blocks + 4U * g;
            maxima[r * MaximaPitch + g] = std::fmax(std::fmax(row[0], row[1]), std::fmax(row[2], row[3]));
        }
    }
    auto* d_maxima = dalloc<float>(maxima.size());
    check(hipMemcpy(d_maxima, maxima.data(), maxima.size() * sizeof(float), hipMemcpyHostToDevice), "preserved maxima");
    float* d_scores[3]{}; unsigned* d_selected[3]{};
    QsaRowsSelectViews<TestRows> views[3]{};
    for (unsigned arm = 0; arm < 3U; ++arm) {
        d_scores[arm] = dalloc<float>(scores.size()); d_selected[arm] = dalloc<unsigned>(poison.size());
        check(hipMemcpy(d_scores[arm], scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice), "preserved scores");
        check(hipMemcpy(d_selected[arm], poison.data(), poison.size() * sizeof(unsigned), hipMemcpyHostToDevice), "preserved selected reset");
        views[arm].preserve_scores = arm != 0U;
        for (unsigned r = 0; r < TestRows; ++r) {
            views[arm].block_scores[r] = d_scores[arm] + r * Blocks;
            views[arm].selected[r] = d_selected[arm] + r * SelectedPitch;
            views[arm].visible[r] = visible[r];
        }
    }
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(select_rows_kernel<TestRows>), dim3(TestRows), dim3(kSelectThreads), 0, stream, views[0], 4U, TopK);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(select_rows_kernel<TestRows>), dim3(TestRows), dim3(kSelectThreads), 0, stream, views[1], 4U, TopK);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_flash::select_kernel<TestRows>), dim3(TestRows), dim3(kSelectThreads), 0, stream,
        views[2], d_maxima, MaximaPitch, 4U, TopK);
    check(hipStreamSynchronize(stream), "preserved selection sync");
    std::vector<unsigned> reference(poison.size()), selected(poison.size());
    check(hipMemcpy(reference.data(), d_selected[0], reference.size() * sizeof(unsigned), hipMemcpyDeviceToHost), "preserved reference");
    bool ok = true;
    for (unsigned arm = 1; arm < 3U; ++arm) {
        std::vector<float> after(scores.size());
        check(hipMemcpy(after.data(), d_scores[arm], after.size() * sizeof(float), hipMemcpyDeviceToHost), "preserved scores result");
        check(hipMemcpy(selected.data(), d_selected[arm], selected.size() * sizeof(unsigned), hipMemcpyDeviceToHost), "preserved selected result");
        ok = ok && selected == reference && std::memcmp(after.data(), scores.data(), scores.size() * sizeof(float)) == 0;
    }
    std::vector<qsa_flash::SelectRow> batch(TestRows);
    for (unsigned r = 0; r < TestRows; ++r) batch[r] = {views[2].block_scores[r], views[2].selected[r],
        d_maxima + r * MaximaPitch, visible[r], 1U};
    auto* d_batch = dalloc<qsa_flash::SelectRow>(batch.size());
    check(hipMemcpy(d_batch, batch.data(), batch.size() * sizeof(batch[0]), hipMemcpyHostToDevice), "preserved batch");
    check(hipMemcpy(d_selected[2], poison.data(), poison.size() * sizeof(unsigned), hipMemcpyHostToDevice), "preserved batch reset");
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, qsa_flash::select_batch_kernel, dim3(TestRows), dim3(kSelectThreads), 0, stream, d_batch, 4U, TopK);
    check(hipStreamSynchronize(stream), "preserved batch sync");
    std::vector<float> after(scores.size());
    check(hipMemcpy(after.data(), d_scores[2], after.size() * sizeof(float), hipMemcpyDeviceToHost), "preserved batch scores");
    check(hipMemcpy(selected.data(), d_selected[2], selected.size() * sizeof(unsigned), hipMemcpyDeviceToHost), "preserved batch result");
    ok = ok && selected == reference && std::memcmp(after.data(), scores.data(), scores.size() * sizeof(float)) == 0;
    for (unsigned arm = 0; arm < 3U; ++arm) { check(hipFree(d_scores[arm]), "free"); check(hipFree(d_selected[arm]), "free"); }
    check(hipFree(d_maxima), "free"); check(hipFree(d_batch), "free");
    std::printf("%s preserved selection dense/ties/zero-fill/NaN/fallback/direct/batched\n", ok ? "PASS" : "FAIL");
    return ok;
}

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
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, true>), dim3(Rows / TR), dim3(256), 0, s, d_va, union_at);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, false>), dim3(Rows / TR), dim3(256), 0, s, d_vb, union_at);
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
    // Diagnostic accounting must not touch any union byte, and a nonbinding cap is a complete no-op.
    auto* d_stats = dalloc<qsa_fast::UnionStats>(1);
    auto* d_result = dalloc<qsa_fast::UnionStatsResult>(1);
    check(hipMemsetAsync(d_stats, 0, sizeof(*d_stats), s), "stats reset");
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, true>), dim3(Rows / TR), dim3(256), 0, s,
        d_va, union_at, 0U, d_stats);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, qsa_fast::union_stats_finish_kernel, dim3(1), dim3(32), 0, s, d_stats, d_result);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, false, true>), dim3(Rows / TR), dim3(256), 0, s,
        d_vb, union_at, UINT32_MAX);
    check(hipStreamSynchronize(s), "stats / nonbinding cap sync");
    std::vector<std::uint32_t> diagnostic(a.size()), nonbinding(b.size());
    check(hipMemcpy(diagnostic.data(), d_a, a.size() * sizeof(a[0]), hipMemcpyDeviceToHost), "stats scratch");
    check(hipMemcpy(nonbinding.data(), d_b, b.size() * sizeof(b[0]), hipMemcpyDeviceToHost), "nonbinding scratch");
    qsa_fast::UnionStatsResult got{}, want{};
    check(hipMemcpy(&got, d_result, sizeof(got), hipMemcpyDeviceToHost), "stats result");
    std::vector<unsigned> sizes;
    for (unsigned tile = 0; tile < Rows / TR; ++tile) {
        const auto at = static_cast<std::size_t>(tile * TR) * per_row + union_at;
        unsigned n = 0U;
        for (unsigned i = 0; i < a[at]; ++i) {
            const auto bits = a[at + 4U + TR + 2U * i + 1U];
            if (bits) ++n;
            for (unsigned r = 0; r < TR; ++r) want.per_row_blocks += (bits >> r) & 1U;
        }
        sizes.push_back(n);
        want.blocks += n; want.chunks += a[at] / 4U; want.padded += a[at] - n;
    }
    std::sort(sizes.begin(), sizes.end());
    want.rows = Rows; want.tiles = Rows / TR;
    want.p50 = sizes[(sizes.size() + 1U) / 2U - 1U];
    want.p90 = sizes[(9U * sizes.size() + 9U) / 10U - 1U]; want.maximum = sizes.back();
    const bool stats_ok = got.rows == want.rows && got.tiles == want.tiles && got.per_row_blocks == want.per_row_blocks &&
        got.blocks == want.blocks && got.chunks == want.chunks && got.padded == want.padded &&
        got.p50 == want.p50 && got.p90 == want.p90 && got.maximum == want.maximum;
    if (diagnostic != a || nonbinding != b || !stats_ok) {
        std::fprintf(stderr, "FAIL %s diagnostic bytes / host recount / nonbinding cap\n", c.name); ok = false;
    }
    check(hipFree(d_stats), "free"); check(hipFree(d_result), "free");
    // Large unions exercise every radix/compaction stride, independently of indexer score ties.
    const std::vector<float> no_scores;
    const std::vector<unsigned> no_score_blocks(Rows, 0U);
    for (const unsigned cap : {1U, 128U, 1024U}) {
        auto expected = a;
        for (unsigned tile = 0; tile < Rows / TR; ++tile) {
            const unsigned r0 = tile * TR;
            const auto at = static_cast<std::size_t>(r0) * per_row + union_at;
            const auto end = static_cast<std::size_t>(r0 + 1U) * per_row;
            std::vector<unsigned> words(expected.begin() + static_cast<std::ptrdiff_t>(at), expected.begin() + static_cast<std::ptrdiff_t>(end));
            cap_oracle(words, no_scores, no_score_blocks, visible, 0U, r0, cap);
            std::copy(words.begin(), words.end(), expected.begin() + static_cast<std::ptrdiff_t>(at));
        }
        for (unsigned builder = 0; builder < 2U; ++builder) {
            check(hipMemcpyAsync(d_b, canary.data(), canary.size() * sizeof(unsigned), hipMemcpyHostToDevice, s), "large cap reset");
            if (builder)
                STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, true, true>), dim3(Rows / TR), dim3(256), 0, s,
                    d_vb, union_at, cap);
            else
                STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, false, true>), dim3(Rows / TR), dim3(256), 0, s,
                    d_vb, union_at, cap);
            check(hipStreamSynchronize(s), "large cap sync");
            std::vector<unsigned> capped(a.size());
            check(hipMemcpy(capped.data(), d_b, capped.size() * sizeof(unsigned), hipMemcpyDeviceToHost), "large cap result");
            if (capped != expected) {
                std::fprintf(stderr, "FAIL %s large cap %u builder %u oracle\n", c.name, cap, builder); ok = false;
            }
        }
    }
    if (ok && (c.mode == 0U || c.mode == 1U || c.mode == 3U)) {
        constexpr unsigned kGroups = 128U, kRounds = 5U;
        std::vector<qsa_fast::Views<Rows>> many(kGroups, va);
        auto* d_many = dalloc<qsa_fast::Views<Rows>>(kGroups);
        check(hipMemcpy(d_many, many.data(), kGroups * sizeof(va), hipMemcpyHostToDevice), "h2d many");
        hipEvent_t e0, e1;
        check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
        const auto timed = [&](bool merge) {
            check(hipEventRecord(e0, s), "record");
            if (merge) STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, true>), dim3(kGroups * (Rows / TR)), dim3(256), 0, s, d_many, union_at);
            else STRIX_HIP_LAUNCH_GGL(Rdna3Layout, HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR, false>), dim3(kGroups * (Rows / TR)), dim3(256), 0, s, d_many, union_at);
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
    ok = preserved_selection_cases(s) && ok;
    for (unsigned mode = 0; mode < 4U; ++mode) {
        ok = capped_cases<16U>(s, mode) && ok;
        ok = capped_cases<32U>(s, mode) && ok;
        ok = capped_cases<64U>(s, mode) && ok;
    }
    // The merge form must actually run for the sorted cases and be declined for the unsorted / over-capacity tiles.
    ok = ok && merged["sparse-1m"] == Rows / TR && merged["sparse-300k"] == Rows / TR && merged["dense"] == 0U && merged["zero-fill"] == 0U &&
        merged["repeats-empty"] == Rows / TR && merged["over-capacity"] == Rows / TR / 2U;
    check(hipStreamDestroy(s), "stream");
    std::puts(ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
