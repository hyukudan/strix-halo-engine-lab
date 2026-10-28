#include <iterator>
#include <map>
// Included by qsa_flash_fixtures.hip. Native gfx1151 byte comparison of the
// production group-major attend and its occupancy variant.
// Direct launches test the bodies independently of launcher admission.
// Late-position / timing selections (Sel::*) follow select_rows_kernel: a row whose complete blocks (visible / 4) fit
// the 512-block budget selects every visible token (dense, through visible 2051); otherwise 512 complete blocks plus
// its incomplete causal tail. The four rows of a tile share their complete blocks; two adjacent tiles share 384
// of 512 (overlap, I/(A+B) = .375), none (disjoint) or all (equal) - fewer shared only where the tiles' common
// complete range cannot hold the private blocks, which the case line reports.
enum class Sel : unsigned { small = 0U, overlap = 1U, disjoint = 2U, equal = 3U };
struct OccCase {
    unsigned pattern = 0U, padding = 0U, numeric = 0U, prefix = 0U, total = 3U * 64U;
    Sel sel = Sel::small;
    bool timing = false, candidate = true;
    bool depth_boundary = false; // tile union sizes 0, 1, 4, 5, 8, 9, 12, 13: 0..4 chunks
};
static const char* sel_name(Sel s) {
    return s == Sel::overlap ? "overlap" : s == Sel::disjoint ? "disjoint" : s == Sel::equal ? "equal" : "small";
}
// Byte-exact comparison of two output buffers on the device (timing mode: every generation, outside the timed interval).
__global__ void occ_diff_kernel(const uint4* a, const uint4* b, std::size_t n, unsigned* mismatches) {
    unsigned local = 0U;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 x = a[i], y = b[i];
        local += (x.x != y.x) | (x.y != y.y) | (x.z != y.z) | (x.w != y.w);
    }
    if (local) atomicAdd(mismatches, local);
}
// Check poison independently in BOTH arms on every timing pass, including warmup.
__global__ void occ_guard_kernel(const unsigned char* a, const unsigned char* b, unsigned total,
    unsigned pitch, unsigned data, unsigned* mismatches) {
    const std::size_t size = static_cast<std::size_t>(total + 1U) * pitch * sizeof(float);
    unsigned local = 0U;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < size;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t row = i / (pitch * sizeof(float));
        const std::size_t in_row = i % (pitch * sizeof(float));
        if (row == total || in_row >= data * sizeof(float)) local += a[i] != 0xA5U || b[i] != 0xA5U;
    }
    if (local) atomicAdd(mismatches, local);
}
static void run_occ_case(const OccCase& c, hipStream_t s) {
    constexpr unsigned Rows = 64U, Heads = 24U, KV = 2U, Dim = 256U, TopK = 512U;
    const bool late = c.sel != Sel::small, timing = c.timing;
    const unsigned total = c.total, ng = total / Rows, prefix = c.prefix;
    const unsigned visible_max = late ? prefix + total : 769U;
    constexpr unsigned maxsel = 2052U; // 512 complete blocks + a 3-token causal tail (2051), padded
    const unsigned union_at = Heads * qsa_fast::kPartial;
    const unsigned scratch_stride = union_at + 8U + 2U * (maxsel + 4U);
    Rng rng{0xA771900ULL + c.numeric + c.pattern * 17U + c.padding + prefix};
    std::vector<std::uint32_t> selections(static_cast<std::size_t>(total) * maxsel);
    std::vector<unsigned> counts(total), visible(total);
    for (unsigned r = 0U; r < total; ++r)
        visible[r] = late ? prefix + r + 1U : visible_max - 3U - ((r / 4U) % 2U == 0U ? 8U : 0U) + r % 4U;
    unsigned dense_rows = 0U, sparse_rows = 0U, min_shared = ~0U, max_shared = 0U;
    // Late: complete-block lists per tile, built jointly for adjacent tiles.
    std::vector<std::vector<unsigned>> tile_blocks(total / 4U);
    if (late) {
        for (unsigned duo = 0U; duo < total / 8U; ++duo) {
            unsigned common = ~0U;
            for (unsigned r = duo * 8U; r < duo * 8U + 8U; ++r) common = std::min(common, visible[r] / 4U);
            if (common <= TopK) continue; // some row is dense: those tiles take every token below
            const unsigned want = c.sel == Sel::equal ? TopK : c.sel == Sel::disjoint ? 0U : 384U;
            const unsigned shared = std::max(want, std::min(TopK, 2U * TopK > common ? 2U * TopK - common : 0U));
            const unsigned pool = 2U * TopK - shared;
            std::vector<unsigned> blocks;
            // Evenly spread distinct complete blocks over [0, common), then a seeded shuffle assigns them.
            for (unsigned i = 0U; i < pool; ++i) blocks.push_back(static_cast<unsigned>(static_cast<std::uint64_t>(i) * common / pool));
            for (unsigned i = pool - 1U; i > 0U; --i) std::swap(blocks[i], blocks[rng.next() % (i + 1U)]);
            auto& t0 = tile_blocks[duo * 2U]; auto& t1 = tile_blocks[duo * 2U + 1U];
            t0.assign(blocks.begin(), blocks.begin() + TopK);
            t1.assign(blocks.begin(), blocks.begin() + shared);
            t1.insert(t1.end(), blocks.begin() + TopK, blocks.end());
            std::sort(t0.begin(), t0.end()); std::sort(t1.begin(), t1.end());
            expect(t0.size() == TopK && t1.size() == TopK, "occ selection 512 blocks per tile");
            std::vector<unsigned> both;
            std::set_intersection(t0.begin(), t0.end(), t1.begin(), t1.end(), std::back_inserter(both));
            expect(both.size() == shared, "occ selection shared block count");
            min_shared = std::min(min_shared, shared); max_shared = std::max(max_shared, shared);
        }
    }
    for (unsigned r = 0U; r < total; ++r) {
        auto* list = selections.data() + static_cast<std::size_t>(r) * maxsel;
        unsigned n = 0U;
        if (late && visible[r] / 4U <= TopK) {
            for (unsigned token = 0U; token < visible[r]; ++token) list[n++] = token;
            ++dense_rows;
        } else if (late) {
            auto blocks = tile_blocks[r / 4U];
            if (blocks.empty()) { // the adjacent tiles include a dense row: this sparse tile takes its own spread
                const unsigned complete = visible[r] / 4U;
                for (unsigned i = 0U; i < TopK; ++i) blocks.push_back(static_cast<unsigned>(static_cast<std::uint64_t>(i) * complete / TopK));
            }
            // Row 0 of each tile drops its first block: that block's tokens stay row-masked for row 0.
            for (std::size_t i = (r % 4U == 0U ? 1U : 0U); i < blocks.size(); ++i)
                for (unsigned t = 0U; t < 4U; ++t) list[n++] = blocks[i] * 4U + t;
            for (unsigned token = visible[r] / 4U * 4U; token < visible[r]; ++token) list[n++] = token; // causal tail
            ++sparse_rows;
        } else if (c.depth_boundary) {
            const unsigned sizes[] = {0U, 1U, 4U, 5U, 8U, 9U, 12U, 13U};
            const unsigned blocks = sizes[(r / 4U) % 8U];
            // Mix complete selected blocks with a partial visible block in the same tile.
            visible[r] = (1U + blocks) * 4U + r % 4U;
            for (unsigned j = 0U; j < blocks; ++j) list[n++] = (2U + j) * 4U;
        } else if (c.pattern != 5U || r / 4U % 2U) {
            const unsigned side = (r / 4U) % 2U;
            const unsigned blocks = 12U - c.padding - (c.pattern == 3U && side ? 4U : 0U);
            for (unsigned j = 0U; j < blocks; ++j) {
                unsigned b = 1U + j;
                if (c.pattern == 1U) b += side * 64U; // disjoint
                if (c.pattern == 2U) b = 1U + j * 2U + side; // interleaved
                if (c.pattern == 3U) b += side * 3U; // equal IDs in shifted chunk slots
                if (c.pattern == 4U && j + 1U == blocks) b = (visible_max - 9U) / 4U;
                list[n++] = b * 4U;
            }
            if (c.numeric == 7U || c.numeric == 8U) { // multiplicities 0..3, including saturation
                for (unsigned token = 0U; token < 4U; ++token)
                    for (unsigned j = 0U; j < (c.numeric == 7U ? 256U + token : (r + token) % 4U); ++j) list[n++] = token;
            }
        }
        expect(n <= maxsel, "occ selection capacity");
        counts[r] = n;
    }
    // Host replica of union_body per tile: ascending (block, row mask), padded to a multiple of four with (0, 0).
    const auto tile_union = [&](unsigned tile) {
        std::map<unsigned, unsigned> mask;
        unsigned blocks = 0U;
        for (unsigned i = 0U; i < 4U; ++i) blocks = std::max(blocks, (visible[tile * 4U + i] + 3U) / 4U);
        for (unsigned i = 0U; i < 4U; ++i) {
            const unsigned r = tile * 4U + i;
            for (unsigned j = 0U; j < counts[r]; ++j) {
                const unsigned b = selections[static_cast<std::size_t>(r) * maxsel + j] / 4U;
                if (b < blocks) mask[b] |= 1U << i;
            }
        }
        return mask;
    };
    const auto tile_visible = [&](unsigned tile) {
        unsigned v = 0U;
        for (unsigned i = 0U; i < 4U; ++i) v = std::max(v, visible[tile * 4U + i]);
        return v;
    };
    unsigned min_chunks = ~0U, max_chunks = 0U, full_tiles = 0U, partial_tiles = 0U;
    for (unsigned tile = 0U; tile < total / 4U; ++tile) {
        const auto u = tile_union(tile);
        const unsigned chunks = (static_cast<unsigned>(u.size()) + 3U) / 4U;
        min_chunks = std::min(min_chunks, chunks); max_chunks = std::max(max_chunks, chunks);
        full_tiles += !u.empty() && u.size() % 4U == 0U;
        partial_tiles += u.size() % 4U != 0U;
        if (c.depth_boundary) {
            const unsigned sizes[] = {0U, 1U, 4U, 5U, 8U, 9U, 12U, 13U};
            expect(u.size() == sizes[tile % 8U], "occ depth-boundary union size");
        }
    }
    if (c.depth_boundary) expect(min_chunks == 0U && max_chunks == 4U && full_tiles && partial_tiles,
        "occ D=3 boundary and mixed complete/partial chunks");
    std::printf("occ coverage groups %u chunks %u..%u full_tiles %u partial_tiles %u depth_boundary %u\n",
        ng, min_chunks, max_chunks, full_tiles, partial_tiles, c.depth_boundary ? 1U : 0U);
    double overlap = 0.0;
    // Exceptional KV seeds that consumers must meet (numeric 1): kinds 0 selected+live, 1 row-masked, 2 padding (block 0),
    // 3 shared block whose token is visible to one tile and zero-staged by the other.
    std::vector<std::uint32_t> seeds[4];
    unsigned padded_tiles = 0U;
    if (late) {
        std::uint64_t requested = 0U, shared = 0U;
        for (unsigned duo = 0U; duo < total / 8U; ++duo) {
            const auto u0 = tile_union(duo * 2U), u1 = tile_union(duo * 2U + 1U);
            requested += u0.size() + u1.size();
            padded_tiles += (u0.size() % 4U != 0U) + (u1.size() % 4U != 0U);
            for (const auto& [b, m] : u0) if (u1.count(b)) ++shared;
            const unsigned v0 = tile_visible(duo * 2U), v1 = tile_visible(duo * 2U + 1U);
            for (const auto& [b, m] : u0) {
                if (u1.count(b)) {
                    for (unsigned t = b * 4U; t < b * 4U + 4U; ++t)
                        if (t >= std::min(v0, v1) && t < std::max(v0, v1)) seeds[3].push_back(t);
                }
                if (m != 0xfU && b * 4U + 3U < v0) seeds[1].push_back(b * 4U + 1U);
                if (m == 0xfU && b * 4U + 3U < v0 && seeds[0].size() < 64U) seeds[0].push_back(b * 4U + 2U);
            }
        }
        overlap = requested ? static_cast<double>(shared) / static_cast<double>(requested) : 0.0;
        if (padded_tiles) seeds[2].push_back(0U);
    }
    std::vector<std::uint16_t> kv(static_cast<std::size_t>(visible_max) * KV * Dim * 2U);
    if (late) {
        // Cheap deterministic payload for long prefixes: |x| in [2^-3, 2^3), random sign.
        std::uint64_t x = 0x9E3779B97F4A7C15ULL ^ prefix;
        for (std::size_t i = 0U; i < kv.size(); ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            const auto u = static_cast<std::uint32_t>(x >> 32U);
            kv[i] = static_cast<std::uint16_t>(((u & 0x8000U) | (0x3e00U + (u >> 16U) % 0x0300U)) & 0xffffU);
        }
        if (c.numeric == 1U) {
            // V of every seeded token: a distinct NaN payload per kind; K of the first selected seed: +Inf / -0.
            const std::uint16_t vnan[4] = {0x7fc1U, 0x7fc3U, 0xffc5U, 0x7fc7U};
            for (unsigned kind = 0U; kind < 4U; ++kind)
                for (const unsigned token : seeds[kind]) {
                    if (token >= visible_max) continue;
                    const auto row = static_cast<std::size_t>(token) * KV * Dim * 2U;
                    for (unsigned d = 0U; d < 8U; ++d) {
                        if (kind == 0U && token == seeds[0].front()) kv[row + d] = d % 2U ? 0x8000U : 0x7f80U;
                        kv[row + KV * Dim + d] = vnan[kind];
                        kv[row + KV * Dim + Dim + d] = vnan[kind]; // KV head 1
                    }
                }
        }
    } else {
        for (auto& value : kv) {
            const float rounded = bf(rng.normal() * .5F);
            value = static_cast<std::uint16_t>(ubits(rounded) >> 16U);
        }
    }
    const std::uint16_t special[] = {0U, 0x8000U, 1U, 0x8001U, 0x7f7fU, 0x7f80U, 0x7fc1U, 0xff80U};
    if (!late && c.numeric >= 2U && c.numeric <= 6U) {
        for (std::size_t i = 0U; i < kv.size(); ++i) {
            if (c.numeric < 5U) kv[i] = special[(i + c.numeric) % (c.numeric == 4U ? 5U : 4U)];
            else if (i % 37U == 0U || i < KV * Dim * 2U)
                kv[i] = special[5U + i % 3U]; // includes masked/padding block zero
        }
    }
    if (!late && c.numeric == 4U) {
        for (unsigned token = 0U; token < visible_max; ++token)
            for (unsigned d = 0U; d < KV * Dim; ++d) {
                const auto at = static_cast<std::size_t>(token) * KV * Dim * 2U + d;
                kv[at] = 0U; kv[at + KV * Dim] = 0x4780U; // 65536 -> FP16 65504 -> BF16 65536
            }
    }
    std::vector<float> qg(static_cast<std::size_t>(total) * Heads * Dim * 2U);
    for (std::size_t i = 0U; i < qg.size(); ++i) {
        qg[i] = rng.normal();
        if (late) continue;
        if (c.numeric == 0U) qg[i] = 0.0F;
        if (c.numeric == 2U) qg[i] = i % 2U ? -0.0F : 0.0F;
        if (c.numeric == 3U) qg[i] = bits(i % 2U ? 0x80010000U : 0x00010000U);
        if (c.numeric == 4U) qg[i] = i % (2U * Dim) < Dim ? 0.0F : 100.0F;
        if (c.numeric == 6U && i % 101U == 0U) qg[i] = bits(0x7f800000U);
    }
    auto* dk = dalloc<std::uint16_t>(kv.size()); auto* dq = dalloc<float>(qg.size());
    auto* ds = dalloc<std::uint32_t>(selections.size());
    auto* scratch = dalloc<float>(static_cast<std::size_t>(total / 4U) * scratch_stride);
    auto* dmis = dalloc<unsigned>(1U);
    check(hipMemset(dmis, 0, sizeof(unsigned)), "occ mismatch counter");
    constexpr unsigned Pitch = Heads * Dim + 32U;
    const std::size_t output_size = static_cast<std::size_t>(total + 1U) * Pitch;
    auto* oa = dalloc<float>(output_size); auto* ob = dalloc<float>(output_size);
    check(hipMemcpy(dk, kv.data(), kv.size() * sizeof(kv[0]), hipMemcpyHostToDevice), "occ KV");
    check(hipMemcpy(dq, qg.data(), qg.size() * sizeof(float), hipMemcpyHostToDevice), "occ Q");
    check(hipMemcpy(ds, selections.data(), selections.size() * sizeof(selections[0]), hipMemcpyHostToDevice), "occ selections");
    std::vector<qsa_fast::Views<Rows>> views(ng);
    for (unsigned r = 0U; r < total; ++r) {
        auto& v = views[r / Rows]; const unsigned row = r % Rows;
        v.q_gate[row] = dq + static_cast<std::size_t>(r) * Heads * Dim * 2U;
        v.selected[row] = ds + static_cast<std::size_t>(r) * maxsel;
        v.scratch[row] = scratch + static_cast<std::size_t>(r / 4U) * scratch_stride;
        v.count[row] = counts[r]; v.visible[row] = visible[r];
    }
    // Exercise mapped descriptors and event-protected reuse beyond both ring slots.
    qsa_fast::Views<Rows>* ring[2]{};
    hipEvent_t done[2]{}, begin{}, mid{}, end{};
    for (unsigned i = 0U; i < 2U; ++i) {
        check(hipHostMalloc(reinterpret_cast<void**>(&ring[i]), ng * sizeof(views[0]), hipHostMallocDefault), "occ ring");
        check(hipEventCreate(&done[i]), "occ ring event");
    }
    check(hipEventCreate(&begin), "occ timer"); check(hipEventCreate(&mid), "occ timer"); check(hipEventCreate(&end), "occ timer");
    double elapsed[2]{}, attend[2]{};
    std::vector<double> samples[2];
    const unsigned passes = timing ? 12U : late ? 4U : 6U;
    std::vector<unsigned char> a(output_size * sizeof(float)), b(a.size());
    const auto check_guards = [total](const std::vector<unsigned char>& bytes) {
        bool intact = true;
        for (unsigned row = 0U; row <= total; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * Pitch;
            const std::size_t first = (base + (row < total ? Heads * Dim : 0U)) * sizeof(float);
            const std::size_t last = (base + Pitch) * sizeof(float);
            for (std::size_t i = first; i < last; ++i) intact = intact && bytes[i] == 0xA5U;
        }
        return intact;
    };
    bool equal = true;
    for (unsigned pass = 0U; pass < passes; ++pass) {
        // AB, BA order alternates to reduce thermal/order bias.
        for (unsigned order = 0U; order < 2U; ++order) {
            const unsigned which = order ^ (pass % 2U), slot = order;
            if (pass) check(hipEventSynchronize(done[slot]), "occ ring wait");
            for (unsigned g = 0U; g < ng; ++g) {
                auto v = views[pass % 2U ? ng - 1U - g : g];
                for (unsigned row = 0U; row < Rows; ++row) {
                    const unsigned original = (pass % 2U ? ng - 1U - g : g) * Rows + row;
                    v.output[row] = (which ? ob : oa) + static_cast<std::size_t>(original) * Pitch;
                }
                ring[slot][g] = v;
            }
            // Clear output and guards for every descriptor generation, outside timing.
            check(hipMemsetAsync(which ? ob : oa, 0xA5, output_size * sizeof(float), s), "occ generation poison");
            check(hipEventRecord(begin, s), "occ time start");
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, 4U>), dim3(ng * 16U), dim3(256), 0, s, ring[slot], union_at);
            check(hipEventRecord(mid, s), "occ time union");
            if (which && c.candidate)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_occ_kernel<Rows, 4U, 3U>), dim3(ng * 16U * KV), dim3(320), 0, s,
                    ring[slot], dk, KV, Heads, union_at, KV);
            else
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_kernel<Rows, 4U, 3U>), dim3(ng * 16U * KV), dim3(320), 0, s,
                    ring[slot], dk, KV, Heads, union_at, KV);
            check(hipGetLastError(), "occ launch");
            check(hipEventRecord(end, s), "occ time end"); check(hipEventRecord(done[slot], s), "occ ring record");
            check(hipEventSynchronize(end), "occ time sync");
            float ms = 0.0F, ams = 0.0F;
            check(hipEventElapsedTime(&ms, begin, end), "occ elapsed");
            check(hipEventElapsedTime(&ams, mid, end), "occ attend elapsed");
            if (pass >= 2U) { elapsed[which] += static_cast<double>(ms); attend[which] += static_cast<double>(ams); samples[which].push_back(ams); }
            if (!timing) {
                auto& result = which ? b : a;
                check(hipMemcpy(result.data(), which ? ob : oa, result.size(), hipMemcpyDeviceToHost), "occ generation result");
                expect(check_guards(result), which ? "occ comparison generation guards" : "occ reference generation guards");
            }
        }
        if (timing) { // both generations of this pass, byte for byte, after the timed intervals
            hipLaunchKernelGGL(occ_diff_kernel, dim3(1024), dim3(256), 0, s, reinterpret_cast<const uint4*>(oa),
                reinterpret_cast<const uint4*>(ob), output_size * sizeof(float) / sizeof(uint4), dmis);
            check(hipGetLastError(), "occ diff launch");
            hipLaunchKernelGGL(occ_guard_kernel, dim3(1024), dim3(256), 0, s,
                reinterpret_cast<const unsigned char*>(oa), reinterpret_cast<const unsigned char*>(ob),
                total, Pitch, Heads * Dim, dmis);
            check(hipGetLastError(), "occ guard launch");
        } else equal = equal && a == b;
    }
    check(hipStreamSynchronize(s), "occ comparison sync");
    unsigned mismatches = 0U;
    check(hipMemcpy(&mismatches, dmis, sizeof(unsigned), hipMemcpyDeviceToHost), "occ mismatch count");
    equal = equal && mismatches == 0U;
    check(hipMemcpy(a.data(), oa, a.size(), hipMemcpyDeviceToHost), "occ result");
    check(hipMemcpy(b.data(), ob, b.size(), hipMemcpyDeviceToHost), "occ result");
    equal = equal && a == b;
    expect(check_guards(a), "occ reference final guards");
    expect(check_guards(b), "occ comparison final guards");
    std::size_t nan_outputs = 0U;
    for (std::size_t i = 0U; i + 4U <= a.size(); i += 4U) {
        std::uint32_t u; std::memcpy(&u, a.data() + i, 4U);
        nan_outputs += (u & 0x7f800000U) == 0x7f800000U && (u & 0x7fffffU) && u != 0xA5A5A5A5U;
    }
    const auto median = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2U]; };
    const auto spread = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v.back() - v.front(); };
    const double k = passes - 2U;
    if (late) {
        std::printf("occ %s prefix %u rows %u dense %u sparse %u shared %u..%u overlap %.4f padded_tiles %u seeds %zu/%zu/%zu/%zu "
            "numeric %u %s %s: %s union+attend %.3f / %.3f ms attend %.3f / %.3f ms median %.3f / %.3f spread %.3f / %.3f "
            "mismatch_vectors_or_guard_bytes %u nan_outputs %zu\n",
            sel_name(c.sel), prefix, total, dense_rows, sparse_rows, min_shared == ~0U ? 0U : min_shared, max_shared, overlap,
            padded_tiles, seeds[0].size(), seeds[1].size(), seeds[2].size(), seeds[3].size(), c.numeric,
            timing ? "timing" : "exact", c.candidate ? "A=production B=occ-D3" : "A/A production", equal ? "PASS" : "FAIL",
            elapsed[0] / k, elapsed[1] / k, attend[0] / k, attend[1] / k, median(samples[0]), median(samples[1]),
            spread(samples[0]), spread(samples[1]), mismatches, nan_outputs);
        if (c.numeric == 1U) {
            // Every seed kind present where the case can hold it, and the NaN payloads reach the outputs.
            expect(!seeds[0].empty() && !seeds[1].empty(), "occ late seeds: selected and row-masked tokens");
            expect(!padded_tiles || !seeds[2].empty(), "occ late seeds: padding");
            expect(!seeds[3].empty(), "occ late seeds: shared block zero-staged by one tile");
            expect(nan_outputs > 0U, "occ late seeds reach outputs");
        }
    } else
        std::printf("occ pattern %u padding %u numeric %u prefix %u rows %u %s: %s union+attend %.3f / %.3f ms\n",
            c.pattern, c.padding, c.numeric, prefix, total, c.candidate ? "A=production B=occ-D3" : "A/A production",
            equal ? "PASS" : "FAIL", elapsed[0] / k, elapsed[1] / k);
    expect(equal, "occ complete output bytes including guards");
    for (unsigned i = 0U; i < 2U; ++i) { check(hipEventDestroy(done[i]), "occ event free"); check(hipHostFree(ring[i]), "occ ring free"); }
    check(hipEventDestroy(begin), "occ timer free"); check(hipEventDestroy(mid), "occ timer free"); check(hipEventDestroy(end), "occ timer free");
    for (void* p : {static_cast<void*>(dk), static_cast<void*>(dq), static_cast<void*>(ds), static_cast<void*>(scratch),
        static_cast<void*>(oa), static_cast<void*>(ob), static_cast<void*>(dmis)}) check(hipFree(p), "occ free");
}

// Occupancy candidate: production attend vs attend_order_occ_kernel, complete output bytes (exact) and ABBA attend times.
static int run_occ_fixtures(hipStream_t s, bool timing) {
    int device = 0;
    hipDeviceProp_t props{};
    check(hipGetDevice(&device), "occ device");
    check(hipGetDeviceProperties(&props, device), "occ device properties");
    // Feature suffixes (e.g. :xnack-) are permitted, other architectures are not a native qualification.
    if (std::strncmp(props.gcnArchName, "gfx1151", 7U) != 0 ||
        (props.gcnArchName[7] != '\0' && props.gcnArchName[7] != ':')) {
        std::fprintf(stderr, "occ fixtures require native gfx1151, got %s\n", props.gcnArchName);
        return 2;
    }
    hipFuncAttributes prod{}, occ{};
    check(hipFuncGetAttributes(&prod, reinterpret_cast<const void*>(&qsa_fast::attend_order_kernel<64U, 4U, 3U>)), "production attributes");
    const void* occ_fn = reinterpret_cast<const void*>(&qsa_fast::attend_order_occ_kernel<64U, 4U, 3U>);
    check(hipFuncGetAttributes(&occ, occ_fn), "occ attributes");
    int resident_prod = 0, resident_occ = 0;
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident_prod,
        reinterpret_cast<const void*>(&qsa_fast::attend_order_kernel<64U, 4U, 3U>), 320, 0U), "production occupancy");
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident_occ, occ_fn, 320, 0U), "occ occupancy");
    std::printf("occ D%u resources production LDS %zu registers %d local %zu resident %d | occ LDS %zu registers %d local %zu resident %d\n",
        3U, prod.sharedSizeBytes, prod.numRegs, prod.localSizeBytes, resident_prod, occ.sharedSizeBytes, occ.numRegs, occ.localSizeBytes,
        resident_occ);
    OccCase c;
    if (timing) {
        c.total = 8192U; c.timing = true;
        c.sel = Sel::overlap; c.candidate = false; c.prefix = 24576U; run_occ_case(c, s);
        c.candidate = true;
        for (unsigned prefix : {0U, 8192U, 16384U, 24576U, 122880U, 253952U}) { c.prefix = prefix; c.sel = Sel::overlap; run_occ_case(c, s); }
        for (Sel sel : {Sel::disjoint, Sel::equal}) { c.prefix = 24576U; c.sel = sel; run_occ_case(c, s); }
        c.sel = Sel::overlap; c.candidate = false; c.prefix = 122880U; run_occ_case(c, s);
    } else {
        for (unsigned pattern = 0U; pattern < 6U; ++pattern)
            for (unsigned padding = 0U; padding < 4U; ++padding)
                for (unsigned numeric = 0U; numeric < 9U; ++numeric) {
                    c.pattern = pattern; c.padding = padding; c.numeric = numeric;
                    run_occ_case(c, s);
                }
        // The original 216 small cases above use three groups. Repeat every class with one group.
        c.total = 64U;
        for (unsigned pattern = 0U; pattern < 6U; ++pattern)
            for (unsigned padding = 0U; padding < 4U; ++padding)
                for (unsigned numeric = 0U; numeric < 9U; ++numeric) {
                    c.pattern = pattern; c.padding = padding; c.numeric = numeric;
                    run_occ_case(c, s);
                }
        for (unsigned total : {64U, 192U}) for (unsigned numeric : {0U, 5U}) {
            c = OccCase{}; c.total = total; c.numeric = numeric; c.depth_boundary = true;
            run_occ_case(c, s);
        }
        // Original 24 late cases: causal tails, dense/sparse transition and exceptional seeds.
        c = OccCase{}; c.total = 128U;
        for (unsigned prefix : {2042U, 32642U, 130946U, 262014U})
            for (Sel sel : {Sel::overlap, Sel::disjoint, Sel::equal})
                for (unsigned numeric : {0U, 1U}) { c.prefix = prefix; c.sel = sel; c.numeric = numeric; run_occ_case(c, s); }
    }
    return failures ? 1 : 0;
}

// Cheap default-mode smoke: one group, mixed 0..4 chunks, finite and exceptional KV.
// The explicit correctness mode retains the full small and long-prefix matrices.
static void run_occ_smoke(hipStream_t s) {
    for (unsigned numeric : {0U, 5U}) {
        OccCase c; c.total = 64U; c.numeric = numeric; c.depth_boundary = true;
        run_occ_case(c, s);
    }
}
