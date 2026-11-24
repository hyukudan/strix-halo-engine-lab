// Host check of the microscaled R1 rows chunking (launch_rows): for every batch size and scratch size, each routed row
// lands in exactly one chunk of at most rows_chunk_cap(scratch rows) rows (kMaxRows by default, at most
// kMaxLaunchRows), at the chunk-local position whose input, ids, weights, failure word and output are the batch row's
// own (row_offsets of the batch row), and every chunk restarts the activation scratch at its base, inside the scratch.
// The kernels then run the one-row body per row, so the row's bits are those of a one-row launch for any batch size.
#include "moe_core/microscaled_r1.hpp"
#include "strix/device/declaration.hpp"
#include "strix/device/gate.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;

constexpr unsigned kTopK = 8U, kHidden = 48U, kIntermediate = 16U;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    (void)_putenv_s(name, value);
#else
    if (*value) (void)setenv(name, value, 1);
    else (void)unsetenv(name);
#endif
}

// scratch_rows == 0: the launchers' default iteration (no cap argument = kMaxRows rows of scratch).
bool check_rows(unsigned rows, bool with_failure, unsigned scratch_rows) {
    const unsigned cap = scratch_rows == 0U ? r1::kMaxRows : r1::rows_chunk_cap(scratch_rows);
    std::vector<unsigned> ids(std::size_t{rows} * kTopK), failure(rows);
    std::vector<float> weights(ids.size()), input(std::size_t{rows} * kHidden), output(input.size());
    std::vector<std::uint16_t> activation(std::size_t{scratch_rows == 0U ? r1::kMaxRows : scratch_rows} * kTopK *
                                          kIntermediate);
    const r1::Args batch{nullptr, ids.data(), weights.data(), with_failure ? failure.data() : nullptr, input.data(),
                         activation.data(), output.data(), 128U, kHidden, kIntermediate, 4.0F, 25.0F};
    std::vector<unsigned> seen(rows, 0U);
    unsigned chunks = 0;
    // The launchers' own iteration (for_each_rows_chunk): `true` = launched, `false` = refused.
    const auto launch_chunk = [&](unsigned first, unsigned count) {
        ++chunks;
        if (count == 0U || count > cap || count > r1::kMaxLaunchRows) return false;
        const auto chunk = r1::rows_chunk<kTopK>(batch, first);
        if (chunk.activation != activation.data() || chunk.matrices != batch.matrices || chunk.hidden != kHidden ||
            chunk.intermediate != kIntermediate || chunk.experts != batch.experts)
            return false;
        for (unsigned local = 0; local < count; ++local) {
            const unsigned row = first + local;
            const auto mine = r1::row_offsets<kTopK>(row, kHidden, kIntermediate);
            const auto at = r1::row_offsets<kTopK>(local, kHidden, kIntermediate);
            if (chunk.input + at.input != input.data() + mine.input ||
                chunk.output + at.output != output.data() + mine.output ||
                chunk.ids + at.route != ids.data() + mine.route ||
                chunk.weights + at.route != weights.data() + mine.route)
                return false;
            if (with_failure ? chunk.failure + local != failure.data() + row : chunk.failure != nullptr) return false;
            // The chunk's activation rows stay inside the caller's scratch.
            if (at.activation + std::size_t{kTopK} * kIntermediate > activation.size()) return false;
            ++seen[row];
        }
        return true;
    };
    const bool ok = scratch_rows == 0U ? r1::for_each_rows_chunk(rows, true, launch_chunk)
                                       : r1::for_each_rows_chunk(rows, true, launch_chunk, cap);
    if (!ok || chunks != r1::rows_chunk_count(rows, cap)) return false;
    for (unsigned count : seen)
        if (count != 1U) return false;
    return true;
}

// Near UINT_MAX (no batch allocated): the count does not wrap, the last chunk ends exactly at rows, every row is
// covered once (sizes sum to rows) and the iteration stops; a refused chunk stops the launches at that chunk.
bool check_wrap_edge() {
    constexpr unsigned kMax = std::numeric_limits<unsigned>::max();
    static_assert(r1::rows_chunk_count(kMax) == kMax / r1::kMaxRows + 1U);
    static_assert(r1::rows_chunk_count(kMax - kMax % r1::kMaxRows) == kMax / r1::kMaxRows);
    static_assert(r1::rows_chunk_size(kMax, kMax - kMax % r1::kMaxRows) == kMax % r1::kMaxRows);
    for (unsigned rows : {kMax, kMax - kMax % r1::kMaxRows}) {
        unsigned calls = 0, last_first = 0, last_count = 0;
        std::uint64_t covered = 0, expected_first = 0;
        bool ordered = true;
        const int status = r1::for_each_rows_chunk(rows, 0, [&](unsigned first, unsigned count) {
            ordered = ordered && first == expected_first && count >= 1U && count <= r1::kMaxRows;
            expected_first = std::uint64_t{first} + count;
            covered += count;
            last_first = first;
            last_count = count;
            ++calls;
            return 0;
        });
        const unsigned tail = rows % r1::kMaxRows == 0U ? r1::kMaxRows : rows % r1::kMaxRows;
        if (status != 0 || !ordered || calls != r1::rows_chunk_count(rows) || covered != rows ||
            last_count != tail || last_first != rows - tail) {
            std::printf("FAIL wrap edge rows %u: calls %u last %u+%u\n", rows, calls, last_first, last_count);
            return false;
        }
    }
    unsigned calls = 0;
    const int refused = r1::for_each_rows_chunk(kMax, 0, [&](unsigned first, unsigned) {
        ++calls;
        return first == 2U * r1::kMaxRows ? 7 : 0;
    });
    return refused == 7 && calls == 3U;
}

// rows_cta: every CTA of a rows launch maps to a distinct (unit, block) and every pair is reached, for any tile; a
// tile's CTAs of one unit are consecutive blocks; tile = blocks is the (block, unit) order of the one-row grids.
bool check_rows_order(unsigned units, unsigned blocks, unsigned tile) {
    const r1::RowsOrder order{units, blocks, tile};
    std::vector<unsigned char> seen(std::size_t{units} * blocks, 0U);
    for (unsigned linear = 0; linear < units * blocks; ++linear) {
        const auto cta = r1::rows_cta(order, linear);
        if (cta.unit >= units || cta.block >= blocks) return false;
        auto& hit = seen[std::size_t{cta.unit} * blocks + cta.block];
        if (hit != 0U) return false;
        hit = 1U;
        if (tile == blocks && (cta.unit != linear / blocks || cta.block != linear % blocks)) return false;
        if (linear > 0U) {
            const auto before = r1::rows_cta(order, linear - 1U);
            const bool same_tile = before.block / tile == cta.block / tile;
            if (same_tile && before.unit == cta.unit && before.block + 1U != cta.block) return false;
        }
    }
    return true;
}

} // namespace

int main() {
    static_assert(r1::rows_chunk_cap(0U) == 0U && r1::rows_chunk_cap(5U) == 5U && r1::rows_chunk_cap(16U) == 16U &&
                  r1::rows_chunk_cap(17U) == r1::kMaxLaunchRows);
    static_assert(r1::rows_chunk_count(16U, r1::kMaxLaunchRows) == 1U &&
                  r1::rows_chunk_count(17U, r1::kMaxLaunchRows) == 2U &&
                  r1::rows_chunk_size(17U, 16U, r1::kMaxLaunchRows) == 1U);
    static_assert(r1::rows_chunk_count(1U) == 1U && r1::rows_chunk_count(r1::kMaxRows) == 1U);
    static_assert(r1::rows_chunk_count(r1::kMaxRows + 1U) == 2U && r1::rows_chunk_count(16U) == 2U);
    // Tile policy on the Inkling-shaped gate_up (intermediate 1152, hidden 2304, top-8, 2 columns per wave): the
    // largest tile whose span fits the measured budget; no budget keeps the (block, unit) order.
    using packed = strix::weight::microscaled::MatrixFormat;
    namespace tiles = strix::r1_rows_tiles;
    // A plan comes only from the device declaration ([r1_rows_tiles.<name>], published by the start gate) and holds
    // only for the workload it was measured on: gfx1151 with 20 WGPs (multiProcessorCount as HIP reports it, not the
    // 40 CUs), 128 experts, 2304 x 1152, top-8, NVFP4 or MXFP8, rows 4 and 8. Every other key, row count, bank or an
    // absent declaration resolves to the incumbent order.
    const auto declaration = strix::device::parse_declaration(R"(schema = "strix-devices-v1"
[r1_rows_tiles.gfx1151-inkling-bank]
arch = "gfx1151"
multiprocessors = 20
experts = 128
hidden = 2304
intermediate = 1152
top_k = 8
formats = ["nvfp4", "mxfp8"]
cache_mib = [0, 0, 0, 0, 24, 0, 0, 0, 24]
)");
    const std::span<const tiles::DeviceTable> declared(declaration.r1_rows_tiles);
    const auto budget = [&](std::string_view arch, unsigned mp, packed format, unsigned experts, unsigned hidden,
                            unsigned intermediate, unsigned top_k, unsigned rows) {
        return tiles::gate_up_cache_bytes(declared, arch, mp, format, experts, hidden, intermediate, top_k, rows);
    };
    constexpr std::uint64_t mib24 = std::uint64_t{24U} << 20U;
    unsigned failures = 0U;
    const auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("FAIL %s\n", what);
            ++failures;
        }
    };
    expect(declared.size() == 1U && declared[0].name == "gfx1151-inkling-bank", "declaration holds the one entry");
    // The measured device finds its row with the runtime's own key (feature suffix, multiprocessor count 20).
    expect(budget("gfx1151:sramecc-:xnack-", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 4U) == mib24, "nvfp4 rows 4");
    expect(budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == mib24, "nvfp4 rows 8");
    expect(budget("gfx1151:sramecc-:xnack-", 20U, packed::mxfp8, 128U, 2304U, 1152U, 8U, 8U) == mib24, "mxfp8 rows 8");
    // Row counts: 1 / 2 measured without gain, 3 / 5 / 6 / 7 not measured, 9+ past the table.
    for (unsigned rows : {0U, 1U, 2U, 3U, 5U, 6U, 7U, 9U, 16U}) {
        for (packed format : {packed::nvfp4, packed::mxfp8}) {
            expect(budget("gfx1151", 20U, format, 128U, 2304U, 1152U, 8U, rows) == 0U, "uncovered row count");
        }
    }
    // Every other key field: architecture, multiprocessors (40 = the CU count, not what HIP reports), bank size (the
    // review's 256-expert bank), hidden, intermediate, top_k.
    expect(budget("gfx1201", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U, "other arch");
    expect(budget("", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U, "no arch");
    expect(budget("gfx1151", 40U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U, "40 multiprocessors");
    expect(budget("gfx1151", 16U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U, "16 multiprocessors");
    expect(budget("gfx1151", 20U, packed::nvfp4, 256U, 2304U, 1152U, 8U, 8U) == 0U, "256 experts");
    expect(budget("gfx1151", 20U, packed::nvfp4, 64U, 2304U, 1152U, 8U, 8U) == 0U, "64 experts");
    expect(budget("gfx1151", 20U, packed::nvfp4, 128U, 4096U, 1152U, 8U, 8U) == 0U, "other hidden");
    expect(budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1536U, 8U, 8U) == 0U, "other intermediate");
    expect(budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 4U, 8U) == 0U, "other top_k");
    // No declaration (nothing published, or a declaration without the section): no plan anywhere.
    expect(tiles::binding_tables({tiles::Runtime::hip, 0}).empty(), "nothing published in a binary without a bind");
    expect(tiles::gate_up_cache_bytes({}, "gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U,
           "no declaration");
    expect(strix::device::parse_declaration("schema = \"strix-devices-v1\"\n").r1_rows_tiles.empty(),
           "declaration without plans");
    // A published table is what the launcher reads for its device (current_rows_cache_bytes looks it up with the
    // calling thread's HIP device and that device's key).
    expect(tiles::publish({tiles::Runtime::hip, 0}, declaration.r1_rows_tiles), "publish HIP device 0");
    expect(tiles::gate_up_cache_bytes(tiles::binding_tables({tiles::Runtime::hip, 0}), "gfx1151", 20U, packed::nvfp4,
                                      128U, 2304U, 1152U, 8U, 8U) == mib24,
           "published table");
    expect(tiles::binding_tables({tiles::Runtime::cuda, 0}).empty(), "another runtime's device 0 has its own tables");
    expect(!tiles::publish({tiles::Runtime::hip, tiles::ordinals}, declaration.r1_rows_tiles) &&
               tiles::binding_tables({tiles::Runtime::hip, tiles::ordinals}).empty() &&
               tiles::binding_tables({tiles::Runtime::hip, -1}).empty(),
           "no slot past the device indices");
    // The declaration refuses an entry that could cover more than was measured.
    const auto refused = [](const std::string& body) {
        try {
            (void)strix::device::parse_declaration("schema = \"strix-devices-v1\"\n[r1_rows_tiles.x]\n" + body);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const std::string bank = "experts = 128\nhidden = 2304\nintermediate = 1152\ntop_k = 8\n";
    const std::string full = "arch = \"gfx1151\"\nmultiprocessors = 20\n" + bank + "formats = [\"nvfp4\"]\n";
    expect(!refused(full + "cache_mib = [0, 0, 0, 0, 24]\n"), "short budget list accepted");
    expect(refused(full), "missing cache_mib");
    expect(!refused(full + "cache_mib = [0, 0, 0, 0, 24, 0, 0, 0, 24, 24, 24, 24, 24, 24, 24, 24, 24]\n"),
           "seventeen row counts (0 .. kMaxLaunchRows) refused");
    expect(refused(full + "cache_mib = [0, 0, 0, 0, 24, 0, 0, 0, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]\n"),
           "eighteen row counts");
    expect(refused(full + "cache_mib = [24]\nextra = 1\n"), "unknown key");
    expect(refused("arch = \"gfx1151\"\nmultiprocessors = 20\n" + bank + "formats = [\"fp16\"]\ncache_mib = [24]\n"),
           "unknown format");
    expect(refused("arch = \"gfx1151:xnack-\"\nmultiprocessors = 20\n" + bank +
                   "formats = [\"nvfp4\"]\ncache_mib = [24]\n"),
           "arch with features");
    expect(refused("arch = \"gfx1151\"\n" + bank + "formats = [\"nvfp4\"]\ncache_mib = [24]\n"),
           "missing multiprocessors");
    // Two entries with distinct names that cover one workload (device, bank and a shared format) are refused at load:
    // otherwise the first entry would mask the second (a 4-row-only table ahead of an 8-row one gives 0 at 8 rows).
    const auto two = [&](const std::string& first, const std::string& second) {
        try {
            (void)strix::device::parse_declaration("schema = \"strix-devices-v1\"\n[r1_rows_tiles.a]\n" + first +
                                                   "[r1_rows_tiles.b]\n" + second);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const std::string device20 = "arch = \"gfx1151\"\nmultiprocessors = 20\n" + bank;
    const std::string rows4 = "cache_mib = [0, 0, 0, 0, 24]\n";
    const std::string rows8 = "cache_mib = [0, 0, 0, 0, 0, 0, 0, 0, 24]\n";
    expect(two(device20 + "formats = [\"nvfp4\"]\n" + rows4, device20 + "formats = [\"nvfp4\"]\n" + rows8),
           "same key, same format refused");
    expect(two(device20 + "formats = [\"nvfp4\"]\n" + rows4, device20 + "formats = [\"mxfp8\", \"nvfp4\"]\n" + rows8),
           "same key, overlapping formats refused");
    expect(!two(device20 + "formats = [\"nvfp4\"]\n" + rows4, device20 + "formats = [\"mxfp8\"]\n" + rows8),
           "same key, disjoint formats accepted");
    expect(!two(device20 + "formats = [\"nvfp4\"]\n" + rows4,
                "arch = \"gfx1151\"\nmultiprocessors = 40\n" + bank + "formats = [\"nvfp4\"]\n" + rows8),
           "other multiprocessors accepted");
    expect(!two(device20 + "formats = [\"nvfp4\"]\n" + rows4,
                "arch = \"gfx1151\"\nmultiprocessors = 20\nexperts = 256\nhidden = 2304\nintermediate = 1152\n"
                "top_k = 8\nformats = [\"nvfp4\"]\n" + rows8),
           "other bank accepted");
    // One process, several starts. A start only resolves its plans (Bound::r1_rows_tiles); the backend binder publishes
    // them for the runtime device it owns (device::publish_bound_plans), so a start the backend refuses publishes
    // nothing. Two bindings alive on HIP devices 1 and 2: one with a declared 4-row budget, one without any declaration
    // (no STRIX_DEVICES, no default file); each device's 4-row launch reads its own binding's tables. A later
    // declaration-free start on device 1 clears device 1 only.
    {
        namespace device = strix::device;
        const auto dir = std::filesystem::temp_directory_path() / "strix-r1-rows-chunk-test";
        std::filesystem::create_directories(dir);
        const auto path = (dir / "devices.toml").string();
        {
            std::ofstream out(path, std::ios::binary);
            out << "schema = \"strix-devices-v1\"\n[devices.igpu]\nvendor = \"amd\"\narch = \"gfx1151\"\n"
                   "pci = \"0000:c5:00.0\"\nattach = { kind = \"integrated\" }\nroles = [\"pair\", \"decode\", \"prefill\"]\n"
                   "[r1_rows_tiles.declared]\n" + device20 + "formats = [\"nvfp4\"]\n" + rows4;
        }
        device::Visible igpu;
        igpu.ordinal = 0;
        igpu.vendor = device::Vendor::amd;
        igpu.name = "AMD Radeon Graphics";
        igpu.pci = device::normalise_pci("c5:00.0");
        igpu.arch = "gfx1151";
        igpu.integrated = true;
        device::FakeEnumerator cards(device::Vendor::amd, {igpu});
        // No declaration from the environment: no STRIX_DEVICE(S), and the default file locations point nowhere.
        const auto nowhere = (dir / "none").string();
        set_env("STRIX_DEVICE", "");
        set_env("STRIX_DEVICES", "");
        set_env("XDG_CONFIG_HOME", nowhere.c_str());
        set_env("APPDATA", nowhere.c_str());
        const auto order4 = [&](int ordinal) {
            return r1::gate_up_rows_order<packed::nvfp4>(
                       1152U, 2304U, 4U, 8U, 2U,
                       tiles::gate_up_cache_bytes(tiles::binding_tables({tiles::Runtime::hip, ordinal}),
                                                  "gfx1151:sramecc-:xnack-", 20U, packed::nvfp4, 128U, 2304U, 1152U,
                                                  8U, 4U))
                .tile;
        };
        try {
            const auto bound_declared = device::start_gate(cards, {}, path);
            expect(bound_declared.r1_rows_tiles.size() == 1U, "declared start resolves its plan");
            expect(tiles::binding_tables({tiles::Runtime::hip, 1}).empty() && order4(1) == 72U,
                   "a start the backend has not bound (or refused) publishes nothing");
            device::publish_bound_plans(bound_declared, tiles::Runtime::hip, 1);
            expect(order4(1) == 18U, "bound declared start: device 1's 4-row launch takes the measured tile");
            const auto plain = device::start_gate(cards, {});
            expect(plain.r1_rows_tiles.empty(), "declaration-free start resolves no plan");
            device::publish_bound_plans(plain, tiles::Runtime::hip, 2);
            expect(order4(2) == 72U, "declaration-free binding: device 2 takes the incumbent order");
            expect(order4(1) == 18U, "a second binding alive leaves device 1's tables");
            expect(tiles::gate_up_cache_bytes(tiles::binding_tables({tiles::Runtime::hip, 0}), "gfx1151", 20U,
                                              packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == mib24,
                   "device 0 keeps its tables through the other starts");
            device::publish_bound_plans(plain, tiles::Runtime::hip, 1);
            expect(tiles::binding_tables({tiles::Runtime::hip, 1}).empty() && order4(1) == 72U,
                   "declaration-free start on device 1 clears device 1 only");
            expect(tiles::binding_tables({tiles::Runtime::cuda, 1}).empty(), "HIP binds leave CUDA devices alone");
            bool past_refused = false;
            try {
                device::publish_bound_plans(bound_declared, tiles::Runtime::hip, tiles::ordinals);
            } catch (const device::Refusal&) {
                past_refused = true;
            }
            expect(past_refused, "a bind past the plan slots refuses");
        } catch (const std::exception& e) {
            std::printf("FAIL start gate: %s\n", e.what());
            ++failures;
        }
        std::filesystem::remove_all(dir);
    }
    // Uncovered -> the incumbent order (tile == blocks: CTA i = (i / blocks, i % blocks)).
    expect(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 5U, 8U, 2U,
                                                  budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 5U))
                   .tile == 72U,
           "rows 5 incumbent order");
    expect(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 8U, 8U, 2U,
                                                  budget("gfx1151", 20U, packed::nvfp4, 256U, 2304U, 1152U, 8U, 8U))
                   .tile == 72U,
           "256-expert bank incumbent order");
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 2U, 8U, 2U, 0U).tile == 72U);
    // Covered -> the measured tiles.
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 4U, 8U, 2U, mib24).tile == 18U);
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 8U, 8U, 2U, mib24).tile == 9U);
    static_assert(r1::gate_up_rows_order<packed::mxfp8>(1152U, 2304U, 8U, 8U, 2U, mib24).tile == 5U);
    static_assert(r1::gate_up_rows_order<packed::mxfp8>(1152U, 2304U, 4U, 8U, 2U, mib24).tile == 10U);
    // A single block's span above the budget keeps the launch order.
    static_assert(r1::rows_order(64U, 72U, (std::uint64_t{1U} << 20U), mib24).tile == 72U);
    static_assert(r1::rows_order(64U, 72U, mib24 / 64U, mib24).tile == 1U);
    // Launch grids are checked in 64 bits against the submitting backend's limits. HIP: x <= 0x7FFFFFFF CTAs, y <=
    // 0xFFFF CTAs, <= 0xFFFFFFFF work-items per axis (with 256 work-items per CTA the last legal x is 0xFFFFFF CTAs).
    // CUDA: x <= 2^31 - 1, y <= 65535 CTAs, no work-item limit per axis.
    constexpr unsigned threads = r1::kBlockThreads;
    constexpr auto hip = r1::kHipGridLimits;
    constexpr auto cuda = r1::kCudaGridLimits;
    static_assert(threads == 256U);
    static_assert(r1::grid_launchable(0xFFFFFFU, 1U, threads, hip) &&
                  !r1::grid_launchable(0x1000000U, 1U, threads, hip));
    static_assert(r1::grid_launchable(0x1000000U, 1U, threads, cuda) &&
                  r1::grid_launchable(0x7FFFFFFFU, 1U, threads, cuda));
    static_assert(!r1::grid_launchable(0x80000000U, 1U, threads, cuda));
    static_assert(r1::grid_launchable(0x7FFFFFFFU, 1U, 1U, hip) && !r1::grid_launchable(0x80000000U, 1U, 1U, hip));
    for (const auto& limits : {hip, cuda}) {
        expect(r1::grid_launchable(1U, 0xFFFFU, threads, limits) && !r1::grid_launchable(1U, 0x10000U, threads, limits),
               "y limit");
        expect(!r1::grid_launchable(0U, 1U, threads, limits) && !r1::grid_launchable(1U, 0U, threads, limits),
               "empty grid");
    }
    // The flat HIP rows grid (units x blocks on x) falls back to the 2D form past these limits.
    static_assert(r1::rows_order_launchable({64U, 72U, 9U}, threads, hip));
    static_assert(r1::rows_order_launchable({1U, 0xFFFFFFU, 1U}, threads, hip));
    static_assert(!r1::rows_order_launchable({1U, 0x1000000U, 1U}, threads, hip));
    static_assert(r1::rows_order_launchable({256U, 65535U, 65535U}, threads, hip));
    static_assert(!r1::rows_order_launchable({256U, 65536U, 65536U}, threads, hip));
    static_assert(!r1::rows_order_launchable({256U, 16777217U, 16777217U}, threads, hip));
    // The 2D forms (gate (gate_blocks, rows x top_k), down (down_blocks, rows)). gate_blocks(intermediate, 2) =
    // intermediate / 16: 268,435,440 -> 0xFFFFFF blocks (x = 2^32 - 256 work-items), 268,435,456 -> 2^24 blocks (exactly
    // 2^32 work-items: the review's NVFP4 rows 2, top-1, hidden 32 call). HIP refuses it; CUDA admits it, as main did.
    constexpr r1::Geometry g = r1::kGeometry;
    static_assert(r1::rows_grids_launchable(32U, 268435440U, 2U, 1U, g, threads, hip));
    static_assert(!r1::rows_grids_launchable(32U, 268435456U, 2U, 1U, g, threads, hip));
    static_assert(r1::rows_grids_launchable(32U, 268435456U, 2U, 1U, g, threads, cuda));
    static_assert(!r1::gate_grid_launchable(268435456U, 2U, 1U, g.gate_columns, threads, hip));
    static_assert(r1::gate_grid_launchable(268435456U, 2U, 1U, g.gate_columns, threads, cuda));
    // The same calls through the launches' host admission (fits_rows: Args shape, then the grids each launch submits).
    // The pointers only need to be non-null: nothing is launched or read.
    {
        alignas(64) static unsigned char storage[256];
        r1::Args a{};
        a.matrices = reinterpret_cast<decltype(a.matrices)>(storage);
        a.ids = reinterpret_cast<const unsigned*>(storage);
        a.weights = reinterpret_cast<const float*>(storage);
        a.input = reinterpret_cast<const float*>(storage);
        a.activation = reinterpret_cast<std::uint16_t*>(storage);
        a.output = reinterpret_cast<float*>(storage);
        a.experts = 1U;
        a.gate_cap = a.up_cap = 1.0F;
        a.hidden = 32U;
        a.intermediate = 268435456U;
        expect(r1::fits<packed::nvfp4, 1U>(a), "review shape passes the shape checks");
        expect(!r1::fits_rows<packed::nvfp4, 1U>(a, 2U, hip), "HIP refuses the review call (combined launch)");
        expect(!r1::fits_rows<packed::nvfp4, 1U>(a, 2U, hip, r1::Grids::gate), "HIP refuses the review gate node");
        expect(r1::fits_rows<packed::nvfp4, 1U>(a, 2U, hip, r1::Grids::down), "HIP admits the review down node");
        expect(r1::fits_rows<packed::nvfp4, 1U>(a, 2U, cuda), "CUDA admits the review call (combined launch)");
        expect(r1::fits_rows<packed::nvfp4, 1U>(a, 2U, cuda, r1::Grids::gate), "CUDA admits the review gate node");
        expect(r1::fits_rows<packed::nvfp4, 1U>(a, 2U, cuda, r1::Grids::down), "CUDA admits the review down node");
        // Each standalone node checks only its own grid: down_blocks(hidden, 4) = hidden / 4 = 2^24 blocks for hidden
        // 2^26 exceeds HIP's work-item extent on the down grid only.
        a.hidden = 67108864U;
        a.intermediate = 1152U;
        expect(r1::fits<packed::nvfp4, 1U>(a), "wide-hidden shape passes the shape checks");
        expect(!r1::fits_rows<packed::nvfp4, 1U>(a, 1U, hip, r1::Grids::down), "HIP refuses the oversized down node");
        expect(r1::fits_rows<packed::nvfp4, 1U>(a, 1U, hip, r1::Grids::gate), "HIP gate node ignores the down grid");
        expect(!r1::fits_rows<packed::nvfp4, 1U>(a, 1U, hip), "HIP combined launch checks both grids up front");
        expect(r1::fits_rows<packed::nvfp4, 1U>(a, 1U, cuda), "CUDA admits the wide down grid");
    }
    // The r12 example (8 rows, top-32, intermediate 1,048,576): flat grid refused, the 2D fallback (65,536 x 256) legal.
    static_assert(!r1::rows_order_launchable(r1::gate_up_rows_order<packed::nvfp4>(1048576U, 2304U, 8U, 32U, 2U, 0U),
                                             threads, hip));
    static_assert(r1::rows_grids_launchable(2304U, 1048576U, 8U, 32U, g, threads, hip));
    static_assert(r1::rows_grids_launchable(2304U, 1048576U, 8U, 32U, g, threads, cuda));
    // down_blocks(hidden, 4) = hidden / 4: hidden 67,108,800 -> 0xFFFFF0 blocks pass, 67,108,864 -> 2^24 refused on HIP.
    static_assert(r1::rows_grids_launchable(67108800U, 1152U, 1U, 8U, g, threads, hip));
    static_assert(!r1::rows_grids_launchable(67108864U, 1152U, 1U, 8U, g, threads, hip));
    static_assert(r1::rows_grids_launchable(67108864U, 1152U, 1U, 8U, g, threads, cuda));
    failures += check_wrap_edge() ? 0U : 1U;
    for (unsigned units : {1U, 2U, 3U, 8U, 16U, 64U}) {
        for (unsigned blocks : {1U, 5U, 72U, 576U}) {
            for (unsigned tile : {1U, 2U, 7U, 8U, 64U, 72U, 576U}) {
                if (tile > blocks) continue;
                if (!check_rows_order(units, blocks, tile)) {
                    std::printf("FAIL rows order units %u blocks %u tile %u\n", units, blocks, tile);
                    ++failures;
                }
            }
        }
    }
    for (unsigned rows = 1; rows <= r1::kMaxLaunchRows; ++rows) {
        const auto nv = r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, rows, 8U, 2U, mib24);
        const auto mx = r1::gate_up_rows_order<packed::mxfp8>(1152U, 2304U, rows, 8U, 2U, mib24);
        if (!check_rows_order(nv.units, nv.blocks, nv.tile) || !check_rows_order(mx.units, mx.blocks, mx.tile)) {
            std::printf("FAIL rows order policy rows %u\n", rows);
            ++failures;
        }
    }
    // Scratch rows: 0 = default iteration (kMaxRows), 1, kMaxRows, 12, kMaxLaunchRows and 40 (capped at one launch).
    for (unsigned scratch : {0U, 1U, r1::kMaxRows, 12U, r1::kMaxLaunchRows, 40U}) {
        for (unsigned rows = 1; rows <= 4U * r1::kMaxLaunchRows + 3U; ++rows) {
            for (bool with_failure : {true, false}) {
                if (!check_rows(rows, with_failure, scratch)) {
                    std::printf("FAIL rows %u scratch %u failure words %d\n", rows, scratch, with_failure ? 1 : 0);
                    ++failures;
                }
            }
        }
    }
    if (failures != 0U) return 1;
    std::printf("microscaled_r1 rows chunking: PASS (rows 1..%u, scratch rows default/1/8/12/16/40, UINT_MAX edge)\n",
                4U * r1::kMaxLaunchRows + 3U);
    return 0;
}
