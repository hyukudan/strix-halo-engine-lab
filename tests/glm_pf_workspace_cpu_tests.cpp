#include "glm/prefill/glm_pf_load_admission.hpp"
#include "glm/engine/glm_kvshard_exchange.hpp"
#include "glm/load/glm_rank_defaults.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

namespace pf = strix::glm::pf;
namespace mem = strix::glm::engine::mem;
namespace {
void check(bool ok) {
    if (!ok) throw std::runtime_error("GLM prefill workspace CPU check failed");
}
template<class F> bool refused(F&& f) {
    try { f(); }
    catch (const std::runtime_error& e) {
        check(std::string(e.what()).find("glm pf path: refusing to allocate ") == 0);
        return true;
    }
    return false;
}
// Inject an identity physical model so these tests isolate owner accounting.
// The runtime supplies the calibrated AllocModel, including its real granule.
struct PhysicalModel { std::uint64_t physical(std::uint64_t bytes) const { return bytes; } };
struct Admission {
    std::vector<mem::Term> terms;
    mem::Snapshot snapshot;
    mem::Margins margins{0, 0};
    std::uint64_t prefill_install = 0, device_free = 0;
    std::uint64_t post_load_device_bytes = 0, post_load_host_bytes = 0;
    std::uint64_t first_forward_reserve = 0;
    PhysicalModel model;
};

bool same_layout(pf::CapacityLayout a, pf::CapacityLayout b) {
    return a.routed == b.routed && a.query_window == b.query_window && a.sharded == b.sharded;
}
void admission_matrix(pf::CapacityPolicy policy, bool initial_sharded, bool shard_auto,
                      const std::vector<pf::CapacityLayout>& expected) {
    // Every per-rank admission combination, including asymmetric refusals and
    // nonmonotone outcomes. A later rank-local success cannot rescue a different
    // candidate on the peer. Also verify early exit at the first joint success.
    const unsigned count = static_cast<unsigned>(expected.size());
    for (unsigned skip = 0; skip <= count; ++skip)
    for (unsigned mask = 0; mask < (1U << (2 * count)); ++mask) {
        unsigned first = count;
        for (unsigned i = skip; i < count; ++i)
            if (((mask >> (2 * i)) & 3U) == 3U) { first = i; break; }
        unsigned calls = 0;
        pf::CapacityLayout selected;
        const bool admitted = pf::select_capacity_layout(policy, initial_sharded, shard_auto,
            [&](pf::CapacityLayout layout) {
                check(calls < count && same_layout(layout, expected[skip + calls]));
                const auto bits = mask >> (2 * (skip + calls++));
                return std::array<bool, 2>{(bits & 1U) != 0, (bits & 2U) != 0};
            }, selected, skip);
        check(admitted == (first < count));
        check(calls == (admitted ? first + 1 : count) - skip);
        if (admitted) check(same_layout(selected, expected[first]));
    }
}
}

int main() {
    using Scratch = pf::ScratchPolicy;
    check(pf::parse_scratch_policy(nullptr) == Scratch::automatic);
    check(pf::parse_scratch_policy("auto") == Scratch::automatic);
    check(pf::parse_scratch_policy("full") == Scratch::full);
    check(pf::parse_scratch_policy("routed") == Scratch::routed);
    for (const char* value : {"", "0", "1", "AUTO", "typo"}) {
        bool invalid = false;
        try { (void)pf::parse_scratch_policy(value); }
        catch (const std::invalid_argument&) { invalid = true; }
        check(invalid);
    }
    check(pf::parse_query_window_policy(nullptr) == -1);
    check(pf::parse_query_window_policy("auto") == -1);
    check(pf::parse_query_window_policy("0") == 0);
    check(pf::parse_query_window_policy("2048") == 2048);
    check(pf::parse_query_window_policy("65536") == 65536);
    admission_matrix({}, false, true, {{false, 0, false}, {true, 0, false}, {true, 2048, false},
                                      {false, 0, true}, {true, 0, true}, {true, 2048, true}});
    admission_matrix({}, false, false, {{false, 0, false}, {true, 0, false}, {true, 2048, false}});
    admission_matrix({}, true, false, {{false, 0, true}, {true, 0, true}, {true, 2048, true}});
    admission_matrix({Scratch::full, -1}, false, true,
                     {{false, 0, false}, {false, 2048, false}, {false, 0, true}, {false, 2048, true}});
    admission_matrix({Scratch::routed, -1}, false, false, {{true, 0, false}, {true, 2048, false}});
    admission_matrix({Scratch::automatic, 0}, false, false, {{false, 0, false}, {true, 0, false}});
    admission_matrix({Scratch::automatic, 17}, false, false, {{false, 17, false}, {true, 17, false}});
    admission_matrix({Scratch::full, 0}, false, false, {{false, 0, false}});
    admission_matrix({Scratch::routed, 2048}, false, true, {{true, 2048, false}, {true, 2048, true}});
    std::cout << "PASS: AUTO capacity admission matrices, explicit pins and first joint-fit ordering\n";
    // Two independent ranks, with load+KV, communication, forward reserve,
    // path/workspace and the layout-independent install/staging allowance.
    // One-byte boundaries include asymmetric headroom. Without the install term
    // the first layout would admit and subsequently fail its install guard.
    for (bool shard_auto : {false, true}) {
        const std::vector<pf::CapacityLayout> order{{false, 0, false}, {true, 0, false}, {true, 2048, false},
                                                   {false, 0, true}, {true, 0, true}, {true, 2048, true}};
        for (unsigned install : {0U, 76U, 192U}) {
            for (unsigned left = 0; left <= 512; ++left) {
                for (int skew : {-1, 0, 1}) {
                    unsigned first = shard_auto ? 6U : 3U;
                    unsigned calls = 0;
                    pf::CapacityLayout selected;
                    const auto need = [&](unsigned i) {
                        return (i < 3 ? 100U : 40U) + (i % 3 == 0 ? 200U : i % 3 == 1 ? 120U : 60U)
                            + 20U + 8U + 16U + 12U + install;
                    };
                    const auto right = static_cast<unsigned>(static_cast<int>(left) + skew + 1);
                    for (unsigned i = 0; i < (shard_auto ? 6U : 3U); ++i)
                        if (left >= need(i) && right >= need(i)) { first = i; break; }
                    const bool admitted = pf::select_capacity_layout({}, false, shard_auto,
                        [&](pf::CapacityLayout layout) {
                            const unsigned i = calls++;
                            check(same_layout(layout, order[i]));
                            std::array<bool, 2> fits{};
                            for (unsigned rank = 0; rank < 2; ++rank) {
                                Admission a;
                                a.snapshot.meminfo_ok = true;
                                a.prefill_install = install;
                                a.post_load_device_bytes = a.post_load_host_bytes = i < 3 ? 100 : 40;
                                a.first_forward_reserve = 12;
                                a.device_free = a.snapshot.mem_available = rank == 0 ? left : right;
                                const auto path = i % 3 == 0 ? 200 : i % 3 == 1 ? 120 : 60;
                                fits[rank] = !refused([&] { pf::admit_prefill_load(a, path, 20, 8, 16); });
                            }
                            return fits;
                        }, selected);
                    check(admitted == (first < (shard_auto ? 6U : 3U)));
                    if (admitted) check(same_layout(selected, order[first]));
                }
            }
        }
    }
    // Late install refusals from either rank advance the shared cursor. Check
    // every success position, alternating which rank refuses, and exhaustion.
    const std::vector<pf::CapacityLayout> retry_order{{false, 0, false}, {true, 0, false}, {true, 2048, false},
                                                    {false, 0, true}, {true, 0, true}, {true, 2048, true}};
    for (unsigned refusing_rank : {0U, 1U}) {
        for (unsigned success = 0; success <= retry_order.size(); ++success) {
            unsigned skip = 0;
            bool installed = false;
            for (;;) {
                pf::CapacityLayout selected;
                const bool admitted = pf::select_capacity_layout({}, false, true, [&](pf::CapacityLayout layout) {
                    check(skip < retry_order.size() && same_layout(layout, retry_order[skip]));
                    return std::array<bool, 2>{true, true};
                }, selected, skip);
                if (!admitted) break;
                check(same_layout(selected, retry_order[skip]));
                std::array<bool, 2> install_fits{true, true};
                if (skip < success) install_fits[(skip + refusing_rank) % 2] = false;
                if (install_fits[0] && install_fits[1]) { installed = true; break; }
                ++skip;
            }
            check(skip == success && installed == (success < retry_order.size()));
        }
    }
    std::cout << "PASS: install/staging admission matrix and post-install retry order\n";
    namespace rd = strix::glm::rank_defaults;
    check(rd::resolve_prefill_auto(false, false, false, false, true, false, 0, false) == "gemv");
    check(rd::resolve_prefill_auto(true, false, false, false, true, false, 0, false, true) == "gemv");
    check(rd::resolve_prefill_auto(false, false, false, false, false, false, 0, false) == "gemv");
    check(rd::resolve_prefill_auto(false, false, true, false, true, false, 0, false) == "ref");
    check(pf::indexer_workspace_bytes(8192) == 144 * mem::kMiB);
    check(pf::indexer_workspace_bytes(4096) == 72 * mem::kMiB);
    for (const bool grouped_shape : {false, true}) {
        for (const bool exact : {false, true}) {
            const auto gemv = pf::prefill_moe_route(pf::TierB::gemv, grouped_shape, exact);
            const auto wmma = pf::prefill_moe_route(pf::TierB::wmma, grouped_shape, exact);
            check(!gemv.legacy_wmma && gemv.exact == (grouped_shape && exact));
            check(wmma.legacy_wmma == grouped_shape && !wmma.exact);
            check(gemv.legacy_arena == gemv.legacy_wmma);
            check(wmma.legacy_arena == wmma.legacy_wmma);
            for (auto tier : {pf::TierB::gemv, pf::TierB::wmma}) {
                const auto full = pf::prefill_moe_route(tier, grouped_shape, exact, false);
                const auto routed = pf::prefill_moe_route(tier, grouped_shape, exact, true);
                check(full.legacy_arena);
                check(full.legacy_wmma == routed.legacy_wmma && full.exact == routed.exact);
            }
        }
    }
    check(pf::parse_attention_query_window(nullptr) == 0);
    check(pf::parse_attention_query_window("0") == 0);
    check(pf::parse_attention_query_window("2048") == 2048);
    check(pf::parse_attention_query_window("65536") == 65536);
    for (const char* value : {"", "-1", "+1", " 1", "1x", "65537", "4294967296"}) {
        bool invalid = false;
        try { (void)pf::parse_attention_query_window(value); }
        catch (const std::invalid_argument&) { invalid = true; }
        check(invalid);
    }
    for (unsigned rows : {0U, 1U, 17U, 2047U, 2048U, 2049U, 8192U, 65536U}) {
        check(pf::attention_scratch_rows(rows, 0) == rows);
        for (unsigned window : {1U, 17U, 2048U, 65536U}) {
            const unsigned scratch = pf::attention_scratch_rows(rows, window);
            check(scratch == std::min(rows, window));
            unsigned covered = 0;
            while (covered < rows) {
                const unsigned piece = std::min(scratch, rows - covered);
                check(piece > 0 && piece <= scratch);
                covered += piece;
            }
            check(covered == rows);
        }
    }

    { // Synthetic replicated startup boundary, not a machine capacity claim.
        // 11 trunk DSA layers + MTP: 12 * (1024 + 256 + 256 + 512/4)
        // bytes/token, plus the slot's FP32 pool scores and u32 work (8/4).
        constexpr std::uint64_t per_token = 12 * 1664 + 2;
        constexpr std::uint64_t legacy = 2899324672ULL; // existing 64-bit ABI bound at 8192 rows
        constexpr std::uint64_t attn = std::uint64_t{8192 - 2048} * 32 * (512 + 2304 + 512) * 4;
        static_assert(attn == 2617245696ULL);
        constexpr std::uint64_t old_capacity = 131072, old_path = 16 * mem::kGiB;
        constexpr std::uint64_t fixed = 64 * mem::kGiB, workspace = 144 * mem::kMiB, margin = 8 * mem::kGiB;
        Admission boundary;
        boundary.snapshot.meminfo_ok = true;
        boundary.device_free = fixed + old_capacity * per_token + old_path + workspace + margin;
        boundary.snapshot.mem_available = boundary.device_free;
        for (const auto saved : {std::uint64_t{0}, legacy, legacy + attn}) {
            const auto capacity = old_capacity + (saved / per_token / 4) * 4;
            const auto path_bytes = old_path - saved;
            for (unsigned extra : {0U, 4U}) {
                boundary.post_load_device_bytes = fixed + (capacity + extra) * per_token;
                boundary.post_load_host_bytes = boundary.post_load_device_bytes;
                const bool startup = !refused([&] { pf::admit_prefill_load(boundary, path_bytes, workspace, margin, 0); });
                const auto free = boundary.device_free - boundary.post_load_device_bytes;
                const bool loaded = !refused([&] { pf::admit_prefill_bytes(path_bytes, workspace, margin, free, free); });
                check(startup == (extra == 0) && loaded == startup);
            }
            std::cout << "synthetic replicated max-context=" << capacity << " saved-bytes=" << saved << " PASS\n";
        }
    }
    pf::admit_prefill_bytes(100, 20, 8, 128, 128);
    check(refused([] { pf::admit_prefill_bytes(100, 20, 8, 127, 128); }));
    check(refused([] { pf::admit_prefill_bytes(100, 20, 8, 128, 127); }));
    check(refused([] { pf::admit_prefill_bytes(std::numeric_limits<std::uint64_t>::max(), 1, 0, 0, 0); }));

    Admission a;
    a.snapshot.meminfo_ok = true;
    a.post_load_device_bytes = 1000;
    a.post_load_host_bytes = 1100;
    // Load terms stay separate; install/staging must survive the projection.
    a.terms = {{"temporary load staging and reserves", 5000, 5000}};
    a.prefill_install = 5000;
    a.snapshot.gpu = true;
    a.snapshot.gtt_total = 1;
    a.margins = {8000, 8000};
    // Device: retained 1000 + comm 40 + install 5000 + path 200 + workspace 30 + margin 20.
    // Host: retained 1100 + comm 40 + the same install/path/workspace/margin.
    a.device_free = 6290;
    a.snapshot.mem_available = 6390;
    pf::admit_prefill_load(a, 200, 30, 20, 40);
    --a.device_free;
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));
    ++a.device_free;
    --a.snapshot.mem_available;
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));
    ++a.snapshot.mem_available;
    a.snapshot.meminfo_ok = false;
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));
    a.snapshot.meminfo_ok = true;
    a.post_load_device_bytes = std::numeric_limits<std::uint64_t>::max();
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));

    a.prefill_install = 0;

    // Receipt-derived, rounded 8192-row fixture, not a device qualification.
    // The post-load oracle is independent of the startup ledger subtraction.
    constexpr std::uint64_t path = 6336 * mem::kMiB, workspace = 144 * mem::kMiB;
    constexpr std::uint64_t margin = 8 * mem::kGiB;
    for (const std::uint64_t capacity : {1090000ULL, 1190000ULL, 1500000ULL, 2000000ULL}) {
        const auto post_device = 306 * mem::kGiB / 1000 + (2027520 - capacity) * 17737 + 43 * mem::kGiB / 10;
        const auto post_host = post_device + 2 * mem::kGiB;
        const bool expected = post_device >= path + workspace + margin && post_host >= path + workspace + margin;
        a.post_load_device_bytes = 65 * mem::kGiB + capacity * 17737;
        a.post_load_host_bytes = a.post_load_device_bytes + 7 * mem::kGiB;
        a.device_free = post_device + a.post_load_device_bytes;
        a.snapshot.mem_available = post_host + a.post_load_host_bytes;
        const bool startup = !refused([&] { pf::admit_prefill_load(a, path, workspace, margin, 0); });
        const bool loaded = !refused([&] { pf::admit_prefill_bytes(path, workspace, margin, post_device, post_host); });
        check(startup == expected && loaded == expected);
        check(expected == (capacity <= 1190000));
        std::cout << "capacity " << capacity << ": startup=" << (startup ? "ADMIT" : "REFUSE")
                  << " post-load=" << (loaded ? "ADMIT" : "REFUSE") << " PASS\n";
    }

    // Check the integer top-k decomposition independently of score arithmetic:
    // retained prefix + next window gives the global set AND ascending tie order.
    std::mt19937 gen(41);
    unsigned cases = 0;
    for (unsigned size : {0U, 1U, 511U, 512U, 513U, 4095U, 4096U, 4097U, 8193U, 16385U}) {
        for (unsigned pattern = 0; pattern < 5; ++pattern) {
            std::vector<std::uint32_t> keys(size);
            for (unsigned i = 0; i < size; ++i)
                keys[i] = static_cast<std::uint32_t>(pattern == 0 ? 7U : pattern == 1 ? i : pattern == 2 ? size - i : pattern == 3 ? gen() % 5U : gen());
            const auto better = [&](unsigned x, unsigned y) { return keys[x] != keys[y] ? keys[x] > keys[y] : x < y; };
            std::vector<unsigned> all, kept;
            for (unsigned i = 0; i < size; ++i) all.push_back(i);
            auto top = [&](auto& v) {
                std::sort(v.begin(), v.end(), better);
                if (v.size() > pf::kIndexerRetainedPools) v.resize(pf::kIndexerRetainedPools);
                std::sort(v.begin(), v.end());
            };
            top(all);
            for (unsigned begin = 0; begin < size; begin += pf::kIndexerPoolWindow) {
                for (unsigned i = begin; i < std::min(size, begin + pf::kIndexerPoolWindow); ++i) kept.push_back(i);
                top(kept);
            }
            check(kept == all);
            ++cases;
        }
    }
    // Boundary fixture anchored at an admitted replicated 1.45M, not a device
    // measurement. Both candidate layouts use the very same post-load oracle.
    namespace ks = strix::glm::kvshard;
    constexpr std::uint64_t cache_slope = 11 * ks::bytes_per_token;
    constexpr std::uint64_t forward = mem::kGiB / 2;
    constexpr std::uint64_t fixed = path + workspace + margin + forward;
    constexpr std::uint64_t available = fixed + 1450000 * cache_slope;
    for (const std::uint64_t capacity : {1190000ULL, 1450000ULL, 1600000ULL, 2000000ULL, 2850000ULL, 3000000ULL}) {
        const auto fits_layout = [&](bool shard) {
            const auto cache = (shard ? ks::storage_tokens(capacity) : capacity) * cache_slope;
            Admission candidate;
            candidate.first_forward_reserve = forward;
            candidate.device_free = available;
            candidate.snapshot.meminfo_ok = true;
            candidate.snapshot.cma_free_ok = candidate.snapshot.zoneinfo_ok = true;
            candidate.snapshot.mem_available = available + 2 * mem::kGiB;
            candidate.post_load_device_bytes = cache;
            candidate.post_load_host_bytes = cache;
            const auto temporary = shard ? pf::kKvShardPrefillTemporaryBytes : 0;
            const bool startup = !refused([&] {
                pf::admit_prefill_load(candidate, path, workspace + temporary, margin, 0);
            });
            const bool loaded = !refused([&] {
                pf::admit_prefill_bytes(path, workspace + temporary, margin + forward,
                                        available > cache ? available - cache : 0,
                                        available + 2 * mem::kGiB > cache ? available + 2 * mem::kGiB - cache : 0);
            });
            check(startup == loaded);
            if (!shard && capacity == 1600000) {
                check(mem::fits({{"cache", cache, 0}, {"first forward", forward, 0}},
                                candidate.snapshot, {margin, margin}, candidate.device_free));
                check(!startup);  // regression: a fitting load plan is insufficient
            }
            return startup;
        };
        const bool replicated = fits_layout(false), sharded = fits_layout(true);
        const auto selected = ks::prefill_layout(replicated, replicated, sharded, sharded);
        const auto expected = capacity <= 1450000 ? ks::PrefillLayout::replicated :
            capacity <= 2850000 ? ks::PrefillLayout::sharded : ks::PrefillLayout::refuse;
        check(selected == expected);
        std::cout << "auto capacity " << capacity << ": replicated=" << replicated << " sharded=" << sharded
                  << " selected=" << (selected == ks::PrefillLayout::replicated ? "replicated" :
                                      selected == ks::PrefillLayout::sharded ? "sharded" : "refuse") << " PASS\n";
    }
    // Host ownership oracle: append deliberately awkward chunks, retain a
    // partial pool on its owner, and reconstruct bounded windows without ever
    // consuming the poisoned padding of the other partition.
    constexpr unsigned total = 2 * pf::kIndexerPoolWindow * 4 + 12;
    const auto storage = static_cast<std::size_t>(ks::storage_tokens(total));
    constexpr std::uint32_t poison = 0xdeadbeefU;
    std::array<std::vector<std::uint32_t>, 2> cache{std::vector<std::uint32_t>(storage, poison),
                                                  std::vector<std::uint32_t>(storage, poison)};
    std::array<std::vector<std::uint32_t>, 2> pools{std::vector<std::uint32_t>(storage / 4, poison),
                                                  std::vector<std::uint32_t>(storage / 4, poison)};
    unsigned frontier = 0;
    for (const unsigned chunk : {1U, 2U, 4U, 4094U, 8193U, total}) {
        const unsigned end = std::min(total, frontier + chunk);
        for (unsigned token = frontier; token < end; ++token)
            cache[ks::owner(token)][static_cast<std::size_t>(ks::local_token(token))] = token;
        for (unsigned pool = frontier / 4; pool < end / 4; ++pool) {
            const unsigned owner = pool % 2;
            const auto local = static_cast<std::size_t>(pool / 2);
            std::uint32_t ordered = 0;
            for (unsigned j = 0; j < 4; ++j) {
                const auto token = cache[owner][local * 4 + j];
                check(token == pool * 4 + j);  // includes prior chunk's carry
                ordered = ordered * 33U + token;
            }
            pools[owner][local] = ordered;
        }
        frontier = end;
    }
    check(frontier == total);
    unsigned windows = 0;
    for (unsigned begin = 0; begin < total / 4; begin += pf::kIndexerPoolWindow) {
        const unsigned end = std::min(total / 4, begin + pf::kIndexerPoolWindow);
        std::vector<std::uint32_t> window(end - begin);
        for (unsigned pool = begin; pool < end; ++pool) {
            window[pool - begin] = pools[pool % 2][pool / 2];
            std::uint32_t expected = 0;
            for (unsigned j = 0; j < 4; ++j) expected = expected * 33U + pool * 4 + j;
            check(window[pool - begin] == expected);
        }
        check(window.size() <= pf::kIndexerPoolWindow);
        ++windows;
    }
    check(windows == 3);
    // Selected-token ordinals preserve canonical order and original cache bits
    // at owner transitions, the 4096-pool boundary and an incomplete row tail.
    for (const unsigned token : {0U, 3U, 4U, 7U, 8U, 16383U, 16384U, total - 1})
        check(cache[ks::owner(token)][static_cast<std::size_t>(ks::local_token(token))] == token);
    std::cout << "PASS: kv-shard ownership/carry, three bounded windows and selected gather oracle\n";
    // Neither a remote refusal nor a local refusal may be hidden by the peer.
    for (unsigned mask = 0; mask < 16; ++mask) {
        const bool r0 = (mask & 1U) != 0, r1 = (mask & 2U) != 0;
        const bool s0 = (mask & 4U) != 0, s1 = (mask & 8U) != 0;
        const auto expected = r0 && r1 ? ks::PrefillLayout::replicated :
            s0 && s1 ? ks::PrefillLayout::sharded : ks::PrefillLayout::refuse;
        check(ks::prefill_layout(r0, r1, s0, s1) == expected);
    }
    std::cout << "PASS: workspace/admission boundaries, 16 two-rank votes and " << cases << " ordered top-k cases\n";
    return EXIT_SUCCESS;
}
