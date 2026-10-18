#include "strix/prefix_state_cache.hpp"
#include "strix/model_content_hash.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

strix::PrefixCacheBuffer heap(std::uint64_t bytes) {
    std::shared_ptr<std::byte> data(new std::byte[static_cast<std::size_t>(bytes ? bytes : 1)],
        std::default_delete<std::byte[]>());
    return {std::move(data), bytes};
}

strix::ModelContentIdentity identity_of(const char* text) {
    return strix::sha256_bytes(std::as_bytes(std::span<const char>(text, std::strlen(text))));
}

constexpr std::uint32_t T = 4;
constexpr std::uint64_t segment_bytes = 64, fixed_bytes = 256;

// Payload bytes are a function of (block or position, salt) so restores can be checked.
std::byte pattern(std::uint64_t a, std::uint64_t i) { return static_cast<std::byte>((a * 131U + i * 7U) & 255U); }

// Builds the checkpoints of `tokens` at every block boundary, sharing segments like the engine does.
std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> build(strix::PrefixStateCache& cache,
    const strix::ModelContentIdentity& identity, const std::vector<std::uint32_t>& tokens) {
    std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> out;
    std::vector<std::shared_ptr<const strix::PrefixCacheSegment>> segments;
    auto previous = identity;
    for (std::uint64_t k = 0; k < tokens.size() / T; ++k) {
        const std::span<const std::uint32_t> block(tokens.data() + k * T, T);
        auto segment = std::make_shared<strix::PrefixCacheSegment>();
        segment->identity = identity;
        segment->parent = previous;
        segment->digest = strix::prefix_cache_block_digest(previous, block);
        segment->block_index = k;
        segment->block_rows = T;
        segment->tokens.assign(block.begin(), block.end());
        segment->payload = cache.allocate(segment_bytes);
        for (std::uint64_t i = 0; i < segment_bytes; ++i) segment->payload.data.get()[i] = pattern(k + tokens[k * T], i);
        previous = segment->digest;
        segments.push_back(segment);
        auto checkpoint = std::make_shared<strix::PrefixCacheCheckpoint>();
        checkpoint->identity = identity;
        checkpoint->digest = previous;
        checkpoint->block_rows = T;
        checkpoint->position = (k + 1) * T;
        checkpoint->segments = segments;
        checkpoint->fixed = cache.allocate(fixed_bytes);
        for (std::uint64_t i = 0; i < fixed_bytes; ++i) checkpoint->fixed.data.get()[i] = pattern(1000 + checkpoint->position, i);
        checkpoint->metadata.assign(3, static_cast<std::byte>(k));
        out.push_back(checkpoint);
    }
    return out;
}

bool same_payloads(const strix::PrefixCacheCheckpoint& a, const strix::PrefixCacheCheckpoint& b) {
    if (a.position != b.position || a.segments.size() != b.segments.size() || a.fixed.bytes != b.fixed.bytes ||
        a.metadata != b.metadata || std::memcmp(a.fixed.data.get(), b.fixed.data.get(), a.fixed.bytes) != 0)
        return false;
    for (std::size_t k = 0; k < a.segments.size(); ++k)
        if (a.segments[k]->tokens != b.segments[k]->tokens || a.segments[k]->payload.bytes != b.segments[k]->payload.bytes ||
            std::memcmp(a.segments[k]->payload.data.get(), b.segments[k]->payload.data.get(), a.segments[k]->payload.bytes) != 0)
            return false;
    return true;
}
}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("strix-prefix-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        const auto id = identity_of("identity A"), other = identity_of("identity B");
        std::vector<std::uint32_t> tokens;
        for (std::uint32_t i = 0; i < 18; ++i) tokens.push_back(100 + i);  // 4 full blocks + 2

        // Digests chain and stop at full blocks.
        const auto digests = strix::prefix_cache_block_digests(id, tokens, T, 100);
        require(digests.size() == 4, "digest count");
        require(digests[1] == strix::prefix_cache_block_digest(digests[0], std::span(tokens).subspan(4, 4)), "digest chain");
        require(strix::prefix_cache_block_digests(other, tokens, T, 100)[0] != digests[0], "identity changes digests");

        {
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            const auto checkpoints = build(cache, id, tokens);
            require(cache.find_longest(id, tokens, T, 17) == nullptr, "empty cache misses");
            for (const auto& checkpoint : checkpoints) require(cache.insert(checkpoint), "insert");
            // Shared segments are counted once: 4 segments + 4 fixed blobs + metadata.
            require(cache.stats().host_bytes == 4 * segment_bytes + 4 * (fixed_bytes + 3), "shared segment accounting");
            auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 16 && same_payloads(*hit, *checkpoints[3]), "longest hit");
            hit = cache.find_longest(id, tokens, T, 15);
            require(hit && hit->position == 12, "limit bounds the hit");
            // A different token in block 3 hits block 2 only.
            auto changed = tokens;
            changed[9] = 7;
            hit = cache.find_longest(id, changed, T, 17);
            require(hit && hit->position == 8 && same_payloads(*hit, *checkpoints[1]), "diverging tokens hit the common prefix");
            // A different identity never hits.
            require(cache.find_longest(other, tokens, T, 17) == nullptr, "identity mismatch misses");
            // A different block size never hits (different digests and positions).
            require(cache.find_longest(id, tokens, 8, 17) == nullptr, "block size mismatch misses");
            cache.flush();
            require(cache.stats().disk_write_failures == 0 && cache.stats().disk_writes == 4, "disk writes");
        }
        {
            // A fresh store finds the checkpoints on disk, byte-identical, and loads them into the host tier.
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            strix::PrefixStateCache reference({1ULL << 20, {}, 0}, heap);
            const auto expected = build(reference, id, tokens);
            auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 16 && same_payloads(*hit, *expected[3]), "disk hit restores payloads");
            require(cache.stats().disk_hits == 1 && cache.stats().host_checkpoints == 1, "disk hit is retained in host tier");
            hit = cache.find_longest(id, tokens, T, 17);
            require(cache.stats().host_hits == 1, "second lookup is a host hit");
            require(cache.find_longest(other, tokens, T, 17) == nullptr, "disk identity mismatch misses");
        }
        {
            // A corrupted fixed blob on disk is rejected (counted) and the next shorter prefix is used.
            const auto digests16 = strix::prefix_cache_block_digests(id, tokens, T, 4);
            const auto path = dir / ("ckpt-" + strix::model_content_identity_hex(digests16[3]) + ".tmpc");
            {
                std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
                file.seekp(-1, std::ios::end);
                file.put('x');
            }
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            const auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 12 && cache.stats().disk_read_failures == 1, "corrupt checkpoint is skipped");
        }
        {
            // Startup sweep: old temporaries and unreferenced segments go, referenced segments stay.
            const auto stale = std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
            const auto orphan = dir / ("seg-" + std::string(64, 'a') + ".tmps");
            const auto partial = dir / "ckpt-partial.tmpc.tmp.1";
            { std::ofstream(orphan) << "x"; std::ofstream(partial) << "x"; }
            std::filesystem::last_write_time(orphan, stale);
            std::filesystem::last_write_time(partial, stale);
            std::size_t before = 0;
            for (const auto& entry : std::filesystem::directory_iterator(dir)) before += entry.path().extension() == ".tmps";
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            std::size_t after = 0;
            for (const auto& entry : std::filesystem::directory_iterator(dir)) after += entry.path().extension() == ".tmps";
            require(!std::filesystem::exists(orphan) && !std::filesystem::exists(partial) && after + 1 == before,
                "sweep removes only stale orphans");
            require(cache.find_longest(id, tokens, T, 13) != nullptr, "referenced segments survive the sweep");
        }
        {
            // LRU within the byte budget: room for two checkpoints of one prompt (shared segments).
            const std::uint64_t budget = 3 * segment_bytes + 2 * (fixed_bytes + 3);
            strix::PrefixStateCache cache({budget, {}, 0}, heap);
            const auto checkpoints = build(cache, id, tokens);
            require(cache.insert(checkpoints[0]) && cache.insert(checkpoints[1]), "insert within budget");
            require(cache.find_longest(id, tokens, T, 9) != nullptr, "touch block 2");
            require(cache.insert(checkpoints[2]), "insert evicts");
            const auto stats = cache.stats();
            require(stats.evicted == 1 && stats.host_bytes <= budget, "evicted least recently used");
            // Block 1 was evicted, block 2 and 3 are retained.
            require(cache.find_longest(id, tokens, T, 4) == nullptr, "evicted entry misses");
            require(cache.find_longest(id, tokens, T, 12)->position == 12, "newest entry hits");
            // A checkpoint larger than the whole budget is rejected.
            strix::PrefixStateCache tiny({fixed_bytes, {}, 0}, heap);
            require(!tiny.insert(checkpoints[3]) && tiny.stats().rejected_over_budget == 1, "over budget rejected");
        }
        std::filesystem::remove_all(dir);
        std::cout << "prefix state cache tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(dir);
        std::cerr << "prefix state cache tests failed: " << error.what() << '\n';
        return 1;
    }
}
