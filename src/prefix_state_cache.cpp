#include "strix/prefix_state_cache.hpp"

#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>

#if defined(__unix__)
#include <unistd.h>
#endif

namespace strix {
namespace {

constexpr std::uint64_t segment_magic = 0x31474553'4350'4d54ULL;    // "TMPCSEG1"
constexpr std::uint64_t checkpoint_magic = 0x31504b43'4350'4d54ULL; // "TMPCCKP1"

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

std::string hex(const ModelContentIdentity& digest) { return model_content_identity_hex(digest); }

std::filesystem::path segment_path(const std::filesystem::path& dir, const ModelContentIdentity& digest) {
    return dir / ("seg-" + hex(digest) + ".tmps");
}
std::filesystem::path checkpoint_path(const std::filesystem::path& dir, const ModelContentIdentity& digest) {
    return dir / ("ckpt-" + hex(digest) + ".tmpc");
}

// Fixed-width little-endian header fields; the engine runs on little-endian hosts only.
static_assert(std::endian::native == std::endian::little, "prefix cache files are little-endian");

struct SegmentHeader {
    std::uint64_t magic = segment_magic;
    std::uint8_t identity[32]{}, digest[32]{}, parent[32]{};
    std::uint64_t block_index = 0;
    std::uint32_t block_rows = 0, reserved = 0;
    std::uint64_t payload_bytes = 0, payload_checksum = 0;
};
struct CheckpointHeader {
    std::uint64_t magic = checkpoint_magic;
    std::uint8_t identity[32]{}, digest[32]{};
    std::uint64_t position = 0;
    std::uint32_t block_rows = 0, reserved = 0;
    std::uint64_t segment_count = 0, fixed_bytes = 0, metadata_bytes = 0;
    std::uint64_t fixed_checksum = 0, metadata_checksum = 0;
};
static_assert(sizeof(SegmentHeader) == 8 + 96 + 8 + 8 + 16);
static_assert(sizeof(CheckpointHeader) == 8 + 64 + 8 + 8 + 40);

void put(std::uint8_t (&out)[32], const ModelContentIdentity& value) { std::memcpy(out, value.bytes.data(), 32); }
ModelContentIdentity get(const std::uint8_t (&in)[32]) {
    ModelContentIdentity value;
    std::memcpy(value.bytes.data(), in, 32);
    return value;
}

class File final {
public:
    File(const std::filesystem::path& path, const char* mode) : file_(std::fopen(path.string().c_str(), mode)) {}
    ~File() { if (file_) std::fclose(file_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    [[nodiscard]] bool open() const noexcept { return file_ != nullptr; }
    void read(void* data, std::uint64_t bytes) {
        auto* out = static_cast<std::byte*>(data);
        while (bytes) {
            const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, 64ULL << 20));
            if (std::fread(out, 1, chunk, file_) != chunk) throw std::runtime_error("prefix cache file is truncated");
            out += chunk;
            bytes -= chunk;
        }
    }
    void write(const void* data, std::uint64_t bytes) {
        const auto* in = static_cast<const std::byte*>(data);
        while (bytes) {
            const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, 64ULL << 20));
            if (std::fwrite(in, 1, chunk, file_) != chunk) throw std::runtime_error("prefix cache file write failed");
            in += chunk;
            bytes -= chunk;
        }
    }
    void close() {
        if (file_ && std::fclose(file_) != 0) { file_ = nullptr; throw std::runtime_error("prefix cache file close failed"); }
        file_ = nullptr;
    }
private:
    std::FILE* file_ = nullptr;
};

std::filesystem::path temporary_path(const std::filesystem::path& path) {
    long pid = 0;
#if defined(__unix__)
    pid = static_cast<long>(::getpid());
#endif
    return path.string() + ".tmp." + std::to_string(pid);
}

std::uint64_t checkpoint_host_bytes(const PrefixCacheCheckpoint& checkpoint) {
    return checkpoint.fixed.bytes + checkpoint.metadata.size();
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

std::vector<std::uint32_t> PrefixCacheCheckpoint::tokens() const {
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(position));
    for (const auto& segment : segments) out.insert(out.end(), segment->tokens.begin(), segment->tokens.end());
    return out;
}

ModelContentIdentity prefix_cache_block_digest(const ModelContentIdentity& previous,
    std::span<const std::uint32_t> block_tokens) {
    std::vector<std::byte> buffer(32 + block_tokens.size_bytes());
    std::memcpy(buffer.data(), previous.bytes.data(), 32);
    if (!block_tokens.empty()) std::memcpy(buffer.data() + 32, block_tokens.data(), block_tokens.size_bytes());
    return sha256_bytes(buffer);
}

std::vector<ModelContentIdentity> prefix_cache_block_digests(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t max_blocks) {
    require(block_rows != 0 && identity.valid(), "prefix cache digests need an identity and a block size");
    const auto blocks = std::min<std::uint64_t>(max_blocks, tokens.size() / block_rows);
    std::vector<ModelContentIdentity> digests;
    digests.reserve(static_cast<std::size_t>(blocks));
    auto previous = identity;
    for (std::uint64_t k = 0; k < blocks; ++k) {
        previous = prefix_cache_block_digest(previous, tokens.subspan(static_cast<std::size_t>(k * block_rows), block_rows));
        digests.push_back(previous);
    }
    return digests;
}

std::uint64_t prefix_cache_checksum(std::span<const std::byte> bytes) noexcept {
    constexpr std::uint64_t prime = 0x9e3779b97f4a7c15ULL;
    std::uint64_t lane[4] = {0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL};
    const auto* data = bytes.data();
    std::size_t size = bytes.size(), offset = 0;
    for (; offset + 32 <= size; offset += 32) {
        for (unsigned i = 0; i < 4; ++i) {
            std::uint64_t word;
            std::memcpy(&word, data + offset + 8 * i, 8);
            lane[i] = std::rotl((lane[i] ^ word) * prime, 29);
        }
    }
    std::uint64_t tail = 0;
    for (unsigned shift = 0; offset < size; ++offset, shift += 8)
        tail ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(data[offset])) << (shift % 64);
    std::uint64_t hash = static_cast<std::uint64_t>(size) * prime;
    for (const auto value : lane) hash = std::rotl((hash ^ value) * prime, 31);
    return std::rotl((hash ^ tail) * prime, 27) ^ (hash >> 17);
}

PrefixStateCache::PrefixStateCache(PrefixStateCachePolicy policy, PrefixCacheAllocator allocate)
    : policy_(std::move(policy)), allocate_(std::move(allocate)) {
    require(policy_.host_byte_budget != 0 && static_cast<bool>(allocate_),
        "prefix state cache needs a positive host byte budget and an allocator");
    if (!policy_.disk_directory.empty()) {
        std::filesystem::create_directories(policy_.disk_directory);
        // Leftovers of interrupted writers (old enough not to belong to a live process).
        sweep_disk(std::chrono::minutes(10));
        writer_ = std::thread([this] { writer_loop(); });
    }
}

PrefixStateCache::~PrefixStateCache() {
    if (writer_.joinable()) {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        writer_.join();
    }
}

PrefixCacheBuffer PrefixStateCache::allocate(std::uint64_t bytes) const {
    auto buffer = allocate_(bytes);
    if (bytes && (!buffer.data || buffer.bytes < bytes)) throw std::runtime_error("prefix cache allocation failed");
    buffer.bytes = bytes;
    return buffer;
}

std::uint64_t PrefixStateCache::host_bytes_locked() {
    std::uint64_t total = 0;
    for (auto it = segments_.begin(); it != segments_.end();) {
        if (const auto segment = it->second.lock()) {
            total += segment->payload.bytes;
            ++it;
        } else {
            it = segments_.erase(it);
        }
    }
    for (const auto& [digest, entry] : entries_) total += checkpoint_host_bytes(*entry.checkpoint);
    return total;
}

std::uint64_t PrefixStateCache::incremental_bytes(const PrefixCacheCheckpoint& checkpoint) {
    std::lock_guard lock(mutex_);
    return incremental_bytes_locked(checkpoint);
}

std::uint64_t PrefixStateCache::incremental_bytes_locked(const PrefixCacheCheckpoint& checkpoint) {
    std::uint64_t total = checkpoint_host_bytes(checkpoint);
    for (const auto& segment : checkpoint.segments) {
        const auto found = segments_.find(segment->digest);
        if (found == segments_.end() || found->second.expired()) total += segment->payload.bytes;
    }
    return total;
}

void PrefixStateCache::touch_locked(Entry& entry) {
    lru_.splice(lru_.end(), lru_, entry.lru);
}

void PrefixStateCache::evict_to_budget_locked(std::uint64_t reserve) {
    while (!lru_.empty() && host_bytes_locked() + reserve > policy_.host_byte_budget) {
        const auto digest = lru_.front();
        lru_.pop_front();
        entries_.erase(digest);
        ++stats_.evicted;
    }
}

bool PrefixStateCache::insert(std::shared_ptr<const PrefixCacheCheckpoint> checkpoint) {
    require(checkpoint && checkpoint->block_rows && checkpoint->position == checkpoint->segments.size() * checkpoint->block_rows,
        "prefix cache checkpoint is malformed");
    {
        std::lock_guard lock(mutex_);
        if (const auto found = entries_.find(checkpoint->digest); found != entries_.end()) {
            touch_locked(found->second);
            return true;
        }
        const auto need = incremental_bytes_locked(*checkpoint);
        if (need > policy_.host_byte_budget) {
            ++stats_.rejected_over_budget;
            return false;
        }
        evict_to_budget_locked(need);
        lru_.push_back(checkpoint->digest);
        entries_.emplace(checkpoint->digest, Entry{checkpoint, std::prev(lru_.end())});
        for (const auto& segment : checkpoint->segments) segments_[segment->digest] = segment;
        ++stats_.inserted;
    }
    if (writer_.joinable()) {
        {
            std::lock_guard lock(queue_mutex_);
            queue_.push_back(std::move(checkpoint));
        }
        queue_cv_.notify_one();
    }
    return true;
}

bool PrefixStateCache::contains(const ModelContentIdentity& digest) {
    {
        std::lock_guard lock(mutex_);
        if (entries_.count(digest)) return true;
    }
    if (policy_.disk_directory.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(checkpoint_path(policy_.disk_directory, digest), error);
}

std::shared_ptr<const PrefixCacheSegment> PrefixStateCache::find_segment(const ModelContentIdentity& digest) {
    std::lock_guard lock(mutex_);
    const auto found = segments_.find(digest);
    return found == segments_.end() ? nullptr : found->second.lock();
}

std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::find_longest(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t limit) {
    require(block_rows != 0, "prefix cache lookup needs a block size");
    const auto max_blocks = std::min<std::uint64_t>(limit, tokens.size()) / block_rows;
    const auto digests = prefix_cache_block_digests(identity, tokens, block_rows, max_blocks);
    {
        std::lock_guard lock(mutex_);
        ++stats_.lookups;
    }
    for (auto k = digests.size(); k > 0; --k) {
        const auto& digest = digests[k - 1];
        std::shared_ptr<const PrefixCacheCheckpoint> found;
        {
            std::lock_guard lock(mutex_);
            if (const auto it = entries_.find(digest); it != entries_.end()) {
                found = it->second.checkpoint;
                touch_locked(it->second);
            }
        }
        if (found) {
            const auto position = static_cast<std::uint64_t>(k) * block_rows;
            const auto stored = found->tokens();
            if (found->identity == identity && found->block_rows == block_rows && found->position == position &&
                stored.size() == position && std::equal(stored.begin(), stored.end(), tokens.begin())) {
                std::lock_guard lock(mutex_);
                ++stats_.host_hits;
                return found;
            }
            continue;  // digest collision or corrupt entry: never used
        }
        if (policy_.disk_directory.empty()) continue;
        std::error_code error;
        if (!std::filesystem::exists(checkpoint_path(policy_.disk_directory, digest), error)) continue;
        try {
            if (auto loaded = load_from_disk(identity, digest, tokens.first(static_cast<std::size_t>(k * block_rows)), block_rows)) {
                {
                    std::lock_guard lock(mutex_);
                    ++stats_.disk_hits;
                }
                // Into the host tier without writing it back.
                {
                    std::lock_guard lock(mutex_);
                    const auto need = checkpoint_host_bytes(*loaded) + [&] {
                        std::uint64_t bytes = 0;
                        for (const auto& segment : loaded->segments) {
                            const auto it = segments_.find(segment->digest);
                            if (it == segments_.end() || it->second.expired()) bytes += segment->payload.bytes;
                        }
                        return bytes;
                    }();
                    if (need <= policy_.host_byte_budget && !entries_.count(loaded->digest)) {
                        evict_to_budget_locked(need);
                        lru_.push_back(loaded->digest);
                        entries_.emplace(loaded->digest, Entry{loaded, std::prev(lru_.end())});
                        for (const auto& segment : loaded->segments) segments_[segment->digest] = segment;
                    }
                }
                std::filesystem::last_write_time(checkpoint_path(policy_.disk_directory, digest),
                    std::filesystem::file_time_type::clock::now(), error);
                return loaded;
            }
        } catch (const std::exception&) {
            std::lock_guard lock(mutex_);
            ++stats_.disk_read_failures;
        }
    }
    std::lock_guard lock(mutex_);
    ++stats_.misses;
    return nullptr;
}

std::shared_ptr<const PrefixCacheSegment> PrefixStateCache::load_segment(const ModelContentIdentity& identity,
    const ModelContentIdentity& digest, std::uint32_t block_rows) {
    if (auto retained = find_segment(digest)) return retained;
    File file(segment_path(policy_.disk_directory, digest), "rb");
    if (!file.open()) throw std::runtime_error("prefix cache segment file is missing");
    SegmentHeader header;
    file.read(&header, sizeof(header));
    if (header.magic != segment_magic || get(header.identity) != identity || get(header.digest) != digest ||
        header.block_rows != block_rows)
        throw std::runtime_error("prefix cache segment header does not match");
    auto segment = std::make_shared<PrefixCacheSegment>();
    segment->identity = identity;
    segment->digest = digest;
    segment->parent = get(header.parent);
    segment->block_index = header.block_index;
    segment->block_rows = block_rows;
    segment->tokens.resize(block_rows);
    file.read(segment->tokens.data(), static_cast<std::uint64_t>(block_rows) * sizeof(std::uint32_t));
    if (prefix_cache_block_digest(segment->parent, segment->tokens) != digest)
        throw std::runtime_error("prefix cache segment tokens do not match their digest");
    segment->payload = allocate(header.payload_bytes);
    file.read(segment->payload.data.get(), header.payload_bytes);
    if (prefix_cache_checksum(segment->payload.span()) != header.payload_checksum)
        throw std::runtime_error("prefix cache segment payload checksum differs");
    std::lock_guard lock(mutex_);
    stats_.disk_read_bytes += sizeof(header) + header.payload_bytes;
    return segment;
}

std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::load_from_disk(const ModelContentIdentity& identity,
    const ModelContentIdentity& digest, std::span<const std::uint32_t> tokens, std::uint32_t block_rows) {
    const auto start = std::chrono::steady_clock::now();
    File file(checkpoint_path(policy_.disk_directory, digest), "rb");
    if (!file.open()) return nullptr;
    CheckpointHeader header;
    file.read(&header, sizeof(header));
    if (header.magic != checkpoint_magic || get(header.identity) != identity || get(header.digest) != digest ||
        header.block_rows != block_rows || header.position != tokens.size() ||
        header.segment_count * block_rows != header.position)
        throw std::runtime_error("prefix cache checkpoint header does not match");
    std::vector<std::uint8_t> segment_digests(static_cast<std::size_t>(header.segment_count * 32));
    file.read(segment_digests.data(), segment_digests.size());
    auto checkpoint = std::make_shared<PrefixCacheCheckpoint>();
    checkpoint->identity = identity;
    checkpoint->digest = digest;
    checkpoint->position = header.position;
    checkpoint->block_rows = block_rows;
    checkpoint->metadata.resize(static_cast<std::size_t>(header.metadata_bytes));
    file.read(checkpoint->metadata.data(), header.metadata_bytes);
    if (prefix_cache_checksum(checkpoint->metadata) != header.metadata_checksum)
        throw std::runtime_error("prefix cache metadata checksum differs");
    checkpoint->fixed = allocate(header.fixed_bytes);
    file.read(checkpoint->fixed.data.get(), header.fixed_bytes);
    if (prefix_cache_checksum(checkpoint->fixed.span()) != header.fixed_checksum)
        throw std::runtime_error("prefix cache fixed-state checksum differs");
    auto previous = identity;
    for (std::uint64_t k = 0; k < header.segment_count; ++k) {
        ModelContentIdentity segment_digest;
        std::memcpy(segment_digest.bytes.data(), segment_digests.data() + k * 32, 32);
        auto segment = load_segment(identity, segment_digest, block_rows);
        // The chain must be exactly the requested tokens.
        if (segment->block_index != k || segment->parent != previous ||
            !std::equal(segment->tokens.begin(), segment->tokens.end(),
                tokens.begin() + static_cast<std::ptrdiff_t>(k * block_rows)))
            throw std::runtime_error("prefix cache checkpoint chain does not match the requested tokens");
        previous = segment_digest;
        checkpoint->segments.push_back(std::move(segment));
    }
    if (previous != digest) throw std::runtime_error("prefix cache checkpoint chain does not end at its digest");
    std::lock_guard lock(mutex_);
    ++stats_.disk_reads;
    stats_.disk_read_bytes += sizeof(header) + header.fixed_bytes + header.metadata_bytes;
    stats_.disk_read_ms += elapsed_ms(start);
    return checkpoint;
}

void PrefixStateCache::write_to_disk(const PrefixCacheCheckpoint& checkpoint) {
    const auto start = std::chrono::steady_clock::now();
    const auto& dir = policy_.disk_directory;
    std::uint64_t written = 0;
    std::error_code error;
    for (const auto& segment : checkpoint.segments) {
        const auto path = segment_path(dir, segment->digest);
        if (std::filesystem::exists(path, error)) continue;
        SegmentHeader header;
        put(header.identity, segment->identity);
        put(header.digest, segment->digest);
        put(header.parent, segment->parent);
        header.block_index = segment->block_index;
        header.block_rows = segment->block_rows;
        header.payload_bytes = segment->payload.bytes;
        header.payload_checksum = prefix_cache_checksum(segment->payload.span());
        const auto temporary = temporary_path(path);
        {
            File file(temporary, "wb");
            if (!file.open()) throw std::runtime_error("prefix cache segment file cannot be created");
            file.write(&header, sizeof(header));
            file.write(segment->tokens.data(), segment->tokens.size() * sizeof(std::uint32_t));
            file.write(segment->payload.data.get(), segment->payload.bytes);
            file.close();
        }
        std::filesystem::rename(temporary, path);
        written += sizeof(header) + segment->payload.bytes;
    }
    const auto path = checkpoint_path(dir, checkpoint.digest);
    if (!std::filesystem::exists(path, error)) {
        CheckpointHeader header;
        put(header.identity, checkpoint.identity);
        put(header.digest, checkpoint.digest);
        header.position = checkpoint.position;
        header.block_rows = checkpoint.block_rows;
        header.segment_count = checkpoint.segments.size();
        header.fixed_bytes = checkpoint.fixed.bytes;
        header.metadata_bytes = checkpoint.metadata.size();
        header.fixed_checksum = prefix_cache_checksum(checkpoint.fixed.span());
        header.metadata_checksum = prefix_cache_checksum(checkpoint.metadata);
        const auto temporary = temporary_path(path);
        {
            File file(temporary, "wb");
            if (!file.open()) throw std::runtime_error("prefix cache checkpoint file cannot be created");
            file.write(&header, sizeof(header));
            for (const auto& segment : checkpoint.segments) file.write(segment->digest.bytes.data(), 32);
            file.write(checkpoint.metadata.data(), checkpoint.metadata.size());
            file.write(checkpoint.fixed.data.get(), checkpoint.fixed.bytes);
            file.close();
        }
        std::filesystem::rename(temporary, path);
        written += sizeof(header) + checkpoint.fixed.bytes + checkpoint.metadata.size();
    }
    std::lock_guard lock(mutex_);
    ++stats_.disk_writes;
    stats_.disk_write_bytes += written;
    stats_.disk_write_ms += elapsed_ms(start);
}

void PrefixStateCache::enforce_disk_budget() {
    if (!policy_.disk_byte_budget) return;
    namespace fs = std::filesystem;
    struct Item { fs::path path; fs::file_time_type time; std::uint64_t bytes; };
    std::vector<Item> checkpoints;
    std::uint64_t total = 0;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        const auto bytes = entry.file_size(error);
        total += bytes;
        if (entry.path().extension() == ".tmpc") checkpoints.push_back({entry.path(), entry.last_write_time(error), bytes});
    }
    if (total <= policy_.disk_byte_budget) return;
    std::sort(checkpoints.begin(), checkpoints.end(), [](const Item& a, const Item& b) { return a.time < b.time; });
    std::size_t removed = 0;
    // Approximate: a removed checkpoint frees its own file now and its unshared segments below.
    while (removed + 1 < checkpoints.size() && total > policy_.disk_byte_budget) {
        fs::remove(checkpoints[removed].path, error);
        total -= std::min(total, checkpoints[removed].bytes);
        ++removed;
        std::lock_guard lock(mutex_);
        ++stats_.disk_evictions;
    }
    if (!removed) return;
    // Delete segment files no remaining checkpoint references.
    std::unordered_set<std::string> referenced;
    for (std::size_t i = removed; i < checkpoints.size(); ++i) {
        File file(checkpoints[i].path, "rb");
        if (!file.open()) continue;
        CheckpointHeader header;
        try {
            file.read(&header, sizeof(header));
            if (header.magic != checkpoint_magic) continue;
            std::vector<std::uint8_t> digests(static_cast<std::size_t>(header.segment_count * 32));
            file.read(digests.data(), digests.size());
            for (std::uint64_t k = 0; k < header.segment_count; ++k) {
                ModelContentIdentity digest;
                std::memcpy(digest.bytes.data(), digests.data() + k * 32, 32);
                referenced.insert(segment_path(policy_.disk_directory, digest).filename().string());
            }
        } catch (const std::exception&) {
        }
    }
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error))
        if (entry.path().extension() == ".tmps" && !referenced.count(entry.path().filename().string()))
            fs::remove(entry.path(), error);
}

void PrefixStateCache::sweep_disk(std::chrono::seconds min_age) {
    namespace fs = std::filesystem;
    std::error_code error;
    const auto now = fs::file_time_type::clock::now();
    const auto old = [&](const fs::directory_entry& entry) {
        const auto time = entry.last_write_time(error);
        return !error && now - time > min_age;
    };
    std::unordered_set<std::string> referenced;
    std::vector<fs::path> segments;
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error)) {
        const auto name = entry.path().filename().string();
        if (name.find(".tmp.") != std::string::npos) {
            if (old(entry)) fs::remove(entry.path(), error);
            continue;
        }
        if (entry.path().extension() == ".tmps") { segments.push_back(entry.path()); continue; }
        if (entry.path().extension() != ".tmpc") continue;
        File file(entry.path(), "rb");
        if (!file.open()) continue;
        try {
            CheckpointHeader header;
            file.read(&header, sizeof(header));
            if (header.magic != checkpoint_magic || header.segment_count > (1ULL << 24)) continue;
            std::vector<std::uint8_t> digests(static_cast<std::size_t>(header.segment_count * 32));
            file.read(digests.data(), digests.size());
            for (std::uint64_t k = 0; k < header.segment_count; ++k) {
                ModelContentIdentity digest;
                std::memcpy(digest.bytes.data(), digests.data() + k * 32, 32);
                referenced.insert(segment_path(policy_.disk_directory, digest).filename().string());
            }
        } catch (const std::exception&) {
        }
    }
    for (const auto& path : segments)
        if (!referenced.count(path.filename().string()) && old(fs::directory_entry(path, error))) fs::remove(path, error);
}

void PrefixStateCache::writer_loop() {
    for (;;) {
        std::shared_ptr<const PrefixCacheCheckpoint> next;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                writer_busy_ = false;
                idle_cv_.notify_all();
                return;
            }
            next = std::move(queue_.front());
            queue_.pop_front();
            writer_busy_ = true;
        }
        try {
            write_to_disk(*next);
            enforce_disk_budget();
        } catch (const std::exception&) {
            std::lock_guard lock(mutex_);
            ++stats_.disk_write_failures;
        }
        {
            std::lock_guard lock(queue_mutex_);
            writer_busy_ = false;
        }
        idle_cv_.notify_all();
    }
}

void PrefixStateCache::flush() {
    if (!writer_.joinable()) return;
    std::unique_lock lock(queue_mutex_);
    idle_cv_.wait(lock, [&] { return queue_.empty() && !writer_busy_; });
}

PrefixStateCacheStats PrefixStateCache::stats() {
    std::lock_guard lock(mutex_);
    auto out = stats_;
    out.host_bytes = host_bytes_locked();
    out.host_checkpoints = entries_.size();
    out.host_segments = segments_.size();
    return out;
}

}  // namespace strix
