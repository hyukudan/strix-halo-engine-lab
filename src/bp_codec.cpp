#include "strix/bp_codec.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace strix::bp_codec {
namespace {

constexpr std::uint32_t kMagic = 0x31565042U;  // "BPV1"
constexpr unsigned kMaxLen = 11U;
constexpr std::size_t kTable = std::size_t{1} << kMaxLen;
constexpr std::size_t kFrameHeader = 16U, kDirEntry = 16U, kHuffHeader = 128U + 16U;
enum : std::uint8_t { kRaw = 0, kConstant = 1, kHuffman = 2 };

inline std::uint64_t load64(const std::uint8_t* p) noexcept {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;  // little-endian hosts only (x86-64)
}
inline std::uint32_t load32(const std::uint8_t* p) noexcept {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline void store32(std::uint8_t* p, std::uint32_t v) noexcept { std::memcpy(p, &v, 4); }
inline void store64(std::uint8_t* p, std::uint64_t v) noexcept { std::memcpy(p, &v, 8); }
inline std::uint64_t rotl(std::uint64_t x, unsigned r) noexcept { return (x << r) | (x >> (64U - r)); }

[[noreturn]] void bad(const char* what) { throw std::invalid_argument(std::string("bp-v1 frame: ") + what); }

// ---- Huffman code construction ----

// Code lengths (<= kMaxLen) of a Huffman code for counts (at least two nonzero). Lengths above the limit: halve the
// counts (keeping them nonzero) and rebuild, which always terminates (equal counts give length 8).
void huffman_lengths(const std::array<std::uint64_t, 256>& counts, std::array<std::uint8_t, 256>& lens) {
    std::array<std::uint64_t, 256> c = counts;
    for (;;) {
        std::array<std::uint16_t, 256> order{};
        unsigned leaves = 0;
        for (unsigned s = 0; s < 256U; ++s) if (c[s]) order[leaves++] = static_cast<std::uint16_t>(s);
        std::sort(order.begin(), order.begin() + leaves, [&](std::uint16_t a, std::uint16_t b) { return c[a] != c[b] ? c[a] < c[b] : a < b; });
        // Two-queue construction: nodes 0..leaves-1 are leaves (sorted), internal nodes follow in creation order.
        std::array<std::uint64_t, 512> weight{};
        std::array<std::uint16_t, 512> parent{};
        for (unsigned i = 0; i < leaves; ++i) weight[i] = c[order[i]];
        unsigned leaf = 0, inner = leaves, made = leaves;
        const auto pick = [&]() -> unsigned {
            if (leaf < leaves && (inner >= made || weight[leaf] <= weight[inner])) return leaf++;
            return inner++;
        };
        while (made < 2U * leaves - 1U) {
            const unsigned a = pick(), b = pick();
            weight[made] = weight[a] + weight[b];
            parent[a] = parent[b] = static_cast<std::uint16_t>(made);
            ++made;
        }
        std::array<std::uint8_t, 512> depth{};
        unsigned longest = 0;
        for (unsigned n = made - 1U; n-- > 0;) {
            depth[n] = static_cast<std::uint8_t>(std::min<unsigned>(depth[parent[n]] + 1U, 255U));
            if (n < leaves) longest = std::max<unsigned>(longest, depth[n]);
        }
        if (longest <= kMaxLen) {
            lens.fill(0);
            for (unsigned i = 0; i < leaves; ++i) lens[order[i]] = depth[i];
            return;
        }
        for (auto& v : c) if (v) v = (v + 1U) >> 1U;
    }
}

// Canonical codes, bit-reversed for LSB-first emission.
void canonical_codes(const std::array<std::uint8_t, 256>& lens, std::array<std::uint16_t, 256>& codes) {
    std::array<unsigned, kMaxLen + 2U> count{}, next{};
    for (const auto l : lens) ++count[l];
    count[0] = 0;
    unsigned code = 0;
    for (unsigned l = 1; l <= kMaxLen; ++l) { code = (code + count[l - 1U]) << 1U; next[l] = code; }
    for (unsigned s = 0; s < 256U; ++s) {
        const unsigned l = lens[s];
        if (!l) { codes[s] = 0; continue; }
        unsigned v = next[l]++, r = 0;
        for (unsigned b = 0; b < l; ++b) { r = (r << 1U) | (v & 1U); v >>= 1U; }
        codes[s] = static_cast<std::uint16_t>(r);
    }
}

// ---- planning (encoder pass 1) ----

struct PlanePlan {
    std::uint8_t mode = kRaw, value = 0;
    std::array<std::uint8_t, 256> lens{};
    std::array<std::uint32_t, 4> stream_bytes{};
    std::uint64_t bytes = 0;  // payload bytes of the plane
};
struct BlockPlan {
    const std::uint8_t* data = nullptr;
    std::size_t plain = 0;
    std::uint32_t width = 1;
    std::array<PlanePlan, 4> planes{};
    std::uint64_t encoded = 0, checksum = 0, offset = 0;
};

inline std::size_t stream_begin(std::size_t n, unsigned s) { return std::min(n, ((n + 3U) / 4U) * s); }

void plan_block(BlockPlan& b, double min_gain) {
    const std::size_t w = b.width, n = b.plain / w;
    b.encoded = 1U + w + b.plain % w;
    for (std::size_t j = 0; j < w; ++j) {
        PlanePlan& p = b.planes[j];
        std::array<std::array<std::uint64_t, 256>, 4> h{};
        for (unsigned s = 0; s < 4U; ++s) {
            auto& hs = h[s];
            const std::uint8_t* in = b.data + j;
            for (std::size_t i = stream_begin(n, s), e = stream_begin(n, s + 1U); i < e; ++i) ++hs[in[i * w]];
        }
        std::array<std::uint64_t, 256> total{};
        unsigned distinct = 0;
        for (unsigned v = 0; v < 256U; ++v) {
            total[v] = h[0][v] + h[1][v] + h[2][v] + h[3][v];
            if (total[v]) { ++distinct; p.value = static_cast<std::uint8_t>(v); }
        }
        if (n == 0 || distinct > 1U) p.mode = kRaw, p.bytes = n;
        if (n > 0 && distinct == 1U) p.mode = kConstant, p.bytes = 1;
        if (distinct > 1U && n > kHuffHeader) {
            huffman_lengths(total, p.lens);
            std::uint64_t sum = kHuffHeader;
            for (unsigned s = 0; s < 4U; ++s) {
                std::uint64_t bits = 0;
                for (unsigned v = 0; v < 256U; ++v) bits += h[s][v] * p.lens[v];
                p.stream_bytes[s] = static_cast<std::uint32_t>((bits + 7U) / 8U);
                sum += p.stream_bytes[s];
            }
            if (static_cast<double>(sum) < static_cast<double>(n) * (1.0 - min_gain)) p.mode = kHuffman, p.bytes = sum;
        }
        b.encoded += p.bytes;
    }
}

// ---- encoder pass 2 ----

// Writes the codes of in[i * stride], i < count, into exactly [out, end).
void encode_stream(const std::uint8_t* in, std::size_t stride, std::size_t count, const std::array<std::uint16_t, 256>& codes,
    const std::array<std::uint8_t, 256>& lens, std::uint8_t* out, std::uint8_t* end) {
    std::uint64_t acc = 0;
    unsigned bits = 0;
    std::uint8_t* p = out;
    std::size_t i = 0;
    for (; i + 4U <= count && end - p >= 8; i += 4U) {
        for (unsigned k = 0; k < 4U; ++k) {
            const std::uint8_t s = in[(i + k) * stride];
            acc |= static_cast<std::uint64_t>(codes[s]) << bits;
            bits += lens[s];
        }
        std::memcpy(p, &acc, 8);
        p += bits >> 3U;
        acc = (bits & ~7U) == 64U ? 0 : acc >> (bits & ~7U);
        bits &= 7U;
    }
    for (; i < count; ++i) {
        const std::uint8_t s = in[i * stride];
        acc |= static_cast<std::uint64_t>(codes[s]) << bits;
        bits += lens[s];
        while (bits >= 8U) {
            if (p >= end) throw std::logic_error("bp-v1 encoder: stream exceeds its planned size");
            *p++ = static_cast<std::uint8_t>(acc);
            acc >>= 8U;
            bits -= 8U;
        }
    }
    if (bits) {
        if (p >= end) throw std::logic_error("bp-v1 encoder: stream exceeds its planned size");
        *p++ = static_cast<std::uint8_t>(acc);
    }
    if (p != end) throw std::logic_error("bp-v1 encoder: stream differs from its planned size");
}

void write_block(const BlockPlan& b, std::uint8_t* out) {
    const std::size_t w = b.width, n = b.plain / w;
    std::uint8_t* p = out;
    *p++ = static_cast<std::uint8_t>(w);
    for (std::size_t j = 0; j < w; ++j) *p++ = b.planes[j].mode;
    for (std::size_t j = 0; j < w; ++j) {
        const PlanePlan& pl = b.planes[j];
        const std::uint8_t* in = b.data + j;
        if (pl.mode == kConstant) {
            *p++ = pl.value;
        } else if (pl.mode == kRaw) {
            if (w == 1) std::memcpy(p, in, n);
            else for (std::size_t i = 0; i < n; ++i) p[i] = in[i * w];
            p += n;
        } else {
            for (unsigned k = 0; k < 128U; ++k) p[k] = static_cast<std::uint8_t>(pl.lens[2U * k] | (pl.lens[2U * k + 1U] << 4U));
            p += 128;
            for (unsigned s = 0; s < 4U; ++s) store32(p + 4U * s, pl.stream_bytes[s]);
            p += 16;
            std::array<std::uint16_t, 256> codes{};
            canonical_codes(pl.lens, codes);
            for (unsigned s = 0; s < 4U; ++s) {
                const std::size_t first = stream_begin(n, s), last = stream_begin(n, s + 1U);
                encode_stream(in + first * w, w, last - first, codes, pl.lens, p, p + pl.stream_bytes[s]);
                p += pl.stream_bytes[s];
            }
        }
    }
    const std::size_t tail = b.plain % w;
    std::memcpy(p, b.data + n * w, tail);
    p += tail;
    if (static_cast<std::uint64_t>(p - out) != b.encoded) throw std::logic_error("bp-v1 encoder: block differs from its planned size");
}

// ---- decoder ----

struct Reader {
    const std::uint8_t* start;
    const std::uint8_t* p;
    std::uint64_t acc = 0;
    unsigned bits = 0;
};

inline void refill_fast(Reader& r) noexcept {
    r.acc |= load64(r.p) << r.bits;
    r.p += (63U - r.bits) >> 3U;
    r.bits |= 56U;
}
inline void refill_safe(Reader& r, const std::uint8_t* limit) noexcept {
    while (r.bits <= 56U) {
        const std::uint64_t byte = r.p < limit ? *r.p : 0U;
        r.acc |= byte << r.bits;
        ++r.p;
        r.bits += 8U;
    }
}
inline std::uint8_t decode_one(Reader& r, const std::uint16_t* table) noexcept {
    const std::uint16_t e = table[r.acc & (kTable - 1U)];
    const unsigned l = e >> 8U;
    r.acc >>= l;
    r.bits -= l;
    return static_cast<std::uint8_t>(e);
}

// Decodes the four streams of one plane into out[i * stride].
void decode_plane(const std::uint8_t* streams, const std::array<std::uint32_t, 4>& sizes, const std::uint16_t* table,
    std::uint8_t* out, std::size_t stride, std::size_t n, const std::uint8_t* limit) {
    std::array<Reader, 4> r{};
    std::array<std::uint8_t*, 4> dst{};
    std::array<std::size_t, 4> left{};
    const std::uint8_t* at = streams;
    for (unsigned s = 0; s < 4U; ++s) {
        r[s].start = r[s].p = at;
        at += sizes[s];
        dst[s] = out + stream_begin(n, s) * stride;
        left[s] = stream_begin(n, s + 1U) - stream_begin(n, s);
    }
    // Interleaved: 5 symbols of <= 11 bits per refill of >= 56 bits.
    const std::uint8_t* const fast_limit = limit - 8;
    while (left[0] >= 5U && left[1] >= 5U && left[2] >= 5U && left[3] >= 5U && r[0].p <= fast_limit && r[1].p <= fast_limit &&
           r[2].p <= fast_limit && r[3].p <= fast_limit) {
        refill_fast(r[0]); refill_fast(r[1]); refill_fast(r[2]); refill_fast(r[3]);
        for (unsigned k = 0; k < 5U; ++k) {
            dst[0][k * stride] = decode_one(r[0], table);
            dst[1][k * stride] = decode_one(r[1], table);
            dst[2][k * stride] = decode_one(r[2], table);
            dst[3][k * stride] = decode_one(r[3], table);
        }
        for (unsigned s = 0; s < 4U; ++s) { dst[s] += 5U * stride; left[s] -= 5U; }
    }
    for (unsigned s = 0; s < 4U; ++s) {
        Reader& x = r[s];
        while (left[s] >= 5U && x.p <= fast_limit) {
            refill_fast(x);
            for (unsigned k = 0; k < 5U; ++k) dst[s][k * stride] = decode_one(x, table);
            dst[s] += 5U * stride;
            left[s] -= 5U;
        }
        while (left[s]) {
            if (x.bits < kMaxLen) refill_safe(x, limit);
            *dst[s] = decode_one(x, table);
            dst[s] += stride;
            --left[s];
        }
        // The stream must end exactly where its size says (zero padding to the byte).
        const std::uint64_t consumed = static_cast<std::uint64_t>(x.p - x.start) * 8U - x.bits;
        if ((consumed + 7U) / 8U != sizes[s]) bad("huffman stream size differs from its coded symbols");
    }
}

void build_table(const std::uint8_t* packed, std::array<std::uint8_t, 256>& lens, std::vector<std::uint16_t>& table) {
    std::uint64_t kraft = 0;
    for (unsigned k = 0; k < 128U; ++k) {
        lens[2U * k] = static_cast<std::uint8_t>(packed[k] & 15U);
        lens[2U * k + 1U] = static_cast<std::uint8_t>(packed[k] >> 4U);
    }
    unsigned used = 0;
    for (const auto l : lens) {
        if (l > kMaxLen) bad("huffman code length above 11");
        if (l) { kraft += std::uint64_t{1} << (kMaxLen - l); ++used; }
    }
    if (used < 2U || kraft != kTable) bad("huffman code is not complete");
    std::array<std::uint16_t, 256> codes{};
    canonical_codes(lens, codes);
    table.assign(kTable, 0);
    for (unsigned s = 0; s < 256U; ++s) {
        const unsigned l = lens[s];
        if (!l) continue;
        const auto entry = static_cast<std::uint16_t>(s | (l << 8U));
        for (std::size_t k = codes[s]; k < kTable; k += std::size_t{1} << l) table[k] = entry;
    }
}

struct DirEntry {
    std::uint32_t plain = 0, encoded = 0;
    std::uint64_t checksum = 0, plain_offset = 0, encoded_offset = 0;
};

std::vector<DirEntry> read_directory(std::span<const std::uint8_t> frame, std::uint64_t& plain_total) {
    if (frame.size() < kFrameHeader) bad("shorter than its header");
    if (load32(frame.data()) != kMagic) bad("bad magic");
    const std::uint64_t count = load32(frame.data() + 4);
    plain_total = load64(frame.data() + 8);
    if (count > (frame.size() - kFrameHeader) / kDirEntry) bad("block directory exceeds the frame");
    std::vector<DirEntry> dir(static_cast<std::size_t>(count));
    std::uint64_t plain = 0, encoded = kFrameHeader + count * kDirEntry;
    for (std::size_t b = 0; b < dir.size(); ++b) {
        const std::uint8_t* e = frame.data() + kFrameHeader + b * kDirEntry;
        dir[b].plain = load32(e);
        dir[b].encoded = load32(e + 4);
        dir[b].checksum = load64(e + 8);
        if (!dir[b].plain || dir[b].encoded < 2U) bad("empty block");
        dir[b].plain_offset = plain;
        dir[b].encoded_offset = encoded;
        plain += dir[b].plain;
        encoded += dir[b].encoded;
        if (encoded > frame.size()) bad("blocks exceed the frame");
    }
    if (plain != plain_total) bad("block sizes do not add up to the plain size");
    if (encoded != frame.size()) bad("blocks do not fill the frame");
    return dir;
}

void decode_block(std::span<const std::uint8_t> frame, const DirEntry& d, std::uint8_t* out) {
    const std::uint8_t* p = frame.data() + d.encoded_offset;
    const std::uint8_t* const end = p + d.encoded;
    const std::uint8_t* const limit = frame.data() + frame.size();
    const std::size_t w = *p++;
    if (w != 1U && w != 2U && w != 4U) bad("block width is not 1, 2 or 4");
    const std::size_t n = d.plain / w, tail = d.plain % w;
    if (static_cast<std::size_t>(end - p) < w) bad("block header is truncated");
    std::array<std::uint8_t, 4> modes{};
    for (std::size_t j = 0; j < w; ++j) modes[j] = *p++;
    std::array<std::uint8_t, 256> lens{};
    std::vector<std::uint16_t> table;
    for (std::size_t j = 0; j < w; ++j) {
        std::uint8_t* dst = out + j;
        const auto avail = static_cast<std::size_t>(end - p);
        if (modes[j] == kConstant) {
            if (n == 0 || avail < 1U) bad("constant plane is truncated");
            const std::uint8_t v = *p++;
            if (w == 1) std::memset(dst, v, n);
            else for (std::size_t i = 0; i < n; ++i) dst[i * w] = v;
        } else if (modes[j] == kRaw) {
            if (avail < n) bad("raw plane is truncated");
            if (w == 1) std::memcpy(dst, p, n);
            else if (w == 2) for (std::size_t i = 0; i < n; ++i) dst[2U * i] = p[i];
            else for (std::size_t i = 0; i < n; ++i) dst[i * w] = p[i];
            p += n;
        } else if (modes[j] == kHuffman) {
            if (n == 0 || avail < kHuffHeader) bad("huffman plane header is truncated");
            build_table(p, lens, table);
            p += 128;
            std::array<std::uint32_t, 4> sizes{};
            std::uint64_t sum = 0;
            for (unsigned s = 0; s < 4U; ++s) { sizes[s] = load32(p + 4U * s); sum += sizes[s]; }
            p += 16;
            if (sum > static_cast<std::uint64_t>(end - p)) bad("huffman streams exceed the block");
            decode_plane(p, sizes, table.data(), dst, w, n, limit);
            p += sum;
        } else {
            bad("unknown plane mode");
        }
    }
    if (static_cast<std::size_t>(end - p) != tail) bad("block size differs from its planes");
    std::memcpy(out + n * w, p, tail);
    if (xxh64(out, d.plain) != d.checksum) bad("block checksum differs (decoded bytes are not the encoded ones)");
}

}  // namespace

// ---- XXH64 ----

std::uint64_t xxh64(const std::uint8_t* data, std::size_t bytes, std::uint64_t seed) noexcept {
    constexpr std::uint64_t P1 = 0x9E3779B185EBCA87ULL, P2 = 0xC2B2AE3D27D4EB4FULL, P3 = 0x165667B19E3779F9ULL,
                            P4 = 0x85EBCA77C2B2AE63ULL, P5 = 0x27D4EB2F165667C5ULL;
    const auto mix = [](std::uint64_t acc, std::uint64_t in) { acc += in * P2; acc = rotl(acc, 31); return acc * P1; };
    const auto fold = [&](std::uint64_t acc, std::uint64_t v) { acc ^= mix(0, v); return acc * P1 + P4; };
    const std::uint8_t* p = data;
    const std::uint8_t* const end = data + bytes;
    std::uint64_t h;
    if (bytes >= 32U) {
        std::uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        for (; end - p >= 32; p += 32) {
            v1 = mix(v1, load64(p)); v2 = mix(v2, load64(p + 8)); v3 = mix(v3, load64(p + 16)); v4 = mix(v4, load64(p + 24));
        }
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = fold(h, v1); h = fold(h, v2); h = fold(h, v3); h = fold(h, v4);
    } else {
        h = seed + P5;
    }
    h += static_cast<std::uint64_t>(bytes);
    for (; end - p >= 8; p += 8) { h ^= mix(0, load64(p)); h = rotl(h, 27) * P1 + P4; }
    if (end - p >= 4) { h ^= static_cast<std::uint64_t>(load32(p)) * P1; h = rotl(h, 23) * P2 + P3; p += 4; }
    for (; p < end; ++p) { h ^= static_cast<std::uint64_t>(*p) * P5; h = rotl(h, 11) * P1; }
    h ^= h >> 33U; h *= P2; h ^= h >> 29U; h *= P3; h ^= h >> 32U;
    return h;
}

// ---- pool ----

namespace {
struct Job {
    const std::function<void(std::size_t)>* fn = nullptr;
    std::size_t tasks = 0;
    std::atomic<std::size_t> next{0}, done{0};
    std::mutex m;
    std::condition_variable cv;
    std::exception_ptr error;
};
void work(Job& job) {
    for (;;) {
        const std::size_t i = job.next.fetch_add(1);
        if (i >= job.tasks) return;
        try {
            (*job.fn)(i);
        } catch (...) {
            const std::lock_guard lock(job.m);
            if (!job.error) job.error = std::current_exception();
        }
        if (job.done.fetch_add(1) + 1U == job.tasks) {
            const std::lock_guard lock(job.m);
            job.cv.notify_all();
        }
    }
}
}  // namespace

struct WorkerPool::State {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Job>> jobs;
    bool stop = false;
    std::vector<std::thread> workers;
};

WorkerPool::WorkerPool(unsigned threads) : state_(std::make_unique<State>()) {
    for (unsigned t = 0; t < threads; ++t)
        state_->workers.emplace_back([s = state_.get()] {
            for (;;) {
                std::shared_ptr<Job> job;
                {
                    std::unique_lock lock(s->m);
                    for (;;) {
                        while (!s->jobs.empty() && s->jobs.front()->next.load() >= s->jobs.front()->tasks) s->jobs.pop_front();
                        if (!s->jobs.empty() || s->stop) break;
                        s->cv.wait(lock);
                    }
                    if (s->jobs.empty()) return;
                    job = s->jobs.front();
                }
                work(*job);
            }
        });
}

WorkerPool::~WorkerPool() {
    {
        const std::lock_guard lock(state_->m);
        state_->stop = true;
    }
    state_->cv.notify_all();
    for (auto& t : state_->workers) t.join();
}

unsigned WorkerPool::threads() const noexcept { return static_cast<unsigned>(state_->workers.size()); }

void WorkerPool::run(std::size_t tasks, const std::function<void(std::size_t)>& fn) {
    if (!tasks) return;
    auto job = std::make_shared<Job>();
    job->fn = &fn;
    job->tasks = tasks;
    if (tasks > 1U && !state_->workers.empty()) {
        {
            const std::lock_guard lock(state_->m);
            state_->jobs.push_back(job);
        }
        state_->cv.notify_all();
    }
    work(*job);
    {
        std::unique_lock lock(job->m);
        job->cv.wait(lock, [&] { return job->done.load() == job->tasks; });
    }
    {
        const std::lock_guard lock(state_->m);
        const auto it = std::find(state_->jobs.begin(), state_->jobs.end(), job);
        if (it != state_->jobs.end()) state_->jobs.erase(it);
    }
    if (job->error) std::rethrow_exception(job->error);
}

// ---- encode / decode ----

std::vector<std::uint8_t> encode(std::span<const Region> regions, WorkerPool* pool, const EncodeOptions& options, EncodeStats* stats) {
    std::vector<BlockPlan> blocks;
    std::uint64_t plain = 0;
    for (const auto& r : regions) {
        const std::uint32_t w = (r.width == 2U || r.width == 4U) ? r.width : 1U;
        const std::size_t step = std::max<std::size_t>(w, std::min<std::size_t>(options.block_bytes, std::size_t{1} << 30) / w * w);
        for (std::size_t at = 0; at < r.bytes; at += step) {
            BlockPlan b;
            b.data = r.data + at;
            b.plain = std::min(step, r.bytes - at);
            b.width = w;
            blocks.push_back(b);
        }
        plain += r.bytes;
    }
    if (blocks.size() > UINT32_MAX) throw std::invalid_argument("bp-v1: too many blocks");
    const auto each = [&](const std::function<void(std::size_t)>& fn) {
        if (pool) pool->run(blocks.size(), fn);
        else for (std::size_t i = 0; i < blocks.size(); ++i) fn(i);
    };
    each([&](std::size_t i) {
        plan_block(blocks[i], options.huffman_min_gain);
        blocks[i].checksum = xxh64(blocks[i].data, blocks[i].plain);
    });
    std::uint64_t at = kFrameHeader + blocks.size() * kDirEntry;
    for (auto& b : blocks) { b.offset = at; at += b.encoded; }
    std::vector<std::uint8_t> frame(static_cast<std::size_t>(at));
    store32(frame.data(), kMagic);
    store32(frame.data() + 4, static_cast<std::uint32_t>(blocks.size()));
    store64(frame.data() + 8, plain);
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        std::uint8_t* e = frame.data() + kFrameHeader + i * kDirEntry;
        store32(e, static_cast<std::uint32_t>(blocks[i].plain));
        store32(e + 4, static_cast<std::uint32_t>(blocks[i].encoded));
        store64(e + 8, blocks[i].checksum);
    }
    each([&](std::size_t i) { write_block(blocks[i], frame.data() + blocks[i].offset); });
    if (stats) {
        *stats = {};
        stats->plain_bytes = plain;
        stats->encoded_bytes = frame.size();
        stats->blocks = blocks.size();
        for (const auto& b : blocks)
            for (std::size_t j = 0; j < b.width; ++j) {
                const auto m = b.planes[j].mode;
                ++(m == kRaw ? stats->raw_planes : m == kConstant ? stats->constant_planes : stats->huffman_planes);
            }
    }
    return frame;
}

std::uint64_t decoded_size(std::span<const std::uint8_t> frame) {
    std::uint64_t plain = 0;
    (void)read_directory(frame, plain);
    return plain;
}

void decode(std::span<const std::uint8_t> frame, std::span<std::uint8_t> out, WorkerPool* pool) {
    std::uint64_t plain = 0;
    const auto dir = read_directory(frame, plain);
    if (plain != out.size()) throw std::invalid_argument("bp-v1: output size differs from the frame's plain size");
    const std::function<void(std::size_t)> fn = [&](std::size_t i) { decode_block(frame, dir[i], out.data() + dir[i].plain_offset); };
    if (pool) pool->run(dir.size(), fn);
    else for (std::size_t i = 0; i < dir.size(); ++i) fn(i);
}

}  // namespace strix::bp_codec
