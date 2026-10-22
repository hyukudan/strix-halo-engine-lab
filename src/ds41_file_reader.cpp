#include "strix/ds41_file_reader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::ds41 {
namespace {

struct AlignedFree {
    void operator()(std::uint8_t* p) const noexcept {
#ifdef _WIN32
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
};
using AlignedPage = std::unique_ptr<std::uint8_t, AlignedFree>;
AlignedPage aligned_alloc_bytes(std::size_t bytes) {
#ifdef _WIN32
    auto* p = static_cast<std::uint8_t*>(_aligned_malloc(bytes, FileReader::alignment));
#else
    void* raw = nullptr;
    if (posix_memalign(&raw, FileReader::alignment, bytes) != 0) raw = nullptr;
    auto* p = static_cast<std::uint8_t*>(raw);
#endif
    if (p == nullptr) throw std::bad_alloc();
    return AlignedPage(p);
}

#ifdef _WIN32
// One positional read on an overlapped handle, waited for: the bytes read (0 at the end of the file).
std::size_t read_at(HANDLE h, std::uint64_t offset, void* dst, std::size_t bytes) {
    thread_local HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    ov.hEvent = event;
    DWORD got = 0;
    if (!ReadFile(h, dst, static_cast<DWORD>(bytes), nullptr, &ov)) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) return 0;
        if (e != ERROR_IO_PENDING) throw std::runtime_error("ReadFile failed, error " + std::to_string(e));
    }
    if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) return 0;
        throw std::runtime_error("ReadFile failed, error " + std::to_string(e));
    }
    return got;
}
#endif

}  // namespace

FileReader::FileReader(const std::filesystem::path& path, bool direct, unsigned threads) : path_(path), direct_(direct) {
#ifdef _WIN32
    // overlapped handle: synchronous handles serialize every read on the file object, so the pool's threads would
    // run at queue depth 1; each read waits on its own event instead
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_SEQUENTIAL_SCAN);
    HANDLE h = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, flags, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path.string());
    handle_ = h;
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(h, &sz)) throw std::runtime_error("cannot size " + path.string());
    size_ = static_cast<std::uint64_t>(sz.QuadPart);
#else
    fd_ = ::open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd_ < 0 && direct) {  // a filesystem without O_DIRECT (9p, tmpfs): buffered reads
        fd_ = ::open(path.c_str(), O_RDONLY);
        direct_ = false;
    }
    if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
    struct stat st {};
    if (::fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat " + path.string());
    size_ = static_cast<std::uint64_t>(st.st_size);
#endif
    for (unsigned i = 0; i < std::max(1U, threads); ++i) threads_.emplace_back([this] { worker(); });
}

FileReader::~FileReader() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
    if (cache_mem_ != nullptr) AlignedPage free_cache(cache_mem_);
#ifdef _WIN32
    if (handle_ != nullptr) CloseHandle(static_cast<HANDLE>(handle_));
#else
    if (fd_ >= 0) ::close(fd_);
#endif
}

FileReader::Stats FileReader::stats() const { return stats_; }

void FileReader::pread_exact(std::uint64_t offset, std::size_t bytes, void* dst) {
    auto* out = static_cast<std::uint8_t*>(dst);
    while (bytes > 0) {
        const std::size_t want = std::min<std::size_t>(bytes, std::size_t{1} << 30);
#ifdef _WIN32
        const std::size_t n = read_at(static_cast<HANDLE>(handle_), offset, out, want);
        if (n == 0) throw std::runtime_error("read failed at " + std::to_string(offset) + " of " + path_.string());
#else
        const ssize_t got = ::pread(fd_, out, want, static_cast<off_t>(offset));
        if (got <= 0) throw std::runtime_error("read failed at " + std::to_string(offset) + " of " + path_.string());
        const auto n = static_cast<std::size_t>(got);
#endif
        out += n;
        offset += n;
        bytes -= n;
    }
}

// Aligned pages [offset, offset + bytes) of which only the part before the end of the file must arrive (an unbuffered
// read of a whole page at the end of the file returns the bytes that exist).
void FileReader::pread_page(std::uint64_t offset, std::size_t bytes, std::uint8_t* dst) {
    const std::uint64_t want = std::min<std::uint64_t>(bytes, size_ - offset);
    std::uint64_t done = 0;
    while (done < want) {
#ifdef _WIN32
        const std::uint64_t at = offset + done;
        const std::size_t got = read_at(static_cast<HANDLE>(handle_), at, dst + done, static_cast<std::size_t>(bytes - done));
        if (got == 0) throw std::runtime_error("page read failed at " + std::to_string(at) + " of " + path_.string());
        done += got;
#else
        const ssize_t got = ::pread(fd_, dst + done, bytes - done, static_cast<off_t>(offset + done));
        if (got <= 0) throw std::runtime_error("page read failed at " + std::to_string(offset + done) + " of " + path_.string());
        done += static_cast<std::uint64_t>(got);
#endif
    }
}

void FileReader::worker() {
    AlignedPage page = aligned_alloc_bytes(2 * alignment);
    std::uint64_t seen = 0;
    for (;;) {
        std::size_t job = 0;
        const std::function<void(std::size_t, std::uint8_t*)>* body = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || (generation_ != seen && next_job_ < job_count_); });
            if (stop_) return;
            job = next_job_++;
            body = job_;
            if (next_job_ >= job_count_) seen = generation_;
        }
        try {
            (*body)(job, page.get());
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (error_.empty()) error_ = e.what();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (++finished_ == job_count_) done_.notify_all();
        }
    }
}

void FileReader::run_parallel(std::size_t jobs, const std::function<void(std::size_t, std::uint8_t*)>& body) {
    if (jobs == 0) return;
    std::unique_lock<std::mutex> lock(mutex_);
    job_ = &body;
    job_count_ = jobs;
    next_job_ = 0;
    finished_ = 0;
    error_.clear();
    ++generation_;
    wake_.notify_all();
    done_.wait(lock, [&] { return finished_ == job_count_; });
    job_ = nullptr;
    if (!error_.empty()) throw std::runtime_error(error_);
}

void FileReader::read_range(std::uint64_t offset, std::size_t bytes, void* dst, std::size_t piece) {
    if (bytes == 0) return;
    if (offset + bytes > size_) throw std::runtime_error("read beyond the end of " + path_.string());
    const auto t0 = std::chrono::steady_clock::now();
    auto* out = static_cast<std::uint8_t*>(dst);
    piece = std::max<std::size_t>(alignment, piece / alignment * alignment);
    // aligned middle [a0, a1) read straight into dst when dst is aligned the same way; the rest through pages
    const bool aligned_dst = (reinterpret_cast<std::uintptr_t>(out) % alignment) == (offset % alignment);
    std::uint64_t a0 = offset, a1 = offset + bytes;
    if (direct_) {
        a0 = (offset + alignment - 1U) / alignment * alignment;
        a1 = (offset + bytes) / alignment * alignment;
        if (!aligned_dst || a1 <= a0) a0 = a1 = offset;  // everything through the bounce pages
    }
    const std::size_t middle = static_cast<std::size_t>(a1 - a0), pieces = (middle + piece - 1U) / piece;
    // unaligned bytes: [offset, a0) and [a1, offset + bytes), page by page
    std::vector<std::uint64_t> pages;
    if (direct_) {
        for (std::uint64_t p = offset / alignment * alignment; p < a0; p += alignment) pages.push_back(p);
        for (std::uint64_t p = a1 / alignment * alignment; p < offset + bytes; p += alignment)
            if (pages.empty() || pages.back() != p) pages.push_back(p);
    }
    const std::function<void(std::size_t, std::uint8_t*)> body = [&](std::size_t job, std::uint8_t* page) {
        if (job < pieces) {
            const std::uint64_t at = a0 + std::uint64_t{job} * piece;
            const std::size_t n = std::min<std::size_t>(piece, static_cast<std::size_t>(a1 - at));
            pread_exact(at, n, out + (at - offset));
            return;
        }
        if (!direct_) return;
        const std::uint64_t p = pages[job - pieces];
        pread_page(p, alignment, page);
        const std::uint64_t lo = std::max(p, offset), hi = std::min(p + alignment, offset + bytes);
        for (std::uint64_t b = lo; b < hi; ++b)
            if (b < a0 || b >= a1) out[b - offset] = page[b - p];
    };
    if (!direct_) {
        // buffered: plain pieces over the whole range
        const std::size_t all = (bytes + piece - 1U) / piece;
        const std::function<void(std::size_t, std::uint8_t*)> plain = [&](std::size_t job, std::uint8_t*) {
            const std::uint64_t at = offset + std::uint64_t{job} * piece;
            pread_exact(at, std::min<std::size_t>(piece, static_cast<std::size_t>(offset + bytes - at)), out + (at - offset));
        };
        run_parallel(all, plain);
    } else {
        run_parallel(pieces + pages.size(), body);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.bytes += bytes;
    stats_.reads += pieces + pages.size();
    stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct FileReader::PageIndex {
    std::unordered_map<std::uint64_t, std::uint32_t> slot;  // page offset -> cache slot
};

namespace {
constexpr std::uint32_t kNil = 0xffffffffU;
}  // namespace

void FileReader::set_page_cache(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_mem_ != nullptr) AlignedPage old(cache_mem_);
    cache_mem_ = nullptr;
    cache_slots_ = bytes / alignment;
    index_ = std::make_unique<PageIndex>();
    slot_page_.assign(cache_slots_, ~std::uint64_t{0});
    lru_prev_.assign(cache_slots_, kNil);
    lru_next_.assign(cache_slots_, kNil);
    slot_stamp_.assign(cache_slots_, 0);
    lru_head_ = lru_tail_ = kNil;
    if (cache_slots_ == 0) return;
    cache_mem_ = aligned_alloc_bytes(cache_slots_ * alignment).release();
    // list in slot order: head 0 (most recent) ... tail (least recent)
    for (std::size_t k = 0; k < cache_slots_; ++k) {
        lru_prev_[k] = k == 0 ? kNil : static_cast<std::uint32_t>(k - 1);
        lru_next_[k] = k + 1 == cache_slots_ ? kNil : static_cast<std::uint32_t>(k + 1);
    }
    lru_head_ = 0;
    lru_tail_ = static_cast<std::uint32_t>(cache_slots_ - 1);
}

void FileReader::read_records(std::span<const std::uint64_t> offsets, std::size_t record, std::uint8_t* out, std::size_t per_job) {
    if (offsets.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    if (record > alignment) throw std::invalid_argument("read_records: records larger than a page");
    // every distinct record once (repeated n-grams read the same rows), in file order, then copied to its requests
    std::vector<std::uint32_t> order(offsets.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::uint32_t>(i);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return offsets[a] < offsets[b]; });
    std::vector<std::uint32_t> firsts;  // positions in `order` where a new offset starts
    for (std::size_t i = 0; i < order.size(); ++i)
        if (i == 0 || offsets[order[i]] != offsets[order[i - 1]]) firsts.push_back(static_cast<std::uint32_t>(i));
    firsts.push_back(static_cast<std::uint32_t>(order.size()));
    const std::size_t unique = firsts.size() - 1U;
    for (std::size_t u = 0; u < unique; ++u)
        if (offsets[order[firsts[u]]] + record > size_) throw std::runtime_error("record beyond the end of " + path_.string());

    per_job = std::max<std::size_t>(1, per_job);
    if (cache_slots_ == 0) {
        // no cache: jobs of per_job distinct records in file order, each read into the worker's page and copied out at
        // once (a record on the page the job read last reuses it: one read per distinct page of the job)
        std::atomic<std::uint64_t> pages_read{0};
        const std::function<void(std::size_t, std::uint8_t*)> body = [&](std::size_t job, std::uint8_t* page) {
            const std::size_t end = std::min(unique, (job + 1U) * per_job);
            std::uint64_t have = ~std::uint64_t{0};
            for (std::size_t u = job * per_job; u < end; ++u) {
                const std::uint64_t off = offsets[order[firsts[u]]];
                std::uint8_t* first = out + std::size_t{order[firsts[u]]} * record;
                const std::uint64_t p = off / alignment * alignment;
                const bool two = static_cast<std::size_t>(off + record - p) > alignment;
                if (p != have || two) {
                    if (direct_) pread_page(p, two ? 2U * alignment : alignment, page);
                    else pread_exact(p, static_cast<std::size_t>(std::min<std::uint64_t>(two ? 2U * alignment : alignment, size_ - p)), page);
                    pages_read += two ? 2U : 1U;
                    have = two ? ~std::uint64_t{0} : p;
                }
                std::memcpy(first, page + (off - p), record);
                for (std::size_t i = firsts[u] + 1U; i < firsts[u + 1U]; ++i) std::memcpy(out + std::size_t{order[i]} * record, first, record);
            }
        };
        run_parallel((unique + per_job - 1U) / per_job, body);
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.bytes += pages_read * alignment;
        stats_.reads += pages_read;
        stats_.records += unique;
        stats_.pages += pages_read;
        stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    auto unlink = [&](std::uint32_t k) {
        if (lru_prev_[k] != kNil) lru_next_[lru_prev_[k]] = lru_next_[k];
        else lru_head_ = lru_next_[k];
        if (lru_next_[k] != kNil) lru_prev_[lru_next_[k]] = lru_prev_[k];
        else lru_tail_ = lru_prev_[k];
    };
    auto touch = [&](std::uint32_t k) {  // to the head (most recent)
        if (k == lru_head_) return;
        unlink(k);
        lru_prev_[k] = kNil;
        lru_next_[k] = lru_head_;
        if (lru_head_ != kNil) lru_prev_[lru_head_] = k;
        lru_head_ = k;
        if (lru_tail_ == kNil) lru_tail_ = k;
    };

    // batches of records whose distinct pages fit kBatchPages: each page gets a place (a cache slot, or the batch's
    // scratch), the missing pages are read in parallel, then the records are copied out
    constexpr std::size_t kBatchPages = 16384;
    AlignedPage scratch = aligned_alloc_bytes(kBatchPages * alignment);
    std::uint64_t pages_total = 0, hits_total = 0;
    ++stamp_;
    std::vector<std::uint64_t> pages;    // distinct page offsets of the batch, ascending
    std::vector<std::uint8_t*> where;    // their bytes
    std::vector<std::uint32_t> to_read;  // indices into pages
    for (std::size_t u0 = 0; u0 < unique;) {
        pages.clear();
        std::size_t u1 = u0;
        for (; u1 < unique; ++u1) {
            const std::uint64_t off = offsets[order[firsts[u1]]];
            const std::uint64_t p0 = off / alignment * alignment, p1 = (off + record - 1U) / alignment * alignment;
            const std::size_t add = (pages.empty() || pages.back() != p0 ? 1U : 0U) + (p1 != p0 ? 1U : 0U);
            if (pages.size() + add > kBatchPages) break;
            if (pages.empty() || pages.back() != p0) pages.push_back(p0);
            if (p1 != p0) pages.push_back(p1);
        }
        where.assign(pages.size(), nullptr);
        to_read.clear();
        std::size_t scratch_used = 0;
        for (std::size_t k = 0; k < pages.size(); ++k) {
            if (cache_slots_ != 0) {
                const auto it = index_->slot.find(pages[k]);
                if (it != index_->slot.end()) {
                    touch(it->second);
                    slot_stamp_[it->second] = stamp_;
                    where[k] = cache_mem_ + std::size_t{it->second} * alignment;
                    ++hits_total;
                    continue;
                }
                const std::uint32_t victim = lru_tail_;
                if (slot_stamp_[victim] != stamp_) {  // not a page of this call: reuse its slot
                    if (slot_page_[victim] != ~std::uint64_t{0}) index_->slot.erase(slot_page_[victim]);
                    slot_page_[victim] = pages[k];
                    index_->slot[pages[k]] = victim;
                    slot_stamp_[victim] = stamp_;
                    touch(victim);
                    where[k] = cache_mem_ + std::size_t{victim} * alignment;
                    to_read.push_back(static_cast<std::uint32_t>(k));
                    continue;
                }
            }
            where[k] = scratch.get() + (scratch_used++) * alignment;
            to_read.push_back(static_cast<std::uint32_t>(k));
        }
        pages_total += pages.size();
        const std::function<void(std::size_t, std::uint8_t*)> read_body = [&](std::size_t job, std::uint8_t*) {
            const std::size_t end = std::min(to_read.size(), (job + 1U) * per_job);
            for (std::size_t q = job * per_job; q < end; ++q) {
                const std::uint32_t k = to_read[q];
                if (direct_) pread_page(pages[k], alignment, where[k]);
                else pread_exact(pages[k], static_cast<std::size_t>(std::min<std::uint64_t>(alignment, size_ - pages[k])), where[k]);
            }
        };
        run_parallel((to_read.size() + per_job - 1U) / per_job, read_body);
        // copy out: the batch's pages are ascending, and so are its records
        std::size_t k = 0;
        for (std::size_t u = u0; u < u1; ++u) {
            const std::uint64_t off = offsets[order[firsts[u]]];
            const std::uint64_t p0 = off / alignment * alignment;
            while (pages[k] != p0) ++k;
            std::uint8_t* first = out + std::size_t{order[firsts[u]]} * record;
            const std::size_t in0 = static_cast<std::size_t>(off - p0), n0 = std::min<std::size_t>(record, alignment - in0);
            std::memcpy(first, where[k] + in0, n0);
            if (n0 < record) std::memcpy(first + n0, where[k + 1U], record - n0);
            for (std::size_t i = firsts[u] + 1U; i < firsts[u + 1U]; ++i) std::memcpy(out + std::size_t{order[i]} * record, first, record);
        }
        u0 = u1;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.bytes += (pages_total - hits_total) * alignment;
    stats_.reads += pages_total - hits_total;
    stats_.records += unique;
    stats_.pages += pages_total;
    stats_.page_hits += hits_total;
    stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void FileReader::read_records_queued(std::span<const std::uint64_t> offsets, std::size_t record, std::uint8_t* out) {
    if (offsets.empty()) return;
#ifdef _WIN32
    // every record's page(s) as one overlapped read issued from this thread (queue depth = records, no pool wake-ups),
    // then all waited for
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t n = offsets.size();
    AlignedPage pages = aligned_alloc_bytes(n * 2U * alignment);
    std::vector<OVERLAPPED> ov(n);
    std::vector<HANDLE> events(n, nullptr);
    std::vector<std::uint64_t> base(n);
    std::vector<DWORD> want(n);
    std::string error;
    std::size_t issued = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t off = offsets[i];
        if (off + record > size_) {
            error = "record beyond the end of " + path_.string();
            break;
        }
        base[i] = direct_ ? off / alignment * alignment : off;
        const std::uint64_t span = direct_ ? ((off + record - base[i]) > alignment ? 2U * alignment : alignment) : record;
        want[i] = static_cast<DWORD>(std::min<std::uint64_t>(span, size_ - base[i]));
        events[i] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ov[i] = OVERLAPPED{};
        ov[i].Offset = static_cast<DWORD>(base[i] & 0xffffffffULL);
        ov[i].OffsetHigh = static_cast<DWORD>(base[i] >> 32U);
        ov[i].hEvent = events[i];
        ++issued;
        if (!ReadFile(static_cast<HANDLE>(handle_), pages.get() + i * 2U * alignment, direct_ ? static_cast<DWORD>(span) : want[i], nullptr, &ov[i])) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                error = "ReadFile failed, error " + std::to_string(e);
                break;
            }
        }
    }
    const double issue = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (std::size_t i = 0; i < issued; ++i) {
        DWORD got = 0;
        if (!GetOverlappedResult(static_cast<HANDLE>(handle_), &ov[i], &got, TRUE) && error.empty())
            error = "ReadFile failed, error " + std::to_string(GetLastError());
        else if (got < want[i] && error.empty())
            error = "short record read in " + path_.string();
        CloseHandle(events[i]);
    }
    if (!error.empty()) throw std::runtime_error(error);
    for (std::size_t i = 0; i < n; ++i) std::memcpy(out + i * record, pages.get() + i * 2U * alignment + (offsets[i] - base[i]), record);
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.bytes += n * record;
    stats_.reads += n;
    stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stats_.issue_seconds += issue;
#else
    read_records(offsets, record, out, 1);
#endif
}

}  // namespace strix::ds41
