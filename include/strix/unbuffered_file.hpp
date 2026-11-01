#pragma once
// Sequential session/prefix-state file I/O that leaves the page cache alone.
//
// A serving process keeps large read-mostly tables resident in the page cache. A buffered
// multi-GB state write or restore displaces those pages, and every later access pays a cold
// read. On Linux these classes move whole aligned chunks with O_DIRECT through a staging
// buffer; where O_DIRECT is unsupported (tmpfs, some network file systems) they fall back to
// buffered I/O that writes back and drops each chunk's pages as soon as it is complete.
// The bytes written and read are identical to plain buffered I/O; only page-cache use differs.
//
// STRIX_SESSION_DIRECT_IO=0 keeps the callers' original buffered paths. Other platforms always
// use the original paths (unbuffered_io_enabled() is false there). STRIX_SESSION_READAHEAD=0
// reads synchronously (no helper thread).
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>
#endif
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::state_file_io {

inline constexpr std::size_t alignment = 4096;
inline constexpr std::size_t chunk_bytes = std::size_t{8} << 20;

inline bool switch_enabled(const char* name) noexcept {
    const char* value = std::getenv(name);
    return !(value && std::string_view(value) == "0");
}
inline bool unbuffered_io_enabled() noexcept {
#if defined(__linux__)
    return switch_enabled("STRIX_SESSION_DIRECT_IO");
#else
    return false;
#endif
}

// Test-only fault injection for the unbuffered paths. max_syscall_bytes caps each write
// system call (short writes); fail_at_offset makes writes at or past that file offset fail;
// refuse_direct makes setting O_DIRECT on an open descriptor fail with EINVAL, as on a file
// system without direct I/O (tmpfs before 6.6, some FUSE).
struct FaultInjection {
    std::atomic<bool> refuse_direct{false};
    std::atomic<std::uint64_t> max_syscall_bytes{0};
    std::atomic<std::uint64_t> fail_at_offset{UINT64_MAX};
};
inline FaultInjection& fault_injection() noexcept { static FaultInjection f; return f; }

struct IoCounters {
    std::atomic<std::uint64_t> direct_writes{0}, fallback_writes{0}, direct_reads{0}, fallback_reads{0};
};
inline IoCounters& io_counters() noexcept { static IoCounters c; return c; }

// Aligned staging chunks are recycled: a restore opens one file per segment, and fresh
// multi-MiB allocations would page-fault on every file. At most four chunks are retained.
class StagingChunk {
public:
    StagingChunk() : data_(acquire()) {}
    ~StagingChunk() { release(data_); }
    StagingChunk(const StagingChunk&) = delete;
    StagingChunk& operator=(const StagingChunk&) = delete;
    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    static constexpr std::size_t capacity = chunk_bytes + alignment;
private:
    struct Pool {
        std::mutex mutex;
        std::vector<std::byte*> free;
        ~Pool() {
            for (auto* chunk : free) {
#if defined(_WIN32)
                ::_aligned_free(chunk);
#else
                std::free(chunk);
#endif
            }
        }
    };
    static Pool& pool() { static Pool p; return p; }
    static std::byte* acquire() {
        {
            auto& p = pool();
            std::lock_guard lock(p.mutex);
            if (!p.free.empty()) { auto* out = p.free.back(); p.free.pop_back(); return out; }
        }
        void* raw = nullptr;
#if defined(_WIN32)
        raw = ::_aligned_malloc(capacity, alignment);
#else
        if (::posix_memalign(&raw, alignment, capacity) != 0) raw = nullptr;
#endif
        if (!raw) throw std::bad_alloc();
        return static_cast<std::byte*>(raw);
    }
    static void release(std::byte* chunk) noexcept {
        if (!chunk) return;
        {
            auto& p = pool();
            std::lock_guard lock(p.mutex);
            if (p.free.size() < 4) {
                try { p.free.push_back(chunk); return; } catch (...) {}
            }
        }
#if defined(_WIN32)
        ::_aligned_free(chunk);
#else
        std::free(chunk);
#endif
    }
    std::byte* data_;
};

#if defined(__linux__)
namespace detail {
inline std::uint64_t round_up(std::uint64_t value) noexcept { return (value + alignment - 1) / alignment * alignment; }
inline bool set_direct(int fd, bool on) noexcept {
    if (on && fault_injection().refuse_direct.load(std::memory_order_relaxed)) { errno = EINVAL; return false; }
    const int flags = ::fcntl(fd, F_GETFL);
    return flags >= 0 && ::fcntl(fd, F_SETFL, on ? (flags | O_DIRECT) : (flags & ~O_DIRECT)) == 0;
}
// Buffered fallback: write back [offset, offset+bytes) and drop its clean pages.
inline void drop_written(int fd, std::uint64_t offset, std::uint64_t bytes) noexcept {
    if (!bytes) return;
    (void)::sync_file_range(fd, static_cast<off_t>(offset), static_cast<off_t>(bytes),
        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
    (void)::posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
}
}  // namespace detail

// Creates a new file and writes it front to back. finish() must be called for a complete
// file; a destroyed unfinished writer only closes its descriptor (callers remove temporaries).
class SequentialWriter {
public:
    SequentialWriter(const std::filesystem::path& path, bool exclusive, unsigned mode = 0644) {
        // Create buffered, then ask for O_DIRECT: open(O_CREAT|O_EXCL|O_DIRECT) on a file system
        // without direct I/O creates the file and still fails, which a retry would see as EEXIST.
        const int flags = O_WRONLY | O_CREAT | O_CLOEXEC | (exclusive ? O_EXCL : O_TRUNC);
        fd_ = ::open(path.c_str(), flags, mode);
        direct_ = fd_ >= 0 && detail::set_direct(fd_, true);
    }
    ~SequentialWriter() { if (fd_ >= 0) ::close(fd_); }
    SequentialWriter(const SequentialWriter&) = delete;
    SequentialWriter& operator=(const SequentialWriter&) = delete;
    [[nodiscard]] bool open() const noexcept { return fd_ >= 0; }
    [[nodiscard]] bool direct() const noexcept { return direct_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return written_ + fill_; }

    void write(const void* data, std::uint64_t bytes) {
        const auto* in = static_cast<const std::byte*>(data);
        while (bytes) {
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, chunk_bytes - fill_));
            std::memcpy(staging_.data() + fill_, in, n);
            fill_ += n; in += n; bytes -= n;
            if (fill_ == chunk_bytes) flush(false);
        }
    }
    // Completes the file at its exact logical length; durable adds a data sync.
    void finish(bool durable) {
        flush(true);
        if (padded_ && ::ftruncate(fd_, static_cast<off_t>(written_)) != 0)
            throw std::runtime_error("state file truncate failed");
        if (durable && ::fdatasync(fd_) != 0) throw std::runtime_error("state file sync failed");
        if (!direct_) detail::drop_written(fd_, 0, written_);
        const int fd = fd_; fd_ = -1;
        if (::close(fd) != 0) throw std::runtime_error("state file close failed");
        (direct_ ? io_counters().direct_writes : io_counters().fallback_writes).fetch_add(1, std::memory_order_relaxed);
    }

private:
    void flush(bool final) {
        if (!fill_) return;
        std::size_t length = fill_;
        if (direct_ && length % alignment) {
            // Only the last chunk is partial: pad to the block size, truncate in finish().
            const auto padded = static_cast<std::size_t>(detail::round_up(length));
            std::memset(staging_.data() + length, 0, padded - length);
            length = padded; padded_ = true;
        }
        put(staging_.data(), length, written_);
        if (!direct_) detail::drop_written(fd_, written_, fill_);
        written_ += fill_;
        fill_ = 0;
        (void)final;
    }
    long syscall_write(const std::byte* data, std::size_t length, std::uint64_t offset) {
        auto& faults = fault_injection();
        const auto fail_at = faults.fail_at_offset.load(std::memory_order_relaxed);
        if (offset >= fail_at) { errno = ENOSPC; return -1; }
        length = static_cast<std::size_t>(std::min<std::uint64_t>(length, fail_at - offset));
        if (const auto cap = faults.max_syscall_bytes.load(std::memory_order_relaxed); cap && length > cap)
            length = static_cast<std::size_t>(cap);
        return static_cast<long>(::pwrite(fd_, data, length, static_cast<off_t>(offset)));
    }
    void put(const std::byte* data, std::size_t length, std::uint64_t offset) {
        while (length) {
            const auto done = syscall_write(data, length, offset);
            if (done < 0 && errno == EINTR) continue;
            if (done < 0 && direct_ && errno == EINVAL && offset == 0) {
                // The file system accepted O_DIRECT at open but refuses this geometry.
                detail::set_direct(fd_, false); direct_ = false; continue;
            }
            if (done <= 0) throw std::runtime_error("state file write failed");
            const auto n = static_cast<std::size_t>(done);
            data += n; offset += n; length -= n;
            // A short direct write that ends off a block boundary cannot continue unbuffered.
            if (direct_ && length && (n % alignment)) { detail::set_direct(fd_, false); direct_ = false; }
        }
    }
    int fd_ = -1;
    bool direct_ = false, padded_ = false;
    StagingChunk staging_;
    std::size_t fill_ = 0;
    std::uint64_t written_ = 0;
};

// Reads an existing file front to back in aligned chunks. With read-ahead (files larger than
// one chunk, STRIX_SESSION_READAHEAD not 0) a helper thread reads the next chunk into a
// second staging buffer while the caller copies (and hashes) the current one.
class SequentialReader {
public:
    explicit SequentialReader(const std::filesystem::path& path, bool readahead = readahead_enabled()) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        direct_ = fd_ >= 0 && detail::set_direct(fd_, true);
        struct stat st{};
        if (fd_ >= 0 && ::fstat(fd_, &st) == 0) size_ = static_cast<std::uint64_t>(st.st_size);
        else if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        if (fd_ >= 0 && readahead && size_ > chunk_bytes) {
            second_ = std::make_unique<StagingChunk>();
            worker_ = std::thread([this] { produce(); });
        }
    }
    ~SequentialReader() {
        if (worker_.joinable()) {
            { std::lock_guard lock(mutex_); stop_ = true; }
            cv_.notify_all();
            worker_.join();
        }
        if (fd_ >= 0) ::close(fd_);
    }
    SequentialReader(const SequentialReader&) = delete;
    SequentialReader& operator=(const SequentialReader&) = delete;
    [[nodiscard]] bool open() const noexcept { return fd_ >= 0; }
    [[nodiscard]] bool direct() const noexcept { return direct_; }
    [[nodiscard]] bool reading_ahead() const noexcept { return worker_.joinable(); }
    // Size observed when the file was opened.
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] std::uint64_t consumed() const noexcept { return consumed_; }

    void read(void* out, std::uint64_t bytes) {
        auto* dst = static_cast<std::byte*>(out);
        while (bytes) {
            if (cursor_ == available_) next_chunk();
            if (cursor_ == available_) throw std::runtime_error("state file is truncated");
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, available_ - cursor_));
            std::memcpy(dst, current_ + cursor_, n);
            dst += n; bytes -= n; cursor_ += n; consumed_ += n;
        }
    }
    // After the opened size has been consumed: true when the file holds more bytes.
    [[nodiscard]] bool grew() {
        if (consumed_ < size_) return false;
        if (cursor_ < available_) return true;
        next_chunk();
        return cursor_ < available_;
    }

private:
    static bool readahead_enabled() noexcept { return switch_enabled("STRIX_SESSION_READAHEAD"); }
    // Reads the next file bytes into `into`; at the observed end it asks for one extra block
    // so growth is visible. A short read at the end marks EOF (later calls return 0).
    std::size_t fill(std::byte* into) {
        if (eof_) return 0;
        const auto remaining = size_ > offset_ ? size_ - offset_ : 0;
        const std::size_t want = static_cast<std::size_t>(remaining >= chunk_bytes ? chunk_bytes
            : detail::round_up(remaining) + alignment);
        std::size_t got = 0;
        while (got < want) {
            const auto n = ::pread(fd_, into + got, want - got, static_cast<off_t>(offset_ + got));
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && direct_ && errno == EINVAL && offset_ == 0 && got == 0) {
                detail::set_direct(fd_, false); direct_ = false; continue;
            }
            if (n < 0) throw std::runtime_error("state file read failed");
            if (n == 0) break;
            got += static_cast<std::size_t>(n);
            if (direct_ && got % alignment) {
                if (got >= remaining) break;  // end of file
                detail::set_direct(fd_, false); direct_ = false;
            }
        }
        if (got < want) eof_ = true;
        if (!direct_ && got) (void)::posix_fadvise(fd_, static_cast<off_t>(offset_), static_cast<off_t>(got), POSIX_FADV_DONTNEED);
        if (!counted_) {
            counted_ = true;
            (direct_ ? io_counters().direct_reads : io_counters().fallback_reads).fetch_add(1, std::memory_order_relaxed);
        }
        offset_ += got;
        return got;
    }
    // Helper thread: fills the two staging chunks alternately, one ahead of the caller.
    void produce() {
        std::byte* chunks[2] = {first_.data(), second_->data()};
        for (unsigned slot = 0;; slot ^= 1U) {
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !full_[slot]; });
                if (stop_) return;
            }
            std::size_t got = 0;
            std::exception_ptr failure;
            try { got = fill(chunks[slot]); } catch (...) { failure = std::current_exception(); }
            {
                std::lock_guard lock(mutex_);
                bytes_[slot] = got; full_[slot] = true; failure_ = failure;
                done_ = failure || got == 0;
            }
            cv_.notify_all();
            if (failure || got == 0) return;
        }
    }
    void next_chunk() {
        cursor_ = available_ = 0;
        if (!worker_.joinable()) {
            current_ = first_.data();
            available_ = fill(first_.data());
            return;
        }
        std::unique_lock lock(mutex_);
        if (holding_) { full_[slot_] = false; slot_ ^= 1U; holding_ = false; cv_.notify_all(); }
        cv_.wait(lock, [&] { return full_[slot_] || done_; });
        if (!full_[slot_]) return;  // the producer finished: nothing more
        if (failure_ && bytes_[slot_] == 0) std::rethrow_exception(failure_);
        holding_ = true;
        current_ = slot_ ? second_->data() : first_.data();
        available_ = bytes_[slot_];
    }
    int fd_ = -1;
    bool direct_ = false, counted_ = false, eof_ = false;
    std::uint64_t size_ = 0, offset_ = 0, consumed_ = 0;
    StagingChunk first_;
    std::unique_ptr<StagingChunk> second_;
    const std::byte* current_ = nullptr;
    std::size_t cursor_ = 0, available_ = 0;
    // Read-ahead state (guarded by mutex_ while the helper runs).
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool full_[2] = {false, false}, stop_ = false, done_ = false, holding_ = false;
    std::size_t bytes_[2] = {0, 0};
    unsigned slot_ = 0;
    std::exception_ptr failure_;
};
#else
// Other platforms keep the callers' original buffered paths; these are never constructed.
class SequentialWriter {
public:
    SequentialWriter(const std::filesystem::path&, bool, unsigned = 0644) {}
    [[nodiscard]] bool open() const noexcept { return false; }
    [[nodiscard]] bool direct() const noexcept { return false; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return 0; }
    void write(const void*, std::uint64_t) { throw std::logic_error("unbuffered state file I/O is Linux only"); }
    void finish(bool) { throw std::logic_error("unbuffered state file I/O is Linux only"); }
};
class SequentialReader {
public:
    explicit SequentialReader(const std::filesystem::path&, bool = false) {}
    [[nodiscard]] bool open() const noexcept { return false; }
    [[nodiscard]] bool direct() const noexcept { return false; }
    [[nodiscard]] bool reading_ahead() const noexcept { return false; }
    [[nodiscard]] std::uint64_t size() const noexcept { return 0; }
    [[nodiscard]] std::uint64_t consumed() const noexcept { return 0; }
    void read(void*, std::uint64_t) { throw std::logic_error("unbuffered state file I/O is Linux only"); }
    [[nodiscard]] bool grew() { return false; }
};
#endif

}  // namespace strix::state_file_io
