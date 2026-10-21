#include "strix/ds41_engram.hpp"

#include "strix/ds41_formats.hpp"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strix::ds41 {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

constexpr std::uint64_t page_size = 4096;

} // namespace

EngramHasher::EngramHasher(const deepseek41::EngramContract& c) : contract_(c) {
    const std::size_t layers = c.layer_ids.size(), cols = c.rows_per_token();
    require(c.max_ngram_size >= 2 && c.n_heads >= 1 && !c.layer_ids.empty() &&
                c.primes.size() == layers * cols && c.multipliers.size() == layers * c.max_ngram_size &&
                c.pad_id < c.token_map.size(),
            "Engram contract geometry");
    pad_ = c.token_map[c.pad_id];
    offsets_.resize(layers * cols);
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t sum = 0;
        for (std::size_t k = 0; k < cols; ++k) {
            offsets_[e * cols + k] = sum;
            sum += c.primes[e * cols + k];
        }
        require(sum == c.rows[e], "Engram primes do not sum to the table rows");
    }
}

void EngramHasher::push(std::uint32_t token, std::span<std::uint64_t> rows) {
    const auto& c = contract_;
    const std::size_t layers = c.layer_ids.size(), cols = c.rows_per_token(), n = c.max_ngram_size;
    require(token < c.token_map.size() && rows.size() == layers * cols, "Engram hash request");
    history_.push_back(c.token_map[token]);
    const std::size_t t = history_.size() - 1;
    std::vector<std::uint64_t> ids(n);
    bool blocked = false;
    for (std::size_t s = 0; s < n; ++s) {
        blocked = blocked || t < s;  // text only: no dead (image) positions
        ids[s] = blocked ? pad_ : history_[t - s];
    }
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t rolling = ids[0] * c.multipliers[e * n];
        for (std::size_t s = 1; s < n; ++s) {
            rolling ^= ids[s] * c.multipliers[e * n + s];
            for (std::uint32_t h = 0; h < c.n_heads; ++h) {
                const std::size_t k = (s - 1) * c.n_heads + h;
                rows[e * cols + k] = rolling % c.primes[e * cols + k] + offsets_[e * cols + k];
            }
        }
    }
}

std::vector<std::uint64_t> EngramHasher::hash_sequence(std::span<const std::uint32_t> tokens) {
    reset();
    const std::size_t per = static_cast<std::size_t>(layers()) * columns();
    std::vector<std::uint64_t> out(tokens.size() * per);
    for (std::size_t i = 0; i < tokens.size(); ++i) push(tokens[i], std::span<std::uint64_t>(out.data() + i * per, per));
    return out;
}

void EngramHasher::truncate(std::uint64_t positions) {
    require(positions <= history_.size(), "Engram history cannot grow by truncation");
    history_.resize(positions);
}

EngramRowReader::EngramRowReader(const std::filesystem::path& file, std::vector<EngramTable> tables, unsigned threads,
                                 bool direct_io)
    : tables_(std::move(tables)) {
    require(threads >= 1 && threads <= 256 && !tables_.empty(), "Engram reader configuration");
    if (direct_io) {
        fd_ = ::open(file.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        direct_ = fd_ >= 0;
    }
    if (fd_ < 0) fd_ = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("Engram reader cannot open the model file: " + std::string(std::strerror(errno)));
    struct stat st{};
    if (::fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("Engram reader cannot stat the model file"); }
    file_bytes_ = static_cast<std::uint64_t>(st.st_size);
    for (const auto& t : tables_)
        if (t.rows == 0 || t.file_offset + t.rows * engram_row_bytes > file_bytes_) {
            ::close(fd_);
            throw std::invalid_argument("Engram table lies outside the model file");
        }
    for (unsigned i = 0; i < threads; ++i) threads_.emplace_back([this] { worker(); });
}

EngramRowReader::~EngramRowReader() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
    if (fd_ >= 0) ::close(fd_);
}

void EngramRowReader::read_one(const EngramRowRequest& r, std::uint8_t* out, std::uint8_t* page) {
    if (r.table >= tables_.size() || r.row >= tables_[r.table].rows) throw std::out_of_range("Engram row out of range");
    const std::uint64_t offset = tables_[r.table].file_offset + r.row * engram_row_bytes;
    if (!direct_) {
        std::size_t got = 0;
        while (got < engram_row_bytes) {
            const auto n = ::pread(fd_, out + got, engram_row_bytes - got, static_cast<off_t>(offset + got));
            if (n <= 0) throw std::runtime_error("Engram row read failed");
            got += static_cast<std::size_t>(n);
        }
        page_reads_.fetch_add(1, std::memory_order_relaxed);
        page_bytes_.fetch_add(engram_row_bytes, std::memory_order_relaxed);
        return;
    }
    const std::uint64_t first = offset & ~(page_size - 1);
    const std::uint64_t last = (offset + engram_row_bytes + page_size - 1) & ~(page_size - 1);
    const std::size_t length = static_cast<std::size_t>(last - first);  // one or two pages
    std::size_t got = 0;
    while (got < length) {
        const auto n = ::pread(fd_, page + got, length - got, static_cast<off_t>(first + got));
        if (n <= 0) throw std::runtime_error("Engram row direct read failed");
        got += static_cast<std::size_t>(n);
        if (got % page_size) throw std::runtime_error("Engram direct read returned a partial page");
    }
    std::memcpy(out, page + (offset - first), engram_row_bytes);
    page_reads_.fetch_add(1, std::memory_order_relaxed);
    page_bytes_.fetch_add(length, std::memory_order_relaxed);
}

void EngramRowReader::worker() {
    void* raw = nullptr;
    if (::posix_memalign(&raw, page_size, 2 * page_size) != 0) std::abort();
    auto* page = static_cast<std::uint8_t*>(raw);
    std::uint64_t seen = 0;
    for (;;) {
        std::span<const EngramRowRequest> batch;
        std::uint8_t* out = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) break;
            seen = generation_;
            batch = batch_;
            out = out_;
            ++active_;
        }
        std::size_t done = 0;
        std::string error;
        for (;;) {
            const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= batch.size()) break;
            try {
                read_one(batch[i], out + i * engram_row_bytes, page);
            } catch (const std::exception& e) {
                if (error.empty()) error = e.what();
            }
            ++done;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ += done;
            --active_;
            if (!error.empty() && !failed_) { failed_ = true; error_ = error; }
        }
        done_.notify_all();
    }
    std::free(page);
}

void EngramRowReader::read(std::span<const EngramRowRequest> requests, std::span<std::uint8_t> out) {
    require(out.size() == requests.size() * engram_row_bytes, "Engram read output size");
    if (requests.empty()) return;
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    batch_ = requests;
    out_ = out.data();
    finished_ = 0;
    failed_ = false;
    error_.clear();
    next_.store(0, std::memory_order_relaxed);
    ++generation_;
    wake_.notify_all();
    // Every request is counted once, and no worker may still hold this batch when read() returns.
    done_.wait(lock, [&] { return finished_ == requests.size() && active_ == 0; });
    batch_ = {};
    out_ = nullptr;
    stats_.requests += requests.size();
    ++stats_.batches;
    stats_.busy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (failed_) throw std::runtime_error("Engram batch read failed: " + error_);
}

EngramReaderStats EngramRowReader::stats() const {
    auto s = stats_;
    s.page_reads = page_reads_.load();
    s.page_bytes = page_bytes_.load();
    return s;
}

std::vector<EngramTable> engram_tables(const deepseek41::TargetModel& model) {
    std::vector<EngramTable> out;
    for (std::size_t e = 0; e < model.engram.layer_ids.size(); ++e) {
        const std::string name = "blk." + std::to_string(model.engram.layer_ids[e]) + ".engram_embd.weight";
        bool found = false;
        for (const auto& t : model.manifest.tensors)
            if (t.name == name) {
                require(t.byte_size == model.engram.rows[e] * engram_row_bytes, "Engram table size differs from its rows");
                out.push_back({t.absolute_offset, model.engram.rows[e]});
                found = true;
            }
        require(found, "Engram table missing from the manifest");
    }
    return out;
}

} // namespace strix::ds41
