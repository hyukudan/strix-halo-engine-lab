#include "strix/ple_reclaim.hpp"
#include "strix/qpf2k_defaults.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(__linux__)
#include <glob.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strix::ple_reclaim {
namespace {
std::atomic<Prefetcher*> active_prefetcher{nullptr};

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("PLE reclaim byte accounting overflow");
    return a + b;
}
[[maybe_unused]] std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error("PLE reclaim byte accounting overflow");
    return a * b;
}
[[maybe_unused]] std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
[[maybe_unused]] bool test_bit(const std::vector<std::uint64_t>& bits, std::uint64_t i) noexcept {
    return (bits[static_cast<std::size_t>(i >> 6U)] >> (i & 63U)) & 1U;
}
[[maybe_unused]] void set_bit(std::vector<std::uint64_t>& bits, std::uint64_t i) noexcept {
    bits[static_cast<std::size_t>(i >> 6U)] |= std::uint64_t{1} << (i & 63U);
}
// Sum of this process's KFD queue eviction time over its GPUs, -1 if unreadable (no KFD or no stats).
std::int64_t kfd_evicted_ms() {
#if defined(__linux__)
    const auto pattern = "/sys/class/kfd/kfd/proc/" + std::to_string(::getpid()) + "/stats_*/evicted_ms";
    glob_t found{};
    if (::glob(pattern.c_str(), 0, nullptr, &found) != 0) { ::globfree(&found); return -1; }
    std::int64_t total = 0;
    bool any = false;
    for (std::size_t i = 0; i < found.gl_pathc; ++i) {
        std::ifstream in(found.gl_pathv[i]);
        long long value = 0;
        if (in >> value) { total += value; any = true; }
    }
    ::globfree(&found);
    return any ? total : -1;
#else
    return -1;
#endif
}
}  // namespace

Mode requested() noexcept {
    if (qpf2k::suppressed(qpf2k::Item::ple_reclaim)) return Mode::off;
    const char* v = std::getenv("STRIX_PLE_RECLAIMABLE");
    if (v) {
        if (std::strcmp(v, "1") == 0) return Mode::required;
        return std::strcmp(v, "auto") == 0 ? Mode::automatic : Mode::off;
    }
    // Also honor the defaults policy when no startup injection has run (host callers/tests).
    const char* defaults = std::getenv("STRIX_QPF2K_DEFAULTS");
    return qpf2k::gsq_model || (defaults && defaults[0] == '0') ? Mode::off : Mode::automatic;
}

void set_active(Prefetcher* prefetcher) noexcept { active_prefetcher.store(prefetcher, std::memory_order_release); }
Prefetcher* active() noexcept { return active_prefetcher.load(std::memory_order_acquire); }

std::vector<Interval> reclaim_intervals(std::uint64_t table_offset, std::uint64_t table_bytes, std::uint64_t page,
    std::vector<Interval> registered) {
    if (!page || (page & (page - 1U))) throw std::invalid_argument("PLE reclaim page size must be a power of two");
    const auto end = checked_add(table_offset, table_bytes);
    const auto first = checked_add(table_offset, page - 1U) & ~(page - 1U);
    const auto last = end & ~(page - 1U);
    std::vector<Interval> result;
    if (last <= first) return result;
    for (auto& r : registered) {
        if (r.end < r.begin) throw std::invalid_argument("PLE reclaim registered interval is inverted");
        r.begin &= ~(page - 1U);
        r.end = checked_add(r.end, page - 1U) & ~(page - 1U);
    }
    std::sort(registered.begin(), registered.end(), [](const Interval& a, const Interval& b) { return a.begin < b.begin; });
    auto at = first;
    for (const auto& r : registered) {
        if (r.end <= at || r.begin >= last) continue;
        if (r.begin > at) result.push_back({at, r.begin});
        at = std::max(at, r.end);
        if (at >= last) break;
    }
    if (at < last) result.push_back({at, last});
    return result;
}

bool Prefetcher::in_reclaim(std::uint64_t page_index) const noexcept {
    const auto byte = page_index * config_.page;
    const auto it = std::upper_bound(config_.reclaim.begin(), config_.reclaim.end(), byte,
        [](std::uint64_t value, const Interval& interval) { return value < interval.begin; });
    return it != config_.reclaim.begin() && byte < std::prev(it)->end;
}

Prefetcher::Prefetcher(Config config, const PleHashContract& contract, std::span<const std::uint32_t> prompt,
    bool automatic)
    : config_(std::move(config)) {
#if !defined(__linux__)
    static_cast<void>(contract); static_cast<void>(prompt); static_cast<void>(automatic);
    throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1 requires Linux (madvise/mincore/fadvise)");
#else
    const auto page = config_.page;
    if (!config_.mapping_base || !page || (page & (page - 1U)) || config_.mapping_base % page || !config_.stored_row_bytes ||
        !config_.table_rows || config_.table_rows > std::numeric_limits<std::uint32_t>::max() || !config_.drop_page_cache)
        throw std::invalid_argument("PLE reclaim configuration is incomplete");
    const auto table_bytes = checked_mul(config_.table_rows, config_.stored_row_bytes);
    if (checked_add(config_.table_offset, table_bytes) > config_.mapping_bytes)
        throw std::invalid_argument("PLE reclaim table exceeds the model mapping");
    if (prompt.empty()) throw std::invalid_argument("PLE reclaim requires the known prompt tokens");
    if (!contract.head_count || contract.head_count > PleHashContract::maximum_heads || contract.ngram_size < 2)
        throw std::invalid_argument("PLE reclaim hash contract is invalid");
    std::uint64_t reclaimable = 0;
    for (std::size_t i = 0; i < config_.reclaim.size(); ++i) {
        const auto& r = config_.reclaim[i];
        if (r.begin >= r.end || r.begin % page || r.end % page || r.end > config_.mapping_bytes ||
            r.begin < config_.table_offset || r.end > config_.table_offset + table_bytes ||
            (i && r.begin < config_.reclaim[i - 1].end))
            throw std::invalid_argument("PLE reclaim intervals must be sorted page-aligned table interior ranges");
        reclaimable = checked_add(reclaimable, r.end - r.begin);
    }
    if (config_.reclaim.empty()) throw std::invalid_argument("PLE reclaim table has no unregistered interior page");
    first_page_ = config_.reclaim.front().begin / page;
    last_page_ = config_.reclaim.back().end / page;
    const auto span_pages = last_page_ - first_page_;
    issued_.assign(static_cast<std::size_t>((span_pages + 63U) / 64U), 0U);
    std::vector<std::uint64_t> touched(issued_.size(), 0U);

    // The launcher's hash chain (hip_ple_launcher.hip launch: position 0 starts from EOS context, then next_context).
    heads_ = contract.head_count;
    positions_ = prompt.size();
    rows_.resize(static_cast<std::size_t>(checked_mul(positions_, heads_)));
    const auto context_count = static_cast<std::size_t>(contract.ngram_size - 1U);
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> previous{};
    std::fill_n(previous.begin(), context_count, contract.eos_token_id);
    std::uint64_t touched_pages = 0;
    for (std::uint64_t p = 0; p < positions_; ++p) {
        const auto hash = hash_ple_token(contract, prompt[static_cast<std::size_t>(p)],
            std::span<const std::uint32_t>(previous.data(), context_count));
        if (hash.row_count != heads_) throw std::logic_error("PLE reclaim hash row count differs from the contract");
        for (std::uint32_t h = 0; h < heads_; ++h) {
            const auto row = hash.row_ids[h];
            if (row >= config_.table_rows) throw std::out_of_range("PLE reclaim row id exceeds the table");
            rows_[static_cast<std::size_t>(p * heads_ + h)] = static_cast<std::uint32_t>(row);
            const auto begin = config_.table_offset + row * config_.stored_row_bytes;
            for (auto pg = begin / page; pg <= (begin + config_.stored_row_bytes - 1U) / page; ++pg) {
                if (pg < first_page_ || pg >= last_page_ || !in_reclaim(pg)) continue;
                if (!test_bit(touched, pg - first_page_)) { set_bit(touched, pg - first_page_); ++touched_pages; }
            }
        }
        std::copy_n(hash.next_context.begin(), context_count, previous.begin());
    }
    const auto touched_bytes = checked_mul(touched_pages, page);
    const auto margin = checked_add(64ULL << 20U, touched_bytes / 64U);
    admitted_ = checked_add(checked_add(touched_bytes, config_.decode_allowance_bytes), margin);
    credit_ = reclaimable > admitted_ ? reclaimable - admitted_ : 0U;
    stats_.reclaimable_bytes = reclaimable;
    stats_.touched_bytes = touched_bytes;
    stats_.admitted_bytes = admitted_;
    stats_.credit_bytes = credit_;

    // Drop before the large allocations: never leave the choice of victim pages to kernel reclaim (registered model
    // pages are unpinned userptr pages; reclaiming one evicts every queue of the process).
    bool failed = false;
    stats_.kfd_evicted_ms_before_drop = kfd_evicted_ms();
    stats_.resident_before_drop_bytes = resident_bytes(&failed);
    for (const auto& r : config_.reclaim) {
        auto* address = reinterpret_cast<void*>(config_.mapping_base + static_cast<std::uintptr_t>(r.begin));
        const auto length = static_cast<std::size_t>(r.end - r.begin);
        (void)madvise(address, length, MADV_RANDOM);
        if (madvise(address, length, MADV_DONTNEED) != 0)
            throw std::runtime_error("PLE reclaim: MADV_DONTNEED on the table interior failed");
        if (const int error = config_.drop_page_cache(r.begin, r.end - r.begin))
            throw std::runtime_error(std::string("PLE reclaim: page-cache drop (posix_fadvise DONTNEED) failed: ") +
                std::strerror(error));
    }
    stats_.resident_after_drop_bytes = resident_bytes(&failed);
    stats_.kfd_evicted_ms_after_drop = kfd_evicted_ms();
    stats_.kfd_evicted_during_drop = stats_.kfd_evicted_ms_before_drop >= 0 &&
        stats_.kfd_evicted_ms_after_drop > stats_.kfd_evicted_ms_before_drop;
    stats_.mincore_failed = failed;
    if (!automatic) report_setup();
    // The credit rests on the drop: refuse unless mincore is authoritative and shows the interior (nearly) empty.
    // Linux reports every page resident to a caller that neither owns nor may write the file, and pages another
    // process still maps survive POSIX_FADV_DONTNEED; both land here.
    if (failed)
        throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1: mincore on the PLE table failed; the drop cannot be verified");
    if (stats_.resident_after_drop_bytes > config_.drop_residual_bytes)
        throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1: " + std::to_string(stats_.resident_after_drop_bytes) +
            " B of the PLE table stay resident after the drop (limit " + std::to_string(config_.drop_residual_bytes) +
            " B): another process maps the model, or mincore is not authoritative (file neither owned nor writable)");
    target_ = std::min(positions_, config_.ahead_rows);
    const std::lock_guard<std::mutex> lock(mutex_);
    thread_ = std::thread([this] { helper(); });
    if (automatic) report_setup();
#endif
}

void Prefetcher::report_setup() const noexcept {
    if (stats_.kfd_evicted_during_drop)
        std::fprintf(stderr, "ple-reclaim: KFD queue eviction time rose across the PLE drop (%lld -> %lld ms)\n",
            static_cast<long long>(stats_.kfd_evicted_ms_before_drop), static_cast<long long>(stats_.kfd_evicted_ms_after_drop));
    std::fprintf(stderr, "ple-reclaim: reclaimable %llu B, prompt %llu tokens touch %llu pages (%llu B), admitted %llu B, "
        "credit %llu B, resident %llu -> %llu B after drop%s\n",
        static_cast<unsigned long long>(stats_.reclaimable_bytes), static_cast<unsigned long long>(positions_),
        static_cast<unsigned long long>(stats_.touched_bytes / config_.page), static_cast<unsigned long long>(stats_.touched_bytes),
        static_cast<unsigned long long>(admitted_), static_cast<unsigned long long>(credit_),
        static_cast<unsigned long long>(stats_.resident_before_drop_bytes),
        static_cast<unsigned long long>(stats_.resident_after_drop_bytes), stats_.mincore_failed ? " (mincore failed)" : "");
}

Prefetcher::~Prefetcher() {
    if (active() == this) set_active(nullptr);
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
}

std::uint64_t Prefetcher::resident_bytes(bool* failed) const {
#if defined(__linux__)
    constexpr std::uint64_t chunk_pages = 1ULL << 20U;
    std::vector<unsigned char> vec(static_cast<std::size_t>(chunk_pages));
    std::uint64_t resident = 0;
    for (const auto& r : config_.reclaim) {
        for (auto at = r.begin; at < r.end; at += chunk_pages * config_.page) {
            const auto bytes = std::min<std::uint64_t>(r.end - at, chunk_pages * config_.page);
            if (mincore(reinterpret_cast<void*>(config_.mapping_base + static_cast<std::uintptr_t>(at)),
                    static_cast<std::size_t>(bytes), vec.data()) != 0) {
                if (failed) *failed = true;
                return 0;
            }
            const auto n = static_cast<std::size_t>(bytes / config_.page);
            for (std::size_t i = 0; i < n; ++i) resident += vec[i] & 1U;
        }
    }
    return resident * config_.page;
#else
    if (failed) *failed = true;
    return 0;
#endif
}

void Prefetcher::prefetch_until(std::uint64_t target) {
#if defined(__linux__)
    const auto t0 = now_ns();
    const auto page = config_.page;
    std::vector<std::uint64_t> pages;
    for (auto p = issued_until_; p < target; ++p)
        for (std::uint32_t h = 0; h < heads_; ++h) {
            const auto begin = config_.table_offset + std::uint64_t{rows_[static_cast<std::size_t>(p * heads_ + h)]} *
                config_.stored_row_bytes;
            for (auto pg = begin / page; pg <= (begin + config_.stored_row_bytes - 1U) / page; ++pg) {
                if (pg < first_page_ || pg >= last_page_ || test_bit(issued_, pg - first_page_) || !in_reclaim(pg)) continue;
                set_bit(issued_, pg - first_page_);
                pages.push_back(pg);
            }
        }
    std::sort(pages.begin(), pages.end());
    std::uint64_t issued = 0;
    std::vector<unsigned char> vec;
    for (std::size_t i = 0; i < pages.size();) {
        auto j = i + 1U;
        while (j < pages.size() && pages[j] == pages[j - 1U] + 1U) ++j;
        const auto count = static_cast<std::size_t>(j - i);
        auto* address = reinterpret_cast<void*>(config_.mapping_base + static_cast<std::uintptr_t>(pages[i] * page));
        const auto length = static_cast<std::size_t>(count * page);
        vec.resize(count);
        std::uint64_t missing = count;
        if (mincore(address, length, vec.data()) == 0) {
            missing = 0;
            for (const auto v : vec) missing += (v & 1U) ? 0U : 1U;
        }
        if (missing) {
            (void)madvise(address, length, MADV_WILLNEED);
            issued += missing;
        }
        i = j;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    stats_.prefetch_pages_considered += pages.size();
    stats_.prefetch_pages_issued += issued;
    stats_.prefetch_ns_total += now_ns() - t0;
#else
    static_cast<void>(target);
#endif
}

void Prefetcher::helper() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stop_ || target_ > issued_until_ || sample_; });
        if (stop_) return;
        const auto target = target_;
        const bool sample = sample_;
        sample_ = false;
        lock.unlock();
        if (target > issued_until_) prefetch_until(target);   // issued_until_ is written only by this thread
        bool failed = false;
        const auto resident = sample ? resident_bytes(&failed) : 0U;
        lock.lock();
        issued_until_ = std::max(issued_until_, target);
        stats_.prefetched_until = issued_until_;
        if (sample) {
            stats_.mincore_failed = stats_.mincore_failed || failed;
            ++stats_.resident_samples;
            stats_.resident_last_bytes = resident;
            stats_.resident_max_bytes = std::max(stats_.resident_max_bytes, resident);
            if (resident > admitted_ && !stats_.bound_exceeded) {
                stats_.bound_exceeded = true;
                std::fprintf(stderr, "ple-reclaim: resident PLE %llu B exceeds the admitted %llu B\n",
                    static_cast<unsigned long long>(resident), static_cast<unsigned long long>(admitted_));
            }
        }
    }
}

std::uint64_t Prefetcher::stage_rows(std::uint64_t first, std::uint64_t max_count, std::vector<std::uint32_t>& out) const {
    if (!config_.stage_prefetch || !max_count || first >= positions_) return 0;
    const auto count = std::min(max_count, positions_ - first);
    const auto begin = rows_.begin() + static_cast<std::ptrdiff_t>(first * heads_);
    out.assign(begin, begin + static_cast<std::ptrdiff_t>(count * heads_));
    return count;
}

void Prefetcher::on_gather(std::uint64_t first_position, std::span<const PleHashResult> hashes, const GatherReceipt& receipt) {
    const auto gather_ns = receipt.gather_ns, minflt = receipt.minflt, majflt = receipt.majflt;
    std::uint64_t mismatches = 0, outside = 0;
    for (std::size_t r = 0; r < hashes.size(); ++r) {
        const auto p = first_position + r;
        if (p >= positions_) { ++outside; continue; }
        const auto& hash = hashes[r];
        bool same = hash.row_count == heads_;
        for (std::uint32_t h = 0; same && h < heads_; ++h)
            same = hash.row_ids[h] == rows_[static_cast<std::size_t>(p * heads_ + h)];
        mismatches += same ? 0U : 1U;
    }
    Stats snapshot;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.blocks;
        stats_.gathered_rows += hashes.size();
        stats_.gather_ns_total += gather_ns;
        stats_.gather_ns_max = std::max(stats_.gather_ns_max, gather_ns);
        stats_.minflt_total += minflt;
        stats_.majflt_total += majflt;
        stats_.majflt_max = std::max(stats_.majflt_max, majflt);
        stats_.hash_mismatches += mismatches;
        if (receipt.stage_error) ++stats_.stage_errors;
        if (receipt.stage_attempted) {
            if (receipt.staged_rows) ++stats_.stage_hits; else ++stats_.stage_misses;
            stats_.stage_rows += receipt.staged_rows;
            stats_.stage_ns_total += receipt.stage_ns;
            stats_.stage_wait_ns_total += receipt.stage_wait_ns;
            stats_.stage_majflt_total += receipt.stage_majflt;
        }
        stats_.rows_outside_prompt += outside;
        if (mismatches && !stats_.bound_exceeded) {
            stats_.bound_exceeded = true;
            std::fprintf(stderr, "ple-reclaim: %llu gathered rows differ from the precomputed prompt hashes\n",
                static_cast<unsigned long long>(mismatches));
        }
        const auto end = first_position + hashes.size();
        target_ = std::max(target_, std::min(positions_, end + config_.ahead_rows));
        sample_ = true;
        snapshot = stats_;
    }
    wake_.notify_one();
    std::fprintf(stderr, "ple-reclaim block %llu rows %zu staged %llu (worker %.2f ms hidden, wait %.2f ms, majflt %llu) "
        "exposed gather %.2f ms minflt %llu majflt %llu prefetched-until %llu issued %llu pages resident %llu B\n",
        static_cast<unsigned long long>(first_position), hashes.size(), static_cast<unsigned long long>(receipt.staged_rows),
        static_cast<double>(receipt.stage_ns) / 1e6, static_cast<double>(receipt.stage_wait_ns) / 1e6,
        static_cast<unsigned long long>(receipt.stage_majflt),
        static_cast<double>(gather_ns) / 1e6, static_cast<unsigned long long>(minflt), static_cast<unsigned long long>(majflt),
        static_cast<unsigned long long>(snapshot.prefetched_until), static_cast<unsigned long long>(snapshot.prefetch_pages_issued),
        static_cast<unsigned long long>(snapshot.resident_last_bytes));
    if (config_.strict && snapshot.bound_exceeded)
        throw std::runtime_error("STRIX_PLE_RECLAIM_STRICT: PLE residency bound or prompt hash check failed");
}

void Prefetcher::finish() {
    bool failed = false;
    const auto resident = resident_bytes(&failed);
    const auto evicted = kfd_evicted_ms();
    bool exceeded = false;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stats_.final_sampled) return;
        stats_.final_sampled = true;
        stats_.kfd_evicted_ms_final = evicted;
        stats_.mincore_failed = stats_.mincore_failed || failed;
        ++stats_.resident_samples;
        stats_.resident_last_bytes = resident;
        stats_.resident_max_bytes = std::max(stats_.resident_max_bytes, resident);
        if (resident > admitted_ && !stats_.bound_exceeded) {
            stats_.bound_exceeded = true;
            std::fprintf(stderr, "ple-reclaim: final resident PLE %llu B exceeds the admitted %llu B\n",
                static_cast<unsigned long long>(resident), static_cast<unsigned long long>(admitted_));
        }
        exceeded = stats_.bound_exceeded;
    }
    if (config_.strict && exceeded) {
        // The run fails before its stdout receipt: leave the evidence on stderr first.
        std::fprintf(stderr, "ple-reclaim receipt: %s\n", receipt_json().c_str());
        throw std::runtime_error("STRIX_PLE_RECLAIM_STRICT: PLE residency bound or prompt hash check failed");
    }
}

Stats Prefetcher::stats() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

std::string Prefetcher::receipt_json() const {
    const auto s = stats();
    const auto u = [](std::uint64_t v) { return std::to_string(v); };
    return std::string("{\"scope\":\"STRIX_PLE_RECLAIMABLE=1 prefill PLE pages dropped at setup and prefetched; admission credit\"") +
        ",\"reclaimable_bytes\":" + u(s.reclaimable_bytes) + ",\"touched_bytes\":" + u(s.touched_bytes) +
        ",\"decode_allowance_bytes\":" + u(config_.decode_allowance_bytes) + ",\"admitted_bytes\":" + u(s.admitted_bytes) +
        ",\"credit_bytes\":" + u(s.credit_bytes) + ",\"resident_before_drop_bytes\":" + u(s.resident_before_drop_bytes) +
        ",\"resident_after_drop_bytes\":" + u(s.resident_after_drop_bytes) + ",\"blocks\":" + u(s.blocks) +
        ",\"gathered_rows\":" + u(s.gathered_rows) + ",\"gather_ns_total\":" + u(s.gather_ns_total) +
        ",\"gather_ns_max\":" + u(s.gather_ns_max) + ",\"minflt_total\":" + u(s.minflt_total) +
        ",\"majflt_total\":" + u(s.majflt_total) + ",\"majflt_max\":" + u(s.majflt_max) +
        ",\"prefetch_ahead_rows\":" + u(config_.ahead_rows) + ",\"prefetched_until\":" + u(s.prefetched_until) +
        ",\"prefetch_pages_considered\":" + u(s.prefetch_pages_considered) +
        ",\"prefetch_pages_issued\":" + u(s.prefetch_pages_issued) + ",\"prefetch_ns_total\":" + u(s.prefetch_ns_total) +
        ",\"resident_samples\":" + u(s.resident_samples) + ",\"resident_max_bytes\":" + u(s.resident_max_bytes) +
        ",\"resident_last_bytes\":" + u(s.resident_last_bytes) + ",\"hash_mismatches\":" + u(s.hash_mismatches) +
        ",\"rows_outside_prompt\":" + u(s.rows_outside_prompt) +
        ",\"stage_prefetch\":" + (config_.stage_prefetch ? "true" : "false") + ",\"stage_hits\":" + u(s.stage_hits) +
        ",\"stage_misses\":" + u(s.stage_misses) + ",\"stage_rows\":" + u(s.stage_rows) +
        ",\"stage_ns_total\":" + u(s.stage_ns_total) + ",\"stage_wait_ns_total\":" + u(s.stage_wait_ns_total) +
        ",\"stage_majflt_total\":" + u(s.stage_majflt_total) + ",\"stage_errors\":" + u(s.stage_errors) +
        ",\"strict\":" + (config_.strict ? "true" : "false") +
        ",\"exposed_faults_scope\":\"process-wide (RUSAGE_SELF) around the launch thread's hash+gather\"" +
        ",\"kfd_evicted_ms_before_drop\":" + std::to_string(s.kfd_evicted_ms_before_drop) +
        ",\"kfd_evicted_ms_after_drop\":" + std::to_string(s.kfd_evicted_ms_after_drop) +
        ",\"kfd_evicted_ms_final\":" + std::to_string(s.kfd_evicted_ms_final) +
        ",\"final_sampled\":" + (s.final_sampled ? "true" : "false") +
        ",\"kfd_evicted_during_drop\":" + (s.kfd_evicted_during_drop ? "true" : "false") +
        ",\"mincore_failed\":" + (s.mincore_failed ? "true" : "false") +
        ",\"bound_exceeded\":" + (s.bound_exceeded ? "true" : "false") + "}";
}

}  // namespace strix::ple_reclaim
