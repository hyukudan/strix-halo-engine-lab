#include "strix/dense_fast_registry.hpp"

#include "strix/env_flags.hpp"
#include "strix/gsq_a8_groups.hpp"

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace strix::dense_fast {
namespace {
struct Registry {
    std::shared_mutex mutex;
    std::unordered_map<const void*, Tensor> tensors;
    std::atomic<std::uint64_t> count{0}, hits{0}, q8_bytes{0}, bytes{0};
};
Registry& registry() {
    static Registry instance;
    return instance;
}
}  // namespace

void register_tensor(const void* original_address, const Tensor& tensor) {
    auto& r = registry();
    const std::unique_lock lock(r.mutex);
    r.tensors[original_address] = tensor;
    r.count.store(r.tensors.size(), std::memory_order_release);
}

void clear() noexcept {
    auto& r = registry();
    const std::unique_lock lock(r.mutex);
    r.tensors.clear();
    r.count.store(0U, std::memory_order_release);
}

bool kernels_enabled() noexcept {
    static const bool enabled = strix::envcfg::flag("STRIX_DENSE_FAST", true);
    return enabled;
}

bool rows_admitted(unsigned window_rows) noexcept {
    static const unsigned mask = strix::env_number<unsigned>("STRIX_DENSE_FAST_ROWS", 0x6U);
    return window_rows < 32U && (mask >> window_rows & 1U) != 0U;
}

const Tensor* find(const void* original_address, unsigned window_rows) noexcept {
    auto& r = registry();
    if (!r.count.load(std::memory_order_acquire) || !kernels_enabled() || !rows_admitted(window_rows)) return nullptr;
    const std::shared_lock lock(r.mutex);
    const auto it = r.tensors.find(original_address);
    if (it == r.tensors.end()) return nullptr;
    const auto& t = it->second;
    r.hits.fetch_add(1U, std::memory_order_relaxed);
    r.q8_bytes.fetch_add(static_cast<std::uint64_t>(t.rows) * (t.cols / 32U) * 34U, std::memory_order_relaxed);
    r.bytes.fetch_add(t.rows * gsq_a8::dense_row_bytes(t.type, t.cols), std::memory_order_relaxed);
    return &t;
}

std::uint64_t registered_count() noexcept { return registry().count.load(std::memory_order_acquire); }
std::uint64_t hits() noexcept { return registry().hits.load(std::memory_order_relaxed); }
std::uint64_t replaced_q8_bytes() noexcept { return registry().q8_bytes.load(std::memory_order_relaxed); }
std::uint64_t read_bytes() noexcept { return registry().bytes.load(std::memory_order_relaxed); }

}  // namespace strix::dense_fast
