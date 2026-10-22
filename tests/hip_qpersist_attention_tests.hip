#include "strix/hip_qpersist_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::qpersist_diagnostic;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Owner {
    strix::HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    struct Allocation { void* base; std::size_t bytes; };
    std::vector<Allocation> allocations;
    Owner() {
        require(guard.architecture().starts_with("gfx1151"), "qpersist gate requires gfx1151");
        guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); });
    }
    ~Owner() {
        guard.recover_or_fail_stop();
        for (const auto& a : allocations) if (hipFree(a.base) != hipSuccess) std::_Exit(3);
        if (stream && hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
    }
    void finish() { check(hipStreamSynchronize(stream)); guard.require_original(); }
    template <class T> T* allocate(std::size_t count) {
        void* p = nullptr;
        guard.run([&] { check(hipMalloc(&p, count * sizeof(T) + 128U)); });
        try { allocations.push_back({p, count * sizeof(T)}); }
        catch (...) { (void)hipFree(p); throw; }
        guard.run([&] { check(hipMemsetAsync(p, 0xa5, count * sizeof(T) + 128U, stream)); });
        return reinterpret_cast<T*>(static_cast<unsigned char*>(p) + 64U);
    }
    template <class T> T* upload(const std::vector<T>& values) {
        auto* p = allocate<T>(values.size()); finish();
        guard.run([&] { check(hipMemcpy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice)); });
        return p;
    }
    std::vector<unsigned char> read(const void* p, std::size_t bytes) {
        finish(); std::vector<unsigned char> out(bytes);
        guard.run([&] { check(hipMemcpy(out.data(), p, bytes, hipMemcpyDeviceToHost)); });
        return out;
    }
    void guards() {
        for (const auto& a : allocations) {
            const auto bytes = read(a.base, a.bytes + 128U);
            require(std::all_of(bytes.begin(), bytes.begin() + 64U, [](auto b) { return b == 0xa5; }), "leading guard changed");
            require(std::all_of(bytes.end() - 64U, bytes.end(), [](auto b) { return b == 0xa5; }), "trailing guard changed");
        }
    }
};
std::uint32_t random_state = 0x49544153U;
float number() {
    random_state = random_state * 1664525U + 1013904223U;
    if ((random_state & 63U) == 0U) return std::bit_cast<float>(0x80000000U);
    if ((random_state & 63U) == 1U) return std::bit_cast<float>(1U);
    return static_cast<float>(static_cast<int>((random_state >> 16U) % 31U) - 15) / 64.0F;
}
std::uint16_t bf16(float x) {
    const auto b = std::bit_cast<std::uint32_t>(x);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}
void equal(Owner& o, const void* a, const void* b, std::size_t bytes, const char* text) {
    require(o.read(a, bytes) == o.read(b, bytes), text);
}
void attention_case(unsigned dim, unsigned rows, unsigned count, bool dense) {
    Owner o;
    constexpr unsigned heads = 24U, kv_heads = 2U, maximum = 2051U;
    const unsigned width = heads * dim, capacity = maximum * 2U + 1U;
    std::vector<std::uint16_t> kv(static_cast<std::size_t>(capacity) * kv_heads * dim * 2U);
    for (auto& x : kv) x = bf16(number());
    const auto* device_kv = o.upload(kv);
    std::array<std::vector<AttentionRow>, 2> row;
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<float> query(width * 2U);
        for (auto& x : query) x = number();
        std::vector<std::uint32_t> selected(maximum, UINT32_MAX);
        const auto n = std::max(1U, count - std::min(count - 1U, r));
        for (unsigned s = 0; s < n; ++s) selected[s] = 2U * s + 1U;
        const auto* q = o.upload(query); const auto* ids = o.upload(selected);
        for (unsigned arm = 0; arm < 2U; ++arm)
            row[arm].push_back({q, ids, o.allocate<float>(heads * maximum), o.allocate<float>(width), n});
    }
    std::array<Attention, 2> a;
    for (unsigned arm = 0; arm < 2U; ++arm) {
        a[arm] = {row[arm], device_kv, heads, kv_heads, dim, maximum};
        if (dense) { a[arm].codes = o.allocate<std::int8_t>(rows * width); a[arm].deltas = o.allocate<float>((width / 32U) * 64U); }
        enqueue_scores(a[arm], arm != 0U, o.stream);
    }
    for (unsigned r = 0; r < rows; ++r)
        equal(o, row[0][r].scores, row[1][r].scores, heads * maximum * sizeof(float), "pre-softmax score bytes differ");
    for (unsigned arm = 0; arm < 2U; ++arm) enqueue_tail(a[arm], arm != 0U, o.stream);
    for (unsigned r = 0; r < rows; ++r) {
        equal(o, row[0][r].scores, row[1][r].scores, heads * maximum * sizeof(float), "probability bytes differ");
        equal(o, row[0][r].output, row[1][r].output, width * sizeof(float), "value/gate bytes differ");
    }
    if (dense) {
        equal(o, a[0].codes, a[1].codes, rows * width, "A8 code bytes differ");
        equal(o, a[0].deltas, a[1].deltas, (width / 32U) * 64U * sizeof(float), "A8 delta bytes differ");
    }
    o.guards();
}
void cache_case(unsigned mode, unsigned width, unsigned fault) {
    Owner o;
    constexpr unsigned index_dim = 128U, capacity = 3U;
    std::vector<float> index(index_dim), key(width), value(width);
    for (auto& x : index) x = number();
    for (auto& x : key) x = number();
    for (auto& x : value) x = number();
    auto& bad = mode == 0U ? index.back() : key.back();
    if (fault == 1U) bad = std::numeric_limits<float>::infinity();
    if (fault == 2U) bad = std::numeric_limits<float>::quiet_NaN();
    if (fault == 3U) bad = std::numeric_limits<float>::max(); // finite -> BF16 infinity
    if (fault == 4U && mode != 0U) value.front() = -std::numeric_limits<float>::infinity();
    const auto* i = o.upload(index); const auto* k = o.upload(key); const auto* v = o.upload(value);
    std::array<Cache, 2> c;
    for (unsigned arm = 0; arm < 2U; ++arm) {
        c[arm] = {i, k, v, o.allocate<std::uint16_t>(capacity * index_dim),
            o.allocate<std::uint16_t>(capacity * width * 2U), mode == 1U ? 0U : index_dim,
            mode == 0U ? 0U : width, fault == 6U ? capacity : 1U, capacity,
            o.upload(std::vector<std::uint32_t>{fault == 5U ? 4U : 0U})};
        enqueue_cache(c[arm], arm != 0U, o.stream);
    }
    equal(o, c[0].raw_state, c[1].raw_state, capacity * index_dim * sizeof(std::uint16_t), "raw cache bytes differ");
    equal(o, c[0].kv_state, c[1].kv_state, capacity * width * 2U * sizeof(std::uint16_t), "KV cache bytes differ");
    equal(o, c[0].status, c[1].status, sizeof(std::uint32_t), "sticky status bytes differ");
    o.guards();
}
}
int main() {
    try {
        unsigned attention = 0U, cache = 0U;
        for (unsigned dim : {128U, 256U}) for (unsigned rows : {1U, 2U, 4U, 8U})
            for (unsigned count : {1U, 31U, 32U, 33U, 1025U, 2048U, 2051U}) for (bool dense : {false, true}) {
                attention_case(dim, rows, count, dense); ++attention;
            }
        for (unsigned mode : {0U, 1U, 2U}) for (unsigned width : {512U, 513U, 1024U})
            for (unsigned fault = 0U; fault < 7U; ++fault) { cache_case(mode, width, fault); ++cache; }
        std::cout << "qpersist components PASS attention=" << attention << " cache=" << cache << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "qpersist components FAIL: " << e.what() << '\n'; return 1; }
}
