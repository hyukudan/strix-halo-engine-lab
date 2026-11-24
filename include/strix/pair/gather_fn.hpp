#pragma once
// The type-erased all-gather callback of an exact tensor-parallel pass (one symmetric swap of the slots, see
// strix/pair/gather_link.hpp) and the two-buffer wire ring a join stages its halves in. Backend-neutral host code: the
// stream and slot pointers cross as void*, so an engine binds any link (e.g. xport::tp_gather::HipPairGather::Link::fn)
// without this header knowing the runtime. Aggregates: an app builds one as `GatherFn{&link, &Link::fn}`.
// Home: shared (host shared, pair facility). Bound by: qwen35 dense HIP exact TP decode (HipEngine::TpGather) and Hy4 HIP
// TP step joins (Forward::TpGather); the HIP executors are src/xport/tp_join_exec_hip.hpp. Host test tests/pair_gather_fn_tests.cpp.
#include <cstddef>
#include <stdexcept>

namespace strix::pair {

struct GatherFn {
    void* ctx = nullptr;
    void (*fn)(void* ctx, void* stream, void* slots, std::size_t slot_bytes, std::size_t bytes) = nullptr;

    // One gather of `bytes` over slots of `slot_bytes` each, stream-ordered after the work that produced this rank's slot.
    void operator()(void* stream, void* slots, std::size_t slot_bytes, std::size_t bytes) const { fn(ctx, stream, slots, slot_bytes, bytes); }
};

// Two [2][slot] wire buffers used alternately (the swap of one join may still read its buffer while the next join packs).
template <class T>
class WireRing {
public:
    void bind(T* base, std::size_t slot) noexcept {
        base_ = base;
        slot_ = slot;
    }
    [[nodiscard]] T* base() const noexcept { return base_; }
    [[nodiscard]] std::size_t slot() const noexcept { return slot_; }
    // The next of the two buffers, for a join of `per_rank` values per rank.
    [[nodiscard]] T* next(std::size_t per_rank) {
        if (per_rank > slot_) throw std::logic_error("pair wire ring: join larger than its wire slot");
        flip_ ^= 1U;
        return base_ + std::size_t{flip_} * 2U * slot_;
    }

private:
    T* base_ = nullptr;
    std::size_t slot_ = 0;
    unsigned flip_ = 0;
};

}  // namespace strix::pair
