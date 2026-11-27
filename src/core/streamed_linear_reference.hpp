#pragma once

#include "core/bf16_reference.hpp"
#include "core/host_lane.hpp"
#include "core/linear_reference.hpp"

#include <list>
#include <map>
#include <set>

namespace strix::core {

template <class Work>
void reference_row_ranges(HostLane& lane, unsigned threads, std::size_t size, const Work& work) {
    const auto jobs = std::min<std::size_t>(threads, size / 16);
    if (jobs < 2) {
        work(0, size);
        return;
    }
    std::vector<std::future<void>> tasks;
    tasks.reserve(jobs);
    std::exception_ptr failure;
    try {
        for (std::size_t job = 0; job < jobs; ++job) {
            const auto first = size * job / jobs;
            const auto last = size * (job + 1) / jobs;
            tasks.push_back(lane.submit(HostPriority::background, [&, first, last] { work(first, last); }));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& task : tasks) {
        try { task.get(); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
}

// Each dot uses the reference's unchanged input order and fused operations.
inline void parallel_linear_reference(HostLane& lane, unsigned threads, std::span<const float> matrix,
                                      std::span<const float> input, std::span<float> output, unsigned groups = 1) {
    if (groups == 0 || input.empty() || input.size() % groups || output.size() % groups)
        throw std::invalid_argument("linear reference: invalid projection groups");
    const auto in = input.size() / groups;
    const auto per_group = output.size() / groups;
    if (per_group == 0 || matrix.size() % in || matrix.size() / in != output.size())
        throw std::invalid_argument("linear reference: matrix dimensions do not match input and output");
    reference_row_ranges(lane, threads, output.size(), [&](std::size_t first, std::size_t last) {
        for (auto row = first; row < last; ++row)
            output[row] = dot_f32(matrix.subspan(row * in, in), input.subspan(row / per_group * in, in));
    });
}

// A bounded decoded cache. Admit on reuse to keep one-off reads from taking
// space away from recurring projections. Matrices that do not fit are used
// from a single scratch buffer. The source must be immutable for this lifetime.
class ReferenceMatrixCache {
public:
    explicit ReferenceMatrixCache(std::size_t capacity) : capacity_(capacity) {}

    template <class Read>
    [[nodiscard]] std::span<const float> get(const std::string& name, std::size_t size, const Read& read) {
        if (const auto it = cached_.find(name); it != cached_.end()) {
            if (it->second.values.size() != size) throw std::invalid_argument("linear reference: matrix resized");
            recent_.splice(recent_.end(), recent_, it->second.position);
            return it->second.values;
        }
        scratch_.resize(size);
        read(name, std::span<float>(scratch_), 0);
        if (scratch_.size() <= capacity_ / sizeof(float) && !observed_.insert(name).second) {
            auto retained = scratch_.capacity() == size ? std::move(scratch_) : std::vector<float>(scratch_);
            const auto bytes = retained.capacity() * sizeof(float);
            if (bytes > capacity_) {
                scratch_ = std::move(retained);
                return scratch_;
            }
            while (bytes > capacity_ - bytes_) {
                const auto old = cached_.find(recent_.front());
                bytes_ -= old->second.values.capacity() * sizeof(float);
                cached_.erase(old);
                recent_.pop_front();
            }
            recent_.push_back(name);
            try {
                const auto it = cached_.try_emplace(name, Slot{std::move(retained), std::prev(recent_.end())}).first;
                bytes_ += bytes;
                return it->second.values;
            } catch (...) {
                recent_.pop_back();
                throw;
            }
        }
        return scratch_;
    }

private:
    struct Slot {
        std::vector<float> values;
        std::list<std::string>::iterator position;
    };
    std::size_t capacity_, bytes_{};
    std::list<std::string> recent_;
    std::map<std::string, Slot, std::less<>> cached_;
    std::set<std::string, std::less<>> observed_;
    std::vector<float> scratch_;
};

// Logits of `positions` rows against a [vocab x hidden] matrix that is read in
// slabs of `slab_rows` rows (`read(span, first element offset)`), so a matrix
// that does not fit in memory is read once for all positions. `store(weight,
// logits, column)` writes column `column` of every row and owns the dot's
// accumulation order. Optionally stores every logit rounded to BF16.
template <class Read, class Store>
[[nodiscard]] std::vector<std::vector<float>> streamed_head_projection(
    HostLane& lane, unsigned threads, std::size_t positions, std::size_t vocab, std::size_t hidden,
    std::size_t slab_rows, bool round_to_bf16, const Read& read, const Store& store) {
    if (!positions || !vocab || !hidden || !slab_rows)
        throw std::invalid_argument("streamed head projection: empty dimension");
    std::vector<float> block(slab_rows * hidden);
    std::vector<std::vector<float>> logits(positions, std::vector<float>(vocab));
    for (std::size_t first = 0; first < vocab; first += slab_rows) {
        const auto count = std::min(slab_rows, vocab - first);
        const auto matrix = std::span(block).first(count * hidden);
        read(matrix, first * hidden);
        reference_row_ranges(lane, threads, count, [&](std::size_t begin, std::size_t finish) {
            for (auto col = begin; col < finish; ++col)
                store(std::span<const float>(matrix).subspan(col * hidden, hidden), logits, first + col);
        });
    }
    if (round_to_bf16)
        for (auto& row : logits) round_bf16_reference(row);
    return logits;
}

} // namespace strix::core
