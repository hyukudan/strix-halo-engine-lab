#pragma once

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
        const auto bytes = scratch_.capacity() * sizeof(float);
        if (bytes <= capacity_ && !observed_.insert(name).second) {
            while (bytes > capacity_ - bytes_) {
                const auto old = cached_.find(recent_.front());
                bytes_ -= old->second.values.capacity() * sizeof(float);
                cached_.erase(old);
                recent_.pop_front();
            }
            recent_.push_back(name);
            try {
                const auto it = cached_.try_emplace(name, Slot{std::move(scratch_), std::prev(recent_.end())}).first;
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

} // namespace strix::core
