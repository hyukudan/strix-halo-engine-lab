#pragma once

#include "strix/weight/source.hpp"
#include "weight_load_queue.hpp"

#include <list>
#include <memory>
#include <span>

namespace strix::core {

// Feed an indexed shard into the common staged loader using the same source
// handle that admitted its header. This retains stale-source checks in the
// reader and works for both file-backed and mapped weights.
class WeightSliceLoader {
public:
    explicit WeightSliceLoader(WeightUploader& uploader, WeightLoadOptions options)
        : queue_(uploader, options), tickets_(queue_) {}

    void load(const weight::Slice& slice, std::span<std::uint8_t> destination) {
        if (!slice.source || destination.size() != slice.range.bytes ||
            slice.range.offset > slice.source->size() ||
            slice.range.bytes > slice.source->size() - slice.range.offset) {
            throw std::invalid_argument("weight load: inconsistent slice and destination");
        }
        if (destination.empty()) return;
        auto reader = std::ranges::find_if(sources_, [&](const auto& source) { return source.get() == slice.source.get(); });
        if (reader == sources_.end()) reader = sources_.emplace(sources_.end(), slice.source);
        tickets_.load(*reader, slice.range.offset, slice.range.bytes, destination.data());
    }

    void ready() {
        tickets_.flush();
        queue_.publish();
    }

    // A resident plan can also be submitted range by range by placement code.
    template <class Plan>
    void load_resident(const Plan& plan, std::span<std::uint8_t> storage) {
        if (storage.size() < plan.bytes())
            throw std::invalid_argument("weight load: incomplete resident storage");
        for (const auto& range : plan.ranges())
            load(range.source, storage.subspan(range.destination_offset, range.bytes));
        ready();
    }

    [[nodiscard]] WeightLoadStats stats() const { return queue_.stats(); }

private:
    class Reader final : public WeightSource {
    public:
        explicit Reader(std::shared_ptr<const weight::Source> source) : source_(std::move(source)) {}
        [[nodiscard]] const weight::Source* get() const { return source_.get(); }
        [[nodiscard]] std::uint64_t alignment() const override { return 1; }
        std::uint64_t read(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* destination) const override {
            const auto status = source_->read({offset, bytes},
                {reinterpret_cast<std::byte*>(destination), static_cast<std::size_t>(bytes)});
            if (!status) throw std::runtime_error("weight load: " + status.detail);
            return bytes;
        }
    private:
        std::shared_ptr<const weight::Source> source_;
    };

    std::list<Reader> sources_;  // Readers outlive worker drain on success or failure.
    WeightLoadQueue queue_;
    WeightRangeTickets tickets_;
};

}  // namespace strix::core
