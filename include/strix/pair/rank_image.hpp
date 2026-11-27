#pragma once
// A rank's packed resident weight image over source slices. The model supplies descriptors (name, shard, slice)
// and an owner predicate; this header owns the rank-scope rule, the source-order offset packing and the bounded
// serial range loader.
#include "strix/pair/rank_memory.hpp"
#include "strix/weight/source.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace strix::pair {

// A tensor of the rank image needs: std::string name, shard; weight::Slice source; std::uint64_t resident_offset.
template<class T>
concept ImageTensor = requires(T t) {
    { t.name } -> std::convertible_to<std::string_view>;
    { t.shard } -> std::convertible_to<std::string_view>;
    { t.source } -> std::convertible_to<weight::Slice>;
    { t.resident_offset } -> std::convertible_to<std::uint64_t>;
};

// Whether a plan scoped to `local` (nullopt: both ranks) covers `rank`.
[[nodiscard]] inline bool covers_rank(std::optional<std::uint32_t> local, std::uint32_t rank) noexcept {
    return !local || *local == rank;
}

// Rank scope: the local rank must be 0 or 1, and every name absent from the local copy must belong to the peer
// (owner(name) is the owning rank, or nullopt for tensors that stay deferred).
template<class Owner>
void check_rank_scope(std::uint32_t local, std::span<const std::string> missing, Owner&& owner) {
    if (local > 1U) throw std::invalid_argument("pair rank scope: rank must be 0 or 1");
    for (const auto& name : missing)
        if (const std::optional<std::uint32_t> who = owner(name); who && *who == local)
            throw std::invalid_argument("pair rank scope: rank " + std::to_string(local) + " copy lacks its tensor " + name);
}

struct PackedImage { std::uint64_t payload{}, resident{}, largest{}; };

// Sort into source read order (shard, offset, name) and assign aligned destination offsets.
template<ImageTensor T>
PackedImage pack_rank_image(std::vector<T>& tensors, std::uint64_t alignment) {
    std::ranges::sort(tensors, {}, [](const T& item) {
        return std::tie(item.shard, item.source.range.offset, item.name);
    });
    PackedImage image;
    for (auto& item : tensors) {
        item.resident_offset = deploy::aligned(image.resident, alignment);
        image.resident = deploy::checked_add(item.resident_offset, item.source.range.bytes);
        image.payload = deploy::checked_add(image.payload, item.source.range.bytes);
        image.largest = std::max(image.largest, item.source.range.bytes);
    }
    image.resident = deploy::aligned(image.resident, alignment);
    return image;
}

using ImageWrite = std::function<void(std::uint64_t resident_offset, std::span<const std::byte>)>;

// Bounded, serial source reads in the packed order. The writer consumes each chunk synchronously.
template<ImageTensor T>
void stream_rank_image(std::span<const T> tensors, std::span<std::byte> staging, const ImageWrite& write) {
    if (staging.empty() || !write) throw std::invalid_argument("pair rank image: invalid weight staging or writer");
    for (const auto& tensor : tensors) {
        for (std::uint64_t offset = 0; offset < tensor.source.range.bytes;) {
            const auto count = std::min<std::uint64_t>(staging.size(), tensor.source.range.bytes - offset);
            const auto chunk = staging.first(static_cast<std::size_t>(count));
            weight::Slice slice;
            if (const auto status = weight::subslice(tensor.source, {offset, count}, slice); !status)
                throw std::runtime_error(std::string(tensor.name) + ": " + status.detail);
            if (const auto status = weight::read_slice(slice, chunk); !status)
                throw std::runtime_error(std::string(tensor.name) + ": " + status.detail);
            write(tensor.resident_offset + offset, chunk);
            offset += count;
        }
    }
}

} // namespace strix::pair
