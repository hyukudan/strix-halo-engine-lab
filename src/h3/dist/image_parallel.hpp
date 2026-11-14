#pragma once
// Image ownership and raw tile wire layout; transport and numerical operators stay shared.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::h3::dist {
enum class ImageParallel { automatic, sequence, request };
inline ImageParallel image_parallel(const std::string& name) {
    if (name == "auto") return ImageParallel::automatic;
    if (name == "sequence") return ImageParallel::sequence;
    if (name == "request") return ImageParallel::request;
    throw std::invalid_argument("image parallel must be auto, sequence or request");
}
struct ImageWork {
    bool sequence = false;
    std::vector<std::uint32_t> requests;
};
inline ImageWork image_work(ImageParallel policy, std::uint32_t count, std::uint32_t seed,
                            std::uint32_t world, std::uint32_t rank) {
    if ((world != 1 && world != 2) || rank >= world || count == 0 || count > 4096 ||
        count - 1 > std::numeric_limits<std::uint32_t>::max() - seed)
        throw std::invalid_argument("image: invalid rank/world, request count or seed range");
    if (policy == ImageParallel::sequence && world != 2)
        throw std::invalid_argument("image: sequence policy needs two ranks");
    ImageWork w;
    w.sequence = world == 2 && (policy == ImageParallel::sequence ||
        (policy == ImageParallel::automatic && count == 1));
    for (std::uint32_t i = 0; i < count; ++i)
        if (w.sequence || i % world == rank) w.requests.push_back(i);
    return w;
}

// Same-sized symmetric messages; unequal/empty halves are zero padded. No FP16 conversion.
class ImageTileExchange {
public:
    explicit ImageTileExchange(const std::vector<std::size_t>& sizes) : offsets_(1, 0) {
        for (auto n : sizes) {
            if (!n || n > std::numeric_limits<std::size_t>::max() / sizeof(float) - offsets_.back())
                throw std::invalid_argument("image: invalid tile payload size");
            offsets_.push_back(offsets_.back() + n);
        }
        cut_ = sizes.size() / 2;
    }
    std::size_t first(unsigned rank) const { check_rank(rank); return rank ? cut_ : 0; }
    std::size_t last(unsigned rank) const { check_rank(rank); return rank ? offsets_.size() - 1 : cut_; }
    std::size_t elements() const { return std::max(offsets_[cut_], offsets_.back() - offsets_[cut_]); }
    std::vector<float> pack(unsigned rank, const std::vector<std::vector<float>>& tiles) const {
        if (tiles.size() + 1 != offsets_.size()) throw std::invalid_argument("image: tile count");
        std::vector<float> out(elements(), 0);
        for (auto i = first(rank); i < last(rank); ++i) {
            if (tiles[i].size() != offsets_[i+1] - offsets_[i]) throw std::invalid_argument("image: tile size");
            std::copy(tiles[i].begin(), tiles[i].end(), out.begin() + (offsets_[i] - offsets_[first(rank)]));
        }
        return out;
    }
    std::vector<std::vector<float>> merge(const std::vector<float>& rank0, const std::vector<float>& rank1) const {
        if (rank0.size() != elements() || rank1.size() != elements()) throw std::invalid_argument("image: tile message size");
        std::vector<std::vector<float>> tiles(offsets_.size() - 1);
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto& message = rank ? rank1 : rank0;
            for (auto i = first(rank); i < last(rank); ++i) {
                const auto offset = offsets_[i] - offsets_[first(rank)];
                tiles[i].assign(message.begin() + offset, message.begin() + offset + offsets_[i+1] - offsets_[i]);
            }
        }
        return tiles;
    }
private:
    static void check_rank(unsigned rank) { if (rank > 1) throw std::invalid_argument("image: tile rank"); }
    std::vector<std::size_t> offsets_;
    std::size_t cut_ = 0;
};
} // namespace strix::h3::dist
