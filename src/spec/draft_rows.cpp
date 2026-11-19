#include "strix/spec/draft_rows.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace strix::spec {

DraftRows::DraftRows(DraftRowsConfig config) : config_(std::move(config)) {
    if (config_.vocab == 0 || config_.group == 0 || config_.vocab % config_.group != 0)
        throw std::invalid_argument("draft rows: vocab must be a positive multiple of the group size");
    groups_total_ = config_.vocab / config_.group;
    capacity_ = config_.capacity_groups ? std::min(config_.capacity_groups, groups_total_) : groups_total_;
    core_ = std::min(config_.core_groups, capacity_);
    for (const auto id : config_.always)
        if (id >= config_.vocab) throw std::invalid_argument("draft rows: always id past the vocabulary");
    for (const auto& p : config_.packs)
        for (const auto id : p)
            if (id >= config_.vocab) throw std::invalid_argument("draft rows: pack id past the vocabulary");
    slot_of_.assign(groups_total_, kNone);
    begin({});
    materialised_ = 0;
}

bool DraftRows::add_group(std::uint32_t g) {
    if (slot_of_[g] != kNone) return false;
    if (slots_.size() >= capacity_) { ++dropped_; return false; }
    slot_of_[g] = std::uint32_t(slots_.size());
    slots_.push_back(g);
    return true;
}

void DraftRows::begin(std::span<const std::uint32_t> prompt) {
    for (std::size_t s = core_; s < slots_.size(); ++s) slot_of_[slots_[s]] = kNone;
    if (slots_.size() < core_) {  // first call: lay out the core in order
        for (std::uint32_t g = 0; g < core_; ++g) add_group(g);
    } else {
        slots_.resize(core_);
    }
    materialised_ = std::min(materialised_, core_);
    dropped_ = 0;
    packs_chosen_ = 0;
    for (const auto id : config_.always) add_group(id / config_.group);
    for (std::size_t i = 0; i < config_.packs.size(); ++i) {
        // A pack joins when the prompt holds pack_min_hits of its ids outside the core (sorted copy: binary search).
        std::vector<std::uint32_t> pack = config_.packs[i];
        std::sort(pack.begin(), pack.end());
        std::uint32_t hits = 0;
        for (const auto id : prompt)
            if (id / config_.group >= core_ && std::binary_search(pack.begin(), pack.end(), id)) ++hits;
        if (hits < config_.pack_min_hits) continue;
        if (i < 32) packs_chosen_ |= 1u << i;
        for (const auto id : config_.packs[i]) add_group(id / config_.group);
    }
    if (packs_chosen_ == 0)
        for (std::uint32_t g = core_; g < std::min(config_.fallback_groups, groups_total_); ++g) add_group(g);
    for (const auto id : prompt)
        if (id < config_.vocab) add_group(id / config_.group);
}

std::uint32_t DraftRows::admit(std::span<const std::uint32_t> ids) {
    std::uint32_t added = 0;
    for (const auto id : ids)
        if (id < config_.vocab && add_group(id / config_.group)) ++added;
    return added;
}

std::uint32_t DraftRows::token(std::uint32_t row) const {
    if (row >= rows()) throw std::out_of_range("draft rows: compact row past the subset");
    return slots_[row / config_.group] * config_.group + row % config_.group;
}

std::string DraftRows::describe() const {
    char b[192];
    std::snprintf(b, sizeof b, "draft rows %u of %u (%.1f %% of the head, group %u, core %u groups), packs 0x%x, dropped %llu", rows(),
                  config_.vocab, 100.0 * double(rows()) / double(config_.vocab), config_.group, core_, packs_chosen_,
                  static_cast<unsigned long long>(dropped_));
    return b;
}

std::vector<std::uint32_t> DraftRows::read_ids(const std::string& path, std::uint32_t vocab) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("draft rows: cannot open " + path);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() % 4 != 0) throw std::runtime_error("draft rows: " + path + " is not a uint32 id list");
    std::vector<std::uint32_t> ids(bytes.size() / 4);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = std::uint32_t(bytes[4 * i]) | std::uint32_t(bytes[4 * i + 1]) << 8 | std::uint32_t(bytes[4 * i + 2]) << 16 |
                 std::uint32_t(bytes[4 * i + 3]) << 24;
        if (ids[i] >= vocab) throw std::runtime_error("draft rows: " + path + " holds an id past the vocabulary");
    }
    return ids;
}

} // namespace strix::spec
