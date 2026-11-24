#include "strix/weight/calib_provenance.hpp"

#include "h3/te/te_sha256.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace strix::weight::calib {
namespace {

void hash_ids(h3::te::Sha256& hash, std::span<const std::uint32_t> ids) {
    std::array<std::uint8_t, 4> le{};
    for (const auto id : ids) {
        for (unsigned i = 0; i < 4U; ++i) le[i] = static_cast<std::uint8_t>((id >> (8U * i)) & 0xffU);
        hash.update(le.data(), le.size());
    }
}

std::string hex_prefix(const Digest256& d) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string s;
    for (std::size_t i = 0; i < 8U; ++i) {
        s.push_back(digits[d.bytes[i] >> 4U]);
        s.push_back(digits[d.bytes[i] & 15U]);
    }
    return s;
}

}  // namespace

Digest256 window_identity(std::span<const std::uint32_t> ids) {
    h3::te::Sha256 hash;
    hash_ids(hash, ids);
    return Digest256{hash.finish()};
}

std::uint64_t shingle_hash(std::span<const std::uint32_t, kShingleTokens> ids) {
    h3::te::Sha256 hash;
    hash_ids(hash, ids);
    const auto d = hash.finish();
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8U; ++i) v |= std::uint64_t{d[i]} << (8U * i);
    return v;
}

void Provenance::add_window(std::span<const std::uint32_t> ids) {
    if (ids.empty()) throw std::invalid_argument("calibration window without ids");
    windows_.push_back({window_identity(ids), ids.size()});
    const auto before = shingles_.size();
    for (std::size_t at = 0; at + kShingleTokens <= ids.size(); ++at)
        shingles_.push_back(shingle_hash(ids.subspan(at).first<kShingleTokens>()));
    std::sort(shingles_.begin() + static_cast<std::ptrdiff_t>(before), shingles_.end());
    std::inplace_merge(shingles_.begin(), shingles_.begin() + static_cast<std::ptrdiff_t>(before), shingles_.end());
    shingles_.erase(std::unique(shingles_.begin(), shingles_.end()), shingles_.end());
}

void Provenance::merge(const Provenance& other) {
    windows_.insert(windows_.end(), other.windows_.begin(), other.windows_.end());
    std::vector<std::uint64_t> all;
    all.reserve(shingles_.size() + other.shingles_.size());
    std::set_union(shingles_.begin(), shingles_.end(), other.shingles_.begin(), other.shingles_.end(), std::back_inserter(all));
    shingles_ = std::move(all);
}

Provenance Provenance::from_parts(std::vector<WindowId> windows, std::vector<std::uint64_t> shingles) {
    Provenance p;
    p.windows_ = std::move(windows);
    std::sort(shingles.begin(), shingles.end());
    shingles.erase(std::unique(shingles.begin(), shingles.end()), shingles.end());
    p.shingles_ = std::move(shingles);
    return p;
}

void Provenance::add_manifest(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read gate manifest " + path);
    const auto dir = std::filesystem::path(path).parent_path();
    std::string line;
    std::size_t cases = 0;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string name, ids_path;
        std::uint64_t tokens = 0;
        if (!(fields >> name) || name.starts_with('#')) continue;
        if (!(fields >> tokens >> ids_path)) throw std::runtime_error(path + ": manifest row needs NAME TOKENS IDS_PATH");
        const std::filesystem::path p(ids_path);
        std::ifstream ids_in(p.is_absolute() ? p : dir / p);
        if (!ids_in) throw std::runtime_error(path + ": cannot read the ids of " + name);
        std::vector<std::uint32_t> ids;
        std::uint64_t id = 0;
        while (ids_in >> id) {
            if (id > 0xffffffffULL) throw std::runtime_error(path + ": id out of range in " + name);
            ids.push_back(static_cast<std::uint32_t>(id));
        }
        if (!ids_in.eof() || ids.size() != tokens)
            throw std::runtime_error(path + ": " + name + " has " + std::to_string(ids.size()) + " ids, the manifest says " +
                                     std::to_string(tokens));
        add_window(ids);
        ++cases;
    }
    if (cases == 0U) throw std::runtime_error("gate manifest " + path + " lists no case");
}

std::string overlap(const Provenance& a, const Provenance& b) {
    for (const auto& x : a.windows())
        for (const auto& y : b.windows())
            if (x.digest.bytes == y.digest.bytes) return "window " + hex_prefix(x.digest) + " (" + std::to_string(x.tokens) + " ids)";
    std::size_t shared = 0;
    auto i = a.shingles().begin();
    auto j = b.shingles().begin();
    while (i != a.shingles().end() && j != b.shingles().end()) {
        if (*i < *j) ++i;
        else if (*j < *i) ++j;
        else {
            ++shared;
            ++i;
            ++j;
        }
    }
    if (shared == 0U) return {};
    return std::to_string(shared) + " runs of " + std::to_string(kShingleTokens) + " ids";
}

}  // namespace strix::weight::calib
