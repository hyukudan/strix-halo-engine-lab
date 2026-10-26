// Copy drafter (see minimax_ngram.hpp).
#include "minimax_ngram.hpp"

#include <algorithm>

namespace strix::minimax::dspark {

std::uint64_t CopyDrafter::key_at(std::size_t end) const {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = end - anchor_; i < end; ++i) h = (h ^ history_[i]) * 1099511628211ULL;
    return h;
}

void CopyDrafter::append(std::span<const std::uint32_t> tokens) {
    for (const std::uint32_t t : tokens) {
        // Index the window that ends at the current end BEFORE this token joins: its continuation is `t`.
        if (history_.size() >= 3) {
            const std::size_t n = history_.size();
            ends3_[(static_cast<std::uint64_t>(history_[n - 3]) << 42U) ^ (static_cast<std::uint64_t>(history_[n - 2]) << 21U) ^ history_[n - 1]]
                .push_back(static_cast<std::uint32_t>(n));
        }
        if (history_.size() >= anchor_) {
            auto& v = ends_[key_at(history_.size())];
            v.push_back(static_cast<std::uint32_t>(history_.size()));
            if (v.size() > max_occ_) v.erase(v.begin());
        }
        history_.push_back(t);
    }
}

std::vector<std::uint32_t> CopyDrafter::propose(std::uint32_t k) const {
    std::vector<std::uint32_t> out;
    const std::size_t n = history_.size();
    if (n < anchor_ || k == 0) return out;
    const auto it = ends_.find(key_at(n));
    if (it == ends_.end()) return out;
    // Occurrences whose window really equals the current suffix (hash collisions excluded), most recent first.
    std::vector<std::uint32_t> occ;
    for (auto p = it->second.rbegin(); p != it->second.rend(); ++p)
        if (std::equal(history_.begin() + (*p - anchor_), history_.begin() + *p, history_.end() - anchor_)) occ.push_back(*p);
    if (occ.empty()) return out;
    for (std::uint32_t i = 0; i < k; ++i) {
        const std::size_t first = occ[0] + i;
        if (first >= n) break;  // the most recent copy runs into the present
        const std::uint32_t t = history_[first];
        bool agree = true;
        for (std::size_t j = 1; j < occ.size() && agree; ++j) agree = occ[j] + i >= n || history_[occ[j] + i] == t;
        if (!agree) break;
        out.push_back(t);
    }
    return out;
}

std::vector<std::uint32_t> CopyDrafter::continuation(std::span<const std::uint32_t> extra, std::uint32_t k,
                                                     std::uint32_t min_match) const {
    std::vector<std::uint32_t> out;
    const std::size_t h = history_.size(), n = h + extra.size();
    if (n < 3 || k == 0) return out;
    const auto at = [&](std::size_t i) { return i < h ? history_[i] : (i < n ? extra[i - h] : out[i - n]); };
    const std::uint64_t key = (static_cast<std::uint64_t>(at(n - 3)) << 42U) ^ (static_cast<std::uint64_t>(at(n - 2)) << 21U) ^ at(n - 1);
    const auto it = ends3_.find(key);
    if (it == ends3_.end()) return out;
    std::size_t best = 0, best_len = 0;
    for (const std::uint32_t p : it->second) {  // ascending positions: a later one wins ties
        if (p >= n) continue;                   // the present itself (extra empty: never indexed yet)
        std::size_t m = 0;
        while (m < p && m < 256 && at(p - 1 - m) == at(n - 1 - m)) ++m;
        if (m >= min_match && m >= best_len) {
            best_len = m;
            best = p;
        }
    }
    if (best_len == 0) return out;
    for (std::uint32_t i = 0; i < k; ++i) out.push_back(at(best + i));  // best + i < n + i: reads through `out` when needed
    return out;
}

} // namespace strix::minimax::dspark
