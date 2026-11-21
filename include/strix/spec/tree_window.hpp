#pragma once

// Engine window of a token tree (strix/spec/block_drafter.hpp TreeRows): the backend-neutral contract a target's kernels
// follow so that every verify row gets the bits a LINEAR window of its own path gives (EXACT, ids == serial).
//
// TreeRows come in depth order with each leaf sibling placed right before the chain node of its depth, so the parent of
// every row is the last CHAIN row before it. A target whose kernels walk the rows in order then needs three rules:
//   - position: row r is at position pos0 + pos[r] (its depth); RoPE and the attention length use it;
//   - K/V index: chain rows store K/V at pos0 + pos[r] (their final place, as a linear window would), leaf i stores at
//     pos0 + chain + i (after the chain rows, so no chain row is overwritten); a row reads key pos0 + pos[r] (its own)
//     from pos0 + index[r] and every earlier key from its natural index: those are exactly its ancestors (chain rows);
//   - recurrent state (conv taps, delta-rule state): a leaf row computes its output from the current state and does
//     NOT advance it (the next row's parent is the last chain row, whose state is the current one).
// Commit of an accepted root path (path[0] = 0, then chain rows, optionally ending on a leaf) = replay rows [0, end) with
// the leaves not on the path skipped (or, when the path is every chain row of the window, keep the window's own state),
// and move the K/V of a final leaf from its index to its position. Rejected leaves are never committed.
// Home: shared (host spec layer); the CUDA core implements it (RowMap::self, gdn_window leaf mask, gdn_commit_path,
// kv_move_row), HIP lists the same kernel bodies.

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "strix/spec/block_drafter.hpp"

namespace strix::spec {

inline constexpr std::uint32_t kTreeWindowMaxRows = 32;  // one 32-bit leaf mask

struct TreeWindow {
    std::uint32_t rows = 0;
    std::uint32_t chain = 0;  // chain rows (anchor + main drafts)
    std::uint32_t leaf = 0;   // bit r: row r is a leaf sibling
    std::array<std::uint32_t, kTreeWindowMaxRows> pos{};    // position offset of row r (its depth)
    std::array<std::uint32_t, kTreeWindowMaxRows> index{};  // K/V index offset of row r
};

// Throws std::invalid_argument unless `t` is a leaf-sibling tree in the layout above with 1..32 rows.
inline TreeWindow tree_window(const TreeRows& t) {
    const std::size_t n = t.tokens.size();
    if (n == 0 || n > kTreeWindowMaxRows || t.parent.size() != n || t.depth.size() != n || t.node.size() != n)
        throw std::invalid_argument("tree_window: 1..32 rows with parent/depth/node per row");
    if (t.parent[0] != -1 || t.depth[0] != 0 || t.node[0] != 0) throw std::invalid_argument("tree_window: row 0 is not the anchor");
    TreeWindow w;
    w.rows = static_cast<std::uint32_t>(n);
    std::uint32_t last_chain = 0, chain = 1;
    for (std::uint32_t r = 1; r < w.rows; ++r) {
        if (t.parent[r] != static_cast<std::int32_t>(last_chain))
            throw std::invalid_argument("tree_window: a row's parent is not the last chain row (subtrees are not supported)");
        if (t.depth[r] != t.depth[last_chain] + 1) throw std::invalid_argument("tree_window: depth != parent depth + 1");
        w.pos[r] = t.depth[r];
        if (t.node[r] < 0) {
            w.leaf |= 1u << r;
        } else {
            if (t.node[r] != static_cast<std::int32_t>(t.depth[r])) throw std::invalid_argument("tree_window: chain node != depth");
            last_chain = r;
            ++chain;
        }
    }
    w.chain = chain;
    std::uint32_t li = 0;
    for (std::uint32_t r = 0; r < w.rows; ++r) w.index[r] = (w.leaf >> r & 1u) ? chain + li++ : w.pos[r];
    return w;
}

struct PathCommit {
    std::uint32_t accepted = 0;  // positions committed (path length)
    std::uint32_t end = 0;       // recurrent replay covers rows [0, end) ...
    std::uint32_t skip = 0;      // ... except these rows (leaves not on the path)
    bool replay = false;         // false: the window's state after its chain rows is the path's state (no replay)
    bool move = false;           // the path ends on a leaf: K/V index offset `from` moves to position offset `to`
    std::uint32_t from = 0, to = 0;
};

// Throws std::invalid_argument unless `path` is a root path of `w` (anchor, consecutive chain rows, optionally a final leaf).
inline PathCommit path_commit(const TreeWindow& w, std::span<const std::uint32_t> path) {
    if (path.empty() || path[0] != 0 || path.size() > w.rows) throw std::invalid_argument("path_commit: path must start at row 0");
    const std::uint32_t last = path.back();
    if (last >= w.rows) throw std::invalid_argument("path_commit: row outside the window");
    PathCommit c;
    c.accepted = static_cast<std::uint32_t>(path.size());
    c.end = last + 1;
    c.skip = w.leaf & ((1u << last) - 1u);  // leaves before the last row (last <= 31)
    std::size_t i = 0;  // the path must be exactly the rows of [0, end) not skipped
    for (std::uint32_t r = 0; r < c.end; ++r) {
        if (c.skip >> r & 1u) continue;
        if (i >= path.size() || path[i] != r) throw std::invalid_argument("path_commit: not a root path of the window");
        if (w.pos[r] != i) throw std::invalid_argument("path_commit: path depth mismatch");
        ++i;
    }
    if (i != path.size()) throw std::invalid_argument("path_commit: not a root path of the window");
    c.replay = !(c.end == w.rows && c.skip == w.leaf);
    if (w.leaf >> last & 1u) {
        c.move = true;
        c.from = w.index[last];
        c.to = w.pos[last];
    }
    return c;
}

}  // namespace strix::spec
