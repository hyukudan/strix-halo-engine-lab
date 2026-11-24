#pragma once
// Shared VerifyTarget over any engine with the window contract (CUDA CudaEngine, HIP HipEngine):
//   forward_rows(tokens, rows, LogitRows) -> Out; read_top(Out, rows, host) (synchronizes); commit(k); sync().
// Greedy verification keeps read_top. Sampled verification optionally reads [rows][vocab()] host floats through
// read_logits_rows(Out, rows, float*). Both reads synchronize. commit(k) = engine commit + sync.
// Optional: decode_committed_row(token, host_logits) -> next token, the engine's plain one-row decode step
// (committed at once; host_logits empty or [vocab()]). A one-row verify (a serial step of the speculator) then
// runs as that decode step and the following commit(1) has nothing left to do: a one-row window is always kept.
// Home: shared (host spec layer): no device types; each backend binds it with its own engine type.
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "strix/spec/block_drafter.hpp"

namespace strix::spec {

template <class Engine>
class EngineVerifyTarget final : public VerifyTarget {
    using Rows = typename Engine::Rows;
    static constexpr bool has_logits = requires(Engine& e, const Rows& out, std::uint32_t rows, float* host) {
        { e.vocab() } -> std::convertible_to<std::uint32_t>;
        e.read_logits_rows(out, rows, host);
    };
    static constexpr bool has_decode_row = requires(Engine& e, std::uint32_t token, std::span<float> logits) {
        { e.decode_committed_row(token, logits) } -> std::convertible_to<std::uint32_t>;
    };
    // An engine that verifies a tree supplies its geometry, forward and path commit together.
    static constexpr bool has_tree = requires(Engine& e, const TreeRows& tree, std::span<const std::uint32_t> path) {
        { e.tree_rows() } -> std::convertible_to<std::uint32_t>;
        { e.forward_tree_rows(tree, Engine::LogitRows::all) } -> std::same_as<Rows>;
        e.commit_path(path);
    };

public:
    explicit EngineVerifyTarget(Engine& e) : e_(e) {}
    // Optional observer of every verify pass (rows of [anchor, drafts...] + the engine's per-row outputs): the MTP chain
    // (strix/spec/mtp_chain.hpp) takes the hidden rows of the pass from it. Needs Engine::Rows.
    using Observer = std::function<void(const Rows&, std::uint32_t)>;
    void set_observer(Observer f) { observer_ = std::move(f); }
    void set_row_rule(const RowRule& rule) override { rule_ = rule; }
    [[nodiscard]] bool supports_row_rule(bool sampled) const override { return !sampled || has_logits; }
    [[nodiscard]] std::uint32_t admissible_verify_rows(std::uint32_t limit) const override {
        if constexpr (requires(const Engine& e) {
            { e.admissible_verify_rows(limit) } -> std::convertible_to<std::uint32_t>;
        }) {
            return std::min(limit, static_cast<std::uint32_t>(e_.admissible_verify_rows(limit)));
        } else {
            return limit;
        }
    }
    void set_first_position(std::uint64_t position) override { first_position_ = position; }

    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        const auto first = verify_position();
        row_committed_ = false;
        if constexpr (has_decode_row) {
            if (rows == 1U && !observer_) {
                greedy.resize(1);
                greedy[0] = decode_row(tokens[0], first);
                row_committed_ = true;
                return;
            }
        }
        const auto out = e_.forward_rows(tokens.data(), rows, Engine::LogitRows::all);
        read_tokens(out, rows, first, {}, greedy, tokens);
        if (observer_) observer_(out, rows);
    }
    void commit(std::uint32_t k) override {
        if (row_committed_) {
            if (k != 1U) throw std::logic_error("engine verify target: a one-row window commits one row");
            row_committed_ = false;
            return;
        }
        e_.commit(k);
        e_.sync();
    }
    [[nodiscard]] std::uint32_t tree_rows() const override {
        if constexpr (has_tree) return static_cast<std::uint32_t>(e_.tree_rows());
        else return 0;
    }
    void verify_tree(const TreeRows& tree, std::vector<std::uint32_t>& greedy) override {
        if constexpr (has_tree) {
            const auto rows = static_cast<std::uint32_t>(tree.tokens.size());
            if (tree.depth.size() != rows) throw std::logic_error("engine verify target: missing tree row depths");
            const auto first = verify_position();
            const auto out = e_.forward_tree_rows(tree, Engine::LogitRows::all);
            read_tokens(out, rows, first, tree.depth, greedy);
            if (observer_) observer_(out, rows);
        } else {
            VerifyTarget::verify_tree(tree, greedy);
        }
    }
    void commit_path(std::span<const std::uint32_t> path) override {
        if constexpr (has_tree) {
            e_.commit_path(path);
            e_.sync();
        } else {
            VerifyTarget::commit_path(path);
        }
    }
    Engine& engine() { return e_; }

private:
    [[nodiscard]] std::uint64_t verify_position() const {
        if (rule_.sampled()) {
            if constexpr (!has_logits) throw std::logic_error("engine verify target: sampled rows require host logits");
            if constexpr (requires(Engine& e) { { e.committed() } -> std::convertible_to<std::uint64_t>; }) {
                return static_cast<std::uint64_t>(e_.committed()) + 1;
            }
        }
        return first_position_;
    }

    [[nodiscard]] std::uint32_t decode_row(std::uint32_t token, std::uint64_t first) {
        if (!rule_.sampled()) return static_cast<std::uint32_t>(e_.decode_committed_row(token, std::span<float>{}));
        if constexpr (has_logits) {
            logits_.resize(static_cast<std::size_t>(e_.vocab()));
            (void)e_.decode_committed_row(token, std::span<float>(logits_));
            return rule_.row_token(logits_, first);
        } else {
            throw std::logic_error("engine verify target: sampled rows require host logits");
        }
    }

    // inputs (a linear window: anchor, drafts): row r's token is only compared with inputs[r + 1], and the first row
    // that differs ends the accepted prefix, so the rows after it are never read. A sampled row's draw is keyed by its
    // position (RowRule::row_token is a pure function of the row and the position), so the rows after the first
    // difference are not drawn: no emitted id changes, and a window pays the host sampler only for the rows it can
    // use. Those rows repeat the differing row's token. Tree windows (depth) and greedy reads are unchanged.
    void read_tokens(const Rows& out, std::uint32_t rows, std::uint64_t first, std::span<const std::uint32_t> depth,
                     std::vector<std::uint32_t>& greedy, std::span<const std::uint32_t> inputs = {}) {
        greedy.resize(rows);
        if constexpr (has_logits) {
            if (rule_.sampled()) {
                const auto vocab = static_cast<std::uint32_t>(e_.vocab());
                if (vocab == 0) throw std::logic_error("engine verify target: sampled rows require a non-empty vocabulary");
                logits_.resize(static_cast<std::size_t>(rows) * vocab);
                e_.read_logits_rows(out, rows, logits_.data());
                for (std::uint32_t r = 0; r < rows; ++r) {
                    const auto row = std::span<const float>(logits_).subspan(static_cast<std::size_t>(r) * vocab, vocab);
                    greedy[r] = rule_.row_token(row, first + (depth.empty() ? r : depth[r]));
                    if (depth.empty() && r + 1U < rows && r + 1U < inputs.size() && greedy[r] != inputs[r + 1U]) {
                        std::fill(greedy.begin() + static_cast<std::ptrdiff_t>(r) + 1, greedy.end(), greedy[r]);
                        break;
                    }
                }
                return;
            }
        }
        e_.read_top(out, rows, greedy.data());
    }

    Engine& e_;
    Observer observer_;
    RowRule rule_;
    std::uint64_t first_position_ = 1;
    std::vector<float> logits_;
    bool row_committed_ = false;  // the last verify was one row, already committed by decode_committed_row
};

}  // namespace strix::spec
