#pragma once
// Shared VerifyTarget over any engine with the window contract (CUDA CudaEngine, HIP HipEngine):
//   forward_rows(tokens, rows, LogitRows) -> Out; read_top(Out, rows, host) (synchronizes); commit(k); sync().
// Greedy verification keeps read_top. Sampled verification optionally reads [rows][vocab()] host floats through
// read_logits_rows(Out, rows, float*). Both reads synchronize. commit(k) = engine commit + sync.
// Home: shared (host spec layer): no device types; each backend binds it with its own engine type.
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
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
    void set_first_position(std::uint64_t position) override { first_position_ = position; }

    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        const auto first = verify_position();
        const auto out = e_.forward_rows(tokens.data(), rows, Engine::LogitRows::all);
        read_tokens(out, rows, first, {}, greedy);
        if (observer_) observer_(out, rows);
    }
    void commit(std::uint32_t k) override {
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

    void read_tokens(const Rows& out, std::uint32_t rows, std::uint64_t first,
                     std::span<const std::uint32_t> depth, std::vector<std::uint32_t>& greedy) {
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
};

}  // namespace strix::spec
