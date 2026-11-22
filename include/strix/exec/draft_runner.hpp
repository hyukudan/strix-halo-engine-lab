#pragma once

#include "strix/exec/compiler.hpp"
#include "strix/exec/descriptor.hpp"
#include "strix/spec/block_parallel.hpp"
#include "strix/spec/draft_decl.hpp"
#include "strix/state/txn.hpp"

#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace strix::exec {

class CpuExecutor;

class DraftRunner final : public spec::ChainDrafter, public spec::BlockForward {
public:
    // One serialized lane. The target and draft programs can use the same
    // executor. The arena must be the executor's workspace and outlive the
    // runner. The compiler is bound to the declared graph and its live-ins,
    // including read-only target taps and state where applicable.
    struct CpuBinding {
        Compiler& compiler;
        CpuExecutor& executor;
        std::span<const std::byte> arena;
        ProgramKey key;  // The runner sets phase and width. graph_id must be nonzero.
        // Dedicated draft state. Only the runner's transaction writes it.
        state::CpuState* state = nullptr;
        std::uint32_t slot = 0;
        // Project the arena epilogue for one row. Views are consumed before
        // the next submission. Draft and state-only bindings must produce
        // equivalent state images for verified inputs that can be reused.
        std::function<void(const Program&, std::span<const std::byte>, std::uint32_t,
                           std::span<state::Row>)> state_rows{};
        std::function<float(const Program&, std::span<const std::byte>, std::uint32_t)> confidence{};
    };

    DraftRunner(spec::DraftDecl decl, CpuBinding binding);
    // Weightless context lookup uses the shared n-gram index over the prompt
    // and subsequent verified tokens.
    explicit DraftRunner(spec::DraftDecl decl, const spec::ProposerConfig& config = {});

    [[nodiscard]] const spec::DraftDecl& declaration() const noexcept { return decl_; }

    // Begin at the unconsumed anchor. A stateful binding's published frontier
    // must already equal position. Lookup learns its initial history via accept().
    void start(std::uint32_t anchor, std::uint64_t position);
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k,
               std::vector<std::uint32_t>& out) override;
    void forward(const spec::BlockRequest&, std::vector<std::uint32_t>& proposals,
                 std::vector<float>& confidence) override;
    // The last emitted token becomes the next unconsumed anchor. Previously
    // staged inputs can be published only if they match the verified stream.
    void accept(std::span<const std::uint32_t> emitted) override;
    // Discard staged graph state and keep the current anchor for a retry.
    void abort() noexcept;

private:
    void chain_draft(std::span<const std::uint32_t> prefix, std::uint32_t k,
                     std::vector<std::uint32_t>& out);
    [[nodiscard]] const Program& run(Phase phase, std::span<const StepRow> rows,
                                     std::span<RowOut> picks);
    void append(const Program& program);

    spec::DraftDecl decl_;
    std::optional<CpuBinding> cpu_;
    spec::BlockPolicy block_{};
    std::optional<spec::ContextProposer> lookup_;
    std::optional<std::uint32_t> anchor_;
    std::uint32_t position_ = 0;
    std::vector<state::Row> epilogues_;
    std::vector<std::uint32_t> staged_inputs_;
    std::optional<state::Txn> pending_;
};

}  // namespace strix::exec
