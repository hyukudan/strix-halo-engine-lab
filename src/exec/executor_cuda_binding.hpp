#pragma once

#include "strix/exec/program.hpp"

#include <utility>

namespace strix::exec::cuda_detail {

enum class GraphAction { eager, capture, rebind, replay };

// One reusable executable per map key. Cache the exact capture inputs so an
// arena epoch change with stable indirection can keep the executable as-is.
class GraphBinding {
public:
    [[nodiscard]] GraphAction select(const Program& program, std::uint64_t epoch) const {
        if (!program.capturable || program.key.tap) return GraphAction::eager;
        if (!source_ || program.key != key_) return GraphAction::capture;
        if (&program == source_ && program.identity == identity_ && epoch == epoch_)
            return ready_ ? GraphAction::replay : GraphAction::eager;
        if (!same_plan(program, epoch)) return GraphAction::capture;
        if (ready_) return program.args == args_ ? GraphAction::replay : GraphAction::rebind;
        return epoch != epoch_ || program.args != args_ ? GraphAction::capture : GraphAction::eager;
    }

    void remember(const Program& program, std::uint64_t epoch, bool ready) {
        if (!matches(program, epoch)) {
            auto args = program.args;
            auto launches = program.launches;
            args_ = std::move(args);
            launches_ = std::move(launches);
        }
        source_ = &program;
        key_ = program.key;
        identity_ = program.identity;
        epoch_ = epoch;
        ready_ = ready;
    }

    [[nodiscard]] bool matches(const Program& program, std::uint64_t epoch) const noexcept {
        return &program == source_ && program.identity == identity_ && epoch == epoch_;
    }

private:
    [[nodiscard]] bool same_plan(const Program& program, std::uint64_t epoch) const {
        // X1's identity includes the arena layout. Across an explicit layout
        // change, a matching kernel recipe can update the existing graph.
        if ((program.identity != identity_ && epoch == epoch_) || program.launches.size() != launches_.size())
            return false;
        for (std::size_t i = 0; i < launches_.size(); ++i) {
            const BoundLaunch& a = launches_[i];
            const BoundLaunch& b = program.launches[i];
            if (a.kernel != b.kernel || a.fn != b.fn || a.entry != b.entry || a.identity != b.identity ||
                a.chain_identity != b.chain_identity || a.after != b.after || a.stream != b.stream ||
                a.grid.blocks != b.grid.blocks || a.grid.threads != b.grid.threads ||
                a.grid.shared_bytes != b.grid.shared_bytes || a.args_offset != b.args_offset ||
                a.args_bytes != b.args_bytes || a.capturable != b.capturable) return false;
        }
        return true;
    }

    const Program* source_ = nullptr;
    ProgramKey key_{};
    std::uint64_t identity_ = 0;
    std::uint64_t epoch_ = 0;
    bool ready_ = false;
    std::vector<std::byte> args_;
    std::vector<BoundLaunch> launches_;
};

}  // namespace strix::exec::cuda_detail
