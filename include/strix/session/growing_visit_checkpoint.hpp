#pragma once

#include "strix/session/kv_row_append.hpp"

namespace strix::session {

// Undo one row of a CPU reference using GrowingVisitCache. Save the recurrent
// histories and the single physical KV slot a row can replace. Dense visits
// grow by a row, so shrinking them hides the discarded suffix.
template <class Traits, class Element>
class GrowingVisitRowCheckpoint {
public:
    using Plan = VisitCachePlan<Traits>;
    using Cache = GrowingVisitCache<Element, Plan::Layout::stream_count>;
    using Storage = std::array<Cache, Plan::visits().size()>;

    void save(const Plan& plan, unsigned position, const Storage& storage) {
        for (const auto visit : Plan::visits()) {
            const auto& layout = plan.visit(visit);
            const auto index = Traits::index(visit);
            const auto& source = storage[index];
            auto& saved = visits_[index];
            saved.key_size = source.keys.size();
            saved.value_size = source.values.size();
            saved.history = source.history;
            saved.offset = attn::cache_rows::row_offset(layout.kv, position);
            if (layout.kv.ring != 0U && !source.keys.empty()) {
                const auto keys = std::span(source.keys).subspan(saved.offset, layout.kv.width);
                const auto values = std::span(source.values).subspan(saved.offset, layout.kv.width);
                saved.keys.assign(keys.begin(), keys.end());
                saved.values.assign(values.begin(), values.end());
            } else {
                saved.keys.clear();
                saved.values.clear();
            }
        }
    }

    void restore(Storage& storage) const {
        for (std::size_t i = 0; i < storage.size(); ++i) {
            auto& target = storage[i];
            const auto& saved = visits_[i];
            target.keys.resize(saved.key_size);
            target.values.resize(saved.value_size);
            if (!saved.keys.empty()) {
                std::ranges::copy(saved.keys, std::span(target.keys).subspan(saved.offset).begin());
                std::ranges::copy(saved.values, std::span(target.values).subspan(saved.offset).begin());
            }
            target.history = saved.history;
        }
    }

private:
    struct Visit {
        std::size_t key_size{}, value_size{}, offset{};
        std::vector<Element> keys, values;
        std::array<std::vector<float>, Plan::Layout::stream_count> history;
    };
    std::array<Visit, Plan::visits().size()> visits_;
};

// The exact CPU reference can evaluate a window by stepping its existing row
// arithmetic. Restore rejected rows in reverse order, retaining the accepted
// histories and ring slots without re-evaluating the accepted tokens.
template <class Traits, class Element>
class GrowingVisitWindow {
public:
    using Checkpoint = GrowingVisitRowCheckpoint<Traits, Element>;
    using Plan = typename Checkpoint::Plan;
    using Storage = typename Checkpoint::Storage;

    template <class Step>
    void verify(const Plan& plan, Storage& storage, unsigned frontier,
                std::span<const std::uint32_t> inputs, std::vector<std::uint32_t>& greedy, Step step) {
        if (pending_ != 0U || !plan.admits_sequence(frontier, inputs.size()))
            throw std::invalid_argument("invalid CPU verify window");
        greedy.resize(inputs.size());
        if (inputs.size() > 1U && checkpoints_.size() < inputs.size())
            checkpoints_.resize(inputs.size());
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            // The anchor always commits. Only its successors may need undo.
            if (row != 0U) checkpoints_[row].save(plan, frontier + static_cast<unsigned>(row), storage);
            greedy[row] = step(inputs[row]);
        }
        pending_ = static_cast<unsigned>(inputs.size());
    }

    unsigned commit(Storage& storage, unsigned keep) {
        if (keep == 0U || keep > pending_)
            throw std::invalid_argument("invalid CPU verify prefix");
        const auto rejected = pending_ - keep;
        while (pending_ > keep) checkpoints_[--pending_].restore(storage);
        pending_ = 0;
        return rejected;
    }

    [[nodiscard]] bool pending() const { return pending_ != 0U; }

private:
    std::vector<Checkpoint> checkpoints_;
    unsigned pending_{};
};

}  // namespace strix::session
