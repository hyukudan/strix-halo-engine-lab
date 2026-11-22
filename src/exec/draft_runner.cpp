#include "strix/exec/draft_runner.hpp"

#include "executor_cpu.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix::exec {
namespace {

std::uint32_t end_position(std::uint64_t position, std::size_t rows) {
    constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
    if (position > limit || rows > limit - position)
        throw std::out_of_range("draft window exceeds X3 positions");
    return static_cast<std::uint32_t>(position + rows);
}

std::uint32_t picked(const RowOut& row) {
    if ((row.status != RowStatus::picked && row.status != RowStatus::stop) || row.id == spec::kNoToken)
        throw std::runtime_error("draft program did not emit a token");
    return row.id;
}

}  // namespace

DraftRunner::DraftRunner(spec::DraftDecl decl, CpuBinding binding)
    : decl_(decl), cpu_(std::move(binding)) {
    const auto& cpu = *cpu_;
    if (decl_.id.empty() || decl_.graph.empty() || cpu.key.graph_id == 0 || cpu.key.tree || cpu.key.sampled ||
        static_cast<bool>(cpu.state_rows) != (cpu.state != nullptr) ||
        static_cast<bool>(cpu.confidence) != !decl_.confidence.empty())
        throw std::invalid_argument("invalid draft graph binding");

    switch (decl_.kind) {
    case spec::DraftKind::chain:
    case spec::DraftKind::model:
        if (!decl_.max_k || decl_.block || cpu.confidence)
            throw std::invalid_argument("invalid chain draft declaration");
        break;
    case spec::DraftKind::block:
        if (decl_.block < 2 || decl_.block > kStepMaxRows)
            throw std::invalid_argument("invalid draft block width");
        block_.block = decl_.block;
        block_.fill_token = decl_.noise_token;
        block_.mask = decl_.attention == spec::DraftAttention::block_noncausal
            ? spec::BlockMask::bidirectional : spec::BlockMask::causal;
        block_.anchor_proposes = decl_.anchor_proposes;
        block_.prefix_in_block = decl_.prefix_in_block;
        if (!decl_.max_k) decl_.max_k = spec::max_drafts(block_);
        if (decl_.max_k > spec::max_drafts(block_))
            throw std::invalid_argument("draft depth exceeds the declared block");
        break;
    default:
        throw std::invalid_argument("draft kind has no CPU graph binding");
    }

    if (cpu.state) {
        (void)cpu.state->entry(0, cpu.slot);
        epilogues_.resize(cpu.state->plan().instances.size());
    }
}

DraftRunner::DraftRunner(spec::DraftDecl decl, const spec::ProposerConfig& config) : decl_(decl) {
    if (decl_.kind != spec::DraftKind::lookup || decl_.id.empty() || !decl_.graph.empty() ||
        std::ranges::any_of(decl_.sources, [](spec::LookupSource source) {
            return source != spec::LookupSource::context && source != spec::LookupSource::ngram;
        }))
        throw std::invalid_argument("invalid context lookup draft declaration");
    if (!decl_.max_k) decl_.max_k = spec::max_proposal;
    lookup_.emplace(config);
}

void DraftRunner::start(std::uint32_t anchor, std::uint64_t position) {
    if (!cpu_ || pending_) throw std::logic_error("draft runner cannot set a new anchor");
    const auto start_pos = end_position(position, 0);
    if (cpu_->state) {
        const auto& lane = cpu_->state->entry(0, cpu_->slot);
        if (lane.pending || lane.frontier != start_pos)
            throw std::invalid_argument("draft anchor disagrees with the published state frontier");
    }
    position_ = start_pos;
    anchor_ = anchor;
}

const Program& DraftRunner::run(Phase phase, std::span<const StepRow> rows, std::span<RowOut> picks) {
    auto key = cpu_->key;
    key.phase = phase;
    key.rows = static_cast<std::uint32_t>(rows.size());
    const auto& program = cpu_->compiler.get(key);
    cpu_->executor.wait(cpu_->executor.submit(program, {rows}), picks);
    return program;
}

void DraftRunner::append(const Program& program) {
    std::ranges::fill(epilogues_, state::Row{});
    cpu_->state_rows(program, cpu_->arena, 0, epilogues_);
    pending_->append(epilogues_);
}

void DraftRunner::draft(std::span<const std::uint32_t> prefix, std::uint32_t k,
                        std::vector<std::uint32_t>& out) {
    out.clear();
    k = std::min(k, decl_.max_k);
    if (lookup_) {
        out = lookup_->propose(prefix, k).tokens;
        return;
    }
    if (!anchor_ || pending_) throw std::logic_error("draft runner needs an idle anchor");
    if (!k) return;

    if (decl_.kind == spec::DraftKind::block) {
        if (prefix.size() >= block_.block) throw std::invalid_argument("prefix exceeds the draft block");
        const auto count = static_cast<std::uint32_t>(prefix.size());
        k = std::min(k, spec::max_drafts(block_, count));
        if (!k) return;
        std::vector<std::uint32_t> ids;
        std::vector<float> confidence;
        spec::block_ids(block_, *anchor_, prefix, k, ids);
        forward({ids, position_, count, k}, out, confidence);
        return;
    }

    chain_draft(prefix, k, out);
}

void DraftRunner::chain_draft(std::span<const std::uint32_t> prefix, std::uint32_t k,
                              std::vector<std::uint32_t>& out) {
    const auto after_prefix = end_position(position_, prefix.size());
    const auto end = end_position(after_prefix, k);
    try {
        out.reserve(k);
        if (cpu_->state) {
            pending_.emplace(*cpu_->state, cpu_->slot);
            pending_->begin(end - position_);
            staged_inputs_.reserve(end - position_);
        }
        auto token = *anchor_;
        for (auto pos = position_; pos < end; ++pos) {
            const bool propose = pos >= after_prefix;
            const StepRow row{token, pos, cpu_->slot, kStepNoParent, 0,
                              step_valid | (propose ? step_sample : 0U) | (pending_ ? step_pending_state : 0U), 0, 0};
            RowOut result{};
            const auto& program = run(Phase::draft, {&row, 1}, {&result, 1});
            if (pending_) {
                append(program);
                staged_inputs_.push_back(token);
            }
            if (propose) {
                token = picked(result);
                out.push_back(token);
            } else {
                token = prefix[pos - position_];
            }
        }
    } catch (...) {
        abort();
        out.clear();
        throw;
    }
}

void DraftRunner::forward(const spec::BlockRequest& req, std::vector<std::uint32_t>& proposals,
                          std::vector<float>& confidence) {
    proposals.clear();
    confidence.clear();
    if (decl_.kind != spec::DraftKind::block || !req.k || req.k > decl_.max_k || req.ids.empty() ||
        req.k > spec::max_drafts(block_, req.prefix) ||
        req.ids.size() != spec::rows_needed(block_, req.prefix, req.k) ||
        !std::ranges::all_of(req.ids.subspan(req.prefix + 1U),
                             [this](std::uint32_t id) { return id == block_.fill_token; }))
        throw std::invalid_argument("invalid draft block request");
    (void)end_position(req.anchor_pos, req.ids.size());
    if (!anchor_) start(req.ids.front(), req.anchor_pos);
    if (req.anchor_pos != position_ || req.ids.front() != *anchor_)
        throw std::invalid_argument("draft block request has the wrong anchor");

    std::array<StepRow, kStepMaxRows> rows{};
    std::array<RowOut, kStepMaxRows> picks{};
    const auto count = static_cast<std::uint32_t>(req.ids.size());
    const auto first = spec::proposal_row(block_, req.prefix);
    const auto full_mask = ~std::uint32_t{0} >> (kStepMaxRows - count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto bit = std::uint32_t{1} << i;
        const auto visible = block_.mask == spec::BlockMask::bidirectional ? full_mask ^ bit : bit - 1U;
        const bool propose = i >= first && i - first < req.k;
        rows[i] = {req.ids[i], position_ + i, cpu_->slot, i ? i - 1U : kStepNoParent, visible,
                   step_valid | (propose ? step_sample : 0U), 0, 0};
    }

    std::vector<std::uint32_t> tokens;
    std::vector<float> scores;
    tokens.reserve(req.k);
    if (cpu_->confidence) scores.reserve(req.k);
    const auto& program = run(Phase::draft, std::span(rows).first(count), std::span(picks).first(count));
    for (auto i = first; i < first + req.k; ++i) {
        tokens.push_back(picked(picks[i]));
        if (cpu_->confidence) scores.push_back(cpu_->confidence(program, cpu_->arena, i));
    }
    proposals = std::move(tokens);
    confidence = std::move(scores);
}

void DraftRunner::accept(std::span<const std::uint32_t> emitted) {
    if (lookup_) {
        lookup_->commit(emitted);
        return;
    }
    if (emitted.empty()) {
        abort();
        return;
    }
    if (!anchor_) throw std::logic_error("draft runner has no anchor to accept");

    try {
        const auto next = end_position(position_, emitted.size());
        const auto count = next - position_;
        if (cpu_->state) {
            if (!(pending_ && count <= staged_inputs_.size() &&
                  std::ranges::equal(std::span(staged_inputs_).subspan(1, count - 1U),
                                     emitted.first(count - 1U)))) {
                // Replay known inputs if the block used noise, a copy prefix
                // failed, or verification consumed beyond the staged inputs.
                // Keep the old image published until the whole replay succeeds.
                abort();
                pending_.emplace(*cpu_->state, cpu_->slot);
                pending_->begin(count);
                auto token = *anchor_;
                for (std::uint32_t i = 0; i < count; ++i) {
                    const StepRow row{token, position_ + i, cpu_->slot, kStepNoParent, 0,
                                      step_valid | step_pending_state, 0, 0};
                    RowOut ignored{};
                    append(run(Phase::state_only, {&row, 1}, {&ignored, 1}));
                    token = emitted[i];
                }
            }
            pending_->commit(count);
        }
        position_ = next;
        anchor_ = emitted.back();
        abort();
    } catch (...) {
        abort();
        throw;
    }
}

void DraftRunner::abort() noexcept {
    pending_.reset();
    staged_inputs_.clear();
}

}  // namespace strix::exec
