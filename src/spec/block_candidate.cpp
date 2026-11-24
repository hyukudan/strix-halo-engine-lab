#include "strix/spec/block_candidate.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix::spec {
BlockPolicy candidate_block_policy(const weight::BlockDraftConfig& c) {
    if (c.target_layers.size() > max_draft_layers || c.layers > max_draft_layers)
        throw std::invalid_argument("candidate block: policy layer limit");
    BlockPolicy p;
    p.block = c.block; p.fill_token = c.mask; p.ring = c.window;
    p.draft_layers = c.layers; p.target_layer_count = static_cast<std::uint32_t>(c.target_layers.size());
    for (std::size_t i = 0; i < c.target_layers.size(); ++i) {
        if (c.target_layers[i] > std::numeric_limits<std::uint16_t>::max())
            throw std::invalid_argument("candidate block: capture id limit");
        p.target_layers[i] = static_cast<std::uint16_t>(c.target_layers[i]);
    }
    validate(p);
    return p;
}
CandidateBlockForward::CandidateBlockForward(weight::BlockDraftConfig c, reference::Weights w, CandidateTargetHooks hooks)
    : config_(std::move(c)), weights_(std::move(w)), hooks_(std::move(hooks)) {
    static_cast<void>(candidate_block_policy(config_));
    if (!weights_ || !hooks_.features || !hooks_.embed || !hooks_.raw_logits || !hooks_.accepted)
        throw std::invalid_argument("candidate block: incomplete target hooks");
}
void CandidateBlockForward::forward(const BlockRequest& req, std::vector<std::uint32_t>& out, std::vector<float>& confidence) {
    if (req.prefix != 0 || req.ids.empty() || req.ids.size() > config_.block || req.k == 0 || req.k >= req.ids.size())
        throw std::invalid_argument("candidate block: invalid block request");
    for (std::size_t i = 0; i < req.ids.size(); ++i)
        if (req.ids[i] >= config_.vocab || (i > 0 && req.ids[i] != config_.mask))
            throw std::invalid_argument("candidate block: expected anchor followed by mask tokens");
    const auto features = hooks_.features();
    if (features.first_position > req.anchor_pos || features.rows.rows != req.anchor_pos - features.first_position)
        throw std::invalid_argument("candidate block: feature frontier differs from anchor");
    if (features.rows.rows < std::min<std::uint64_t>(req.anchor_pos, config_.window - 1U))
        throw std::invalid_argument("candidate block: incomplete sliding context");
    const auto noise = hooks_.embed(req.ids);
    if (noise.rows != req.ids.size())
        throw std::invalid_argument("candidate block: embedding row count differs from request");
    const auto all = reference::forward(config_, weights_, features.rows, noise, features.first_position);
    reference::Matrix hidden(req.k, config_.hidden);
    std::copy_n(all.data.begin() + config_.hidden, hidden.data.size(), hidden.data.begin());
    const auto logits = hooks_.raw_logits(hidden);
    auto selected = reference::select(config_, weights_, hidden, logits, req.ids.front());
    out = std::move(selected.path);
    // A greedy point mass is NOT an estimate of target acceptance. Leave empty.
    confidence.clear();
}
void CandidateBlockForward::accept(std::span<const std::uint32_t> emitted) { hooks_.accepted(emitted); }
} // namespace strix::spec
