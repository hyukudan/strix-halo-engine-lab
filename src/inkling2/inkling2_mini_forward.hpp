#pragma once

#include "inkling2_mini_messages.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace strix::inkling2 {

inline std::string mini_layer_parameter(unsigned physical_layer, std::string_view suffix) {
    return "layers." + std::to_string(physical_layer) + "." + std::string(suffix);
}

inline void mini_round_bf16(std::span<float> row) {
    for (float& value : row) value = attn::group_indexer::bf16_round(value);
}

inline std::vector<float> mini_rms_norm(std::span<const float> input, std::span<const float> scale) {
    if (input.size() != kHiddenWidth || scale.size() != kHiddenWidth) {
        throw std::invalid_argument("invalid Inkling 2 Mini RMS norm row");
    }
    const float inverse = attn::group_pool::rms_inv(input.data(), kHiddenWidth, 1e-6F);
    std::vector<float> result(kHiddenWidth);
    for (unsigned i = 0; i < kHiddenWidth; ++i) result[i] = (input[i] * inverse) * scale[i];
    mini_round_bf16(result);
    return result;
}

// Backend operators consume one decoder row. Attention and convolution own
// persistent state keyed by the visit; residual banks are token-local.
template <class Backend>
class MiniForwardRow {
public:
    MiniForwardRow(Backend& backend, unsigned position, std::span<const float> embedding)
        : backend_(backend), position_(position), hidden_(embedding.begin(), embedding.end()) {
        if (embedding.size() != kHiddenWidth || position >= kMaxContext) {
            throw std::invalid_argument("invalid Inkling 2 Mini decoder input");
        }
    }

    std::vector<float> run() {
        MiniLayers::run(*this);
        const auto readout = residual_read(hidden_, "attn_res.output_query", 1);
        auto normalized = mini_rms_norm(readout, backend_.parameter("norm.weight", kHiddenWidth));
        capture_draft_tap(kLogicalVisits, normalized);
        return normalized;
    }

    void save_core_entry() {
        entry_hidden_ = hidden_;
        entry_blocks_ = blocks_;
    }

    void capture_previous_pass(unsigned) {
        boundary_.insert(boundary_.end(), hidden_.begin(), hidden_.end());
        constexpr unsigned query_size = 2U * MiniLayers::core_width * kHiddenWidth;
        messages_.emplace(boundary_, backend_.parameter("middle_cycle_cross_pass_queries", query_size),
                          backend_.parameter("middle_cycle_cross_pass_kv_queries", query_size));
    }

    void restore_core_entry() {
        hidden_ = entry_hidden_;
        blocks_ = entry_blocks_;
    }

    void run_layer(LayerVisit visit) {
        capture_draft_tap(visit.logical_layer, hidden_);
        const auto physical = visit.physical_layer;
        const bool full = is_full_attention(physical);
        if (captures_message_boundary(visit)) boundary_ = hidden_;

        const auto attention_inputs = physical == 0 ? hidden_ :
            residual_read(hidden_, "attn_res.attn_queries." + std::to_string(physical - 1U), full ? 3U : 1U);
        const auto qk_norm = full ? "q_input_norm.weight" : "attn_norm.weight";
        const auto query = layer_input(visit, attention_inputs, 0U, MessagePath::query, "attn", qk_norm);

        std::vector<float> key;
        std::vector<float> value;
        if (full || (is_core(physical) && visit.pass != 0)) {
            key = layer_input(visit, attention_inputs, full ? 1U : 0U, MessagePath::key, "k",
                              full ? "k_input_norm.weight" : qk_norm);
            value = layer_input(visit, attention_inputs, full ? 2U : 0U, MessagePath::value, "v",
                                full ? "v_input_norm.weight" : qk_norm);
        }
        const auto attention = backend_.attention(visit, position_, query, key.empty() ? query : key,
                                                   value.empty() ? query : value);
        auto residual = backend_.convolve(visit, false, attention);
        if (starts_residual_block(visit)) {
            blocks_.insert(blocks_.end(), hidden_.begin(), hidden_.end());
        } else {
            for (unsigned i = 0; i < kHiddenWidth; ++i) residual[i] += hidden_[i];
        }

        const auto moe_readout = residual_read(residual, "attn_res.mlp_queries." + std::to_string(physical), 1U);
        const auto moe_input = layer_input(visit, moe_readout, 0U, MessagePath::moe, "mlp", "mlp_norm.weight");
        const auto expert_sum = backend_.moe(visit, moe_input);
        const auto branch = backend_.convolve(visit, true, expert_sum);
        for (unsigned i = 0; i < kHiddenWidth; ++i) hidden_[i] = residual[i] + branch[i];
        mini_round_bf16(hidden_);
    }

private:
    void capture_draft_tap(unsigned visit, std::span<const float> hidden) {
        if constexpr (requires { backend_.capture_draft_tap(visit, hidden); })
            backend_.capture_draft_tap(visit, hidden);
    }

    std::vector<float> residual_read(std::span<const float> current, std::string_view query_name, unsigned queries) const {
        namespace ref = core::residual_reference;
        std::vector<float> bank = blocks_;
        bank.insert(bank.end(), current.begin(), current.end());
        return ref::mix(ref::normalize_bank(bank, kHiddenWidth, 1e-6F),
                        backend_.parameter(query_name, queries * kHiddenWidth), kHiddenWidth);
    }

    std::vector<float> layer_input(LayerVisit visit, std::span<const float> input, unsigned row,
                                  MessagePath path, std::string_view controller, std::string_view norm) const {
        namespace ref = core::residual_reference;
        const auto raw = input.subspan(row * kHiddenWidth, kHiddenWidth);
        const auto scale = backend_.parameter(mini_layer_parameter(visit.physical_layer, norm), kHiddenWidth);
        if (visit.pass == 0) return mini_rms_norm(raw, scale);

        const auto base = std::string("controllers.") + std::string(controller) + ".";
        const auto down = backend_.parameter(mini_layer_parameter(visit.physical_layer, base + "down_weight"),
                                             2U * kHiddenWidth * kControllerRank);
        const auto forget = backend_.parameter(mini_layer_parameter(visit.physical_layer, base + "forget_weight"),
                                               kHiddenWidth * kControllerRank);
        const auto gain = backend_.parameter(mini_layer_parameter(visit.physical_layer, base + "gain_weight"),
                                             kHiddenWidth * kControllerRank);
        const auto gated = ref::apply_low_rank_gate(raw, messages_.value().message(visit, path),
                                                    {down, forget, gain}, 1e-6F,
                                                    ref::DownProjectionPrecision::bf16);
        return mini_rms_norm(gated, scale);
    }

    Backend& backend_;
    unsigned position_;
    std::vector<float> hidden_;
    std::vector<float> blocks_;
    std::vector<float> entry_hidden_;
    std::vector<float> entry_blocks_;
    std::vector<float> boundary_;
    std::optional<MiniMessageBank> messages_;
};

template <class Backend>
std::vector<float> mini_forward_row(Backend& backend, unsigned position, std::span<const float> embedding) {
    return MiniForwardRow<Backend>(backend, position, embedding).run();
}

}  // namespace strix::inkling2
