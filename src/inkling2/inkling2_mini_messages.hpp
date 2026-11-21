#pragma once

#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"
#include "core/residual_reference.hpp"

namespace strix::inkling2 {

enum class MessagePath : unsigned { query, key, value, moe };

class MiniMessageBank {
public:
    // One token's capture is [incoming layer-12 residual, output after layer 17].
    // Normalize these two rows once, then use them for all learned queries.
    MiniMessageBank(std::span<const float> captured_states, std::span<const float> attention_moe_queries,
                    std::span<const float> key_value_queries) {
        constexpr unsigned query_elements = 2U * MiniLayers::core_width * kHiddenWidth;
        if (captured_states.size() != 2U * kHiddenWidth || attention_moe_queries.size() != query_elements ||
            key_value_queries.size() != query_elements) {
            throw std::invalid_argument("invalid Inkling 2 Mini message tensor shape");
        }
        namespace ref = core::residual_reference;
        const auto bank = ref::normalize_bank(captured_states, kHiddenWidth, 1e-6F);
        attention_moe_ = ref::mix(bank, attention_moe_queries, kHiddenWidth);
        key_value_ = ref::mix(bank, key_value_queries, kHiddenWidth);
    }

    // Each tensor stores a pair of queries per physical core layer.
    // The same learned query reads the newly captured bank on each repeat.
    std::span<const float> message(LayerVisit visit, MessagePath path) const {
        if (visit.pass == 0 || visit.pass >= kCorePasses || !is_core(visit.physical_layer)) {
            throw std::invalid_argument("Inkling 2 Mini messages require a repeated core visit");
        }
        const auto offset = 2U * (visit.physical_layer - kCoreFirst) * kHiddenWidth;
        switch (path) {
            case MessagePath::query: return std::span<const float>(attention_moe_).subspan(offset, kHiddenWidth);
            case MessagePath::key: return std::span<const float>(key_value_).subspan(offset, kHiddenWidth);
            case MessagePath::value: return std::span<const float>(key_value_).subspan(offset + kHiddenWidth, kHiddenWidth);
            case MessagePath::moe: return std::span<const float>(attention_moe_).subspan(offset + kHiddenWidth, kHiddenWidth);
        }
        throw std::invalid_argument("invalid Inkling 2 Mini message path");
    }
private:
    std::vector<float> attention_moe_;
    std::vector<float> key_value_;
};

}  // namespace strix::inkling2
