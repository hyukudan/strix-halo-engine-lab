#include "inkling2_mini_pack_source.hpp"

#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_schedule.hpp"

#include <charconv>

namespace strix::inkling2 {

std::string mini_expert_pack_name(unsigned layer, unsigned expert, std::string_view role) {
    return "model.llm.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) + "." +
           std::string(role) + ".weight";
}

weight::Status mini_pack_source(const weight::PackCheckpoint& checkpoint, const MiniTextWeights& weights,
                               std::string_view name, weight::PackMatrixSource& out) {
    if (const auto* tensor = checkpoint.find(name)) return weight::PackMatrixSource::bind(checkpoint, *tensor, out);
    const auto missing = [&] { return weight::Status{weight::Error::malformed, "unknown pack matrix " + std::string(name)}; };
    constexpr std::string_view prefix = "model.llm.layers.", expert_path = ".mlp.experts.";
    if (!name.starts_with(prefix)) return missing();
    auto fields = name.substr(prefix.size());
    const auto number = [&](unsigned& value) {
        const auto* end = fields.data() + fields.size();
        const auto parsed = std::from_chars(fields.data(), end, value);
        fields.remove_prefix(static_cast<std::size_t>(parsed.ptr - fields.data()));
        return parsed.ec == std::errc{};
    };
    unsigned layer{}, expert{};
    if (!number(layer) || layer >= kPhysicalLayers || !fields.starts_with(expert_path)) return missing();
    fields.remove_prefix(expert_path.size());
    if (!number(expert) || expert >= kExperts) return missing();
    const bool down = fields == ".down.weight";
    const bool up = fields == ".up.weight";
    if (!down && !up && fields != ".gate.weight") return missing();
    const auto projection = down ? MiniExpertProjection::down : MiniExpertProjection::gate_up;
    weight::microscaled::MatrixView view;
    if (auto status = weights.expert_matrix(layer, expert, projection, view); !status) return status;
    return weight::PackMatrixSource::bind(view, mini_expert_row_order(projection), up ? kExpertWidth : 0U,
                                          down ? kHiddenWidth : kExpertWidth, out);
}

}  // namespace strix::inkling2
