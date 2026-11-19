#include "stage_bridge.hpp"

#include <stdexcept>
#include <string>

namespace strix::model_graph {

std::vector<graph::Traits> stage_traits(const ModelGraph& g) {
    std::vector<graph::Traits> out;
    out.reserve(g.layers.size());
    for (const auto& l : g.layers) {
        graph::Traits t = 0;
        switch (l.attention) {
            case AttentionFamily::dense: t |= graph::trait::attn_dense; break;
            case AttentionFamily::gdn: t |= graph::trait::attn_gdn; break;
            case AttentionFamily::compressed_indexed: t |= graph::trait::attn_indexed; break;
            case AttentionFamily::mla: {
                const auto w = l.dimensions.find("window");
                t |= (w != l.dimensions.end() && w->second > 0U) ? graph::trait::attn_window : graph::trait::attn_indexed;
                break;
            }
            default: throw std::invalid_argument("stage graph: no stage trait for the attention family of layer " + std::to_string(l.id));
        }
        for (const auto& m : l.mlps) {
            if (m.kind == MlpKind::dense) t |= graph::trait::ffn_dense;
            if (m.kind == MlpKind::moe) t |= graph::trait::ffn_moe;
        }
        out.push_back(t);
    }
    return out;
}

graph::StageGraph to_stage_graph(const ModelGraph& g, graph::StageGraph declaration) {
    declaration.layers = stage_traits(g);
    graph::validate(declaration);
    return declaration;
}

}  // namespace strix::model_graph
