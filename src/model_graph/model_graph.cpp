#include "strix/model_graph.hpp"

#include <iomanip>
#include <initializer_list>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace strix::model_graph {
namespace {
std::string quote(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (const char c : value) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (byte < 32) { out += "\\u00"; out += hex[byte >> 4]; out += hex[byte & 15]; }
        else out += c;
    }
    return out + '"';
}
template<class T, class Encode>
std::string array(const std::vector<T>& values, Encode encode) {
    std::string out = "[";
    bool comma = false;
    for (const auto& value : values) {
        if (comma) out += ',';
        comma = true;
        out += encode(value);
    }
    return out + ']';
}
template<class Enum>
std::string name(Enum value, std::initializer_list<const char*> names) {
    const auto index = static_cast<unsigned>(value);
    if (index >= names.size()) throw std::invalid_argument("model graph: invalid enum");
    return quote(*(names.begin() + index));
}
std::string tensor(const TensorRequirement& t) {
    return "{\"role\":" + quote(t.role) + ",\"format\":" + quote(t.format) + ",\"shape\":" +
        array(t.shape, [](const auto& extent) { return quote(extent); }) + ",\"ownership\":" + quote(t.ownership) + '}';
}
std::string layer(const LayerDesc& l) {
    std::string out = "{\"id\":" + std::to_string(l.id) + ",\"block\":" +
        name(l.block, {"transformer", "recurrent", "auxiliary"}) + ",\"attention\":" +
        name(l.attention, {"none", "dense", "gdn", "qsa", "kda", "mla", "compressed_indexed", "msa"});
    out += ",\"mlps\":" + array(l.mlps, [](const auto& m) {
        return "{\"kind\":" + name(m.kind, {"dense", "moe", "shared_expert"}) +
            ",\"intermediate\":" + std::to_string(m.intermediate) + ",\"experts\":" + std::to_string(m.experts) +
            ",\"selected\":" + std::to_string(m.selected) + '}';
    });
    out += ",\"norms\":" + array(l.norms, [](const auto& n) {
        return "{\"kind\":" + quote(n.kind) + ",\"epsilon\":" + quote(n.epsilon) + ",\"placement\":" +
            name(n.placement, {"pre_attention", "post_attention", "pre_mlp", "post_mlp", "input", "output"}) + '}';
    });
    out += ",\"hyper_connection\":";
    if (l.hyper_connection) {
        const auto& h = *l.hyper_connection;
        out += "{\"kind\":" + name(h.kind, {"hc", "mhc"}) + ",\"streams\":" + std::to_string(h.streams) +
            ",\"iterations\":" + std::to_string(h.iterations) + '}';
    } else out += "null";
    out += ",\"tensors\":" + array(l.tensors, tensor);
    out += ",\"states\":" + array(l.states, [](const auto& s) {
        return "{\"tensor\":" + tensor(s.tensor) + ",\"lifetime\":" + quote(s.lifetime) + '}';
    });
    // Omit empty extensions to preserve existing adapters' descriptive identities.
    if (!l.semantics.empty()) {
        out += ",\"semantics\":{";
        bool comma = false;
        for (const auto& [key, value] : l.semantics) {
            if (comma) out += ',';
            comma = true;
            out += quote(key) + ':' + quote(value);
        }
        out += '}';
    }
    if (!l.dimensions.empty()) {
        out += ",\"dimensions\":{";
        bool comma = false;
        for (const auto& [key, value] : l.dimensions) {
            if (comma) out += ',';
            comma = true;
            out += quote(key) + ':' + std::to_string(value);
        }
        out += '}';
    }
    return out + '}';
}
}  // namespace

std::string canonical(const ModelGraph& g) {
    std::string out = "{\"schema\":\"strix-model-graph/v" + std::to_string(ModelGraph::schema_version) +
        "\",\"model\":" + quote(g.model) + ",\"semantic_version\":" + quote(g.semantic_version) + ",\"semantics\":{";
    bool comma = false;
    for (const auto& [key, value] : g.semantics) {
        if (comma) out += ',';
        comma = true;
        out += quote(key) + ':' + quote(value);
    }
    out += "},\"dimensions\":{";
    comma = false;
    for (const auto& [key, value] : g.dimensions) {
        if (comma) out += ',';
        comma = true;
        out += quote(key) + ':' + std::to_string(value);
    }
    out += "},\"inputs\":" + array(g.inputs, tensor) + ",\"outputs\":" + array(g.outputs, tensor);
    out += ",\"tensors\":" + array(g.tensors, tensor);
    out += ",\"layers\":" + array(g.layers, layer);
    out += ",\"attachments\":" + array(g.attachments, [](const auto& a) {
        return "{\"kind\":" + name(a.kind, {"ple", "mtp", "draft"}) + ",\"name\":" + quote(a.name) +
            ",\"target\":" + quote(a.target) + ",\"layers\":" + array(a.layers, layer) + '}';
    });
    out += ",\"edges\":" + array(g.edges, [](const auto& e) {
        return "{\"from\":" + quote(e.from) + ",\"to\":" + quote(e.to) + ",\"kind\":" +
            name(e.kind, {"data", "residual", "hyper_connection", "cross_layer"}) + '}';
    });
    if (!g.states.empty()) out += ",\"states\":" + array(g.states, [](const auto& s) {
        return "{\"tensor\":" + tensor(s.tensor) + ",\"lifetime\":" + quote(s.lifetime) + '}';
    });
    return out + '}';
}
std::string identity(const ModelGraph& graph) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const char c : canonical(graph)) {
        hash ^= static_cast<unsigned char>(c);
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}
std::string dump(const ModelGraph& graph) {
    validate(graph);
    return "{\"identity\":" + quote(identity(graph)) + ',' + canonical(graph).substr(1) + '\n';
}
void validate(const ModelGraph& graph) {
    if (graph.model.empty() || graph.semantic_version.empty() || graph.layers.empty())
        throw std::invalid_argument("model graph: missing model, semantic version or trunk");
    std::set<std::uint64_t> ids;
    auto check = [&ids](const LayerDesc& l) {
        if (!ids.insert(l.id).second) throw std::invalid_argument("model graph: duplicate layer id");
        for (const auto& s : l.states)
            if (s.tensor.role.empty() || s.tensor.format.empty() || s.lifetime.empty())
                throw std::invalid_argument("model graph: incomplete state requirement");
    };
    for (const auto& l : graph.layers) check(l);
    for (const auto& s : graph.states)
        if (s.tensor.role.empty() || s.tensor.format.empty() || s.lifetime.empty())
            throw std::invalid_argument("model graph: incomplete global state requirement");
    for (const auto& a : graph.attachments) {
        if (a.name.empty() || a.target.empty()) throw std::invalid_argument("model graph: incomplete attachment");
        for (const auto& l : a.layers) check(l);
    }
    (void)canonical(graph);  // validate enum values too
}
}  // namespace strix::model_graph
