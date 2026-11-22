#pragma once

#include "strix/graph/stage_graph.hpp"
#include "strix/model_description.hpp"
#include "strix/weight/tensor_map.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <memory>

namespace strix::weight { struct TensorIndex; }

namespace strix::graph {

struct LayerVisit {
    std::uint32_t physical_layer, pass, logical_layer;
    friend bool operator==(const LayerVisit&, const LayerVisit&) = default;
};

// D1 owns the parsed schema. Catalogue, state-layout, draft and tokenizer
// consumers can bind these records without rereading TOML or losing fields.
using DeclFields = model_description::Value::Table;

struct StageFields {
    std::string id, table;
    DeclFields fields;
    std::vector<LayerVisit> active_visits;
};

struct DeclGraph {
    std::string id, kind;
    StageGraph stages;
    std::vector<StageFields> rows;
};

struct TensorRequirement {
    std::string name, stage, format;
    // Logical shape in contiguous-axis-first order, as in tensor_map::Entry.
    std::array<std::uint64_t, 4> dims{};
    std::uint8_t rank{};
};

struct ModelDecl {
    std::string name, architecture, license, reference;
    weight::tensor_map::Model symbols;
    std::vector<weight::tensor_map::Entry> tensors;
    std::vector<TensorRequirement> requirements;
    StageGraph stages;
    std::vector<LayerVisit> schedule;
    DeclFields fields;
    std::vector<DeclFields> states, drafts;
    std::string chat, tokenizer;
    std::uint64_t identity{};

    // The first graph is the primary graph. Its StageGraph is also exposed as stages.
    [[nodiscard]] std::span<const DeclGraph> graphs() const noexcept;

private:
    struct Storage;
    std::shared_ptr<const Storage> storage_;
    friend class ModelDeclLoader;
};

struct ModelDeclOptions {
    std::filesystem::path templates = "data/graph/templates";
};

// Inspection resolves the schema and template without opening a checkpoint.
[[nodiscard]] ModelDecl inspect_model_decl(const model_description::ModelDescription&,
                                           const ModelDeclOptions& = {});
[[nodiscard]] ModelDecl load_model_decl(const model_description::ModelDescription&,
                                        const weight::TensorIndex&, const ModelDeclOptions& = {});

enum class DeclKeyKind { operation, decoder, head_dim, state_layout, transform };
struct DeclKey {
    DeclKeyKind kind;
    std::string_view graph, stage, key;
};
// A backend catalogue supplies this lookup. All keys must be present before admission.
using DeclLookup = std::function<bool(const DeclKey&)>;
void validate(const ModelDecl&, const DeclLookup&);

} // namespace strix::graph
