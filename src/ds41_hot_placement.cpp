#include "strix/ds41_hot_placement.hpp"

#include "strix/serve/json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strix::ds41 {

namespace {

using serve::Json;

constexpr std::array<std::string_view, hot_device_count> device_names{"rtx-pro", "rtx-4090", "pc-ram", "amd"};

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("hot placement: " + what); }

std::size_t index_of(HotDevice device) noexcept { return static_cast<std::size_t>(device); }

// Accessors return pointers (never null): a reference result with temporary string arguments trips the
// dangling-reference warning.
const Json* member(const Json& object, std::string_view key, const std::string& where) {
    const Json* value = object.find(key);
    if (value == nullptr) fail(where + " lacks \"" + std::string(key) + "\"");
    return value;
}
const Json* object_member(const Json& object, std::string_view key, const std::string& where) {
    const Json* value = member(object, key, where);
    if (!value->is_object()) fail(where + "." + std::string(key) + " is not an object");
    return value;
}
std::string string_member(const Json& object, std::string_view key, const std::string& where) {
    const Json* value = member(object, key, where);
    if (!value->is_string()) fail(where + "." + std::string(key) + " is not a string");
    return value->as_string();
}
std::uint64_t unsigned_value(const Json& value, const std::string& where) {
    if (!value.is_number()) fail(where + " is not a number");
    const double d = value.as_double();
    if (!(d >= 0.0) || d != std::floor(d) || d > 9.0e15) fail(where + " is not a non-negative integer");
    return static_cast<std::uint64_t>(value.as_int());
}
std::uint64_t unsigned_member(const Json& object, std::string_view key, const std::string& where) {
    return unsigned_value(*member(object, key, where), where + "." + std::string(key));
}
std::uint32_t u32(std::uint64_t value, const std::string& where) {
    if (value > std::numeric_limits<std::uint32_t>::max()) fail(where + " is out of range");
    return static_cast<std::uint32_t>(value);
}
std::optional<double> optional_number(const Json& object, std::string_view key, const std::string& where) {
    const Json* value = object.find(key);
    if (value == nullptr) return std::nullopt;
    if (!value->is_number()) fail(where + "." + std::string(key) + " is not a number");
    const double d = value->as_double();
    if (!(d >= 0.0) || !std::isfinite(d)) fail(where + "." + std::string(key) + " is not a finite non-negative number");
    return d;
}
HotDevice device_member(const Json& object, std::string_view key, const std::string& where) {
    const auto name = string_member(object, key, where);
    const auto device = hot_device_from_name(name);
    if (!device) fail(where + "." + std::string(key) + ": unknown device \"" + name + "\"");
    return *device;
}

void check_geometry(const HotPlacementGeometry& got, const HotPlacementGeometry& want) {
    const auto check = [](std::uint64_t a, std::uint64_t b, const char* name) {
        if (a != b) fail(std::string("model.") + name + " is " + std::to_string(a) + ", expected " + std::to_string(b));
    };
    check(got.layers, want.layers, "layers");
    check(got.experts, want.experts, "experts");
    check(got.top_k, want.top_k, "top_k");
    check(got.expert_bytes, want.expert_bytes, "expert_bytes");
    check(got.canonical_split, want.canonical_split, "canonical_split");
    if (got.layers == 0U || got.experts == 0U || got.top_k == 0U || got.top_k > got.experts || got.canonical_split > got.experts)
        fail("model geometry is degenerate");
}

void parse_devices(const Json& root, HotPlacement& out) {
    const Json& devices = *member(root, "devices", "file");
    if (!devices.is_array()) fail("devices is not an array");
    std::array<bool, hot_device_count> seen{};
    for (std::size_t i = 0; i < devices.as_array().size(); ++i) {
        const Json& entry = devices.as_array()[i];
        const std::string where = "devices[" + std::to_string(i) + "]";
        if (!entry.is_object()) fail(where + " is not an object");
        HotPlacementDevice device;
        device.device = device_member(entry, "id", where);
        if (seen[index_of(device.device)]) fail(where + ": device \"" + std::string(hot_device_name(device.device)) + "\" repeats");
        seen[index_of(device.device)] = true;
        if (const Json* kind = entry.find("kind"); kind != nullptr && kind->is_string()) device.kind = kind->as_string();
        if (const Json* role = entry.find("role"); role != nullptr && role->is_string()) device.role = role->as_string();
        device.budget_gib = optional_number(entry, "budget_gib", where).value_or(0.0);
        device.reserve_gib = optional_number(entry, "reserve_gib", where).value_or(0.0);
        device.expert_gib = optional_number(entry, "expert_gib", where);
        if (device.device == HotDevice::amd) {
            const Json& servers = *member(entry, "servers", where);
            if (!servers.is_array() || servers.as_array().empty()) fail(where + ".servers is not a non-empty array");
            for (std::size_t s = 0; s < servers.as_array().size(); ++s) {
                const Json& server = servers.as_array()[s];
                const std::string at = where + ".servers[" + std::to_string(s) + "]";
                if (!server.is_object()) fail(at + " is not an object");
                const Json& range = *member(server, "experts", at);
                if (!range.is_array() || range.as_array().size() != 2U) fail(at + ".experts is not [first, end]");
                HotAmdServer amd;
                amd.node = string_member(server, "node", at);
                amd.first = u32(unsigned_value(range.as_array()[0], at + ".experts[0]"), at);
                amd.end = u32(unsigned_value(range.as_array()[1], at + ".experts[1]"), at);
                out.amd_servers.push_back(std::move(amd));
            }
        }
        out.devices.push_back(std::move(device));
    }
    if (!seen[index_of(HotDevice::amd)]) fail("devices lacks \"amd\" (the servers of the experts not listed)");
    std::sort(out.amd_servers.begin(), out.amd_servers.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::uint32_t next = 0;
    for (const auto& server : out.amd_servers) {
        if (server.first != next || server.end <= server.first) fail("amd server ranges do not cover the experts contiguously");
        next = server.end;
    }
    if (next != out.geometry.experts) fail("amd server ranges do not end at the expert count");
}

void parse_layers(const Json& root, HotPlacement& out) {
    const Json& layers = *member(root, "layers", "file");
    if (!layers.is_array()) fail("layers is not an array");
    const auto count = out.geometry.layers, experts = out.geometry.experts;
    if (layers.as_array().size() != count)
        fail("layers has " + std::to_string(layers.as_array().size()) + " entries, expected " + std::to_string(count));
    out.layers.resize(count);
    for (std::uint32_t l = 0; l < count; ++l) {
        const Json& entry = layers.as_array()[l];
        const std::string where = "layers[" + std::to_string(l) + "]";
        if (!entry.is_object()) fail(where + " is not an object");
        if (unsigned_member(entry, "layer", where) != l) fail(where + ".layer is not " + std::to_string(l));
        auto& layer = out.layers[l];
        layer.device.assign(experts, HotDevice::amd);
        std::vector<bool> listed(experts, false);
        for (const auto& [key, value] : entry.as_object()) {
            if (key == "layer") continue;
            const auto device = hot_device_from_name(key);
            if (!device) {
                // Additive keys are allowed in v1, but an unknown list would silently move experts to amd.
                if (value.is_array()) fail(where + ": unknown device list \"" + key + "\"");
                continue;
            }
            if (*device == HotDevice::amd) fail(where + ": \"amd\" is implicit (every expert not listed elsewhere)");
            if (!value.is_array()) fail(where + "." + key + " is not an array");
            auto& ids = layer.experts[index_of(*device)];
            for (std::size_t i = 0; i < value.as_array().size(); ++i) {
                const auto id = unsigned_value(value.as_array()[i], where + "." + key + "[" + std::to_string(i) + "]");
                if (id >= experts) fail(where + "." + key + ": expert " + std::to_string(id) + " out of range");
                if (!ids.empty() && id <= ids.back()) fail(where + "." + key + ": ids are not strictly ascending");
                if (listed[id]) fail(where + ": expert " + std::to_string(id) + " is listed on two devices");
                listed[id] = true;
                ids.push_back(static_cast<std::uint32_t>(id));
                layer.device[id] = *device;
            }
        }
        auto& amd = layer.experts[index_of(HotDevice::amd)];
        for (std::uint32_t e = 0; e < experts; ++e)
            if (!listed[e]) amd.push_back(e);
        layer.slot.assign(experts, 0U);
        for (const auto& ids : layer.experts)
            for (std::size_t i = 0; i < ids.size(); ++i) layer.slot[ids[i]] = static_cast<std::uint32_t>(i);
    }
}

void check_memory(const HotPlacement& out) {
    for (const auto& device : out.devices) {
        if (device.device == HotDevice::amd) continue;
        const auto resident = out.resident_experts(device.device);
        const double bytes = static_cast<double>(resident) * static_cast<double>(out.geometry.expert_bytes);
        if (bytes > device.expert_capacity_bytes())
            fail("device \"" + std::string(hot_device_name(device.device)) + "\" holds " + std::to_string(resident) +
                 " experts, over its expert memory");
    }
    for (std::size_t d = 0; d < hot_device_count; ++d) {
        const auto device = static_cast<HotDevice>(d);
        if (device != HotDevice::amd && out.find_device(device) == nullptr && out.resident_experts(device) != 0U)
            fail("layers place experts on \"" + std::string(hot_device_name(device)) + "\", which devices does not declare");
    }
}

} // namespace

std::string_view hot_device_name(HotDevice device) noexcept { return device_names[index_of(device)]; }

std::optional<HotDevice> hot_device_from_name(std::string_view name) noexcept {
    for (std::size_t d = 0; d < hot_device_count; ++d)
        if (device_names[d] == name) return static_cast<HotDevice>(d);
    return std::nullopt;
}

double HotPlacementDevice::expert_capacity_bytes() const noexcept {
    const double gib = expert_gib ? *expert_gib : std::max(0.0, budget_gib - reserve_gib);
    return gib * 1073741824.0;
}

HotDevice HotPlacement::device_of(std::uint32_t layer, std::uint32_t expert) const {
    if (layer >= layers.size() || expert >= geometry.experts) throw std::out_of_range("hot placement: (layer, expert) out of range");
    return layers[layer].device[expert];
}
std::uint32_t HotPlacement::slot_of(std::uint32_t layer, std::uint32_t expert) const {
    if (layer >= layers.size() || expert >= geometry.experts) throw std::out_of_range("hot placement: (layer, expert) out of range");
    return layers[layer].slot[expert];
}
std::uint32_t HotPlacement::amd_server_of(std::uint32_t expert) const {
    for (std::size_t s = 0; s < amd_servers.size(); ++s)
        if (expert >= amd_servers[s].first && expert < amd_servers[s].end) return static_cast<std::uint32_t>(s);
    throw std::out_of_range("hot placement: no amd server holds the expert");
}
std::uint64_t HotPlacement::resident_experts(HotDevice device) const {
    std::uint64_t total = 0;
    for (const auto& layer : layers) total += layer.experts[index_of(device)].size();
    return total;
}
const HotPlacementDevice* HotPlacement::find_device(HotDevice device) const noexcept {
    for (const auto& d : devices)
        if (d.device == device) return &d;
    return nullptr;
}

HotPlacement parse_hot_placement(std::string_view json_text, const HotPlacementExpect& expect) {
    Json root;
    try {
        root = serve::parse_json(json_text);
    } catch (const std::exception& error) {
        fail(std::string("invalid JSON: ") + error.what());
    }
    if (!root.is_object()) fail("the file is not a JSON object");
    const auto format = string_member(root, "format", "file");
    if (format != hot_placement_format) fail("unknown format \"" + format + "\"");

    HotPlacement out;
    const Json& model = *object_member(root, "model", "file");
    if (const Json* family = model.find("family"); family != nullptr && family->is_string()) out.model_family = family->as_string();
    if (const Json* file = model.find("file"); file != nullptr && file->is_string()) out.model_file = file->as_string();
    if (const Json* identity = model.find("model_identity"); identity != nullptr && !identity->is_null()) {
        if (!identity->is_string()) fail("model.model_identity is neither a string nor null");
        out.model_identity = identity->as_string();
    }
    out.geometry.layers = u32(unsigned_member(model, "layers", "model"), "model.layers");
    out.geometry.experts = u32(unsigned_member(model, "experts", "model"), "model.experts");
    out.geometry.top_k = u32(unsigned_member(model, "top_k", "model"), "model.top_k");
    out.geometry.expert_bytes = unsigned_member(model, "expert_bytes", "model");
    out.geometry.canonical_split = u32(unsigned_member(model, "canonical_split", "model"), "model.canonical_split");
    check_geometry(out.geometry, expect.geometry);
    if (expect.model_identity && out.model_identity && *out.model_identity != *expect.model_identity)
        fail("model identity " + *out.model_identity + " differs from the engine's " + *expect.model_identity);

    if (const Json* dense = root.find("dense"); dense != nullptr) {
        if (!dense->is_object()) fail("dense is not an object");
        out.dense_device = device_member(*dense, "device", "dense");
    }
    const Json& combine = *object_member(root, "combine", "file");
    out.combine_identity = string_member(combine, "identity", "combine");
    if (out.combine_identity != hot_placement_combine) fail("unknown combine identity \"" + out.combine_identity + "\"");
    if (const Json* ret = combine.find("remote_return"); ret != nullptr && ret->is_string()) out.remote_return = ret->as_string();

    parse_devices(root, out);
    parse_layers(root, out);
    check_memory(out);
    return out;
}

HotPlacement load_hot_placement(const std::string& path, const HotPlacementExpect& expect) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    std::ostringstream text;
    text << in.rdbuf();
    if (!in.good() && !in.eof()) fail("cannot read " + path);
    return parse_hot_placement(text.str(), expect);
}

HotLayerSplit split_hot_layer(const HotPlacement& placement, std::uint32_t layer, std::span<const std::uint32_t> ids,
                              std::span<const float> weights, std::uint32_t rows, const HotRequestSizes& sizes) {
    const auto& g = placement.geometry;
    if (layer >= placement.layers.size()) throw std::invalid_argument("hot split: layer out of range");
    const std::size_t entries = static_cast<std::size_t>(rows) * g.top_k;
    if (rows == 0U || ids.size() != entries || weights.size() != entries)
        throw std::invalid_argument("hot split needs rows * top_k ids and weights");
    const auto& map = placement.layers[layer];

    HotLayerSplit out;
    out.layer = layer;
    out.rows = rows;
    out.top_k = g.top_k;
    out.split = g.canonical_split;
    out.order.resize(entries);
    out.low_counts.assign(rows, 0U);
    out.folded_prefix.assign(static_cast<std::size_t>(rows) * 2U, 0U);

    // Targets: rtx_4090 at index 0, AMD server s at 1 + s.
    const std::size_t target_count = 1U + placement.amd_servers.size();
    std::vector<HotTargetWork> work(target_count);
    work[0].target = {HotDevice::rtx_4090, 0U};
    for (std::size_t s = 0; s < placement.amd_servers.size(); ++s) work[1U + s].target = {HotDevice::amd, static_cast<std::uint32_t>(s)};
    const auto target_index = [](const HotSlot& slot) -> std::size_t {
        return slot.device == HotDevice::rtx_4090 ? 0U : 1U + slot.server;
    };
    std::vector<std::vector<bool>> target_rows(target_count, std::vector<bool>(rows, false));
    std::array<std::vector<bool>, hot_device_count> used;
    for (auto& u : used) u.assign(g.experts, false);
    std::vector<std::vector<bool>> target_used(target_count, std::vector<bool>(g.experts, false));

    for (std::uint32_t r = 0; r < rows; ++r) {
        HotSlot* row = out.order.data() + static_cast<std::size_t>(r) * g.top_k;
        for (std::uint32_t k = 0; k < g.top_k; ++k) {
            const auto route_slot = r * g.top_k + k;
            const auto expert = ids[route_slot];
            if (expert >= g.experts) throw std::invalid_argument("hot split: expert id out of range");
            for (std::uint32_t j = 0; j < k; ++j)
                if (row[j].expert == expert) throw std::invalid_argument("hot split: an expert repeats within a row");
            HotSlot& slot = row[k];
            slot.expert = expert;
            slot.route_slot = route_slot;
            slot.weight_slot = map.slot[expert];
            slot.weight = weights[route_slot];
            slot.device = map.device[expert];
            slot.server = slot.device == HotDevice::amd ? placement.amd_server_of(expert) : 0U;
        }
        std::sort(row, row + g.top_k, [](const HotSlot& a, const HotSlot& b) { return a.expert < b.expert; });
        std::uint32_t low = 0;
        while (low < g.top_k && row[low].expert < g.canonical_split) ++low;
        out.low_counts[r] = low;

        for (std::uint8_t half = 0; half < 2U; ++half) {
            const std::uint32_t begin = half == 0U ? 0U : low, end = half == 0U ? low : g.top_k;
            if (begin == end) continue;
            // Leading run of the half on one target: folded when it has at least two positions. Later positions
            // of the same target (after a position elsewhere) go raw; the parts-host continues the fold.
            std::uint32_t prefix = 0;
            if (hot_device_returns_vectors(row[begin].device)) {
                const auto lead = target_index(row[begin]);
                std::uint32_t run = 1;
                while (begin + run < end && hot_device_returns_vectors(row[begin + run].device) && target_index(row[begin + run]) == lead)
                    ++run;
                if (run >= 2U) prefix = run;
            }
            out.folded_prefix[static_cast<std::size_t>(r) * 2U + half] = prefix;
            if (prefix != 0U) {
                auto& t = work[target_index(row[begin])];
                HotVectorRequest request;
                request.row = r;
                request.half = half;
                request.kind = HotVectorKind::folded_prefix;
                request.first = 0;
                request.count = prefix;
                request.expert = row[begin].expert;
                request.route_slot = row[begin].route_slot;
                t.requests.push_back(request);
                ++t.folded_vectors;
                t.folded_positions += prefix;
                for (auto p = begin; p < begin + prefix; ++p) row[p].folded = true;
            }
            for (auto p = begin; p < end; ++p) {
                if (!hot_device_returns_vectors(row[p].device) || row[p].folded) continue;
                auto& t = work[target_index(row[p])];
                HotVectorRequest request;
                request.row = r;
                request.half = half;
                request.kind = HotVectorKind::raw;
                request.first = p - begin;
                request.count = 1;
                request.expert = row[p].expert;
                request.route_slot = row[p].route_slot;
                t.requests.push_back(request);
                ++t.raw_vectors;
            }
        }
        for (std::uint32_t p = 0; p < g.top_k; ++p) {
            const auto d = index_of(row[p].device);
            ++out.positions[d];
            used[d][row[p].expert] = true;
            if (hot_device_returns_vectors(row[p].device)) {
                const auto t = target_index(row[p]);
                target_rows[t][r] = true;
                target_used[t][row[p].expert] = true;
            }
        }
    }

    for (std::size_t d = 0; d < hot_device_count; ++d)
        for (std::uint32_t e = 0; e < g.experts; ++e)
            if (used[d][e]) out.experts[d].push_back(e);
    out.upload_bytes = out.experts[index_of(HotDevice::pc_ram)].size() * g.expert_bytes;
    for (std::size_t t = 0; t < target_count; ++t) {
        auto& target = work[t];
        if (target.requests.empty()) continue;
        for (std::uint32_t e = 0; e < g.experts; ++e)
            if (target_used[t][e]) target.experts.push_back(e);
        target.rows = static_cast<std::uint32_t>(std::count(target_rows[t].begin(), target_rows[t].end(), true));
        target.input_bytes = target.rows * sizes.input_bytes_per_row;
        target.output_bytes = (std::uint64_t{target.raw_vectors} + target.folded_vectors) * sizes.output_bytes_per_vector;
        if (target.target.device == HotDevice::rtx_4090) {
            out.pcie_input_bytes += target.input_bytes;
            out.pcie_output_bytes += target.output_bytes;
        } else {
            out.network_input_bytes += target.input_bytes;
            out.network_output_bytes += target.output_bytes;
        }
        out.targets.push_back(std::move(target));
    }
    return out;
}

} // namespace strix::ds41
