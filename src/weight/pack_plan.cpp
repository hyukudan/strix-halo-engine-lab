#include "strix/weight/pack_plan.hpp"

#include "strix/serve/json.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_q8.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace strix::weight {
namespace {
using serve::Json;

const Json& required(const Json& object, const char* key) {
    const auto* value = object.find(key);
    if (!value) throw std::invalid_argument(std::string("missing ") + key);
    return *value;
}

std::uint64_t nonnegative(const Json& value) {
    const auto number = value.as_int();
    if (number < 0) throw std::invalid_argument("expected nonnegative integer");
    return static_cast<std::uint64_t>(number);
}

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::invalid_argument("pack plan arithmetic overflow");
    return a + b;
}

using ExpertKey = std::pair<std::uint64_t, std::uint64_t>;
struct Promotion {
    ExpertKey key;
    std::uint64_t routes{}, extra{};
    std::size_t first{}, last{};
};

std::string expert_name(std::string_view pattern, const ExpertKey& key) {
    std::string result(pattern);
    const auto substitute = [&](std::string_view token, std::uint64_t index) {
        const auto position = result.find(token);
        if (position == std::string::npos) throw std::invalid_argument("missing marker " + std::string(token));
        result.replace(position, token.size(), std::to_string(index));
    };
    substitute("{layer}", key.first);
    substitute("{expert}", key.second);
    if (result.find_first_of("{}") != std::string::npos) throw std::invalid_argument("invalid expert name template");
    return result;
}
}  // namespace

FormatId strixpack_format(std::string_view name) {
    if (name == "sp2") return kSP2WeightStorage;
    if (name == "sp3") return kSP3WeightStorage;
    if (name == "sp4") return kSP4WeightStorage;
    throw std::invalid_argument("expected sp2, sp3 or sp4");
}

std::string_view strixpack_label(FormatId format) {
    const auto bits = strixpack_bits(format);
    if (bits == 0) throw std::invalid_argument("expected a Strix pack format");
    return format.name.substr(6U);
}

std::uint64_t strixpack_matrix_bytes(FormatId format, std::uint64_t rows, std::uint64_t columns) {
    Shape shape{};
    shape.rank = 2;
    shape.dims = {rows, columns, 0, 0};
    MatrixLoadGeometry geometry{};
    if (const auto status = strixpack_matrix_geometry(format, shape, geometry); !status)
        throw std::invalid_argument(status.detail);
    return geometry.bytes;
}

PackPlan plan_strix_pack(const Json& description, FormatId cold, std::uint64_t hot_limit,
                        std::uint64_t budget, std::uint64_t reserve) {
    if (required(description, "schema").as_string() != "strix-pack-model/v1" || budget < reserve ||
        !is_strixpack_storage(cold) || (hot_limit != 0 && cold != kSP2WeightStorage))
        throw std::invalid_argument("invalid pack schema, budget, reserve or hot/cold format");
    PackPlan plan;
    plan.budget = budget;
    plan.reserve = reserve;
    if (const auto* retained = description.find("retained_bytes")) plan.bytes = nonnegative(*retained);
    std::set<std::string> names;
    const auto emit = [&](std::string name, FormatId storage, std::uint64_t bytes) {
        if (name.empty() || name.find_first_of("\t\r\n") != std::string::npos || !names.insert(name).second)
            throw std::invalid_argument("invalid or duplicate tensor name: " + name);
        plan.bytes = add(plan.bytes, bytes);
        plan.tensors.push_back({std::move(name), storage, bytes, {}});
    };
    if (const auto* retained = description.find("retained"))
        for (const auto& tensor : retained->as_array())
            emit(required(tensor, "name").as_string(), {}, nonnegative(required(tensor, "bytes")));
    const auto emit_matrix = [&](const Json& tensor, std::string name, FormatId fallback, bool routed = true) {
        auto storage = fallback;
        if (const auto* format = tensor.find("format")) {
            if (format->as_string() == "keep") {
                const auto bytes = nonnegative(required(tensor, "source_bytes"));
                emit(std::move(name), {}, bytes);
                return false;
            }
            storage = strixpack_format(format->as_string());
        }
        const auto bytes = strixpack_matrix_bytes(storage, nonnegative(required(tensor, "rows")),
                                                   nonnegative(required(tensor, "columns")));
        emit(std::move(name), storage, bytes);
        const auto* coder = routed ? nullptr : tensor.find("unobserved_coder");
        if (!coder) coder = tensor.find("coder");
        if (coder) {
            const auto& name_coder = coder->as_string();
            if (name_coder != "rtn" && name_coder != "importance" && name_coder != "feedback" && name_coder != "both")
                throw std::invalid_argument("invalid pack plan coder");
            plan.tensors.back().coder = name_coder;
        }
        return storage == cold;
    };
    if (const auto* matrices = description.find("matrices"))
        for (const auto& tensor : matrices->as_array())
            (void)emit_matrix(tensor, required(tensor, "name").as_string(), cold);
    std::map<ExpertKey, std::uint64_t> routes;
    if (const auto* counts = description.find("routes"))
        for (const auto& row : counts->as_array()) {
            const auto& fields = row.as_array();
            if (fields.size() != 3) throw std::invalid_argument("expected routes as [layer, expert, count]");
            const ExpertKey key{nonnegative(fields[0]), nonnegative(fields[1])};
            if (!routes.emplace(key, nonnegative(fields[2])).second)
                throw std::invalid_argument("duplicate expert route count");
        }

    std::vector<Promotion> candidates;
    const auto* groups = description.find("expert_groups");
    std::set<ExpertKey> experts_seen;
    if (groups) {
        for (const auto& group : groups->as_array()) {
            const auto first_layer = nonnegative(required(group, "first_layer"));
            const auto layers = nonnegative(required(group, "layers"));
            const auto experts = nonnegative(required(group, "experts"));
            const auto& projections = required(group, "projections").as_array();
            if (layers == 0 || experts == 0 || projections.empty() || layers > 1000000U / experts ||
                layers * experts > 1000000U / projections.size())
                throw std::invalid_argument("invalid expert group size");
            const auto layer_end = add(first_layer, layers);
            for (auto layer = first_layer; layer < layer_end; ++layer)
                for (std::uint64_t expert = 0; expert < experts; ++expert) {
                    const ExpertKey key{layer, expert};
                    if (!experts_seen.insert(key).second)
                        throw std::invalid_argument("expert appears in multiple groups");
                    auto route = routes.extract(key);
                    const auto count = route ? route.mapped() : 0U;
                    Promotion candidate{key, count, 0U, plan.tensors.size(), 0U};
                    for (const auto& projection : projections) {
                        const auto name = expert_name(required(projection, "name").as_string(), key);
                        const bool promotable = emit_matrix(projection, name, cold, count != 0);
                        if (hot_limit && !promotable)
                            throw std::invalid_argument("hot/cold promotion requires the cold format on every expert projection");
                        if (hot_limit && count)
                            candidate.extra = add(candidate.extra, plan.tensors.back().bytes / sp::SP2Storage::bytes *
                                                                   (sp::SP3Storage::bytes - sp::SP2Storage::bytes));
                    }
                    candidate.last = plan.tensors.size();
                    if (hot_limit && count) candidates.push_back(candidate);
                }
        }
    }
    if (!routes.empty()) throw std::invalid_argument("route counts name undeclared experts");
    if (hot_limit && candidates.empty()) throw std::invalid_argument("hot experts need positive route counts");
    std::ranges::sort(candidates, [](const Promotion& a, const Promotion& b) {
        return a.routes == b.routes ? a.key < b.key : a.routes > b.routes;
    });
    for (const auto& candidate : candidates) {
        if (plan.hot_experts == hot_limit || !plan.fits()) break;
        if (candidate.extra > budget - reserve - plan.bytes) continue;
        for (auto i = candidate.first; i < candidate.last; ++i) {
            auto& tensor = plan.tensors[i];
            // Both sizes have the same number of 128-weight superblocks.
            tensor.bytes = tensor.bytes / sp::SP2Storage::bytes * sp::SP3Storage::bytes;
            tensor.storage = kSP3WeightStorage;
        }
        plan.bytes += candidate.extra;
        ++plan.hot_experts;
    }
    return plan;
}

}  // namespace strix::weight
