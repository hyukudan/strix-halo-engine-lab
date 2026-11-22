#include "strix/state/table.hpp"

#include "strix/graph/model_decl.hpp"
#include "strix/session/kv_row_append.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace strix::state {
namespace {

using Size = std::uint64_t;

Size add(Size a, Size b) {
    if (b > std::numeric_limits<Size>::max() - a) throw std::overflow_error("state layout size overflow");
    return a + b;
}

Size mul(Size a, Size b) {
    if (b && a > std::numeric_limits<Size>::max() / b) throw std::overflow_error("state layout size overflow");
    return a * b;
}

Size align(Size n, std::uint32_t alignment) {
    return add(n, (Size{0} - n) & (alignment - 1U));
}

// A versioned, byte-order-independent physical layout identity.
struct Identity {
    Size value = 14695981039346656037ULL;

    void byte(std::uint8_t b) { value = (value ^ b) * 1099511628211ULL; }
    void number(Size n) {
        for (unsigned i = 0; i < 8; ++i) {
            byte(static_cast<std::uint8_t>(n));
            n >>= 8U;
        }
    }
    void word(std::string_view s) {
        number(s.size());
        for (const char c : s) byte(static_cast<std::uint8_t>(c));
    }
};

void hash_decl(Identity& h, const Decl& d) {
    h.word(d.id);
    h.word(d.when);
    h.word(d.canonical.layout);
    for (const auto n : d.dims) h.number(n);
    for (const auto n : {static_cast<Size>(d.kind), static_cast<Size>(d.per), static_cast<Size>(d.dtype),
                         Size{d.planes}, Size{d.window}, Size{d.ratio}, Size{d.group}, static_cast<Size>(d.rollback),
                         Size{static_cast<std::uint32_t>(d.producer)}, static_cast<Size>(d.owner), Size{d.block},
                         Size{d.plane_bytes}, static_cast<Size>(d.shadow.dtype), Size{d.shadow.planes},
                         Size{d.shadow.plane_bytes}, static_cast<Size>(d.canonical.dtype), static_cast<Size>(d.absent),
                         Size{d.position_range}, Size{d.replay_input_bytes}, Size{d.checkpoint_interval},
                         Size{d.row_alignment}}) h.number(n);
}

Size plane_bytes(Dtype dtype, Size elements, std::uint32_t explicit_bytes) {
    if (explicit_bytes) return explicit_bytes;
    switch (dtype) {
    case Dtype::f32:
    case Dtype::u32: return mul(elements, 4);
    case Dtype::bf16:
    case Dtype::f16: return mul(elements, 2);
    case Dtype::e4m3_shadow:
    case Dtype::i8_token:
    case Dtype::i8_g64:
    case Dtype::i4_had:
    case Dtype::trellis:
    case Dtype::fp8_block:
    case Dtype::fp4_block: throw std::invalid_argument("packed state requires explicit plane bytes");
    }
    throw std::invalid_argument("unknown state dtype");
}

bool selected(const Decl& d, const graph::ModelDecl& model, const graph::LayerVisit& visit,
              const LayoutOptions& options) {
    if (d.when.empty()) return true;
    if (options.matches) return options.matches(d.when, visit);
    constexpr std::pair<std::string_view, graph::Traits> named[] = {
        {"full", graph::trait::attn_dense}, {"window", graph::trait::attn_window},
        {"gdn", graph::trait::attn_gdn}, {"indexed", graph::trait::attn_indexed},
        {"dense", graph::trait::ffn_dense}, {"moe", graph::trait::ffn_moe}, {"inject", graph::trait::inject},
    };
    const auto name = d.when.starts_with('!') ? d.when.substr(1) : d.when;
    for (const auto& [key, flag] : named) {
        if (key == name && visit.physical_layer < model.stages.layers.size())
            return ((model.stages.layers[visit.physical_layer] & flag) != 0U) == (name == d.when);
    }
    throw std::invalid_argument("state layer predicate needs a D1 visit selector");
}

InstanceLayout geometry(const Decl& d, const Layout& plan, const LayoutOptions& options) {
    if (!d.dims[0] || (!d.dims[1] && d.dims[2]) || !d.planes || !d.block || !d.ratio || !d.group ||
        !std::has_single_bit(d.row_alignment) || (d.position_range && d.owner != Owner::per_rank) ||
        (d.producer < -1) || (d.per == Per::model && (!d.when.empty() || d.producer >= 0)))
        throw std::invalid_argument("invalid state declaration");
    if (d.rollback != Rollback::replay && (d.replay_input_bytes || d.checkpoint_interval))
        throw std::invalid_argument("replay workspace declared for a different rollback policy");
    if (d.rollback == Rollback::replay && !d.replay_input_bytes)
        throw std::invalid_argument("replay requires a retained input image");
    if (d.checkpoint_interval == 1 || d.checkpoint_interval > plan.max_pending)
        throw std::invalid_argument("invalid state checkpoint interval");

    const Size width = mul(mul(d.dims[0], std::max(1U, d.dims[1])), std::max(1U, d.dims[2]));
    InstanceLayout out{};
    out.rollback = d.rollback;
    out.owner = d.owner;
    out.position_range = d.position_range;
    out.master_bytes = mul(plane_bytes(d.dtype, width, d.plane_bytes), d.planes);
    if (d.shadow.planes) out.shadow_bytes = mul(plane_bytes(d.shadow.dtype, width, d.shadow.plane_bytes), d.shadow.planes);
    const auto payload = add(out.master_bytes, out.shadow_bytes);
    out.row_stride = align(d.rollback == Rollback::rewind ? add(payload, sizeof(std::uint32_t)) : payload,
                           d.row_alignment);
    out.replay_input_bytes = d.replay_input_bytes;
    out.checkpoint_interval = d.checkpoint_interval;
    out.window = std::min(d.window, plan.context);
    out.row_divisor = d.kind == Kind::index_pool && d.ratio == 1 ? d.group : d.ratio;

    if ((d.kind == Kind::kv_window && !out.window) || (d.kind == Kind::kv_compressed && out.row_divisor == 1) ||
        (out.window && out.row_divisor > 1) || (d.rollback != Rollback::rewind && out.row_divisor > 1) ||
        (d.kind == Kind::recurrent && d.rollback == Rollback::rewind) ||
        (d.kind == Kind::taps && d.rollback == Rollback::rewind && !out.window))
        throw std::invalid_argument("incompatible state row geometry");

    if (d.rollback == Rollback::snapshot) {
        out.rows = Size{plan.max_pending} + 1U;
    } else if (d.rollback != Rollback::rewind) {
        out.rows = 3;  // committed plus two working images
        if (d.checkpoint_interval) out.rows += plan.max_pending / d.checkpoint_interval;
    } else if (out.row_divisor > 1) {
        out.rows = plan.context / out.row_divisor;
        out.carry_rows = static_cast<std::uint32_t>(std::min(Size{plan.context}, Size{plan.max_pending} + 1U));
    } else {
        out.rows = plan.context;
        if (out.window && plan.max_pending < plan.context) {
            // The incumbent sizes for W-1 history rows plus the new launch.
            // Keep one more row to preserve the old last query's full window on abort.
            const auto ring = session::make_visit_cache_layout<0>(plan.context, plan.max_pending + 1U,
                                                                 1, out.window, {}, 0);
            out.rows = ring.kv_rows;
            out.ring = ring.kv.ring;
        } else if (!out.window && d.kind == Kind::kv_dense &&
                   mul(out.rows, out.row_stride) > options.dense_slot_budget) {
            out.pages = (plan.context - 1U) / page_rows + 1U;
            out.rows = Size{out.pages} * page_rows;
        }
    }

    out.carry_offset = mul(out.row_stride, out.rows);
    out.input_offset = add(out.carry_offset, mul(out.row_stride, out.carry_rows));
    out.slot_bytes = align(add(out.input_offset, mul(plan.max_pending, d.replay_input_bytes)), d.row_alignment);
    return out;
}

}  // namespace

Layout layout(std::span<const Decl> decls, const graph::ModelDecl& model, Size context,
              std::uint32_t slots, const LayoutOptions& options) {
    if (!context || context > std::numeric_limits<std::uint32_t>::max() || !slots || !options.max_pending ||
        options.max_pending > context || decls.size() >= no_instance)
        throw std::invalid_argument("invalid state admission capacity");
    Layout out{};
    out.context = static_cast<std::uint32_t>(context);
    out.slots = slots;
    out.max_pending = options.max_pending;
    Identity hash;
    hash.word("strix.state.layout.v1");
    hash.number(model.identity);
    hash.number(context);
    hash.number(slots);
    hash.number(out.max_pending);
    hash.number(decls.size());

    std::set<std::string_view> ids;
    for (std::uint32_t index = 0; index < decls.size(); ++index) {
        const auto& d = decls[index];
        if (d.id.empty() || !ids.insert(d.id).second) throw std::invalid_argument("empty or duplicate state id");
        const auto shape = geometry(d, out, options);
        hash_decl(hash, d);
        out.prefix_block = mul(out.prefix_block / std::gcd(out.prefix_block, d.block), d.block);
        const auto first = out.instances.size();
        std::set<std::uint32_t> layers;
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> producers;
        const auto emit = [&](std::uint32_t layer, std::uint32_t visit, std::uint32_t pass) {
            if (out.instances.size() >= no_instance) throw std::overflow_error("too many state instances");
            auto instance = shape;
            instance.declaration = index;
            instance.layer = layer;
            instance.visit = visit;
            instance.pass = pass;
            instance.storage = static_cast<std::uint32_t>(out.instances.size());
            if (d.producer >= 0) {
                const auto producer = static_cast<std::uint32_t>(d.producer);
                if (layer != producer) instance.storage = no_instance;
                else if (!producers.emplace(std::pair{producer, pass}, instance.storage).second)
                    throw std::invalid_argument("ambiguous state producer visit");
            }
            if (instance.storage != no_instance) {
                out.pool_alignment = std::max(out.pool_alignment, d.row_alignment);
                instance.pool_offset = align(out.pool_bytes, d.row_alignment);
                out.pool_bytes = add(instance.pool_offset, mul(instance.slot_bytes, slots));
                instance.page_index = out.page_entries;
                out.page_entries = add(out.page_entries, mul(instance.pages, slots));
            }
            out.instances.push_back(instance);
        };

        if (d.per == Per::model) emit(no_instance, no_instance, 0);
        else for (const auto& visit : model.schedule) {
            if (!selected(d, model, visit, options)) continue;
            if (d.per == Per::visit) emit(visit.physical_layer, visit.logical_layer, visit.pass);
            else if (layers.insert(visit.physical_layer).second) emit(visit.physical_layer, no_instance, 0);
        }

        for (auto i = first; i < out.instances.size(); ++i) {
            auto& instance = out.instances[i];
            if (instance.storage == no_instance) {
                const auto producer = producers.find({static_cast<std::uint32_t>(d.producer), instance.pass});
                if (producer == producers.end()) throw std::invalid_argument("state producer is unavailable on visit");
                instance.storage = producer->second;
                instance.slot_bytes = 0;
            }
            hash.number(instance.layer);
            hash.number(instance.visit);
            hash.number(instance.pass);
            hash.number(instance.storage);
            hash.number(instance.pool_offset);
            hash.number(instance.slot_bytes);
            hash.number(instance.row_stride);
            hash.number(instance.pages);
            hash.number(instance.ring);
        }
        hash.number(out.instances.size() - first);
    }
    out.identity = hash.value;
    out.table_entries = mul(out.instances.size(), slots);
    return out;
}

}  // namespace strix::state
