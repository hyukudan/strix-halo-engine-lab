#include "strix/ernie/pair_plan.hpp"

#include "strix/deploy/binding.hpp"
#include "strix/dist/transport_spec.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/pair/first_token.hpp"
#include "strix/pair/rank_image.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

namespace strix::ernie {
namespace {

namespace layers = pair::layers;

constexpr auto kWindow = model::narrow<std::uint32_t>(data::attention::sliding_window);
constexpr auto kHcaWidth = model::narrow<std::uint64_t>(data::dims::head_dim);
constexpr auto kKvBlock = model::narrow<std::uint64_t>(data::weights::checkpoint::weight_block_size[1]);
constexpr std::uint32_t kPairProtocol = 0x324E5245U;  // "ERN2", the FourCC style of the other engines' pair hellos
constexpr std::uint64_t kPlanBlocks = 16;
constexpr std::string_view kHelloArch = "cpu";          // host first-token path: no device architecture
constexpr SparseLatentRowDims kMlaRow{model::narrow<std::uint64_t>(data::attention::mla::kv_lora_rank), kKvBlock,
                                      model::narrow<std::uint64_t>(data::attention::mla::qk_rope_head_dim),
                                      model::narrow<std::uint64_t>(data::attention::indexer::index_head_dim),
                                      model::narrow<std::uint64_t>(data::weights::checkpoint::expert_block_columns)};

// Candidate compact layouts (shared formulas, ERNIE dimensions).
static_assert(kHcaWidth % kKvBlock == 0 && kMlaRow.valid());
constexpr bool compress_ratios_valid() {
    for (unsigned layer = 0; layer < kLayerCount; ++layer)
        if (data::attention::compress_ratios[layer] == 0) return false;  // negative marks a sparse-MLA layer
    return true;
}
static_assert(compress_ratios_valid(), "a compressed-history layer needs a ratio of at least 1");
constexpr auto kHcaEntry = block_scaled_fp8_row_bytes(kHcaWidth, kKvBlock);
constexpr auto kMlaEntry = sparse_latent_row_bytes(kMlaRow);

using deploy::checked_add;

PairLayer describe_layer(std::uint32_t layer, std::uint32_t context) {
    PairLayer result{};
    result.id = layer;
    result.routed = layer_has_moe(layer);
    if (layer_is_mla(layer)) {
        result.attention = PairAttention::sparse_mla;
        result.history_rows = context;
        result.kv_bytes = context * kMlaEntry;
    } else {
        result.attention = PairAttention::hca;
        result.compression_ratio = static_cast<std::uint32_t>(data::attention::compress_ratios[layer]);
        const auto rows = compressed_history_rows(context, kWindow, result.compression_ratio);
        result.history_rows = rows.history_rows;
        result.compressed_rows = rows.compressed_rows;
        result.kv_bytes = (result.history_rows + result.compressed_rows) * kHcaEntry;
        // Unfused carry budget: retain projection and gate in F32 for a
        // complete incomplete-group window; APE itself is in the weights.
        result.carry_bytes = compressed_history_carry_bytes(result.compression_ratio, kHcaWidth);
    }
    return result;
}

// Chain declaration: no measured ERNIE rates exist, so every layer costs one unit (the head one more), both ranks read at one declared
// rate, and a link hop moves one activation row per ms. State is the exact per-layer cache bytes at the context.
}  // namespace

PairLayer describe_pair_layer(std::uint32_t layer, std::uint32_t context) { return describe_layer(layer, context); }

PairChainInput pair_chain_input(const PairLimits& limits) {
    if (limits.context == 0 || limits.context > data::dims::max_context || limits.prefill_rows == 0 ||
        limits.prefill_rows > limits.context)
        throw std::invalid_argument("ERNIE pair: invalid context or prefill width");
    PairChainInput in;
    const auto rate = static_cast<double>(kPairRowBytes);
    for (std::uint32_t rank = 0; rank < 2U; ++rank) {
        pair::chain::Device device;
        device.name = "rank" + std::to_string(rank);
        device.layer_ms_per_row.assign(kLayerCount, 1.0);
        device.tail_ms_per_row = 1.0;  // final norm + head count as one more unit on the last stage
        device.read_bytes_per_ms = rate;
        in.devices.push_back(std::move(device));
    }
    in.links.push_back({rate, rate, 0.0});
    in.shape.layers = kLayerCount;
    in.shape.seam_bytes_per_row = rate;
    in.shape.state_bytes_per_row.assign(kLayerCount, 0.0);
    for (std::uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const auto state = describe_layer(layer, limits.context);
        const auto bytes = static_cast<double>(state.kv_bytes + state.carry_bytes);
        in.shape.state_fixed_bytes.push_back(bytes);
        in.shape.read_bytes_per_token.push_back(bytes);
    }
    in.shape.head_bytes = static_cast<double>(kPairLogitsBytes);
    // A representative prompt of kPlanBlocks windows of the prefill width: the flow shop only separates cuts once
    // two or more windows overlap across the ranks.
    in.request.split = pair::chain::Split::layer_cut;
    in.request.cuts = {0U};
    in.request.rows = std::uint64_t{limits.prefill_rows} * kPlanBlocks;
    in.request.blocks = {limits.prefill_rows};
    in.request.decoders = {1U};
    in.request.max_stages = 2U;
    in.request.state_rows = limits.context;
    in.request.decode_tokens = 1.0;
    in.request.layer_owned_state = true;
    in.request.price_layer_owned_decode = true;
    return in;
}

std::uint32_t pair_cut(const PairLimits& limits) {
    const auto in = pair_chain_input(limits);
    const auto plan = pair::chain::choose(in.devices, in.links, in.shape, in.request).best;
    if (!plan.valid || plan.cuts.size() != 3U)
        throw std::invalid_argument("ERNIE pair: the layer planner found no cut: " + plan.reason);
    return plan.cuts[1];
}

namespace {

PairPlan build_pair_plan(const Inventory& inventory, const device::Topology& topology,
                          const PairLimits& limits, std::string transport,
                          std::optional<std::uint32_t> local_rank) {
    const auto owns = [&](std::uint32_t rank) { return pair::covers_rank(local_rank, rank); };
    if (!inventory.has_manifest ||
        (!local_rank && (!inventory.missing_shards.empty() || inventory.missing_known_tensors != 0)))
        throw std::invalid_argument("ERNIE pair: a complete validated checkpoint inventory is required");
    const auto cut = pair_cut(limits);
    if (local_rank) {
        // Range admission: a rank copy may omit shards that hold only the peer's tensors.
        if (inventory.missing_known.size() != inventory.missing_known_tensors)
            throw std::invalid_argument("ERNIE pair: rank scope needs the missing tensor names");
        pair::check_rank_scope(*local_rank, inventory.missing_known, [&](std::string_view name) { return pair_owner(name, cut); });
    }
    if (topology.nodes.size() != 2 || topology.nodes[0].rank != 0 || topology.nodes[1].rank != 1 ||
        !std::ranges::any_of(topology.links, [](const auto& link) { return link.a == 0 && link.b == 1; }))
        throw std::invalid_argument("ERNIE pair: topology requires ranks 0 and 1 with a direct link");
    // The bring-up plan uses the existing USB4 host transport. Device buffers,
    // TCP endpoints and disjoint directed paths can bind after qualification.
    const dist::TransportSpellings host_only{"usb4auto", {}, "usb4-auto", false, false, false, {}};
    (void)dist::parse_transport(transport, host_only);
    if (limits.context == 0 || limits.context > data::dims::max_context ||
        limits.prefill_rows == 0 || limits.prefill_rows > limits.context ||
        !std::has_single_bit(limits.alignment) || limits.load_chunk_bytes == 0)
        throw std::invalid_argument("ERNIE pair: invalid context, prefill width, alignment, or staging limit");

    PairPlan plan{};
    plan.transport = std::string(dist::pair_transport.host);
    plan.limits = limits;
    std::array<std::size_t, kLayerCount> layer_tensor_count{};
    bool embedding = false, head = false;
    for (const auto& [name, tensor] : inventory.tensors) {
        const auto bytes = tensor.payload.range.bytes;
        if (tensor.module == Module::unclassified || bytes == 0 || !tensor.payload.source)
            throw std::invalid_argument("ERNIE pair: invalid source descriptor for " + name);
        weight::Slice verified;
        if (const auto status = weight::subslice(tensor.payload, {0, bytes}, verified); !status)
            throw std::invalid_argument(name + ": " + status.detail);
        if (tensor.drafter) {
            plan.draft_bytes = checked_add(plan.draft_bytes, bytes);
            continue;
        }
        if (tensor.layer && *tensor.layer >= kLayerCount)
            throw std::invalid_argument("ERNIE pair: target layer out of range for " + name);
        const bool first = tensor.layer ? *tensor.layer < cut : tensor.module == Module::embedding;
        if (!owns(first ? 0U : 1U)) continue; // Peer-owned tensor in a shard both ranks read.
        if (tensor.layer) ++layer_tensor_count[*tensor.layer];
        embedding |= !tensor.layer && tensor.module == Module::embedding;
        head |= !tensor.layer && tensor.module == Module::lm_head && name == "lm_head.weight";
        plan.ranks[first ? 0U : 1U].tensors.push_back(
            {name, verified.source->shard_name(), tensor.module, tensor.layer, tensor.wire_dtype,
             tensor.shape, std::move(verified), 0});
    }
    bool all_layers = true;
    for (std::uint32_t layer = 0; layer < kLayerCount; ++layer)
        all_layers = all_layers && (!owns(layer < cut ? 0U : 1U) || layer_tensor_count[layer] != 0U);
    if ((owns(0U) && !embedding) || (owns(1U) && !head) || !all_layers)
        throw std::invalid_argument("ERNIE pair: missing embedding, head or target layer");
    plan.local = local_rank;

    for (std::uint32_t rank = 0; rank < plan.ranks.size(); ++rank) {
        auto& out = plan.ranks[rank];
        out.placement = {{0U, cut, kLayerCount}, rank};
        out.placement.check();
        const auto image = pair::pack_rank_image(out.tensors, limits.alignment);
        auto& memory = out.memory;
        memory.usable = topology.nodes[rank].memory_bytes;
        memory.weight_payload = image.payload;
        memory.weight_resident = image.resident;
        std::vector<pair::LayerState> state;
        for (auto layer = out.placement.first(); layer < out.placement.last(); ++layer) {
            const auto descriptor = describe_layer(layer, limits.context);
            state.push_back({descriptor.kv_bytes, descriptor.carry_bytes});
            out.layers.push_back(descriptor);
        }
        pair::RankTerms terms;
        terms.alignment = limits.alignment;
        terms.load_chunk_bytes = limits.load_chunk_bytes;
        terms.largest_tensor = image.largest;
        terms.row_bytes = kPairRowBytes;
        terms.rows = limits.prefill_rows;
        terms.logits_bytes = out.placement.head() ? kPairLogitsBytes : 0U;
        terms.scratch = limits.scratch[rank];
        terms.reserve = limits.reserve[rank];
        terms.layers = state;
        pair::account_rank_memory(memory, terms);
    }
    return plan;
}

} // namespace

std::optional<std::uint32_t> pair_owner(std::string_view name, std::uint32_t cut) {
    if (name.starts_with("model.mtp.")) return std::nullopt;
    if (const auto layer = target_layer(name)) return *layer < cut ? 0U : 1U;
    return name == "model.embed_tokens.weight" ? 0U : 1U;
}

PairPlan make_pair_plan(const Inventory& inventory, const device::Topology& topology,
                        const PairLimits& limits, std::string transport,
                        std::optional<std::uint32_t> local_rank) {
    try {
        return build_pair_plan(inventory, topology, limits, std::move(transport), local_rank);
    } catch (const std::overflow_error&) {
        throw std::invalid_argument("ERNIE pair: byte count overflows");
    }
}

void stream_pair_weights(const PairRank& rank, std::span<std::byte> staging, const PairWeightWrite& write) {
    if (!rank.memory.admitted()) throw std::invalid_argument("ERNIE pair: rank exceeds usable memory");
    if (staging.size() > rank.memory.load_staging)
        throw std::invalid_argument("ERNIE pair: weight staging exceeds the admitted chunk");
    pair::stream_rank_image(std::span<const PairTensor>(rank.tensors), staging, write);
}

std::uint32_t run_pair_first_token(const PairPlan& plan, std::uint32_t rank,
    std::span<const std::uint32_t> prompt, std::string_view asset_sha256,
    std::string_view numeric_identity, dist::Transport& link, const PairFirstTokenOps& ops) {
    if (!plan.admitted() || rank >= plan.ranks.size() || (plan.local && *plan.local != rank)) {
        link.abort("ERNIE pair: first-token session failed");
        throw std::invalid_argument("ERNIE pair: an admitted two-rank plan is required");
    }
    pair::layers::FirstTokenSpec spec;
    spec.protocol = kPairProtocol;
    spec.cut = plan.ranks[rank].placement.cuts.at(1);
    spec.layers = kLayerCount;
    spec.context = plan.limits.context;
    spec.prefill_rows = plan.limits.prefill_rows;
    spec.vocab = data::dims::vocab;
    spec.row_bytes = kPairRowBytes;
    spec.gpu_arch = kHelloArch;
    return pair::layers::first_token(plan.ranks[rank].placement, rank, prompt, spec, asset_sha256, numeric_identity,
                                     link, ops);
}

} // namespace strix::ernie
