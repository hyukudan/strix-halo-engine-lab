#include "strix/ernie/pair_plan.hpp"

#include "strix/dist/transport_spec.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

namespace strix::ernie {
namespace {

namespace layers = pair::layers;

constexpr auto kWindow = model::narrow<std::uint32_t>(data::attention::sliding_window);
constexpr auto kHcaWidth = model::narrow<std::uint64_t>(data::dims::head_dim);
constexpr auto kMlaRank = model::narrow<std::uint64_t>(data::attention::mla::kv_lora_rank);
constexpr auto kMlaRope = model::narrow<std::uint64_t>(data::attention::mla::qk_rope_head_dim);
constexpr auto kIndexWidth = model::narrow<std::uint64_t>(data::attention::indexer::index_head_dim);
constexpr auto kKvBlock = model::narrow<std::uint64_t>(data::weights::checkpoint::weight_block_size[1]);
constexpr auto kIndexBlock = model::narrow<std::uint64_t>(data::weights::checkpoint::expert_block_columns);

// Candidate compact layouts: FP8 KV + F32 scales; split MLA rotary FP8 and
// separate E2M1 index keys (one byte of E8M0 per group).
static_assert(kHcaWidth % kKvBlock == 0 && kMlaRank % kKvBlock == 0 && kIndexWidth % kIndexBlock == 0);
constexpr auto kHcaEntry = kHcaWidth + (kHcaWidth / kKvBlock) * sizeof(float);
constexpr auto kMlaEntry = kMlaRank + (kMlaRank / kKvBlock) * sizeof(float) + kMlaRope +
                           kIndexWidth / 2U + kIndexWidth / kIndexBlock;

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
        throw std::invalid_argument("ERNIE pair: byte count overflows");
    return a + b;
}

std::uint64_t aligned(std::uint64_t bytes, std::uint64_t alignment) {
    const auto remainder = bytes & (alignment - 1U);
    return remainder == 0 ? bytes : add(bytes, alignment - remainder);
}

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
        result.history_rows = std::min(context, kWindow);
        result.compressed_rows = (context + result.compression_ratio - 1U) / result.compression_ratio;
        result.kv_bytes = (result.history_rows + result.compressed_rows) * kHcaEntry;
        // Unfused carry budget: retain projection and gate in F32 for a
        // complete incomplete-group window; APE itself is in the weights.
        result.carry_bytes = result.compression_ratio * kHcaWidth * 2U * sizeof(float);
    }
    return result;
}

bool sha256_spelling(std::string_view digest) {
    return digest.size() == 64 && std::ranges::all_of(digest, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

} // namespace

PairPlan make_pair_plan(const Inventory& inventory, const device::Topology& topology,
                        const PairLimits& limits, std::string transport) {
    if (!inventory.has_manifest || !inventory.missing_shards.empty() || inventory.missing_known_tensors != 0)
        throw std::invalid_argument("ERNIE pair: a complete validated checkpoint inventory is required");
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
            plan.draft_bytes = add(plan.draft_bytes, bytes);
            continue;
        }
        if (tensor.layer && *tensor.layer >= kLayerCount)
            throw std::invalid_argument("ERNIE pair: target layer out of range for " + name);
        if (tensor.layer) ++layer_tensor_count[*tensor.layer];
        embedding |= !tensor.layer && tensor.module == Module::embedding;
        head |= !tensor.layer && tensor.module == Module::lm_head && name == "lm_head.weight";
        const bool first = tensor.layer ? *tensor.layer < kPairCut : tensor.module == Module::embedding;
        plan.ranks[first ? 0U : 1U].tensors.push_back(
            {name, verified.source->shard_name(), tensor.module, tensor.layer, tensor.wire_dtype,
             tensor.shape, std::move(verified), 0});
    }
    if (!embedding || !head || std::ranges::find(layer_tensor_count, 0U) != layer_tensor_count.end())
        throw std::invalid_argument("ERNIE pair: missing embedding, head or target layer");

    for (std::uint32_t rank = 0; rank < plan.ranks.size(); ++rank) {
        auto& out = plan.ranks[rank];
        out.placement = {{0U, kPairCut, kLayerCount}, rank};
        out.placement.check();
        // Compute the packed destination offsets in the source read order.
        std::ranges::sort(out.tensors, {}, [](const PairTensor& item) {
            return std::tie(item.shard, item.source.range.offset, item.name);
        });
        auto& memory = out.memory;
        std::uint64_t largest_tensor = 0;
        memory.usable = topology.nodes[rank].memory_bytes;
        for (auto& item : out.tensors) {
            item.resident_offset = aligned(memory.weight_resident, limits.alignment);
            memory.weight_resident = add(item.resident_offset, item.source.range.bytes);
            memory.weight_payload = add(memory.weight_payload, item.source.range.bytes);
            largest_tensor = std::max(largest_tensor, item.source.range.bytes);
        }
        memory.weight_resident = aligned(memory.weight_resident, limits.alignment);
        for (auto layer = out.placement.first(); layer < out.placement.last(); ++layer) {
            const auto descriptor = describe_layer(layer, limits.context);
            memory.kv_payload = add(memory.kv_payload, descriptor.kv_bytes);
            memory.kv_resident = add(memory.kv_resident, aligned(descriptor.kv_bytes, limits.alignment));
            memory.carry = add(memory.carry, aligned(descriptor.carry_bytes, limits.alignment));
            out.layers.push_back(descriptor);
        }
        memory.logits = out.placement.head() ? aligned(kPairLogitsBytes, limits.alignment) : 0U;
        // Two seam slots plus the runner's in/out activation buffer.
        const auto seam_slot = aligned(kPairRowBytes * limits.prefill_rows, limits.alignment);
        memory.exchange = add(add(seam_slot, seam_slot), seam_slot);
        memory.load_staging = aligned(std::min(limits.load_chunk_bytes, largest_tensor), limits.alignment);
        memory.scratch = aligned(limits.scratch[rank], limits.alignment);
        memory.reserve = aligned(limits.reserve[rank], limits.alignment);
        for (const auto bytes : {memory.weight_resident, memory.kv_resident, memory.carry, memory.logits,
                                 memory.exchange, memory.load_staging, memory.scratch, memory.reserve})
            memory.required = add(memory.required, bytes);
    }
    return plan;
}

void stream_pair_weights(const PairRank& rank, std::span<std::byte> staging, const PairWeightWrite& write) {
    if (!rank.memory.admitted()) throw std::invalid_argument("ERNIE pair: rank exceeds usable memory");
    if (staging.empty() || staging.size() > rank.memory.load_staging || !write)
        throw std::invalid_argument("ERNIE pair: invalid weight staging or writer");
    for (const auto& tensor : rank.tensors) {
        for (std::uint64_t offset = 0; offset < tensor.source.range.bytes;) {
            const auto count = std::min<std::uint64_t>(staging.size(), tensor.source.range.bytes - offset);
            const auto chunk = staging.first(static_cast<std::size_t>(count));
            weight::Slice slice;
            if (const auto status = weight::subslice(tensor.source, {offset, count}, slice); !status)
                throw std::runtime_error(tensor.name + ": " + status.detail);
            if (const auto status = weight::read_slice(slice, chunk); !status)
                throw std::runtime_error(tensor.name + ": " + status.detail);
            write(tensor.resident_offset + offset, chunk);
            offset += count;
        }
    }
}

std::uint32_t run_pair_first_token(const PairPlan& plan, std::uint32_t rank,
    std::span<const std::uint32_t> prompt, std::string_view asset_sha256,
    std::string_view numeric_identity, dist::Transport& link, const PairFirstTokenOps& ops) {
    try {
        if (!plan.admitted() || rank >= plan.ranks.size() || !link.supports_peer_abort())
            throw std::invalid_argument("ERNIE pair: an admitted two-rank plan and abortable link are required");
        const auto& placement = plan.ranks[rank].placement;
        const auto valid_token = [](std::uint32_t id) { return id < data::dims::vocab; };
        if (prompt.empty() || prompt.size() > plan.limits.context || !std::ranges::all_of(prompt, valid_token) ||
            !sha256_spelling(asset_sha256) || numeric_identity.empty() ||
            !ops.forward || !ops.seam || !ops.fence || (placement.head() && !ops.top))
            throw std::invalid_argument("ERNIE pair: invalid first-token prompt, identity or device hooks");

        const std::array<std::uint32_t, 5> layout{
            kPairCut, kLayerCount, static_cast<std::uint32_t>(kPairRowBytes), plan.limits.context, plan.limits.prefill_rows};
        const auto assets = layers::fnv(14695981039346656037ULL, asset_sha256);
        const auto layout_key = layers::fnv(layers::fnv(assets, numeric_identity), layout);
        const auto identity = layers::fnv(layout_key, prompt);
        constexpr std::uint64_t magic = 0x3152494150455253ULL;
        std::array<std::uint64_t, 12> hello{magic, rank, identity, prompt.size()};
        std::memcpy(hello.data() + 4, asset_sha256.data(), asset_sha256.size());
        std::array<std::uint64_t, hello.size()> peer{};
        link.exchange(std::as_bytes(std::span(hello)), std::as_writable_bytes(std::span(peer)));
        if (peer[0] != magic || peer[1] != 1U - rank || peer[2] != identity || peer[3] != prompt.size() ||
            !std::equal(hello.begin() + 4, hello.end(), peer.begin() + 4))
            throw std::runtime_error("ERNIE pair: peer rank, assets, geometry, numerics or prompt identity mismatch");

        const layers::Link channel{
            [&](std::span<const std::byte> bytes) { link.send(bytes); },
            [&](std::span<std::byte> bytes) { link.receive(bytes); }};
        const auto windows = layers::windows(prompt.size(), plan.limits.prefill_rows).size();
        pair::PipeOptions options;
        options.validate = layers::validator(identity, windows, kPairCut, plan.limits.prefill_rows, kPairRowBytes);
        pair::Pipe pipe(rank, windows, channel.send, channel.receive,
                        [&] { link.abort("ERNIE pair: prefill pipe failed"); }, std::move(options));
        layers::StageOps prefill;
        prefill.forward = [&](std::span<const std::uint32_t> ids, const layers::Window& window, bool head) {
            ops.forward(ids, window, placement.first(), placement.last(), head);
        };
        prefill.seam = ops.seam;
        prefill.fence = ops.fence;
        prefill.row_bytes = kPairRowBytes;
        (void)layers::prefill(placement, prompt, plan.limits.prefill_rows, identity,
                              placement.embeds() ? nullptr : &pipe, placement.head() ? nullptr : &pipe, prefill);

        layers::DecodeOps first;
        first.row_bytes = kPairRowBytes;
        const auto token = placement.head() ? ops.top() : 0U;
        if (!valid_token(token)) throw std::runtime_error("ERNIE pair: first token is out of vocabulary");
        const auto result = layers::decode(placement, token, 1U,
            placement.embeds() ? nullptr : &channel, placement.head() ? nullptr : &channel, first).ids.front();
        if (!valid_token(result)) throw std::runtime_error("ERNIE pair: peer relayed an out-of-vocabulary token");
        return result;
    } catch (...) {
        link.abort("ERNIE pair: first-token session failed");
        throw;
    }
}

} // namespace strix::ernie
