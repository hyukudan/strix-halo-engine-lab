#include "strix/ernie/pair_plan.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <thread>

namespace {
using namespace strix::ernie;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class Operation>
void rejects(Operation operation, const char* message) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

Inventory fake_checkpoint() {
    const std::array bytes{std::byte{11}, std::byte{22}, std::byte{33}, std::byte{44}};
    const auto a = strix::weight::memory_source(bytes, "a.safetensors");
    const auto b = strix::weight::memory_source(bytes, "b.safetensors");
    Inventory inventory;
    inventory.has_manifest = true;
    inventory.tensors.emplace("model.embed_tokens.weight",
        Tensor{Module::embedding, false, {}, "BF16", {1U}, {a, {2U, 2U}}});
    inventory.tensors.emplace("lm_head.weight",
        Tensor{Module::lm_head, false, {}, "BF16", {1U}, {b, {2U, 2U}}});
    inventory.tensors.emplace("model.mtp.weight",
        Tensor{Module::draft_input, true, {}, "BF16", {1U}, {b, {0U, 2U}}});
    for (unsigned i = 0; i < kLayerCount; ++i)
        inventory.tensors.emplace(target_layer_prefix(i) + "self_attn.test",
            Tensor{layer_is_mla(i) ? Module::mla : Module::hca, false, i, "BF16", {1U}, {a, {0U, 2U}}});
    return inventory;
}

strix::device::Topology fake_devices() {
    strix::device::Topology topology;
    topology.nodes.resize(2);
    topology.nodes[0].rank = 0;
    topology.nodes[1].rank = 1;
    topology.nodes[0].memory_bytes = std::uint64_t{1} << 30U;
    topology.nodes[1].memory_bytes = std::uint64_t{2} << 30U;
    strix::device::TopologyLink link{};
    link.a = 0;
    link.b = 1;
    link.link_class = "usb4";
    topology.links.push_back(link);
    return topology;
}

void offsets(const PairRank& rank) {
    std::vector<std::byte> image(static_cast<std::size_t>(rank.memory.weight_resident), std::byte{255});
    std::array<std::byte, 1> staging{};
    std::uint64_t written = 0;
    stream_pair_weights(rank, staging, [&](std::uint64_t offset, std::span<const std::byte> bytes) {
        check(bytes.size() == 1U, "staging chunk exceeds caller's buffer");
        image.at(static_cast<std::size_t>(offset)) = bytes.front();
        ++written;
    });
    check(written == rank.memory.weight_payload, "streamed byte count mismatch");
    for (const auto& t : rank.tensors) {
        check(t.resident_offset % 256U == 0U, "unaligned tensor destination");
        const auto at = static_cast<std::size_t>(t.resident_offset);
        const bool global = t.module == Module::embedding || t.module == Module::lm_head;
        check(image[at] == (global ? std::byte{33} : std::byte{11}) &&
              image[at + 1U] == (global ? std::byte{44} : std::byte{22}) &&
              image[at + 2U] == std::byte{255}, "wrong tensor bytes or padding overwritten");
    }
}

struct Wire {
    std::mutex mutex;
    std::condition_variable ready;
    std::array<std::deque<std::vector<std::byte>>, 2> messages;
    bool cancelled = false;
};

class Transport final : public strix::dist::Transport {
public:
    Transport(Wire& wire, unsigned rank) : wire_(wire), rank_(rank) {}
    void send(std::span<const std::byte> data) override {
        const std::lock_guard lock(wire_.mutex);
        if (wire_.cancelled) throw std::runtime_error("Fake link cancelled");
        wire_.messages[rank_].emplace_back(data.begin(), data.end());
        wire_.ready.notify_all();
    }
    void receive(std::span<std::byte> out) override {
        std::unique_lock lock(wire_.mutex);
        auto& incoming = wire_.messages[1U - rank_];
        if (!wire_.ready.wait_for(lock, std::chrono::seconds(5),
            [&] { return wire_.cancelled || !incoming.empty(); }))
            throw std::runtime_error("Fake link timed out");
        if (incoming.empty() || incoming.front().size() != out.size())
            throw std::runtime_error("Fake link cancelled or framed incorrectly");
        std::ranges::copy(incoming.front(), out.begin());
        incoming.pop_front();
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        send(out);
        receive(in);
    }
    std::string name() const override { return "fake"; }
    bool supports_peer_abort() const noexcept override { return true; }
    void abort(const std::string&) override {
        const std::lock_guard lock(wire_.mutex);
        wire_.cancelled = true;
        wire_.ready.notify_all();
    }
private:
    Wire& wire_;
    unsigned rank_;
};

void first_token(const PairPlan& plan, bool mismatch) {
    constexpr std::size_t floats_per_row = kPairRowBytes / sizeof(float);
    constexpr std::array prompt{2U, 7U, 19U};
    Wire wire;
    std::array<std::uint32_t, 2> output{}, calls{};
    std::array<std::exception_ptr, 2> errors{};
    std::array<std::jthread, 2> threads;
    for (unsigned rank = 0; rank < 2; ++rank) {
        threads[rank] = std::jthread([&, rank] {
            try {
                Transport link(wire, rank);
                std::vector<float> residual(floats_per_row * plan.limits.prefill_rows);
                std::uint32_t best = 0;
                const auto row_value = [](std::uint32_t token, std::size_t column) {
                    return static_cast<float>(token + column);
                };
                PairFirstTokenOps ops;
                ops.forward = [&](std::span<const std::uint32_t> ids, const strix::pair::layers::Window& window,
                                  std::uint32_t begin, std::uint32_t end, bool head) {
                    const auto& p = plan.ranks[rank].placement;
                    check(begin == p.first() && end == p.last(), "device received a foreign layer range");
                    check(head == (rank == 1U && window.start + window.rows == prompt.size()), "LM head ownership");
                    ++calls[rank];
                    for (std::size_t i = 0; i < ids.size(); ++i) {
                        for (std::size_t j = 0; j < floats_per_row; ++j) {
                            const float original = row_value(ids[i], j);
                            auto& value = residual[i * floats_per_row + j];
                            if (rank == 0) value = original;
                            else check(value == original + static_cast<float>(begin), "F32 seam payload changed");
                            value += static_cast<float>(end - begin);
                        }
                    }
                    if (head) best = static_cast<std::uint32_t>(residual[(ids.size() - 1U) * floats_per_row]);
                };
                ops.seam = [&](strix::pair::PipeFrame& frame, bool save, std::size_t bytes) {
                    check(save == (rank == 0) && bytes <= residual.size() * sizeof(float), "wrong seam extent");
                    if (save) std::memcpy(frame.payload.data(), residual.data(), bytes);
                    else std::memcpy(residual.data(), frame.payload.data(), bytes);
                };
                ops.fence = [] {};
                if (rank == 1U) ops.top = [&] { return best; };
                const std::string assets(64U, rank == 1U && mismatch ? 'b' : 'a');
                output[rank] = run_pair_first_token(plan, rank, prompt, assets, "fake-fp32", link, ops);
            } catch (...) {
                errors[rank] = std::current_exception();
            }
        });
    }
    for (auto& thread : threads) thread.join();
    if (mismatch) {
        check(errors[0] && errors[1] && calls[0] == 0U && calls[1] == 0U,
              "asset mismatch did not stop both ranks before prefill");
    } else {
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        const auto expected = prompt.back() + kLayerCount;
        check(output[0] == expected && output[1] == expected && calls[0] == 2U && calls[1] == 2U,
              "two-rank prefill or greedy relay diverged");
    }
}

} // namespace

int main() try {
    auto inventory = fake_checkpoint();
    auto topology = fake_devices();
    const PairLimits limits{4096U, 2U, 256U, 65536U, {0U, 1024U}, {0U, 2048U}};
    const auto plan = make_pair_plan(inventory, topology, limits, "usb4-auto");
    const auto& a = plan.ranks[0];
    const auto& b = plan.ranks[1];
    check(plan.admitted() && plan.transport == "usb4auto", "fake devices should be admitted");
    check(a.placement.embeds() && !a.placement.head() && a.placement.last() == 22U &&
          !b.placement.embeds() && b.placement.head() && b.placement.first() == 22U && b.placement.last() == 43U,
          "22/21 cut and global ownership");
    check(a.layers.size() == 22U && b.layers.size() == 21U && !a.layers.front().routed &&
          a.layers[7].attention == PairAttention::sparse_mla &&
          b.layers.back().attention == PairAttention::sparse_mla, "native layer descriptors");
    check(a.layers.front().history_rows == 128U && a.layers.front().compressed_rows == 32U &&
          a.layers.front().compression_ratio == 128U && a.layers[7].history_rows == 4096U,
          "local attention state capacities");
    check(a.memory.kv_payload == 9715200U && b.memory.kv_payload == 9630720U &&
          a.memory.carry == 19U * 524288U && b.memory.carry == 18U * 524288U &&
          a.memory.logits == 0U && b.memory.logits == 804864U && kPairRowBytes == 65536U,
          "KV, carry, logits and seam bytes");
    check(a.memory.weight_payload == 46U && b.memory.weight_payload == 44U &&
          a.memory.weight_resident == 23U * 256U && b.memory.weight_resident == 22U * 256U &&
          plan.draft_bytes == 2U, "resident offsets and draft exclusion");
    // The plan's cut is the shared layer-chain planner's choice for ERNIE's per-layer state, with the priced
    // layer-owned decode, and the rank memory admission accepts that cut.
    const auto input = pair_chain_input(limits);
    check(input.request.price_layer_owned_decode && input.request.layer_owned_state &&
          input.shape.state_fixed_bytes.size() == kLayerCount, "chain input declares layer-owned priced decode");
    for (std::uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const auto state = describe_pair_layer(layer, limits.context);
        check(input.shape.state_fixed_bytes[layer] == static_cast<double>(state.kv_bytes + state.carry_bytes),
              "chain layer state differs from the plan's layer state");
    }
    const auto chosen = strix::pair::chain::choose(input.devices, input.links, input.shape, input.request).best;
    const auto cut = pair_cut(limits);
    check(chosen.valid && chosen.cuts.size() == 3U && chosen.cuts[1] == cut && chosen.decoder == 1U &&
          chosen.decode_ms_per_token > 0.0, "pair_cut is the chain planner's cut");
    check(a.placement.cuts == chosen.cuts && b.placement.cuts == chosen.cuts &&
          a.memory.admitted() && b.memory.admitted(), "plan places and admits the chain's cut");
    offsets(a);
    offsets(b);
    first_token(plan, false);
    first_token(plan, true);

    topology.nodes[1].memory_bytes = b.memory.required;
    check(make_pair_plan(inventory, topology, limits, "usb4auto").admitted(), "exact admission boundary");
    --topology.nodes[1].memory_bytes;
    const auto small = make_pair_plan(inventory, topology, limits, "usb4auto");
    check(small.ranks[0].memory.admitted() && !small.admitted(), "second rank must refuse one byte short");
    std::array<std::byte, 1> staging{};
    rejects([&] { stream_pair_weights(small.ranks[1], staging, [](auto, auto) {}); },
            "rejected rank loaded weights");
    rejects([&] { (void)make_pair_plan(inventory, topology, limits, "usb4gpu"); },
            "unqualified direct transport accepted");
    auto overflow = limits;
    overflow.alignment = std::uint64_t{1} << 63U;
    rejects([&] { (void)make_pair_plan(inventory, topology, overflow, "usb4auto"); }, "memory wraparound accepted");
    check(pair_owner("model.layers.21.x", cut) == 0U && pair_owner("model.layers.22.x", cut) == 1U &&
          pair_owner("model.embed_tokens.weight", cut) == 0U && pair_owner("model.norm.weight", cut) == 1U &&
          !pair_owner("model.mtp.0.x", cut), "tensor ownership follows the planner's cut");

    // Range admission: the rank-0 copy omits the shard holding only rank-1 and drafter tensors.
    auto rank0 = inventory;
    for (unsigned i = cut; i < kLayerCount; ++i) {
        auto name = target_layer_prefix(i) + "self_attn.test";
        rank0.tensors.erase(name);
        rank0.missing_known.push_back(std::move(name));
    }
    for (const char* name : {"lm_head.weight", "model.mtp.weight"}) {
        rank0.tensors.erase(name);
        rank0.missing_known.emplace_back(name);
    }
    rank0.missing_shards.emplace_back("b.safetensors");
    rank0.missing_known_tensors = rank0.missing_known.size();
    rejects([&] { (void)make_pair_plan(rank0, topology, limits, "usb4auto"); },
            "whole plan accepted a single-rank copy");
    const auto local = make_pair_plan(rank0, topology, limits, "usb4auto", 0U);
    bool same = local.ranks[0].tensors.size() == a.tensors.size();
    for (std::size_t i = 0; same && i < a.tensors.size(); ++i)
        same = local.ranks[0].tensors[i].name == a.tensors[i].name &&
               local.ranks[0].tensors[i].resident_offset == a.tensors[i].resident_offset;
    check(local.admitted() && local.local == 0U && same && local.ranks[1].tensors.empty() &&
          local.ranks[0].memory.required == a.memory.required,
          "rank-scoped plan differs from the whole plan for its own rank");
    rejects([&] { (void)make_pair_plan(rank0, topology, limits, "usb4auto", 1U); },
            "rank 1 admitted from a copy without its tensors");
    rank0.missing_known.push_back(target_layer_prefix(0) + "self_attn.other");
    rank0.missing_known_tensors = rank0.missing_known.size();
    rejects([&] { (void)make_pair_plan(rank0, topology, limits, "usb4auto", 0U); },
            "rank 0 admitted without one of its own tensors");

    inventory.missing_shards.push_back("missing");
    rejects([&] { (void)make_pair_plan(inventory, topology, limits, "usb4auto"); },
            "partial checkpoint accepted");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
