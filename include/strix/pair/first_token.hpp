#pragma once
// Two-rank first-token driver over the shared layer pipeline: a pair hello that binds assets, geometry, numerics
// and prompt, the shared prefill pipe, and the first greedy decode relay. The model supplies the geometry and its
// device hooks; a fresh device session is required after any failure (the link is aborted).
#include "strix/dist/transport.hpp"
#include "strix/pair/handshake.hpp"
#include "strix/pair/layer_pipeline.hpp"
#include "strix/weight/levers.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <span>
#include <stdexcept>
#include <string_view>

namespace strix::pair::layers {

struct FirstTokenSpec {
    std::uint32_t protocol{}, cut{}, layers{}, context{}, prefill_rows{}, vocab{};
    std::size_t row_bytes{};
    // Names of the contract slots reported on a hello mismatch (see first_token_contract).
    static constexpr std::array<std::string_view, 7> kFields{
        "layer cut", "layer count", "activation row bytes", "context", "prefill width",
        "numeric or prompt identity", "prompt length"};
};

struct FirstTokenOps {
    decltype(StageOps::range) forward;
    decltype(StageOps::seam) seam;
    std::function<void()> fence;
    std::function<std::uint32_t()> top;
};

// Both ranks verify the same asset digest, geometry, numeric identity and prompt before any activation is sent.
[[nodiscard]] inline std::uint32_t first_token(const Placement& placement, std::uint32_t rank,
    std::span<const std::uint32_t> prompt, const FirstTokenSpec& spec, std::string_view asset_sha256,
    std::string_view numeric_identity, dist::Transport& link, const FirstTokenOps& ops) {
    try {
        const auto digest = weight::digest_from_hex(asset_sha256);
        const auto valid_token = [&](std::uint32_t id) { return id < spec.vocab; };
        if (rank >= 2U || !link.supports_peer_abort())
            throw std::invalid_argument("pair first token: a two-rank plan and an abortable link are required");
        if (prompt.empty() || prompt.size() > spec.context || !std::ranges::all_of(prompt, valid_token) ||
            digest.bytes == std::array<std::uint8_t, 32>{} || numeric_identity.empty() ||
            !ops.forward || !ops.seam || !ops.fence || (placement.head() && !ops.top))
            throw std::invalid_argument("pair first token: invalid prompt, identity or device hooks");

        const std::array<std::uint32_t, 5> layout{spec.cut, spec.layers, static_cast<std::uint32_t>(spec.row_bytes),
                                                  spec.context, spec.prefill_rows};
        const auto assets = fnv(14695981039346656037ULL, asset_sha256);
        const auto identity = fnv(fnv(fnv(assets, numeric_identity), layout), prompt);
        PairHello hello;
        hello.protocol = spec.protocol;
        hello.rank = rank;
        hello.model_sha256 = digest.bytes;
        hello.context = spec.context;
        hello.max_rows = spec.prefill_rows;
        hello.placement = spec.cut;
        hello.contract = {spec.cut, spec.layers, spec.row_bytes, spec.context, spec.prefill_rows, identity,
                          prompt.size()};
        (void)exchange_hello(hello, [&](std::span<const std::byte> out, std::span<std::byte> in) {
            link.exchange(out, in);
        }, FirstTokenSpec::kFields);

        const Link channel{
            [&](std::span<const std::byte> bytes) { link.send(bytes); },
            [&](std::span<std::byte> bytes) { link.receive(bytes); }};
        const auto windows_count = windows(prompt.size(), spec.prefill_rows).size();
        PipeOptions options;
        options.validate = validator(identity, windows_count, spec.cut, spec.prefill_rows, spec.row_bytes);
        Pipe pipe(rank, windows_count, channel.send, channel.receive,
                  [&] { link.abort("pair first token: prefill pipe failed"); }, std::move(options));
        StageOps prefill_ops;
        prefill_ops.forward = [&](std::span<const std::uint32_t> ids, const Window& window, bool head) {
            ops.forward(ids, window, placement.first(), placement.last(), head);
        };
        prefill_ops.seam = ops.seam;
        prefill_ops.fence = ops.fence;
        prefill_ops.row_bytes = spec.row_bytes;
        (void)prefill(placement, prompt, spec.prefill_rows, identity,
                      placement.embeds() ? nullptr : &pipe, placement.head() ? nullptr : &pipe, prefill_ops);

        DecodeOps first;
        first.row_bytes = spec.row_bytes;
        const auto token = placement.head() ? ops.top() : 0U;
        if (!valid_token(token)) throw std::runtime_error("pair first token: first token is out of vocabulary");
        const auto result = decode(placement, token, 1U, placement.embeds() ? nullptr : &channel,
                                   placement.head() ? nullptr : &channel, first).ids.front();
        if (!valid_token(result)) throw std::runtime_error("pair first token: peer relayed an out-of-vocabulary token");
        return result;
    } catch (...) {
        link.abort("pair first token: session failed");
        throw;
    }
}

} // namespace strix::pair::layers
