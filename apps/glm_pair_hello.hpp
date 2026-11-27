#pragma once

#include "glm/load/glm_rank_defaults.hpp"
#include "glm_route_levers.hpp"
#include "attn_core/index_score_mode.hpp"
#include "glm/xfer/glm_xfer.hpp"
#include "strix/dist/transport.hpp"
#include "strix/glm/engine.hpp"
#include "strix/pair/handshake_hip.hpp"

namespace strix::glm::rank_defaults {

// Runs on the raw host link before an automatic row or layout admission can enter a different sequence.
inline bool agree_glm_pair(dist::Transport& transport, const engine::Options& opt, const Effective& eff,
                           bool serving, bool gpu_direct, std::uint32_t capabilities,
                           std::uint64_t prompt_and_decode = 0, std::uint64_t lane_contract = 0, bool turbo = false) {
    try {
        auto hello = pair::native_hello(serving ? 0x32534c47U : 0x32524c47U, opt.rank);  // "GLS2" / "GLR2"
        std::vector<ModelContentFile> files;
        for (const auto& entry : std::filesystem::directory_iterator(opt.model_dir)) {
            const auto name = entry.path().filename().string();
            if (entry.is_regular_file() && (entry.path().extension() == ".safetensors" || name == "config.json" ||
                                           name == "model.safetensors.index.json" || name == "tokenizer.json"))
                files.push_back({name, entry.path()});
        }
        hello.model_sha256 = build_model_content_identity(files).bytes;
        if (eff.product) hello.profile = eff.product->fingerprint();
        // Hash only effective values and the drafter payload, so path and configuration provenance may differ.
        std::string route = eff.identity.value + '|' + eff.prefill + '|' + eff.mtp.value + '|' + eff.mtp_spec.value +
            '|' + eff.spec.value + '|' + eff.decode_xch.value + '|' + eff.xch + '|' + eff.head.mtp_digest;
        for (const auto& item : env_items()) route += '|' + eff.env.at(item.env).value;
        hello.route = strix::glm::record_route_levers(serving ? "http" : "cli", opt.rank, eff.prefill_requested.value,
                                                     eff.prefill, eff.identity.value, opt.max_rows);
        std::uint64_t config = 14695981039346656037ULL;
        for (const char value : route) config = (config ^ static_cast<unsigned char>(value)) * 1099511628211ULL;
        hello.slots = opt.slots;
        hello.context = opt.capacity;
        hello.max_rows = opt.auto_rows ? 0U : opt.max_rows;
        hello.transport_capabilities = capabilities;
        namespace idx = strix::attn_core::idx;
        const auto idx_mode = idx::wmma_mode(strix::glm::pf::kIndexerWmmaUnsetMode);
        xfer::Options wire;
        wire.capacity = xfer::GlmSchedule::row_bytes(8U);
        wire.from_env();
        const std::uint64_t wire_flags = (wire.crc ? 1U : 0U) | (wire.crc && wire.crc_deferred ? 2U : 0U);
        hello.contract = {(std::uint64_t{opt.split_frame_mode} << 32U) | opt.split_frame_crossover,
            (wire_flags << 32U) | wire.fold_capacity,
            (opt.auto_rows ? 1U : 0U) | (gpu_direct ? 2U : 0U) | (opt.prefill_units ? 4U : 0U) |
                (eff.kv_shard_auto ? 8U : 0U) | (eff.kv_shard ? 16U : 0U),
            (std::uint64_t{opt.layers} << 32U) | eff.kmax, eff.adaptive ? 1U : 0U, prompt_and_decode, lane_contract, config,
            idx_mode == idx::kModeExact ? 0U : idx_mode};
        constexpr std::array<std::string_view, 9> fields{"split frame policy", "CRC and row fold", "row, device or KV layout policy",
            "model depth or MTP window", "MTP adaptive policy", "prompt or decode count", "serve lane or import policy",
            "effective runtime flags", "indexer score mode"};
        hello.offers[0] = turbo ? 1U : 0U;
        const auto peer = pair::exchange_hello(hello, [&](auto out, auto in) { transport.exchange(out, in); }, fields);
        return turbo && peer.offers[0] != 0U;
    } catch (const std::exception& e) {
        transport.abort(e.what());
        throw;
    }
}

}  // namespace strix::glm::rank_defaults
