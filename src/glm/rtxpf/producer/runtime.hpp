#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <cstdint>
#include "verified_manifest.hpp"
namespace strix::glm::rtxpf::producer {
// Decode expert-tier selection and tierstream startup budget (decode only).
struct DecodeOptions {
    std::string expert_tier="tierstream"; // the product cache (default); "reference": uncached backend, only inside the untimed compare
    std::string unit_file;               // tierstream disk tier: the GLM unit file (manifest beside it)
    double vram_total_gib=-1,ram_total_gib=-1; // whole-device caps (< 0: none = the device room / host above the floor)
    double tier_vram_gib=-1,tier_ram_gib=-1; // < 0: everything left after prefill residency / the host floor
    unsigned tier_spare_slots=16,tier_host_spares=8,tier_staging=8;
    // Arena spares (retained misses): max(tier_spare_slots, permille/1000 of the arena);
    // 0 = tier_spare_slots exactly (an explicit --tier-spare-slots sets it).
    unsigned tier_spare_permille=50;
    // Placement, the DS way (expert_tier.hpp seed_order): residency file of the previous
    // run ("" = <unit_file>.residency, "none" = off; written back after every request),
    // routes census files (DS routes format; comma-separated), the round-robin fallback.
    std::string expert_residency,expert_routes;
    std::string routes_out;              // record each decoded token's routes (DS routes format; receipts carry them too)
    std::string tier_policy="auto";      // boundary victims: lru | scored | auto (SCORED_AUTO, the DS default)
    bool tier_predict=false;             // overlapped next-layer prefetch (ss_dg_enqueue_predict), off by default
    // dg_flush_ready: the library flushes its demand stream after each layer's ready
    // signal, so WDDM submits the ready write at once instead of batching it behind
    // later work (a device wait on that layer could otherwise outlive the service's
    // flag deadline). Submission only: outputs unchanged. On by default (profile row flush_ready).
    bool tier_flush_ready=true;
    // Exact decode levers (profile rows, pending until a window proves equal digests):
    // routed MoE launches only the expert groups holding routed rows (device compaction);
    // shared experts and dense FFNs keep a persistent FP16 copy decoded once at startup
    // (a named whole-device plan part; dropped when the plan cannot afford it).
    bool skip_empty_expert_groups=false,persistent_dense_fp16=false;
    bool disk_prefetch=false;
    bool reference_compare=false;        // tierstream: also keep the uncached reference tier for same-window comparison
    std::string unit_manifest_sha256;    // pinned digest of PRIMARY.units.json (trusted manifest)
};
using StateSink=std::function<void(unsigned rank,unsigned layer,unsigned component,std::span<const std::byte>)>;
// Constructor is the first device work. The service validates the UUID and all
// host arguments before constructing it. No CPU emulation hides a CUDA failure.
class Runtime {
    struct Impl;std::unique_ptr<Impl> p_;
public:
    // decode=true additionally loads the root lm_head (decode-capable cache or
    // shards), allocates resident session state and selects the expert tier
    // backend ("reference": uncached per-expert reads; "tierstream": shared cache).
    Runtime(const std::string& path,unsigned capacity,unsigned row_tile,bool from_shards=false,bool graphs=true,unsigned host_floor_gib=8,bool legacy_index=false,bool serial_stage=false,const VerifiedManifest* manifest=nullptr,unsigned max_device_layers=43,unsigned max_pinned_layers=43,bool require_file_tier=false,bool decode=false,const DecodeOptions& decode_options={});
    ~Runtime();
    // Dry tierstream budget plan from shard headers (no hashing, no CUDA calls):
    // the planned VRAM/host split for the given device figures, as JSON.
    static std::string plan_budget(const std::string& model,unsigned capacity,unsigned row_tile,unsigned host_floor_gib,const DecodeOptions& decode_options,std::uint64_t device_total,std::uint64_t device_free);
    std::string health()const;
    // Measured device/host memory now and at peak, ledger charges, caps (JSON).
    std::string memory_report(const std::string& stage);
    // Expert cache diagnostics for failure receipts (JSON; "{}" without the tierstream cache).
    std::string expert_diagnostics();
    // Opt-in decode timeline: every decode_one record then carries "timeline" (CUDA events per phase; outputs unchanged).
    void set_decode_profile(bool on);
    // Writes the cache residency (most recent first) to the residency file; "" when off. Between requests.
    std::string save_residency();
    std::string model_sha256()const;
    std::string timings()const;
    void prefill(std::span<const std::uint32_t> tokens,const std::string& identity,const StateSink&);
    std::string self_check(std::span<const std::uint32_t> tokens,const std::string& identity,unsigned reference_tile=256);
    // Prefill row tile: the current one, the one allocated at startup, and a
    // per-request override (valid_schedule, at most the allocated tile; graph
    // templates recapture for the new shape, as the self-check does).
    unsigned prefill_row_tile()const;
    unsigned allocated_prefill_row_tile()const;
    void use_prefill_row_tile(unsigned tile);
    // ---- single session decode, identity glm-dec-rtx-native-v1 (draft) ----
    // One pending token: after begin_session(S) the frontier is |S|-1 and S.back()
    // is pending. decode_one consumes it and appends the greedy token. Any
    // failure invalidates the session; begin_session always starts from zero.
    bool decode_capable()const;
    std::string begin_session(std::span<const std::uint32_t> tokens);
    std::string decode_one(bool qualification=false);
    std::string generate(std::span<const std::uint32_t> prompt,unsigned max_new_tokens,std::span<const std::uint32_t> stop_token_ids,bool qualification=false);
    // All 46 layers at the current frontier, exactly the prefill sink order/layout.
    void export_session(const StateSink&);
    // SHA-256 over every KDA block-start checkpoint (state, convolution) and retained open rows.
    std::string replay_digest();
    std::vector<std::uint32_t> session_tokens()const;
    void snapshot_session();
    void restore_session();
    void reset_session();
    // Between tokens only: reset the fetched-expert cache to its open state (no-op without a cache).
    void reset_expert_cache(bool open);
    // "tierstream" or "reference" (with --reference-compare): backend of the next session.
    void select_expert_tier(const std::string& name);
    std::string expert_tier()const;
};
}
