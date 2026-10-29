#pragma once
#include "strix/file_snapshot.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/ds41_dspark_context.hpp"
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <set>
#include <algorithm>

namespace strix::deepseek41 {
using SessionBytes = std::vector<std::uint8_t>;
// Each invocation must exchange exactly these bytes with the other rank, in order.
using SessionHostExchange = std::function<void(std::span<const std::byte>, std::span<std::byte>)>;
[[nodiscard]] SessionBytes session_peer_exchange(std::span<const std::uint8_t>, const SessionHostExchange &);
using SessionExchange = std::function<SessionBytes(std::span<const std::uint8_t>)>;
struct SessionCompat {
    std::map<std::string, std::string> fields;
    [[nodiscard]] std::string record() const;
    [[nodiscard]] std::string key() const;
};
// Full-file SHA-256, memoized only by stable file identity; detects mutation during hashing.
[[nodiscard]] std::string session_file_sha256(const std::filesystem::path &);
// Includes all STRIX_ and HIP_/ROCR_/HSA_ environment settings, plus full hashes of supplied
// files/directories. Callers must additionally supply every effective CLI/layout setting.
[[nodiscard]] SessionCompat session_compat(std::map<std::string, std::string> effective,
                                           const std::map<std::string, std::filesystem::path> &assets,
                                           const std::filesystem::path &executable,
                                           const LoaderSnapshot* loaded = nullptr);
// Always reach peer agreement after setup, including explicit startup failures.
template <class Factory, class Log>
auto admit_session_startup(Factory factory, Log log) -> decltype(factory()) {
    try { return factory(); }
    catch (const std::exception& e) {
        log(e.what());
        return {};
    }
    catch (...) {
        log("unknown session setup failure");
        return {};
    }
}
// A failure disables automatic sessions; an explicit request on either rank
// refuses both ranks only after the exchange has completed.
template <class Exchange>
[[nodiscard]] bool session_startup_ready(bool ready, bool explicit_use, const Exchange& exchange) {
    const std::uint64_t local = (ready ? 1U : 0U) | (explicit_use ? 2U : 0U);
    std::uint64_t peer = 0;
    exchange(std::as_bytes(std::span(&local, 1)), std::as_writable_bytes(std::span(&peer, 1)));
    if (ready && (peer & 1U)) return true;
    if (explicit_use || (peer & 2U)) throw std::runtime_error("session setup failed on one or both ranks");
    return false;
}
[[nodiscard]] std::string session_generation();
void validate_session_id(const std::string &);
struct SessionFrontier {
    std::uint64_t consumed = 0, emitted = 0, accepted = 0, drafter = 0;
    std::uint32_t pending = 0;
    bool pending_valid = true;
    std::string sampling = "greedy", lineage = "ds41-target-committed-turn-v1";
};
struct SessionDraftState {
    ds41::dspark::ContextSnapshot context;
    ds41::dspark::ControllerSnapshot controller;
    std::uint64_t exchange_request = 0, windows = 0, verified = 0, accepted = 0;
    // No pending drafts or staged hidden rows at this producer's committed-turn boundary.
};
[[nodiscard]] SessionBytes serialize_session_draft(const SessionDraftState&);
[[nodiscard]] SessionDraftState deserialize_session_draft(std::span<const std::uint8_t>);
struct SessionCheckpoint {
    std::string id, generation, compat_record;
    std::uint32_t world = 2, rank = 0;
    SessionFrontier frontier;
    AttentionState attention;
    std::optional<SessionDraftState> draft;
};
[[nodiscard]] inline bool session_prompt_extends(const SessionCheckpoint &c, std::span<const std::uint32_t> prompt) {
    const auto &history = c.attention.tokens;
    return prompt.size() >= history.size() + (c.frontier.pending_valid ? 1U : 0U) &&
        std::equal(history.begin(), history.end(), prompt.begin()) &&
        (!c.frontier.pending_valid || prompt[history.size()] == c.frontier.pending);
}
// Preserves all existing attention-state bytes; top-k remains in the rank-local section.
[[nodiscard]] std::string session_state_sha256(const AttentionState &);
[[nodiscard]] std::string session_state_sha256(const SessionCheckpoint &);
[[nodiscard]] SessionBytes serialize_session(const SessionCheckpoint &);
[[nodiscard]] SessionCheckpoint deserialize_session(std::span<const std::uint8_t>, const SessionCompat &, std::uint32_t world,
                                                    std::uint32_t rank, std::uint64_t max_bytes);
struct SessionRestoreTiming {
    std::uint64_t payload_bytes = 0;
    double read_seconds = 0, verify_seconds = 0, peer_seconds = 0;
};
struct SessionInfo {
    SessionCheckpoint checkpoint;
    std::uint64_t bytes = 0;
    bool ram = false, disk = false;
};
struct SessionCacheStats {
    std::uint64_t bytes = 0, entries = 0, evictions = 0, refused = 0;
};
// Extension of PrefixCache, no second attention/KV representation. Engine-thread owned.
// Both ranks participate in every mutation and restore. Returned host state is unpublished;
// callers must import into unused slots and agree success before scheduling either slot.
class PrefixCache::Sessions {
  public:
    Sessions(SessionCompat compat, std::uint32_t rank, std::uint64_t budget_bytes, std::filesystem::path directory = {},
             std::uint64_t max_parked = 8, std::uint64_t disk_budget = ~std::uint64_t{0});
    void capture(const SessionCheckpoint &, const SessionExchange &);
    void checkpoint(const std::string &id, const SessionExchange &); // durable group commit
    [[nodiscard]] SessionCheckpoint resume(const std::string &id, bool disk, const SessionExchange &);
    void erase(const std::string &id, const SessionExchange &);
    [[nodiscard]] std::vector<std::string> list() const;
    [[nodiscard]] SessionCacheStats stats() const;
    [[nodiscard]] SessionInfo inspect(const std::string &) const;
    [[nodiscard]] std::uint64_t disk_bytes() const;
    [[nodiscard]] SessionRestoreTiming last_restore() const { return restore_timing_; }
    [[nodiscard]] const SessionCompat &compat() const { return compat_; }
    // Final GPU import barrier: false on either rank refuses publication on both.
    static void agree(bool success, const SessionExchange &, const std::string &phase);

  private:
    struct Entry {
        SessionBytes bytes;
        std::uint64_t used = 0;
    };
    SessionCompat compat_;
    std::uint32_t rank_;
    std::uint64_t budget_, max_parked_, disk_budget_, clock_ = 0;
    std::filesystem::path dir_;
    std::map<std::string, Entry> ram_;
    SessionCacheStats stats_;
    std::set<std::string> dirty_;
    SessionRestoreTiming restore_timing_;
    void remove_disk(const std::string& id, const std::filesystem::path& keep = {});
    void reserve_disk(const std::string& id, std::uint64_t bytes, const SessionExchange&);
    [[nodiscard]] std::filesystem::path manifest(const std::string &) const;
};
} // namespace strix::deepseek41
