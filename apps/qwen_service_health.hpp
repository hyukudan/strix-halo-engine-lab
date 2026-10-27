#pragma once

#include "hip_decode_options.hpp"
#include "strix/asset_manifest.hpp"
#include "strix/build_info.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/health.hpp"

namespace strix::decode_cli {

// Only this exception crosses the release startup boundary. Its text is public and bounded.
enum class ReleaseAssetError { invalid, binding, inventory, io };
class ReleaseAssetFailure final : public std::runtime_error {
public:
    explicit ReleaseAssetFailure(ReleaseAssetError code) : std::runtime_error(
        code == ReleaseAssetError::invalid ? "release manifest invalid" :
        code == ReleaseAssetError::binding ? "release manifest loader binding failure" :
        code == ReleaseAssetError::inventory ? "release manifest inventory limit" :
        "release manifest I/O failure") {}
    explicit ReleaseAssetFailure(const AssetMismatch& mismatch) : std::runtime_error(message(mismatch)) {}
private:
    static std::string message(const AssetMismatch& mismatch) {
        std::string out = "release asset mismatch";
        for (const auto* role : {"target", "target_shard", "draft", "vocabulary", "tokenizer", "profile", "calibration"}) {
            if (mismatch.role != role) continue;
            out += " role=" + mismatch.role + " index=" + std::to_string(mismatch.index);
            for (const auto* key : {"target", "target_shard", "draft", "vocabulary", "tokenizer", "profile", "calibration", "distill-g"})
                if (mismatch.key == key) out += " key=" + mismatch.key;
            if (mismatch.key_digest.size() == 16 &&
                mismatch.key_digest.find_first_not_of("0123456789abcdef") == std::string::npos)
                out += " key_digest=" + mismatch.key_digest;
            break;
        }
        return out;
    }
};
// Cover the reader, verifier (including its filesystem operations), parsing and loader binding.
// Never forward exception text, including from unexpected/non-standard failures.
template<class Operation>
inline serve::Json release_asset_boundary(Operation&& operation) {
    try { return operation(); }
    catch (const ReleaseAssetFailure&) { throw; }
    catch (const AssetMismatch& error) { throw ReleaseAssetFailure(error); }
    catch (const std::invalid_argument&) { throw ReleaseAssetFailure(ReleaseAssetError::invalid); }
    catch (const serve::JsonError&) { throw ReleaseAssetFailure(ReleaseAssetError::invalid); }
    catch (...) { throw ReleaseAssetFailure(ReleaseAssetError::io); }
}

// Startup only: the package root is the manifest's containing directory. No HTTP read/hash.
inline serve::Json verify_http_release_assets(const Options& options) {
    if (options.release_manifest.empty() || options.print_effective_config) return {};
    return release_asset_boundary([&]() -> serve::Json {
        const std::filesystem::path manifest(options.release_manifest);
        const auto root = manifest.has_parent_path() ? manifest.parent_path() : std::filesystem::path(".");
        const auto text = read_asset_json(manifest);
        const auto verified = serve::parse_json(verify_asset_manifest(text, root));
        const auto input = serve::parse_json(text);
        if (input.find("engine")->as_string() != "qwen")
            throw ReleaseAssetFailure(ReleaseAssetError::invalid);
        // Bind explicit loader inputs to the verified package; never export member/path strings.
        const auto bound = [&](const std::string& path, std::string_view role) {
            if (path.empty()) return;
            std::error_code ec;
            const auto selected = std::filesystem::canonical(path, ec);
            if (ec) throw ReleaseAssetFailure(ReleaseAssetError::io);
            for (const auto& file : input.find("files")->as_array()) {
                if (file.find("role")->as_string() != role) continue;
                const auto member = std::filesystem::canonical(root / file.find("member")->as_string(), ec);
                if (ec) throw ReleaseAssetFailure(ReleaseAssetError::io);
                if (member == selected) return;
            }
            throw ReleaseAssetFailure(ReleaseAssetError::binding);
        };
        bound(options.model_path, "target");
        bound(options.mtp.sidecar, "draft");
        bound(options.mtp.golden, "calibration");
        if (options.mtp.draft_vocab.starts_with("auto:")) {
            const auto directory = options.mtp.draft_vocab.substr(5);
            unsigned entries = 0;
            std::error_code ec;
            std::filesystem::directory_iterator it(directory, ec), end;
            if (ec) throw ReleaseAssetFailure(ReleaseAssetError::io);
            while (it != end) {
                if (++entries > 1024) throw ReleaseAssetFailure(ReleaseAssetError::inventory);
                const bool regular = it->is_regular_file(ec);
                if (ec) throw ReleaseAssetFailure(ReleaseAssetError::io);
                if (regular && it->path().extension() == ".bin") bound(it->path().string(), "vocabulary");
                it.increment(ec);
                if (ec) throw ReleaseAssetFailure(ReleaseAssetError::io);
            }
        } else {
            // The explicit vocabulary option accepts a comma-separated file list.
            std::string_view list = options.mtp.draft_vocab;
            while (!list.empty()) {
                const auto comma = list.find(',');
                bound(std::string(list.substr(0, comma)), "vocabulary");
                if (comma == std::string_view::npos) break;
                list.remove_prefix(comma + 1);
            }
        }
        bound(options.dense_q6p_pack_path, "profile");
        bound(options.dense_trellis_pack_path, "profile");
        bound(options.offline_tuning_cache_path, "profile");
        // The API-key file is an operator secret, not a release asset: never bound or hashed.
        return verified;
    });
}
inline std::shared_ptr<const serve::ServeHealthRecord> http_health_record(const Options& options,
    std::uint64_t slots,
    const std::optional<StartupPreflight>& startup, const serve::Json& assets) {
    auto c = std::make_shared<serve::ServeHealthRecord>();
    const auto info = strix::build_info_json();
    c->build_info = serve::parse_json(info);
    const auto hash = sha256_bytes(std::as_bytes(std::span(info.data(), info.size())));
    constexpr char digits[] = "0123456789abcdef";
    for (const auto b : hash.bytes) { c->build_info_sha256 += digits[b >> 4U]; c->build_info_sha256 += digits[b & 15U]; }
    c->asset_verification = assets;
    c->context_tokens = options.context_tokens;
    c->slots = slots;
    const auto pair = numerical_profile_pair(options);
    c->decode_profile = pair.decode;
    c->prefill_profile = pair.prefill;
    c->prefill_tile_rows = options.known_prefix_prefill_rows;
    c->draft_profile = options.mtp.draft_profile;
    c->prefix_enabled = options.prefix_cache_host_bytes > 0;
    c->prefix_host_bytes = options.prefix_cache_host_bytes;
    c->prefix_disk_enabled = c->prefix_enabled && !options.prefix_cache_dir.empty();
    c->prefix_disk_bytes = options.prefix_cache_disk_bytes;
    if (startup) { c->preflight = startup->observation; c->preflight_assessment = startup->assessment; }
    const auto* probe = std::getenv("STRIX_SERVE_BACKEND_PROBE");
    c->backend_probe_enabled = !probe || std::string_view(probe) != "0";
    c->interim_unchecked_http_success = true;
    return c;
}

} // namespace strix::decode_cli
