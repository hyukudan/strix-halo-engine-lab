#include "strix/asset_manifest.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/mtp_draft_profiles.hpp"
#include "strix/serve/json.hpp"
#include <algorithm>
#include <random>
#include <functional>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>
#include <stdexcept>

namespace strix {
std::function<void()> asset_test_after_hash_callback;
void asset_test_after_hash() { if (asset_test_after_hash_callback) asset_test_after_hash_callback(); }
}
namespace {
using strix::serve::Json;
void check(bool ok) { if (!ok) throw std::runtime_error("asset test failed"); }
template<class E = std::invalid_argument, class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const E&) { rejected = true; }
    check(rejected);
}
void write(const std::filesystem::path& p, const std::string& s) { std::ofstream(p, std::ios::binary) << s; }
}
// Per-backend draft min-p: HIP measured values, CUDA legacy 0.7 for every draft (no promotion).
void draft_min_p_profiles() {
    using B = strix::MtpDraftBackend;
    const auto* base = strix::find_mtp_draft_profile("base");
    check(base != nullptr && base->hip_min_probability == 0.85F && base->cuda_min_probability == 0.7F);
    for (const auto& p : strix::mtp_draft_profiles) {
        check(p.cuda_min_probability == 0.7F);
        check(strix::mtp_draft_backend_min_probability(&p, B::hip) == p.hip_min_probability);
    }
    check(strix::find_mtp_draft_profile("distill-g")->hip_min_probability == 0.7F);
    check(strix::mtp_draft_backend_min_probability(nullptr, B::hip) == 0.5F);
    check(strix::mtp_draft_backend_min_probability(nullptr, B::cuda) == 0.7F);
    auto r = strix::resolve_mtp_draft_min_probability(std::nullopt, base, B::cuda);
    check(r.value == 0.7F && r.source == "profile");
    r = strix::resolve_mtp_draft_min_probability(std::nullopt, nullptr, B::cuda);
    check(r.value == 0.7F && r.source == "unprofiled");
    r = strix::resolve_mtp_draft_min_probability(std::nullopt, base, B::hip);
    check(r.value == 0.85F && r.source == "profile");
    r = strix::resolve_mtp_draft_min_probability(0.0F, base, B::cuda);  // explicit 0 is a real value
    check(r.value == 0.0F && r.source == "explicit");
    r = strix::resolve_mtp_draft_min_probability(1.0F, nullptr, B::hip);
    check(r.value == 1.0F && r.source == "explicit");
    for (const float bad : {-0.01F, 1.01F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        bool rejected = false;
        try { (void)strix::resolve_mtp_draft_min_probability(bad, base, B::cuda); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
    }
}
// Drafter discovery binds a registry draft only to the target it was trained for (architecture, layers, hidden).
void draft_targets() {
    for (const auto& p : strix::mtp_draft_profiles) {
        check(strix::mtp_draft_target_matches(p, "qwen4exp", 48, 2560));
        check(!strix::mtp_draft_target_matches(p, "qwen4exp", 64, 5120));
        check(!strix::mtp_draft_target_matches(p, "qwen35", 48, 2560));
        check(!strix::mtp_draft_target_matches(p, "qwen4exp", 49, 2560));  // the draft sidecar's own layer count
        check(!strix::mtp_draft_target_matches(p, "", 0, 0));               // unreadable model identity
    }
}
int main() try {
    draft_min_p_profiles();
    draft_targets();
    std::random_device random;
    std::filesystem::path root;
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        auto candidate = std::filesystem::temp_directory_path() / ("strix-assets-" + std::to_string(random()));
        if (std::filesystem::create_directory(candidate)) { root = std::move(candidate); break; }
    }
    check(!root.empty());
    struct Cleanup {
        std::filesystem::path p;
        ~Cleanup() { std::error_code ec; std::filesystem::remove_all(p, ec); }
    } cleanup{root};
    write(root / "a", "abc"); write(root / "b", "def");
    const std::string spec = R"({"format":"strix-assets","revision":1,"engine":"qwen","model_key":"tiny","profile_pair":"test","required":{"target":["a"],"profile":["b"]},"files":[{"role":"target","key":"a","member":"a","source_revision":"test","content_key":"model"},{"role":"profile","key":"b","member":"b","source_revision":"test"}],"provenance":[],"licenses":[]})";
    const auto manifest = strix::create_asset_manifest(spec, root);
    check(manifest == strix::create_asset_manifest(spec, root));
    check(strix::serve::parse_json(strix::verify_asset_manifest(manifest, root)).find("verification")->as_string() == "full");
    auto reordered = strix::serve::parse_json(spec);
    std::reverse(reordered.as_object().begin(), reordered.as_object().end());
    auto& ordered_files = reordered.set("files", *reordered.find("files")).as_array();
    std::reverse(ordered_files.begin(), ordered_files.end());
    check(strix::create_asset_manifest(reordered.dump(), root) == manifest);
    check(strix::asset_manifest_digest(manifest) == strix::asset_manifest_digest(strix::create_asset_manifest(reordered.dump(), root)));
    auto parsed = strix::serve::parse_json(manifest);
    const auto& target = parsed.find("files")->as_array().back();
    check(target.find("sha256")->as_string() != target.find("content_identity")->as_string());
    write(root / "a", "abd"); rejects<strix::AssetMismatch>([&] { (void)strix::verify_asset_manifest(manifest, root); });
    check(strix::serve::parse_json(strix::verify_asset_manifest(manifest, root, {"profile"})).find("verification")->as_string() == "partial");
    write(root / "a", "a"); rejects<strix::AssetMismatch>([&] { (void)strix::verify_asset_manifest(manifest, root); });
    std::filesystem::remove(root / "a"); rejects<strix::AssetMismatch>([&] { (void)strix::verify_asset_manifest(manifest, root); });
    write(root / "a", "abc");
    for (const auto& path : {"/tmp/private", "../secret", "C:/private", "\\\\host\\share"}) {
        auto j = strix::serve::parse_json(spec);
        j.set("files", *j.find("files")).as_array()[0].set("member", path);
        rejects([&] { (void)strix::create_asset_manifest(j.dump(), root); });
    }
    std::filesystem::create_directory_symlink(root.parent_path(), root / "escape");
    auto j = strix::serve::parse_json(spec);
    j.set("files", *j.find("files")).as_array()[0].set("member", "escape/secret");
    rejects([&] { (void)strix::create_asset_manifest(j.dump(), root); });
    j = strix::serve::parse_json(spec);
    j.set("files", *j.find("files")).as_array().push_back(j.find("files")->as_array()[0]);
    rejects([&] { (void)strix::create_asset_manifest(j.dump(), root); });
    rejects<strix::serve::JsonError>([&] { (void)strix::create_asset_manifest("{\"revision\":1," + spec.substr(1), root); });
    for (const auto& hash : {"bad", "0000000000000000000000000000000000000000000000000000000000000000"}) {
        j = strix::serve::parse_json(spec);
        j.set("files", *j.find("files")).as_array()[0].set("expected_sha256", hash);
        if (std::string(hash) == "bad") rejects([&] { (void)strix::create_asset_manifest(j.dump(), root); });
        else rejects<strix::AssetMismatch>([&] { (void)strix::create_asset_manifest(j.dump(), root); });
    }
    j = strix::serve::parse_json(manifest);
    j.set("files", *j.find("files")).as_array().pop_back();
    rejects<strix::AssetMismatch>([&] { (void)strix::verify_asset_manifest(j.dump(), root); });
    j = strix::serve::parse_json(spec);
    j.set("files", *j.find("files")).as_array()[0].set("bytes", Json::number_text("18446744073709551616"));
    rejects([&] { (void)strix::create_asset_manifest(j.dump(), root); });
    rejects([&] { (void)strix::create_asset_manifest(std::string(1024 * 1024 + 1, ' '), root); });
    j = strix::serve::parse_json(spec);
    j.set("files", *j.find("files")).as_array()[0].set("audit", "gguf");
    rejects<strix::AssetMismatch>([&] { (void)strix::create_asset_manifest(j.dump(), root); });
    rejects([&] { (void)strix::verify_asset_manifest(manifest, root, {"absent"}); });
    j = strix::serve::parse_json(manifest);
    j.set("files", *j.find("files")).as_array().back().set("content_identity", std::string(64, '0'));
    rejects<strix::AssetMismatch>([&] { (void)strix::verify_asset_manifest(j.dump(), root); });
    j = strix::serve::parse_json(spec);
    auto& fields = j.set("files", *j.find("files")).as_array()[0].as_object();
    std::erase_if(fields, [](const auto& kv) { return kv.first == "source_revision"; });
    rejects([&] { (void)strix::create_asset_manifest(j.dump(), root); });
    // Test-only object supplies a deterministic mutation between hashing and recheck.
    for (const auto* key : {"target", "PRIVACY_HOST.internal", "SECRET_123"}) {
        j = strix::serve::parse_json(spec);
        j.set("required", *j.find("required")).set("target", Json(Json::Array{Json(key)}));
        j.set("files", *j.find("files")).as_array()[0].set("key", key);
        strix::asset_test_after_hash_callback = [&] {
            const auto path = root / "a";
            std::filesystem::last_write_time(path, std::filesystem::last_write_time(path) + std::chrono::seconds(1));
        };
        bool changed = false;
        try { (void)strix::create_asset_manifest(j.dump(), root); }
        catch (const strix::AssetMismatch& e) {
            changed = std::string(e.what()) == "asset changed while hashing";
            check(e.role == "target" && e.index == 1);
            if (std::string(key) == "target") check(e.key == key && e.key_digest.empty());
            else {
                check(e.key.empty() && e.key_digest.size() == 16);
                const std::string name = key;
                check(e.key_digest == strix::model_content_identity_hex(strix::sha256_bytes(
                    std::as_bytes(std::span(name.data(), name.size())))).substr(0, 16));
            }
        }
        strix::asset_test_after_hash_callback = {};
        check(changed);
    }
    const auto real = strix::serve::parse_json(strix::read_asset_json(std::filesystem::path(STRIX_SOURCE_DIR) / "release/qwen-assets.spec.json"));
    std::size_t vocab = 0;
    for (const auto& f : real.find("files")->as_array()) {
        if (f.find("role")->as_string() == "draft") {
            const auto* profile = strix::find_mtp_draft_profile("distill-g");
            check(profile != nullptr);
            check(f.find("member")->as_string() == profile->logical_name);
            check(f.find("expected_sha256")->as_string() == profile->sidecar_sha256);
            check(f.find("bytes")->number_source() == std::to_string(profile->sidecar_bytes));
            check(f.find("source_revision")->as_string() == profile->revision);
        }
        if (f.find("role")->as_string() != "vocabulary") continue;
        const auto p = std::filesystem::path(STRIX_SOURCE_DIR) / "data" / f.find("member")->as_string();
        check(strix::model_content_identity_hex(strix::sha256_file_content(p)) == f.find("expected_sha256")->as_string());
        ++vocab;
    }
    check(vocab == 20);
    std::cout << "asset manifest tests passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
