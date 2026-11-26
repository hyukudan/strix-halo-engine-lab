#include "deepseek41_quality_arms.hpp"
#include "strix/ds41_prefill_precision.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

using strix::deepseek41::PrefillOptions;
using strix::deepseek41::PrefillResult;
namespace quality = strix::deepseek41::quality;
#include "replay_host.hpp"
#include "iu4_host.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void set_env(const char* key, const char* value) {
#ifdef _WIN32
    require(_putenv_s(key, value ? value : "") == 0, "set IU4 environment");
#else
    require((value ? setenv(key, value, 1) : unsetenv(key)) == 0, "set IU4 environment");
#endif
}

static void resident_iu4_arms() {
    struct Arm { const char* label; bool i4; bool relaxed; bool relaxed_f16; const char* profile; };
    const Arm arms[] = {
        {"exact", false, false, false, "exact"},
        {"n_l0_s1", false, false, false, "exact"},
        {"fast3", false, true, false, "fast-prefill"},
        {"m16_sh_eng_fast3", false, true, true, "fast-prefill"},
        {"i4_m16_sh_eng_fast3", true, false, false, "diagnostic"},
        {"m16_i4_fast3", true, false, false, "diagnostic"},
        {"fast3", false, true, false, "fast-prefill"},
        {"exact", false, false, false, "exact"},
    };
    for (const char* inherited : {"1", static_cast<const char*>(nullptr)}) {
        set_env("STRIX_DS41_FAST_I4", inherited);
        PrefillOptions loaded;
        require(resolve_iu4(loaded) == (inherited != nullptr), "ordinary options must retain IU4 inheritance");
        for (const auto& arm : arms) {
            const auto precision = quality::arm_precision(arm.label);
            apply_iu4(loaded, precision);
            if (resolve_iu4(loaded) != arm.i4 || loaded.fast_moe != arm.relaxed || loaded.fast_moe_f16 != arm.relaxed_f16)
                throw std::runtime_error(std::string("wrong resident routed-expert dispatch: ") + arm.label +
                                         " with inherited IU4 " + (inherited ? inherited : "unset"));
            require(quality::arm_profile(precision) == arm.profile, "receipt must name the executed arm profile");
        }
        require(strix::ds41::fast_i4() == (inherited != nullptr), "resident arms must leave the process setting alone");
    }
    PrefillOptions unresolved_iu4;
    unresolved_iu4.decoder_replay_override = true;
    require(quality::arm_profile(unresolved_iu4) == "diagnostic", "inherited IU4 cannot attest exact");
}

static void resident_iu4_scope() {
    using strix::ds41::fast_profile::MoeI4Scope;
    struct Peer {
        struct Tag { std::uint32_t kind, layer; std::uint64_t request; };
        MoeI4Scope local_scope, peer_scope;
        bool exchanged = false;
        void exchange_host(std::span<const std::byte> local, std::span<std::byte> remote, const Tag& tag) {
            require(tag.kind == 4U, "IU4 agreement must be a control frame");
            require(local.size() == 1 && remote.size() == 1 && local[0] == static_cast<std::byte>(local_scope),
                    "both ranks must send the same shape and their effective scope");
            remote[0] = static_cast<std::byte>(peer_scope);
            exchanged = true;
        }
    };
    const auto exact = quality::arm_precision("exact");
    const auto candidate = quality::arm_precision("i4_fast3");
    const auto check_scope = [&](const char* setting, MoeI4Scope local, MoeI4Scope remote, bool reject) {
        set_env("STRIX_DS41_FAST_I4_SCOPE", setting);
        Peer peer{local, remote};
        quality::verify_i4_scope(exact, peer, 4U);
        require(!peer.exchanged, "controls must not use the candidate scope or wire exchange");
        bool refused = false;
        try { quality::verify_i4_scope(candidate, peer, 4U); }
        catch (const std::invalid_argument&) { refused = true; }
        require(peer.exchanged && refused == reject, "candidate scope must be validated after both ranks exchange");
    };
    check_scope(nullptr, MoeI4Scope::all, MoeI4Scope::all, false);
    check_scope("all", MoeI4Scope::all, MoeI4Scope::all, false);
    check_scope("gu", MoeI4Scope::gate_up, MoeI4Scope::gate_up, false);
    check_scope("all", MoeI4Scope::all, MoeI4Scope::gate_up, true);
    check_scope("gu", MoeI4Scope::gate_up, MoeI4Scope::all, true);
    check_scope("bogus", MoeI4Scope::invalid, MoeI4Scope::all, true);
    check_scope("all", MoeI4Scope::all, MoeI4Scope::invalid, true);
    require(strix::ds41::fast_profile::moe_i4_scope("") == MoeI4Scope::invalid, "empty scope remains invalid");
}

static void replay_reference() {
    for (const auto inherited : {0U, 256U, 512U}) {
        PrefillOptions loaded;
        loaded.decoder_replay_rows = 512U;
        for (const auto* label : {"fast3b256", "exact", "n_l0_s1", "n_l0_s2"}) {
            const auto arm = quality::arm_precision(label);
            apply_replay(loaded, arm);
            const auto run = resolve_replay(loaded, inherited);
            const bool bounded = std::string_view(label) == "fast3b256";
            require(run.decoder_replay_rows == (bounded ? 256U : 0U), "reference/null must stay OFF under inherited replay");
            const auto executed = quality::executed_precision(arm, run);
            require(quality::arm_profile(executed) == (bounded ? "fast-prefill" : "exact"), "profile must reflect execution");
            require(resolve_replay(loaded, inherited, 8192U).decoder_replay_rows == 0U, "continuation must use exact plan");
        }
        PrefillOptions inherited_arm;
        require(resolve_replay(inherited_arm, inherited).decoder_replay_rows == inherited, "ordinary options must retain inheritance");
        // a served prompt split into pieces (decoder deferral to the next piece), in-run prefix snapshots or a group pass
        // need every decoder row: the run takes the exact plan instead of refusing (the serve was poisoned before)
        require(resolve_replay(inherited_arm, inherited, 0U, 14592U).decoder_replay_rows == 0U, "a deferred piece runs the exact plan");
        require(resolve_replay(inherited_arm, inherited, 0U, 0U, true).decoder_replay_rows == 0U, "a snapshot run runs the exact plan");
        require(resolve_replay(inherited_arm, inherited, 0U, 0U, false, true).decoder_replay_rows == 0U, "a group pass runs the exact plan");
        require(resolve_replay(loaded, inherited, 0U, 14592U).decoder_replay_rows == 0U, "an explicit bound never applies to a deferred piece");
    }
    PrefillOptions unresolved;
    require(quality::arm_profile(unresolved) == "diagnostic", "unresolved inheritance cannot attest exact");
    const auto bounded = resolve_replay(unresolved, 256U);
    const auto executed = quality::executed_precision(quality::arm_precision("exact"), bounded);
    require(quality::arm_profile(executed) == "diagnostic", "resolved bounded execution cannot attest exact");
    unresolved.decoder_replay_rows = 512U;
    require(resolve_replay(unresolved, 256U).decoder_replay_rows == 512U, "explicit nonzero option must beat environment");
}

static void write_bytes(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("fixture write failed");
}

static void model_content() {
    const std::filesystem::path target = "pr605-target.bin", overlay = "pr605-overlay.bin", copy = "pr605-copy.bin";
    struct Cleanup {
        std::array<std::filesystem::path, 3> paths;
        ~Cleanup() { for (const auto& p : paths) { std::error_code ec; std::filesystem::remove(p, ec); } }
    } cleanup{{target, overlay, copy}};
    write_bytes(target, "metadata|tensor payload A");
    write_bytes(overlay, "overlay payload A");
    const auto original = quality::model_witness(target, overlay);
    const auto stamp = std::filesystem::last_write_time(target);
    write_bytes(target, "metadata|tensor payload B");
    std::filesystem::last_write_time(target, stamp);
    require(original != quality::model_witness(target, overlay), "same-size target edit must change witness even with restored timestamp");
    write_bytes(target, "metadata|tensor payload A");
    require(original == quality::model_witness(target, overlay), "restored content must restore witness");
    write_bytes(overlay, "overlay payload B");
    require(original != quality::model_witness(target, overlay), "effective overlay content must change witness");
    require(quality::model_witness(target, {}) != quality::model_witness(target, overlay), "overlay absence must be bound");
    write_bytes(copy, "metadata|tensor payload A");
    require(quality::model_witness(target, overlay) == quality::model_witness(copy, overlay), "paths must not change model content identity");
    bool missing_rejected = false;
    try { (void)quality::model_witness(target, "pr605-missing-overlay.bin"); }
    catch (const std::runtime_error&) { missing_rejected = true; }
    require(missing_rejected, "missing effective overlay must not attest a witness");
}

static void manifest_positions() {
    std::istringstream short_rows("# name tokens path arms\ns8k-a 8128 a.ids exact\nt0k-a 256 b.ids exact\n");
    require(quality::manifest_max_positions(short_rows) == 8192, "<= 8184-token manifests keep the 8192 state");
    std::istringstream long_rows("s8k-a 8128 a.ids exact\n\nd32-a 32768 c.ids exact,m16_sh_eng_fast3\n");
    require(quality::manifest_max_positions(long_rows) == 32776, "a 32K window sizes the state to 32768 + 8");
    std::istringstream bad_rows("only-a-name\n");
    require(quality::manifest_max_positions(bad_rows) == 8192, "rows without a token count keep the floor");
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected replay, iu4 or model");
        if (std::string_view(argv[1]) == "replay") { replay_reference(); manifest_positions(); }
        else if (std::string_view(argv[1]) == "iu4") { resident_iu4_arms(); resident_iu4_scope(); }
        else if (std::string_view(argv[1]) == "model") model_content();
        else throw std::runtime_error("unknown test");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
