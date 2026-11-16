#include "strix/minimax_layercut.hpp"
#include "strix/deploy/minimax_layercut.hpp"
#include <cstdio>
#include <map>
#include <stdexcept>

namespace {
namespace lc = strix::pair::layercut;
namespace mm = strix::minimax::layercut;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class F> void rejects(F f, const char* key) {
    try { f(); } catch (const std::invalid_argument& e) {
        require(std::string(e.what()).find(key) != std::string::npos, "error must name key/field"); return;
    }
    throw std::runtime_error("invalid input accepted");
}
}
int main() try {
    lc::Inputs input{mm::chain_shape(), 32769, 4320, 3200, 4600};
    const auto set_env = [](const char* key, const char* value) {
#ifdef _WIN32
        if (_putenv_s(key, value) != 0) throw std::runtime_error("set test env");
#else
        if (setenv(key, value, 1) != 0) throw std::runtime_error("set test env");
#endif
    };
    for (const char* key : {"STRIX_SPLIT", "STRIX_SPLIT_CUT", "STRIX_SPLIT_BLOCK", "STRIX_SPLIT_ACT"}) set_env(key, "");
    require(lc::resolve({}, input).split == lc::chain::Split::device_only, "envcfg default");
    set_env("STRIX_SPLIT", "layer-cut"); set_env("STRIX_SPLIT_CUT", "7");
    require(lc::resolve({}, input).cut == 7, "actual envcfg lookup");
    lc::Cli actual; actual.cut = "8";
    require(lc::resolve(actual, input).cut == 8, "actual envcfg override");
    bool registered = false;
    for (const auto& e : strix::envcfg::registered()) registered |= e.name == "STRIX_SPLIT" && e.value == "layer-cut" && e.set;
    require(registered, "envcfg registration");
    set_env("STRIX_SPLIT", ""); set_env("STRIX_SPLIT_CUT", "");
    std::map<std::string, std::string> env;
    auto lookup = [&](const char* k) -> const char* { auto it = env.find(k); return it == env.end() ? nullptr : it->second.c_str(); };
    auto r = lc::resolve({}, input, lookup);
    require(r.split == lc::chain::Split::device_only && r.block == 2048 && r.act == lc::Act::relay, "default route");
    require(r.lines.find("split split device-only (default)") != std::string::npos, "default origin");
    env = {{"STRIX_SPLIT", "layer-cut"}, {"STRIX_SPLIT_CUT", "12"}, {"STRIX_SPLIT_BLOCK", "512"}, {"STRIX_SPLIT_ACT", "direct"}};
    r = lc::resolve({}, input, lookup);
    require(r.cut == 12 && r.block == 512 && r.act == lc::Act::direct && r.lines.find("(env)") != std::string::npos, "env policy");
    lc::Cli cli; cli.cut = "24"; cli.block = "1024"; cli.act = "relay";
    env["STRIX_SPLIT_CUT"] = "invalid-overridden";
    r = lc::resolve(cli, input, lookup);
    require(r.cut == 24 && r.block == 1024 && r.act == lc::Act::relay && r.lines.find("split cut 24 (cli)") != std::string::npos, "cli wins");
    cli.split = "device-only"; env["STRIX_SPLIT"] = "overridden";
    require(lc::resolve(cli, input, lookup).split == lc::chain::Split::device_only, "split cli wins");
    env.clear(); cli = {}; cli.split = "layer-cut";
    for (const char* bad : {"0", "60", "61", "-1", "12x", "+2", " 2", "4294967296", ""}) {
        cli.cut = bad; rejects([&] { (void)lc::resolve(cli, input, lookup); }, "STRIX_SPLIT_CUT");
    }
    cli.cut = "20";
    for (const char* bad : {"0", "17", "16x", "-16", ""}) {
        cli.block = bad; rejects([&] { (void)lc::resolve(cli, input, lookup); }, "STRIX_SPLIT_BLOCK");
    }
    cli.block = "2048"; cli.act = "bad";
    rejects([&] { (void)lc::resolve(cli, input, lookup); }, "STRIX_SPLIT_ACT");
    cli.act = "direct"; cli.split = "bad";
    rejects([&] { (void)lc::resolve(cli, input, lookup); }, "STRIX_SPLIT");
    cli.split = "layer-cut"; cli.cut = "auto";
    input.link_mbps = 118;
    const auto slow = lc::resolve(cli, input, lookup);
    input.link_mbps = 4600;
    const auto fast = lc::resolve(cli, input, lookup);
    require(slow.cut != fast.cut && slow.cut < fast.cut, "link changes automatic cut");
    require(fast.lines.find("split cut " + std::to_string(fast.cut) + " (planner)") != std::string::npos, "planner origin");
    std::printf("synthetic MiniMax auto cuts: 1gbe=%u usb4=%u\n", slow.cut, fast.cut);
    cli.block = "auto";
    r = lc::resolve(cli, input, lookup);
    require(r.block % 16 == 0 && r.lines.find("(planner)") != std::string::npos, "auto block");
    cli.split = "auto";
    r = lc::resolve(cli, input, lookup);
    require(r.split != lc::chain::Split::auto_plan, "auto split resolved");
    input.link_mbps = 0; env["STRIX_HANDOFF_LINK"] = "usb4";
    require(lc::resolve(cli, input, lookup).block > 0, "link class input");
    for (const auto& entry : std::map<std::string, double>{{"1gbe",118},{"2.5gbe",290},{"10gbe",1180},{"usb4",4600}})
        require(lc::link_class(entry.first) == entry.second, "link class rate");
    env["STRIX_HANDOFF_LINK"] = "bad";
    rejects([&] { (void)lc::resolve(cli, input, lookup); }, "STRIX_HANDOFF_LINK");
    input.rtx_rows_s = 0; env.clear();
    rejects([&] { (void)lc::resolve(cli, input, lookup); }, "STRIX_SPLIT");

    const auto schedule = lc::blocks(4103, 2048);
    require(schedule.size() == 3 && schedule[2].row0 == 4096 && schedule[2].rows == 7, "partial block real rows");
    require(lc::blocks(0, 16).empty() && lc::blocks(32, 16).size() == 2, "empty/exact block schedule");
    rejects([] { (void)lc::blocks(16, 0); }, "STRIX_SPLIT_BLOCK");
    lc::Meta m{1, 2, 16, 6144, 19, lc::token_fnv(std::vector<std::uint32_t>{1, 0x10203040}), mm::identity(2, "fp32")};
    const auto bytes = lc::pack(m);
    const auto n = lc::unpack(bytes);
    lc::validate(n, m, 60);
    require(lc::pack(n) == bytes && bytes[0] == 1 && bytes[4] == 2 && bytes[32] == m.identity.size(), "meta little endian round trip");
    for (unsigned field = 0; field < 8; ++field) {
        auto bad = m;
        switch (field) {
        case 0: ++bad.version; break; case 1: ++bad.cut; break; case 2: bad.block += 16; break;
        case 3: ++bad.hidden; break; case 4: ++bad.rows; break; case 5: ++bad.token_fnv64; break;
        case 6: bad.identity += "changed"; break; default: bad.cut = 0; break;
        }
        rejects([&] { lc::validate(bad, m, 60); }, "mismatch");
    }
    for (std::size_t len = 0; len < bytes.size(); ++len)
        rejects([&] { (void)lc::unpack(std::span(bytes).first(len)); }, "meta");
    auto trailing = bytes; trailing.push_back(0);
    rejects([&] { (void)lc::unpack(trailing); }, "meta");
    // Valid transport sessions can still omit/repeat model pieces: reject them before import/head/reply.
    for (bool relay : {false, true}) {
        lc::Pieces order(m, true, true, relay);
        std::uint32_t b = 0;
        for (auto desc : lc::blocks(m.rows, m.block)) {
            order.accept({lc::kAct, b++, relay ? strix::handoff::kRelay : 0, desc.row0, desc.rows}, std::size_t{desc.rows} * 24576, 1280);
            for (unsigned l = 0; l < m.cut; ++l)
                order.accept({lc::kKv, l, 0, desc.row0, desc.rows}, std::size_t{desc.rows} * 1280, 1280);
        }
        order.finish();
    }
    lc::Pieces missing(m, true, true, false);
    rejects([&] { missing.finish(); }, "incomplete");
    rejects([&] { missing.accept({lc::kKv, 0, 0, 0, 16}, 16 * 1280, 1280); }, "piece");
    rejects([&] { missing.accept({lc::kAct, 0, 0, 0, 16}, 16 * 24576 - 1, 1280); }, "piece");
    missing.accept({lc::kAct, 0, 0, 0, 16}, 16 * 24576, 1280);
    rejects([&] { missing.accept({lc::kAct, 0, 0, 0, 16}, 16 * 24576, 1280); }, "piece");
    lc::Pieces kv_only(m, false, true, false), act_only(m, true, false, true);
    std::uint32_t b = 0;
    for (auto desc : lc::blocks(m.rows, m.block)) {
        act_only.accept({lc::kAct, b++, strix::handoff::kRelay, desc.row0, desc.rows}, std::size_t{desc.rows} * 24576, 1280);
        for (unsigned l = 0; l < m.cut; ++l) kv_only.accept({lc::kKv, l, 0, desc.row0, desc.rows}, std::size_t{desc.rows} * 1280, 1280);
    }
    kv_only.finish(); act_only.finish();
    // Verify scatter reconstructs all three component planes, including a short tail.
    std::vector<std::uint8_t> full(m.rows * 1280, 0xff), reference(full.size());
    for (std::size_t i = 0; i < reference.size(); ++i) reference[i] = static_cast<std::uint8_t>(i * 17 + i / 512);
    for (auto desc : lc::blocks(m.rows, m.block)) {
        std::vector<std::uint8_t> part(std::size_t{desc.rows} * 1280);
        std::size_t src = 0, dst = 0;
        for (auto w : mm::rank_state_widths) {
            std::memcpy(part.data() + dst, reference.data() + src + desc.row0 * w, std::size_t{desc.rows} * w);
            src += static_cast<std::size_t>(m.rows) * w; dst += std::size_t{desc.rows} * w;
        }
        lc::scatter_state(full, part, m.rows, desc.row0, desc.rows, mm::rank_state_widths);
    }
    require(full == reference, "component-major KV reassembly");
    require(mm::chain_shape().state_bytes_per_row[0] == 2560 && mm::dedup_state_bytes == 2304, "model binding raw state bytes");
    struct Footprints final : strix::deploy::FootprintProvider {
        std::string model() const override { return "minimax-m3.1-flash"; }
        std::vector<std::string> topologies() const override { return {"tp2"}; }
        std::vector<strix::deploy::Format> formats(const std::string&) const override { return {}; }
        strix::deploy::Footprint footprint(const strix::deploy::Spec&, const strix::deploy::Candidate&, std::uint32_t) const override { return {}; }
    } fp;
    strix::deploy::MinimaxLayercutProvider provider(fp);
    require(provider.layers() == 60 && provider.seam_bytes(24, 19) == 24576 && !provider.seam_bytes(0, 19), "deployment seam binding");
    require(provider.state_bytes(19, 0, 24) == 19 * 24 * 2560, "deployment raw both-rank state");
    require(provider.protocols()[0].gate == "pending" && provider.protocols()[0].name == mm::format &&
        provider.protocols()[0].import_ms < 0 && !provider.range_footprint({}, {}, 0, 0, 24), "unqualified deployment adapter");
    rejects([&] { (void)provider.state_bytes(19, 24, 61); }, "range");
    std::puts("PASS: layercut resolver, schedule, metadata, coverage and KV reassembly");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
