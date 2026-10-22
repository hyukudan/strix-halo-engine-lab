#include "strix/ds41_hot_placement.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Hot-expert placement reader and layer split: valid and invalid files, (layer, expert) -> device / weight slot,
// the canonical order and folded prefixes of the split (remote prefix, interior expert, whole half, mixed devices),
// payload accounting, and that combining the requested vectors as specified gives the bits of the canonical combine.
namespace {

using namespace strix::ds41;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::string list(const std::vector<std::uint32_t>& ids) {
    std::string out = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) out += (i ? "," : "") + std::to_string(ids[i]);
    return out + "]";
}

struct FileSpec {
    std::string format = "tensormeld/hot-expert-placement-v1";
    std::string identity = "null";
    std::uint32_t layers = 40, experts = 384, top_k = 6, split = 192;
    std::uint64_t expert_bytes = 9953280;
    std::string combine = "ds41-canonical-halves-v1";
    std::string servers = R"([{"node": "node1", "experts": [0, 192]}, {"node": "node2", "experts": [192, 384]}])";
    std::string pc_ram_device = R"({"id": "pc-ram", "kind": "host", "role": "expert-cache", "budget_gib": 8.0})";
    // layer -> device name -> ids; layers absent here get empty lists.
    std::map<std::uint32_t, std::map<std::string, std::vector<std::uint32_t>>> lists;
    std::uint32_t layer_entries = 40;
    std::function<std::string(std::uint32_t)> layer_override; // non-empty result replaces the entry
};

std::string make_file(const FileSpec& s) {
    std::string out = "{\n  \"format\": \"" + s.format + "\",\n";
    out += "  \"model\": {\"family\": \"deepseek-v4.1-flash\", \"file\": \"m.gguf\", \"model_identity\": " + s.identity +
           ", \"layers\": " + std::to_string(s.layers) + ", \"experts\": " + std::to_string(s.experts) +
           ", \"top_k\": " + std::to_string(s.top_k) + ", \"expert_bytes\": " + std::to_string(s.expert_bytes) +
           ", \"canonical_split\": " + std::to_string(s.split) + "},\n";
    out += "  \"dense\": {\"device\": \"rtx-pro\", \"layers\": \"all\"},\n  \"devices\": [\n";
    out += R"(    {"id": "rtx-pro", "kind": "cuda", "role": "parts-host", "budget_gib": 80.0, "reserve_gib": 9.5, "expert_gib": 70.5},)" "\n";
    out += R"(    {"id": "rtx-4090", "kind": "cuda", "role": "expert-tier", "budget_gib": 20.0, "reserve_gib": 0.0, "expert_gib": 20.0},)" "\n";
    out += "    " + s.pc_ram_device + ",\n";
    out += R"(    {"id": "amd", "kind": "hip", "role": "expert-server", "servers": )" + s.servers + "}\n  ],\n  \"layers\": [\n";
    for (std::uint32_t l = 0; l < s.layer_entries; ++l) {
        std::string entry = s.layer_override ? s.layer_override(l) : "";
        if (entry.empty()) {
            entry = "{\"layer\": " + std::to_string(l);
            const auto it = s.lists.find(l);
            for (const char* name : {"rtx-pro", "rtx-4090", "pc-ram"}) {
                std::vector<std::uint32_t> ids;
                if (it != s.lists.end())
                    if (const auto jt = it->second.find(name); jt != it->second.end()) ids = jt->second;
                entry += std::string(", \"") + name + "\": " + list(ids);
            }
            entry += "}";
        }
        out += "    " + entry + (l + 1U < s.layer_entries ? ",\n" : "\n");
    }
    out += "  ],\n  \"combine\": {\"identity\": \"" + s.combine +
           "\", \"remote_return\": \"raw-per-expert; a remote canonical prefix of a half may return one folded partial\"},\n";
    out += R"(  "fit": {"seed": 1, "tool_version": "w1-hotplace-1"}, "expected": {"R1": {"tok_s": 0.0}}, "extra_key": 1)" "\n}\n";
    return out;
}

bool rejects(const std::string& text, const std::string& needle, const HotPlacementExpect& want = {}) {
    try {
        (void)parse_hot_placement(text, want);
    } catch (const std::invalid_argument& error) {
        const std::string what = error.what();
        if (what.find(needle) == std::string::npos) {
            std::cerr << "  message was: " << what << '\n';
            return false;
        }
        return true;
    }
    return false;
}

// Layer 0 of the split tests: rtx-pro holds 20, 200, 300, 310; rtx-4090 holds 2, 4; pc-ram holds 5, 250; the rest is amd.
HotPlacement split_placement() {
    FileSpec s;
    s.lists[0] = {{"rtx-pro", {20, 200, 300, 310}}, {"rtx-4090", {2, 4}}, {"pc-ram", {5, 250}}};
    s.lists[1] = {{"rtx-pro", {1, 5, 9}}};
    return parse_hot_placement(make_file(s));
}

void test_valid() {
    FileSpec s;
    s.identity = "\"0123456789abcdef\"";
    s.lists[0] = {{"rtx-pro", {1, 5, 9}}, {"rtx-4090", {2, 200}}, {"pc-ram", {7}}};
    s.lists[39] = {{"rtx-pro", {383}}};
    const auto text = make_file(s);
    HotPlacementExpect want;
    want.model_identity = "0123456789abcdef";
    const auto p = parse_hot_placement(text, want);
    expect(p.layers.size() == 40U && p.geometry.experts == 384U && p.geometry.canonical_split == 192U, "geometry");
    expect(p.model_identity && *p.model_identity == "0123456789abcdef", "model identity");
    expect(p.dense_device == HotDevice::rtx_pro && p.devices.size() == 4U, "dense device and devices");
    expect(p.amd_servers.size() == 2U && p.amd_servers[1].node == "node2" && p.amd_servers[1].first == 192U, "amd servers");
    expect(p.device_of(0, 1) == HotDevice::rtx_pro && p.slot_of(0, 1) == 0U, "0/1 rtx-pro slot 0");
    expect(p.device_of(0, 9) == HotDevice::rtx_pro && p.slot_of(0, 9) == 2U, "0/9 rtx-pro slot 2");
    expect(p.device_of(0, 200) == HotDevice::rtx_4090 && p.slot_of(0, 200) == 1U, "0/200 rtx-4090 slot 1");
    expect(p.device_of(0, 7) == HotDevice::pc_ram && p.slot_of(0, 7) == 0U, "0/7 pc-ram slot 0");
    // amd: complement 0, 3, 4, 6, 8, 10, ... dense
    expect(p.device_of(0, 0) == HotDevice::amd && p.slot_of(0, 0) == 0U, "0/0 amd slot 0");
    expect(p.device_of(0, 3) == HotDevice::amd && p.slot_of(0, 3) == 1U, "0/3 amd slot 1");
    expect(p.device_of(0, 10) == HotDevice::amd && p.slot_of(0, 10) == 5U, "0/10 amd slot 5");
    expect(p.layers[0].experts[3].size() == 384U - 6U, "amd complement size");
    expect(p.device_of(39, 383) == HotDevice::rtx_pro && p.device_of(1, 1) == HotDevice::amd, "other layers");
    expect(p.amd_server_of(191) == 0U && p.amd_server_of(192) == 1U, "amd server of");
    expect(p.resident_experts(HotDevice::rtx_pro) == 4U && p.resident_experts(HotDevice::rtx_4090) == 2U, "resident counts");
    // A null identity in the file is accepted whatever the engine's identity.
    FileSpec n;
    expect(!parse_hot_placement(make_file(n), want).model_identity, "null identity accepted");

    // load from a file
    const std::string path = "ds41_hot_placement_test.json";
    { std::ofstream(path, std::ios::binary) << text; }
    const auto loaded = load_hot_placement(path, want);
    expect(loaded.slot_of(0, 200) == 1U, "load_hot_placement");
    std::remove(path.c_str());
    bool threw = false;
    try { (void)load_hot_placement("does-not-exist.json"); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "missing file rejected");
}

void test_invalid() {
    FileSpec base;
    base.lists[0] = {{"rtx-pro", {1, 5, 9}}};
    expect(rejects("{not json", "invalid JSON"), "invalid JSON");
    { auto s = base; s.format = "tensormeld/hot-expert-placement-v2"; expect(rejects(make_file(s), "unknown format"), "format v2"); }
    { auto s = base; s.top_k = 8; expect(rejects(make_file(s), "model.top_k"), "top_k mismatch"); }
    { auto s = base; s.expert_bytes = 1; expect(rejects(make_file(s), "model.expert_bytes"), "expert_bytes mismatch"); }
    { auto s = base; s.split = 128; expect(rejects(make_file(s), "model.canonical_split"), "split mismatch"); }
    {
        HotPlacementExpect want;
        want.geometry.layers = 61;
        expect(rejects(make_file(base), "model.layers", want), "layers mismatch");
    }
    {
        auto s = base;
        s.identity = "\"aaaaaaaaaaaaaaaa\"";
        HotPlacementExpect want;
        want.model_identity = "bbbbbbbbbbbbbbbb";
        expect(rejects(make_file(s), "model identity", want), "identity mismatch");
    }
    { auto s = base; s.combine = "per-device-partials"; expect(rejects(make_file(s), "combine identity"), "combine identity"); }
    { auto s = base; s.lists[3] = {{"rtx-pro", {5, 4}}}; expect(rejects(make_file(s), "strictly ascending"), "descending ids"); }
    { auto s = base; s.lists[3] = {{"rtx-pro", {5, 5}}}; expect(rejects(make_file(s), "strictly ascending"), "repeated id"); }
    { auto s = base; s.lists[3] = {{"rtx-pro", {5}}, {"pc-ram", {5}}}; expect(rejects(make_file(s), "two devices"), "id on two devices"); }
    { auto s = base; s.lists[3] = {{"rtx-4090", {384}}}; expect(rejects(make_file(s), "out of range"), "id out of range"); }
    { auto s = base; s.layer_entries = 39; expect(rejects(make_file(s), "39 entries"), "missing layer"); }
    {
        auto s = base;
        s.layer_override = [](std::uint32_t l) { return l == 7U ? std::string(R"({"layer": 8, "rtx-pro": []})") : std::string(); };
        expect(rejects(make_file(s), "layers[7].layer"), "layer index");
    }
    {
        auto s = base;
        s.layer_override = [](std::uint32_t l) { return l == 2U ? std::string(R"({"layer": 2, "rtx-5090": [1]})") : std::string(); };
        expect(rejects(make_file(s), "unknown device list"), "unknown device list");
    }
    {
        auto s = base; // an additive non-list key is accepted
        s.layer_override = [](std::uint32_t l) { return l == 2U ? std::string(R"({"layer": 2, "note": "x", "rtx-pro": [3]})") : std::string(); };
        expect(parse_hot_placement(make_file(s)).device_of(2, 3) == HotDevice::rtx_pro, "additive layer key");
    }
    {
        auto s = base;
        s.layer_override = [](std::uint32_t l) { return l == 2U ? std::string(R"({"layer": 2, "amd": [1]})") : std::string(); };
        expect(rejects(make_file(s), "implicit"), "explicit amd list");
    }
    {
        auto s = base;
        s.servers = R"([{"node": "node1", "experts": [0, 190]}, {"node": "node2", "experts": [192, 384]}])";
        expect(rejects(make_file(s), "contiguously"), "server gap");
    }
    {
        auto s = base;
        s.servers = R"([{"node": "node1", "experts": [0, 192]}])";
        expect(rejects(make_file(s), "end at the expert count"), "servers short");
    }
    {
        // 20 GiB of rtx-4090 hold floor(20 * 2^30 / 9953280) = 2157 experts: 54 per layer over 40 layers is over.
        auto s = base;
        std::vector<std::uint32_t> ids;
        for (std::uint32_t e = 100; e < 154; ++e) ids.push_back(e);
        for (std::uint32_t l = 0; l < 40; ++l) s.lists[l]["rtx-4090"] = ids;
        expect(rejects(make_file(s), "\"rtx-4090\" holds 2160 experts"), "4090 over memory");
        ids.pop_back();
        for (std::uint32_t l = 0; l < 40; ++l) s.lists[l]["rtx-4090"] = ids; // 2120 fits
        expect(parse_hot_placement(make_file(s)).resident_experts(HotDevice::rtx_4090) == 2120U, "4090 within memory");
    }
    {
        // pc-ram as in the format example: budget 0, no expert_gib -> cannot hold any expert.
        auto s = base;
        s.pc_ram_device = R"({"id": "pc-ram", "kind": "host", "role": "expert-cache", "budget_gib": 0.0})";
        expect(parse_hot_placement(make_file(s)).resident_experts(HotDevice::pc_ram) == 0U, "empty pc-ram with budget 0");
        s.lists[5]["pc-ram"] = {17};
        expect(rejects(make_file(s), "\"pc-ram\" holds 1"), "pc-ram over budget");
    }
}

std::vector<float> weights_for(std::size_t n) {
    std::vector<float> w(n);
    for (std::size_t i = 0; i < n; ++i) w[i] = 0.05F + 0.01F * static_cast<float>(i);
    return w;
}

void test_split_cases() {
    const auto p = split_placement();
    const auto amd = HotDevice::amd;
    {
        // All on the parts-host (rtx-pro + pc-ram), router order shuffled.
        const std::vector<std::uint32_t> ids{310, 20, 5, 300, 250, 200};
        const auto s = split_hot_layer(p, 0, ids, weights_for(6), 1);
        expect(s.all_local() && s.network_input_bytes == 0U && s.network_output_bytes == 0U && s.pcie_input_bytes == 0U &&
                   s.pcie_output_bytes == 0U,
               "all local: no targets");
        expect(s.low_counts[0] == 2U, "all local: low count");
        const std::vector<std::uint32_t> want{5, 20, 200, 250, 300, 310};
        for (std::uint32_t k = 0; k < 6; ++k) expect(s.order[k].expert == want[k], "all local: ascending order");
        expect(s.order[0].route_slot == 2U && s.order[0].weight == weights_for(6)[2], "all local: route slot and weight carried");
        expect(s.order[0].device == HotDevice::pc_ram && s.order[0].weight_slot == 0U, "all local: pc-ram slot");
        expect(s.order[3].device == HotDevice::pc_ram && s.order[3].weight_slot == 1U, "all local: pc-ram slot 1");
        expect(s.order[2].weight_slot == 1U && s.order[5].weight_slot == 3U, "all local: rtx-pro slots");
        expect(s.experts[2].size() == 2U && s.upload_bytes == 2U * 9953280U && s.local_positions() == 6U, "all local: upload");
    }
    {
        // Remote prefix in the low half: 3, 10 on amd (node1) lead the low half {3, 10, 20}; high all rtx-pro.
        const std::vector<std::uint32_t> ids{200, 10, 300, 20, 3, 310};
        const auto s = split_hot_layer(p, 0, ids, weights_for(6), 1);
        expect(s.targets.size() == 1U && s.targets[0].target == HotTarget{amd, 0}, "low prefix: one target, node1");
        expect(s.folded_prefix[0] == 2U && s.folded_prefix[1] == 0U, "low prefix: folded 2");
        const auto& t = s.targets[0];
        expect(t.requests.size() == 1U && t.requests[0].kind == HotVectorKind::folded_prefix && t.requests[0].count == 2U &&
                   t.requests[0].expert == 3U && t.requests[0].route_slot == 4U && t.requests[0].half == 0U,
               "low prefix: one folded request");
        expect(t.folded_vectors == 1U && t.raw_vectors == 0U && t.folded_positions == 2U, "low prefix: counts");
        expect(t.input_bytes == 5808U && t.output_bytes == 20480U && t.rows == 1U, "low prefix: bytes");
        expect(t.experts == std::vector<std::uint32_t>{3, 10}, "low prefix: target experts");
        expect(s.order[0].folded && s.order[1].folded && !s.order[2].folded, "low prefix: folded flags");
    }
    {
        // amd positions not contiguous from the start: low {1 amd, 20 rtx-pro, 30 amd} -> two raw vectors.
        const std::vector<std::uint32_t> ids{30, 20, 1, 200, 300, 310};
        const auto s = split_hot_layer(p, 0, ids, weights_for(6), 1);
        expect(s.folded_prefix[0] == 0U, "split prefix: not folded (1 and 30 not contiguous)");
        expect(s.targets.size() == 1U && s.targets[0].raw_vectors == 2U && s.targets[0].output_bytes == 40960U, "split prefix: two raw");
        expect(s.targets[0].requests[0].first == 0U && s.targets[0].requests[1].first == 2U, "split prefix: positions 0 and 2");
        const std::vector<std::uint32_t> ids2{25, 20, 30, 200, 300, 310}; // low {20 local, 25 amd, 30 amd}: interior
        const auto s2 = split_hot_layer(p, 0, ids2, weights_for(6), 1);
        expect(s2.folded_prefix[0] == 0U && s2.targets[0].raw_vectors == 2U, "interior: raw");
        expect(s2.targets[0].requests[0].first == 1U && s2.targets[0].requests[0].expert == 25U, "interior: position 1");
        const std::vector<std::uint32_t> ids3{25, 20, 21, 200, 300, 310}; // low {20 rtx-pro, 21 amd, 25 amd}: trailing pair
        const auto s3 = split_hot_layer(p, 0, ids3, weights_for(6), 1);
        expect(s3.targets[0].raw_vectors == 2U && s3.targets[0].requests[0].expert == 21U, "interior pair: raw");
        // remote, remote, local, remote in one half: low {3 amd, 6 amd, 20 rtx-pro, 30 amd} -> 1 folded (3, 6) + 1 raw (30).
        const std::vector<std::uint32_t> ids4{30, 300, 6, 20, 200, 3};
        const auto s4 = split_hot_layer(p, 0, ids4, weights_for(6), 1);
        expect(s4.low_counts[0] == 4U && s4.folded_prefix[0] == 2U && s4.targets.size() == 1U, "prefix then raw: folded 2");
        const auto& t4 = s4.targets[0];
        expect(t4.folded_vectors == 1U && t4.raw_vectors == 1U && t4.folded_positions == 2U && t4.output_bytes == 2U * 20480U,
               "prefix then raw: 1 folded + 1 raw");
        expect(t4.requests.size() == 2U && t4.requests[0].kind == HotVectorKind::folded_prefix && t4.requests[0].expert == 3U &&
                   t4.requests[1].kind == HotVectorKind::raw && t4.requests[1].first == 3U && t4.requests[1].expert == 30U &&
                   t4.requests[1].route_slot == 0U,
               "prefix then raw: requests");
        expect(s4.order[0].folded && s4.order[1].folded && !s4.order[2].folded && !s4.order[3].folded, "prefix then raw: flags");
        // Same with rtx-4090 leading: low {2, 4 rtx-4090, 20 rtx-pro, 30 amd} -> 4090 folded, amd raw.
        const std::vector<std::uint32_t> ids5{30, 300, 4, 20, 200, 2};
        const auto s5 = split_hot_layer(p, 0, ids5, weights_for(6), 1);
        expect(s5.folded_prefix[0] == 2U && s5.targets.size() == 2U && s5.targets[0].folded_vectors == 1U && s5.targets[1].raw_vectors == 1U,
               "4090 prefix, amd raw later");
    }
    {
        // Single leading remote expert stays raw.
        const std::vector<std::uint32_t> ids{1, 20, 5, 200, 300, 310};
        const auto s = split_hot_layer(p, 0, ids, weights_for(6), 1);
        expect(s.folded_prefix[0] == 0U && s.targets[0].raw_vectors == 1U && s.targets[0].requests[0].first == 0U, "single lead: raw");
    }
    {
        // Whole half remote: high {201, 202, 203} on amd node2; low local + pc-ram.
        const std::vector<std::uint32_t> ids{203, 5, 201, 20, 202, 250};
        const auto s = split_hot_layer(p, 0, ids, weights_for(6), 1);
        expect(s.low_counts[0] == 2U, "whole half: low count"); // 5, 20; 250 is high
        // high = {201, 202, 203, 250}: 250 pc-ram trails, amd holds the leading three and all of its positions.
        expect(s.folded_prefix[1] == 3U, "whole half: folded 3 of the high half");
        expect(s.targets.size() == 1U && s.targets[0].target == HotTarget{amd, 1} && s.targets[0].folded_vectors == 1U,
               "whole half: node2 folded");
        const std::vector<std::uint32_t> ids2{203, 5, 201, 20, 202, 204};
        const auto s2 = split_hot_layer(p, 0, ids2, weights_for(6), 1);
        expect(s2.folded_prefix[1] == 4U && s2.targets[0].requests[0].count == 4U && s2.network_output_bytes == 20480U,
               "whole half: entire high half one vector");
    }
    {
        // Mixed devices: low {2 rtx-4090, 3 amd, 6 amd}: 4090 leads with one -> raw; amd not a prefix -> raw.
        const std::vector<std::uint32_t> ids{6, 3, 2, 200, 300, 310};
        const auto s = split_hot_layer(p, 0, ids, weights_for(6), 1);
        expect(s.targets.size() == 2U && s.targets[0].target == HotTarget{HotDevice::rtx_4090, 0} &&
                   s.targets[1].target == HotTarget{amd, 0},
               "mixed: 4090 then node1");
        expect(s.folded_prefix[0] == 0U && s.targets[0].raw_vectors == 1U && s.targets[1].raw_vectors == 2U, "mixed: all raw");
        expect(s.pcie_input_bytes == 5808U && s.pcie_output_bytes == 20480U && s.network_input_bytes == 5808U &&
                   s.network_output_bytes == 2U * 20480U,
               "mixed: pcie and network bytes");
        // low {2, 4 rtx-4090, 6 amd}: 4090 prefix folded, amd raw.
        const std::vector<std::uint32_t> ids2{6, 4, 2, 200, 300, 310};
        const auto s2 = split_hot_layer(p, 0, ids2, weights_for(6), 1);
        expect(s2.folded_prefix[0] == 2U && s2.targets[0].folded_vectors == 1U && s2.targets[1].raw_vectors == 1U &&
                   s2.targets[1].requests[0].first == 2U,
               "mixed: 4090 prefix folded, amd raw");
        // low {3 amd, 4 rtx-4090, 6 amd}: interleaved -> raw everywhere
        const std::vector<std::uint32_t> ids3{6, 4, 3, 200, 300, 310};
        const auto s3 = split_hot_layer(p, 0, ids3, weights_for(6), 1);
        expect(s3.folded_prefix[0] == 0U && s3.targets[1].raw_vectors == 2U && s3.targets[0].raw_vectors == 1U, "mixed: interleaved raw");
        // Both halves remote on different servers: low {3, 6, 7} node1, high {193, 194, 195} node2.
        const std::vector<std::uint32_t> ids4{195, 3, 193, 7, 194, 6};
        const auto s4 = split_hot_layer(p, 0, ids4, weights_for(6), 1);
        expect(s4.targets.size() == 2U && s4.targets[0].target == HotTarget{amd, 0} && s4.targets[1].target == HotTarget{amd, 1} &&
                   s4.network_output_bytes == 2U * 20480U && s4.network_input_bytes == 2U * 5808U && s4.pcie_input_bytes == 0U,
               "both halves: one folded vector per server");
    }
    {
        // Rejections.
        bool threw = false;
        try { (void)split_hot_layer(p, 0, std::vector<std::uint32_t>{1, 1, 2, 3, 4, 5}, weights_for(6), 1); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "repeat rejected");
        threw = false;
        try { (void)split_hot_layer(p, 0, std::vector<std::uint32_t>{1, 2, 3, 4, 5, 384}, weights_for(6), 1); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "id range rejected");
        threw = false;
        try { (void)split_hot_layer(p, 40, std::vector<std::uint32_t>{1, 2, 3, 4, 5, 6}, weights_for(6), 1); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "layer range rejected");
        threw = false;
        try { (void)split_hot_layer(p, 0, std::vector<std::uint32_t>{1, 2, 3, 4, 5, 6}, weights_for(6), 2); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "size rejected");
    }
}

// R = 8: random routes over a random placement. Rebuild each row's combine from the requested vectors the way the
// parts-host would (folded partial as the starting accumulator, then raw / local terms in canonical order) and
// compare with the canonical fold bit for bit; check the accounting against a direct count.
void test_split_r8() {
    std::mt19937 rng(7);
    FileSpec s;
    for (std::uint32_t l = 0; l < 40; ++l) {
        std::vector<std::uint32_t> pro, g4090, ram;
        for (std::uint32_t e = 0; e < 384; ++e) {
            const auto u = rng() % 10U;
            if (u < 4U) pro.push_back(e);
            else if (u < 5U) g4090.push_back(e);
            else if (u < 6U) ram.push_back(e);
        }
        if (l % 5U == 0U) g4090.clear();
        s.lists[l] = {{"rtx-pro", pro}, {"rtx-4090", g4090}, {"pc-ram", ram}};
    }
    s.pc_ram_device = R"({"id": "pc-ram", "kind": "host", "role": "expert-cache", "budget_gib": 48.0, "reserve_gib": 4.0})";
    // rtx-4090 at 10 % of 384 over 40 layers (~1536) fits 20 GiB; rtx-pro at 40 % (~6144) fits 70.5 GiB.
    const auto p = parse_hot_placement(make_file(s));
    constexpr std::uint32_t hidden = 16, rows = 8, top_k = 6;
    std::uniform_real_distribution<float> value(-2.0F, 2.0F), weight(0.01F, 0.6F);
    std::size_t folded_seen = 0, raw_after_fold = 0;
    for (std::uint32_t trial = 0; trial < 400; ++trial) {
        const auto layer = static_cast<std::uint32_t>(rng() % 40U);
        // Concentrated routes so that rows share experts and halves often fall on one target.
        std::vector<std::uint32_t> ids(rows * top_k);
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t k = 0; k < top_k; ++k) {
                std::uint32_t e = 0;
                bool fresh = false;
                while (!fresh) {
                    e = trial % 3U == 0U ? static_cast<std::uint32_t>(rng() % 384U) : static_cast<std::uint32_t>((rng() % 24U) * 16U + (trial % 16U));
                    fresh = true;
                    for (std::uint32_t j = 0; j < k; ++j) fresh = fresh && ids[r * top_k + j] != e;
                }
                ids[r * top_k + k] = e;
            }
        std::vector<float> w(rows * top_k), raw(rows * top_k * hidden), shared(rows * hidden);
        for (auto& x : w) x = weight(rng);
        for (auto& x : raw) x = value(rng);
        for (auto& x : shared) x = value(rng);
        const auto split = split_hot_layer(p, layer, ids, w, rows);

        // Canonical combine directly from the routes.
        std::vector<float> want(rows * hidden), got(rows * hidden);
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<std::pair<std::uint32_t, std::uint32_t>> sorted;
            for (std::uint32_t k = 0; k < top_k; ++k) sorted.emplace_back(ids[r * top_k + k], r * top_k + k);
            std::sort(sorted.begin(), sorted.end());
            for (std::uint32_t o = 0; o < hidden; ++o) {
                float low = 0.0F, high = 0.0F;
                for (const auto& [e, slot] : sorted) {
                    const float term = w[slot] * raw[slot * hidden + o];
                    if (e < 192U) low = low + term; else high = high + term;
                }
                want[r * hidden + o] = (shared[r * hidden + o] + low) + high;
            }
        }
        // What the targets send: folded partials computed on the server from its own raws and the weights.
        std::map<std::pair<std::uint32_t, unsigned>, std::vector<float>> partial;
        std::vector<bool> received(rows * top_k, false); // raw vectors received, by route slot
        std::uint64_t input = 0, output = 0, vectors = 0;
        for (const auto& t : split.targets) {
            input += t.input_bytes;
            output += t.output_bytes;
            for (const auto& q : t.requests) {
                ++vectors;
                const auto* row = split.order.data() + q.row * top_k;
                const auto begin = q.half == 0U ? 0U : split.low_counts[q.row];
                if (q.kind == HotVectorKind::raw) {
                    expect(row[begin + q.first].route_slot == q.route_slot && hot_device_returns_vectors(row[begin + q.first].device),
                           "r8: raw request points at a remote position");
                    received[q.route_slot] = true;
                    continue;
                }
                ++folded_seen;
                std::vector<float> acc(hidden, 0.0F);
                for (std::uint32_t i = 0; i < q.count; ++i) {
                    const auto& slot = row[begin + i];
                    expect(slot.device == t.target.device && slot.folded, "r8: folded prefix on its target");
                    for (std::uint32_t o = 0; o < hidden; ++o) acc[o] = acc[o] + slot.weight * raw[slot.route_slot * hidden + o];
                }
                partial[{q.row, q.half}] = acc;
            }
        }
        // The parts-host combine.
        std::uint32_t remote_positions = 0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto* row = split.order.data() + r * top_k;
            const auto low_count = split.low_counts[r];
            std::vector<float> half_sum[2];
            for (unsigned half = 0; half < 2U; ++half) {
                const auto begin = half == 0U ? 0U : low_count, end = half == 0U ? low_count : top_k;
                const auto prefix = split.folded_prefix[r * 2U + half];
                auto& acc = half_sum[half];
                acc.assign(hidden, 0.0F);
                if (prefix != 0U) acc = partial.at({r, half});
                for (auto pos = begin + prefix; pos < end; ++pos) {
                    const auto& slot = row[pos];
                    expect(pos == begin || row[pos - 1U].expert < slot.expert, "r8: ascending");
                    expect(!slot.folded, "r8: no folded position after the prefix");
                    if (hot_device_returns_vectors(slot.device)) {
                        expect(received[slot.route_slot], "r8: remote raw requested");
                        ++remote_positions;
                        const auto& lead = row[begin];
                        if (prefix != 0U && slot.device == lead.device && slot.server == lead.server) ++raw_after_fold;
                    }
                    for (std::uint32_t o = 0; o < hidden; ++o) acc[o] = acc[o] + slot.weight * raw[slot.route_slot * hidden + o];
                }
                remote_positions += prefix;
            }
            for (std::uint32_t o = 0; o < hidden; ++o) got[r * hidden + o] = (shared[r * hidden + o] + half_sum[0][o]) + half_sum[1][o];
        }
        bool same = true;
        for (std::size_t i = 0; i < want.size(); ++i) same = same && std::bit_cast<std::uint32_t>(want[i]) == std::bit_cast<std::uint32_t>(got[i]);
        expect(same, "r8: combine from the split is bitwise canonical (trial " + std::to_string(trial) + ")");

        // Accounting.
        const std::uint32_t remote = split.positions[1] + split.positions[3];
        expect(remote == remote_positions, "r8: remote positions");
        expect(split.positions[0] + split.positions[1] + split.positions[2] + split.positions[3] == rows * top_k, "r8: positions");
        expect(output == vectors * 20480U && output == split.network_output_bytes + split.pcie_output_bytes &&
                   input == split.network_input_bytes + split.pcie_input_bytes,
               "r8: bytes");
        for (const auto& t : split.targets) {
            std::vector<bool> has_row(rows, false);
            std::vector<std::uint32_t> experts;
            for (std::uint32_t i = 0; i < rows * top_k; ++i) {
                const auto& slot = split.order[i];
                const bool on = slot.device == t.target.device && (slot.device != HotDevice::amd || slot.server == t.target.server);
                if (!on) continue;
                has_row[i / top_k] = true;
                experts.push_back(slot.expert);
            }
            std::sort(experts.begin(), experts.end());
            experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
            expect(t.experts == experts, "r8: target distinct experts");
            expect(t.rows == static_cast<std::uint32_t>(std::count(has_row.begin(), has_row.end(), true)) &&
                       t.input_bytes == t.rows * 5808U,
                   "r8: target rows and input bytes");
        }
        std::vector<std::uint32_t> ram;
        for (const auto& slot : split.order)
            if (slot.device == HotDevice::pc_ram) ram.push_back(slot.expert);
        std::sort(ram.begin(), ram.end());
        ram.erase(std::unique(ram.begin(), ram.end()), ram.end());
        expect(split.experts[2] == ram && split.upload_bytes == ram.size() * 9953280U, "r8: pc-ram upload");
    }
    expect(folded_seen > 50U, "r8: folded prefixes exercised (" + std::to_string(folded_seen) + ")");
    expect(raw_after_fold > 10U, "r8: raw after a folded prefix exercised (" + std::to_string(raw_after_fold) + ")");
    std::cout << "r8: 400 windows, " << folded_seen << " folded partials, " << raw_after_fold
              << " raw vectors of a folding target after a position elsewhere\n";
}

} // namespace

int main() {
    try {
        test_valid();
        test_invalid();
        test_split_cases();
        test_split_r8();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ds41 hot placement tests passed\n";
    return 0;
}
