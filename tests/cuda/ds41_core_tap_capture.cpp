// Capture of the deepseek41 host forward for the CUDA layer-core check (host only, no device code).
//   strix-ds41-tap-capture MODEL.gguf --layers N --prompt-ids a,b,c OUT.tap
// Runs HostForward over layers [0, N) one token per step and writes, per step: the token and position, the Engram
// rows the step reads (264-byte rows per Engram layer below N, [layer][column], from the model file), every op output
// buffer of the step in plan order (buffer_tap) and the step's op hashes (last_step_hashes, which also cover the
// logits, the greedy token and the route ids). strix-cuda-ds41-core-tap-check replays the ops on the device.
//
// File: "DS41TAP1", u32 layers, then records {u32 kind, ...}: 1 step {u32 token, u32 position}; 2 engram rows
// {u32 layer, u32 columns, columns * 264 bytes}; 3 buffer {u32 label bytes, label, u64 n, n floats}; 4 hashes
// {u32 count, count x {u32 label bytes, label, u64 hash}}; 0 end.

#include "strix/deepseek41_host_forward.hpp"
#include "strix/ds41_engram.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void put_u32(std::ofstream& o, std::uint32_t v) { o.write(reinterpret_cast<const char*>(&v), 4); }
void put_u64(std::ofstream& o, std::uint64_t v) { o.write(reinterpret_cast<const char*>(&v), 8); }
void put_label(std::ofstream& o, const std::string& s) {
    put_u32(o, static_cast<std::uint32_t>(s.size()));
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

}  // namespace

int main(int argc, char** argv) try {
    std::string model, prompt, out_path;
    std::uint32_t layers = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--layers") layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--prompt-ids") prompt = next();
        else if (model.empty()) model = a;
        else if (out_path.empty()) out_path = a;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (model.empty() || prompt.empty() || out_path.empty())
        throw std::invalid_argument("usage: strix-ds41-tap-capture MODEL.gguf --layers N --prompt-ids a,b,c OUT.tap");
    std::vector<std::uint32_t> ids;
    std::stringstream ss(prompt);
    for (std::string t; std::getline(ss, t, ',');) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));

    std::vector<std::pair<std::string, std::vector<float>>> taps;
    strix::deepseek41::HostForwardOptions options;
    options.layers = layers;
    options.buffer_tap = [&](const std::string& label, const std::vector<float>& v) { taps.emplace_back(label, v); };
    strix::deepseek41::HostForward fwd(model, options);
    const auto& m = fwd.model();
    strix::ds41::EngramHasher hasher(m.engram);
    const auto tables = strix::ds41::engram_tables(m);
    std::ifstream file(model, std::ios::binary);
    std::ofstream out(out_path, std::ios::binary);
    out.write("DS41TAP1", 8);
    put_u32(out, layers);
    std::vector<std::uint64_t> rows(std::size_t{hasher.layers()} * hasher.columns());
    for (std::size_t pos = 0; pos < ids.size(); ++pos) {
        hasher.push(ids[pos], rows);
        taps.clear();
        fwd.step(ids[pos]);
        put_u32(out, 1);
        put_u32(out, ids[pos]);
        put_u32(out, static_cast<std::uint32_t>(pos));
        for (std::uint32_t e = 0; e < hasher.layers(); ++e) {
            if (m.engram.layer_ids[e] >= layers) continue;
            std::vector<char> bytes(std::size_t{hasher.columns()} * 264U);
            for (std::uint32_t k = 0; k < hasher.columns(); ++k) {
                file.seekg(static_cast<std::streamoff>(tables[e].file_offset + rows[std::size_t{e} * hasher.columns() + k] * 264U));
                file.read(bytes.data() + std::size_t{k} * 264U, 264);
                if (!file) throw std::runtime_error("cannot read an Engram row");
            }
            put_u32(out, 2);
            put_u32(out, m.engram.layer_ids[e]);
            put_u32(out, hasher.columns());
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        for (const auto& [label, v] : taps) {
            put_u32(out, 3);
            put_label(out, label);
            put_u64(out, v.size());
            out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4U));
        }
        const auto& hashes = fwd.last_step_hashes();
        put_u32(out, 4);
        put_u32(out, static_cast<std::uint32_t>(hashes.size()));
        for (const auto& [label, h] : hashes) {
            put_label(out, label);
            put_u64(out, h);
        }
        std::printf("step %zu token %u -> %u (%zu buffers)\n", pos, ids[pos], fwd.next_token(), taps.size());
    }
    put_u32(out, 0);
    if (!out) throw std::runtime_error("cannot write " + out_path);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
