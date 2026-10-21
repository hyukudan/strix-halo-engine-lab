#include "strix/deepseek41_host_forward.hpp"
#include "strix/ds41_expert_tp.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Greedy decode of a trimmed deepseek41 model (layers [0, N) + head) on the host reference, one token per step.
//   strix-deepseek41-trimmed MODEL.gguf --layers 4 --prompt-ids 0,671,... --generate 8 [--logits OUT.f32] [--op-hashes]
//   [--streams OUT.f32: residual stream after every layer of every step] [--dump-op op@layer[/site] OUT.f32]
// Prints one line per step (fed token, greedy token, logits hash); --logits writes the FP32 logits of every step
// (steps x vocab, little endian) for comparison with the reference model run on the same layers.
int main(int argc, char** argv) try {
    std::string model, prompt, logits_path;
    std::uint32_t layers = 4, generate = 8;
    bool op_hashes = false;
    std::string streams_path, dump_op, dump_path, overlay, calib_path;
    std::map<std::string, std::pair<std::vector<double>, std::uint64_t>> calib;  // per-channel sums of squares of projection inputs
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--layers") layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--prompt-ids") prompt = next();
        else if (a == "--generate") generate = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--logits") logits_path = next();
        else if (a == "--op-hashes") op_hashes = true;
        else if (a == "--streams") streams_path = next();
        else if (a == "--dump-op") { dump_op = next(); dump_path = next(); }
        else if (a == "--overlay") overlay = next();
        else if (a == "--calib") calib_path = next();
        else if (model.empty()) model = a;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (model.empty() || prompt.empty())
        throw std::invalid_argument("usage: strix-deepseek41-trimmed MODEL.gguf --layers N --prompt-ids a,b,c [--generate G] [--logits OUT.f32] [--op-hashes]");
    std::vector<std::uint32_t> ids;
    std::stringstream ss(prompt);
    for (std::string t; std::getline(ss, t, ',');) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));

    const auto t0 = std::chrono::steady_clock::now();
    strix::deepseek41::HostForwardOptions options;
    options.layers = layers;
    options.weights_overlay = overlay;
    if (!calib_path.empty())
        options.buffer_tap = [&](const std::string& label, const std::vector<float>& v) {
            if (!label.starts_with("rms_norm@") && !label.starts_with("attention@") && !label.starts_with("o_a@")) return;
            auto& [sum, n] = calib[label];
            if (sum.empty()) sum.assign(v.size(), 0.0);
            for (std::size_t i = 0; i < v.size(); ++i) sum[i] += static_cast<double>(v[i]) * static_cast<double>(v[i]);
            ++n;
        };
    strix::deepseek41::HostForward fwd(model, options);
    if (!dump_op.empty()) fwd.set_dump(dump_op);
    if (!overlay.empty()) std::cout << fwd.overlay_description() << '\n';
    std::ofstream dump_out;
    if (!dump_path.empty()) dump_out.open(dump_path, std::ios::binary);
    std::ofstream streams_out;
    if (!streams_path.empty()) streams_out.open(streams_path, std::ios::binary);
    std::cout << "plan: layers [0, " << layers << "), " << fwd.plan().steps.size() << " steps; host stand-ins:";
    for (const auto op : strix::deepseek41::HostForward::stand_in_ops()) std::cout << ' ' << strix::deepseek41::to_string(op);
    std::cout << '\n';
    if (strix::ds41::expert_tp::enabled_from_env()) std::cout << "expert tp: " << strix::ds41::expert_tp::identity << '\n';
    std::ofstream logits_out;
    if (!logits_path.empty()) logits_out.open(logits_path, std::ios::binary);
    std::vector<std::uint32_t> seq = ids;
    for (std::size_t pos = 0; pos < seq.size(); ++pos) {
        const auto& logits = fwd.step(seq[pos]);
        const auto next = fwd.next_token();
        std::uint64_t h = 14695981039346656037ULL;
        for (const float f : logits) {
            std::uint32_t b;
            std::memcpy(&b, &f, 4);
            for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("pos %zu fed %u argmax %u top %.4f logits_fnv %016llx (%.1f s)\n", pos, seq[pos], next, static_cast<double>(logits[next]),
                    static_cast<unsigned long long>(h), s);
        if (op_hashes)
            for (const auto& [name, hash] : fwd.last_step_hashes())
                std::printf("  %s %016llx\n", name.c_str(), static_cast<unsigned long long>(hash));
        if (dump_out) dump_out.write(reinterpret_cast<const char*>(fwd.dumped().data()), static_cast<std::streamsize>(fwd.dumped().size() * 4U));
        if (streams_out)
            for (const auto& st : fwd.last_step_streams())
                streams_out.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 4U));
        if (logits_out) logits_out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * 4U));
        if (pos + 1U == seq.size() && seq.size() < ids.size() + generate) seq.push_back(next);
        std::fflush(stdout);
    }
    if (!calib_path.empty()) {  // "label n count\n" then n doubles, per projection input (requant.py --imatrix)
        std::ofstream f(calib_path, std::ios::binary);
        for (const auto& [label, e] : calib) {
            const std::string head = label + " " + std::to_string(e.first.size()) + " " + std::to_string(e.second) + "\n";
            f.write(head.data(), static_cast<std::streamsize>(head.size()));
            f.write(reinterpret_cast<const char*>(e.first.data()), static_cast<std::streamsize>(e.first.size() * 8U));
        }
        if (!f) throw std::runtime_error("cannot write " + calib_path);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-deepseek41-trimmed: " << error.what() << '\n';
    return 1;
}
