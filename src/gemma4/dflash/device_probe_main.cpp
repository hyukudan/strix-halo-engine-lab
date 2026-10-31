// Explicit full-model device qualification/measurement. Never registered with ctest.
#include "device.hpp"
#include "gemma4/kernels/device.hpp"
#include "gemma4/ref/npz.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace g4 = strix::gemma4;
namespace df = g4::dflash;
using Json = strix::serve::Json;
using Clock = std::chrono::steady_clock;
namespace {
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::uint32_t greedy(std::span<const float> logits) {
    unsigned best = 0;
    for (unsigned i = 1; i < logits.size(); ++i) if (g4::better(logits[i], i, logits[best], best)) best = i;
    return best;
}
template<class T> Json array(std::span<const T> values) {
    Json out = Json::array(); for (auto x : values) out.push(Json(std::uint64_t(x))); return out;
}
void bytes(std::span<const float> a, std::span<const float> b, const char* stage) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size_bytes()))
        throw std::runtime_error(std::string("host/device bytes: ") + stage);
}
struct Resident {
    g4::gpu::Buffer storage;
    g4::LinearView view;
    explicit Resident(g4::LinearView host) : storage(std::size_t(host.outputs) * host.row_bytes), view(host) {
        storage.upload(host.data, storage.size()); view.data = storage.as<std::uint8_t>();
    }
};
std::vector<float> host_logits(const df::Weights& w, std::span<const float> hidden, g4::LinearView head) {
    std::vector<float> result(std::size_t(df::max_drafts) * head.outputs);
    std::vector<float> coefficients(head.inputs);
    for (unsigned t = 0; t < head.outputs; ++t) {
        for (unsigned i = 0; i < head.inputs; ++i) coefficients[i] = g4::weight_at(head, t, i);
        for (unsigned r = 0; r < df::max_drafts; ++r) {
            float value = g4::chunk16_dot([&](unsigned i) { return coefficients[i]; },
                hidden.data() + std::size_t(r + 1) * head.inputs, head.inputs) * w.logit_scale;
            if (w.logit_cap > 0) value = g4::declared_tanh(value / w.logit_cap) * w.logit_cap;
            result[std::size_t(r) * head.outputs + t] = value;
        }
    }
    return result;
}
struct Prefill {
    unsigned anchor = 0;
    std::vector<float> taps;
};
Prefill prefill(g4::gpu::DeviceTarget& target, g4::gpu::DeviceStream& stream, std::span<const unsigned> prompt,
    df::DeviceDrafter* drafter, df::DraftContext* host_context, const df::Weights& w, df::RopeView rope) {
    Prefill result;
    for (std::size_t p = 0, batch = 1; p < prompt.size(); ++batch) {
        const auto n = std::min<std::size_t>(1 + (batch - 1) % 8, prompt.size() - p);
        auto txn = stream.begin(); std::vector<g4::RowPlan> rows(n);
        std::vector<std::uint64_t> positions(n);
        for (std::size_t r = 0; r < n; ++r) {
            positions[r] = p + r; rows[r] = {0, 0, p + r, prompt[p + r], txn.get(), r, true};
        }
        const auto out = target.forward_rows(rows); txn->commit(n);
        result.anchor = greedy(std::span(out.logits).last(out.vocab));
        if (drafter) drafter->append(out.taps, positions);
        if (host_context) {
            host_context->append(w, rope, out.taps, positions);
            for (unsigned l = 0; l < w.shape.layers; ++l)
                for (auto pos : positions) {
                    bytes(host_context->key(l, pos), drafter->context_key(l, pos), "context K");
                    bytes(host_context->value(l, pos), drafter->context_value(l, pos), "context V");
                }
        }
        p += n;
    }
    return result;
}
void run(g4::gpu::DeviceTarget& target, const g4::HostWeights& hw, const df::Weights& w, df::RopeView rope,
    g4::LinearView assets, const std::filesystem::path& fixture, unsigned count, bool check_bytes, const std::string& modes) {
    const auto arrays = g4::read_npz(fixture);
    const auto raw = arrays.at("prompt_ids").values<std::int32_t>("<i4");
    require(!raw.empty(), "empty prompt"); std::vector<unsigned> prompt;
    for (auto t : raw) { require(t >= 0 && unsigned(t) < hw.config.vocab, "prompt ID"); prompt.push_back(unsigned(t)); }
    const auto capacity = unsigned(prompt.size()) + count + df::block;
    require(capacity <= rope.positions, "draft rotary reservation");
    std::vector<unsigned> serial;
    std::string serial_digest;
    double plain_ms = 0;
    {
        g4::gpu::DeviceStream stream(hw.config, capacity);
        auto anchor = prefill(target, stream, prompt, nullptr, nullptr, w, rope).anchor;
        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i) {
            auto txn = stream.begin(); g4::RowPlan row{0, 0, txn->frontier(), anchor, txn.get(), 0, true};
            const auto out = target.forward_rows(std::span(&row, 1)); txn->commit(1);
            anchor = greedy(out.logits); serial.push_back(anchor);
        }
        plain_ms = ms(start); serial_digest = stream.digest();
    }
    Json baseline = Json::object(); baseline.set("event", "serial"); baseline.set("fixture", fixture.stem().string());
    baseline.set("tokens", array<unsigned>(serial)); baseline.set("ms", plain_ms); baseline.set("digest", serial_digest);
    std::cout << baseline.dump() << std::endl;
    if (check_bytes) {
        g4::gpu::DeviceStream stream(hw.config, capacity);
        df::DeviceDrafter device(w, rope, assets, assets); df::DraftContext host(w.shape);
        auto byte_prompt = prompt;
        // The first eight append groups have sizes 1..8 (36 real target rows).
        // Continue the same real greedy prefix when a short prompt needs rows.
        for (unsigned i = 0; byte_prompt.size() < 36; ++i) {
            require(i < serial.size(), "byte gate needs at least 36 processed rows");
            byte_prompt.push_back(serial[i]);
        }
        auto anchor = prefill(target, stream, byte_prompt, &device, &host, w, rope).anchor;
        const auto embedded = df::embed_block(anchor, hw.embedding);
        const auto hidden = df::block_forward(w, host, rope, embedded, stream.frontier());
        bytes(hidden, device.forward(embedded, stream.frontier()), "eight-row block");
        const auto ids = device.select(hidden);
        const auto logits = host_logits(w, hidden, hw.embedding);
        bytes(logits, device.selected_logits(), "seven-row shared head");
        for (unsigned r = 0; r < df::max_drafts; ++r)
            require(ids[r] == greedy(std::span(logits).subspan(std::size_t(r) * hw.config.vocab, hw.config.vocab)), "proposal ID");
        require(device.draft(anchor, stream.frontier()) == ids, "device embedding/block/head");
        Json receipt = Json::object(); receipt.set("event", "draft_bytes"); receipt.set("verdict", "EXACT");
        receipt.set("fixture", fixture.stem().string()); receipt.set("proposal_ids", array<unsigned>(ids));
        receipt.set("block_ms", device.timings().block_ms); receipt.set("head_ms", device.timings().head_ms);
        std::cout << receipt.dump() << std::endl;
    }
    std::vector<unsigned> windows;
    if (modes == "all") for (unsigned k = 0; k <= 7; ++k) windows.push_back(k);
    else if (modes == "auto") windows.push_back(0);
    else windows.push_back(unsigned(std::stoul(modes)));
    for (unsigned k : windows) {
        require(k <= 7, "k must be auto/all/0..7");
        g4::gpu::DeviceStream stream(hw.config, capacity);
        df::DeviceDrafter device(w, rope, assets, assets);
        auto anchor = prefill(target, stream, prompt, &device, nullptr, w, rope).anchor;
        df::Acceptance acceptance; df::DeviceTimings total;
        std::vector<unsigned> ids;
        const auto start = Clock::now();
        while (ids.size() < count) {
            auto txn = stream.begin(); df::DeviceTimings timing;
            const auto admitted = df::admit_window(k, txn->frontier(), count - ids.size(), true, false, 0, k == 0);
            const auto cycle = df::run_device_cycle(target, *txn, device, anchor, admitted, count - ids.size(), {}, &acceptance, &timing);
            ids.insert(ids.end(), cycle.verdict.emitted.begin(), cycle.verdict.emitted.end()); anchor = cycle.next_anchor;
            total.block_ms += timing.block_ms; total.head_ms += timing.head_ms; total.verify_ms += timing.verify_ms;
            total.append_ms += timing.append_ms; total.commit_ms += timing.commit_ms;
            require(std::equal(ids.begin(), ids.end(), serial.begin()), "DFlash/serial token mismatch");
        }
        const auto elapsed = ms(start); const auto digest = stream.digest();
        require(digest == serial_digest, "DFlash/serial committed-state mismatch");
        Json receipt = Json::object(); receipt.set("event", "cycle_gate"); receipt.set("verdict", "EXACT");
        receipt.set("fixture", fixture.stem().string()); receipt.set("k", std::uint64_t(k)); receipt.set("auto", k == 0);
        receipt.set("count", std::uint64_t(ids.size())); receipt.set("tokens", array<unsigned>(ids)); receipt.set("digest", digest);
        receipt.set("cycles", acceptance.cycles); receipt.set("attempted", acceptance.attempted); receipt.set("eligible", acceptance.eligible);
        receipt.set("accepted", acceptance.accepted); receipt.set("emitted", acceptance.emitted); receipt.set("all_accepted", acceptance.all_accepted);
        receipt.set("depth_eligible", array<std::uint64_t>(acceptance.depth_eligible));
        receipt.set("depth_accepted", array<std::uint64_t>(acceptance.depth_accepted));
        receipt.set("rejection_depth", array<std::uint64_t>(acceptance.rejection_depth));
        receipt.set("wall_ms", elapsed); receipt.set("serial_ms", plain_ms);
        receipt.set("block_ms", total.block_ms); receipt.set("head_ms", total.head_ms);
        receipt.set("verify_ms", total.verify_ms); receipt.set("append_ms", total.append_ms); receipt.set("commit_ms", total.commit_ms);
        std::cout << receipt.dump() << std::endl;
    }
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "MODEL.gguf TARGET_ROPE_DIR DRAFT.gguf DRAFT_ROPE.g4dr PROMPT.npz TOKENS K(auto/all/0..7) [--bytes]\n";
            return 0;
        }
        require(argc == 8 || (argc == 9 && std::string(argv[8]) == "--bytes"), "use --help for arguments");
        g4::gpu::require_device();
        const auto model = g4::load_gguf_model(argv[1]); const auto hw = g4::bind_host_weights(model);
        const auto target_rope = g4::load_rope_tables(argv[2]); const auto draft = df::DraftFile::open(argv[3]);
        const auto rope = df::RopeFile::open(argv[4]);
        Resident assets(hw.embedding); g4::gpu::DeviceTarget target(hw, target_rope.view);
        Json identity = Json::object(); identity.set("event", "identity"); identity.set("target", g4::arithmetic_id);
        identity.set("draft", df::arithmetic_id); identity.set("model_hash", model.model_hash);
        identity.set("target_rope_hash", target_rope.view.hash); identity.set("draft_rope_layout", "interleaved-128-pairs-v1");
        identity.set("draft_cosine_hash", g4::hash_bytes(rope.cosine.data(), rope.cosine.size() * 4));
        identity.set("draft_sine_hash", g4::hash_bytes(rope.sine.data(), rope.sine.size() * 4));
        std::cout << identity.dump() << std::endl;
        const auto count = unsigned(std::stoul(argv[6])); require(count > 0 && count <= 4096, "token count 1..4096");
        run(target, hw, draft.weights(), rope.view(), assets.view, argv[5], count, argc == 9, argv[7]);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
