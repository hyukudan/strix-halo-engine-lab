// Explicit full-model diagnostic. Never registered with ctest.
#include "dflash.hpp"
#include "gemma4/ref/npz.hpp"
#include "gemma4/ref/target.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace g4 = strix::gemma4;
namespace df = g4::dflash;
using Clock = std::chrono::steady_clock;
using Json = strix::serve::Json;

namespace {
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::uint32_t greedy(std::span<const float> logits) {
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < logits.size(); ++i)
        if (g4::better(logits[i], i, logits[best], best)) best = i;
    return best;
}
template<class T> void dump(const std::filesystem::path& path, std::span<const T> values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
    if (!out) throw std::runtime_error("cannot write diagnostic output");
}
template<class T> Json array(std::span<const T> values) {
    Json out = Json::array();
    for (auto value : values) out.push(Json(std::uint64_t(value)));
    return out;
}
struct Draft {
    std::size_t offset;
    std::array<std::uint32_t, df::max_drafts> ids;
    double block_ms, head_ms;
    df::ForwardDiagnostics diagnostics;
};

Json range(std::span<const float> values) {
    if (values.empty()) throw std::runtime_error("empty diagnostic tensor");
    const auto limits = std::minmax_element(values.begin(), values.end());
    double square = 0;
    for (float value : values) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite diagnostic tensor");
        square += double(value) * double(value);
    }
    Json out = Json::object();
    out.set("minimum", double(*limits.first)); out.set("maximum", double(*limits.second));
    out.set("rms", std::sqrt(square / double(values.size())));
    return out;
}

void run(g4::HostTarget& target, const g4::HostWeights& target_weights, const df::Weights& weights,
         df::RopeView rope, const std::filesystem::path& fixture, const std::filesystem::path& output,
         std::size_t last_offset) {
    const auto arrays = g4::read_npz(fixture);
    const auto prompt = arrays.at("prompt_ids").values<std::int32_t>("<i4");
    if (prompt.empty() || prompt.size() + last_offset + df::block > rope.positions)
        throw std::runtime_error("prompt or rotary reservation invalid");
    for (auto token : prompt)
        if (token < 0 || std::uint32_t(token) >= target_weights.config.vocab)
            throw std::runtime_error("invalid prompt token");
    const auto dir = output / fixture.stem();
    std::filesystem::create_directories(dir);
    dump<std::int32_t>(dir / "prompt.i32", prompt);
    g4::HostStream stream(target_weights.config);
    df::DraftContext context;
    std::vector<float> taps;
    double context_ms = 0;
    auto append = [&](const g4::ForwardResult& result, std::uint64_t start) {
        std::vector<std::uint64_t> positions(result.rows);
        std::iota(positions.begin(), positions.end(), start);
        const auto begin = Clock::now();
        context.append(weights, rope, result.taps, positions);
        context_ms += milliseconds(begin);
        taps.insert(taps.end(), result.taps.begin(), result.taps.end());
    };
    const auto prefill_start = Clock::now();
    std::uint32_t anchor = 0;
    for (std::size_t b = 0; b < prompt.size(); b += df::block) {
        const auto n = std::min<std::size_t>(df::block, prompt.size() - b);
        auto tx = stream.begin();
        std::vector<g4::RowPlan> rows;
        for (std::size_t r = 0; r < n; ++r)
            rows.push_back({0, 0, b + r, std::uint32_t(prompt[b + r]), tx.get(), r, true});
        const auto result = target.forward_rows(rows);
        tx->commit(n);
        append(result, b);
        anchor = greedy(std::span<const float>(result.logits).last(result.vocab));
    }
    const double prefill_ms = milliseconds(prefill_start);
    std::vector<std::uint32_t> serial{anchor};
    std::vector<Draft> drafts;
    const auto decode_start = Clock::now();
    // Observe the first window and an optional later serial prefix, always with a full block.
    for (std::size_t step = 0; step < last_offset + df::max_drafts; ++step) {
        if (step == 0 || step == last_offset) {
            const auto embedded = df::embed_block(anchor, target_weights.embedding);
            const auto block_start = Clock::now();
            df::ForwardDiagnostics diagnostics;
            const auto hidden = df::block_forward(weights, context, rope, embedded, stream.frontier(), &diagnostics);
            const double block_ms = milliseconds(block_start);
            const auto head_start = Clock::now();
            const auto ids = df::proposals_from_hidden(weights, hidden, target_weights.embedding);
            const double head_ms = milliseconds(head_start);
            drafts.push_back({step, ids, block_ms, head_ms, std::move(diagnostics)});
            const auto prefix = dir / ("window-" + std::to_string(step));
            dump<float>(prefix.string() + "-taps.f32", taps);
            dump<float>(prefix.string() + "-embedded.f32", embedded);
            dump<float>(prefix.string() + "-hidden.f32", hidden);
            std::cerr << fixture.stem().string() << " offset=" << step << " block_ms=" << block_ms
                      << " head_ms=" << head_ms << '\n';
        }
        auto tx = stream.begin();
        const std::uint64_t position = stream.frontier();
        const g4::RowPlan row{0, 0, position, anchor, tx.get(), 0, true};
        const auto result = target.forward_rows({&row, 1});
        tx->commit(1);
        append(result, position);
        anchor = greedy(result.logits);
        serial.push_back(anchor);
    }
    Json receipt = Json::object();
    receipt.set("fixture", fixture.stem().string());
    receipt.set("target_identity", g4::arithmetic_id);
    receipt.set("draft_identity", df::arithmetic_id);
    receipt.set("prompt_rows", std::uint64_t(prompt.size()));
    receipt.set("serial_ids", array<std::uint32_t>(serial));
    dump<std::uint32_t>(dir / "serial.u32", serial);
    if (const auto found = arrays.find("gen_ids"); found != arrays.end()) {
        const auto expected = found->second.values<std::int32_t>("<i4");
        const auto n = std::min(expected.size(), serial.size());
        bool equal = n > 0;
        for (std::size_t i = 0; i < n; ++i) equal = equal && std::int64_t(serial[i]) == expected[i];
        receipt.set("hf_prefix_rows", std::uint64_t(n)); receipt.set("hf_prefix_equal", equal);
    }
    receipt.set("prefill_with_context_ms", prefill_ms);
    receipt.set("decode_with_draft_context_ms", milliseconds(decode_start));
    receipt.set("context_ms", context_ms);
    receipt.set("state_digest", stream.digest());
    receipt.set("tap_range", range(taps));
    Json windows = Json::array();
    for (const auto& draft : drafts) {
        Json window = Json::object();
        window.set("offset", std::uint64_t(draft.offset));
        window.set("position", std::uint64_t(prompt.size() + draft.offset));
        window.set("proposals", array<std::uint32_t>(draft.ids));
        window.set("block_ms", draft.block_ms); window.set("head_ms", draft.head_ms);
        Json gates = Json::array();
        for (const auto& gate : draft.diagnostics.gates) {
            Json entry = Json::object();
            entry.set("minimum", double(gate.minimum)); entry.set("maximum", double(gate.maximum));
            entry.set("negative_saturated", std::uint64_t(gate.negative_saturated));
            gates.push(std::move(entry));
        }
        window.set("gates", std::move(gates));
        Json by_k = Json::array();
        std::size_t prefix = 0;
        while (prefix < df::max_drafts && draft.ids[prefix] == serial[draft.offset + prefix + 1]) ++prefix;
        for (std::uint32_t k = 1; k <= df::max_drafts; ++k) {
            // Serial continuation is a reference only until the first rejected proposal.
            // Later hypothetical verify rows are not evaluated by this diagnostic.
            const auto accepted = std::min<std::size_t>(prefix, k);
            Json result = Json::object();
            result.set("k", std::uint64_t(k)); result.set("accepted", std::uint64_t(accepted));
            result.set("eligible", std::uint64_t(std::min<std::size_t>(k, accepted + 1)));
            result.set("emitted", std::uint64_t(accepted + 1));
            by_k.push(std::move(result));
        }
        window.set("prefix_accepted", std::uint64_t(prefix));
        window.set("by_k", std::move(by_k)); windows.push(std::move(window));
    }
    receipt.set("windows", std::move(windows));
    receipt.set("qualification", "short host diagnostic; no device or throughput qualification");
    std::ofstream file(dir / "receipt.json"); file << receipt.dump() << '\n';
    if (!file) throw std::runtime_error("cannot write receipt");
    std::cout << receipt.dump() << std::endl;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "strix-gemma4-dflash-reference MODEL_DIR TARGET_TABLE_DIR DRAFT.gguf DRAFT.g4dr OUTPUT_DIR LAST_OFFSET FIXTURE.npz...\n"
                      << "Loads the full target on the host. Use only on an admitted qualification machine.\n"
                      << "Dumps private taps/embeddings/hidden rows and serial-greedy acceptance; not a ctest.\n";
            return 0;
        }
        if (argc < 8) throw std::runtime_error("use --help for required arguments");
        const auto offset = std::stoull(argv[6]);
        if (offset > 64) throw std::runtime_error("diagnostic offset exceeds 64");
        const auto model = g4::load_safetensors_model(argv[1]);
        const auto tables = g4::load_rope_tables(argv[2]);
        const auto weights = g4::bind_host_weights(model);
        g4::HostTarget target(weights, tables.view);
        const auto drafter = df::DraftFile::open(argv[3]);
        const auto rope = df::RopeFile::open(argv[4]);
        Json identity = Json::object();
        identity.set("model_hash", model.model_hash); identity.set("target_rope_hash", tables.view.hash);
        const g4::MappedFile draft_bytes(argv[3]), rope_bytes(argv[4]);
        identity.set("draft_hash", g4::hash_bytes(draft_bytes.data(), draft_bytes.size()));
        identity.set("draft_rope_hash", g4::hash_bytes(rope_bytes.data(), rope_bytes.size()));
        std::cout << identity.dump() << std::endl;
        for (int i = 7; i < argc; ++i) run(target, weights, drafter.weights(), rope.view(), argv[i], argv[5], offset);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DFlash reference: " << error.what() << '\n'; return 1;
    }
}
