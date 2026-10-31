#include "target.hpp"
#include "npz.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace g4 = strix::gemma4;
namespace {
std::uint32_t argmax(std::span<const float> x) {
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < x.size(); ++i) if (g4::better(x[i], i, x[best], best)) best = i;
    return best;
}
double lse(std::span<const float> x) {
    const double m = *std::max_element(x.begin(), x.end());
    double sum = 0;
    for (float v : x) sum += std::exp(double(v) - m);
    return m + std::log(sum);
}
struct Metrics {
    std::size_t rows = 0, agreed = 0, greedy_agreed = 0;
    std::vector<double> lower, upper;
    double max_abs = 0, rmse = 0, full_kl = 0;
    std::vector<std::uint32_t> greedy;
    strix::serve::Json disagreements = strix::serve::Json::array();
    std::optional<std::size_t> first_greedy_difference;
};
void compare(Metrics& m, std::span<const float> ours, std::span<const std::int32_t> ids,
             std::span<const float> hf, double hf_lse, std::optional<std::uint32_t> greedy_reference) {
    if (ids.empty() || ids.size() != hf.size()) throw std::runtime_error("invalid top-K fixture");
    const auto top = std::size_t(std::max_element(hf.begin(), hf.end()) - hf.begin());
    std::uint32_t expected = std::uint32_t(ids[top]);
    for (std::size_t k = 0; k < hf.size(); ++k) {
        if (ids[k] < 0 || std::size_t(ids[k]) >= ours.size()) throw std::runtime_error("top-K id outside vocabulary");
        if (hf[k] == hf[top]) expected = std::min(expected, std::uint32_t(ids[k]));
    }
    if (greedy_reference) expected = *greedy_reference;
    const auto actual = argmax(ours);
    if (actual != expected) {
        strix::serve::Json difference = strix::serve::Json::object();
        difference.set("row",std::uint64_t(m.rows)); difference.set("hf_token",std::uint64_t(expected));
        difference.set("engine_token",std::uint64_t(actual));
        difference.set("engine_margin",double(ours[actual])-ours[expected]);
        for (std::size_t i = 0; i < ids.size(); ++i) if (std::uint32_t(ids[i]) == actual) difference.set("hf_margin",double(hf[top])-hf[i]);
        m.disagreements.push(std::move(difference));
    }
    ++m.rows; m.agreed += actual == expected;
    const double ours_lse = lse(ours);
    double term = 0, mass_p = 0, mass_q = 0;
    std::vector<bool> seen(ours.size());
    for (std::size_t k = 0; k < ids.size(); ++k) {
        if (seen[std::size_t(ids[k])]) throw std::runtime_error("duplicate top-K fixture token");
        seen[std::size_t(ids[k])] = true;
        const double logp = double(hf[k]) - hf_lse, logq = double(ours[std::size_t(ids[k])]) - ours_lse;
        const double p = std::exp(logp);
        mass_p += p; mass_q += std::exp(logq); term += p * (logp - logq);
    }
    if (mass_p > 1.000001 || mass_q > 1.000001) throw std::runtime_error("inconsistent top-K probability mass");
    const double tail_p = std::max(0.0, 1.0 - mass_p), tail_q = std::max(1e-300, 1.0 - mass_q);
    double min_tail = 30;
    for (std::size_t i = 0; i < ours.size(); ++i) if (!seen[i]) min_tail = std::min(min_tail, double(ours[i]));
    m.lower.push_back(std::max(0.0, term + (tail_p ? tail_p * std::log(tail_p / tail_q) : 0.0)));
    m.upper.push_back(std::max(0.0, term + (tail_p ? tail_p * (std::log(tail_p) - min_tail + ours_lse) : 0.0)));
}
void run(g4::HostTarget& target, const g4::ModelConfig& config, const std::filesystem::path& fixture, std::size_t generate) {
    const auto arrays = g4::read_npz(fixture);
    const auto prompt = arrays.at("prompt_ids").values<std::int32_t>("<i4");
    const auto gold = arrays.at("gen_ids").values<std::int32_t>("<i4");
    const auto top_ids = arrays.at("topk_ids").values<std::int32_t>("<i4");
    const auto top_vals = arrays.at("topk_vals").values<float>("<f4");
    const auto hf_lse = arrays.at("lse").values<double>("<f8");
    const auto last = arrays.at("last_logits").values<float>("<f4");
    const auto& shape = arrays.at("topk_ids").shape;
    if (prompt.empty() || shape.size() != 2 || shape[0] != gold.size() + 1 || shape != arrays.at("topk_vals").shape || hf_lse.size() != shape[0] || last.size() != config.vocab) throw std::runtime_error("fixture dimensions inconsistent");
    if (arrays.at("prompt_ids").shape.size() != 1 || arrays.at("gen_ids").shape.size() != 1 ||
        arrays.at("lse").shape.size() != 1 || arrays.at("last_logits").shape.size() != 1) throw std::runtime_error("fixture vectors must be one-dimensional");
    for (const auto& tokens : {std::span<const std::int32_t>(prompt),std::span<const std::int32_t>(gold)})
        for (const auto token : tokens) if (token < 0 || std::uint32_t(token) >= config.vocab) throw std::runtime_error("fixture token outside vocabulary");
    generate = std::min(generate, gold.size());
    if (prompt.size() + generate > config.max_positions) throw std::runtime_error("fixture exceeds context limit");
    const auto k = shape[1];
    g4::HostStream teacher(config), greedy(config);
    Metrics metrics;
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> teacher_logits, greedy_logits;
    // Two independent streams share only weight reads. Chunk arithmetic is unchanged.
    for (std::size_t begin = 0; begin < prompt.size(); begin += 4) {
        const auto count = std::min<std::size_t>(4, prompt.size() - begin);
        auto tt = teacher.begin(), gt = greedy.begin();
        std::vector<g4::RowPlan> rows;
        for (std::size_t r = 0; r < count; ++r) {
            if (prompt[begin+r] < 0) throw std::runtime_error("negative prompt token");
            rows.push_back({0,0,begin+r,std::uint32_t(prompt[begin+r]),tt.get(),r,true});
            rows.push_back({1,0,begin+r,std::uint32_t(prompt[begin+r]),gt.get(),count+r,true});
        }
        const auto out = target.forward_rows(rows); tt->commit(count); gt->commit(count);
        teacher_logits.assign(out.logits.begin() + std::ptrdiff_t((count-1)*config.vocab), out.logits.begin() + std::ptrdiff_t(count*config.vocab));
        greedy_logits.assign(out.logits.begin() + std::ptrdiff_t((2*count-1)*config.vocab), out.logits.end());
    }
    const double ours_lse = lse(teacher_logits);
    double squares = 0;
    const double full_hf_lse = lse(last);
    for (std::size_t i = 0; i < last.size(); ++i) {
        const double delta = double(teacher_logits[i]) - last[i];
        metrics.max_abs = std::max(metrics.max_abs, std::abs(delta)); squares += delta * delta;
        const double logp = double(last[i]) - full_hf_lse;
        metrics.full_kl += std::exp(logp) * (logp - double(teacher_logits[i]) + ours_lse);
    }
    metrics.rmse = std::sqrt(squares / double(last.size()));
    for (std::size_t step = 0; step <= generate; ++step) {
        const auto reference = step < gold.size() ? std::optional<std::uint32_t>(std::uint32_t(gold[step])) : std::nullopt;
        compare(metrics, teacher_logits, {top_ids.data()+step*k,k}, {top_vals.data()+step*k,k}, hf_lse[step],reference);
        if (step == generate) break;
        const auto next = argmax(greedy_logits);
        if (next != std::uint32_t(gold[step]) && !metrics.first_greedy_difference) metrics.first_greedy_difference = step;
        metrics.greedy.push_back(next); metrics.greedy_agreed += next == std::uint32_t(gold[step]);
        auto tt = teacher.begin(), gt = greedy.begin();
        const auto p = teacher.frontier();
        const g4::RowPlan rows[] = {{0,0,p,std::uint32_t(gold[step]),tt.get(),0,true}, {1,0,p,next,gt.get(),1,true}};
        const auto out = target.forward_rows(rows); tt->commit(1); gt->commit(1);
        teacher_logits.assign(out.logits.begin(), out.logits.begin()+config.vocab);
        greedy_logits.assign(out.logits.begin()+config.vocab,out.logits.end());
        if ((step+1)%8 == 0) std::cerr << fixture.stem().string() << " processed " << step+1 << '/' << generate << '\n';
    }
    const auto mean = [](const auto& v) { return std::accumulate(v.begin(),v.end(),0.0)/double(v.size()); };
    auto sorted = metrics.lower; std::sort(sorted.begin(), sorted.end());
    strix::serve::Json result = strix::serve::Json::object();
    result.set("fixture", fixture.stem().string()); result.set("rows", std::uint64_t(metrics.rows));
    result.set("top1_agreement", double(metrics.agreed)/double(metrics.rows));
    result.set("kl_lower_mean", mean(metrics.lower)); result.set("kl_lower_p99", sorted[std::min(sorted.size()-1,std::size_t(std::ceil(0.99*double(sorted.size())))-1)]);
    result.set("kl_upper_mean", mean(metrics.upper)); result.set("last_logit_max_abs", metrics.max_abs);
    result.set("last_logit_rmse", metrics.rmse); result.set("last_full_kl", metrics.full_kl);
    result.set("greedy_equal", metrics.greedy_agreed == generate); result.set("greedy_matching_tokens", std::uint64_t(metrics.greedy_agreed));
    result.set("generated", std::uint64_t(generate));
    result.set("teacher_disagreements",std::move(metrics.disagreements));
    if (metrics.first_greedy_difference) result.set("first_greedy_difference",std::uint64_t(*metrics.first_greedy_difference));
    strix::serve::Json ids = strix::serve::Json::array();
    for (const auto token : metrics.greedy) ids.push(std::uint64_t(token));
    result.set("greedy_ids", std::move(ids));
    result.set("seconds", std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    result.set("status", "FIDELITY-OBSERVATION; top-K KL bounds are not full-vocabulary qualification");
    std::cout << result.dump() << std::endl;
}
}
int main(int argc, char** argv) {
    try {
        std::filesystem::path model_dir, gguf, tables;
        std::vector<std::filesystem::path> fixtures;
        std::uint32_t threads = 16; std::size_t generate = 64; bool inspect = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("missing option value"); return argv[i]; };
            if (arg == "--q4km") gguf = value();
            else if (arg == "--rope-tables") tables = value();
            else if (arg == "--threads") threads = std::uint32_t(std::stoul(value()));
            else if (arg == "--generate") generate = std::stoull(value());
            else if (arg == "--inspect-model") inspect = true;
            else if (arg == "--help") {
                std::cout << "strix-gemma4-host-ref MODEL_DIR --rope-tables TABLE_DIR FIXTURE.npz... [--q4km MODEL.gguf] [--generate 64] [--threads 16]\n"
                    "--inspect-model loads/validates weights without executing; no GPU is used.\n"; return 0;
            } else if (arg.starts_with("--")) throw std::runtime_error("unknown option: " + arg);
            else if (model_dir.empty()) model_dir = arg;
            else fixtures.emplace_back(arg);
        }
        if (!threads || threads > 16 || model_dir.empty() || (!inspect && (tables.empty() || fixtures.empty()))) throw std::runtime_error("use --help for required model, tables and fixtures");
        const auto begin = std::chrono::steady_clock::now();
        auto model = gguf.empty() ? g4::load_safetensors_model(model_dir) : g4::load_gguf_model(gguf);
        std::cerr << "model identity=" << model.identity << " hash=" << model.model_hash << " tensors=" << model.tensors.size() << " load_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count() << '\n';
        for (const auto& ignored : model.ignored_tensors) std::cerr << "ignored tensor: " << ignored << '\n';
        if (inspect) return 0;
        const auto rope = g4::load_rope_tables(tables); model.rope_hash = rope.view.hash;
        std::cerr << "rotary hash=" << model.rope_hash << " cache=F32 route=" << (gguf.empty() ? "BF16" : "Q4_K_M per-tensor") << '\n';
        g4::HostTarget target(g4::bind_host_weights(model), rope.view, threads);
        for (const auto& fixture : fixtures) run(target, model.config, fixture, generate);
        return 0;
    } catch (const std::exception& e) { std::cerr << "gemma4 host reference: " << e.what() << '\n'; return 1; }
}
