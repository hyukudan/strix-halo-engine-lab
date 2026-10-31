#include "pack.hpp"
#include "format.hpp"
#include "gemma4/kernels/device.hpp"
#include "gemma4/prefill/device.hpp"
#include "gemma4/ref/npz.hpp"
#include "gemma4/dflash/device.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace g4 = strix::gemma4;
namespace gpu = g4::gpu;
namespace ex = g4::exl3;
using Json = strix::serve::Json;
using Clock = std::chrono::steady_clock;
namespace {
struct Args {
    std::filesystem::path model, bf16, rope, teacher, draft, draft_rope, q4;
    std::string mode = "gate";
    unsigned steps = 12, skip = 0, limit = 40;
    bool host = false;
};
Args arguments(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check-host") { a.host = true; continue; }
        if (++i == argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (key == "--model") a.model = value;
        else if (key == "--bf16") a.bf16 = value;
        else if (key == "--rope") a.rope = value;
        else if (key == "--mode") a.mode = value;
        else if (key == "--teacher") a.teacher = value;
        else if (key == "--draft") a.draft = value;
        else if (key == "--draft-rope") a.draft_rope = value;
        else if (key == "--q4") a.q4 = value;
        else if (key == "--steps" || key == "--skip" || key == "--limit") {
            std::size_t end = 0; const auto n = std::stoul(value, &end);
            if (end != value.size() || n > 65536) throw std::invalid_argument("bounded integer required");
            if (key == "--steps") a.steps = unsigned(n);
            else if (key == "--skip") a.skip = unsigned(n);
            else a.limit = unsigned(n);
        } else throw std::invalid_argument("unknown option: " + key);
    }
    if (a.model.empty() || a.rope.empty() || !a.steps || !a.limit || a.steps > 1024 ||
        (a.mode != "gate" && a.mode != "speed" && a.mode != "speed-all" && a.mode != "quality" && a.mode != "quality-decode")) throw std::invalid_argument("invalid check arguments");
    return a;
}
void print(Json value) { std::cout << value.dump() << std::endl; }
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
unsigned greedy(std::span<const float> x) {
    unsigned best = 0;
    for (unsigned i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i])) throw std::runtime_error("nonfinite capped logit");
        if (g4::better(x[i], i, x[best], best)) best = i;
    }
    return best;
}
void same(std::span<const float> a, std::span<const float> b, const char* name) {
    if (a.size() != b.size()) throw std::runtime_error(std::string(name) + " shape mismatch");
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(a.data() + i, b.data() + i, 4)) throw std::runtime_error(std::string(name) + " byte mismatch at " + std::to_string(i));
}
void gate(const g4::HostWeights& weights, g4::RopeTables rope, const Args& args) {
    gpu::DeviceTarget target(weights, rope);
    const auto& c = weights.config;
    gpu::DeviceStream serial(c, 64);
    std::vector<g4::ForwardResult> expected;
    std::vector<std::string> digest;
    for (unsigned r = 0; r < 8; ++r) {
        auto tx = serial.begin(); g4::RowPlan row{0, 0, r, r == 0 ? 2U : 100U + r, tx.get(), 0, true};
        expected.push_back(target.forward_rows({&row, 1})); tx->commit(1); digest.push_back(serial.digest());
    }
    if (args.host) {
        g4::HostTarget host(weights, rope, 16); g4::HostStream stream(c);
        for (unsigned r = 0; r < 2; ++r) {
            auto tx = stream.begin(); g4::RowPlan row{0, 0, r, r == 0 ? 2U : 100U + r, tx.get(), 0, true};
            const auto out = host.forward_rows({&row, 1});
            same(out.logits, expected[r].logits, "full-model host/device logits");
            same(out.taps, expected[r].taps, "full-model host/device taps");
            tx->commit(1);
            if (stream.digest() != digest[r]) throw std::runtime_error("host/device KV digest");
        }
        print(Json::Object{{"gate", "full-model-host-device"}, {"verdict", "EXACT"}, {"rows", 2}});
    }
    for (unsigned count = 1; count <= 8; ++count) {
        gpu::DeviceStream grouped(c, 64); auto tx = grouped.begin(); std::vector<g4::RowPlan> rows;
        for (unsigned r = 0; r < count; ++r) rows.push_back({0, 0, r, r == 0 ? 2U : 100U + r, tx.get(), r, true});
        const auto out = target.forward_rows(rows);
        for (unsigned r = 0; r < count; ++r) {
            same(std::span(out.logits).subspan(std::size_t(r) * c.vocab, c.vocab), expected[r].logits, "R/serial logits");
            same(std::span(out.taps).subspan(std::size_t(r) * 5 * c.hidden, 5 * c.hidden), expected[r].taps, "R/serial taps");
        }
        tx->commit(count);
        if (grouped.digest() != digest[count - 1]) throw std::runtime_error("R/serial state digest");
    }
    print(Json::Object{{"gate", "R1..8-serial-logits-taps-KV"}, {"verdict", "EXACT"}});
    std::vector<std::unique_ptr<gpu::DeviceStream>> streams, solos;
    for (unsigned s = 0; s < 4; ++s) { streams.push_back(std::make_unique<gpu::DeviceStream>(c, 64)); solos.push_back(std::make_unique<gpu::DeviceStream>(c, 64)); }
    for (unsigned s = 0; s < 4; ++s)
        for (unsigned p = 0; p < s; ++p)
            for (auto* stream : {streams[s].get(), solos[s].get()}) {
                auto tx = stream->begin(); g4::RowPlan row{s, 0, p, 2, tx.get(), 0, true};
                (void)target.forward_rows({&row, 1}); tx->commit(1);
            }
    for (unsigned step = 0; step < 2; ++step) {
        std::vector<std::unique_ptr<g4::KvTxn>> tx; std::vector<g4::RowPlan> rows;
        for (unsigned s = 0; s < 4; ++s) {
            tx.push_back(streams[s]->begin()); rows.push_back({s, 0, s + step, 2 + s + step, tx.back().get(), 3 - s, true});
        }
        const auto grouped = target.forward_rows(rows);
        for (unsigned s = 0; s < 4; ++s) {
            auto st = solos[s]->begin(); auto row = rows[s]; row.transaction = st.get(); row.output_row = 0;
            const auto solo = target.forward_rows({&row, 1});
            same(std::span(grouped.logits).subspan(std::size_t(3 - s) * c.vocab, c.vocab), solo.logits, "C4/C1 logits");
            same(std::span(grouped.taps).subspan(std::size_t(3 - s) * 5 * c.hidden, 5 * c.hidden), solo.taps, "C4/C1 taps");
            st->commit(1); tx[s]->commit(1);
            if (streams[s]->digest() != solos[s]->digest()) throw std::runtime_error("C4/C1 state digest");
        }
    }
    print(Json::Object{{"gate", "C4-C1-permuted-logits-taps-KV"}, {"verdict", "EXACT"}});
    if (!args.draft.empty()) {
        const auto file = g4::dflash::DraftFile::open(args.draft);
        const auto dr = g4::dflash::RopeFile::open(args.draft_rope);
        auto dm = std::make_shared<g4::dflash::DraftModel>(file.weights(), dr.view(), target.resident_embedding(), target.resident_head());
        for (unsigned k : {1U, 7U}) {
            gpu::DeviceStream spec(c, 64), plain(c, 64); g4::dflash::DeviceDrafter drafter(dm);
            unsigned anchor = 2, solo_anchor = 2, emitted = 0;
            while (emitted < 8) {
                auto tx = spec.begin();
                const auto result = g4::dflash::run_device_cycle(target, *tx, drafter, anchor, k, 8 - emitted);
                for (auto token : result.verdict.emitted) {
                    auto st = plain.begin(); g4::RowPlan row{0, 0, st->frontier(), solo_anchor, st.get(), 0, true};
                    const auto out = target.forward_rows({&row, 1});
                    if (token != greedy(out.logits)) throw std::runtime_error("DFlash token mismatch");
                    st->commit(1); solo_anchor = token;
                }
                anchor = result.next_anchor; emitted += unsigned(result.verdict.emitted.size());
                if (spec.digest() != plain.digest()) throw std::runtime_error("DFlash state digest");
            }
        }
        print(Json::Object{{"gate", "DFlash-k1-k7-off-tokens-KV"}, {"verdict", "EXACT"}, {"tokens_per_k", 8}});
    }
}
void speed(gpu::DeviceTarget& target, g4::RopeTables rope, const Args& args,
           const std::string& label = "single", unsigned round = 0, unsigned prefill_samples = 3) {
    const auto& w = target.borrow_device_weights();
    for (unsigned count : {1U, 4U, 8U}) {
        std::vector<std::unique_ptr<gpu::DeviceStream>> streams;
        for (unsigned r = 0; r < count; ++r) streams.push_back(std::make_unique<gpu::DeviceStream>(w.config, 2048));
        std::vector<unsigned> tokens(count, 2); std::vector<double> times;
        for (unsigned step = 0; step < args.steps + 2; ++step) {
            std::vector<std::unique_ptr<g4::KvTxn>> tx; std::vector<g4::RowPlan> rows;
            for (unsigned r = 0; r < count; ++r) {
                tx.push_back(streams[r]->begin()); rows.push_back({r, 0, step, tokens[r], tx.back().get(), r, true});
            }
            const auto t = Clock::now(); const auto out = target.forward_rows(rows);
            for (unsigned r = 0; r < count; ++r) tx[r]->commit(1);
            for (unsigned r = 0; r < count; ++r) tokens[r] = greedy(std::span(out.logits).subspan(std::size_t(r) * w.config.vocab, w.config.vocab));
            if (step >= 2) times.push_back(elapsed(t));
        }
        std::sort(times.begin(), times.end());
        const auto n = times.size(); const auto median = (times[(n - 1) / 2] + times[n / 2]) / 2;
        print(Json::Object{{"metric", "decode"}, {"arm", label}, {"round", int(round)}, {"rows", int(count)}, {"tok_s", count / median}, {"median_pass_s", median}, {"steps", int(args.steps)}});
    }
    const auto prepare = Clock::now();
    g4::prefill::DevicePrefill prompt(w, rope, g4::prefill::Residency::device);
    print(Json::Object{{"metric", "panel-setup"}, {"arm", label}, {"round", int(round)}, {"seconds", elapsed(prepare)}});
    for (unsigned length : {512U, 2048U}) {
        std::vector<double> samples;
        std::vector<unsigned> tokens(length, 100); tokens[0] = 2;
        for (unsigned trial = 0; trial < prefill_samples; ++trial) {
            gpu::DeviceStream stream(w.config, 4096);
            const auto t = Clock::now(); (void)prompt.run(stream, tokens); samples.push_back(elapsed(t));
        }
        std::sort(samples.begin(), samples.end());
        const auto median = samples[samples.size() / 2];
        print(Json::Object{{"metric", "prefill"}, {"arm", label}, {"round", int(round)}, {"length", int(length)}, {"tok_s", length / median}, {"median_s", median}, {"samples", int(prefill_samples)}});
    }
}
void speed_all(const Args& args, g4::RopeTables rope) {
    if (args.q4.empty() || args.bf16.empty()) throw std::invalid_argument("speed-all requires --q4 and --bf16; model is the arm root");
    auto plan = Json::object(); plan.set("metric", "benchmark-plan"); plan.set("rounds", 3);
    plan.set("attention", "canonical"); plan.set("qualified", false);
    auto switches = Json::object();
    for (const auto* name : {"STRIX_G4_SHARED_WEIGHTS", "STRIX_G4_GRAPHS", "STRIX_G4_FUSION", "STRIX_G4_EXL3_OUTPUT_TILE", "STRIX_G4_EXL3_COLUMN_CACHE", "STRIX_G4_PREFILL_PREPARE"}) {
        const auto* value = std::getenv(name); switches.set(name, !value || std::string(value) != "0");
    }
    plan.set("off_switches", switches);
#if defined(__linux__)
    utsname info{}; if (uname(&info) == 0) plan.set("kernel", info.release);
    std::ifstream parameter("/sys/module/amdgpu/parameters/iommu_perfopt"); std::string perfopt;
    plan.set("iommu_perfopt", parameter >> perfopt ? perfopt : "not-present");
#endif
    print(plan);
    std::vector<std::unique_ptr<gpu::DeviceTarget>> targets;
    const std::array<std::string, 5> labels{"Q4", "M4", "M5", "M6", "U5"};
    for (const auto& label : labels) {
        const auto start = Clock::now();
        auto model = label == "Q4" ? g4::load_gguf_model(args.q4) : ex::load_arm(args.model / label, args.bf16);
        targets.push_back(std::make_unique<gpu::DeviceTarget>(g4::bind_host_weights(model), rope));
        print(Json::Object{{"metric", "resident-arm"}, {"arm", label}, {"identity", model.identity}, {"model_hash", model.model_hash},
            {"rope_hash", rope.hash}, {"setup_s", elapsed(start)}, {"resident_bytes", std::uint64_t(targets.back()->stats().resident_weight_bytes)}});
        // Construction owns packed bytes; mappings close here, keeping host
        // memory bounded. Only one arm's FP16 panels exists during each trial.
    }
    for (unsigned round = 0; round < 3; ++round)
        for (unsigned step = 0; step < labels.size(); ++step) {
            const auto trial = round * unsigned(labels.size()) + step;
            if (trial < args.skip) continue;
            const auto index = round % 2 ? unsigned(labels.size()) - 1 - step : step;
            print(Json::Object{{"metric", "trial-start"}, {"case", int(trial)}, {"arm", labels[index]}, {"round", int(round)}});
            speed(*targets[index], rope, args, labels[index], round, 1);
            print(Json::Object{{"metric", "trial-complete"}, {"case", int(trial)}, {"arm", labels[index]}, {"round", int(round)}});
        }
}
double lse(std::span<const float> values) {
    const auto m = double(*std::max_element(values.begin(), values.end())); double sum = 0;
    for (float x : values) { if (!std::isfinite(x)) throw std::runtime_error("nonfinite quality logit"); sum += std::exp(double(x) - m); }
    return m + std::log(sum);
}
// decode=true teacher-forces through the decode path (forward_rows, eight
// consecutive rows of one stream per pass), so the receipt measures the decode
// identity itself rather than the WMMA prefill identity.
void quality(const g4::HostWeights& w, g4::RopeTables rope, const Args& args, bool decode = false) {
    if (args.teacher.empty()) throw std::invalid_argument("quality requires ordinary teacher directory");
    using g4::model_detail::require;
    const auto manifest = g4::read_json(args.teacher / "manifest.json");
    for (const auto& run : require(manifest, "runs").as_array())
        if (require(run, "variant").as_string() != "ordinary") throw std::runtime_error("quant quality requires ordinary teacher");
    std::unique_ptr<g4::prefill::DevicePrefill> prompt_owner;
    std::unique_ptr<gpu::DeviceTarget> target;
    if (decode) target = std::make_unique<gpu::DeviceTarget>(w, rope);
    else prompt_owner = std::make_unique<g4::prefill::DevicePrefill>(w, rope);
    std::vector<std::filesystem::path> paths;
    for (const auto& e : std::filesystem::directory_iterator(args.teacher)) if (e.path().extension() == ".npz") paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    if (paths.size() != 40) throw std::runtime_error("quality requires all 40 teacher parents");
    for (std::size_t doc = args.skip; doc < paths.size() && doc < std::size_t(args.skip) + args.limit; ++doc) {
        const auto t = Clock::now();
        const auto id = paths[doc].stem().string();
        const auto& record = require(require(manifest, "docs"), id.c_str());
        const auto hash = require(record, "npz_sha256").as_string();
        {
            const g4::MappedFile file(paths[doc]);
            if (g4::hash_bytes(file.data(), file.size()) != hash) throw std::runtime_error("teacher hash mismatch");
        }
        const auto teacher = g4::read_npz(paths[doc]);
        const auto ids = teacher.at("ids").values<std::int32_t>("<i4");
        const auto argmax = teacher.at("argmax").values<std::int32_t>("<i4");
        const auto ref_lse = teacher.at("lse").values<double>("<f8");
        const auto next = teacher.at("next_logit").values<float>("<f4");
        const auto full_rows = teacher.at("full_rows").values<std::int32_t>("<i4");
        const auto full = teacher.at("full_logits").values<std::uint16_t>("<f2");
        if (ids.empty() || ids.size() != argmax.size() || ref_lse.size() != ids.size() || next.size() + 1 != ids.size() ||
            full.size() != full_rows.size() * w.config.vocab) throw std::runtime_error("teacher geometry");
        std::vector<double> kl(full_rows.size()), dnll(ids.size() - 1);
        std::vector<unsigned> matches(ids.size());
        gpu::DeviceStream stream(w.config, unsigned(ids.size() + 8));
        const std::size_t step = decode ? 8 : g4::prefill::chunk_rows;
        for (std::size_t base = 0; base < ids.size(); base += step) {
            const auto count = std::min<std::size_t>(step, ids.size() - base);
            std::vector<unsigned> tokens(count);
            for (std::size_t i = 0; i < count; ++i) { if (ids[base + i] < 0) throw std::runtime_error("negative teacher token"); tokens[i] = unsigned(ids[base + i]); }
            auto tx = stream.begin();
            struct { std::vector<float> all_logits; } out;
            if (decode) {
                std::vector<g4::RowPlan> rows;
                for (std::size_t i = 0; i < count; ++i) rows.push_back({0, 0, base + i, tokens[i], tx.get(), i, true});
                out.all_logits = std::move(target->forward_rows(rows).logits);
            } else out.all_logits = std::move(prompt_owner->forward(*tx, tokens, true).all_logits);
            tx->commit(count);
            std::atomic<unsigned> cursor{0}; std::exception_ptr error; std::mutex mutex;
            const auto work = [&] {
                try { for (;;) {
                    const auto r = cursor.fetch_add(1); if (r >= count) break;
                    const auto pos = base + r; const auto row = std::span(out.all_logits).subspan(std::size_t(r) * w.config.vocab, w.config.vocab);
                    matches[pos] = greedy(row) == unsigned(argmax[pos]); const double logsum = lse(row);
                    if (pos + 1 < ids.size()) dnll[pos] = (logsum - row[unsigned(ids[pos + 1])]) - (ref_lse[pos] - next[pos]);
                    const auto it = std::lower_bound(full_rows.begin(), full_rows.end(), std::int32_t(pos));
                    if (it != full_rows.end() && *it == std::int32_t(pos)) {
                        const auto index = std::size_t(it - full_rows.begin()); std::vector<float> ref(w.config.vocab);
                        for (unsigned v = 0; v < w.config.vocab; ++v) ref[v] = g4::fp16_value(full[index * w.config.vocab + v]);
                        const double refsum = lse(ref); double sum = 0;
                        for (unsigned v = 0; v < w.config.vocab; ++v) {
                            const double lp = double(ref[v]) - refsum;
                            sum += std::exp(lp) * (lp - (double(row[v]) - logsum));
                        }
                        kl[index] = sum;
                    }
                } } catch (...) { std::lock_guard lock(mutex); if (!error) error = std::current_exception(); }
            };
            std::vector<std::thread> workers; for (unsigned i = 0; i < 16; ++i) workers.emplace_back(work);
            for (auto& worker : workers) worker.join(); if (error) std::rethrow_exception(error);
        }
        auto values = Json::array(); for (double value : kl) values.push(value);
        print(Json::Object{{"metric", "quality"}, {"parent", id}, {"language", require(record, "lang")}, {"teacher_sha256", hash},
            {"rows", std::uint64_t(ids.size())}, {"matches", std::uint64_t(std::accumulate(matches.begin(), matches.end(), 0ULL))},
            {"nll_positions", std::uint64_t(dnll.size())}, {"sum_dnll", std::accumulate(dnll.begin(), dnll.end(), 0.0)},
            {"kl_full", values}, {"seconds", elapsed(t)}});
    }
}
}
int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv); const auto t = Clock::now();
        if (args.mode != "gate" && g4::prefill::attention_mode() != g4::prefill::AttentionMode::canonical)
            throw std::invalid_argument("EXL3 qualification requires canonical attention");
        if (args.mode == "speed-all") {
            const auto rope = g4::load_rope_tables(args.rope);
            speed_all(args, rope.view); return 0;
        }
        auto model = args.model.extension() == ".gguf" ? g4::load_gguf_model(args.model) : ex::load_arm(args.model, args.bf16);
        const auto rope = g4::load_rope_tables(args.rope); const auto weights = g4::bind_host_weights(model);
        auto meta = Json::object(); meta.set("identity", model.identity); meta.set("model_hash", model.model_hash);
        meta.set("rope_hash", rope.view.hash); meta.set("mode", args.mode); meta.set("qualified", false);
        meta.set("prefill_identity", args.mode == "gate" ? "prefill-exact" : args.mode == "quality-decode" ? "decode-path"
                 : model.identity == ex::identity || model.identity == ex::identity_v2 || model.identity == ex::identity_v3 ? ex::prefill_identity : g4::prefill::identity);
        meta.set("load_s", elapsed(t));
        auto switches = Json::object();
        for (const auto* name : {"STRIX_G4_SHARED_WEIGHTS", "STRIX_G4_GRAPHS", "STRIX_G4_FUSION", "STRIX_G4_EXL3_OUTPUT_TILE", "STRIX_G4_EXL3_COLUMN_CACHE", "STRIX_G4_PREFILL_PREPARE"}) {
            const auto* value = std::getenv(name);
            switches.set(name, !value || std::string(value) != "0");
        }
        meta.set("off_switches", switches);
#if defined(__linux__)
        utsname info{}; if (uname(&info) == 0) meta.set("kernel", info.release);
        std::ifstream f("/sys/module/amdgpu/parameters/iommu_perfopt"); std::string perfopt;
        meta.set("iommu_perfopt", f >> perfopt ? perfopt : "not-present");
#endif
        print(meta);
        if (args.mode == "gate") gate(weights, rope.view, args);
        else if (args.mode == "speed") {
            gpu::DeviceTarget target(weights, rope.view); speed(target, rope.view, args);
        }
        else quality(weights, rope.view, args, args.mode == "quality-decode");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
