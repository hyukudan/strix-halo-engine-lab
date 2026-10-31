#include "device.hpp"
#include "gemma4/engine/profile.hpp"
#include "a8_host.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace g4 = strix::gemma4;
namespace gpu = g4::gpu;
namespace {
using Clock = std::chrono::steady_clock;
std::pair<std::string, std::string> host_kernel() {
    std::string kernel = "unavailable", perfopt = "unavailable";
#if defined(__linux__)
    utsname info{};
    if (uname(&info) == 0)
        kernel = info.release;
    std::ifstream parameter("/sys/module/amdgpu/parameters/iommu_perfopt");
    if (!(parameter >> perfopt))
        perfopt = "not-present";
#endif
    return {kernel, perfopt};
}
struct Args {
    std::filesystem::path model, rope;
    std::string route, identity; // --identity; empty: the profile's STRIX_G4_IDENTITY, else g4-v1
    unsigned rows = 1, steps = 8, warmup = 2, token = 2, context = 1024, verify_rows = 0;
    bool host = false, serial = false, profile = false, all_rows = false, verify_all = false;
    bool graph_timing = false;
    bool digest = false;
    std::optional<strix::gemma4::profile::Name> product; // --profile default|fast|quality
    bool allow_pending = false;                           // --allow-pending (benchmarking only)
};
unsigned number(const std::string& x) {
    std::size_t end = 0;
    const auto n = std::stoul(x, &end);
    if (end != x.size() || n > UINT32_MAX)
        throw std::runtime_error("invalid bounded integer");
    return unsigned(n);
}
Args arguments(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (strix::gemma4::profile::take_option(argc, argv, i, a.product))
            continue;
        if (key == "--check-host")
            a.host = true;
        else if (key == "--check-serial")
            a.serial = true;
        else if (key == "--profile" || key == "--kernel-profile")
            a.profile = true;
        else if (key == "--allow-pending")
            a.allow_pending = true;
        else if (key == "--time-graph")
            a.graph_timing = true;
        else if (key == "--digest")
            a.digest = true;
        else {
            if (++i >= argc)
                throw std::runtime_error("missing option value");
            const std::string v = argv[i];
            if (key == "--model")
                a.model = v;
            else if (key == "--rope")
                a.rope = v;
            else if (key == "--route")
                a.route = v;
            else if (key == "--identity")
                a.identity = v;
            else if (key == "--rows") {
                if (v == "all")
                    a.all_rows = true;
                else
                    a.rows = number(v);
            } else if (key == "--steps")
                a.steps = number(v);
            else if (key == "--warmup")
                a.warmup = number(v);
            else if (key == "--token")
                a.token = number(v);
            else if (key == "--context")
                a.context = number(v);
            else if (key == "--verify-rows") {
                a.verify_all = v == "all";
                a.verify_rows = a.verify_all ? 8U : number(v);
            }
            else
                throw std::runtime_error("unknown option: " + key);
        }
    }
    if (a.model.empty() || a.rope.empty() || (a.route != "bf16" && a.route != "q4_k_m") || !a.rows || a.rows > 8 ||
        !a.steps || !a.context || a.verify_rows > 8 ||
        (!a.identity.empty() && a.identity != g4::arithmetic_id && a.identity != g4::a8::arithmetic_id))
        throw std::runtime_error("usage: --model MODEL --route bf16|q4_k_m --rope TABLE_DIR [--rows 1..8|all --steps N "
                                 "--warmup N --context N --check-host --check-serial --verify-rows 1..8|all --kernel-profile "
                                 "--time-graph --digest --profile default|fast|quality --allow-pending "
                                 "--identity g4-v1|g4-a8-v1]");
    if (a.profile && a.graph_timing)
        throw std::runtime_error("profile and time-graph are mutually exclusive");
    if (a.verify_rows && (a.rows != 1 || a.all_rows))
        throw std::runtime_error("verify-rows uses one stream");
    if (!a.verify_rows && std::uint64_t(a.warmup) + a.steps > a.context)
        throw std::runtime_error("benchmark exceeds admitted context");
    return a;
}
void byte_equal(std::span<const float> a, std::span<const float> b, const char* what) {
    if (a.size() != b.size())
        throw std::runtime_error(std::string(what) + " shape mismatch");
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4))
            throw std::runtime_error(std::string(what) + " byte mismatch at " + std::to_string(i));
}
void same(const g4::ForwardResult& a, const g4::ForwardResult& b) {
    byte_equal(a.logits, b.logits, "logits");
    byte_equal(a.taps, b.taps, "taps");
}
unsigned greedy(std::span<const float> x) {
    unsigned best = 0;
    for (unsigned i = 1; i < x.size(); ++i)
        if (g4::better(x[i], i, x[best], best))
            best = i;
    return best;
}
void verify(gpu::DeviceTarget& target, const g4::HostWeights& weights, g4::RopeTables tables, const Args& args) {
    const auto n = args.verify_rows;
    const auto& c = weights.config;
    if (n > args.context)
        throw std::runtime_error("verify rows exceed admitted context");
    gpu::DeviceStream serial(c, args.context);
    g4::HostStream reference(c);
    const bool a8 = std::string_view(target.stats().identity) == g4::a8::arithmetic_id;
    g4::HostTarget host(weights, tables, 8, a8 ? g4::a8::host_recipe() : g4::LinearRecipe{});
    std::vector<g4::ForwardResult> expected;
    std::vector<std::string> digests;
    for (unsigned r = 0; r < n; ++r) {
        auto ht = reference.begin();
        g4::RowPlan hp{0, 0, r, (args.token + r) % c.vocab, ht.get(), 0, true};
        expected.push_back(host.forward_rows({&hp, 1}));
        ht->commit(1);
        digests.push_back(reference.digest());
        auto st = serial.begin();
        g4::RowPlan row{0, 0, r, (args.token + r) % c.vocab, st.get(), 0, true};
        const auto solo = target.forward_rows({&row, 1});
        same(solo, expected.back());
        st->commit(1);
        if (serial.digest() != digests.back())
            throw std::runtime_error("serial host KV digest mismatch");
    }
    for (unsigned count = args.verify_all ? 1U : n; count <= n; ++count) {
        gpu::DeviceStream batch(c, args.context);
        auto tx = batch.begin();
        std::vector<g4::RowPlan> dp;
        for (unsigned r = 0; r < count; ++r)
            dp.push_back({0, 0, r, (args.token + r) % c.vocab, tx.get(), r, true});
        std::reverse(dp.begin(), dp.end());
        const auto out = target.forward_rows(dp);
        const auto selection = target.selected();
        for (unsigned r = 0; r < count; ++r) {
            byte_equal(expected[r].logits, std::span(out.logits).subspan(std::size_t(r) * c.vocab, c.vocab),
                       "verify logits");
            byte_equal(expected[r].taps, std::span(out.taps).subspan(std::size_t(r) * out.tap_width, out.tap_width),
                       "verify taps");
            const auto logits = std::span(out.logits).subspan(std::size_t(r) * c.vocab, c.vocab);
            if (selection[r].id != greedy(logits) ||
                std::memcmp(&selection[r].value, &logits[selection[r].id], sizeof(float)))
                throw std::runtime_error("verify capped selection mismatch");
        }
        tx->commit(count);
        if (batch.digest() != digests[count - 1])
            throw std::runtime_error("verify KV digest mismatch");
        std::cout << "{\"gate\":\"verify_rows\",\"identity\":\"" << target.stats().identity
                  << "\",\"verdict\":\"EXACT\",\"rows\":" << count << ",\"kv\":\"" << batch.digest()
                  << "\"}\n";
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        // Product profile: resolved and exported before any switch is read.
        namespace pr = strix::gemma4::profile;
        const auto product = pr::resolve(args.product.value_or(pr::Name::standard), pr::App::decode_bench,
                                    {.allow_pending = args.allow_pending});
        pr::apply(product);
        std::cout << "{\"profile\":" << product.json() << "}\n";
        const auto [kernel, perfopt] = host_kernel();
        auto model = args.route == "bf16" ? g4::load_safetensors_model(args.model) : g4::load_gguf_model(args.model);
        const auto weights = g4::bind_host_weights(model);
        auto tables = g4::load_rope_tables(args.rope);
        if (args.token >= model.config.vocab || args.context > tables.view.positions ||
            args.context > model.config.max_positions)
            throw std::runtime_error("token/context exceeds assets");
        const auto load_start = Clock::now();
        gpu::DeviceTarget target(weights, tables.view,
                                 {true, true, args.profile, true, args.graph_timing,
                                  args.identity.empty() ? nullptr : args.identity.c_str()});
        const auto upload_s = std::chrono::duration<double>(Clock::now() - load_start).count();
        if (args.verify_rows) {
            verify(target, weights, tables.view, args);
            return 0;
        }
        const std::vector<unsigned> row_counts =
            args.all_rows ? std::vector<unsigned>{1, 2, 4, 8} : std::vector<unsigned>{args.rows};
        for (const auto row_count : row_counts) {
            std::unique_ptr<g4::HostTarget> host;
            if (args.host)
                host = std::make_unique<g4::HostTarget>(
                    weights, tables.view, 8,
                    std::string_view(target.stats().identity) == g4::a8::arithmetic_id ? g4::a8::host_recipe()
                                                                                       : g4::LinearRecipe{});
            std::vector<std::unique_ptr<gpu::DeviceStream>> streams, serial;
            std::vector<std::unique_ptr<g4::HostStream>> refs;
            for (unsigned r = 0; r < row_count; ++r) {
                streams.push_back(std::make_unique<gpu::DeviceStream>(model.config, args.context));
                if (args.serial)
                    serial.push_back(std::make_unique<gpu::DeviceStream>(model.config, args.context));
                if (args.host)
                    refs.push_back(std::make_unique<g4::HostStream>(model.config));
            }
            std::vector<unsigned> tokens(row_count, args.token);
            for (unsigned r = 0; r < row_count; ++r)
                tokens[r] = (args.token + r) % model.config.vocab;
            std::vector<double> timings;
            std::map<std::string, double> kernel_ms;
            std::size_t traffic = 0;
            gpu::DeviceStats applied;
            double gpu_ms = 0.0, forward_ms = 0.0, commit_ms = 0.0;
            std::uint64_t output_hash = 1469598103934665603ULL;
            for (unsigned step = 0; step < args.warmup + args.steps; ++step) {
                std::vector<std::unique_ptr<g4::KvTxn>> transactions, host_transactions;
                std::vector<g4::RowPlan> plans, host_plans;
                for (unsigned r = 0; r < row_count; ++r) {
                    transactions.push_back(streams[r]->begin());
                    plans.push_back({r, 0, step, tokens[r], transactions.back().get(), r, true});
                    if (host) {
                        host_transactions.push_back(refs[r]->begin());
                        host_plans.push_back({r, 0, step, tokens[r], host_transactions.back().get(), r, true});
                    }
                }
                const auto start = Clock::now();
                const auto out = target.forward_rows(plans);
                const auto selection = target.selected();
                const auto commit_start = Clock::now();
                for (auto& tx : transactions)
                    tx->commit(1);
                const auto end = Clock::now();
                // Untimed: hashes every pass's logits and taps.
                if (args.digest)
                    for (const auto* values : {&out.logits, &out.taps})
                        for (const float v : *values) {
                            std::uint32_t bits;
                            std::memcpy(&bits, &v, sizeof(bits));
                            output_hash = (output_hash ^ bits) * 1099511628211ULL;
                        }
                const auto elapsed = std::chrono::duration<double>(end - start).count();
                if (step >= args.warmup) {
                    timings.push_back(elapsed);
                    traffic = target.stats().pass_weight_bytes;
                    applied = target.stats();
                    gpu_ms += applied.gpu_pass_ms;
                    forward_ms += std::chrono::duration<double, std::milli>(commit_start - start).count();
                    commit_ms += std::chrono::duration<double, std::milli>(end - commit_start).count();
                    for (const auto& [k, v] : target.stats().kernel_ms)
                        kernel_ms[k] += v;
                }
                if (host) {
                    const auto expected = host->forward_rows(host_plans);
                    same(out, expected);
                    for (auto& tx : host_transactions)
                        tx->commit(1);
                }
                for (unsigned r = 0; r < row_count; ++r) {
                    const auto logits =
                        std::span(out.logits).subspan(std::size_t(r) * model.config.vocab, model.config.vocab);
                    const auto taps = std::span(out.taps).subspan(std::size_t(r) * out.tap_width, out.tap_width);
                    if (args.serial) {
                        auto st = serial[r]->begin();
                        g4::RowPlan row{r, 0, step, tokens[r], st.get(), 0, true};
                        const auto solo = target.forward_rows({&row, 1});
                        byte_equal(logits, solo.logits, "solo logits");
                        byte_equal(taps, solo.taps, "solo taps");
                        st->commit(1);
                        if (streams[r]->digest() != serial[r]->digest())
                            throw std::runtime_error("solo KV mismatch");
                    }
                    if (host && streams[r]->digest() != refs[r]->digest())
                        throw std::runtime_error("host KV mismatch");
                    if ((args.serial || args.host) && selection[r].id != greedy(logits))
                        throw std::runtime_error("capped device selection mismatch");
                    tokens[r] = selection[r].id;
                }
            }
            if (args.digest) {
                // Final per-stream KV digests and a hash of the last pass's logits/taps bytes,
                // for comparing two schedules over long runs without a reference.
                std::cout << "{\"gate\":\"digest\",\"rows\":" << row_count << ",\"kv\":[";
                for (unsigned r = 0; r < row_count; ++r)
                    std::cout << (r ? "," : "") << '"' << streams[r]->digest() << '"';
                std::cout << "],\"outputs\":\"" << std::hex << output_hash << std::dec << "\",\"tokens\":[";
                for (unsigned r = 0; r < row_count; ++r)
                    std::cout << (r ? "," : "") << tokens[r];
                std::cout << "]}\n";
            }
            std::sort(timings.begin(), timings.end());
            const auto m = timings.size() / 2;
            const double median = timings.size() % 2 ? timings[m] : (timings[m - 1] + timings[m]) * 0.5;
            std::cout << std::setprecision(10) << "{\"identity\":\"" << applied.identity << "\",\"profile\":\""
                      << pr::name(product.name) << "\",\"qualification\":\"" << product.qualification() << "\",\"route\":\"" << args.route
                      << "\",\"model_hash\":\"" << model.model_hash << "\",\"rotary_hash\":\"" << tables.view.hash
                      << "\",\"cache_dtype\":\"F32\",\"qualified\":false,\"rows\":" << row_count
                      << ",\"steps\":" << args.steps << ",\"median_pass_s\":" << median
                      << ",\"aggregate_tok_s\":" << row_count / median
                      << ",\"weight_bytes_per_token\":" << double(traffic) / row_count
                      << ",\"effective_weight_gb_s\":" << double(traffic) / median / 1e9
                      << ",\"resident_weight_bytes\":" << applied.resident_weight_bytes
                      << ",\"upload_validate_s\":" << upload_s << ",\"profiled\":" << (args.profile ? "true" : "false")
                      << ",\"shared_weights\":" << (applied.shared_weights ? "true" : "false")
                      << ",\"gate_up_fusion\":" << (applied.fusion ? "true" : "false")
                      << ",\"graphs\":" << (applied.graphs ? "true" : "false")
                      << ",\"norm_group\":" << (applied.norm_group ? "true" : "false")
                      << ",\"chunk_decode\":" << (applied.chunk_decode ? "true" : "false")
                      << ",\"norm_search\":" << (applied.norm_search ? "true" : "false")
                      << ",\"kv_pool\":" << (applied.kv_pool ? "true" : "false")
                      << ",\"output_tile\":" << (applied.output_tile ? "true" : "false")
                      << ",\"kv_publish\":" << (applied.kv_publish ? "true" : "false")
                      << ",\"linear_finite\":" << (applied.linear_finite ? "true" : "false")
                      << ",\"norm_prefetch\":" << (applied.norm_prefetch ? "true" : "false")
                      << ",\"norm_finite\":" << (applied.norm_finite ? "true" : "false")
                      << ",\"pinned_io\":" << (applied.pinned_io ? "true" : "false")
                      << ",\"norm_bounds\":" << (applied.norm_bounds ? "true" : "false")
                      << ",\"norm_stage\":" << (applied.norm_stage ? "true" : "false")
                      << ",\"head_registers\":" << (applied.head_registers ? "true" : "false")
                      << ",\"input_cache\":" << (applied.input_cache ? "true" : "false")
                      << ",\"attention_prefetch\":" << (applied.attention_prefetch ? "true" : "false")
                      << ",\"norm_intervals\":" << (applied.norm_intervals ? "true" : "false")
                      << ",\"one_chunk_attention\":" << (applied.one_chunk_attention ? "true" : "false")
                      << ",\"chunk_prefetch\":" << (applied.chunk_prefetch ? "true" : "false")
                      << ",\"chunk_prefetch_bf16_only\":" << (applied.chunk_prefetch_bf16_only ? "true" : "false")
                      << ",\"packed_header\":" << (applied.packed_header ? "true" : "false")
                      << ",\"streaming_weights\":" << (applied.streaming_weights ? "true" : "false")
                      << ",\"streaming_weights_auto\":" << (applied.streaming_weights_auto ? "true" : "false")
                      << ",\"streaming_weights_bf16_only\":" << (applied.streaming_weights_bf16_only ? "true" : "false")
                      << ",\"layer_norm_fusion\":" << (applied.layer_norm_fusion ? "true" : "false")
                      << ",\"fused_attention\":" << (applied.fused_attention ? "true" : "false")
                      << ",\"store_fusion\":" << (applied.store_fusion ? "true" : "false")
                      << ",\"multi_linear\":" << (applied.multi_linear ? "true" : "false")
                      << ",\"graph_timed\":" << (args.graph_timing ? "true" : "false")
                      << ",\"gpu_ms_per_pass\":" << gpu_ms / args.steps
                      << ",\"forward_ms_per_pass\":" << forward_ms / args.steps
                      << ",\"commit_ms_per_pass\":" << commit_ms / args.steps
                      << ",\"kernel\":\"" << kernel << "\",\"amdgpu_iommu_perfopt\":\"" << perfopt << '"'
                      << ",\"host_verdict\":\"" << (args.host ? "EXACT" : "NO-REFERENCE") << "\",\"solo_verdict\":\""
                      << (args.serial ? "EXACT" : "NO-REFERENCE") << "\",\"kernel_ms_per_pass\":{";
            bool first = true;
            for (const auto& [k, v] : kernel_ms) {
                if (!first)
                    std::cout << ',';
                first = false;
                std::cout << '"' << k << "\":" << v / args.steps;
            }
            std::cout << "}}\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gemma4 device: " << e.what() << '\n';
        return 1;
    }
}
