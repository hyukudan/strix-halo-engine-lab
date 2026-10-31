// Teacher-forced fidelity receipts of the opt-in g4-a8-v1 identity against g4-v1 on the same weights.
// Explicit full-model CLI (never a ctest). Every parent runs through both identities, eight causal rows
// per pass (row invariance makes that equal to serial decoding). Per counted position t < T-1 it
// records top-1 against the teacher argmax, next-token NLL, KL(teacher || engine) on the teacher's
// full FP16 rows (self-normalized in FP64), and the cross-identity top-1 agreement and full-vocabulary
// KL(g4-v1 || g4-a8-v1). One JSON line per parent; decisions belong to the qualification lane.
#include "device.hpp"
#include "gemma4/ref/npz.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace g4 = strix::gemma4;
namespace gpu = g4::gpu;
namespace {
struct Args {
    std::filesystem::path model, rope, teacher, out;
    unsigned skip = 0, limit = 1000;
};
struct RowStats {
    std::uint32_t top1 = 0;
    double nll = 0.0;
    double lse = 0.0;
};
double logsumexp(std::span<const float> z) {
    double m = -INFINITY;
    for (float v : z)
        m = std::max(m, double(v));
    double s = 0.0;
    for (float v : z)
        s += std::exp(double(v) - m);
    return m + std::log(s);
}
std::uint32_t greedy(std::span<const float> z) {
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < z.size(); ++i)
        if (g4::better(z[i], i, z[best], best))
            best = i;
    return best;
}
// KL(p || q) for log-probability rows a = za - la, b = zb - lb.
double kl(std::span<const double> za, double la, std::span<const float> zb, double lb) {
    double s = 0.0;
    for (std::size_t i = 0; i < za.size(); ++i) {
        const double lp = za[i] - la, lq = double(zb[i]) - lb;
        s += std::exp(lp) * (lp - lq);
    }
    return s;
}
std::string json_list(const std::vector<double>& v) {
    std::ostringstream o;
    o << std::setprecision(9) << '[';
    for (std::size_t i = 0; i < v.size(); ++i)
        o << (i ? "," : "") << v[i];
    o << ']';
    return o.str();
}
} // namespace
int main(int argc, char** argv) {
    try {
        Args a;
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--model")
                a.model = v;
            else if (k == "--rope")
                a.rope = v;
            else if (k == "--teacher")
                a.teacher = v;
            else if (k == "--out")
                a.out = v;
            else if (k == "--skip")
                a.skip = unsigned(std::stoul(v));
            else if (k == "--limit")
                a.limit = unsigned(std::stoul(v));
            else
                throw std::runtime_error("unknown option " + k);
        }
        if (a.model.empty() || a.rope.empty() || a.teacher.empty() || a.out.empty())
            throw std::runtime_error("usage: --model Q4.gguf --rope DIR --teacher ORDINARY_DIR --out FILE.jsonl "
                                     "[--skip N --limit N]");
        auto model = g4::load_gguf_model(a.model);
        const auto weights = g4::bind_host_weights(model);
        auto tables = g4::load_rope_tables(a.rope);
        const char* names[2] = {g4::arithmetic_id, g4::a8::arithmetic_id};
        std::unique_ptr<gpu::DeviceTarget> targets[2];
        for (int k = 0; k < 2; ++k) {
            gpu::DeviceOptions options;
            options.identity = names[k];
            targets[k] = std::make_unique<gpu::DeviceTarget>(weights, tables.view, options);
        }
        std::vector<std::filesystem::path> parents;
        for (const auto& e : std::filesystem::directory_iterator(a.teacher))
            if (e.path().extension() == ".npz")
                parents.push_back(e.path());
        std::sort(parents.begin(), parents.end());
        std::ofstream out(a.out, std::ios::app);
        const auto vocab = model.config.vocab;
        for (std::size_t pi = a.skip; pi < parents.size() && pi < std::size_t(a.skip) + a.limit; ++pi) {
            const auto arrays = g4::read_npz(parents[pi]);
            const auto ids = arrays.at("ids").values<std::int32_t>("<i4");
            const auto argmax = arrays.at("argmax").values<std::int32_t>("<i4");
            const auto lse = arrays.at("lse").values<double>("<f8");
            const auto next = arrays.at("next_logit").values<float>("<f4");
            const auto full_rows = arrays.at("full_rows").values<std::int32_t>("<i4");
            const auto full = arrays.at("full_logits").values<std::uint16_t>("<f2");
            const auto T = ids.size();
            if (T < 2 || argmax.size() != T || lse.size() != T || next.size() != T - 1 ||
                full.size() != full_rows.size() * std::size_t(vocab))
                throw std::runtime_error("teacher schema: " + parents[pi].string());
            std::map<std::size_t, std::size_t> full_index;
            for (std::size_t k = 0; k < full_rows.size(); ++k)
                full_index[std::size_t(full_rows[k])] = k;
            const auto id = parents[pi].stem().string();
            const auto first = id.find('_'), second = id.find('_', first + 1);
            const auto lang = id.substr(first + 1, second - first - 1);
            std::vector<std::unique_ptr<gpu::DeviceStream>> streams;
            for (int k = 0; k < 2; ++k)
                streams.push_back(std::make_unique<gpu::DeviceStream>(model.config, std::uint32_t(T)));
            std::uint32_t top1[2] = {0, 0}, agree = 0;
            double nll[2] = {0, 0}, teacher_nll = 0;
            std::vector<double> teacher_kl[2], cross_kl;
            std::vector<std::uint32_t> flips; // positions where the identities pick different tokens
            for (std::size_t p = 0; p < T; p += 8) {
                const auto n = std::min<std::size_t>(8, T - p);
                g4::ForwardResult results[2];
                for (int k = 0; k < 2; ++k) {
                    auto tx = streams[k]->begin();
                    std::vector<g4::RowPlan> rows;
                    for (std::size_t r = 0; r < n; ++r)
                        rows.push_back({0, 0, p + r, std::uint32_t(ids[p + r]), tx.get(), r, true});
                    results[k] = targets[k]->forward_rows(rows);
                    tx->commit(n);
                }
                struct Row {
                    RowStats arm[2];
                    double teacher_kl[2] = {-1, -1}, cross = 0;
                    std::uint32_t pick[2] = {0, 0};
                };
                std::vector<Row> stats(n);
                std::vector<std::thread> workers;
                for (std::size_t r = 0; r < n; ++r)
                    workers.emplace_back([&, r] {
                        const auto t = p + r;
                        auto& s = stats[r];
                        std::span<const float> z[2];
                        for (int k = 0; k < 2; ++k) {
                            z[k] = std::span<const float>(results[k].logits).subspan(r * vocab, vocab);
                            s.arm[k].lse = logsumexp(z[k]);
                            s.pick[k] = greedy(z[k]);
                            if (t + 1 < T)
                                s.arm[k].nll = s.arm[k].lse - double(z[k][std::uint32_t(ids[t + 1])]);
                        }
                        std::vector<double> zd(vocab);
                        for (std::size_t i = 0; i < vocab; ++i)
                            zd[i] = z[0][i];
                        s.cross = kl(zd, s.arm[0].lse, z[1], s.arm[1].lse);
                        const auto it = full_index.find(t);
                        if (it != full_index.end()) {
                            for (std::size_t i = 0; i < vocab; ++i)
                                zd[i] = g4::fp16_value(full[it->second * vocab + i]);
                            double m = -INFINITY, sum = 0;
                            for (double v : zd)
                                m = std::max(m, v);
                            for (double v : zd)
                                sum += std::exp(v - m);
                            const double tl = m + std::log(sum);
                            for (int k = 0; k < 2; ++k)
                                s.teacher_kl[k] = kl(zd, tl, z[k], s.arm[k].lse);
                        }
                    });
                for (auto& w : workers)
                    w.join();
                for (std::size_t r = 0; r < n; ++r) {
                    const auto t = p + r;
                    if (t + 1 >= T)
                        continue; // counted positions predict an existing next token
                    const auto& s = stats[r];
                    for (int k = 0; k < 2; ++k) {
                        top1[k] += s.pick[k] == std::uint32_t(argmax[t]);
                        nll[k] += s.arm[k].nll;
                        if (s.teacher_kl[k] >= 0)
                            teacher_kl[k].push_back(s.teacher_kl[k]);
                    }
                    agree += s.pick[0] == s.pick[1];
                    if (s.pick[0] != s.pick[1])
                        flips.push_back(std::uint32_t(t));
                    cross_kl.push_back(s.cross);
                    teacher_nll += lse[t] - double(next[t]);
                }
            }
            std::ostringstream line;
            line << std::setprecision(12) << "{\"id\":\"" << id << "\",\"lang\":\"" << lang << "\",\"positions\":" << T - 1
                 << ",\"teacher_nll_sum\":" << teacher_nll << ",\"agree\":" << agree << ",\"flips\":[";
            for (std::size_t i = 0; i < flips.size(); ++i)
                line << (i ? "," : "") << flips[i];
            line << "],\"cross_kl\":" << json_list(cross_kl);
            for (int k = 0; k < 2; ++k)
                line << ",\"" << names[k] << "\":{\"top1\":" << top1[k] << ",\"nll_sum\":" << nll[k]
                     << ",\"teacher_kl\":" << json_list(teacher_kl[k]) << '}';
            line << ",\"model_hash\":\"" << model.model_hash << "\",\"rope_hash\":\"" << tables.view.hash
                 << "\",\"identities\":[\"" << targets[0]->stats().identity << "\",\"" << targets[1]->stats().identity
                 << "\"]}";
            out << line.str() << '\n';
            out.flush();
            std::cout << id << ' ' << lang << " T=" << T << " top1 " << top1[0] << '/' << top1[1] << " agree " << agree
                      << '/' << T - 1 << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "a8-teacher: " << e.what() << '\n';
        return 1;
    }
}
