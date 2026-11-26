#include "inkling2/inkling2_mini_host.hpp"
#include "inkling2/inkling2_mini_pack_source.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "moe_core/grouped_strixpack.hpp"
#include "strix/moe/route_file.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/weight/calib_moments.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace {
namespace ink = strix::inkling2;
namespace wei = strix::weight;
namespace cal = wei::calib;
namespace sp = strix::moe_core::grouped_strixpack;
namespace fs = std::filesystem;

struct Window {
    std::string name;
    std::vector<std::uint32_t> ids;
};

void require(const wei::Status& status) {
    if (!status) throw std::runtime_error(status.detail);
}

std::vector<Window> read_windows(const fs::path& manifest) {
    std::ifstream in(manifest);
    std::vector<Window> windows;
    std::set<std::string> names;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        Window window;
        std::string file;
        std::size_t count = 0;
        if (!(fields >> window.name) || window.name.starts_with('#')) continue;
        if (!(fields >> count >> file) || count < 2 || count > ink::kMaxContext ||
            window.name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_") !=
                std::string::npos || !names.insert(window.name).second)
            throw std::runtime_error("invalid study window: " + window.name);
        std::ifstream ids(manifest.parent_path() / file);
        std::uint32_t id = 0;
        while (ids >> id) {
            if (id >= ink::kOutputVocabulary || window.ids.size() == count)
                throw std::runtime_error("invalid study token: " + window.name);
            window.ids.push_back(id);
        }
        if (window.ids.size() != count || !ids.eof()) throw std::runtime_error("invalid study window IDs: " + window.name);
        windows.push_back(std::move(window));
    }
    if (windows.empty()) throw std::runtime_error("no study windows");
    return windows;
}

class Study final : public ink::MiniHostStudy {
public:
    Study(bool capture, const fs::path& output) : capture_(capture), output_(output) {}

    void open_pack(const std::string& checkpoint, const std::string& path) {
        require(wei::PackCheckpoint::open(checkpoint, checkpoint_));
        wei::Digest256 digest;
        require(checkpoint_.digest(digest));
        require(wei::LowbitPack::open_host(path, digest, pack_));
        ink::MiniTextWeights weights;
        require(weights.open(checkpoint));
        std::uint64_t bytes = 0;
        for (const auto& entry : pack_.entries()) {
            if (entry.bytes > (24ULL << 30U) - bytes) throw std::runtime_error("pack exceeds 24 GiB host limit");
            bytes += entry.bytes;
            wei::PackMatrixSource source;
            require(ink::mini_pack_source(checkpoint_, weights, entry.tensor, source));
            require(matrices_[entry.tensor].load(pack_, entry, source));
        }
        require(pack_.verify_identity());
    }

    void begin(const Window& prompt) {
        if (capture_) window_ = moments_.add_window(prompt.ids);
        else {
            auto config = strix::moe::route_tap_config_from_env();
            config.out_path = (output_ / (prompt.name + ".rte")).string();
            if (!config.in_path.empty()) config.in_path = (fs::path(config.in_path) / (prompt.name + ".rte")).string();
            route_tap_ = std::make_unique<strix::moe::RouteTap>(std::move(config));
            hidden_.open(output_ / (prompt.name + ".hidden.f32"), std::ios::binary);
            if (!hidden_) throw std::runtime_error("cannot create hidden rows file");
        }
    }

    void end(const Window& prompt) {
        if (!route_tap_) return;
        std::ofstream receipt(output_ / (prompt.name + ".routes.log"));
        receipt << "strix-route-file/v1 " << route_tap_->receipt() << '\n';
        if (!receipt) throw std::runtime_error("cannot write route receipt");
        route_tap_.reset();
        hidden_.close();
        if (!hidden_) throw std::runtime_error("cannot write hidden rows");
    }

    void input(std::string_view name, std::span<const float> row) override {
        if (!capture_) {
            if (name == "model.llm.unembed.weight")
                hidden_.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size_bytes()));
            return;
        }
        auto pos = inputs_.find(name);
        if (pos == inputs_.end()) {
            const std::array consumer{std::string(name)};
            auto& moments = moments_.declare(name, static_cast<std::uint32_t>(row.size()), consumer);
            pos = inputs_.emplace(consumer[0], &moments).first;
        }
        moments_.add(window_, std::span(&pos->second, 1), row.data(), 1);
    }

    void embedding(unsigned token) override {
        if (!capture_) return;
        ++embeddings_[token];
        // Each lookup exposes every channel of its selected row with unit gain.
        std::array<float, ink::kHiddenWidth> unit;
        unit.fill(1.0F);
        input("model.llm.embed.weight", unit);
    }

    void routes(unsigned logical, std::span<std::uint32_t> ids, std::span<float> weights) override {
        if (capture_) {
            for (const auto id : ids) ++routes_.at(logical).at(id);
        } else {
            route_tap_->on_call(logical, 1, ink::kExpertsPerToken, ids.data(), weights.data());
            if (std::ranges::any_of(ids, [](std::uint32_t id) { return id >= ink::kExperts; }) ||
                std::ranges::any_of(weights, [](float weight) { return !std::isfinite(weight) || weight < 0.0F; }))
                throw std::runtime_error("invalid study routes");
        }
    }

    bool project(std::string_view name, std::span<const float> row, std::span<float> out,
                 std::size_t first) override {
        const auto it = matrices_.find(name);
        if (it == matrices_.end()) return false;
        sp::matvec(it->second, row, out, first);
        return true;
    }

    bool lookup(std::string_view name, unsigned token, std::span<float> row) override {
        const auto it = matrices_.find(name);
        if (it == matrices_.end()) return false;
        sp::lookup(it->second, token, row);
        return true;
    }

    void score_head(const Window& prompt, const fs::path& reference) {
        constexpr std::string_view name = "model.llm.unembed.weight";
        if (matrices_.size() != 1 || !matrices_.contains(name))
            throw std::runtime_error("head scoring requires a head-only pack");
        std::ifstream rows(reference / (prompt.name + ".hidden.f32"), std::ios::binary);
        std::ofstream out(output_ / (prompt.name + ".f32"), std::ios::binary);
        if (!rows || !out) throw std::runtime_error("cannot open head study rows");
        std::array<float, ink::kHiddenWidth> row;
        std::vector<float> logits(ink::kOutputVocabulary);
        for (std::size_t i = 0; i < prompt.ids.size(); ++i) {
            if (!rows.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(row))))
                throw std::runtime_error("missing exact head input row");
            project(name, row, logits, 0);
            ink::mini_cpu_finalize_logits(logits);
            out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
        }
        if (rows.peek() != EOF || !out) throw std::runtime_error("invalid head study rows");
    }

    void finish() const {
        if (!capture_) return;
        moments_.write((output_ / "fit.moments").string());
        std::ofstream routes(output_ / "routes.json");
        routes << "{\"routes\":[";
        std::array<std::array<std::uint64_t, ink::kExperts>, ink::kPhysicalLayers> physical{};
        for (const auto visit : ink::mini_schedule())
            for (unsigned expert = 0; expert < ink::kExperts; ++expert)
                physical[visit.physical_layer][expert] += routes_[visit.logical_layer][expert];
        for (unsigned layer = 0; layer < physical.size(); ++layer)
            for (unsigned expert = 0; expert < ink::kExperts; ++expert)
                routes << (layer || expert ? "," : "") << "[" << layer << "," << expert << ","
                       << physical[layer][expert] << "]";
        routes << "]}\n";
        std::ofstream lookup(output_ / "embedding-counts.tsv");
        lookup << "token\tcount\n";
        for (const auto& [token, count] : embeddings_) lookup << token << '\t' << count << '\n';
        if (!routes || !lookup) throw std::runtime_error("cannot write study counters");
        std::cerr << "inputs=" << inputs_.size() << " moment_bytes=" << moments_.moment_bytes() << '\n';
    }

private:
    bool capture_;
    fs::path output_;
    std::size_t window_{};
    cal::MomentSet moments_{1, cal::MomentLayout::diagonal, 192U * 1024U * 1024U};
    std::map<std::string, cal::InputMoments*, std::less<>> inputs_;
    std::array<std::array<std::uint64_t, ink::kExperts>, ink::kLogicalVisits> routes_{};
    std::map<unsigned, std::uint64_t> embeddings_;
    std::unique_ptr<strix::moe::RouteTap> route_tap_;
    std::ofstream hidden_;
    wei::PackCheckpoint checkpoint_;
    wei::LowbitPack pack_;
    std::map<std::string, sp::Matrix, std::less<>> matrices_;
};
}  // namespace

int mini_cpu_pack_study(int argc, char** argv) {
    try {
        const std::string_view mode(argc > 2 ? argv[2] : "");
        if (argc == 5 && mode == "tokenize") {
            const auto tokenizer = strix::serve::BpeTokenizer::load_hf_json(
                argv[3], strix::serve::inkling2_mini_tokenizer_identity);
            std::ifstream prompt(argv[4], std::ios::binary);
            if (!prompt) throw std::runtime_error("cannot open prompt");
            std::ostringstream text;
            text << prompt.rdbuf();
            for (const auto id : tokenizer->encode(text.str(), false)) std::cout << id << '\n';
            return 0;
        }
        const bool capture = mode == "capture";
        const bool head = mode == "score-head";
        if ((head && argc != 8) || (!head && (argc < 6 || argc > 7 || (!capture && mode != "score") ||
                                             (capture && argc != 6))))
            throw std::invalid_argument("usage: --pack-study tokenize TOKENIZER_JSON TEXT or "
                                        "--pack-study capture|score CHECKPOINT MANIFEST NEW_DIRECTORY [PACK] or "
                                        "--pack-study score-head CHECKPOINT MANIFEST NEW_DIRECTORY PACK EXACT_ROWS");
        const auto windows = read_windows(argv[4]);
        const fs::path output(argv[5]);
        if (!fs::create_directory(output)) throw std::runtime_error("study output directory must be new");
        Study study(capture, output);
        if (argc >= 7) study.open_pack(argv[3], argv[6]);
        for (const auto& window : windows) {
            if (head) {
                study.score_head(window, argv[7]);
                continue;
            }
            study.begin(window);
            const auto count = window.ids.size();
            ink::MiniHostDecoder decoder(argv[3], static_cast<unsigned>(window.ids.size()), nullptr, &study);
            std::ofstream out;
            if (!capture) {
                out.open(output / (window.name + ".f32"), std::ios::binary);
                if (!out) throw std::runtime_error("cannot create study logits");
            }
            for (std::size_t i = 0; i < count; ++i) {
                const auto logits = decoder.step(window.ids[i], !capture);
                if (!capture) {
                    out.write(reinterpret_cast<const char*>(logits.data()),
                              static_cast<std::streamsize>(logits.size() * sizeof(float)));
                    if (!out) throw std::runtime_error("cannot write study logits");
                }
                if (i == 0 || (i + 1) % 16 == 0 || i + 1 == count)
                    std::cerr << window.name << " " << i + 1 << "/" << count << '\n';
            }
            study.end(window);
        }
        study.finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini CPU study: " << error.what() << '\n';
        return 1;
    }
}
