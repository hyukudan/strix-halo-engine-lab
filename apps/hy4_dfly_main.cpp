// strix-hy4-dfly: DFly2 drafter tools on the Hy4-flash CPU oracle.
//   strix-hy4-dfly taps PART1.gguf [PART2.gguf] --ids-file F [--gen N] [--threads T] --out DIR
// runs the oracle (strix-hy4-ref arithmetic) over the prompt plus N greedy ids and writes the target state a DFly2 draft
// step reads: per fed position the last-layer hidden (hc_head output, before and after output_norm), the latent rows of
// the drafter's KV-share layers, the embedding rows of the fed ids and of the mask id, and the BF16 head table.
//   strix-hy4-dfly accept SHARD.safetensors TAPDIR[,TAPDIR...] [--variant V[;V...]]
// drafts one DFly2 block (src/hy4/ref/hy4_dfly_ref) at every anchor of the greedy region of each taps run and prints the
// acceptance against the target's greedy ids: E = 1 + mean accepted run (0..5), hit rate per proposal row, ms per block.
// Files (little-endian): ids.txt (fed ids, then the last argmax), hidden_pre.f32 / hidden_post.f32 [P][hidden],
// latents.f32 [kv layer][P][kv_lora + qk_rope], embed.f32 [P + 1][hidden] (row P = mask id), head.bf16
// [vocab][hidden], manifest.txt.
#include "hy4/ref/hy4_dfly_ref.hpp"
#include "hy4/ref/hy4_host_ref.hpp"
#include "strix/ds41_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <thread>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strix::hy4::Config;
using strix::hy4::HostRef;

constexpr std::uint32_t kMaskId = 120009U;                                  // config.json dfly_mask_token_id
constexpr std::uint32_t kKvLayers[] = {25U, 29U, 33U, 37U, 41U};            // dfly_target_kv_layers

std::vector<std::uint32_t> read_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ids file " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    for (auto& ch : text)
        if (ch == ',') ch = ' ';
    std::istringstream words(text);
    std::vector<std::uint32_t> ids;
    std::uint64_t v = 0;
    while (words >> v) ids.push_back(static_cast<std::uint32_t>(v));
    return ids;
}

void put(std::ofstream& out, const std::vector<float>& v) {
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
}

int taps(int argc, char** argv) {
    std::vector<std::filesystem::path> parts;
    std::vector<std::uint32_t> prompt;
    std::uint32_t gen = 0;
    unsigned threads = 1;
    std::filesystem::path out_dir;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--ids-file" && more) prompt = read_ids(argv[++i]);
        else if (a == "--gen" && more) gen = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (a == "--threads" && more) threads = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (a == "--out" && more) out_dir = argv[++i];
        else parts.emplace_back(a);
    }
    if (parts.empty() || prompt.empty() || out_dir.empty())
        throw std::invalid_argument("usage: strix-hy4-dfly taps PART1.gguf [PART2.gguf] --ids-file F [--gen N] [--threads T] --out DIR");
    std::filesystem::create_directories(out_dir);
    strix::hy4::set_ref_threads(threads);
    const auto config = Config::from_gguf(parts.front());
    auto weights = strix::hy4::pack_weights(parts);
    HostRef ref(config, *weights);
    const auto h = config.hidden;
    std::ofstream pre(out_dir / "hidden_pre.f32", std::ios::binary), post(out_dir / "hidden_post.f32", std::ios::binary);
    std::vector<std::uint32_t> fed;
    const auto total = static_cast<std::uint32_t>(prompt.size()) + gen;
    for (std::uint32_t p = 0; p < total; ++p) {
        const std::uint32_t token = p < prompt.size() ? prompt[p] : ref.next_token();
        fed.push_back(token);
        (void)ref.step(token);
        put(pre, ref.last_hidden(false));
        put(post, ref.last_hidden(true));
        std::cout << "pos " << p << " in " << token << " argmax " << ref.next_token() << '\n' << std::flush;
    }
    {
        std::ofstream ids(out_dir / "ids.txt");
        for (const auto id : fed) ids << id << ' ';
        ids << ref.next_token() << '\n';
    }
    {
        std::ofstream lat(out_dir / "latents.f32", std::ios::binary);
        for (const auto layer : kKvLayers)
            for (std::uint32_t p = 0; p < total; ++p) put(lat, ref.latent_row(layer, p));
    }
    {
        const auto& embd = weights->tensor("token_embd.weight");
        std::ofstream emb(out_dir / "embed.f32", std::ios::binary);
        auto row = [&](std::uint32_t id) {
            emb.write(reinterpret_cast<const char*>(embd.data() + std::size_t{id} * h), static_cast<std::streamsize>(h * sizeof(float)));
        };
        for (const auto id : fed) row(id);
        row(kMaskId);
    }
    {
        const auto& head = weights->tensor("output.weight");  // BF16 in the pack: the round trip is exact
        std::ofstream bf(out_dir / "head.bf16", std::ios::binary);
        std::vector<std::uint16_t> buf(h);
        for (std::size_t r = 0; r < config.vocab; ++r) {
            for (std::size_t i = 0; i < h; ++i) {
                std::uint32_t bits = 0;
                const float v = strix::ds41::round_to_bf16(head[r * h + i]);
                std::memcpy(&bits, &v, sizeof(bits));
                buf[i] = static_cast<std::uint16_t>(bits >> 16U);
            }
            bf.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(std::uint16_t)));
        }
    }
    std::ofstream manifest(out_dir / "manifest.txt");
    manifest << "positions " << total << "\nprompt " << prompt.size() << "\nhidden " << h << "\nvocab " << config.vocab
             << "\nlatent " << config.kv_lora + config.qk_rope << "\nkv_layers 25 29 33 37 41\nmask " << kMaskId << '\n';
    return 0;
}


std::vector<float> read_f32(const std::filesystem::path& path, std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    std::vector<float> v(count);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<std::size_t>(in.gcount()) != count * sizeof(float)) throw std::runtime_error("short file " + path.string());
    return v;
}

std::map<std::string, std::string> read_manifest(const std::filesystem::path& dir) {
    std::ifstream in(dir / "manifest.txt");
    std::map<std::string, std::string> m;
    std::string key, rest;
    while (in >> key && std::getline(in, rest)) m[key] = rest.substr(rest.find_first_not_of(' '));
    return m;
}

// The target side of a taps run: embedding rows of the fed ids + mask, the KV-share latent rows, the BF16 head.
class TapTarget final : public strix::hy4::dfly::Target {
public:
    TapTarget(const std::filesystem::path& dir, const std::vector<float>* head) : head_(head) {
        auto m = read_manifest(dir);
        P = static_cast<std::uint32_t>(std::stoul(m.at("positions")));
        prompt = static_cast<std::uint32_t>(std::stoul(m.at("prompt")));
        hidden_ = std::stoul(m.at("hidden"));
        latent_ = std::stoul(m.at("latent"));
        ids = read_ids((dir / "ids.txt").string());
        pre = read_f32(dir / "hidden_pre.f32", std::size_t{P} * hidden_);
        post = read_f32(dir / "hidden_post.f32", std::size_t{P} * hidden_);
        lat_ = read_f32(dir / "latents.f32", std::size_t{5} * P * latent_);
        emb_ = read_f32(dir / "embed.f32", std::size_t{P + 1U} * hidden_);
        for (std::uint32_t p = 0; p < P; ++p) row_.emplace(ids[p], p);
        row_[kMaskId] = P;
    }
    std::span<const float> embed(std::uint32_t token) override {
        return {emb_.data() + std::size_t{row_.at(token)} * hidden_, hidden_};
    }
    std::span<const float> latent(std::uint32_t layer, std::uint32_t pos) override {
        return {lat_.data() + (std::size_t{layer} * P + pos) * latent_, latent_};
    }
    void logits(std::span<const float> y, std::vector<float>& out) override {
        const std::size_t vocab = head_->size() / hidden_;
        out.assign(vocab, 0.0F);
        const std::size_t threads = std::max(1U, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        const std::size_t step = (vocab + threads - 1U) / threads;
        for (std::size_t first = 0; first < vocab; first += step)
            pool.emplace_back([&, first] {
                for (std::size_t r = first; r < std::min(vocab, first + step); ++r) {
                    float s = 0.0F;
                    const float* w = head_->data() + r * hidden_;
                    for (std::size_t i = 0; i < hidden_; ++i) s = s + w[i] * y[i];
                    out[r] = s;
                }
            });
        for (auto& t : pool) t.join();
    }
    std::uint32_t P = 0, prompt = 0;
    std::vector<std::uint32_t> ids;  // fed ids + the last argmax
    std::vector<float> pre, post;

private:
    const std::vector<float>* head_;
    std::size_t hidden_ = 0, latent_ = 0;
    std::vector<float> lat_, emb_;
    std::map<std::uint32_t, std::uint32_t> row_;
};

std::vector<float> read_head(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::vector<std::uint16_t> raw(bytes / 2U);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
    std::vector<float> v(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const std::uint32_t bits = std::uint32_t{raw[i]} << 16U;
        std::memcpy(&v[i], &bits, 4U);
    }
    return v;
}

int accept(int argc, char** argv) {
    if (argc < 4) throw std::invalid_argument("usage: strix-hy4-dfly accept SHARD TAPDIR[,TAPDIR...] [--variant V[;V...]]");
    const std::filesystem::path shard = argv[2];
    std::vector<std::string> dirs, variants{"h=post,fus=e+g,hc=off,mask=causal"};
    {
        std::stringstream in(argv[3]);
        std::string d;
        while (std::getline(in, d, ',')) dirs.push_back(d);
    }
    for (int i = 4; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--variant") {
            variants.clear();
            std::stringstream in(argv[++i]);
            std::string v;
            while (std::getline(in, v, ';')) variants.push_back(v);
        }
    }
    const auto g = strix::hy4::dfly::Geometry::official();
    auto w = strix::hy4::dfly::shard_weights(shard);
    const auto head = read_head(std::filesystem::path(dirs.front()) / "head.bf16");
    std::vector<std::unique_ptr<TapTarget>> sets;
    for (const auto& d : dirs) sets.push_back(std::make_unique<TapTarget>(d, &head));
    for (const auto& text : variants) {
        const auto v = strix::hy4::dfly::parse_variant(text);
        std::vector<std::uint32_t> hit(g.block - 1U, 0U);
        std::uint64_t accepted = 0, blocks = 0;
        double ms = 0.0;
        for (std::size_t s = 0; s < sets.size(); ++s) {
            auto& t = *sets[s];
            for (std::uint32_t p = t.prompt - 1U; p + g.block - 1U <= t.P; ++p) {
                const auto& hs = v.hidden == strix::hy4::dfly::HiddenForm::pre_norm ? t.pre : t.post;
                const std::span<const float> h(hs.data() + std::size_t{p - 1U} * g.hidden, g.hidden);
                const auto t0 = std::chrono::steady_clock::now();
                const auto b = strix::hy4::dfly::draft(g, v, *w, t, t.ids[p], p, h, g.block);
                ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::uint32_t run = 0;
                bool alive = true;
                std::cout << "block " << s << " p " << p << " drafts";
                for (std::uint32_t j = 0; j + 1U < g.block; ++j) {
                    const bool ok = b.proposals[j] == t.ids[p + 1U + j];
                    hit[j] += ok ? 1U : 0U;
                    alive = alive && ok;
                    run += alive ? 1U : 0U;
                    std::cout << ' ' << b.proposals[j] << (ok ? "+" : "-");
                }
                std::cout << " run " << run << '\n';
                accepted += run;
                ++blocks;
            }
        }
        std::cout << "E " << 1.0 + static_cast<double>(accepted) / static_cast<double>(blocks) << " blocks " << blocks << " rows";
        for (const auto x : hit) std::cout << ' ' << static_cast<double>(x) / static_cast<double>(blocks);
        std::cout << " ms_per_block " << ms / static_cast<double>(blocks) << " | " << text << '\n' << std::flush;
    }
    return 0;
}

// strix-hy4-dfly kvb PART1.gguf [PART2.gguf] --out DIR: the target's absorbed-form k_b / v_b of the KV-share layers
// (blk.L.attn_k_b.weight [H][kv_lora][nope], blk.L.attn_v_b.weight [H][v_head][kv_lora]) decoded to F32 (kvb_L.f32 =
// k_b then v_b), to compare with the drafter's own kv_b and to test the target-expanded KV-share reading.
int kvb(int argc, char** argv) {
    std::vector<std::filesystem::path> parts;
    std::filesystem::path out_dir;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else parts.emplace_back(a);
    }
    if (parts.empty() || out_dir.empty()) throw std::invalid_argument("usage: strix-hy4-dfly kvb PART1.gguf [PART2.gguf] --out DIR");
    std::filesystem::create_directories(out_dir);
    auto weights = strix::hy4::pack_weights(parts);
    for (const auto layer : kKvLayers) {
        std::ofstream out(out_dir / ("kvb_" + std::to_string(layer) + ".f32"), std::ios::binary);
        put(out, weights->tensor("blk." + std::to_string(layer) + ".attn_k_b.weight"));
        put(out, weights->tensor("blk." + std::to_string(layer) + ".attn_v_b.weight"));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "taps") return taps(argc, argv);
        if (argc >= 2 && std::string(argv[1]) == "accept") return accept(argc, argv);
        if (argc >= 2 && std::string(argv[1]) == "kvb") return kvb(argc, argv);
        std::cerr << "usage: strix-hy4-dfly taps|accept|kvb ...\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "strix-hy4-dfly: " << e.what() << '\n';
        return 1;
    }
}
