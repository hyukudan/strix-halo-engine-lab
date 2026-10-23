#include "strix/ds41_prepared_load.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace strix::ds41::prepared;
void check(bool x) { if (!x) throw std::runtime_error("prepared contract failed"); }
template<class F> void rejects(F f) { bool threw = false; try { f(); } catch (const std::exception&) { threw = true; } check(threw); }
int main() {
    const auto root = std::filesystem::temp_directory_path() / ("strix-prepared-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try {
        std::vector<std::uint8_t> data((staging_slots + 1) * chunk_bytes + 713);
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>((i * 71 + i / 991) % 256);
        // Both accelerated and portable backends implement the published SHA-256 contract.
        check(digest({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        const auto first = Bytes(data).first(chunk_bytes);
        check(digest(first) == strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(first))));
        // Published device-audit formula, independent of parallel readback scheduling/slot reuse.
        std::string sequential;
        for (std::size_t offset = 0; offset < data.size(); offset += chunk_bytes)
            sequential += digest(Bytes(data).subspan(offset, std::min(chunk_bytes, data.size() - offset)));
        const auto audit_expected = digest(Bytes(reinterpret_cast<const std::uint8_t*>(sequential.data()), sequential.size()));
        {
            std::vector<std::uint8_t> scratch(staging_slots * chunk_bytes);
            Staging staging;
            for (std::size_t i = 0; i < staging.size(); ++i) staging[i] = {scratch.data() + i * chunk_bytes, chunk_bytes};
            std::size_t readback_seen = 0;
            check(chunked_digest(data.size(), [&](std::span<std::uint8_t> out, std::uint64_t offset) {
                check(offset == readback_seen);
                const auto part = Bytes(data).subspan(readback_seen, out.size());
                std::copy(part.begin(), part.end(), out.begin());
                readback_seen += out.size();
            }, staging) == audit_expected && readback_seen == data.size());
        }
        check(chunked_digest(0, [](std::span<std::uint8_t>, std::uint64_t) { check(false); }) == digest({}));
        rejects([&] { (void)chunked_digest(data.size(), [](std::span<std::uint8_t> out, std::uint64_t offset) {
            if (offset != 0) throw std::runtime_error("readback failed");
            std::fill(out.begin(), out.end(), 0);
        }); });
        const auto dir = root / "rank0";
        { Writer w(dir, "model0 overlay0 mixB tp1152 packed rank0\n");
          w.tensor("blk.0.ffn_down_exps.weight", data.size(), [&](const Sink& sink) {
              for (std::size_t o = 0; o < data.size(); o += 777777) sink(Bytes(data).subspan(o, std::min<std::size_t>(777777, data.size()-o)), o);
          }); w.commit(); }
        Reader r(dir, "model0 overlay0 mixB tp1152 packed rank0\n");
        std::size_t seen = 0;
        LoadStats stats;
        r.load("blk.0.ffn_down_exps.weight", data.size(), [&](Bytes part, std::uint64_t offset) {
            check(offset == seen && std::equal(part.begin(), part.end(), data.begin()+static_cast<std::ptrdiff_t>(seen))); seen += part.size();
        }, &stats); check(seen == data.size());
        for (const auto* ns : {"model1", "overlay1", "mixA", "tp1024", "dup", "rank1", "ep"}) rejects([&] { Reader bad(dir, ns); });
        // A quoted source name may contain newlines and the delimiter's literal text.
        // Keep accepting exactly matching v1 namespaces, including this case.
        {
            const std::string embedded_namespace = "model \"source\nEND-NAMESPACE\nfile\" hash\n";
            Writer embedded_writer(root / "embedded", embedded_namespace);
            embedded_writer.tensor("tiny", 1, [](const Sink& sink) {
                const std::uint8_t value = 7; sink(Bytes(&value, 1), 0);
            });
            embedded_writer.commit();
            Reader embedded_reader(root / "embedded", embedded_namespace);
            embedded_reader.load("tiny", 1, [](Bytes part, std::uint64_t offset) {
                check(offset == 0 && part.size() == 1 && part[0] == 7);
            });
        }
        rejects([&] { r.load("missing", 1, [](Bytes, std::uint64_t) {}); });
        rejects([&] { r.load("blk.0.ffn_down_exps.weight", data.size()-1, [](Bytes, std::uint64_t) {}); });
        // Consumer exceptions must join all in-flight reads/verifications before staging is freed.
        rejects([&] { r.load("blk.0.ffn_down_exps.weight", data.size(), [](Bytes, std::uint64_t) { throw std::runtime_error("consumer failed"); }); });
        Staging invalid;
        std::vector<std::uint8_t> unaligned(chunk_bytes + 1);
        for (auto& slot : invalid) slot = {unaligned.data() + 1, chunk_bytes};
        rejects([&] { r.load("blk.0.ffn_down_exps.weight", data.size(), [](Bytes, std::uint64_t) {}, nullptr, invalid); });
        // Version rejection still applies when the manifest's outer checksum is valid.
        const auto manifest = dir / "manifest";
        std::ifstream mf(manifest, std::ios::binary);
        const std::string original{std::istreambuf_iterator<char>(mf), {}}; mf.close();
        auto body = original.substr(65); body.replace(0, 16, "STRIX-PREPARED-9\n");
        { std::ofstream f(manifest, std::ios::binary); f << digest(Bytes(reinterpret_cast<const std::uint8_t*>(body.data()), body.size())) << '\n' << body; }
        rejects([&] { Reader bad(dir, "model0 overlay0 mixB tp1152 packed rank0\n"); });
        { std::ofstream f(manifest, std::ios::binary); f << original; }
        // Corruption after one consumed chunk forces the integration caller to overwrite the entire tensor.
        { std::fstream f(dir / "blk.0.ffn_down_exps.weight.bin", std::ios::binary|std::ios::in|std::ios::out); f.seekp(chunk_bytes+12); f.put(static_cast<char>(data[chunk_bytes+12]^1)); }
        seen = 0;
        rejects([&] { r.load("blk.0.ffn_down_exps.weight", data.size(), [&](Bytes b, std::uint64_t) { seen += b.size(); }); });
        check(seen == chunk_bytes);
        { std::fstream f(dir / "manifest", std::ios::binary|std::ios::in|std::ios::out); f.seekp(80); f.put('!'); }
        rejects([&] { Reader bad(dir, "model0 overlay0 mixB tp1152 packed rank0\n"); });
        { Writer interrupted(root / "aborted", "ns\n"); rejects([&] { interrupted.tensor("../bad", 1, [](const Sink&) {}); }); }
        check(!std::filesystem::exists(root / "aborted") && !std::filesystem::exists(root / "aborted.incomplete"));
        rejects([&] { Writer duplicate(dir, "ns\n"); });
        // Explicit application options must bind the effective dense context, independent of getenv.
        const auto model = root / "model.bin", pack_a = root / "pack-a", pack_b = root / "pack-b";
        { std::ofstream f(model); f << "model"; }
        for (const auto& pack : {pack_a, pack_b}) {
            std::filesystem::create_directory(pack);
            std::ofstream f(pack / "dense.head.safetensors"); f << pack.filename().string();
        }
        strix::deepseek41::TargetModel target;
        DenseBinding binding{pack_a.string(), "", "head", "1"};
        const auto ns_a = name_space(model, {}, target, 0, 2, true, true, binding);
        // A matching EP pack is accepted; a TP pack cannot be mistaken for EP (or another rank/layout).
        const auto ns_ep = name_space(model, {}, target, 0, 2, false, true, binding);
        { Writer ep(root / "ep", ns_ep); ep.tensor("expert", 1, [](const Sink& sink) {
              const std::uint8_t b = 7; sink(Bytes(&b, 1), 0);
          }); ep.commit(); }
        Reader ep(root / "ep", ns_ep);
        const auto mismatch = [&](const std::string& expected, const std::string& field) {
            try { Reader bad(root / "ep", expected); check(false); }
            catch (const std::runtime_error& e) {
                const std::string why = e.what();
                check(why.find("namespace mismatch at line") != std::string::npos &&
                      why.find("expected") != std::string::npos && why.find("artifact") != std::string::npos &&
                      why.find(field) != std::string::npos);
            }
        };
        mismatch(ns_a, "identity ds41-expert-tp2-i1152-v1");
        mismatch(name_space(model, {}, target, 1, 2, false, true, binding), "rank 1 world 2");
        mismatch(name_space(model, {}, target, 0, 2, false, false, binding), "down dup");
        binding.trellis = pack_b.string();
        check(ns_a != name_space(model, {}, target, 0, 2, true, true, binding));
        binding.trellis = pack_a.string();
        const auto mapping = root / "mix.txt";
        { std::ofstream f(mapping); f << "# ignored=missing\r\nhead = " << pack_b.string() << " \t\r\n"; }
        binding.mix = mapping.string();
        const auto ns_mix = name_space(model, {}, target, 0, 2, true, true, binding);
        check(ns_mix != ns_a && ns_mix.find("pack-b") != std::string::npos);
        std::filesystem::remove_all(root);
        std::cout << digest_backend() << ": prepared load roundtrip, slot reuse, SHA256 backend/readback identity, EP acceptance and precise namespace/geometry/version/corruption/partial-load/atomic publication PASS\n";
        return 0;
    } catch (...) { std::filesystem::remove_all(root); throw; }
}
