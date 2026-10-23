#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_attention.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>
#include <random>
#include <stdexcept>
#include <string>

// The attention state file: a state of the released plan with random contents survives save + load bitwise (same
// hashes and digest), a flipped byte or a truncated file is refused, and the per-token device size is the documented
// 890 B (720 B of compressed KV + 170 B of index keys) plus the rings.
namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

strix::deepseek41::AttentionState random_state(std::uint32_t position, std::mt19937& rng) {
    namespace attn = strix::ds41::attn;
    strix::deepseek41::AttentionState s;
    s.position = position;
    s.next_token = static_cast<std::uint32_t>(rng() % 129280U);
    for (std::uint32_t i = 0; i < position; ++i) s.tokens.push_back(static_cast<std::uint32_t>(rng() % 129280U));
    const auto plan = attn::released_plan();
    for (const auto& r : plan) {
        strix::deepseek41::AttentionLayerState L;
        L.owns_kv = r.owns_kv;
        L.owns_index = r.owns_index;
        L.ratio = r.ratio;
        L.ring.resize(std::size_t{strix::deepseek41::AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
        for (auto& b : L.ring) b = static_cast<std::uint8_t>(rng());
        if (r.owns_kv) {
            const std::size_t entries = position / r.ratio;
            L.kv.resize(entries * attn::kKvEntryBytes);
            L.idx_keys.resize(entries * attn::kIdxEntryBytes);
            for (auto& b : L.kv) b = static_cast<std::uint8_t>(rng());
            for (auto& b : L.idx_keys) b = static_cast<std::uint8_t>(rng());
            if (r.ratio == 2U) {
                L.carry.resize(2U * attn::kHeadDim);
                for (auto& f : L.carry) f = static_cast<float>(rng() % 1000U) * 0.001F;
                L.carry_valid = position % 2U == 1U;
            }
        }
        if (r.owns_index) {
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(position, attn::kTopK); ++i) L.topk.push_back(i * 3U);
            L.topk_known = (rng() & 1U) != 0U;
        }
        s.layers.push_back(std::move(L));
    }
    return s;
}

// The bytes of serialize_attention_state(s) built in place: plan the frame from the pieces' sizes, copy each piece to
// its offset, write the frame, seal (as a device capture does into its pinned buffer).
std::vector<std::uint8_t> in_place(const strix::deepseek41::AttentionState& s, unsigned threads, std::uint64_t& digest) {
    namespace ds = strix::deepseek41;
    ds::AttentionFrame f;
    f.position = s.position;
    f.next_token = s.next_token;
    f.token_count = s.tokens.size();
    for (const auto& L : s.layers) {
        ds::AttentionFrameLayer F;
        F.owns_kv = L.owns_kv;
        F.owns_index = L.owns_index;
        F.carry_valid = L.carry_valid;
        F.topk_known = L.topk_known;
        F.ratio = L.ratio;
        F.ring_bytes = L.ring.size();
        F.kv_bytes = L.kv.size();
        F.idx_bytes = L.idx_keys.size();
        F.carry_floats = L.carry.size();
        F.topk_count = L.topk.size();
        f.layers.push_back(F);
    }
    ds::plan_attention_frame(f);
    std::vector<std::uint8_t> out(f.total + 5U, 0xA5U);  // a larger buffer (a reused pinned one): only [0, total) is the state
    for (std::size_t l = 0; l < s.layers.size(); ++l) {
        const auto& L = s.layers[l];
        const auto& F = f.layers[l];
        const auto put = [&](std::uint64_t at, const void* p, std::size_t n) {
            if (n != 0U) std::memcpy(out.data() + at, p, n);
        };
        put(F.ring_at, L.ring.data(), L.ring.size());
        put(F.kv_at, L.kv.data(), L.kv.size());
        put(F.idx_at, L.idx_keys.data(), L.idx_keys.size());
        put(F.carry_at, L.carry.data(), L.carry.size() * 4U);
        put(F.topk_at, L.topk.data(), L.topk.size() * 4U);
    }
    ds::write_attention_frame(f, s.tokens, out);
    digest = ds::seal_attention_frame(f, out, threads).digest;
    out.resize(f.total);
    return out;
}

}  // namespace

int main() {
    try {
        namespace fs = std::filesystem;
        std::mt19937 rng(41);
        const fs::path dir = fs::temp_directory_path() / ("ds41-state-test-" + std::to_string(rng()));
        fs::create_directories(dir);
        for (const std::uint32_t position : {1U, 127U, 1001U, 70000U}) {
            const auto s = random_state(position, rng);
            const auto file = dir / ("s" + std::to_string(position) + ".state");
            strix::deepseek41::save_attention_state(file, s);
            const auto back = strix::deepseek41::load_attention_state(file);
            {  // in memory: the file's bytes; split by layer range and merged back: the same state
                namespace ds = strix::deepseek41;
                const auto bytes = ds::serialize_attention_state(s);
                std::ifstream in(file, std::ios::binary);
                const std::vector<std::uint8_t> disk{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
                expect(bytes == disk, "serialized bytes == the file's bytes");
                expect(ds::deserialize_attention_state(bytes).digest() == s.digest(), "deserialize");
                for (const unsigned threads : {1U, 3U, 0U}) {  // in place (a capture's frame) == serialize, whole and as a TAIL range
                    std::uint64_t d = 0;
                    expect(in_place(s, threads, d) == bytes && d == s.digest(), "in-place frame == serialize (threads " + std::to_string(threads) + ")");
                    const auto tail = ds::slice_attention_state(s, 21, 40);
                    expect(in_place(tail, threads, d) == ds::serialize_attention_state(tail) && d == tail.digest(), "in-place frame of a layer range");
                }
                auto bad = bytes;
                bad[bad.size() / 3U] ^= 0x01U;
                bool refused = false;
                try {
                    (void)ds::deserialize_attention_state(bad);
                } catch (const std::runtime_error&) {
                    refused = true;
                }
                expect(refused, "corrupt state bytes refused");
                for (const std::uint32_t k : {1U, 20U, 21U, 23U}) {
                    auto front = ds::slice_attention_state(s, 0, k);
                    front.layers.resize(k);  // a FRONT's export has K layers
                    front.next_token = 0;    // the FRONT has no head
                    const auto tail = ds::slice_attention_state(s, k, 40);
                    const auto merged = ds::merge_attention_states(front, tail);
                    expect(merged.digest() == s.digest() && merged.hashes() == s.hashes(), "split at " + std::to_string(k) + " and merged");
                    expect(ds::deserialize_attention_state(ds::serialize_attention_state(tail)).hashes() == tail.hashes(), "a TAIL state round trip");
                    bool overlap = false;
                    try {
                        (void)ds::merge_attention_states(s, tail);
                    } catch (const std::invalid_argument&) {
                        overlap = true;
                    }
                    expect(overlap, "overlapping layers refused");
                }
            }
            expect(back.hashes() == s.hashes() && back.digest() == s.digest() && back.tokens == s.tokens && back.next_token == s.next_token,
                   "round trip at position " + std::to_string(position));
            expect(back.digest(true) == s.digest(true) && back.digest() != 0U, "digest without top-k");
            std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);  // flip one byte in the middle
            f.seekg(static_cast<std::streamoff>(fs::file_size(file) / 2U));
            char c = 0;
            f.read(&c, 1);
            f.seekp(static_cast<std::streamoff>(fs::file_size(file) / 2U));
            c = static_cast<char>(c ^ 0x10);
            f.write(&c, 1);
            f.close();
            bool refused = false;
            try {
                (void)strix::deepseek41::load_attention_state(file);
            } catch (const std::runtime_error&) {
                refused = true;
            }
            expect(refused, "a flipped byte is refused at position " + std::to_string(position));
            fs::resize_file(file, fs::file_size(file) - 9U);
            refused = false;
            try {
                (void)strix::deepseek41::load_attention_state(file);
            } catch (const std::runtime_error&) {
                refused = true;
            }
            expect(refused, "a truncated file is refused");
        }
        fs::remove_all(dir);
        // device bytes: rings 40 x 136 x 528, plus 890 B per position (sources 2/8/14 at ratio 2, 20 at ratio 1)
        const auto rings = std::uint64_t{40} * strix::deepseek41::AttentionHost::ring_capacity * 528U;
        const auto at = [](std::uint32_t n) { return strix::deepseek41::attention_state_device_bytes(n); };
        const double per_token = static_cast<double>(at(1U << 20U) - at(1U << 19U)) / static_cast<double>(1U << 19U);
        expect(per_token > 889.9 && per_token < 890.1, "per-token device bytes " + std::to_string(per_token));
        expect(at(0) > rings && at(0) < rings + 200000U, "fixed device bytes (rings, carries, top-k)");
        std::printf("attention state: 512K positions = %.1f MiB of device state, 1M = %.1f MiB (%.1f B per token + %.2f MiB fixed)\n",
                    static_cast<double>(at(1U << 19U)) / 1048576.0, static_cast<double>(at(1U << 20U)) / 1048576.0, per_token,
                    static_cast<double>(at(0)) / 1048576.0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "deepseek41 attention state tests pass" : "FAILED");
    return failures == 0 ? 0 : 1;
}
