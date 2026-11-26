// H3 text encoder host tests.
//   strix-h3-te-tests host                       NFC, pre-tokenizer, presentation helpers (no files)
//   strix-h3-te-tests tokenizer CASES.json DIR   token ids vs the reference dump (DIR: vocab.json,
//                                                merges.txt, tokenizer_config.json); skipped if DIR is empty
//   strix-h3-te-tests vision DIR                 patches / MRoPE positions / tags vs the vision fixtures
//                                                (DIR with manifest.json + <case>.*.f32); skipped if empty
#include "h3/te/te_tokenizer.hpp"
#include "h3/te/te_vision_host.hpp"

#include "strix/serve/json.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace strix::h3::te;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}

std::string hex(std::string_view s) {
    std::string out;
    char buf[8];
    for (const unsigned char c : s) { std::snprintf(buf, sizeof buf, "%02X ", c); out += buf; }
    return out;
}

void check_nfc(const char* in, const char* want, const char* name) {
    const auto got = Tokenizer::nfc(in);
    check(got == want, std::string("nfc ") + name + ": got " + hex(got) + " want " + hex(want));
}

void check_split(const char* in, std::vector<std::string> want, const char* name) {
    const auto got = Tokenizer::pre_tokenize(in);
    std::string g;
    for (const auto& p : got) g += "[" + p + "]";
    std::string w;
    for (const auto& p : want) w += "[" + p + "]";
    check(got == want, std::string("split ") + name + ": got " + g + " want " + w);
}

// ---- malformed assets and the BPE merge loop (host only, synthetic files) ----

std::string make_asset_dir(const std::string& name, const std::string& vocab, const std::string& merges, const std::string& config) {
    const auto dir = std::filesystem::temp_directory_path() / ("strix_te_tok_audit_" + name);
    std::filesystem::create_directories(dir);
    const auto put = [&](const char* file, const std::string& text) { std::ofstream(dir / file, std::ios::binary) << text; };
    put("vocab.json", vocab);
    put("merges.txt", merges);
    put("tokenizer_config.json", config);
    return dir.string();
}

// True when load throws a runtime/invalid-argument error whose text holds `needle` (not an allocation failure).
bool load_fails_cleanly(const std::string& dir, const std::string& needle) {
    try { (void)Tokenizer::load(dir); }
    catch (const std::bad_alloc&) { return false; } catch (const std::length_error&) { return false; }
    catch (const std::runtime_error& e) { return std::string(e.what()).find(needle) != std::string::npos; }
    catch (const std::invalid_argument& e) { return std::string(e.what()).find(needle) != std::string::npos; }
    catch (const std::exception&) { return false; }
    return false;
}

// The merge loop as it was before the rank-queue rewrite: the oracle.
std::vector<std::string> reference_bpe(const std::string& word, const std::unordered_map<std::string, std::uint32_t>& ranks) {
    std::vector<std::string> sym;
    for (const char ch : word) sym.emplace_back(1, ch);
    std::string key;
    while (sym.size() > 1U) {
        std::uint32_t best = UINT32_MAX;
        std::size_t at = 0;
        for (std::size_t k = 0; k + 1U < sym.size(); ++k) {
            key.assign(sym[k]); key.push_back(' '); key += sym[k + 1U];
            if (const auto r = ranks.find(key); r != ranks.end() && r->second < best) { best = r->second; at = k; }
        }
        if (best == UINT32_MAX) break;
        std::vector<std::string> merged;
        merged.reserve(sym.size());
        const std::string a = sym[at], b = sym[at + 1U];
        for (std::size_t k = 0; k < sym.size();) {
            if (k + 1U < sym.size() && sym[k] == a && sym[k + 1U] == b) { merged.push_back(a + b); k += 2U; }
            else merged.push_back(std::move(sym[k++]));
        }
        sym = std::move(merged);
    }
    return sym;
}

struct MergeTable {
    std::string vocab_json, merges_txt;
    std::unordered_map<std::string, std::uint32_t> ranks;
    std::unordered_map<std::string, std::int32_t> ids;
};

// Random (not rank-consistent) merge table over the first `letters` lowercase letters.
MergeTable random_table(std::mt19937& rng, int letters, int merges, std::size_t max_len) {
    MergeTable t;
    std::vector<std::string> symbols;
    for (int i = 0; i < letters; ++i) { symbols.emplace_back(1, static_cast<char>('a' + i)); t.ids[symbols.back()] = i; }
    std::uint32_t rank = 0;
    for (int m = 0; m < merges; ++m) {
        const auto& x = symbols[rng() % symbols.size()];
        const auto& y = symbols[rng() % symbols.size()];
        if (x.size() + y.size() > max_len) continue;
        const std::string line = x + " " + y;
        if (t.ranks.count(line)) continue;
        const std::string joined = x + y;
        t.ranks[line] = rank++;
        t.merges_txt += line + "\n";
        if (!t.ids.count(joined)) { t.ids[joined] = static_cast<std::int32_t>(t.ids.size()); symbols.push_back(joined); }
    }
    t.vocab_json = "{";
    for (const auto& [text, id] : t.ids) t.vocab_json += (t.vocab_json.size() > 1U ? ",\"" : "\"") + text + "\":" + std::to_string(id);
    t.vocab_json += "}";
    return t;
}

std::vector<std::int32_t> reference_ids(const MergeTable& t, const std::string& word) {
    std::vector<std::int32_t> out;
    for (const auto& s : reference_bpe(word, t.ranks)) out.push_back(t.ids.at(s));
    return out;
}

void malformed_assets_and_merge_loop() {
    const std::string vocab = R"({"a":0,"b":1,"ab":2})";
    const std::string config = R"({"added_tokens_decoder":{"100":{"content":"<x>"}}})";
    const std::string merges = "#version: 0.2\na b\n";
    check(Tokenizer::load(make_asset_dir("valid", vocab, merges, config)) != nullptr, "valid asset dir loads");
    struct Case { const char* name; std::string vocab, config; const char* needle; };
    const std::vector<Case> cases{
        {"trailing-backslash", "{\"a\\", config, "vocab.json"},
        {"trailing-backslash-escape-pair", "{\"a\\\\\\", config, "vocab.json"},
        {"id-above-int32", R"({"a":2147483648})", config, "token ID out of range"},
        {"id-20-digits", R"({"a":99999999999999999999})", config, "token ID out of range"},
        {"id-over-cap", R"({"a":4194304})", config, "token ID out of range"},
        {"added-no-content", vocab, R"({"added_tokens_decoder":{"5":{}}})", "added token needs content"},
        {"added-key-not-number", vocab, R"({"added_tokens_decoder":{"x":{"content":"<y>"}}})", "token ID out of range"},
        {"added-key-negative", vocab, R"({"added_tokens_decoder":{"-1":{"content":"<y>"}}})", "token ID out of range"},
        {"added-key-huge", vocab, R"({"added_tokens_decoder":{"99999999999":{"content":"<y>"}}})", "token ID out of range"},
    };
    for (const auto& c : cases)
        check(load_fails_cleanly(make_asset_dir(c.name, c.vocab, merges, c.config), c.needle), std::string("not a clean error: ") + c.name);
    {
        const auto dir = make_asset_dir("oversized", vocab, merges, config);
        std::filesystem::resize_file(std::filesystem::path(dir) / "merges.txt", (std::uintmax_t{128} << 20U) + 1U);
        check(load_fails_cleanly(dir, "size limit"), "oversized merges.txt is not a clean error");
    }
    // A negative slot size is a clean error, not a SIZE_MAX insert.
    Presented p;
    p.ids = {kVisionSlot};
    bool refused = false;
    try { (void)token_tags(p, {-1}); } catch (const std::invalid_argument&) { refused = true; } catch (const std::exception&) {}
    check(refused, "token_tags refuses a negative slot size");

    // The rank-queue merge equals the old loop on random (inconsistent) tables, for every word.
    std::mt19937 rng(20261126U);
    int words = 0;
    for (int table = 0; table < 60; ++table) {
        const auto t = random_table(rng, 2 + table % 4, 5 + static_cast<int>(rng() % 60U), 10);
        const auto tok = Tokenizer::load(make_asset_dir("table", t.vocab_json, t.merges_txt, "{}"));
        for (int w = 0; w < 40; ++w) {
            std::string word;
            for (auto n = 1U + rng() % 50U; n > 0U; --n) word.push_back(static_cast<char>('a' + static_cast<int>(rng() % static_cast<unsigned>(2 + table % 4))));
            check(tok->encode(word) == reference_ids(t, word), "merge loop differs from the reference on \"" + word + "\"");
            ++words;
        }
    }
    // One long word: cost of the old loop against the queue (printed, not asserted).
    const auto big = random_table(rng, 26, 6000, 16);
    const auto big_tok = Tokenizer::load(make_asset_dir("big", big.vocab_json, big.merges_txt, "{}"));
    for (const int length : {10000, 30000}) {
        std::string word;
        for (int i = 0; i < length; ++i) word.push_back(static_cast<char>('a' + static_cast<int>(rng() % 26U)));
        const auto t0 = std::chrono::steady_clock::now();
        const auto want = reference_ids(big, word);
        const auto t1 = std::chrono::steady_clock::now();
        const auto got = big_tok->encode(word);
        const auto t2 = std::chrono::steady_clock::now();
        check(got == want, "merge loop differs from the reference on the long word");
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::printf("bpe merge, %d-letter word: old loop %.1f ms, rank queue %.1f ms\n", length, ms(t0, t1), ms(t1, t2));
    }
    std::printf("bpe merge: %d random words equal the old loop\n", words);
}

int host() {
    check_nfc("caf\x65\xCC\x81", "caf\xC3\xA9", "e+acute");
    check_nfc("\xE2\x84\xAB", "\xC3\x85", "angstrom singleton");
    check_nfc("\xE1\x84\x80\xE1\x85\xA1", "\xEA\xB0\x80", "hangul LV");
    check_nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", "\xEA\xB0\x81", "hangul LVT");
    // a + dot below (220) + grave (230) in either order -> a-with-dot-below + grave
    check_nfc("a\xCC\x80\xCC\xA3", "\xE1\xBA\xA1\xCC\x80", "reorder+compose");
    check_nfc("\xEF\xAC\x81", "\xEF\xAC\x81", "ligature fi is compat only");
    check_nfc("\xFF\xFE", "\xFF\xFE", "invalid bytes pass through");
    check_split("Hello world", {"Hello", " world"}, "words");
    check_split("I'm it's", {"I", "'m", " it", "'s"}, "contractions");
    check_split("a  b", {"a", " ", " b"}, "double space");
    check_split("x\n\n y", {"x", "\n\n", " y"}, "newlines");
    check_split("12345", {"1", "2", "3", "4", "5"}, "digits");
    check_split("ok!!!\n", {"ok", "!!!\n"}, "punct+newline");
    check_split("end   ", {"end", "   "}, "trailing ws");
    check_split("e\xCC\x81t\xC3\xA9", {"e", "\xCC\x81t\xC3\xA9"}, "mark is not a letter");
    check(format_seconds(0.25) == "0.2", "format 0.25");
    check(format_seconds(1.25) == "1.2", "format 1.25");
    check(format_seconds(0.75) == "0.8", "format 0.75");
    check(format_seconds(2.0) == "2.0", "format 2.0");
    Presented p;
    p.ids = {5, kVisionStart, kVisionSlot, kVisionEnd, 6};
    const auto tags = token_tags(p, {3});
    check(tags == std::vector<std::int32_t>{1, 0, 0, 0, 0, 0, 1}, "token tags");
    malformed_assets_and_merge_loop();
    std::printf("host: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::int32_t> ints(const strix::serve::Json& a) {
    std::vector<std::int32_t> out;
    for (const auto& v : a.as_array()) out.push_back(static_cast<std::int32_t>(v.as_int()));
    return out;
}

std::string show(const std::vector<std::int32_t>& v) {
    std::string s;
    for (const auto x : v) s += std::to_string(x) + " ";
    return s;
}

int tokenizer(const std::string& cases_path, const std::string& dir) {
    if (dir.empty()) { std::printf("tokenizer: SKIPPED (STRIX_H3_TE_TOKENIZER_DIR not set)\n"); return 0; }
    const auto tok = Tokenizer::load(dir);
    const auto doc = strix::serve::parse_json(read_file(cases_path));
    // the dump's embedding directory holds one 3-row embedding named "fx"
    const EmbeddingLookup lookup = [](std::string_view name) { return name == "fx" ? 3 : -1; };
    int n = 0;
    for (const auto& c : doc.find("cases")->as_array()) {
        const auto& text = c.find("text")->as_string();
        const auto run = [&](const EmbeddingLookup* lk, const char* key) {
            auto got = present(*tok, text, 0, {}, lk).ids;
            const auto want = ints(*c.find(key));
            check(got == want, std::string(key) + " case \"" + text + "\": got " + show(got) + " want " + show(want));
            ++n;
        };
        run(&lookup, "ids");
        if (const auto* nd = c.find("ids_no_embedding_dir"); nd != nullptr) run(nullptr, "ids_no_embedding_dir");
    }
    for (const auto& c : doc.find("presentation")->as_array()) {
        std::vector<RefItem> items;
        for (const auto& it : c.find("items")->as_array()) {
            RefItem r;
            const auto& type = it.find("type")->as_string();
            r.kind = type == "image" ? RefItem::Kind::image : type == "audio" ? RefItem::Kind::audio : RefItem::Kind::video;
            if (r.kind == RefItem::Kind::video) {
                const auto* ts = it.find("timestamps");
                if (ts != nullptr && ts->is_array()) for (const auto& t : ts->as_array()) r.timestamps.push_back(t.as_double());
                r.frames = static_cast<int>(it.find("frames")->as_int());  // used when timestamps are absent
            }
            items.push_back(std::move(r));
        }
        const auto& prompt = c.find("prompt")->as_string();
        const auto got = present(*tok, prompt, static_cast<int>(c.find("images")->as_int()), items, &lookup).ids;
        const auto want = ints(*c.find("ids"));
        check(got == want, "presentation \"" + prompt + "\": got " + show(got) + " want " + show(want));
        ++n;
    }
    std::printf("tokenizer: %d cases, %s\n", n, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

template <typename T>
std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<std::size_t>(f.tellg());
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

int vision(const std::string& dir) {
    if (dir.empty()) { std::printf("vision: SKIPPED (no fixture dir)\n"); return 0; }
    const auto man = strix::serve::parse_json(read_file(dir + "/manifest.json"));
    int n = 0;
    for (const auto& c : man.find("cases")->as_array()) {
        const auto name = c.find("name")->as_string();
        std::vector<Patches> blocks;
        int b = 0;
        for (const auto& v : c.find("vision")->as_array()) {
            const auto base = dir + "/" + name + ".vis" + std::to_string(b);
            const auto& shape = c.find("files")->find("vis" + std::to_string(b) + ".image.f32")->find("shape")->as_array();
            Frames f;
            f.t = static_cast<int>(shape.at(0).as_int());
            f.h = static_cast<int>(shape.at(1).as_int());
            f.w = static_cast<int>(shape.at(2).as_int());
            f.rgb = read_bin<float>(base + ".image.f32");
            auto p = make_patches(f);
            const auto want = read_bin<float>(base + ".patches.f32");
            const auto& g = v.find("grid_thw")->as_array().at(0).as_array();
            check(p.grid_h == g.at(1).as_int() && p.grid_w == g.at(2).as_int(), name + " vis" + std::to_string(b) + " grid");
            check(p.data == want, name + " vis" + std::to_string(b) + " patches bitwise");
            blocks.push_back(std::move(p));
            ++b;
            ++n;
        }
        const auto ids = read_bin<std::int32_t>(dir + "/" + name + ".ids.i32");
        const auto lay = expand_layout(ids, blocks);
        const auto tags = read_bin<std::int32_t>(dir + "/" + name + ".tags.i32");
        const auto pos = read_bin<float>(dir + "/" + name + ".positions.f32");
        check(lay.tags == tags, name + " tags");
        check(lay.positions == pos, name + " MRoPE positions bitwise");
        ++n;
    }
    std::printf("vision: %d checks, %s\n", n, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "host";
    try {
        if (mode == "host") return host();
        if (mode == "tokenizer" && argc > 2) return tokenizer(argv[2], argc > 3 ? argv[3] : "");
        if (mode == "vision") return vision(argc > 2 ? argv[2] : "");
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("usage: strix-h3-te-tests host | tokenizer CASES.json DIR\n");
    return 2;
}
