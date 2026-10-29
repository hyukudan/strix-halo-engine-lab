#include "strix/model_input.hpp"
#include "strix/ds41_dspark_vocab.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace strix::ds41::dspark {
namespace {

bool exists(const std::string& path) { return std::ifstream(path, std::ios::binary).good(); }

// The union of the lists in order, first occurrence kept.
Vocab from_ids(const std::vector<std::uint32_t>& ids) {
    Vocab v;
    v.core = 0;
    add_context(v, ids);
    return v;
}

} // namespace

std::vector<std::string> vocabulary_asset_files(const std::string& spec) {
    std::vector<std::string> out;
    if (spec.starts_with("auto:"))
        throw std::runtime_error("session cache: auto vocabulary selection is not supported; use files: or full");
    if (spec.starts_with("files:")) {
        std::stringstream paths(spec.substr(6));
        for (std::string path; std::getline(paths, path, ',');) out.push_back(path);
    }
    return out;
}

std::vector<std::uint32_t> read_vocabulary_file(const std::string& path) {
    strix::ModelInput file(path);
    file.seekg(0, std::ios::end);
    if (!file) throw std::runtime_error("cannot open draft vocabulary " + path);
    const auto bytes = static_cast<std::size_t>(file.tellg());
    if (bytes % 4U != 0U) throw std::runtime_error("draft vocabulary " + path + " is not a list of uint32 ids");
    std::vector<std::uint32_t> ids(bytes / 4U);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(bytes));
    for (const auto id : ids)
        if (id >= vocab) throw std::runtime_error("draft vocabulary " + path + " has an id outside the vocabulary");
    return ids;
}

bool contains(const Vocab& vocabulary, std::uint32_t id) {
    return id < vocabulary.core || std::find(vocabulary.extra.begin(), vocabulary.extra.end(), id) != vocabulary.extra.end();
}

std::size_t add_context(Vocab& vocabulary, std::span<const std::uint32_t> ids) {
    std::vector<std::uint8_t> member(vocab, 0U);
    for (const auto id : vocabulary.extra) member[id] = 1U;
    std::size_t added = 0;
    for (const auto id : ids) {
        if (id >= vocab || id < vocabulary.core || member[id]) continue;
        member[id] = 1U;
        vocabulary.extra.push_back(id);
        ++added;
    }
    return added;
}

VocabularyChoice choose_vocabulary(const std::string& spec, std::span<const std::uint32_t> prompt) {
    VocabularyChoice out;
    if (spec.empty() || spec == "full") {
        out.label = "full";
        return out;
    }
    if (spec.rfind("files:", 0) == 0) {
        std::vector<std::uint32_t> ids;
        for (const auto& path : vocabulary_asset_files(spec)) {
            const auto part = read_vocabulary_file(path);
            ids.insert(ids.end(), part.begin(), part.end());
        }
        out.vocabulary = from_ids(ids);
        out.label = spec;
    } else if (spec.rfind("auto:", 0) == 0) {
        const auto dir = spec.substr(5);
        static constexpr const char* languages[] = {"es", "fr", "de", "it", "pt", "ru", "ko", "ja", "zh"};
        std::string best;
        std::size_t best_count = 0;
        for (const char* language : languages) {
            const auto path = dir + "/pack-" + language + "16k.bin";
            if (!exists(path)) continue;
            std::vector<std::uint8_t> member(vocab, 0U);
            for (const auto id : read_vocabulary_file(path)) member[id] = 1U;
            std::size_t count = 0;
            for (const auto t : prompt) count += t < vocab && member[t] ? 1U : 0U;
            if (count > best_count) { best_count = count; best = language; }
        }
        std::vector<std::uint32_t> ids;
        if (best.empty() || best_count < 2U || best_count * 20U < prompt.size()) {
            ids = read_vocabulary_file(dir + "/core48k.bin");
            out.label = "core48k";
        } else {
            ids = read_vocabulary_file(dir + "/core32k.bin");
            const auto pack = read_vocabulary_file(dir + "/pack-" + best + ".bin");
            ids.insert(ids.end(), pack.begin(), pack.end());
            out.label = "core32k+" + best + " (" + std::to_string(best_count) + " prompt tokens)";
        }
        out.vocabulary = from_ids(ids);
    } else {
        std::size_t used = 0;
        const unsigned long n = std::stoul(spec, &used);
        if (used != spec.size() || n == 0U || n > vocab) throw std::invalid_argument("draft vocabulary spec " + spec);
        out.vocabulary.core = static_cast<std::uint32_t>(n);
        out.label = "ids < " + spec;
        return out;
    }
    const auto before = out.vocabulary.size();
    add_context(out.vocabulary, prompt);
    out.label += " + " + std::to_string(out.vocabulary.size() - before) + " context rows = " + std::to_string(out.vocabulary.size()) + " rows";
    return out;
}

} // namespace strix::ds41::dspark
