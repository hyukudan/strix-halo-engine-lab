#include "strix/serve/json.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>

#ifndef STRIX_WFMT_SOURCE_DIR
#error STRIX_WFMT_SOURCE_DIR must point at the audited source tree
#endif
namespace {
using strix::serve::Json;
std::string read(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("census: cannot read " + p.generic_string());
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
const Json& field(const Json& j, const char* key) {
    const auto* value = j.find(key);
    if (!value) throw std::runtime_error(std::string("census: missing field ") + key);
    return *value;
}
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("census: " + message);
}
std::set<std::string> strings(const Json& j) {
    std::set<std::string> out;
    for (const auto& v : j.as_array()) require(out.insert(v.as_string()).second, "duplicate string");
    return out;
}
}
int wfmt_census_test() {
    const std::filesystem::path root(STRIX_WFMT_SOURCE_DIR);
    const auto manifest = strix::serve::parse_json(read(root / "tests/wfmt/bindings.json"));
    require(field(manifest, "schema").as_string() == "strix.wfmt.bindings@1", "schema revision");
    const auto keywords = strings(field(manifest, "operation_scan_keywords"));
    std::set<std::string> ids, files, assigned;
    const std::regex operation(R"(\b([A-Za-z_][A-Za-z_0-9]*)\s*\()");
    for (const auto& binding : field(manifest, "bindings").as_array()) {
        const auto& id = field(binding, "id").as_string();
        require(ids.insert(id).second, "duplicate binding " + id);
        const auto& source = field(binding, "source").as_string();
        require(!std::filesystem::path(source).is_absolute() && source.find("..") == std::string::npos,
                "source must be repository relative");
        const auto text = read(root / source);
        files.insert(source);
        require(!field(binding, "owner").as_string().empty(), "unowned binding " + id);
        assigned.insert(field(binding, "migration").as_string());
        require(!field(binding, "formats").as_array().empty(), "missing storage identity " + id);
        require(!field(binding, "arithmetic").as_string().empty(), "missing arithmetic identity " + id);
        std::set<std::string> discovered;
        for (auto i = std::sregex_iterator(text.begin(), text.end(), operation); i != std::sregex_iterator(); ++i) {
            std::string name = (*i)[1].str(), lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            if (std::any_of(keywords.begin(), keywords.end(), [&](const auto& k) { return lower.find(k) != std::string::npos; }))
                discovered.insert(name);
        }
        require(!discovered.empty(), "empty operation census " + source);
        require(discovered == strings(field(binding, "operations")), "operation census changed; review bindings for " + source);
    }
    require(files == strings(field(manifest, "closure_files")), "unbound closure file");
    require(files.size() >= 89, "consumer census lost a source");
    std::set<std::string> work;
    for (const auto& item : field(manifest, "work_items").as_array()) {
        require(!field(item, "owner").as_string().empty(), "unowned migration");
        require(work.insert(field(item, "pr").as_string()).second, "duplicate work item");
    }
    for (int i = 2; i <= 14; ++i) require(work.count("P" + std::string(i < 10 ? "0" : "") + std::to_string(i)), "missing migration");
    for (const auto& pr : assigned) require(work.count(pr), "binding lacks work item " + pr);
    // Independently enumerate the actual reader switch, including parse-only and typed entries.
    const auto reader = read(root / "src/gguf_reader.cpp");
    const std::regex geometry(R"(case\s+(\d+):\s*return TensorTypeGeometry\{(\d+),\s*(\d+)\})");
    std::set<int> current, declared;
    for (auto i = std::sregex_iterator(reader.begin(), reader.end(), geometry); i != std::sregex_iterator(); ++i)
        current.insert(std::stoi((*i)[1].str()));
    for (const auto& format : field(manifest, "gguf").as_array()) {
        const int wire = int(field(format, "wire_id").as_int());
        require(declared.insert(wire).second, "duplicate wire ID");
        const auto expected = "case " + std::to_string(wire) + ": return TensorTypeGeometry{" +
            std::to_string(field(format, "block_elements").as_int()) + ", " + std::to_string(field(format, "block_bytes").as_int()) + "}";
        require(reader.find(expected) != std::string::npos, "geometry drift for " + std::to_string(wire));
        const auto state = field(format, "execution").as_string();
        require(state == "legacy-oracle" || state == "parse-only" || state == "typed-only", "storage admission class");
        if (wire == 3 || wire == 9 || wire == 15 || wire == 17 || wire == 19 || wire == 29 || wire == 34 || wire == 35)
            require(state == "parse-only", "parse recognition advertised as arithmetic");
        if (wire >= 24 && wire <= 28) require(state == "typed-only", "typed payload advertised as arithmetic");
    }
    require(current == declared, "reader has an omitted or removed storage type");
    for (const auto& removed : field(manifest, "rejected_gguf_ids").as_array())
        require(!current.count(int(removed.as_int())), "removed wire ID admitted");
    std::cout << "census: " << ids.size() << " bindings, " << files.size() << " source files, "
              << declared.size() << " GGUF geometries, " << work.size() << " owned migrations\n";
    return 0;
}
