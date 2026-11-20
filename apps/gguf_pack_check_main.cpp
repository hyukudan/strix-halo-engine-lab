// strix-gguf-pack-check: header-level check of a (split) GGUF pack against the shared tensor naming map of its architecture.
//   inspect PART... [--expect-bytes N1,N2,...] [--tsv OUT]   reads only the headers; a partial (downloading) part is accepted when
//                                                           its final size is given, tensor ranges are checked against it
//   verify  PART... [--expect-bytes N1,N2,...] [--tsv OUT]   every part complete (size == expected when given), ranges inside the file
// Both: GGUF v3 bounds (shared reader), split.* consistency, unique names across parts, every tensor claimed exactly once by the
// family table with shapes from the pack's own metadata, and byte totals per type / component / layer. Last line: PASS or FAIL.
#include "strix/gguf_reader.hpp"
#include "strix/weight/tensor_map.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace gguf = strix::gguf;
namespace tm = strix::weight::tensor_map;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::uint64_t meta_u(const gguf::FileInfo& file, const std::string& key) {
    const auto* entry = file.find_metadata(key);
    if (entry == nullptr) fail("missing " + key);
    if (entry->unsigned_value) return *entry->unsigned_value;
    if (entry->signed_value && *entry->signed_value >= 0) return static_cast<std::uint64_t>(*entry->signed_value);
    fail(key + " is not a non-negative integer");
}

std::string type_name(std::uint32_t type) {
    switch (type) {
    case 0: return "F32";
    case 1: return "F16";
    case 8: return "Q8_0";
    case 10: return "Q2_K";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 30: return "BF16";
    case 43: return "F8_E4M3";
    default: return "T" + std::to_string(type);
    }
}

std::string gib(std::uint64_t bytes) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << static_cast<double>(bytes) / static_cast<double>(1ULL << 30);
    return out.str();
}

int run(int argc, char** argv) {
    if (argc < 3) fail("usage: strix-gguf-pack-check inspect|verify PART.gguf... [--expect-bytes N1,N2,...] [--tsv OUT]");
    const std::string mode = argv[1];
    if (mode != "inspect" && mode != "verify") fail("mode must be inspect or verify");
    std::vector<std::filesystem::path> parts;
    std::vector<std::uint64_t> expected;
    std::string tsv;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--expect-bytes" && i + 1 < argc) {
            std::stringstream list(argv[++i]);
            for (std::string item; std::getline(list, item, ',');) expected.push_back(std::stoull(item));
        } else if (arg == "--tsv" && i + 1 < argc) {
            tsv = argv[++i];
        } else if (arg.rfind("--", 0) == 0) {
            fail("unknown option " + arg);
        } else {
            parts.emplace_back(arg);
        }
    }
    if (parts.empty()) fail("no GGUF part given");
    if (!expected.empty() && expected.size() != parts.size()) fail("--expect-bytes needs one size per part");

    gguf::Limits limits;
    limits.max_retained_array_elements = 4096;
    std::vector<gguf::FileInfo> files;
    bool partial = false;
    for (std::size_t p = 0; p < parts.size(); ++p) {
        const auto actual = std::filesystem::file_size(parts[p]);
        std::uint64_t extent = actual;
        if (!expected.empty()) {
            if (actual > expected[p]) fail(parts[p].filename().string() + ": larger than expected");
            if (actual < expected[p]) {
                if (mode == "verify") fail(parts[p].filename().string() + ": incomplete (" + std::to_string(actual) + " of " +
                                           std::to_string(expected[p]) + " bytes)");
                partial = true;
                extent = expected[p];  // header-only: tensor ranges are checked against the declared final size
            }
        }
        std::ifstream input(parts[p], std::ios::binary);
        if (!input) fail("cannot open " + parts[p].string());
        try {
            files.push_back(gguf::read(input, extent, limits));
        } catch (const std::exception& e) {
            fail(parts[p].filename().string() + ": " + e.what());
        }
    }

    // Split consistency: split.no = index, split.count = parts, split.tensors.count = sum, names unique across parts.
    std::uint64_t tensor_total = 0;
    for (const auto& file : files) tensor_total += file.tensors.size();
    if (files.size() > 1 || files[0].find_metadata("split.count") != nullptr) {
        for (std::size_t p = 0; p < files.size(); ++p) {
            if (meta_u(files[p], "split.no") != p) fail("split.no of part " + std::to_string(p + 1) + " is out of order");
            if (meta_u(files[p], "split.count") != files.size()) fail("split.count differs from the number of parts given");
            if (meta_u(files[p], "split.tensors.count") != tensor_total)
                fail("split.tensors.count " + std::to_string(meta_u(files[p], "split.tensors.count")) + " != " +
                     std::to_string(tensor_total) + " tensors in the parts");
        }
    }
    const auto* arch = files[0].find_metadata("general.architecture");
    if (arch == nullptr || !arch->string_value) fail("first part has no general.architecture");
    const auto* family = tm::find_family(*arch->string_value);
    if (family == nullptr) fail("no tensor naming map for architecture " + *arch->string_value);

    std::vector<gguf::TensorInfo> all;
    std::vector<std::size_t> part_of;
    std::set<std::string> names;
    for (std::size_t p = 0; p < files.size(); ++p) {
        for (auto tensor : files[p].tensors) {
            if (!names.insert(tensor.name).second) fail("tensor in two parts: " + tensor.name);
            // A Strix tile pack's 43 is not upstream F8_E4M3: keep it out of the lookup-type admission.
            if (tensor.ggml_type == gguf::kGgmlF8E4M3 && !files[p].is_f8_e4m3(tensor)) tensor.ggml_type = 0xFFFFU;
            all.push_back(std::move(tensor));
            part_of.push_back(p);
        }
    }
    tm::Model model;
    std::string error;
    if (!family->model(files[0], model, error)) fail("metadata: " + error);
    const auto result = tm::check(all, family->table, model);
    if (!result.ok()) {
        for (std::size_t i = 0; i < result.errors.size() && i < 20; ++i) std::cout << "  " << result.errors[i] << "\n";
        fail(std::to_string(result.errors.size()) + " naming-map error(s)");
    }

    std::map<std::string, std::uint64_t> by_type, by_component;
    std::map<std::int64_t, std::uint64_t> by_layer;
    std::uint64_t total = 0;
    std::ofstream out;
    if (!tsv.empty()) {
        out.open(tsv, std::ios::binary);
        if (!out) fail("cannot write " + tsv);
        out << "name\tcomponent\tlayer\ttype\tbytes\tpart\toffset\n";
    }
    for (const auto& claim : result.claims) {
        const auto& tensor = all[claim.tensor];
        const auto& entry = family->table[claim.entry];
        by_type[type_name(tensor.ggml_type)] += tensor.byte_size;
        by_component[std::string(entry.component)] += tensor.byte_size;
        by_layer[claim.layer] += tensor.byte_size;
        total += tensor.byte_size;
        if (!tsv.empty()) out << tensor.name << '\t' << entry.component << '\t' << claim.layer << '\t' << type_name(tensor.ggml_type) << '\t'
                     << tensor.byte_size << '\t' << part_of[claim.tensor] + 1 << '\t' << tensor.absolute_offset << '\n';
    }
    std::cout << "architecture " << *arch->string_value << ", blocks " << model.blocks << ", parts " << files.size()
              << ", tensors " << all.size() << ", tensor bytes " << total << " (" << gib(total) << " GiB)\n";
    std::cout << "by type GiB:";
    for (const auto& [name, bytes] : by_type) std::cout << ' ' << name << '=' << gib(bytes);
    std::cout << "\nby component GiB:";
    for (const auto& [name, bytes] : by_component) std::cout << ' ' << name << '=' << gib(bytes);
    std::cout << "\nby layer GiB (-1 = global):";
    for (const auto& [layer, bytes] : by_layer) std::cout << ' ' << layer << '=' << gib(bytes);
    std::cout << "\n";
    std::cout << "PASS " << mode << ' ' << *arch->string_value << " parts=" << files.size() << " tensors=" << all.size()
              << " bytes=" << total << (partial ? " (headers of a partial download; run verify when complete)" : "") << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cout << "FAIL " << e.what() << "\n";
        return 1;
    }
}
