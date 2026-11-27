// Calibration statistics from recorded input rows (any engine, any model): the rows a device tap wrote
// (DIR/index.tsv "id name outputs inputs", DIR/sites.tsv "first_id segment_ids... # width W", DIR/in-<first_id>.f32 =
// FP32 rows of that input in run order) become strix-calib-moments files (strix/weight/calib_moments.hpp), one per
// layer, with the run's token ids as its provenance window. Layer = the number after "blk." in the first consumer's
// name (GGUF naming), else layer 0. Each run is one window; pass the fit runs and the evaluation runs to separate calls.
//
// usage: strix-calib-moments-from-rows --out-prefix P [--layout full|diagonal] [--threads N]
//            --run DIR IDS [--run DIR IDS ...]      (IDS: the run's token ids, whitespace or comma separated)
// writes P.L<layer>.moments for every layer that has rows.
#include "strix/weight/calib_moments.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace calib = strix::weight::calib;

struct Site {
    std::uint32_t first = 0, width = 0;
    std::vector<std::uint32_t> segments;
};
struct Run {
    std::string dir;
    std::vector<std::uint32_t> ids;
    std::map<std::uint32_t, Site> sites;  // by first id
};

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

std::vector<std::uint32_t> read_ids(const std::string& path) {
    auto text = read_text(path);
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream in(text);
    std::vector<std::uint32_t> ids;
    for (unsigned long long v = 0; in >> v;) {
        if (v > 0xFFFFFFFFULL) throw std::runtime_error(path + ": token id out of range");
        ids.push_back(static_cast<std::uint32_t>(v));
    }
    if (ids.empty()) throw std::runtime_error(path + ": no token ids");
    return ids;
}

std::map<std::uint32_t, std::string> read_index(const std::string& dir) {
    std::istringstream in(read_text(dir + "/index.tsv"));
    std::map<std::uint32_t, std::string> names;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::uint32_t id = 0;
        std::string name;
        if (!(row >> id >> name)) throw std::runtime_error(dir + "/index.tsv: malformed row");
        if (!names.emplace(id, name).second) throw std::runtime_error(dir + "/index.tsv: duplicate id");
    }
    return names;
}

std::map<std::uint32_t, Site> read_sites(const std::string& dir) {
    std::istringstream in(read_text(dir + "/sites.tsv"));
    std::map<std::uint32_t, Site> sites;
    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash == std::string::npos) throw std::runtime_error(dir + "/sites.tsv: row without width");
        Site site;
        std::istringstream ids(line.substr(0, hash)), tail(line.substr(hash + 1U));
        for (std::uint32_t id = 0; ids >> id;) site.segments.push_back(id);
        std::string word;
        if (site.segments.empty() || !(tail >> word >> site.width) || word != "width" || site.width == 0U)
            throw std::runtime_error(dir + "/sites.tsv: malformed row");
        site.first = site.segments.front();
        sites.emplace(site.first, site);
    }
    return sites;
}

int layer_of(const std::string& name) {
    const auto at = name.find("blk.");
    if (at == std::string::npos) return 0;
    return std::stoi(name.substr(at + 4U));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string prefix;
        auto layout = calib::MomentLayout::full;
        unsigned threads = 0;
        std::vector<Run> runs;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " requires a value");
                return argv[++i];
            };
            if (arg == "--out-prefix") prefix = value();
            else if (arg == "--threads") threads = static_cast<unsigned>(std::stoul(value()));
            else if (arg == "--layout") {
                const auto v = value();
                if (v == "full") layout = calib::MomentLayout::full;
                else if (v == "diagonal") layout = calib::MomentLayout::diagonal;
                else throw std::invalid_argument("--layout full|diagonal");
            } else if (arg == "--run") {
                Run run;
                run.dir = value();
                run.ids = read_ids(value());
                runs.push_back(std::move(run));
            } else throw std::invalid_argument("unknown argument " + arg);
        }
        if (prefix.empty() || runs.empty()) throw std::invalid_argument("usage: --out-prefix P --run DIR IDS [--run DIR IDS ...]");
        const auto names = read_index(runs.front().dir);
        for (auto& run : runs) {
            if (read_index(run.dir) != names) throw std::runtime_error(run.dir + ": index differs from the first run");
            run.sites = read_sites(run.dir);
        }
        // Inputs per layer (the union over runs; a site must have one shape everywhere).
        std::map<int, std::map<std::uint32_t, Site>> layers;
        for (const auto& run : runs)
            for (const auto& [first, site] : run.sites) {
                for (const auto id : site.segments)
                    if (!names.count(id)) throw std::runtime_error(run.dir + ": site names an unknown id");
                auto& slot = layers[layer_of(names.at(first))];
                const auto [it, added] = slot.emplace(first, site);
                if (!added && (it->second.width != site.width || it->second.segments != site.segments))
                    throw std::runtime_error(run.dir + ": input " + std::to_string(first) + " differs between runs");
            }
        constexpr std::size_t kChunkRows = 64;
        for (const auto& [layer, inputs] : layers) {
            calib::MomentSet set(threads, layout);
            std::vector<std::size_t> windows;
            for (const auto& run : runs) windows.push_back(set.add_window(run.ids));
            std::size_t rows_total = 0;
            for (const auto& [first, site] : inputs) {
                std::vector<std::string> consumers;
                for (const auto id : site.segments) consumers.push_back(names.at(id));
                auto& input = set.declare("tap-in-" + std::to_string(first), site.width, consumers);
                calib::InputMoments* const targets[] = {&input};
                for (std::size_t r = 0; r < runs.size(); ++r) {
                    const auto path = runs[r].dir + "/in-" + std::to_string(first) + ".f32";
                    if (!std::filesystem::exists(path)) continue;
                    const auto bytes = std::filesystem::file_size(path);
                    const std::size_t row_bytes = std::size_t{site.width} * sizeof(float);
                    if (bytes % row_bytes) throw std::runtime_error(path + ": not whole rows");
                    std::ifstream in(path, std::ios::binary);
                    std::vector<float> chunk(kChunkRows * site.width);
                    for (std::size_t left = bytes / row_bytes; left;) {
                        const auto n = std::min(left, kChunkRows);
                        if (!in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(n * row_bytes)))
                            throw std::runtime_error(path + ": short read");
                        set.add(windows[r], targets, chunk.data(), n);
                        rows_total += n;
                        left -= n;
                    }
                }
            }
            const auto out = prefix + ".L" + std::to_string(layer) + ".moments";
            set.write(out);
            std::cout << out << ": " << inputs.size() << " inputs, " << rows_total << " rows, " << runs.size() << " windows\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "strix-calib-moments-from-rows: " << error.what() << '\n';
        return 1;
    }
}
