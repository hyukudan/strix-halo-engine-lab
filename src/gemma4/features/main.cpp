#include "features.hpp"
#include "gemma4/engine/runtime.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

namespace f = strix::gemma4::features;
namespace e = strix::gemma4::engine;
using Json = strix::serve::Json;
int main(int argc, char** argv) {
    try {
        std::string input = "-", output = "-", directory;
        bool resume = false;
        std::vector<char*> args{argv[0]};
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--input" || a == "--receipts" || a == "--npz-dir") {
                if (++i == argc) throw std::invalid_argument("missing feature CLI value");
                (a == "--input" ? input : a == "--receipts" ? output : directory) = argv[i];
            } else if (a == "--resume") resume = true;
            else args.push_back(argv[i]);
        }
        const auto options = e::parse_options(int(args.size()),args.data());
        if (options.help) {
            std::cout << "strix-gemma4-features --model MODEL --rope-tables TABLES [--tokenizer TOKENIZER] "
                "[--backend host|device] --input INPUT.jsonl --receipts OUTPUT.jsonl [--npz-dir DIR] [--resume]\n"
                "--print-effective-config prints the frozen feature identity without execution.\n";
            return 0;
        }
        if (resume && (input == "-" || output == "-")) throw std::invalid_argument("resume requires input and receipt files");
        if (input != "-" && output != "-" && std::filesystem::weakly_canonical(input) == std::filesystem::weakly_canonical(output))
            throw std::invalid_argument("input and receipts must be different files");
        e::Runtime runtime(options);
        const auto identity = f::backend_identity(*runtime.backend);
        if (options.inspect) { std::cout << identity.json().dump() << '\n'; return 0; }
        f::Exporter exporter(*runtime.backend,identity);
        std::unique_ptr<strix::gemma4::serve::Adapter> adapter;
        if (runtime.tokenizer) adapter = std::make_unique<strix::gemma4::serve::Adapter>(runtime.tokenizer);
        std::vector<Json> completed;
        if (resume && std::filesystem::exists(output)) {
            auto text = e::read_text(output);
            const auto complete = text.find_last_of('\n');
            const auto length = complete == std::string::npos ? 0 : complete+1;
            if (length != text.size()) { std::filesystem::resize_file(output,length); text.resize(length); }
            std::istringstream lines(text); std::string line;
            while (std::getline(lines,line)) completed.push_back(strix::serve::parse_json_unique(line));
        }
        std::ifstream in_file;
        if (input != "-") { in_file.open(input,std::ios::binary); if (!in_file) throw std::runtime_error("cannot open feature input"); }
        std::istream& in = input == "-" ? std::cin : in_file;
        std::ofstream out_file;
        if (output != "-") { out_file.open(output,std::ios::binary|(resume ? std::ios::app : std::ios::trunc)); if (!out_file) throw std::runtime_error("cannot open feature receipts"); }
        std::ostream& out = output == "-" ? std::cout : out_file;
        if (!directory.empty()) std::filesystem::create_directories(directory);
        std::size_t index = 0; std::string line;
        while (std::getline(in,line)) {
            if (line.empty()) throw std::invalid_argument("empty JSONL request");
            const auto hash = strix::gemma4::hash_bytes(line.data(),line.size());
            const auto request = f::parse_request(strix::serve::parse_json_unique(line),adapter.get());
            if (resume && (!request.prefix || request.retain_prefix || request.release_prefix))
                throw std::invalid_argument("resume requires independent prefix requests; handles are process-local");
            const auto archive = "features-"+std::to_string(index)+".npz";
            if (index < completed.size()) {
                const auto& old = completed[index];
                if (!old.find("input_hash") || old.find("input_hash")->as_string() != hash ||
                    !old.find("result") || !old.find("result")->find("identity") ||
                    old.find("result")->find("identity")->dump() != identity.json().dump() ||
                    (!directory.empty() && (!old.find("npz") || old.find("npz")->as_string() != archive ||
                        !std::filesystem::exists(std::filesystem::path(directory)/archive))))
                    throw std::invalid_argument("resume input, identity or archive mismatch");
            } else {
                const auto result = exporter.export_features(request);
                Json record(Json::Object{{"request_index",std::uint64_t(index)},{"input_hash",hash},{"result",result.json(directory.empty())}});
                if (!directory.empty()) {
                    const auto path = std::filesystem::path(directory)/archive;
                    // An orphan is a completed archive without its durable journal row.
                    if (resume && std::filesystem::exists(path)) std::filesystem::remove(path);
                    f::write_npz(path,result); record.set("npz",archive);
                }
                out << record.dump() << '\n'; out.flush(); if (!out) throw std::runtime_error("cannot write feature receipt");
            }
            ++index;
        }
        if (index < completed.size()) throw std::invalid_argument("resume input is shorter than journal");
        return 0;
    } catch (const std::exception& error) { std::cerr << "gemma4 features: " << error.what() << '\n'; return 1; }
}
