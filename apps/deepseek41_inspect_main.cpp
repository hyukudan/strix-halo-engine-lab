#include "strix/deepseek41_family.hpp"
#include "strix/deepseek41_model.hpp"
#include "strix/model_family_registry.hpp"
#include "strix/model_content_hash.hpp"

#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// Validates a deepseek41 GGUF (and optionally its draft) and prints the derived contract as JSON. With --hash it
// also hashes the full files (minutes for the target) and applies the SHA-256 admission.
int main(int argc, char** argv) try {
    std::string target, draft;
    bool hash = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hash") hash = true;
        else if (arg == "--draft" && i + 1 < argc) draft = argv[++i];
        else if (target.empty() && arg.rfind("--", 0) != 0) target = arg;
        else throw std::invalid_argument("usage: strix-deepseek41-inspect TARGET.gguf [--draft DRAFT.gguf] [--hash]");
    }
    if (target.empty()) throw std::invalid_argument("usage: strix-deepseek41-inspect TARGET.gguf [--draft DRAFT.gguf] [--hash]");

    using namespace strix::deepseek41;
    register_family();
    const auto admitted = strix::admit_model_file(target);  // through the model family registry
    const auto model = validate_target(admitted.info);
    const auto& c = model.config;
    std::map<std::string, std::uint64_t> role_bytes;
    for (const auto& t : model.manifest.tensors) role_bytes[std::string(strix::to_string(t.role))] += t.byte_size;

    std::cout << "{\n  \"target\": {\"tensors\": " << model.manifest.tensors.size()
              << ", \"tensor_bytes\": " << model.manifest.total_tensor_bytes
              << ", \"file_backed_bytes\": " << model.manifest.file_backed_candidate_bytes
              << ", \"layout_identity\": " << model.manifest.layout_identity << ", \"bytes_by_role\": {";
    bool first = true;
    for (const auto& [role, bytes] : role_bytes) {
        std::cout << (first ? "" : ", ") << '"' << role << "\": " << bytes;
        first = false;
    }
    std::cout << "}},\n  \"geometry\": {\"layers\": " << c.num_hidden_layers << ", \"hidden\": " << c.hidden_size
              << ", \"vocab\": " << c.vocab_size << ", \"heads\": " << c.num_attention_heads << ", \"head_dim\": " << c.head_dim
              << ", \"experts\": " << c.n_routed_experts << ", \"top_k\": " << c.num_experts_per_tok
              << ", \"sliding_window\": " << c.sliding_window << ", \"hc_mult\": " << c.hc_mult << "},\n"
              << "  \"engram\": {\"layers\": " << model.engram.layer_ids.size() << ", \"rows_per_token_per_layer\": "
              << model.engram.rows_per_token() << ", \"row_bytes\": " << model.engram.row_bytes
              << ", \"compressed_vocab\": " << model.engram.compressed_vocab_size << "},\n  \"layers\": [\n";
    for (std::size_t l = 0; l < model.layers.size(); ++l) {
        const auto& a = model.layers[l];
        std::cout << "    {\"layer\": " << l << ", \"compress_ratio\": " << a.compress_ratio << ", \"kv_source\": " << a.kv_source
                  << ", \"index_source\": " << a.index_source << ", \"kv_from\": " << a.kv_source_layer
                  << ", \"index_from\": " << a.index_source_layer << ", \"candidate_source\": " << a.candidate_source
                  << ", \"uses_candidates\": " << a.uses_candidates << ", \"engram\": " << a.engram << '}'
                  << (l + 1 < model.layers.size() ? ",\n" : "\n");
    }
    std::cout << "  ]";
    if (!draft.empty()) {
        const auto d = validate_draft(read_file(draft), c);
        std::cout << ",\n  \"draft\": {\"tensors\": " << d.manifest.tensors.size() << ", \"tensor_bytes\": "
                  << d.manifest.total_tensor_bytes << ", \"stages\": " << d.stages << ", \"block_size\": " << d.block_size
                  << ", \"experts\": " << d.n_routed_experts << ", \"top_k\": " << d.num_experts_per_tok << '}';
    }
    if (hash) {
        const auto admit = [](const std::string& path, PinnedFileKind kind) {
            const auto digest = strix::sha256_file_content(path);
            admit_file(kind, std::filesystem::file_size(path), digest);
            return strix::model_content_identity_hex(digest);
        };
        std::cout << ",\n  \"admission\": {\"target_sha256\": \"" << admit(target, PinnedFileKind::target_q2) << '"';
        if (!draft.empty()) std::cout << ", \"draft_sha256\": \"" << admit(draft, PinnedFileKind::draft_mxfp4) << '"';
        std::cout << ", \"admitted\": true}";
    }
    std::cout << "\n}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-deepseek41-inspect: " << error.what() << '\n';
    return 1;
}
