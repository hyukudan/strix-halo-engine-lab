#include "test.hpp"
#include "gemma4/model/model.hpp"
#include <filesystem>
#include <fstream>

using namespace strix::gemma4;
namespace {
void write(const std::filesystem::path& p, const std::string& header, const std::vector<std::uint8_t>& payload) {
    std::ofstream f(p,std::ios::binary); const auto size = std::uint64_t(header.size());
    f.write(reinterpret_cast<const char*>(&size),8); f.write(header.data(),std::streamsize(header.size()));
    f.write(reinterpret_cast<const char*>(payload.data()),std::streamsize(payload.size()));
}
}
int main() {
    const auto path = std::filesystem::current_path() / "g4-synthetic.safetensors";
    try {
        write(path,R"({"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"n":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}})",std::vector<std::uint8_t>(16));
        { auto f = read_safetensors(path); CHECK(f.tensors.size() == 2); CHECK(f.tensors[0].shape == std::vector<std::uint64_t>({2,2})); }
        for (const auto& header : {
            R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,8]}})",
            R"({"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"n":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
            R"({"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,99999]}})",
            R"({"w":{"dtype":"UNKNOWN","shape":[2,2],"data_offsets":[0,8]}})",
            R"({"w":{},"w":{}})",
            R"({"w":{"dtype":"BF16","shape":[9223372036854775807,8],"data_offsets":[0,8]}})"}) {
            write(path,header,std::vector<std::uint8_t>(16)); refuses([&] { (void)read_safetensors(path); });
        }
        ModelConfig c;
        for (unsigned l = 0; l < 42; ++l) c.layer.push_back({l%6==5 ? AttentionKind::global : AttentionKind::sliding,l%6==5 ? 512U : 256U,l<24 ? l : (l%6==5 ? 23U : 22U)});
        const auto expected = required_tensors(c); CHECK(expected.size() == 665);
        CHECK(expected.contains("model.language_model.layers.23.self_attn.k_proj.weight"));
        CHECK(!expected.contains("model.language_model.layers.24.self_attn.k_proj.weight"));
        CHECK(expected.at("model.language_model.layers.41.self_attn.o_proj.weight") == std::vector<std::uint64_t>({2560,4096}));
        unsigned frozen = 0;
        for (unsigned l = 0; l < 44; ++l) for (const auto* tail : {"k_proj.weight","v_proj.weight","k_norm.weight","v_norm.weight"})
            frozen += frozen_compatibility_tensor("model.language_model.layers." + std::to_string(l) + ".self_attn." + tail);
        CHECK(frozen == 54);
        CHECK(gguf_tensor_name("blk.23.attn_q.weight") == "model.language_model.layers.23.self_attn.q_proj.weight");
        CHECK(gguf_tensor_name("blk.0.attn_norm.weight") == "model.language_model.layers.0.input_layernorm.weight");
        CHECK(gguf_tensor_name("blk.41.layer_output_scale.weight") == "model.language_model.layers.41.layer_scalar");
        CHECK(gguf_tensor_name("per_layer_token_embd.weight") == "model.language_model.embed_tokens_per_layer.weight");
        refuses([] { (void)gguf_tensor_name("blk.0.attn_q.bias"); });
        refuses([] { (void)gguf_tensor_name("vision.weight"); });
        ModelManifest synthetic; synthetic.config = c;
        for (const auto& [name,shape] : expected) synthetic.tensors.push_back({name,shape,{},{}});
        model_detail::validate_manifest(synthetic);
        synthetic.tensors.back().shape = {1}; refuses([&] { model_detail::validate_manifest(synthetic); });
        synthetic.tensors.pop_back(); refuses([&] { model_detail::validate_manifest(synthetic); });
        synthetic.tensors.push_back(synthetic.tensors.front()); refuses([&] { model_detail::validate_manifest(synthetic); });
        CHECK(packed_row_bytes(WeightFormat::q4_k,512) == 288); refuses([] { (void)packed_row_bytes(WeightFormat::q6_k,255); });
        const std::uint32_t nan = 0x7fc00000U;
        refuses([&] { validate_finite({reinterpret_cast<const std::uint8_t*>(&nan),1,1,4,WeightFormat::f32}); });
        CHECK(hash_bytes("abc",3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        const auto config_path = std::filesystem::current_path()/"g4-config.json";
        const auto valid = read_json(std::string(G4_FIXTURE_DIR)+"/config.json");
        auto config = valid;
        const auto save = [&] { std::ofstream f(config_path); f << config.dump(); };
        save(); CHECK(parse_text_config(config_path).layer[41].producer == 23);
        config = read_json(std::string(G4_FIXTURE_DIR)+"/config.json");
        auto text = *config.find("text_config"); text.set("unknown_override",true); config.set("text_config",text);
        save(); refuses([&] { (void)parse_text_config(config_path); });
        config = read_json(std::string(G4_FIXTURE_DIR)+"/config.json");
        text = *config.find("text_config"); auto rope = *text.find("rope_parameters"); auto full = *rope.find("full_attention");
        full.set("factor",2); rope.set("full_attention",full); text.set("rope_parameters",rope); config.set("text_config",text);
        save(); refuses([&] { (void)parse_text_config(config_path); });
        std::filesystem::remove(config_path);
        std::filesystem::remove(path);
        std::cout << "manifest: defensive header, shapes, exclusions, formats and hash passed\n"; return 0;
    } catch (const std::exception& e) { std::filesystem::remove(path); std::cerr << e.what() << '\n'; return 1; }
}
