#include "strix/deepseek41_model.hpp"
#include "strix/model_content_hash.hpp"

#include <bit>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_invalid(const std::function<void()>& function, const std::string& message, const std::string& needle = "") {
    try {
        function();
        expect(false, message + " (no exception)");
    } catch (const std::invalid_argument& error) {
        expect(needle.empty() || std::string(error.what()).find(needle) != std::string::npos,
               message + " (wrong reason: " + error.what() + ")");
    }
}

// ---- a tiny GGUF v3 writer ----

template <typename T>
void put(std::string& bytes, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i)
        bytes.push_back(static_cast<char>((static_cast<std::uint64_t>(value) >> (i * 8U)) & 0xffU));
}
void put_text(std::string& bytes, const std::string& value) {
    put<std::uint64_t>(bytes, value.size());
    bytes.append(value);
}

struct U32 { std::uint32_t v; };
struct F32 { float v; };
struct Bool { bool v; };
struct U32Array { std::vector<std::uint64_t> v; };
struct U64Array { std::vector<std::uint64_t> v; };
struct TextArray { std::uint64_t n; };
using Value = std::variant<std::string, U32, F32, Bool, U32Array, U64Array, TextArray>;

struct Fixture {
    std::vector<std::pair<std::string, Value>> metadata;
    std::vector<strix::deepseek41::ExpectedTensor> tensors;

    void set(const std::string& key, Value value) {
        for (auto& [k, v] : metadata)
            if (k == key) {
                v = std::move(value);
                return;
            }
        metadata.emplace_back(key, std::move(value));
    }
    void erase_tensor(const std::string& name) {
        std::erase_if(tensors, [&](const auto& t) { return t.name == name; });
    }
    strix::deepseek41::ExpectedTensor& tensor(const std::string& name) {
        for (auto& t : tensors)
            if (t.name == name) return t;
        throw std::logic_error("fixture has no tensor " + name);
    }

    std::string bytes() const {
        std::string b = "GGUF";
        put<std::uint32_t>(b, 3);
        put<std::uint64_t>(b, tensors.size());
        put<std::uint64_t>(b, metadata.size() + 1);
        put_text(b, "general.alignment");
        put<std::uint32_t>(b, 4);
        put<std::uint32_t>(b, 32);
        for (const auto& [key, value] : metadata) {
            put_text(b, key);
            if (const auto* s = std::get_if<std::string>(&value)) {
                put<std::uint32_t>(b, 8);
                put_text(b, *s);
            } else if (const auto* u = std::get_if<U32>(&value)) {
                put<std::uint32_t>(b, 4);
                put<std::uint32_t>(b, u->v);
            } else if (const auto* f = std::get_if<F32>(&value)) {
                put<std::uint32_t>(b, 6);
                put<std::uint32_t>(b, std::bit_cast<std::uint32_t>(f->v));
            } else if (const auto* o = std::get_if<Bool>(&value)) {
                put<std::uint32_t>(b, 7);
                put<std::uint8_t>(b, o->v ? 1 : 0);
            } else if (const auto* a = std::get_if<U32Array>(&value)) {
                put<std::uint32_t>(b, 9);
                put<std::uint32_t>(b, 4);
                put<std::uint64_t>(b, a->v.size());
                for (const auto x : a->v) put<std::uint32_t>(b, static_cast<std::uint32_t>(x));
            } else if (const auto* a64 = std::get_if<U64Array>(&value)) {
                put<std::uint32_t>(b, 9);
                put<std::uint32_t>(b, 10);
                put<std::uint64_t>(b, a64->v.size());
                for (const auto x : a64->v) put<std::uint64_t>(b, x);
            } else {
                const auto n = std::get<TextArray>(value).n;
                put<std::uint32_t>(b, 9);
                put<std::uint32_t>(b, 8);
                put<std::uint64_t>(b, n);
                for (std::uint64_t i = 0; i < n; ++i) put_text(b, "t" + std::to_string(i));
            }
        }
        std::uint64_t offset = 0;
        for (const auto& t : tensors) {
            put_text(b, t.name);
            put<std::uint32_t>(b, static_cast<std::uint32_t>(t.dimensions.size()));
            for (const auto d : t.dimensions) put<std::uint64_t>(b, d);
            put<std::uint32_t>(b, t.ggml_type);
            put<std::uint64_t>(b, offset);
            const auto g = strix::gguf::tensor_type_geometry(t.ggml_type).value();
            std::uint64_t size = t.dimensions[0] / g.block_elements * g.block_bytes;
            for (std::size_t i = 1; i < t.dimensions.size(); ++i) size *= t.dimensions[i];
            offset += (size + 31U) / 32U * 32U;
        }
        while (b.size() % 32 != 0) b.push_back('\0');
        b.append(static_cast<std::size_t>(offset), '\0');
        return b;
    }
    strix::gguf::FileInfo info() const {
        const auto b = bytes();
        std::istringstream input(b, std::ios::in | std::ios::binary);
        return strix::gguf::read(input, b.size(), strix::deepseek41::reader_limits());
    }
};

// A four-layer model with every attention organisation of the real one: window only (0), ratio 2 with a gated
// compressor (KV and index source 1), ratio 1 (KV, index and candidate source 2) and a candidate user (3).
std::string small_config(const std::string& override_key = "", const std::string& override_value = "") {
    std::vector<std::pair<std::string, std::string>> t = {
        {"vocab_size", "64"}, {"hidden_size", "256"}, {"moe_intermediate_size", "256"}, {"num_hidden_layers", "4"},
        {"num_attention_heads", "4"}, {"num_key_value_heads", "1"}, {"head_dim", "64"}, {"qk_rope_head_dim", "16"},
        {"q_lora_rank", "64"}, {"o_lora_rank", "32"}, {"o_groups", "2"}, {"n_routed_experts", "4"},
        {"n_shared_experts", "1"}, {"num_experts_per_tok", "2"}, {"max_position_embeddings", "4096"},
        {"sliding_window", "8"}, {"index_n_heads", "2"}, {"index_head_dim", "32"}, {"index_topk", "4"},
        {"candidate_source_layer_id", "2"}, {"candidate_topk_blocks", "4"}, {"candidate_block_size", "2"},
        {"hc_mult", "2"}, {"hc_sinkhorn_iters", "20"}, {"num_nextn_predict_layers", "1"}, {"rope_theta", "10000"},
        {"compress_rope_theta", "160000"}, {"rms_norm_eps", "1e-20"}, {"hc_eps", "1e-06"}, {"swiglu_limit", "10.0"},
        {"routed_scaling_factor", "1.5"}, {"scoring_func", "\"sqrtsoftplus\""}, {"hidden_act", "\"silu\""},
        {"topk_method", "\"noaux_tc\""}, {"norm_topk_prob", "true"},
        {"rope_scaling",
         R"({"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"rope_type":"yarn"})"},
        {"compress_ratios", "[0,2,1,1,0]"}, {"kv_source_layer_ids", "[1,2]"}, {"index_source_layer_ids", "[1,2,3]"},
        {"engram_layer_ids", "[0,3]"}, {"engram_num_embeddings", "[420,508]"}, {"engram_n_heads", "2"},
        {"engram_head_dim", "256"}, {"engram_max_ngram_size", "3"}, {"engram_vocab_size", "100"},
        {"engram_compressed_vocab_size", "40"}, {"engram_pad_token_id", "2"}, {"dspark_block_size", "3"},
        {"dspark_markov_rank", "32"}, {"dspark_n_routed_experts", "4"}, {"dspark_num_experts_per_tok", "2"},
        {"dspark_noise_token_id", "63"}, {"dspark_target_layer_ids", "[2,3]"}};
    std::string json =
        R"({"bos_token_id":0,"eos_token_id":1,"pad_token_id":2,"image_token_id":60,"vision_config":{"max_wh_ratio":null},"text_config":{)";
    bool first = true;
    for (auto& [k, v] : t) {
        if (k == override_key) v = override_value;
        json += (first ? "\"" : ",\"") + k + "\":" + v;
        first = false;
    }
    return json + "}}";
}

// Next unused primes above engram_vocab_size - 1 = 99, per (layer, n-gram size, head).
const std::vector<std::uint64_t> small_primes = {101, 103, 107, 109, 113, 127, 131, 137};
const std::vector<std::uint64_t> small_multipliers = {12345, 777, 31, 99991, 5, 1};

Fixture small_target() {
    Fixture f;
    const auto json = small_config();
    const auto c = strix::deepseek41::parse_config_json(json);
    f.set("general.architecture", std::string("deepseek41"));
    f.set("deepseek41.config", json);
    f.set("deepseek41.engram.encoding", std::string("e4m3_e8m0_32_row264"));
    f.set("deepseek41.engram.layer_ids", U32Array{{0, 3}});
    f.set("deepseek41.engram.rows", U32Array{{420, 508}});
    f.set("deepseek41.engram.compressed_vocab_size", U32{40});
    f.set("deepseek41.engram.pad_id", U32{2});
    std::vector<std::uint64_t> map;
    for (std::uint64_t i = 0; i < 64; ++i) map.push_back(i % 40U);
    f.set("deepseek41.engram.token_map", U32Array{map});
    f.set("deepseek41.engram.primes", U32Array{small_primes});
    f.set("deepseek41.engram.multipliers", U64Array{small_multipliers});
    f.set("tokenizer.ggml.tokens", TextArray{64});
    f.set("tokenizer.ggml.bos_token_id", U32{0});
    f.set("tokenizer.ggml.eos_token_id", U32{1});
    const std::vector<std::pair<const char*, std::uint32_t>> u = {
        {"vocab_size", 64}, {"hidden_size", 256}, {"moe_intermediate_size", 256}, {"num_hidden_layers", 4},
        {"num_attention_heads", 4}, {"num_key_value_heads", 1}, {"head_dim", 64}, {"qk_rope_head_dim", 16},
        {"q_lora_rank", 64}, {"o_lora_rank", 32}, {"o_groups", 2}, {"n_routed_experts", 4}, {"n_shared_experts", 1},
        {"num_experts_per_tok", 2}, {"max_position_embeddings", 4096}, {"sliding_window", 8}, {"index_n_heads", 2},
        {"index_head_dim", 32}, {"index_topk", 4}, {"candidate_source_layer_id", 2}, {"candidate_topk_blocks", 4},
        {"candidate_block_size", 2}, {"hc_mult", 2}, {"hc_sinkhorn_iters", 20}, {"rope_theta", 10000},
        {"compress_rope_theta", 160000}};
    for (const auto& [k, v] : u) f.set(std::string("deepseek41.") + k, U32{v});
    f.set("deepseek41.rms_norm_eps", F32{1e-20f});
    f.set("deepseek41.hc_eps", F32{1e-6f});
    f.set("deepseek41.swiglu_limit", F32{10.0f});
    f.set("deepseek41.routed_scaling_factor", F32{1.5f});
    f.set("deepseek41.rope_scaling.factor", F32{16.0f});
    f.set("deepseek41.rope_scaling.beta_fast", F32{32.0f});
    f.set("deepseek41.rope_scaling.beta_slow", F32{1.0f});
    f.set("deepseek41.rope_scaling.original_max_position_embeddings", F32{65536.0f});
    f.set("deepseek41.scoring_func", std::string("sqrtsoftplus"));
    f.set("deepseek41.hidden_act", std::string("silu"));
    f.set("deepseek41.topk_method", std::string("noaux_tc"));
    f.set("deepseek41.norm_topk_prob", Bool{true});
    f.set("deepseek41.compress_ratios", U32Array{{0, 2, 1, 1}});
    f.set("deepseek41.kv_source_layer_ids", U32Array{{1, 2}});
    f.set("deepseek41.index_source_layer_ids", U32Array{{1, 2, 3}});
    f.tensors = strix::deepseek41::expected_target_tensors(c);
    return f;
}

Fixture small_draft() {
    Fixture f;
    f.set("general.architecture", std::string("deepseek41-dspark"));
    f.set("dspark.block_size", U32{3});
    f.set("dspark.markov_rank", U32{32});
    f.set("dspark.noise_token_id", U32{63});
    f.set("dspark.target_layer_ids", U32Array{{2, 3}});
    f.set("dspark.n_routed_experts", U32{4});
    f.set("dspark.num_experts_per_tok", U32{2});
    f.tensors = strix::deepseek41::expected_draft_tensors(strix::deepseek41::parse_config_json(small_config()));
    return f;
}

#include "deepseek41_real_config.inc"

strix::ModelContentIdentity identity_from_hex(std::string_view hex) {
    strix::ModelContentIdentity id;
    for (std::size_t i = 0; i < 32; ++i)
        id.bytes[i] = static_cast<std::uint8_t>(std::stoul(std::string(hex.substr(2 * i, 2)), nullptr, 16));
    return id;
}

}  // namespace

int main() try {
    using namespace strix::deepseek41;

    // GGUF reader: numeric arrays are retained only on request (the default reader keeps summaries).
    {
        const auto f = small_target();
        const auto b = f.bytes();
        std::istringstream input(b, std::ios::in | std::ios::binary);
        const auto plain = strix::gguf::read(input, b.size());
        const auto* primes = plain.find_metadata("deepseek41.engram.primes");
        expect(primes != nullptr && primes->array_length == 8U && primes->unsigned_array.empty(),
               "default reader keeps array summaries only");
        const auto retained = f.info();
        expect(retained.find_metadata("deepseek41.engram.primes")->unsigned_array == small_primes,
               "deepseek41 reader retains numeric arrays");
        expect(retained.find_metadata("deepseek41.engram.multipliers")->unsigned_array == small_multipliers,
               "64-bit arrays are retained exactly");
        expect(retained.find_metadata("tokenizer.ggml.tokens")->unsigned_array.empty(), "string arrays are never retained");
    }

    // The small model validates, with the attention organisation derived from its config.
    {
        const auto model = validate_target(small_target().info());
        expect(model.manifest.tensors.size() == expected_target_tensors(model.config).size(), "small target inventory");
        expect(model.engram.rows_per_token() == 4 && model.engram.row_bytes == 264, "small Engram contract");
        const auto& L = model.layers;
        expect(L[0].compress_ratio == 0 && L[0].kv_source_layer == -1 && L[0].engram, "layer 0: window only, Engram");
        expect(L[1].kv_source && L[1].compressor_gate && L[1].index_source && !L[1].uses_candidates,
               "layer 1: gated ratio-2 source");
        expect(L[2].kv_source && !L[2].compressor_gate && L[2].candidate_source, "layer 2: ratio-1 candidate source");
        expect(!L[3].kv_source && L[3].kv_source_layer == 2 && L[3].index_source && L[3].uses_candidates,
               "layer 3: reads layer 2's KV, own indexer over the candidates");
        expect(model.manifest.file_backed_candidate_bytes == (420U + 508U) * 264U, "Engram tables are file-backed");
        expect(model.manifest.layout_identity == strix::model_manifest_layout_identity(model.manifest), "layout identity");
        const auto draft = validate_draft(small_draft().info(), model.config);
        expect(draft.manifest.tensors.size() == 2U + 25U + 4U && draft.block_size == 3, "small draft inventory");
    }

    // Inventory and metadata violations are rejected, naming the first one.
    expect_invalid([] { auto f = small_target(); f.erase_tensor("blk.1.attn_compressor_gate.weight"); (void)validate_target(f.info()); },
                   "missing tensor", "tensor count");
    expect_invalid([] {
        auto f = small_target();
        f.tensors.push_back({"blk.0.attn_compressor_gate.weight", {256, 64}, 1, strix::TensorRole::unknown,
                             strix::ResidencyPolicy::resident, 0});
        (void)validate_target(f.info());
    }, "extra tensor", "tensor count");
    expect_invalid([] { auto f = small_target(); f.tensor("blk.3.attn_kv.weight").name = "blk.3.attn_kv_extra.weight"; (void)validate_target(f.info()); },
                   "unknown tensor name", "unexpected tensor");
    expect_invalid([] { auto f = small_target(); f.tensor("blk.2.ffn_down_exps.weight").ggml_type = 12; (void)validate_target(f.info()); },
                   "wrong storage type", "storage type");
    expect_invalid([] { auto f = small_target(); f.tensor("blk.2.attn_q_b.weight").dimensions = {64, 128}; (void)validate_target(f.info()); },
                   "wrong dimensions", "dimensions");
    expect_invalid([] { auto f = small_target(); f.set("deepseek41.hidden_size", U32{512}); (void)validate_target(f.info()); },
                   "metadata/config mismatch", "hidden_size");
    expect_invalid([] { auto f = small_target(); f.set("deepseek41.compress_ratios", U32Array{{0, 2, 1, 2}}); (void)validate_target(f.info()); },
                   "compress ratio mismatch", "compress_ratios");
    expect_invalid([] {
        auto f = small_target();
        auto p = small_primes;
        p[5] = 139;
        f.set("deepseek41.engram.primes", U32Array{p});
        (void)validate_target(f.info());
    }, "prime out of order", "prime");
    expect_invalid([] {
        auto f = small_target();
        auto m = small_multipliers;
        m[2] = 30;
        f.set("deepseek41.engram.multipliers", U64Array{m});
        (void)validate_target(f.info());
    }, "even multiplier", "multiplier");
    expect_invalid([] {
        auto f = small_target();
        f.set("deepseek41.engram.token_map", U32Array{std::vector<std::uint64_t>(64, 3)});
        (void)validate_target(f.info());
    }, "compressed vocabulary not covered", "unused");
    expect_invalid([] { auto f = small_target(); f.set("deepseek41.engram.encoding", std::string("e4m3_row256")); (void)validate_target(f.info()); },
                   "unknown Engram encoding", "encoding");
    expect_invalid([] { auto f = small_target(); f.set("general.architecture", std::string("qwen4exp")); (void)validate_target(f.info()); },
                   "wrong architecture", "architecture");
    expect_invalid([] { auto f = small_draft(); f.set("dspark.block_size", U32{5}); (void)validate_draft(f.info(), parse_config_json(small_config())); },
                   "draft metadata mismatch", "draft metadata");
    expect_invalid([] { (void)parse_config_json(small_config("kv_source_layer_ids", "[2,1,]")); }, "trailing comma in JSON");
    expect_invalid([] { (void)parse_config_json(R"({"text_config":{"a":1,"a":2}})"); }, "duplicate JSON key", "duplicate");
    expect_invalid([] { (void)parse_config_json(small_config("hidden_size", "256.5")); }, "non-integer size", "hidden_size");
    expect_invalid([] { (void)parse_config_json(small_config("compress_ratios", "[0,2,1,1,2]")); }, "draft stage that compresses", "draft");
    expect_invalid([] { (void)layer_attributes(parse_config_json(small_config("compress_ratios", "[0,1,2,1,0]"))); },
                   "layer reads KV of another ratio", "ratio");

    // The admitted model's own configuration: the full inventory and attention organisation.
    {
        const auto c = parse_config_json(real_config_json);
        const auto target = expected_target_tensors(c);
        const auto draft = expected_draft_tensors(c);
        expect(target.size() == 1046 && draft.size() == 81, "real config: 1,046 target and 81 draft tensors");
        const auto L = layer_attributes(c);
        expect(L[0].compress_ratio == 0 && L[1].engram && L[14].engram, "real config: window-only layers 0-1, Engram 1/14");
        expect(L[2].compressor_gate && L[14].compressor_gate && !L[20].compressor_gate, "real config: gated ratio-2 sources");
        expect(L[19].kv_source_layer == 14 && L[21].kv_source_layer == 20 && L[39].kv_source_layer == 20,
               "real config: KV sharing");
        expect(L[23].index_source_layer == 20 && L[24].index_source && L[24].uses_candidates && L[39].index_source_layer == 36,
               "real config: index reuse and candidates");
        expect(c.engram_max_ngram_size == 4 && c.engram_n_heads == 8, "real config: 24 Engram rows per token per layer");
        bool engram_shape = false;
        for (const auto& t : target)
            if (t.name == "blk.14.engram_embd.weight")
                engram_shape = t.dimensions == std::vector<std::uint64_t>{264, 384016682} && t.ggml_type == 24 &&
                               t.residency == strix::ResidencyPolicy::file_backed_candidate;
        expect(engram_shape, "real config: Engram table shape");
    }

    // Admission is by SHA-256 and byte count of the pinned files.
    {
        const auto& pin = pinned_file(PinnedFileKind::target_q2);
        admit_file(PinnedFileKind::target_q2, pin.bytes, identity_from_hex(pin.sha256_hex));
        auto other = identity_from_hex(pin.sha256_hex);
        other.bytes[31] ^= 1U;
        expect_invalid([&] { admit_file(PinnedFileKind::target_q2, pin.bytes, other); }, "SHA-256 mismatch", "SHA-256");
        expect_invalid([&] { admit_file(PinnedFileKind::target_q2, pin.bytes - 1U, identity_from_hex(pin.sha256_hex)); },
                       "byte count mismatch", "byte count");
        expect_invalid([&] { admit_file(PinnedFileKind::draft_mxfp4, pin.bytes, identity_from_hex(pin.sha256_hex)); },
                       "the target hash does not admit the draft");
    }

    // Optional: the real files' headers (set the paths on a node).
    if (const char* path = std::getenv("STRIX_DEEPSEEK41_GGUF")) {
        const auto model = validate_target(read_file(path));
        expect(model.manifest.tensors.size() == 1046, "real file: 1,046 tensors validate");
        if (const char* draft = std::getenv("STRIX_DEEPSEEK41_DRAFT_GGUF"))
            expect(validate_draft(read_file(draft), model.config).manifest.tensors.size() == 81, "real draft file: 81 tensors validate");
        std::cout << "real file headers validated\n";
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "deepseek41 model tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
}
