#include "strix/deepseek41_describe.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/gguf_reader.hpp"

#include "deepseek41_real_config.inc"

#include <cstdio>
#include <fstream>
#include <map>
#include <string>

// The describe of the released model's geometry (manifest built from the real configuration with the GGUF byte sizes
// of each storage type): one anchor set per layer, the parts' bytes add up to the tensor bytes, the cone rows of the
// exact suffix replay plan, the TP replicated fraction, the pipeline cuts. With an argument the JSON is written there
// (the TensorMeld parser checks it: tensormeld.engine_describe.parse_engine_describe).
namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

strix::deepseek41::TargetModel real_geometry() {
    using namespace strix::deepseek41;
    TargetModel m;
    m.config = parse_config_json(real_config_json);
    m.layers = layer_attributes(m.config);
    m.engram.max_ngram_size = m.config.engram_max_ngram_size;
    m.engram.n_heads = m.config.engram_n_heads;
    m.engram.row_bytes = m.config.engram_head_dim + m.config.engram_head_dim / 32U;
    m.engram.layer_ids = m.config.engram_layer_ids;
    m.manifest.layout_identity = 0x1017;
    for (const auto& e : expected_target_tensors(m.config)) {
        strix::ManifestTensor t;
        t.name = e.name;
        t.dimensions = e.dimensions;
        t.ggml_type = e.ggml_type;
        const auto g = strix::gguf::tensor_type_geometry(e.ggml_type).value();
        std::uint64_t size = e.dimensions[0] / g.block_elements * g.block_bytes;
        for (std::size_t i = 1; i < e.dimensions.size(); ++i) size *= e.dimensions[i];
        t.byte_size = size;
        if (e.layer >= 0) t.layer = static_cast<std::uint32_t>(e.layer);
        m.manifest.total_tensor_bytes += size;
        m.manifest.tensors.push_back(t);
    }
    return m;
}

std::uint64_t num(const strix::serve::Json* j) { return j ? static_cast<std::uint64_t>(j->as_int()) : 0U; }

}  // namespace

int main(int argc, char** argv) try {
    using namespace strix::deepseek41;
    using strix::serve::Json;
    const auto model = real_geometry();
    EngineDescribeOptions o;
    o.engine_name = "strix-deepseek41-rank";
    o.revision = "test";
    o.executable_sha256 = std::string(64, 'a');
    o.roles = {"whole", "pipeline-stage", "prefill", "decode"};
    o.state_export = o.state_import = true;
    o.state_transports = {"tcp", "file"};
    o.parts.revision = model_revision("DeepSeek-V4.1-Flash-1017-Q2.gguf", model);
    const auto d = engine_describe(model, o);
    const auto text = d.dump();
    expect(text.size() < 4U * 1024U * 1024U, "describe below 4 MiB");
    expect(d.find("schema")->as_string() == "tensormeld/engine-describe-v1", "schema");
    const auto& parts_doc = d.find("models")->as_array().at(0);
    const auto& parts = parts_doc.find("parts")->as_array();
    expect(num(parts_doc.find("layers")) == 40U && num(parts_doc.find("stream_bytes_per_row")) == 81920U, "layers and stream");
    std::map<std::string, const Json*> by_id;
    std::uint64_t sum = 0;
    for (const auto& p : parts) {
        by_id[p.find("id")->as_string()] = &p;
        sum += num(p.find("weight_bytes")) + num(p.find("storage_bytes"));
    }
    expect(sum == model.manifest.total_tensor_bytes, "parts add up to the tensor bytes (" + std::to_string(sum) + " vs " +
                                                         std::to_string(model.manifest.total_tensor_bytes) + ")");
    for (std::uint32_t l = 0; l < 40U; ++l) {
        const std::string id = "L" + std::to_string(l);
        expect(by_id.contains(id + ".attn") && by_id.contains(id + ".mix") && by_id.contains(id + ".shared") && by_id.contains(id + ".experts"),
               "parts of layer " + std::to_string(l));
    }
    expect(by_id.contains("L1.engram_kv") && by_id.contains("L14.engram_table") && by_id.contains("L19.ced_bridge") && by_id.contains("head") &&
               by_id.contains("embed"),
           "engram, bridge, head, embed");
    const auto& ex = *by_id["L5.experts"];
    expect(num(ex.find("weight_bytes")) == 384U * 9953280U, "routed experts of a layer: 384 x 9 953 280 B");
    expect(num(ex.find("decode")->find("read_bytes")) == 6U * 9953280U, "6 experts read per row");
    const auto* fraction = by_id["L5.attn"]->find("replicated_fraction");
    expect(fraction && fraction->as_double() > 0.03 && fraction->as_double() < 0.2, "attention replicated fraction");
    std::printf("attention: %llu B per layer (L5), replicated fraction %.4f; experts %llu B; shared %llu B\n",
                static_cast<unsigned long long>(num(by_id["L5.attn"]->find("weight_bytes"))), fraction->as_double(),
                static_cast<unsigned long long>(num(ex.find("weight_bytes"))),
                static_cast<unsigned long long>(num(by_id["L5.shared"]->find("weight_bytes"))));
    // cone: encoder layers all rows, layer 38 128, layer 37 255, layer 39 the last row
    const auto rows = [&](const std::string& id) { return by_id[id]->find("prefill")->find("rows"); };
    expect(rows("L19.attn")->is_string() && rows("L19.attn")->as_string() == "all", "encoder rows");
    expect(num(rows("L38.attn")->find("max")) == 128U && num(rows("L37.attn")->find("max")) == 255U, "cone rows 128 / 255");
    expect(rows("L39.attn")->as_string() == "last", "last layer: the last row");
    expect(rows("L19.ced_bridge")->as_string() == "all", "bridge: every row");
    expect(num(by_id["L20.attn"]->find("state")->find("bytes_per_token")) == 356U && num(by_id["L2.attn"]->find("state")->find("bytes_per_token")) == 178U,
           "KV source state per token");
    std::uint64_t cut22 = 0;
    for (const auto& c : parts_doc.find("cuts")->as_array())
        if (num(c.find("after_layer")) == 22U) cut22 = num(c.find("side_bytes_per_token"));
    expect(cut22 == 96U + 356U + 2052U + 1024U, "cut after 22: mix + kv@20 + topk@20 + candidates at 8K");
    expect(parts_doc.find("cuts")->as_array().size() == 39U, "39 cuts");
    if (argc > 1) std::ofstream(argv[1]) << text << '\n';
    std::printf("describe: %zu bytes, %zu parts\n", text.size(), parts.size());
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("deepseek41 describe tests pass\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
