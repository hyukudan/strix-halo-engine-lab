#include "strix/decode_execution_options.hpp"
#include "strix/decode_integer_head.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool value) { if (!value) throw std::runtime_error("decode execution policy assertion failed"); }
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
}
int main() try {
    strix::DecodeExecutionOptions eager;
    eager.validate(false, false);
    eager.validate(false, true);
    strix::validate_decode_prefix_roundtrip(1, 2, 2);
    rejects([] { strix::validate_decode_prefix_roundtrip(0, 2, 2); });
    rejects([] { strix::validate_decode_prefix_roundtrip(1, 2, 1); });
    rejects([] { strix::validate_decode_prefix_roundtrip(2, 2, 2); });
    auto graph = eager;
    graph.mode = strix::DecodeExecutionMode::graph;
    rejects([&] { graph.validate(true, false); });
    graph.graph_workspace_bytes = 1024;
    graph.graph_executable_bytes = 1024;
    graph.graph_capture_peak_bytes = 1024;
    graph.graph_budget_bytes = 3072;
    graph.validate(true, false);
    rejects([&] { graph.validate(false, false); });
    graph.validate(true, true);
    graph.graph_budget_bytes = 3071;
    rejects([&] { graph.validate(true, false); });
    graph.graph_budget_bytes = std::numeric_limits<std::uint64_t>::max();
    rejects([&] { graph.validate(true, false); });
    graph.graph_budget_bytes = 3072;
    graph.integer_head = strix::DecodeIntegerHead::a4_iu4;
    rejects([&] { graph.validate(true, false); });
    auto integer = eager;
    integer.integer_head = strix::DecodeIntegerHead::a8_iu8;
    rejects([&] { integer.validate(true, false); });
    integer.integer_internal_bytes = 65536;
    integer.integer_retained_bytes = 1ULL << 32U;
    integer.validate(true, false);
    integer.validate(true, true);
    rejects([&] { integer.validate(false, false); });
    integer.mode = static_cast<strix::DecodeExecutionMode>(99);
    rejects([&] { integer.validate(true, false); });
    integer = eager;
    integer.integer_head = static_cast<strix::DecodeIntegerHead>(99);
    rejects([&] { integer.validate(true, false); });
    eager.graph_workspace_bytes = 1;
    rejects([&] { eager.validate(true, false); });
    eager.graph_workspace_bytes = 0;
    eager.integer_retained_bytes = 1;
    rejects([&] { eager.validate(true, false); });
    require(strix::parse_decode_execution_mode("graph") == strix::DecodeExecutionMode::graph);
    require(strix::parse_decode_integer_head("a4-iu8") == strix::DecodeIntegerHead::a4_iu8);
    rejects([] { (void)strix::parse_decode_execution_mode("automatic"); });
    rejects([] { (void)strix::parse_decode_integer_head("a8-iu4"); });
    auto model = strix::ModelContract::qwen38_flash_next();
    strix::ModelManifest manifest;
    strix::ManifestTensor head;
    head.name = "output.weight";
    head.role = strix::TensorRole::output_head;
    head.dimensions = {model.hidden_size, model.vocab_size};
    head.ggml_type = 2;
    head.byte_size = model.vocab_size * (model.hidden_size / 32) * 18;
    manifest.tensors = {head};
    integer = {};
    integer.integer_head = strix::DecodeIntegerHead::a4_iu4;
    integer.integer_internal_bytes = strix::integer_projection::max_working_bytes;
    integer.integer_retained_bytes = 1ULL << 32U;
    const strix::integer_projection::DeviceIdentity device{0, 1, "gfx1151"};
    const auto config = strix::make_decode_integer_head_config(model, manifest, integer, device);
    require(config.input == strix::integer_projection::RuntimeInput::bf16 &&
        config.weights == strix::integer_projection::WeightEncoding::q4_0 &&
        config.instruction == strix::integer_projection::Instruction::iu4);
    manifest.tensors[0].ggml_type = 23;
    manifest.tensors[0].byte_size = model.vocab_size * (model.hidden_size / 256) * 136;
    rejects([&] { (void)strix::make_decode_integer_head_config(model, manifest, integer, device); });
    integer.integer_head = strix::DecodeIntegerHead::a4_iu8;
    require(strix::make_decode_integer_head_config(model, manifest, integer, device).weights ==
        strix::integer_projection::WeightEncoding::iq4_xs);
    model.hidden_size = 640;
    manifest.tensors[0].dimensions[0] = 640;
    rejects([&] { (void)strix::make_decode_integer_head_config(model, manifest, integer, device); });
    manifest.tensors[0].ggml_type = 2;
    manifest.tensors[0].byte_size = model.vocab_size * 20 * 18;
    (void)strix::make_decode_integer_head_config(model, manifest, integer, device);
    ++manifest.tensors[0].byte_size;
    rejects([&] { (void)strix::make_decode_integer_head_config(model, manifest, integer, device); });
    --manifest.tensors[0].byte_size;
    manifest.tensors.push_back(manifest.tensors[0]);
    rejects([&] { (void)strix::make_decode_integer_head_config(model, manifest, integer, device); });
    std::cout << "Explicit decode execution admission contracts passed; no native execution.\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
