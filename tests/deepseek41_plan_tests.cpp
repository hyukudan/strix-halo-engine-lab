#include "strix/deepseek41_plan.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_invalid(const std::function<void()>& function, const std::string& message) {
    try {
        function();
        expect(false, message + " (no exception)");
    } catch (const std::invalid_argument&) {
    }
}

#include "deepseek41_real_config.inc"

// The admitted model's contract without the file: manifest entries straight from the expected inventory.
strix::deepseek41::TargetModel real_model() {
    using namespace strix::deepseek41;
    TargetModel m;
    m.config = parse_config_json(real_config_json);
    m.layers = layer_attributes(m.config);
    for (const auto& t : expected_target_tensors(m.config)) {
        strix::ManifestTensor mt;
        mt.name = t.name;
        mt.dimensions = t.dimensions;
        mt.ggml_type = t.ggml_type;
        mt.role = t.role;
        mt.residency = t.residency;
        m.manifest.tensors.push_back(mt);
    }
    return m;
}

std::size_t count(const strix::deepseek41::Plan& plan, strix::deepseek41::Op op, std::int32_t layer = -2) {
    return static_cast<std::size_t>(std::count_if(plan.steps.begin(), plan.steps.end(), [&](const auto& s) {
        return s.op == op && (layer == -2 || s.layer == layer);
    }));
}

}  // namespace

int main() try {
    using namespace strix::deepseek41;
    const auto model = real_model();

    // Full model: every layer, the organisation of the released config, 890 context bytes per token.
    {
        const auto plan = build_decode_plan(model, 0, 40);
        expect(count(plan, Op::attention) == 40 && count(plan, Op::routed_experts) == 40, "one attention and one MoE per layer");
        expect(count(plan, Op::engram) == 2 && count(plan, Op::engram, 1) == 1 && count(plan, Op::engram, 14) == 1, "Engram at 1 and 14");
        expect(count(plan, Op::compress) == 4 && count(plan, Op::compress, 20) == 1, "compressors at the KV sources");
        expect(count(plan, Op::index) == 8 && count(plan, Op::index, 36) == 1, "indexers at the index sources");
        expect(count(plan, Op::hc_mix) == 80 && count(plan, Op::hc_post) == 80, "two hc sites per layer");
        expect(plan.state_bytes_per_token == 890, "compressed KV + index keys = 890 B per token");
        expect(plan.fixed_state_bytes == 40U * 128U * 528U + 3U * 2U * 2U * 512U * 4U, "window rings + ratio-2 carries");
        expect(plan.steps.front().op == Op::embed && plan.steps.back().op == Op::argmax, "embed first, argmax last");
        expect(plan.expert_split == 192, "experts split in two halves");
        // the FFN collapse uses the attention site's pre; the next attention uses the FFN site's pre (carry)
        const auto it = std::find_if(plan.steps.begin(), plan.steps.end(),
                                     [](const auto& s) { return s.op == Op::hc_pre && s.site == "ffn"; });
        expect(it != plan.steps.end() && it->inputs.back() == Buffer::mix_attn, "FFN collapses with the attention site's pre");
        expect(plan.buffer_elements(Buffer::stream) == 20480 && plan.buffer_elements(Buffer::logits) == 129280 &&
                   plan.buffer_elements(Buffer::o_low) == 8192,
               "buffer geometry");
    }

    // Trimmed model of P4: layers 0-3 + head, every attention organisation of the encoder half present.
    {
        const auto plan = build_decode_plan(model, 0, 4);
        expect(count(plan, Op::engram) == 1 && count(plan, Op::compress) == 1 && count(plan, Op::index) == 1, "layers 0-3");
        expect(plan.state.size() == 4 && plan.state[2].kv_bytes_per_position == 288 && plan.state[2].index_key_bytes_per_position == 68 &&
                   plan.state[2].compress_carry_bytes == 8192 && plan.state[3].kv_bytes_per_position == 0,
               "layer 2 owns the compressed KV, layer 3 reads it");
        expect(plan.state_bytes_per_token == 178, "one ratio-2 source: (288 + 68) / 2 B per token");
        expect(count(plan, Op::head) == 1, "trimmed model keeps the head");
    }

    // A layer slice without embed/head (one rank's or one stage's part).
    {
        const auto plan = build_decode_plan(model, 20, 24, false, false);
        expect(count(plan, Op::embed) == 0 && count(plan, Op::head) == 0 && count(plan, Op::compress, 20) == 1, "slice 20-23");
    }

    expect_invalid([&] { (void)build_decode_plan(model, 4, 4); }, "empty range");
    expect_invalid([&] { (void)build_decode_plan(model, 0, 41); }, "range past the last layer");
    expect_invalid([&] {
        auto broken = model;
        for (auto& t : broken.manifest.tensors)
            if (t.name == "blk.2.attn_q_b.weight") t.ggml_type = 1;
        (void)build_decode_plan(broken, 0, 4);
    }, "weight of an unexpected type");
    expect_invalid([&] {
        auto broken = model;
        std::erase_if(broken.manifest.tensors, [](const auto& t) { return t.name == "blk.1.engram_kv.weight"; });
        (void)build_decode_plan(broken, 0, 4);
    }, "weight missing from the manifest");

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "deepseek41 plan tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
}
