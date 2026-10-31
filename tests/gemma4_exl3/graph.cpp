#include "gemma4/exl3/format.hpp"
#include "gemma4/kernels/device.hpp"
#include "gemma4/prefill/device.hpp"
#include "gemma4/prefill/reference.hpp"
#include "../gemma4_gpu/fixture.hpp"
#include <iostream>

namespace g4 = strix::gemma4;
namespace pf = g4::prefill;
namespace {
g4::ModelManifest model() {
    auto out = fixture::tiny(6, true);
    out.identity = g4::exl3::identity; out.config.vocab = 128;
    std::mt19937 rng(1847); unsigned index = 0;
    auto quant = [&](unsigned outputs, unsigned inputs, unsigned k) {
        auto data = std::make_shared<std::vector<std::uint8_t>>(std::size_t(outputs) * inputs * k / 8 + 2ULL * (inputs + outputs));
        for (unsigned i = 0; i < inputs + outputs; ++i)
            fixture::put16(data->data() + 2ULL * i, std::uint16_t(0x2800U | ((rng() & 1U) << 15U)));
        for (std::size_t i = 2ULL * (inputs + outputs); i < data->size(); ++i) (*data)[i] = std::uint8_t(rng());
        out.storage.push_back(data);
        return g4::LinearView{data->data(), outputs, inputs, std::size_t(inputs) * k / 8, g4::WeightFormat::exl3_mul1};
    };
    for (auto& t : out.tensors) {
        if (t.shape.size() != 2) continue;
        if (t.name.find("embed_tokens") != std::string::npos) {
            t.shape[0] = out.config.vocab;
            auto data = std::make_shared<std::vector<std::uint8_t>>(fixture::packed(g4::WeightFormat::bf16,
                out.config.vocab, t.view.inputs, unsigned(rng())));
            out.storage.push_back(data);
            t.view = {data->data(), out.config.vocab, t.view.inputs, 2ULL * t.view.inputs, g4::WeightFormat::bf16};
        } else {
            const unsigned rates[] = {4, 5, 6, 8};
            t.view = quant(t.view.outputs, t.view.inputs, rates[index++ % 4]);
        }
    }
    out.tensors.push_back({"lm_head.weight", {128, out.config.hidden}, quant(128, out.config.hidden, 8), "synthetic"});
    return out;
}
void same(std::span<const float> a, std::span<const float> b, const char* stage) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size_bytes()))
        throw std::runtime_error(std::string("EXL3 graph bytes: ") + stage);
}
}
int main() {
    try {
        const auto manifest = model(); const auto weights = g4::bind_host_weights(manifest);
        fixture::Tables tables(32);
        pf::Reference host(weights, tables.view(), 32, pf::AttentionMode::canonical);
        pf::DevicePrefill device(weights, tables.view(), pf::Residency::host, false, pf::AttentionMode::canonical);
        g4::gpu::DeviceStream stream(weights.config, 32);
        const std::uint32_t tokens[] = {2, 7, 11};
        const auto reference = host.forward(tokens, true, true);
        { auto tx = stream.begin(); const auto actual = device.forward(*tx, tokens, true);
          same(actual.all_logits, reference.all_logits, "WMMA logits"); same(actual.taps, reference.taps, "WMMA taps"); tx->commit(3); }
        if (stream.digest() != host.digest()) throw std::runtime_error("EXL3 WMMA state digest");
        g4::gpu::DeviceTarget decode(weights, tables.view());
        const std::uint32_t token = 17; const auto expected = host.forward({&token, 1}, false, true);
        { auto tx = stream.begin(); g4::RowPlan row{0, 0, 3, token, tx.get(), 0, true};
          const auto actual = decode.forward_rows({&row, 1}); same(actual.logits, expected.all_logits, "decode after WMMA");
          same(actual.taps, expected.taps, "decode taps after WMMA"); tx->commit(1); }
        if (stream.digest() != host.digest()) throw std::runtime_error("EXL3 post-prefill state digest");
        std::cout << "EXL3 mixed-K sliding/global PLE/head WMMA graph and exact continuation EXACT\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
