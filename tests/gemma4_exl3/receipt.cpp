#include "gemma4/engine/coordinator.hpp"
#include "gemma4/exl3/format.hpp"
#include "gemma4/prefill/prefill.hpp"
#include "gemma4/prefill/attention.hpp"
#include <iostream>

namespace g4 = strix::gemma4;
namespace {
class Backend final : public g4::engine::Backend {
public:
    bool exl3 = true;
    const char* prefill_identity = nullptr;
    const char* prefill_attention = nullptr;
    std::unique_ptr<g4::engine::StreamState> create_stream() override { throw std::logic_error("metadata-only fixture"); }
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan>) override { throw std::logic_error("metadata-only fixture"); }
    const g4::ModelConfig& config() const override { return config_; }
    strix::serve::Json identity() const override {
        auto out = strix::serve::Json::object();
        if (exl3) out.set("identity", g4::exl3::identity);
        if (prefill_identity) out.set("prefill_identity", prefill_identity);
        if (prefill_attention) out.set("prefill_attention", prefill_attention);
        return out;
    }
private:
    g4::ModelConfig config_;
};
}
int main() {
    try {
        Backend backend; g4::engine::Request request;
        auto record = g4::engine::effective_config(backend, request, false);
        if (record.find("decode_identity")->as_string() != g4::exl3::identity) throw std::runtime_error("EXL3 receipt identity");
        backend.exl3 = false;
        record = g4::engine::effective_config(backend, request, false);
        if (record.find("decode_identity")->as_string() != g4::arithmetic_id) throw std::runtime_error("legacy metadata fallback");
        if (record.find("prefill_identity")->as_string() != "prefill-exact") throw std::runtime_error("host prefill fallback");
        // Resolved backend metadata is authoritative, including an exact override under FAST.
        for (const auto* prefill : {"prefill-exact", g4::prefill::identity, g4::prefill::v2_identity,
                                   "prefill-a8", g4::exl3::prefill_identity}) {
            backend.prefill_identity = prefill;
            record = g4::engine::effective_config(backend, request, false);
            if (record.find("prefill_identity")->as_string() != prefill) throw std::runtime_error("resolved prefill identity");
        }
        // FAST on Q4_K_M: WMMA linears and flash prompt attention remain separate identities.
        backend.prefill_identity = g4::prefill::identity;
        backend.prefill_attention = g4::prefill::flash_attention_identity;
        record = g4::engine::effective_config(backend, request, false);
        if (record.find("prefill_identity")->as_string() != "g4-wmma-pf-v1" ||
            record.find("prefill_attention")->as_string() != "g4-flash-attn-v1") throw std::runtime_error("fast prefill identities");
        std::cout << "EXL3, legacy, exact and fast prefill receipt identities PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
