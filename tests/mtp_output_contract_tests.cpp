#include "strix/mtp_output_contract.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto contract =
        strix::build_mtp_output_structural_contract(
            model);

    expect(contract.wide_input_size == 10240 &&
               contract.mixed_hidden_size == 2560 &&
               contract.vocab_size == 248320,
           "MTP output collapses wide residual to model hidden width before logits");
    expect(contract.uses_gated_residual_head_mix &&
               !contract.head_mix_has_injection &&
               contract.lm_head_required,
           "MTP output requires non-injecting gated residual head mix followed by LM head");
    expect(contract.mixer_semantics_resolved &&
               !contract.head_binding_resolved &&
               !contract.output_semantics_resolved,
           "MTP output separates resolved mixer formula from unresolved checkpoint head binding");

    strix::validate_mtp_output_structural_contract(
        model,
        contract);

    auto overclaimed = contract;
    overclaimed.head_binding_resolved = true;
    bool rejected = false;
    try {
        strix::validate_mtp_output_structural_contract(
            model,
            overclaimed);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected,
           "MTP output validator rejects unverified LM-head binding");

    if (failures == 0) {
        std::cout << "MTP output structural contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
