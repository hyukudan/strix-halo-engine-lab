#include "strix/mtp_input_fusion.hpp"

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

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
    {
        auto model =
            strix::ModelContract::qwen38_flash_next();
        model.hidden_size = 2;
        model.hc_count = 3;

        const auto plan =
            strix::build_mtp_input_fusion_plan(
                model);
        expect(plan.hidden_size == 2 &&
                   plan.hc_count == 3 &&
                   plan.wide_hidden_size == 6,
               "MTP fusion plan preserves per-stream hidden geometry");

        const std::vector<double> embedding{
            2.0, -1.0};
        const std::vector<double> hidden{
            1.0, 2.0,
            -2.0, 3.0,
            4.0, -1.0};

        // We = [[1,2],[3,4]], Wh = [[2,0],[0,-1]].
        const std::vector<double> we{
            1.0, 2.0,
            3.0, 4.0};
        const std::vector<double> wh{
            2.0, 0.0,
            0.0, -1.0};

        const auto output =
            strix::mtp_input_fusion_reference(
                plan,
                embedding,
                hidden,
                we,
                wh);

        // We*e = [0,2].
        // Stream outputs:
        // [0,2] + [2,-2] = [2,0]
        // [0,2] + [-4,-3] = [-4,-1]
        // [0,2] + [8,1] = [8,3]
        expect(output ==
                   std::vector<double>({
                       2.0, 0.0,
                       -4.0, -1.0,
                       8.0, 3.0}),
               "MTP fusion broadcasts embedding projection and preserves independent hidden streams");
    }

    {
        auto model =
            strix::ModelContract::qwen38_flash_next();
        model.hidden_size = 2;
        model.hc_count = 3;

        const auto plan =
            strix::build_mtp_input_fusion_plan(
                model);

        const std::vector<double> embedding{
            2.0, -1.0};
        const std::vector<double> hidden{
            1.0, 2.0,
            -2.0, 3.0,
            4.0, -1.0};

        const std::vector<double> we{
            1.0, 2.0,
            3.0, 4.0};
        const std::vector<double> wh{
            2.0, 0.0,
            0.0, -1.0};

        // Row-major [H, 2H], with canonical concat order [embedding, hidden].
        const std::vector<double> eh_proj{
            1.0, 2.0, 2.0, 0.0,
            3.0, 4.0, 0.0, -1.0,
        };

        const auto split =
            strix::mtp_input_fusion_reference(
                plan,
                embedding,
                hidden,
                we,
                wh);
        const auto packed =
            strix::mtp_eh_projection_reference(
                plan,
                embedding,
                hidden,
                eh_proj);

        expect(
            packed == split &&
                plan.eh_proj_input_size == 4 &&
                plan.eh_proj_elements == 8,
            "small exactly representable real-arithmetic fixture agrees, without FP32 or BF16 equivalence claim");

        auto reversed =
            eh_proj;
        // Swap the embedding/hidden halves to prove concat ordering is semantic.
        reversed = {
            2.0, 0.0, 1.0, 2.0,
            0.0, -1.0, 3.0, 4.0,
        };
        const auto wrong_order =
            strix::mtp_eh_projection_reference(
                plan,
                embedding,
                hidden,
                reversed);
        expect(
            wrong_order != split,
            "eh_proj oracle is sensitive to embedding-before-hidden concat order");
    }

    {
        auto model =
            strix::ModelContract::qwen38_flash_next();
        model.hidden_size = 1;
        model.hc_count = 2;
        const auto plan =
            strix::build_mtp_input_fusion_plan(
                model);

        bool rejected = false;
        try {
            const std::vector<double> embedding{1.0};
            const std::vector<double> hidden{1.0};
            const std::vector<double> weight{1.0};
            static_cast<void>(
                strix::mtp_input_fusion_reference(
                    plan,
                    embedding,
                    hidden,
                    weight,
                    weight));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected,
               "MTP fusion rejects flattened hidden state with wrong stream width");
    }

    {
        auto model=strix::ModelContract::qwen38_flash_next();model.hidden_size=2;model.hc_count=2;
        const auto plan=strix::build_mtp_input_fusion_plan(model);
        const std::vector<double> embedding{0.0,0.0},hidden{1.0,1.0,3.0,3.0};
        const std::vector<double> ew{0.0,0.0},hw{0.0,1.0,0.0,-1.0};
        const auto normalized=strix::mtp_input_normalization_reference(plan,embedding,hidden,ew,hw,4.0,
            strix::MtpNormWeightEncoding::gemma_offset);
        // Whole residual mean square=(1+1+9+9)/4=5; epsilon4 gives denominator3.
        // Effective gains [1,2,1,0] give the independently derived rational values.
        expect(normalized.embedding==embedding && std::abs(normalized.wide_hidden[0]-1.0/3.0)<1e-15 &&
            std::abs(normalized.wide_hidden[1]-2.0/3.0)<1e-15 && normalized.wide_hidden[2]==1.0 && normalized.wide_hidden[3]==0.0,
            "wide RMS must combine unequal stream variances before reshaping");
        const std::vector<double> eg{1.0,1.0},hg{1.0,2.0,1.0,0.0};
        const auto converted=strix::mtp_input_normalization_reference(plan,embedding,hidden,eg,hg,4.0,
            strix::MtpNormWeightEncoding::effective_scale);
        expect(converted.wide_hidden==normalized.wide_hidden,
            "preconverted checkpoint gains must not receive Gemma plus-one twice");
        const auto double_offset=strix::mtp_input_normalization_reference(plan,embedding,hidden,eg,hg,4.0,
            strix::MtpNormWeightEncoding::gemma_offset);
        expect(double_offset.wide_hidden!=normalized.wide_hidden &&
            std::abs(normalized.wide_hidden[0]-1.0/std::sqrt(5.0))>0.1,
            "normalization fixture discriminates double offset and per-stream RMS mutants");
        bool rejected=false;
        try {(void)strix::mtp_input_normalization_reference(plan,embedding,hidden,ew,std::span<const double>(hw).first(2),4.0,
            strix::MtpNormWeightEncoding::gemma_offset);} catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"hidden normalization scale must cover the complete wide residual");
        rejected=false;try {(void)strix::mtp_input_normalization_reference(plan,embedding,hidden,ew,hw,0.0,
            strix::MtpNormWeightEncoding::effective_scale);} catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"normalization rejects unspecified or zero epsilon");
    }
    {
        auto model=strix::ModelContract::qwen38_flash_next();model.hidden_size=2;model.hc_count=2;
        const auto plan=strix::build_mtp_input_fusion_plan(model);
        const std::vector<float> embedding{1.0F,1.0F},hidden{1.0F,1.0F,1.0F,1.0F};
        constexpr float a=33554432.0F; //2^25: adding one to -a rounds back to -a in FP32.
        const std::vector<float> weights{a,0.0F,-a,1.0F, 0.0F,0.0F,0.0F,0.0F};
        const auto split=strix::mtp_input_fusion_fp32_reference(plan,embedding,hidden,weights,
            strix::MtpInputFusionArithmetic::split_projections_then_add);
        const auto packed=strix::mtp_input_fusion_fp32_reference(plan,embedding,hidden,weights,
            strix::MtpInputFusionArithmetic::packed_concat_chain);
        // Split: e=a, h=round(-a+1)=-a, e+h=0.
        // Continuous packed: ((a+0)-a)+1=1. No hardware observation is assumed.
        expect(split==std::vector<float>({0.0F,0.0F,0.0F,0.0F}) &&
            packed==std::vector<float>({1.0F,0.0F,1.0F,0.0F}),
            "separate and packed FP32 projection chains are not numerically interchangeable");
    }

    if (failures == 0) {
        std::cout << "MTP input fusion tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
