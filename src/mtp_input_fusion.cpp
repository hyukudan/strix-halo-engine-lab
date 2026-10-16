#include "strix/mtp_input_fusion.hpp"

#include <limits>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "MTP input-fusion geometry overflow");
    }
    return left * right;
}

}  // namespace

MtpInputFusionPlan
build_mtp_input_fusion_plan(
    const ModelContract& model) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (model.hc_count <= 1) {
        throw std::invalid_argument(
            "MTP residual-linear shared fusion requires hc_count > 1");
    }

    MtpInputFusionPlan plan;
    plan.hidden_size =
        model.hidden_size;
    plan.hc_count =
        model.hc_count;
    plan.wide_hidden_size =
        checked_mul(
            plan.hidden_size,
            plan.hc_count);
    plan.fc_embedding_elements =
        checked_mul(
            plan.hidden_size,
            plan.hidden_size);
    plan.fc_hidden_elements =
        plan.fc_embedding_elements;
    plan.eh_proj_input_size =
        checked_mul(
            plan.hidden_size,
            2ULL);
    plan.eh_proj_elements =
        checked_mul(
            plan.hidden_size,
            plan.eh_proj_input_size);
    return plan;
}

MtpNormalizedInputReference mtp_input_normalization_reference(const MtpInputFusionPlan& p,
    std::span<const double> embedding,std::span<const double> hidden,
    std::span<const double> ew,std::span<const double> hw,double epsilon,MtpNormWeightEncoding encoding) {
    if(!p.hidden_size || p.hc_count<=1U || p.wide_hidden_size!=checked_mul(p.hidden_size,p.hc_count) ||
       embedding.size()!=p.hidden_size || hidden.size()!=p.wide_hidden_size || ew.size()!=embedding.size() || hw.size()!=hidden.size() ||
       !std::isfinite(epsilon) || epsilon<=0.0 ||
       (encoding!=MtpNormWeightEncoding::gemma_offset && encoding!=MtpNormWeightEncoding::effective_scale))
        throw std::invalid_argument("MTP normalization reference requires complete wide geometry, positive epsilon and explicit weight encoding");
    const auto normalize=[&](std::span<const double> x,std::span<const double> w) {
        double squares=0.0;
        for(auto value:x) {if(!std::isfinite(value)) throw std::invalid_argument("MTP normalization input must be finite");squares+=value*value;}
        const auto denominator=std::sqrt(squares/static_cast<double>(x.size())+epsilon);
        if(!std::isfinite(denominator)) throw std::overflow_error("MTP normalization reference squared norm overflow");
        std::vector<double> out(x.size());
        for(std::size_t i=0;i<x.size();++i) {
            if(!std::isfinite(w[i])) throw std::invalid_argument("MTP normalization weight must be finite");
            const auto gain=encoding==MtpNormWeightEncoding::gemma_offset?1.0+w[i]:w[i];
            out[i]=(x[i]/denominator)*gain;
            if(!std::isfinite(out[i])) throw std::overflow_error("MTP normalization reference output overflow");
        }
        return out;
    };
    return {normalize(embedding,ew),normalize(hidden,hw)};
}
std::vector<float> mtp_input_fusion_fp32_reference(const MtpInputFusionPlan& p,
    std::span<const float> embedding,std::span<const float> hidden,std::span<const float> weights,
    MtpInputFusionArithmetic arithmetic) {
    if(!p.hidden_size || p.hc_count<=1U || p.wide_hidden_size!=checked_mul(p.hidden_size,p.hc_count) ||
       p.eh_proj_input_size!=checked_mul(p.hidden_size,2U) || p.eh_proj_elements!=checked_mul(p.hidden_size,p.eh_proj_input_size) ||
       embedding.size()!=p.hidden_size || hidden.size()!=p.wide_hidden_size || weights.size()!=p.eh_proj_elements ||
       (arithmetic!=MtpInputFusionArithmetic::split_projections_then_add && arithmetic!=MtpInputFusionArithmetic::packed_concat_chain))
        throw std::invalid_argument("MTP FP32 fusion reference geometry or profile is invalid");
    for(const auto values:{embedding,hidden,weights})
        if(!std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}))
            throw std::invalid_argument("MTP FP32 fusion reference inputs must be finite");
    const auto h=static_cast<std::size_t>(p.hidden_size);
    std::vector<float> out(hidden.size());
    for(std::size_t stream=0;stream<static_cast<std::size_t>(p.hc_count);++stream)
        for(std::size_t row=0;row<h;++row) {
            float e=0.0F;
            for(std::size_t col=0;col<h;++col) e=std::fma(weights[row*2U*h+col],embedding[col],e);
            float v=arithmetic==MtpInputFusionArithmetic::packed_concat_chain?e:0.0F;
            for(std::size_t col=0;col<h;++col) v=std::fma(weights[row*2U*h+h+col],hidden[stream*h+col],v);
            out[stream*h+row]=arithmetic==MtpInputFusionArithmetic::packed_concat_chain?v:e+v;
        }
    return out;
}

std::vector<double>
mtp_eh_projection_reference(
    const MtpInputFusionPlan& plan,
    std::span<const double> normalized_embedding,
    std::span<const double> normalized_wide_hidden,
    std::span<const double> eh_proj_row_major) {
    if (plan.hidden_size == 0 ||
        plan.hc_count <= 1 ||
        plan.wide_hidden_size !=
            plan.hidden_size * plan.hc_count ||
        plan.eh_proj_input_size !=
            plan.hidden_size * 2ULL ||
        normalized_embedding.size() !=
            plan.hidden_size ||
        normalized_wide_hidden.size() !=
            plan.wide_hidden_size ||
        eh_proj_row_major.size() !=
            plan.eh_proj_elements) {
        throw std::invalid_argument(
            "MTP eh_proj buffers do not match canonical geometry");
    }

    const auto hidden =
        static_cast<std::size_t>(
            plan.hidden_size);
    const auto streams =
        static_cast<std::size_t>(
            plan.hc_count);
    const auto input_width =
        static_cast<std::size_t>(
            plan.eh_proj_input_size);

    std::vector<double> output(
        normalized_wide_hidden.size(),
        0.0);

    for (std::size_t stream = 0;
         stream < streams;
         ++stream) {
        const auto hidden_base =
            stream * hidden;
        for (std::size_t row = 0;
             row < hidden;
             ++row) {
            double sum = 0.0;

            // Packed storage is [normalized token, slice of WIDE-normalized hidden].
            // This continuous reduction is not a split-projection rounding proof.
            for (std::size_t column = 0;
                 column < hidden;
                 ++column) {
                sum +=
                    eh_proj_row_major[
                        row * input_width +
                        column] *
                    normalized_embedding[column];
            }
            for (std::size_t column = 0;
                 column < hidden;
                 ++column) {
                sum +=
                    eh_proj_row_major[
                        row * input_width +
                        hidden +
                        column] *
                    normalized_wide_hidden[
                        hidden_base + column];
            }

            output[
                hidden_base + row] =
                sum;
        }
    }
    return output;
}

std::vector<double>
mtp_input_fusion_reference(
    const MtpInputFusionPlan& plan,
    std::span<const double> normalized_embedding,
    std::span<const double> normalized_wide_hidden,
    std::span<const double> fc_embedding_row_major,
    std::span<const double> fc_hidden_row_major) {
    if (plan.hidden_size == 0 ||
        plan.hc_count <= 1 ||
        plan.wide_hidden_size !=
            plan.hidden_size * plan.hc_count ||
        normalized_embedding.size() !=
            plan.hidden_size ||
        normalized_wide_hidden.size() !=
            plan.wide_hidden_size ||
        fc_embedding_row_major.size() !=
            plan.fc_embedding_elements ||
        fc_hidden_row_major.size() !=
            plan.fc_hidden_elements) {
        throw std::invalid_argument(
            "MTP input-fusion buffers do not match plan geometry");
    }

    const auto hidden =
        static_cast<std::size_t>(
            plan.hidden_size);
    const auto streams =
        static_cast<std::size_t>(
            plan.hc_count);

    std::vector<double> embedding_projection(
        hidden,
        0.0);
    for (std::size_t row = 0;
         row < hidden;
         ++row) {
        double sum = 0.0;
        for (std::size_t column = 0;
             column < hidden;
             ++column) {
            sum +=
                fc_embedding_row_major[
                    row * hidden + column] *
                normalized_embedding[column];
        }
        embedding_projection[row] = sum;
    }

    std::vector<double> output(
        normalized_wide_hidden.size(),
        0.0);
    for (std::size_t stream = 0;
         stream < streams;
         ++stream) {
        const auto base =
            stream * hidden;
        for (std::size_t row = 0;
             row < hidden;
             ++row) {
            double hidden_projection = 0.0;
            for (std::size_t column = 0;
                 column < hidden;
                 ++column) {
                hidden_projection +=
                    fc_hidden_row_major[
                        row * hidden + column] *
                    normalized_wide_hidden[
                        base + column];
            }
            output[base + row] =
                embedding_projection[row] +
                hidden_projection;
        }
    }
    return output;
}

}  // namespace strix
