#include "strix/mtp_input_fusion.hpp"

#include <limits>
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
    return plan;
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
