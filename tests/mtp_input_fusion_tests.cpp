#include "strix/mtp_input_fusion.hpp"

#include <iostream>
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

    if (failures == 0) {
        std::cout << "MTP input fusion tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
