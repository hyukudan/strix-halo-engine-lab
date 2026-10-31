#include "gemma4/prefill/reference.hpp"
#include "../gemma4_gpu/fixture.hpp"

using namespace strix::gemma4;
namespace pf = strix::gemma4::prefill;
int main() try {
    for (unsigned count : {6U, 42U}) {
        const auto model = fixture::tiny(count);
        fixture::Tables tables;
        const auto weights = bind_host_weights(model);
        pf::Reference reference(weights, tables.view(), 1024);
        HostTarget exact(weights, tables.view(), 1);
        HostStream state(model.config);
        for (unsigned n = 1; n <= 3; ++n) {
            std::vector<unsigned> tokens(n);
            auto tx = state.begin();
            std::vector<RowPlan> rows;
            for (unsigned r = 0; r < n; ++r) {
                tokens[r] = (n * 3 + r) % model.config.vocab;
                rows.push_back({1, 2, state.frontier() + r, tokens[r], tx.get(), r, true});
            }
            const auto expected = exact.forward_rows(rows); tx->commit(n);
            const auto actual = reference.forward(tokens, false, true);
            bytes(actual.all_logits, expected.logits); bytes(actual.taps, expected.taps);
            CHECK(reference.digest() == state.digest());
        }
        pf::Reference matrix(weights, tables.view(), 1024), serial(weights, tables.view(), 1024);
        const std::array<unsigned, 3> tokens = {1, 5, 8};
        const auto batched = matrix.forward(tokens, true, true);
        for (unsigned r = 0; r < tokens.size(); ++r) {
            const auto solo = serial.forward({&tokens[r], 1}, true);
            bytes(solo.last_logits, std::vector<float>(batched.all_logits.begin() + std::ptrdiff_t(r * model.config.vocab), batched.all_logits.begin() + std::ptrdiff_t((r + 1) * model.config.vocab)));
            bytes(solo.taps, std::vector<float>(batched.taps.begin() + std::ptrdiff_t(r * batched.tap_width), batched.taps.begin() + std::ptrdiff_t((r + 1) * batched.tap_width)));
        }
        CHECK(matrix.digest() == serial.digest());
    }
    std::puts("prefill independent host graph/reference state PASS");
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
