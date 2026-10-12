#include "strix/native_setup_quiescence.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void tests() {
    using strix::prove_native_setup_quiescence;
    std::vector<int> order;
    int observations = 0;
    expect(prove_native_setup_quiescence([&] { order.push_back(++observations); },
                                         [&] { order.push_back(0); }), "successful original-device proof");
    expect(order == std::vector<int>({1, 0, 2}), "binding observation brackets the drain");

    for (int failure = 0; failure != 3; ++failure) {
        order.clear(); observations = 0;
        auto owned_source = std::make_shared<int>(42);
        std::weak_ptr<int> source = owned_source;
        bool released = false;
        const bool proof = prove_native_setup_quiescence(
            [&] {
                order.push_back(++observations);
                expect(!source.expired(), "source retained during binding observation");
                if ((failure == 0 && observations == 1) || (failure == 2 && observations == 2))
                    throw std::runtime_error("same index but changed UUID/runtime/owner");
            },
            [&] {
                order.push_back(0);
                expect(!source.expired(), "source retained while draining queued copy");
                if (failure == 1) throw std::runtime_error("drain failed");
            });
        if (proof) { owned_source.reset(); released = true; }
        expect(!proof && !released && !source.expired(), "failed proof cannot authorize source/destination cleanup");
        const std::vector<int> expected = failure == 0 ? std::vector<int>{1}
                                        : failure == 1 ? std::vector<int>{1, 0}
                                                       : std::vector<int>{1, 0, 2};
        expect(order == expected, "failed step stops proof without retry or invented completion");
    }
    int calls = 0;
    expect(!prove_native_setup_quiescence({}, [&] { ++calls; }) && calls == 0,
           "absent original binding fails before drain");
    expect(!prove_native_setup_quiescence([&] { ++calls; }, {}) && calls == 0,
           "absent drain fails before observation");
}
}
int main() {
    try { tests(); std::cout << "native setup original-binding/drain ordering tests; no GPU execution\n"; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
