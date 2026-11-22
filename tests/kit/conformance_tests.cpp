#include "compare.hpp"

#include <array>
#include <bit>
#include <iostream>

namespace {
using namespace strix::kit;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

Bytes little_endian(std::uint32_t value) {
    Bytes bytes;
    for (unsigned shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    return bytes;
}

float operand(const Bytes& bytes, std::size_t element) {
    std::uint32_t bits = 0;
    for (unsigned i = 0; i < 4; ++i) bits |= std::uint32_t{bytes.at(element * 4 + i)} << (i * 8U);
    return std::bit_cast<float>(bits);
}

void caught(std::string_view test, std::string_view detail, std::string_view expected) {
    require(detail == expected, detail.empty() ? "mutation escaped" : detail);
    std::cout << test << ": CAUGHT " << detail << '\n';
}

void reduction(const Capture& capture) {
    const auto& operands = buffer(capture.inputs, "operands-f32le");
    const auto run = [&](std::string_view key) -> Buffers {
        const auto& order = buffer(capture.inputs, key);
        const float left = operand(operands, order.at(0)) + operand(operands, order.at(1));
        const float right = operand(operands, order.at(2)) + operand(operands, order.at(3));
        return {{"result-f32le", little_endian(std::bit_cast<std::uint32_t>(left + right))}};
    };
    require(compare_exact(capture.expected, run("reference-order")).empty(), "reduction control differs");
    caught("kernel_conformance", compare_exact(capture.expected, run("candidate-order")), "result-f32le: byte 2 differs");
}

void chain(const Capture& capture) {
    const auto run = [&](std::string_view key) -> Buffers {
        std::uint32_t result = buffer(capture.inputs, "input").at(0);
        for (const auto op : buffer(capture.inputs, key)) {
            switch (op) {
            case 0: result *= 2; break;
            case 1: result += 3; break;
            default: throw std::invalid_argument("unknown chain operation");
            }
        }
        return {{"result-u32le", little_endian(result)}};
    };
    require(compare_exact(capture.expected, run("reference-chain")).empty(), "chain control differs");
    caught("chain_fixtures", compare_exact(capture.expected, run("candidate-chain")), "result-u32le: byte 0 differs");
}

void invariance(const Capture& capture) {
    const auto run = [&](std::string_view weight, bool cohort) -> Buffers {
        const std::uint32_t focal = buffer(capture.inputs, "focal-row").at(0);
        const std::uint32_t neighbor = cohort ? buffer(capture.inputs, "cohort-row").at(0) : 0U;
        const std::uint32_t token = focal + neighbor * buffer(capture.inputs, weight).at(0);
        return {{"token-u32le", little_endian(token)},
                {"logit-f32le", little_endian(std::bit_cast<std::uint32_t>(static_cast<float>(token)))},
                {"state-digest-u32le", little_endian(3U * token + 1U)}};
    };
    const auto solo = run("reference-neighbor-weight", false);
    require(compare_exact(capture.expected, solo).empty(), "solo reference differs");
    require(compare_exact(solo, run("reference-neighbor-weight", true)).empty(), "cohort control differs");
    require(compare_exact(solo, run("candidate-neighbor-weight", false)).empty(), "candidate solo differs");
    caught("row_invariance", compare_exact(solo, run("candidate-neighbor-weight", true)), "token-u32le: byte 0 differs");
}

void compare_tests() {
    const Buffers expected{{"a", {0x00, 0x80}}, {"b", {}}};
    require(compare_exact(expected, {{"b", {}}, {"a", {0x00, 0x80}}}).empty(), "buffer names were not matched");
    require(compare_exact(expected, {{"a", {0x00, 0x00}}, {"b", {}}}) == "a: byte 1 differs", "byte difference escaped");
    constexpr std::array reference{1.0F, -2.0F};
    constexpr std::array candidate{1.0F, -2.25F};
    require(compare_bounded(reference, candidate, "fast-v1", "fast-v1", 0.125, 0.0625).empty(), "boundary failed");
    require(compare_bounded(reference, candidate, "fast-v1", "fast-v2", 0.125, 0.0625) == "numerical identity differs",
            "identity difference escaped");
    require(compare_bounded(reference, candidate, "fast-v1", "fast-v1", 0.125, 0).starts_with("element 1"),
            "bound violation escaped");
    std::cout << "capture_compare: PASS\n";
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 3) throw std::invalid_argument("usage: conformance-tests CASE SEED_DIR");
    const std::string_view test = argv[1];
    if (test == "capture_compare") {
        compare_tests();
        return 0;
    }
    struct Case { std::string_view test, file; void (*run)(const Capture&); };
    constexpr std::array cases{
        Case{"kernel_conformance", "swapped-reduction-pair.json", reduction},
        Case{"chain_fixtures", "wrong-chain-order.json", chain},
        Case{"row_invariance", "leaked-row-dependence.json", invariance},
    };
    for (const auto& candidate : cases) if (test == candidate.test) {
        const auto capture = read_capture(std::filesystem::path(argv[2]) / candidate.file);
        const auto encoded = write_capture(capture);
        const auto restored = parse_capture(encoded);
        require(capture.identity == restored.identity && capture.inputs == restored.inputs && capture.expected == restored.expected,
                "capture round trip changed fixture bytes");
        candidate.run(restored);
        return 0;
    }
    throw std::invalid_argument("unknown conformance test");
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
