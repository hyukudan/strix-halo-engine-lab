#include "strix/serial_decode_reference.hpp"

#include "strix/model_content_hash.hpp"

#include <filesystem>
#include <fstream>
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

void write_text(
    const std::filesystem::path& path,
    std::string_view value) {
    std::ofstream output(path);
    output << value;
    if (!output) {
        throw std::runtime_error(
            "failed to write serial replay fixture");
    }
}

}  // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() /
        "strix-serial-reference-tests";
    std::error_code error;
    std::filesystem::remove_all(
        root,
        error);
    std::filesystem::create_directories(
        root);

    const auto valid =
        root / "valid.txt";
    write_text(
        valid,
        "STRIX_SERIAL_REPLAY_V1\n"
        "content_sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
        "reference_id 42\n"
        "step 11 44 2\n"
        "logit 44 1.25\n"
        "logit 12 -0.5\n"
        "step 22 55 1\n"
        "logit 55 2.5\n");

    const auto reference =
        strix::read_serial_decode_reference(
            valid);

    expect(reference.steps.size() == 2 &&
               reference.steps[0].input_token == 11 &&
               reference.steps[0].expected_argmax_token == 44 &&
               reference.steps[0].logits.size() == 2 &&
               reference.independent_reference_identity == 42,
           "versioned serial reference parses deterministic steps");
    expect(
        strix::model_content_identity_hex(
            reference.model_content_identity) ==
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad",
        "serial reference preserves checkpoint SHA-256");

    bool vocabulary_valid = true;
    try {
        strix::validate_serial_decode_reference_vocabulary(
            reference,
            56);
    } catch (const std::invalid_argument&) {
        vocabulary_valid = false;
    }
    expect(vocabulary_valid,
           "serial reference accepts token and probe ids below vocabulary size");

    auto token_out_of_range = reference;
    token_out_of_range.steps[0].input_token = 56;
    bool token_rejected = false;
    try {
        strix::validate_serial_decode_reference_vocabulary(
            token_out_of_range,
            56);
    } catch (const std::invalid_argument&) {
        token_rejected = true;
    }
    expect(token_rejected,
           "serial reference rejects input token at vocabulary boundary");

    auto argmax_out_of_range = reference;
    argmax_out_of_range.steps[0].expected_argmax_token = 56;
    bool argmax_rejected = false;
    try {
        strix::validate_serial_decode_reference_vocabulary(
            argmax_out_of_range,
            56);
    } catch (const std::invalid_argument&) {
        argmax_rejected = true;
    }
    expect(argmax_rejected,
           "serial reference rejects expected argmax at vocabulary boundary");

    auto probe_out_of_range = reference;
    probe_out_of_range.steps[0].logits[0].index = 56;
    bool probe_rejected = false;
    try {
        strix::validate_serial_decode_reference_vocabulary(
            probe_out_of_range,
            56);
    } catch (const std::invalid_argument&) {
        probe_rejected = true;
    }
    expect(probe_rejected,
           "serial reference rejects logit probe at vocabulary boundary");

    bool zero_vocab_rejected = false;
    try {
        strix::validate_serial_decode_reference_vocabulary(
            reference,
            0);
    } catch (const std::invalid_argument&) {
        zero_vocab_rejected = true;
    }
    expect(zero_vocab_rejected,
           "serial reference rejects zero-sized vocabulary");

    const auto duplicate =
        root / "duplicate.txt";
    write_text(
        duplicate,
        "STRIX_SERIAL_REPLAY_V1\n"
        "content_sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
        "reference_id 42\n"
        "step 11 44 2\n"
        "logit 44 1.25\n"
        "logit 44 1.0\n");

    bool duplicate_rejected = false;
    try {
        static_cast<void>(
            strix::read_serial_decode_reference(
                duplicate));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    expect(duplicate_rejected,
           "serial reference rejects duplicate logit probes");

    const auto zero_hash =
        root / "zero-hash.txt";
    write_text(
        zero_hash,
        "STRIX_SERIAL_REPLAY_V1\n"
        "content_sha256 0000000000000000000000000000000000000000000000000000000000000000\n"
        "reference_id 42\n"
        "step 11 44 1\n"
        "logit 44 1.0\n");

    bool zero_hash_rejected = false;
    try {
        static_cast<void>(
            strix::read_serial_decode_reference(
                zero_hash));
    } catch (const std::invalid_argument&) {
        zero_hash_rejected = true;
    }
    expect(zero_hash_rejected,
           "serial reference rejects unset checkpoint content identity");

    std::filesystem::remove_all(
        root,
        error);

    if (failures == 0) {
        std::cout << "serial decode reference tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
