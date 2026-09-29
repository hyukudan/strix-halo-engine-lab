#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
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

void write_file(
    const std::filesystem::path& path,
    std::string_view content) {
    std::ofstream output(
        path,
        std::ios::binary);
    output.write(
        content.data(),
        static_cast<std::streamsize>(
            content.size()));
    if (!output) {
        throw std::runtime_error(
            "failed to write hash fixture");
    }
}

}  // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() /
        "strix-model-content-hash-test";
    std::error_code error;
    std::filesystem::remove_all(
        root,
        error);
    std::filesystem::create_directories(
        root);

    const auto empty_path =
        root / "empty.bin";
    const auto abc_path =
        root / "abc.bin";
    write_file(empty_path, "");
    write_file(abc_path, "abc");

    const auto empty =
        strix::sha256_file_content(
            empty_path);
    const auto abc =
        strix::sha256_file_content(
            abc_path);

    expect(
        strix::model_content_identity_hex(
            empty) ==
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855",
        "SHA-256 empty-file vector matches");
    expect(
        strix::model_content_identity_hex(
            abc) ==
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc vector matches");

    const auto shard_a =
        root / "a.gguf";
    const auto shard_b =
        root / "b.gguf";
    write_file(shard_a, "first shard");
    write_file(shard_b, "second shard");

    std::vector<strix::ModelContentFile>
        files{
            {"model-00002.gguf", shard_b},
            {"model-00001.gguf", shard_a},
        };
    const auto identity =
        strix::build_model_content_identity(
            files);

    std::reverse(
        files.begin(),
        files.end());
    const auto reordered =
        strix::build_model_content_identity(
            files);
    expect(identity == reordered,
           "aggregate model identity is stable under input file ordering");

    write_file(
        shard_b,
        "second shard changed");
    const auto changed =
        strix::build_model_content_identity(
            files);
    expect(changed != identity,
           "aggregate model identity changes when shard content changes");

    files[1].logical_name =
        files[0].logical_name;
    bool duplicate_rejected = false;
    try {
        static_cast<void>(
            strix::build_model_content_identity(
                files));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    expect(duplicate_rejected,
           "aggregate identity rejects duplicate logical shard names");

    std::filesystem::remove_all(
        root,
        error);

    if (failures == 0) {
        std::cout << "model content hash tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
