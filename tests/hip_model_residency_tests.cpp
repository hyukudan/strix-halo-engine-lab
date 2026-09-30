#include "strix/hip_model_residency.hpp"

#include "strix/model_manifest.hpp"
#include "strix/runtime_allocator.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}

std::filesystem::path fixture_path() {
#if defined(__linux__)
    const auto suffix =
        std::to_string(
            static_cast<unsigned long long>(
                ::getpid()));
#else
    const auto suffix = "portable";
#endif
    return std::filesystem::temp_directory_path() /
           ("strix-hip-model-residency-" +
            suffix + ".bin");
}

void write_fixture(
    const std::filesystem::path& path) {
    constexpr std::size_t bytes = 16U * 1024U;
    std::vector<std::uint8_t> data(bytes);
    for (std::size_t index = 0;
         index < data.size();
         ++index) {
        data[index] =
            static_cast<std::uint8_t>(
                (index * 37U + 11U) &
                0xffU);
    }

    std::ofstream output(
        path,
        std::ios::binary |
        std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot create residency fixture");
    }
    output.write(
        reinterpret_cast<const char*>(
            data.data()),
        static_cast<std::streamsize>(
            data.size()));
    if (!output) {
        throw std::runtime_error(
            "cannot write residency fixture");
    }
}

strix::ModelManifest fixture_manifest() {
    strix::ModelManifest manifest;
    manifest.architecture =
        "residency-fixture";

    const auto add =
        [&](std::string name,
            std::uint64_t offset,
            strix::ResidencyPolicy residency) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.dimensions = {1024};
            tensor.ggml_type = 0;
            tensor.byte_size = 1024;
            tensor.absolute_offset =
                offset;
            tensor.role =
                strix::TensorRole::
                    dense_projection;
            tensor.residency =
                residency;
            manifest.total_tensor_bytes +=
                tensor.byte_size;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add(
        "dense.a",
        4096,
        strix::ResidencyPolicy::resident);
    add(
        "ple.table",
        8192,
        strix::ResidencyPolicy::
            file_backed_candidate);
    add(
        "dense.b",
        12288,
        strix::ResidencyPolicy::resident);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);
    return manifest;
}

std::uint8_t expected_byte(
    std::uint64_t file_offset) {
    return static_cast<std::uint8_t>(
        (file_offset * 37ULL + 11ULL) &
        0xffULL);
}

void verify_binding(
    const strix::MappedModelFile& mapped,
    const strix::ModelRuntimeBinding& binding,
    strix::HipModelResidencyMode mode) {
    expect(
        binding.tensors.size() == 3,
        "residency binding preserves tensor count");

    const auto promoted_storage =
        mode ==
                strix::HipModelResidencyMode::
                    registered_file_mapping
            ? strix::RuntimeTensorStorage::
                  registered_file_mapped
            : strix::RuntimeTensorStorage::
                  resident;

    expect(
        binding.tensors[0].storage ==
                promoted_storage &&
            binding.tensors[2].storage ==
                promoted_storage &&
            strix::runtime_tensor_gpu_visible(
                binding.tensors[0].storage) &&
            strix::runtime_tensor_gpu_visible(
                binding.tensors[2].storage),
        "dense tensors become GPU-visible under selected residency mode");

    expect(
        binding.tensors[1].storage ==
                strix::RuntimeTensorStorage::
                    file_mapped &&
            !strix::runtime_tensor_gpu_visible(
                binding.tensors[1].storage) &&
            binding.tensors[1].address ==
                mapped.base_address() +
                    8192U,
        "file-backed candidate remains ordinary host mapping");

    std::vector<std::uint8_t> copied(1024);
    check(
        hipMemcpy(
            copied.data(),
            reinterpret_cast<const void*>(
                binding.tensors[0].address),
            copied.size(),
            hipMemcpyDeviceToHost),
        "copy promoted tensor back");
    expect(
        copied.front() ==
                expected_byte(4096) &&
            copied.back() ==
                expected_byte(
                    4096 + 1023),
        "GPU-visible tensor bytes match mapped file contents");

    const auto* host_table =
        reinterpret_cast<
            const std::uint8_t*>(
                binding.tensors[1].address);
    expect(
        host_table[0] ==
                expected_byte(8192) &&
            host_table[1023] ==
                expected_byte(
                    8192 + 1023),
        "unpromoted file-backed tensor remains host-readable");
}

}  // namespace

int main() try {
    const auto path =
        fixture_path();
    write_fixture(path);

    try {
        strix::AllocationGenerationSource
            generations;
        auto mapped =
            strix::MappedModelFile::
                open_read_only(
                    path,
                    "residency-fixture",
                    generations);
        const auto manifest =
            fixture_manifest();

        {
            strix::HipModelResidencyPolicy
                policy;
            policy.mode =
                strix::HipModelResidencyMode::
                    resident_copy;
            auto residency =
                strix::HipModelResidency::create(
                    mapped,
                    manifest,
                    generations,
                    policy);

            expect(
                residency.copied_bytes() ==
                    2048,
                "resident-copy mode copies only promoted tensor bytes");
            expect(
                residency.registered_bytes() ==
                    0,
                "resident-copy mode registers no host pages");
            verify_binding(
                mapped,
                residency.binding(),
                policy.mode);
        }

        {
            strix::HipModelResidencyPolicy
                policy;
            policy.mode =
                strix::HipModelResidencyMode::
                    registered_file_mapping;
            auto residency =
                strix::HipModelResidency::create(
                    mapped,
                    manifest,
                    generations,
                    policy);

            expect(
                residency.registered_bytes() >=
                    8192,
                "registered mode accounts page-granular promoted ranges");
            expect(
                residency.copied_bytes() ==
                    0,
                "registered mode copies no model bytes");
            verify_binding(
                mapped,
                residency.binding(),
                policy.mode);
        }
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }

    std::filesystem::remove(path);

    if (failures == 0) {
        std::cout
            << "HIP model residency tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP model residency tests: "
        << error.what() << '\n';
    return 2;
}
