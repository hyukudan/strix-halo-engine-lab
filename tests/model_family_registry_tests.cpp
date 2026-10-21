#include "strix/model_family_registry.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_invalid(const std::function<void()>& function, const std::string& message) {
    try {
        function();
        expect(false, message + " (no exception)");
    } catch (const std::invalid_argument&) {
    }
}

template <typename T>
void put(std::string& b, T v) {
    for (std::size_t i = 0; i < sizeof(T); ++i) b.push_back(static_cast<char>((static_cast<std::uint64_t>(v) >> (8U * i)) & 0xffU));
}

// A GGUF with only general.architecture and one F32 tensor.
std::filesystem::path write_gguf(const std::string& architecture) {
    std::string b = "GGUF";
    put<std::uint32_t>(b, 3);
    put<std::uint64_t>(b, 1);
    put<std::uint64_t>(b, 1);
    put<std::uint64_t>(b, 20);
    b += "general.architecture";
    put<std::uint32_t>(b, 8);
    put<std::uint64_t>(b, architecture.size());
    b += architecture;
    put<std::uint64_t>(b, 3);
    b += "w.0";
    put<std::uint32_t>(b, 1);
    put<std::uint64_t>(b, 8);
    put<std::uint32_t>(b, 0);
    put<std::uint64_t>(b, 0);
    while (b.size() % 32 != 0) b.push_back('\0');
    b.append(32, '\0');
    const auto path = std::filesystem::temp_directory_path() / ("strix-registry-" + architecture + ".gguf");
    std::ofstream(path, std::ios::binary).write(b.data(), static_cast<std::streamsize>(b.size()));
    return path;
}

}  // namespace

int main() try {
    using namespace strix;
    expect(registered_model_families().empty(), "the core starts with no family");
    expect(find_model_family("toy") == nullptr, "nothing admits an unknown architecture");

    int validated = 0;
    ModelFamily toy;
    toy.name = "toy";
    toy.architectures = {"toy", "toy-draft"};
    toy.validate = [&](const gguf::FileInfo& info) {
        ++validated;
        if (info.tensors.size() != 1U) throw std::invalid_argument("toy expects one tensor");
    };
    toy.describe = [](const gguf::FileInfo&) { return std::string("toy model"); };
    register_model_family(toy);
    register_model_family(toy);  // idempotent
    expect(model_family_registered("toy") && registered_model_families().size() == 1U, "registered once");
    expect(find_model_family("toy-draft") != nullptr && find_model_family("toy-draft")->name == "toy", "lookup by architecture");

    ModelFamily clash = toy;
    clash.name = "other";
    expect_invalid([&] { register_model_family(clash); }, "an architecture belongs to one family");
    ModelFamily changed = toy;
    changed.architectures = {"toy"};
    expect_invalid([&] { register_model_family(changed); }, "a name cannot be re-registered differently");
    expect_invalid([] { register_model_family(ModelFamily{}); }, "incomplete registration");

    const auto file = write_gguf("toy");
    const auto admitted = admit_model_file(file);
    expect(admitted.family != nullptr && admitted.family->name == "toy" && validated == 1 && admitted.info.tensors.size() == 1U,
           "a file is admitted by the family of its architecture");
    const auto unknown = write_gguf("nobody");
    expect_invalid([&] { (void)admit_model_file(unknown); }, "a file of an unregistered architecture is refused");
    std::filesystem::remove(file);
    std::filesystem::remove(unknown);

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "model family registry tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
}
