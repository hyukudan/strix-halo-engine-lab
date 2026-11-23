// Host test of vision::LookupWeights: a WeightSource over weight::TensorLookup reads BF16 / F32 safetensors tensors
// to FP32 views shaped [extent 0][rest], owns them, and reports absent names. No GPU, no checkpoint.
#include "strix/vision/lookup_weights.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {

int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

void put_u16(std::string& s, unsigned v) {
    s.push_back(static_cast<char>(v & 0xFFU));
    s.push_back(static_cast<char>((v >> 8U) & 0xFFU));
}

}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    // a.weight BF16 [2][3] = 1, -2, 0.5, 0.25, 3, -0.125; b F32 [4] = 1.5, 2.5, -1, 7.
    std::string data;
    for (const unsigned v : {0x3F80U, 0xC000U, 0x3F00U, 0x3E80U, 0x4040U, 0xBE00U}) put_u16(data, v);
    for (const float f : {1.5F, 2.5F, -1.0F, 7.0F}) {
        char b[4];
        std::memcpy(b, &f, 4);
        data.append(b, 4);
    }
    std::string header =
        R"({"a.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"F32","shape":[4],"data_offsets":[12,28]}})";
    while (header.size() % 8U) header.push_back(' ');
    const auto path = std::filesystem::temp_directory_path() / "strix_vision_lookup_weights_test.safetensors";
    {
        std::ofstream f(path, std::ios::binary);
        const std::uint64_t n = header.size();
        char len[8];
        for (unsigned i = 0; i < 8; ++i) len[i] = static_cast<char>((n >> (8U * i)) & 0xFFU);
        f.write(len, 8);
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
        f.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    const std::vector<strix::weight::TensorSpec> specs{{"a.weight", "BF16", {2, 3}}, {"b", "F32", {4}}};
    strix::weight::TensorLookup lookup;
    const strix::weight::TensorSchema schema{strix::weight::Container::safetensors, "safetensors@1", specs, {}};
    const strix::weight::Status st = lookup.open_safetensors(path.string(), schema);
    CHECK(static_cast<bool>(st));
    if (st) {
        strix::vision::LookupWeights w(lookup);
        CHECK(w.has("a.weight") && w.has("b") && !w.has("c"));
        const strix::vision::Tensor a = w.tensor("a.weight");
        CHECK(a.rows == 2U && a.cols == 3U && a.type == strix::vision::Tensor::Type::f32);
        const float want_a[6] = {1.0F, -2.0F, 0.5F, 0.25F, 3.0F, -0.125F};
        for (std::size_t i = 0; i < 6; ++i) CHECK(a.at(i) == want_a[i]);
        const strix::vision::Tensor b = w.tensor("b");
        CHECK(b.rows == 4U && b.cols == 1U && b.at(3) == 7.0F && b.at(0) == 1.5F);
        CHECK(w.tensor("a.weight").data == a.data);  // read once, owned
        bool threw = false;
        try {
            (void)w.tensor("c");
        } catch (const std::out_of_range&) {
            threw = true;
        }
        CHECK(threw);
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("vision lookup weights: PASS");
    return 0;
}
