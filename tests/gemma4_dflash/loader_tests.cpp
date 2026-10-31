#include "fixture.hpp"

#include <chrono>
#include <iostream>

int main() {
    try {
        const auto w = fixture::weights();
        const auto path = std::filesystem::temp_directory_path() /
            ("g4-dflash-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
        struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code e; std::filesystem::remove(p, e); } } cleanup{path};
        fixture::gguf_file(path, w);
        const auto file = strix::gemma4::dflash::DraftFile::open_fixture(path, fixture::shape());
        const auto& got = file.weights();
        fixture::check(got.fc.bits == w.fc.bits && got.layer[1].v.bits == w.layer[1].v.bits, "matrix layout/decode");
        fixture::check(got.layer[1].q_norm == w.layer[1].q_norm && got.final_norm == w.final_norm, "direct norm values");
        fixture::check(got.license == "fixture" && got.logit_scale == 1.25F && got.logit_cap == 6.0F, "metadata retention");
        fixture::rejects([&] { (void)strix::gemma4::dflash::DraftFile::open(path); });
        fixture::gguf_file(path, w, "unsigned_taps");
        (void)strix::gemma4::dflash::DraftFile::open_fixture(path, fixture::shape());
        for (const auto* mutation : {"missing", "extra", "shape", "dtype", "nan", "norm_nan", "window", "taps", "mask"}) {
            fixture::gguf_file(path, w, mutation);
            fixture::rejects([&] { (void)strix::gemma4::dflash::DraftFile::open_fixture(path, fixture::shape()); });
        }
        fixture::gguf_file(path, w);
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 64);
        fixture::rejects([&] { (void)strix::gemma4::dflash::DraftFile::open_fixture(path, fixture::shape()); });
        std::cout << "DFlash loader passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
