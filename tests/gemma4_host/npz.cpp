#include "test.hpp"
#include "gemma4/ref/npz.hpp"
#include <filesystem>
#include <fstream>
int main() {
    try {
        const auto f = strix::gemma4::read_npz(std::string(G4_FIXTURE_DIR)+"/arrays.npz");
        CHECK(f.at("ids").values<std::int32_t>("<i4") == std::vector<std::int32_t>({2,17,4,-1}));
        CHECK(f.at("values").shape == std::vector<std::size_t>({3,7}));
        CHECK(f.at("values").values<float>("<f4")[20] == 5.0F);
        refuses([&] { (void)f.at("ids").values<float>("<f4"); });
        const auto path = std::filesystem::current_path()/"g4-broken.npz";
        std::ifstream in(std::string(G4_FIXTURE_DIR)+"/arrays.npz",std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)),{}); bytes[60] ^= 0x40;
        { std::ofstream out(path,std::ios::binary); out.write(bytes.data(),std::streamsize(bytes.size())); }
        refuses([&] { (void)strix::gemma4::read_npz(path); });
        std::filesystem::remove(path);
        std::cout << "NPZ: compressed fixtures, dimensions, dtype and corruption gates passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
