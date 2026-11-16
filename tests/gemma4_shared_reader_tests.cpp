#include "gemma4/model/model.hpp"
#include <fstream>
#include <iostream>
int main() {
    try {
        const auto path = std::filesystem::current_path()/"gemma4-shared-reader.safetensors";
        const std::string header = R"({"n":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
        {
            std::ofstream f(path,std::ios::binary);
            const auto size = static_cast<std::uint64_t>(header.size()); const float values[2] = {1,2};
            f.write(reinterpret_cast<const char*>(&size),8); f.write(header.data(),static_cast<std::streamsize>(header.size()));
            f.write(reinterpret_cast<const char*>(values),sizeof(values));
        }
        auto file = strix::gemma4::read_safetensors(path);
        if (file.tensors.size() != 1 || file.tensors[0].bytes != 8 || file.tensors[0].shape != std::vector<std::uint64_t>{2}) return 1;
        const auto& t = file.tensors[0];
        const strix::weights::Tensor view{file.storage,t.shape,t.dtype,0,t.offset,t.bytes};
        file = {};
        if (strix::weights::decode_row(view,0) != std::vector<float>{1,2}) return 1;
        std::cout << "Gemma shared reader and view ownership passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
