// Host-only census of the production Qwen GGUF: every tensor's name, ggml type, dimensions and bytes (header only).
#include <cstdio>
#include "qwen_gguf_host.hpp"

int main(int argc, char** argv) {
    const strix::qwen_gguf::Model m(argc > 1 ? argv[1] : "");
    std::printf("tensors %zu data_offset %llu\n", m.info().tensors.size(), static_cast<unsigned long long>(m.info().data_offset));
    for (const auto& t : m.info().tensors) {
        std::printf("%s type %u dims", t.name.c_str(), t.ggml_type);
        for (auto d : t.dimensions) std::printf(" %llu", static_cast<unsigned long long>(d));
        std::printf(" bytes %llu\n", static_cast<unsigned long long>(t.byte_size));
    }
    return 0;
}
