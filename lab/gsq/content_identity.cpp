// content_identity FILE : the file sha256 and the loader's STRIX-MODEL-CONTENT-V1 identity (logical name "model",
// as strix-hip-decode-token maps its model): the --mtp-target-sha256 / --mtp-target-content of a composed GGUF.
#include "strix/model_content_hash.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) try {
    if (argc != 2) { std::fprintf(stderr, "usage: content_identity FILE\n"); return 2; }
    const auto id = strix::hash_single_model_content_file({"model", argv[1]});
    std::printf("file_sha256 %s\ncontent %s\nbytes %llu\n", strix::model_content_identity_hex(id.file_sha256).c_str(),
        strix::model_content_identity_hex(id.content_identity).c_str(), static_cast<unsigned long long>(id.file_bytes));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
