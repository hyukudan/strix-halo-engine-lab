#include "inkling2/inkling2_mini_checkpoint.hpp"

#include <iostream>

namespace {

strix::weight::Status check_index(const strix::weight::TensorIndex& tensors, const char* path) {
    std::shared_ptr<const strix::weight::Source> source;
    if (const auto status = strix::weight::open_source(path, source); !status) return status;
    if (source->size() > 1024U * 1024U)
        return {strix::weight::Error::malformed, "Mini index exceeds 1 MiB"};
    std::string json(static_cast<std::size_t>(source->size()), '\0');
    if (const auto status = source->read({0, source->size()},
                                        {reinterpret_cast<strix::weight::Byte*>(json.data()), json.size()}); !status)
        return status;
    return strix::inkling2::validate_mini_checkpoint_index(tensors, json);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: strix-inkling2-mini-inspect MODEL.safetensors [model.safetensors.index.json]\n";
        return 2;
    }
    const auto fail = [](const strix::weight::Status& status) {
        std::cerr << "Inkling 2 Mini checkpoint: " << status.detail << '\n';
        return 1;
    };
    std::shared_ptr<const strix::weight::Source> source;
    if (const auto status = strix::weight::open_source(argv[1], source); !status) return fail(status);
    strix::weight::TensorIndex tensors;
    if (const auto status = strix::weight::index_safetensors(std::move(source), tensors); !status)
        return fail(status);
    strix::inkling2::MiniCheckpointSummary summary;
    if (const auto status = strix::inkling2::validate_mini_text_checkpoint(tensors, summary); !status)
        return fail(status);
    if (argc == 3) {
        if (const auto status = check_index(tensors, argv[2]); !status) return fail(status);
    }

    std::cout << "Inkling 2 Mini text tensor metadata valid for schema revision "
              << strix::inkling2::kMiniCheckpointRevision << '\n'
              << "Text tensors: " << summary.text_tensors << " (" << summary.text_bytes << " payload bytes)\n"
              << "Vision tensors indexed: " << summary.vision_tensors << '\n';
    return 0;
}
