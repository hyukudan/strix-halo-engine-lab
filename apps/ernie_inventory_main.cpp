#include "strix/ernie/inventory.hpp"

#include <iostream>

int main(int argc, char** argv) {
    const bool list = argc == 3 && std::string_view(argv[2]) == "--list";
    if (argc != 2 && !list) {
        std::cerr << "usage: strix-ernie-inventory CHECKPOINT_DIR [--list]\n";
        return 2;
    }
    strix::ernie::Inventory inventory;
    if (const auto status = strix::ernie::inspect_weights(argv[1], inventory); !status) {
        std::cerr << status.detail << '\n';
        return 1;
    }
    std::size_t unclassified = 0;
    std::uint64_t bytes = 0;
    if (list) std::cout << "tensor\tmodule\tshard\tdtype\tshape\tbyte_offset\tbytes\n";
    for (const auto& [name, tensor] : inventory.tensors) {
        if (tensor.module == strix::ernie::Module::unclassified) ++unclassified;
        bytes += tensor.payload.range.bytes;
        if (!list) continue;
        std::cout << name << '\t' << strix::ernie::module_name(tensor.module) << '\t'
                  << tensor.payload.source->shard_name() << '\t' << tensor.wire_dtype << '\t';
        for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
            if (axis != 0) std::cout << ',';
            std::cout << tensor.shape[axis];
        }
        std::cout << '\t' << tensor.payload.range.offset << '\t' << tensor.payload.range.bytes << '\n';
    }
    std::cerr << "tensors=" << inventory.tensors.size() << " payload_bytes=" << bytes
              << " missing_shards=" << inventory.missing_shards.size()
              << " missing_known_tensors=" << inventory.missing_known_tensors
              << " unclassified=" << unclassified << '\n';
}
