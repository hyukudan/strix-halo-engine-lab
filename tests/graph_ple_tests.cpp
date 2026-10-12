#include "strix/graph_ple.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>

int main() try {
    using namespace strix;
    PleHashContract h;
    h.ngram_size = 3; h.heads_per_ngram = 1; h.head_count = 2;
    h.head_dim = 16; h.row_bytes = 32; h.eos_token_id = 1;
    h.head_offsets[0] = 0; h.head_vocab_sizes[0] = 11;
    h.head_offsets[1] = 11; h.head_vocab_sizes[1] = 13; h.minimum_table_rows = 24;
    RuntimeTensorBinding table{0, RuntimeTensorStorage::resident, 0x1000, 768, 1, 0};
    auto args = bind_graph_inputs(graph_input_layout(1), ExecutionMemoryBinding{7, 0x2000, 16, 2});
    ExecutionMemoryBinding status{7, 0x3000, 4, 3}, previous{7, 0x4000, 8, 4}, next{7, 0x5000, 8, 5};
    ExecutionBufferAddress gathered;
    gathered.address = 0x6000; gathered.bytes = 64; gathered.allocation_generation = 6; gathered.use.generation = 1;
    auto validate = [&] { validate_graph_ple_bindings(h, 64, 128, table, args, status, gathered, previous, next); };
    validate();
    auto rejects = [&](const std::function<void()>& mutate, const std::function<void()>& restore) {
        mutate(); bool rejected = false;
        try { validate(); } catch (const std::exception&) { rejected = true; }
        restore(); if (!rejected) throw std::runtime_error("invalid graph PLE binding accepted");
    };
    rejects([&]{ table.storage = RuntimeTensorStorage::file_mapped; }, [&]{ table.storage = RuntimeTensorStorage::resident; });
    rejects([&]{ table.bytes = 767; }, [&]{ table.bytes = 768; });
    rejects([&]{ next.arena_base = previous.arena_base; }, [&]{ next.arena_base = 0x5000; });
    rejects([&]{ status.execution_identity = 8; }, [&]{ status.execution_identity = 7; });
    rejects([&]{ previous.allocation_generation = 0; }, [&]{ previous.allocation_generation = 4; });
    rejects([&]{ gathered.bytes = 63; }, [&]{ gathered.bytes = 64; });
    rejects([&]{ h.head_vocab_sizes[1] = 14; }, [&]{ h.head_vocab_sizes[1] = 13; });
    rejects([&]{ h.eos_token_id = 64; }, [&]{ h.eos_token_id = 1; });
    rejects([&]{ args.position_offset = 0; }, [&]{ args.position_offset = 8; });
    std::cout << "graph PLE preflight tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
