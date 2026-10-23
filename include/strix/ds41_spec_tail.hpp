#pragma once
// The DSpark drafter of a pipeline TAIL (rank 0 drafts alone): host-only interface over the HIP drafter, fed with the
// committed rows' inputs of the target's last three layers (DeviceForward::captured_streams with capture_main_hidden).

#include "strix/ds41_dspark.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::dist {
class Communicator;
}

namespace strix::ds41::dspark {

class SpecTailDrafter {
public:
    // model: the target file (token embedding, head); draft: the drafter file; vocab: a choose_vocabulary spec
    SpecTailDrafter(const std::string& model, const std::string& draft, std::string vocabulary);
    ~SpecTailDrafter();
    SpecTailDrafter(const SpecTailDrafter&) = delete;
    SpecTailDrafter& operator=(const SpecTailDrafter&) = delete;

    // Both ranks of a world-2 TAIL draft together, the head split by vocabulary (the pair's "split" placement): every
    // draft() is collective (5 small exchanges on the communicator); both ranks must stage the same rows.
    void set_split(std::uint32_t rank, strix::dist::Communicator& comm);
    // A new sequence: rings cleared, nothing staged.
    void reset();
    // The first `rows` rows of the last step (committed) into the staging buffer (device to device on `stream`, flushed
    // to the drafter every 128 rows and before a draft).
    void stage(const std::array<const float*, 3>& captures, std::uint32_t rows, void* stream);
    void flush(void* stream);
    // At the start of drafting: the vocabulary chosen from the context, the context's ids and `first` added.
    std::string begin(std::span<const std::uint32_t> context, std::uint32_t first);
    void add_vocabulary(std::span<const std::uint32_t> ids);
    // Flushes the staged rows, then drafts 5 tokens after `token`.
    [[nodiscard]] Draft draft(std::uint32_t token, void* stream);
    [[nodiscard]] std::uint64_t next_position() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::ds41::dspark
