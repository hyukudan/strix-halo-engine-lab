#pragma once

// DSpark draft vocabularies: token-id lists (little-endian uint32, score order) that restrict the rows
// of the drafter's head (the target's output rows and the Markov head rows). The target verifies every
// draft with its full vocabulary, so a draft vocabulary changes acceptance and cost, never the text.
//
// Directory layout (data built for the deepseek41 tokenizer): core16k/24k/32k/48k/64k.bin (prefixes of
// one ranking), pack-<L>.bin (ids outside core32k, 8k rows; ja 12k, zh 32k, en 16k) and pack-<L>16k.bin.
// Automatic choice (like the MTP draft vocabulary of the main engine): the language pack among
// es fr de it pt ru ko ja zh whose 16k list holds the most prompt tokens, when that is at least 2 tokens
// and 5 % of the prompt: core32k + pack-<L>.bin; otherwise core48k. The prompt tokens are added as
// context rows, and generated tokens are added as they are committed.
//
// Specs (STRIX_DS41_DSPARK_VOCAB): "full" (default), "<N>" (ids below N), "auto:<dir>", or
// "files:<a.bin>,<b.bin>,..." (union in file order).

#include "strix/ds41_dspark.hpp"

#include <span>
#include <string>
#include <vector>

namespace strix::ds41::dspark {

[[nodiscard]] std::vector<std::uint32_t> read_vocabulary_file(const std::string& path);

// Session inventory for fixed file lists. Prompt-dependent auto selection is
// deliberately refused for sessions; the ordinary vocabulary loader is unchanged.
[[nodiscard]] std::vector<std::string> vocabulary_asset_files(const std::string& spec);

struct VocabularyChoice {
    Vocab vocabulary;
    std::string label;
};
[[nodiscard]] VocabularyChoice choose_vocabulary(const std::string& spec, std::span<const std::uint32_t> prompt);
// Adds the ids not yet in the vocabulary (as extra rows, in order); returns how many were added.
std::size_t add_context(Vocab& vocabulary, std::span<const std::uint32_t> ids);
[[nodiscard]] bool contains(const Vocab& vocabulary, std::uint32_t id);

} // namespace strix::ds41::dspark
