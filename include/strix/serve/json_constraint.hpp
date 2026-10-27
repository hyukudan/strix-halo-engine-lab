#pragma once

#include "strix/serve/json.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::serve {
struct ConstraintLimits {
    std::size_t schema_bytes = 65536;
    std::size_t nodes = 4096;
    std::size_t depth = 64;
    std::size_t properties = 128;
    std::size_t enum_values = 1024;
    std::size_t max_string_length = 1048576;
    std::size_t max_output_bytes = 1048576;
    std::size_t compilation_steps = 65536; // Also bounds each cursor transition.
    // Aggregate limits for one compilation, including all references/alternatives.
    std::size_t literal_bytes = 1048576;
    std::size_t trie_nodes = 65536;
    std::size_t alternative_expansions = 65536;
    std::size_t mask_cache_bytes = 0; // Optional per-program LRU; zero disables it.
};
class SchemaError : public std::runtime_error {
public:
    SchemaError(std::string path, std::string keyword, std::string message);
    std::string path;
    std::string keyword;
};
class ConstraintError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct ConstraintData;
struct ConstraintState;
// Optional compilation diagnostics, including failed compilations. No global state.
struct ConstraintCompilationStats {
    std::size_t expansion_recursion_high_water = 0;
    std::size_t expansion_stack_high_water = 0;
    std::size_t compilation_steps = 0;
};
class ConstraintProgram {
private:
    friend ConstraintProgram compile_json_schema(const Json&, const ConstraintLimits&, ConstraintCompilationStats*);
    explicit ConstraintProgram(std::shared_ptr<const ConstraintData> data) : data_(std::move(data)) {}
    std::shared_ptr<const ConstraintData> data_;
    friend class ConstraintCursor;
    friend struct ConstraintAccess;
};
// strix-json-schema-v1: JSON with 0..2 spaces, tabs or newlines per gap
// (including leading/trailing gaps); no CR. Members may occur in any order,
// without duplicates. Strings use strict UTF-8 and paired UTF-16 escapes;
// length counts decoded Unicode scalars. Integers have no fraction/exponent.
// enum/const use only Json::dump(): compact, original member order, UTF-8.
// The cursor begins at the answer boundary: reasoning/framing is a caller concern.
// Copies share immutable pushdown state; admission consumes only the new byte.
// advance(false) leaves the cursor unchanged. complete() marks an accepting
// boundary (a number may still extend); EOS may be selected at that boundary.
// Output bytes, depth and string lengths are bounded. Admission reserves a
// completing suffix within those limits. Byte/shape exhaustion refuses admission;
// evaluation exhaustion throws ConstraintError, never relaxes the language. An optional bounded mask LRU uses exact serialized state keys;
// states with additional names or oversized keys bypass it. It defaults off.
ConstraintProgram compile_json_schema(const Json&, const ConstraintLimits& = {}, ConstraintCompilationStats* = nullptr);
ConstraintProgram compile_json_object();
class ConstraintCursor {
public:
    explicit ConstraintCursor(const ConstraintProgram&);
    bool advance(unsigned char byte);
    [[nodiscard]] bool complete() const;
    // A refused byte never poisons the cursor; admitted prefixes remain productive.
    [[nodiscard]] bool dead() const;
private:
    std::shared_ptr<const ConstraintData> data_;
    std::shared_ptr<const ConstraintState> state_;
    friend struct ConstraintAccess;
};
// Atomic helper: throws ConstraintError on refusal; original cursor is unchanged.
ConstraintCursor advance_token(ConstraintCursor cursor, std::string_view bytes);
class TokenByteTrie {
public:
    // Only normal/byte tokens are eligible; excluded IDs further restrict them.
    explicit TokenByteTrie(const BpeTokenizer&, std::span<const std::uint32_t> excluded = {});
    // Also useful for tokenizer-independent host fixtures. IDs are vector positions.
    explicit TokenByteTrie(std::span<const std::string> bytes, std::span<const std::uint32_t> excluded = {});
    [[nodiscard]] std::size_t vocab_size() const { return vocab_size_; }
    // Whether the ID is an eligible (inserted) token: exactly the IDs a mask walk can admit.
    [[nodiscard]] bool contains(std::uint32_t id) const { return id < members_.size() && members_[id]; }
    struct Node { std::vector<std::pair<unsigned char, std::size_t>> edges; std::vector<std::uint32_t> ids; };
    [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }
private:
    void insert(std::string_view, std::uint32_t);
    std::size_t vocab_size_ = 0;
    std::vector<Node> nodes_{1};
    std::vector<bool> members_;
    std::shared_ptr<const unsigned char> identity_ = std::make_shared<const unsigned char>(0);
    friend struct ConstraintAccess;
};
struct AllowedTokens {
    std::size_t vocab_size = 0;
    std::vector<std::uint64_t> words;
    [[nodiscard]] bool contains(std::size_t id) const;
    void set(std::size_t id, bool value = true);
};
AllowedTokens allowed(const ConstraintProgram&, const ConstraintCursor&, const TokenByteTrie&,
                      std::span<const std::uint32_t> eos_ids = {});
std::uint32_t constrained_greedy(std::span<const float> logits, const AllowedTokens&);
}  // namespace strix::serve

