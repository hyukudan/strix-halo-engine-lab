#include "strix/pair/rank_image.hpp"

#include <array>
#include <cstdio>

namespace {
using namespace strix;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct Item {
    std::string name, shard;
    weight::Slice source;
    std::uint64_t resident_offset{};
};

template<class Operation>
void rejects(Operation operation, const char* message) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

std::optional<std::uint32_t> owner(std::string_view name) {
    if (name == "deferred") return std::nullopt;
    return name.starts_with("a") ? 0U : 1U;
}

} // namespace

int main() try {
    const std::array<std::byte, 4> bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    const auto first = weight::memory_source(bytes, "x.bin");
    const auto second = weight::memory_source(bytes, "y.bin");
    std::vector<Item> items{{"b", "y.bin", {second, {1U, 3U}}}, {"a", "x.bin", {first, {2U, 2U}}},
                            {"c", "x.bin", {first, {0U, 2U}}}};
    const auto image = pair::pack_rank_image(items, 16U);
    check(items[0].name == "c" && items[1].name == "a" && items[2].name == "b", "source read order");
    check(items[0].resident_offset == 0U && items[1].resident_offset == 16U && items[2].resident_offset == 32U,
          "aligned offsets");
    check(image.payload == 7U && image.resident == 48U && image.largest == 3U, "packed totals");

    std::vector<std::byte> out(image.resident, std::byte{255});
    std::array<std::byte, 2> staging{};
    pair::stream_rank_image(std::span<const Item>(items), staging, [&](std::uint64_t at, std::span<const std::byte> chunk) {
        check(chunk.size() <= staging.size(), "chunk exceeds staging");
        std::copy(chunk.begin(), chunk.end(), out.begin() + static_cast<std::ptrdiff_t>(at));
    });
    check(out[0] == std::byte{1} && out[16] == std::byte{3} && out[32] == std::byte{2} && out[34] == std::byte{4} &&
          out[2] == std::byte{255}, "streamed bytes and padding");
    rejects([&] { pair::stream_rank_image(std::span<const Item>(items), {}, [](auto, auto) {}); }, "empty staging");

    const std::array<std::string, 2> missing{"b0", "deferred"};
    pair::check_rank_scope(0U, missing, owner);
    rejects([&] { pair::check_rank_scope(1U, missing, owner); }, "own tensor absent accepted");
    rejects([&] { pair::check_rank_scope(2U, missing, owner); }, "bad rank accepted");
    check(pair::covers_rank(std::nullopt, 1U) && pair::covers_rank(0U, 0U) && !pair::covers_rank(0U, 1U), "covers_rank");

    pair::RankMemory memory;
    memory.usable = 1000U;
    memory.weight_resident = 48U;
    const std::array<pair::LayerState, 2> layers{{{10U, 3U}, {20U, 0U}}};
    pair::RankTerms terms;
    terms.alignment = 16U;
    terms.load_chunk_bytes = 100U;
    terms.largest_tensor = 3U;
    terms.row_bytes = 5U;
    terms.rows = 4U;
    terms.logits_bytes = 17U;
    terms.scratch = 1U;
    terms.reserve = 0U;
    terms.layers = layers;
    pair::account_rank_memory(memory, terms);
    check(memory.kv_payload == 30U && memory.kv_resident == 48U && memory.carry == 16U && memory.logits == 32U &&
          memory.exchange == 96U && memory.load_staging == 16U && memory.scratch == 16U && memory.reserve == 0U &&
          memory.required == 48U + 48U + 16U + 32U + 96U + 16U + 16U && memory.admitted(), "rank memory terms");
    terms.alignment = std::uint64_t{1} << 63U;
    try { auto m = memory; pair::account_rank_memory(m, terms); throw std::logic_error("overflow accepted"); }
    catch (const std::overflow_error&) {}
    std::puts("pair rank image tests passed");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
