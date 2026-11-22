#include "strix/state/txn.hpp"

#include "strix/attn/cache_rows.hpp"
#include "strix/graph/model_decl.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace strix::state;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Error, class Run>
void rejects(Run&& run, const char* message) {
    try {
        run();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

template<class T>
Row image(const T& value) { return std::as_bytes(std::span{&value, 1}); }

template<class T>
T load(Row bytes) {
    check(bytes.size() == sizeof(T), "wrong image size");
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}

void layout_data() {
    using namespace strix::graph;
    ModelDecl model;
    model.identity = 1207;
    model.stages.layers = {trait::attn_dense, trait::attn_dense, trait::attn_window};
    model.schedule = {{0, 0, 0}, {1, 0, 1}, {2, 0, 2}, {0, 1, 3}, {1, 1, 4}, {2, 1, 5}};
    const Decl decls[] = {
        {.id = "kv", .kind = Kind::kv_dense, .per = Per::visit, .dtype = Dtype::bf16, .dims = {3, 2},
         .planes = 2, .when = "full", .producer = 0, .block = 3, .shadow = {Dtype::e4m3_shadow, 2, 8}},
        {.id = "bank", .kind = Kind::cache_rows, .per = Per::layer, .dims = {8}, .when = "window", .block = 4},
        {.id = "index", .kind = Kind::index_pool, .dtype = Dtype::fp4_block, .dims = {128}, .group = 3,
         .owner = Owner::per_rank, .block = 3, .plane_bytes = 68, .canonical = {Dtype::f32, "keys"},
         .absent = Absent::recompute, .position_range = true},
    };
    const LayoutOptions options{.max_pending = 4, .dense_slot_budget = 64};
    const auto plan = layout(decls, model, 129, 2, options);
    check(plan.identity == layout(decls, model, 129, 2, options).identity, "unstable layout identity");
    check(plan.identity != layout(decls, model, 129, 2, {.max_pending = 4}).identity, "paging missing in identity");
    check(plan.identity != layout(decls, model, 130, 2, options).identity, "capacity missing in identity");
    check(plan.prefix_block == 12 && plan.instances.size() == 6 && plan.table_entries == 12, "wrong S1 expansion");
    const auto& kv = plan.instances[0];
    check(kv.rows == 192 && kv.pages == 3 && kv.master_bytes == 24 && kv.shadow_bytes == 16,
          "wrong paged planes");
    check(plan.instances[1].storage == 0 && plan.instances[1].slot_bytes == 0 &&
          plan.instances[3].storage == 2 && plan.instances[4].layer == 2, "wrong layer or visit aliases");
    const auto& keys = plan.instances[5];
    check(keys.row_divisor == 3 && keys.rows == 43 && keys.carry_rows == 5 && keys.position_range,
          "wrong index pool geometry");

    std::array<SlotEntry, 12> entries{};
    std::array<std::uint64_t, 12> pages{};
    check(plan.page_entries == pages.size(), "aliases consumed pages");
    const auto table = bind_table(plan, entries, pages, 1024, 2048);
    check(table.entries == entries.data() && table.states == 6 && table.slots == 2, "bad slot table ABI");
    check(entries[0].base == 2048 && entries[1].base == 2072 && entries[2].base == entries[0].base &&
          entries[0].frontier == 0 && entries[0].pending == 0 && entries[0].capacity == 129 &&
          pages[0] == 1024 && pages[1] == 1024 + 64 * kv.row_stride && pages[3] == 1024 + kv.slot_bytes,
          "bad page relocation or slot stride");
    rejects<std::invalid_argument>([&] { (void)bind_table(plan, entries, {}, 0, 0); }, "accepted short page directory");
    rejects<std::overflow_error>([&] {
        (void)bind_table(plan, entries, pages, std::numeric_limits<std::uint64_t>::max() - 3, 0);
    }, "accepted overflowing device address");

    auto custom = decls[0];
    custom.producer = -1;
    custom.when = "second pass";
    const auto second = layout({&custom, 1}, model, 129, 1, {.matches = [](std::string_view name, const LayerVisit& v) {
        return name == "second pass" && v.pass == 1;
    }});
    check(second.instances.size() == 3 && second.instances.front().visit == 3, "lost visit-sensitive selector");
    rejects<std::invalid_argument>([&] { (void)layout({&custom, 1}, model, 129, 1); }, "unknown selector was ignored");
    custom = decls[0];
    custom.producer = 2;
    rejects<std::invalid_argument>([&] { (void)layout({&custom, 1}, model, 129, 1); }, "missing producer admitted");
    custom = decls[2];
    custom.plane_bytes = 0;
    rejects<std::invalid_argument>([&] { (void)layout({&custom, 1}, model, 129, 1); }, "missing packed image geometry");
    custom = decls[1];
    std::ranges::fill(custom.dims, std::numeric_limits<std::uint32_t>::max());
    rejects<std::overflow_error>([&] { (void)layout({&custom, 1}, model, 129, 1); }, "state dimensions overflow ignored");
}

using Words = std::array<std::uint32_t, 2>;

std::uint32_t input_word(std::uint32_t position, std::uint32_t salt) {
    return std::rotl((position + 113U) * 65537U ^ salt, 7);
}

Words update_words(Words previous, std::uint32_t input, std::uint32_t position) {
    return {std::rotl(previous[0], 11) + input, std::rotl(previous[1] ^ input, 19) + position};
}

void rollback_equivalence() {
    const Decl states[] = {
        {.id = "ring", .kind = Kind::recurrent, .dims = {2}, .rollback = Rollback::snapshot},
        {.id = "replay", .kind = Kind::recurrent, .dims = {2}, .rollback = Rollback::replay,
         .replay_input_bytes = 4},
        {.id = "recompute", .kind = Kind::recurrent, .dims = {2}, .rollback = Rollback::recompute},
        {.id = "checkpoints", .kind = Kind::recurrent, .dims = {2}, .rollback = Rollback::replay,
         .replay_input_bytes = 4, .checkpoint_interval = 2},
        {.id = "window", .kind = Kind::kv_window, .dtype = Dtype::bf16, .dims = {2}, .window = 5},
        {.id = "compressed", .kind = Kind::kv_compressed, .dtype = Dtype::u32, .dims = {1}, .ratio = 3},
    };
    constexpr std::uint32_t window = 5;
    constexpr std::uint32_t context = 53;
    const auto plan = layout(states, strix::graph::ModelDecl{}, context, 2, {.max_pending = 6});
    check(plan.instances[0].rows == 7 && plan.instances[1].rows == 3 && plan.instances[2].rows == 3 &&
          plan.instances[3].rows == 6 && plan.instances[4].ring == 16, "incorrect rollback workspace");
    std::uint32_t salt = 0;
    std::array<unsigned, 4> calls{};
    std::array<Update, std::size(states)> updates;
    for (std::size_t kind = 0; kind < calls.size(); ++kind) {
        updates[kind] = [&, kind](std::uint32_t position, Row previous, Row input, std::span<std::byte> next) {
            ++calls[kind];
            const auto result = update_words(load<Words>(previous),
                kind == 2 ? input_word(position, salt) : load<std::uint32_t>(input), position);
            std::ranges::copy(image(result), next.begin());
        };
    }
    CpuState state(plan, updates);
    Words serial{};
    using Cache = std::array<std::uint16_t, 2>;
    std::vector<Cache> cache;
    std::vector<std::uint32_t> compressed;
    const auto inspect = [&](std::uint32_t frontier, std::uint32_t pending) {
        for (std::uint32_t i = 0; i < std::size(states); ++i) {
            check(state.entry(i, 0).frontier == frontier && state.entry(i, 0).pending == pending &&
                  state.entry(i, 1).frontier == 0 && state.entry(i, 1).pending == 0, "bad frontier or slot isolation");
        }
        for (std::uint32_t i = 0; i < calls.size(); ++i) {
            check(std::ranges::equal(state.current(i, 0), image(serial)), "rollback policy differs from serial bytes");
            check(load<Words>(state.current(i, 1)) == Words{}, "fixed state leaked across slots");
        }
        for (std::uint32_t p = 0; p < frontier; ++p) {
            const auto row = state.committed_row(4, 0, p);
            if (frontier - p <= window) check(load<Cache>(row) == cache[p], "committed window changed");
            else check(row.empty(), "expired window row visible");
            const auto key = state.committed_row(5, 0, p);
            if ((p + 1U) % 3U == 0 || p + 1U == frontier)
                check(load<std::uint32_t>(key) == compressed[p], "committed pool/carry changed");
            else check(key.empty(), "superseded pool carry visible");
        }
        check(state.committed_row(4, 0, frontier).empty() && state.committed_row(5, 0, frontier).empty(),
              "pending row published before commit");
    };

    Txn txn(state, 0);
    Txn other(state, 0);
    std::uint32_t frontier = 0;
    const auto exercise = [&](std::uint32_t pending, std::uint32_t accepted, bool abort) {
        ++salt;
        const auto before = calls;
        txn.begin(pending);
        rejects<std::logic_error>([&] { other.begin(1); }, "two transactions own the same slot");
        rejects<std::logic_error>([&] { state.reset(0); }, "reset touched a pending slot");
        rejects<std::logic_error>([&] { txn.commit(1); }, "unwritten row committed");
        rejects<std::invalid_argument>([&] { txn.append({}); }, "incomplete epilogue accepted");
        for (auto p = frontier; p < frontier + pending; ++p) {
            const auto word = input_word(p, salt);
            const auto value = std::array{static_cast<float>(p) * 1.375F, -static_cast<float>(salt) * 0.015625F};
            Cache row{};
            check(strix::attn::cache_rows::store_rows_ref(value.data(), 2, nullptr, 0, 1, {2, 0}, row.data()),
                  "nonfinite test input");
            const std::array<Row, std::size(states)> inputs{image(word), image(word), {}, image(word), image(row), image(word)};
            txn.append(inputs);
            inspect(frontier, pending);
            if (p < frontier + accepted) {
                // Save the independent serial inputs after checking old visibility.
                cache.push_back(row);
                compressed.push_back(word);
            }
        }
        if (abort) txn.abort();
        else txn.commit(accepted);
        for (auto p = frontier; p < frontier + accepted; ++p) serial = update_words(serial, compressed[p], p);
        frontier += accepted;
        inspect(frontier, 0);
        const auto replayed = accepted == pending ? 0U : accepted;
        check(calls[0] - before[0] == pending && calls[1] - before[1] == pending + replayed &&
              calls[2] - before[2] == pending + replayed && calls[3] - before[3] == pending + replayed % 2U,
              "unexpected replay work or checkpoint bypass");
    };
    exercise(6, 5, false);
    exercise(6, 0, true);
    exercise(6, 6, false);
    exercise(6, 2, false);
    exercise(6, 1, false);
    exercise(6, 0, false);
    exercise(6, 4, false);
    exercise(6, 6, false);
    exercise(6, 3, false);
    exercise(6, 6, false);
    exercise(6, 6, false);
    exercise(6, 6, false);
    exercise(6, 6, false);
    exercise(2, 2, false);
    check(frontier == context, "fixture did not fill admitted context");
    rejects<std::invalid_argument>([&] { txn.begin(1); }, "state capacity overrun");
    state.reset(0);
    check(state.entry(0, 0).generation == 1 && state.entry(0, 1).generation == 0 &&
          state.entry(0, 0).frontier == 0 && load<Words>(state.current(0, 0)) == Words{} &&
          state.committed_row(4, 0, 0).empty(), "slot reset leaked an old generation");
    { Txn abandoned(state, 0); abandoned.begin(1); }
    check(state.entry(0, 0).pending == 0, "abandoned window retained ownership");
}

void paged_and_failure() {
    strix::graph::ModelDecl model;
    model.schedule = {{1, 0, 0}, {0, 0, 1}};  // the alias precedes its producer
    const Decl decls[] = {
        {.id = "kv", .kind = Kind::kv_dense, .dims = {1}},
        {.id = "fixed", .kind = Kind::taps, .per = Per::visit, .dims = {1},
         .rollback = Rollback::snapshot, .producer = 0},
        {.id = "replay", .kind = Kind::recurrent, .dims = {1}, .rollback = Rollback::replay,
         .replay_input_bytes = 4},
    };
    bool fail = false;
    const Update add = [](std::uint32_t, Row previous, Row input, std::span<std::byte> next) {
        const auto value = load<std::uint32_t>(previous) + load<std::uint32_t>(input);
        std::ranges::copy(image(value), next.begin());
    };
    const std::array<Update, 4> updates{{{}, {}, add, [&](std::uint32_t p, Row previous, Row input, std::span<std::byte> next) {
        add(p, previous, input, next);
        if (fail) throw std::runtime_error("injected update failure");
    }}};
    CpuState state(layout(decls, model, 70, 2, {.max_pending = 2, .dense_slot_budget = 0}), updates);
    Txn a(state, 0);
    Txn b(state, 1);
    b.begin(1);
    const auto append = [](Txn& txn, std::uint32_t value) {
        txn.append(std::array<Row, 4>{image(value), {}, image(value), image(value)});
    };
    append(b, 101);
    for (std::uint32_t p = 0; p < 63; ++p) {
        a.begin(1);
        append(a, p + 1U);
        a.commit(1);
    }
    const auto fixed = load<std::uint32_t>(state.current(2, 0));
    a.begin(2);
    append(a, 64);
    append(a, 65);
    fail = true;
    rejects<std::runtime_error>([&] { a.commit(1); }, "partial-accept update failure did not propagate");
    check(state.entry(0, 0).pending == 0 && state.entry(0, 0).frontier == 63 &&
          load<std::uint32_t>(state.current(1, 0)) == fixed && load<std::uint32_t>(state.current(3, 0)) == fixed &&
          state.committed_row(0, 0, 63).empty() && state.entry(0, 1).pending == 1, "failed accept published state");
    a.begin(2);
    rejects<std::runtime_error>([&] { append(a, 88); }, "append update failure did not propagate");
    check(state.entry(0, 0).pending == 0 && state.entry(0, 0).frontier == 63, "failed append left pending rows");
    fail = false;
    a.begin(2);
    append(a, 164);
    append(a, 165);
    a.commit(2);
    b.commit(1);
    check(load<std::uint32_t>(state.committed_row(0, 0, 63)) == 164 &&
          load<std::uint32_t>(state.committed_row(0, 0, 64)) == 165 &&
          load<std::uint32_t>(state.committed_row(0, 1, 0)) == 101 &&
          state.entry(1, 0).base == state.entry(2, 0).base &&
          load<std::uint32_t>(state.current(1, 0)) == fixed + 164U + 165U, "page, slot or fixed alias lost data");
}

}  // namespace

int main() {
    try {
        layout_data();
        rollback_equivalence();
        paged_and_failure();
        std::cout << "state_txn: ring = replay = recompute = serial (bitwise)\n"
                     "state_txn: page, window, carry, alias and slot checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "state_txn: " << error.what() << '\n';
        return 1;
    }
}
