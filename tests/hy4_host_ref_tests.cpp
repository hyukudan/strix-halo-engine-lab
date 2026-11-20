// Hy4 host reference (src/hy4/ref): the synthetic tiny configuration runs end to end through every component
// (Engram, hyper-connections, sliding and indexed attention with the indexer selection, dense FFN, LatentMoE, head),
// and the invariants of the contract hold:
//   * determinism: two instances give the same logits bit for bit;
//   * sliding window: keys q-window+1 .. q (the query included): a token `window` positions back no longer matters,
//     one position closer does;
//   * indexed attention falls back to dense while kv <= index_groups * ratio: the indexer weights cannot change the
//     logits there, and do change them once selection starts;
//   * router: selection by biased key (lower id on ties), weights from the unbiased scores, sum = scale;
//   * Engram rows: the hash of the n-gram window (pad before the sequence start) inside each head's row range.
//   strix-hy4-host-ref-tests [HY4_GGUF]   (with the pack: the configuration read from its metadata)
#include "hy4/ref/hy4_host_ref.hpp"
#include "strix/weight_archive.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix::hy4;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::cout << "FAIL " << what << "\n";
}

std::vector<std::vector<float>> run(const Config& config, Weights& weights, const std::vector<std::uint32_t>& tokens) {
    HostRef ref(config, weights);
    std::vector<std::vector<float>> out;
    for (const auto t : tokens) out.push_back(ref.step(t));
    return out;
}

std::vector<std::uint32_t> sequence(std::size_t n, std::uint32_t vocab) {
    std::vector<std::uint32_t> s;
    for (std::size_t i = 0; i < n; ++i) s.push_back(static_cast<std::uint32_t>((i * 37U + 11U) % vocab));
    return s;
}

Config one_layer(bool indexed) {
    auto c = Config::tiny();
    c.layers = 1;
    c.indexed = {indexed ? std::uint8_t{1} : std::uint8_t{0}};
    c.heads = {2};
    c.dense = {1};
    c.engram_layers.clear();
    c.engram_mult.clear();
    c.head_offsets.clear();
    c.head_sizes.clear();
    return c;
}

// Shared weights::decode_row on a stacked-expert tensor [E][rows][cols] (the pack's *_exps layout): row = e * rows + r.
void rank3_rows() {
    const auto path = std::filesystem::temp_directory_path() / "strix_hy4_rank3_rows.bin";
    std::vector<float> values(2U * 3U * 4U);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) * 0.5F;
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    {
        const strix::weights::Tensor t{std::make_shared<strix::weights::MappedFile>(path), {2, 3, 4}, "F32", 0, 0,
                                       values.size() * sizeof(float)};
        bool ok = true;
        for (std::uint64_t r = 0; r < 6U; ++r) {
            const auto row = strix::weights::decode_row(t, r);
            ok = ok && row == std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(r * 4U),
                                                 values.begin() + static_cast<std::ptrdiff_t>(r * 4U + 4U));
        }
        bool threw = false;
        try {
            (void)strix::weights::decode_row(t, 6);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        check(ok && threw, "rank-3 decode_row rows / bound");
    }
    std::filesystem::remove(path);
}

// The oracle's row loops split over threads (every row its own dot, routed experts read row by row): same bits.
void thread_split() {
    const auto c = Config::tiny();
    const auto tokens = sequence(20, c.vocab);
    auto w = synthetic_weights(c, 7);
    const auto serial = run(c, *w, tokens);
    set_ref_threads(5, 1);
    const auto split = run(c, *w, tokens);
    set_ref_threads(1);
    check(serial == split, "thread split changed the logits");
}

// Config::truncated: depth n keeps the first n layers and only the Engram layers below n; full depth is the identity;
// a truncated model differs from the full one (the head follows an earlier layer); out-of-range depth throws.
void truncated_depth() {
    const auto c = Config::tiny();
    const auto tokens = sequence(12, c.vocab);
    auto w = synthetic_weights(c, 7);
    const auto t3 = c.truncated(3), t2 = c.truncated(2), t1 = c.truncated(1);
    check(run(c, *w, tokens) == run(t3, *w, tokens), "truncated(layers) must equal the full model");
    check(t2.layers == 2U && t2.engram_layers == std::vector<std::uint32_t>{1} && t2.engram_mult.size() == c.ngram &&
              t2.head_offsets.size() == c.engram_heads() && t2.head_sizes.size() == c.engram_heads(),
          "truncated(2) Engram arrays");
    check(t1.engram_layers.empty() && t1.indexed.size() == 1U && t1.heads.size() == 1U && t1.dense.size() == 1U,
          "truncated(1) arrays");
    check(run(t2, *w, tokens) != run(c, *w, tokens), "truncated(2) must differ from the full model");
    bool threw = false;
    try { (void)c.truncated(0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "truncated(0) must throw");
}

void full_model() {
    const auto c = Config::tiny();
    const auto tokens = sequence(20, c.vocab);  // positions past 8: the indexer selects groups
    auto w = synthetic_weights(c, 7);
    const auto a = run(c, *w, tokens), b = run(c, *w, tokens);
    check(a == b, "two runs differ");
    bool finite = true, varied = false;
    for (std::size_t q = 0; q < a.size(); ++q) {
        for (const auto v : a[q]) finite = finite && std::isfinite(v);
        if (q && a[q] != a[q - 1U]) varied = true;
    }
    check(finite, "non-finite logits");
    check(varied, "logits do not depend on the position");
    auto pad = tokens;
    pad[5] = c.pad_token;  // an Engram-masked position still runs the rest of the layer
    const auto p = run(c, *w, pad);
    check(p[5] != a[5], "pad token changed nothing");
}

void sliding_window() {
    const auto c = one_layer(false);
    auto w = synthetic_weights(c, 11);
    auto x = sequence(12, c.vocab), y = x;
    y[1] = (y[1] + 1U) % c.vocab;
    const auto a = run(c, *w, x), b = run(c, *w, y);
    const std::size_t last_seen = 1U + c.window - 1U;  // the query at 1 + window - 1 still has position 1 as a key
    check(a[last_seen] != b[last_seen], "position 1 must be visible from " + std::to_string(last_seen));
    for (std::size_t q = last_seen + 1U; q < a.size(); ++q) check(a[q] == b[q], "position 1 visible from " + std::to_string(q));
}

void indexed_fallback() {
    const auto c = one_layer(true);
    auto w = synthetic_weights(c, 13);
    auto v = synthetic_weights(c, 13, "indexer");
    const auto tokens = sequence(24, c.vocab);
    const auto a = run(c, *w, tokens), b = run(c, *v, tokens);
    const std::size_t dense_until = std::size_t{c.index_groups} * c.index_ratio;  // kv <= 8: dense
    for (std::size_t q = 0; q < dense_until; ++q) check(a[q] == b[q], "indexer changed a dense-fallback position " + std::to_string(q));
    bool selected_differs = false;
    for (std::size_t q = dense_until; q < a.size(); ++q) selected_differs = selected_differs || a[q] != b[q];
    check(selected_differs, "the indexer never changed a selecting position");
}

void router() {
    const auto r = route({0, 0, 0, 0}, {0, 0, 0, 1}, 2, 3.0F);
    check(r.ids == std::vector<std::uint32_t>{0, 3}, "tie order / bias selection");
    check(r.weights.size() == 2U && r.weights[0] == r.weights[1] && std::fabs(r.weights[0] + r.weights[1] - 3.0F) < 1e-6F, "weights");
    const auto s = route({2, 0, 0, 0}, {0, 0, 0, 5}, 2, 1.0F);
    const float s0 = 1.0F / (1.0F + std::exp(-2.0F)), s3 = 0.5F;
    check(s.ids == std::vector<std::uint32_t>{0, 3} && std::fabs(s.weights[0] - s0 / (s0 + s3)) < 1e-6F,
          "weights must use the unbiased scores");
}

std::vector<std::uint64_t> rows_of(const EngramRows& h, const std::vector<std::uint32_t>& tokens, std::uint32_t ordinal) {
    std::vector<std::uint64_t> rows(h.columns());
    h.at(tokens, static_cast<std::uint32_t>(tokens.size() - 1U), ordinal, rows);
    return rows;
}

void engram_hash() {
    auto c = Config::tiny();
    const std::uint32_t pad = c.pad_token;
    const EngramRows h(c);
    // window (t, t-1, t-2) = (5, pad, pad): the sequence start reads the pad id
    const auto rows = rows_of(h, {5}, 0);
    check(rows.size() == c.engram_heads() && rows == rows_of(h, {pad, pad, 5}, 0), "rows per token / start padding");
    for (std::size_t k = 0; k < rows.size(); ++k)
        check(rows[k] >= c.head_offsets[k] && rows[k] < c.head_offsets[k] + c.head_sizes[k], "row outside head " + std::to_string(k));
    // head 0 (order 2) by hand: (map[5] * m0) xor (map[pad] * m1) mod 13
    const std::int64_t p0 = std::int64_t{c.token_map[5]} * c.engram_mult[0], p1 = std::int64_t{c.token_map[pad]} * c.engram_mult[1];
    check(rows[0] == static_cast<std::uint64_t>((p0 ^ p1) % 13), "head 0 hash");
    check(rows_of(h, {6, 5}, 0) != rows, "the previous token must change the rows");
    // second Engram layer: flat-table rows (head_offsets of layer 1), order-3 head by hand
    const auto r1 = rows_of(h, {7, 6, 5}, 1);
    const std::int64_t q0 = std::int64_t{c.token_map[5]} * c.engram_mult[3], q1 = std::int64_t{c.token_map[6]} * c.engram_mult[4],
                       q2 = std::int64_t{c.token_map[7]} * c.engram_mult[5];
    check(r1[2] == c.head_offsets[6] + static_cast<std::uint64_t>((q0 ^ q1 ^ q2) % 37), "layer 1 order-3 head by hand");
    // Reference parity: ids at or past the reference table size read the pad's compressed id at every n-gram slot
    // (the pack: 120,818..120,831 of the 120,832 map); ids below it are unchanged.
    c.engram_table_size = 290;
    const EngramRows cut(c);
    for (std::uint32_t id = 290; id < c.vocab; ++id)
        for (std::uint32_t e = 0; e < 2U; ++e) {
            check(rows_of(cut, {id, 6, 5}, e) == rows_of(cut, {pad, 6, 5}, e), "unmapped id at slot t-2");
            check(rows_of(cut, {7, id, 5}, e) == rows_of(cut, {7, pad, 5}, e), "unmapped id at slot t-1");
            check(rows_of(cut, {7, 6, id}, e) == rows_of(cut, {7, 6, pad}, e), "unmapped id at slot t");
            check(rows_of(h, {7, 6, id}, e) != rows_of(h, {7, 6, pad}, e) || c.token_map[id] == c.token_map[pad],
                  "the full table keeps id " + std::to_string(id));
        }
    for (std::uint32_t id = 0; id < 290; ++id)
        check(rows_of(cut, {id, 6, 5}, 1) == rows_of(h, {id, 6, 5}, 1), "ids below the table size unchanged");
}

// Reference parity: the Engram RMSNorm eps is torch nn.RMSNorm(eps=None) = FLT_EPSILON (2^-23), not finfo(bf16).eps.
void engram_norm_eps() {
    const Config c;
    check(c.engram_norm_eps == std::numeric_limits<float>::epsilon(), "engram_norm_eps == 2^-23");
    // a constant 1e-3 row of 96: x / sqrt(mean(x^2) + eps) = 0.94525 with 2^-23 (2^-7 gave 0.0113)
    const float x = 1e-3F;
    const float y = x / std::sqrt(x * x + c.engram_norm_eps);
    check(std::fabs(y - 0.94525F) < 1e-4F, "rms of a constant 1e-3 row");
}

void pack_config(const std::string& path) {
    const auto c = Config::from_gguf(path);
    std::size_t indexed = 0;
    for (const auto v : c.indexed) indexed += v;
    check(c.layers == 42U && c.hidden == 3072U && c.vocab == 120832U, "pack dimensions");
    check(indexed == 12U && c.indexed[0] && c.indexed[1] && !c.indexed[2] && c.heads[0] == 32U && c.heads[2] == 64U, "layer types");
    check(c.engram_layers == std::vector<std::uint32_t>{1, 11, 21, 31} && c.engram_heads() == 16U && c.head_offsets.size() == 64U,
          "engram layers");
    check(c.token_map.size() == 120832U && c.engram_mult.size() == 12U, "engram tables");
    check(c.qk_nope == 192U && c.qk_rope == 64U && c.v_head == 256U && c.index_groups == 512U && c.top_k == 16U, "attention / moe");
    std::cout << "pack config: " << c.layers << " layers, " << indexed << " indexed, window " << c.window << ", routed scale "
              << c.routed_scale << ", clamp " << c.swiglu_limit << "\n";
}

// The next-token (MTP) block: it never changes the text layers' results; it attends over its own cached positions;
// truncate + re-step reproduces the same bits (draft rollback), across an indexer group boundary; misuse throws.
void next_token_block() {
    auto c = Config::tiny();
    c.nextn = 1;
    c.nextn_heads = 2;
    c.nextn_indexed = 1;
    c.validate();
    auto w = synthetic_weights(c, 17);
    const auto tokens = sequence(10, c.vocab);
    const auto plain = run(c, *w, tokens);
    HostRef ref(c, *w);
    std::vector<std::vector<float>> text, hidden;
    std::vector<HostRef::NextnStep> first;
    for (std::uint32_t p = 0; p + 1U < tokens.size(); ++p) {
        text.push_back(ref.step(tokens[p]));
        hidden.push_back(ref.last_hidden(false));
        first.push_back(ref.nextn_step(tokens[p + 1U], hidden.back(), p));
    }
    text.push_back(ref.step(tokens.back()));
    check(text == plain, "next-token steps changed the text layers' logits");
    check(ref.nextn_positions() == tokens.size() - 1U, "next-token cached positions");
    bool finite = true;
    for (const auto& f : first)
        for (const auto x : f.logits) finite = finite && std::isfinite(x);
    check(finite && first[0].hidden.size() == c.hidden, "next-token logits / hidden");
    // rollback: positions 5.. dropped (tiny ratio groups straddle 5), then the same inputs give the same bits
    ref.nextn_truncate(5);
    check(ref.nextn_positions() == 5U, "truncate(5)");
    bool same = true;
    for (std::uint32_t p = 5; p + 1U < tokens.size(); ++p) {
        const auto again = ref.nextn_step(tokens[p + 1U], hidden[p], p);
        same = same && again.logits == first[p].logits && again.hidden == first[p].hidden && again.argmax == first[p].argmax;
    }
    check(same, "truncate + re-step must reproduce the bits");
    // a chained draft (its own hidden, its own argmax) then rollback leaves the teacher path unchanged
    const auto d1 = ref.nextn_step(first.back().argmax, first.back().hidden, ref.nextn_positions());
    check(d1.logits.size() == c.vocab, "chained draft");
    ref.nextn_truncate(static_cast<std::uint32_t>(tokens.size() - 1U));
    // the block reads its own cache: a different input at position 0 changes position 3's logits
    HostRef other(c, *w);
    for (std::uint32_t p = 0; p < 4U; ++p) {
        (void)other.step(tokens[p]);
        const auto tok = p == 0U ? (tokens[1] + 1U) % c.vocab : tokens[p + 1U];
        const auto o = other.nextn_step(tok, hidden[p], p);
        if (p == 3U) check(o.logits != first[3].logits, "position 3 must see the cached position 0");
    }
    bool threw = false;
    try { (void)other.nextn_step(tokens[0], hidden[0], 7U); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a position other than the cached count must throw");
    threw = false;
    auto none = Config::tiny();
    auto wn = synthetic_weights(none, 17);
    HostRef plain_ref(none, *wn);
    (void)plain_ref.step(tokens[0]);
    try { (void)plain_ref.nextn_step(tokens[1], plain_ref.last_hidden(false), 0U); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a configuration without the block must throw");
    check(c.truncated(2).nextn == 0U, "truncated models drop the next-token block");
    threw = false;
    try { (void)w->tensor("blk." + std::to_string(c.layers) + ".hc_attn_fn.weight"); } catch (const std::out_of_range&) { threw = true; }
    check(threw, "the next-token block has no hyper-connection tensors");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        rank3_rows();
        truncated_depth();
        full_model();
        thread_split();
        sliding_window();
        indexed_fallback();
        router();
        engram_hash();
        engram_norm_eps();
        next_token_block();
        if (argc > 1) pack_config(argv[1]);
    } catch (const std::exception& error) {
        std::cout << "FAIL exception: " << error.what() << "\n";
        ++failures;
    }
    std::cout << (failures ? "FAILED " + std::to_string(failures) : std::string("OK")) << "\n";
    return failures ? 1 : 0;
}
