// Hy4-flash placement on the shared planners (host only): the pack's component table (loader claims summed per layer and
// component, tests/data/hy4/q4km-component-bytes.tsv) -> shared placement inputs -> one-node / layer-chain / tensor-parallel
// decode plans. Numbers checked here are placement plan arithmetic (PREDICTIONS, not measurements).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "strix/hy4/placement.hpp"
#include "strix/qwen35/pair_plan.hpp"

namespace {
namespace hp = strix::hy4::placement;
namespace place = strix::pair::place;
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++failures; }
}
bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
constexpr double GiB = 1073741824.0;

struct Table {
    std::vector<std::string> names;   // owns the component strings the records view
    std::vector<place::Record> records;
};
Table load(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    Table t;
    std::vector<std::pair<std::int64_t, double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        std::int64_t layer = 0;
        std::string component;
        double bytes = 0.0;
        if (!(s >> layer >> component >> bytes)) throw std::runtime_error("bad fixture line: " + line);
        t.names.push_back(component);
        rows.emplace_back(layer, bytes);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) t.records.push_back({rows[i].first, t.names[i], rows[i].second});
    return t;
}

void test_owner_rule_and_qwen_identity() {
    check(place::block_of("blk.12.attn_q.weight") == 12, "block_of layer");
    check(place::block_of("token_embd.weight") == -1, "block_of global");
    bool threw = false;
    try { (void)place::block_of("blk.x.weight"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "block_of malformed throws");
    threw = false;
    try { (void)place::block_of("blk.7"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "block_of without dot throws");
    // qwen35 layer_table binds the shared owner rule: same sums as before (MTP block 2 >= layers left out).
    struct T { std::string name; std::uint64_t byte_size; };
    const std::vector<T> ts{{"blk.0.a", 10}, {"blk.1.b", 20}, {"blk.1.c", 5}, {"blk.2.mtp", 99}, {"token_embd.weight", 7},
                            {"output.weight", 3}, {"output_norm.weight", 1}, {"rope_freqs", 50}};
    const auto lt = strix::qwen35::pair_plan::layer_table(ts, 2U);
    check(lt.weight_bytes.size() == 2U && lt.weight_bytes[0] == 10.0 && lt.weight_bytes[1] == 25.0, "qwen35 layer bytes");
    check(lt.embed_bytes == 7.0 && lt.head_bytes == 4.0, "qwen35 embed/head");
}

void test_inputs(const Table& t, const hp::Facts& f) {
    const auto in = hp::inputs(t.records, f);
    double all = place::sum(in.global) + place::sum(in.beyond) + in.trunk_bytes();
    check(all == 168341638068.0, "every byte is placed exactly once");
    check(near(in.trunk_bytes() / GiB, 89.740, 0.005), "trunk 89.74 GiB");
    check(near(in.total(place::Role::experts) / GiB, 83.347, 0.005), "expert banks 83.35 GiB");
    check(near(in.total(place::Role::split) / GiB, 4.707 + 0.700, 0.01), "split roles (heads, FFN slices)");
    check(near(in.total(place::Role::head) / GiB, 0.692, 0.001), "head role (vocabulary split)");
    check(place::at(in.global, place::Role::lookup) == 63360615936.0, "Engram table bytes");
    check(near(place::sum(in.beyond) / GiB, 6.972, 0.005), "next-token block kept apart");
    const auto st = hp::state_layout(t.records, f);
    std::uint32_t sparse = 0;
    for (const auto v : st.per_row) sparse += v > 0.0 ? 1U : 0U;
    check(sparse == 12U, "12 sparse-attention layers (indexer)");
    check(st.seam_bytes_per_row == 24576.0, "seam = 4 bf16 streams");
    check(hp::layers_with(t.records, "engram.proj", f) == std::vector<std::uint32_t>({1U, 11U, 21U, 31U}), "Engram layers");
    // Decode read per token, one row: 10.40 GB (attention 5.05 + experts 2.80 + head 0.74 + rest).
    const auto active = place::expert_union_fraction(f.top_k, f.experts, 1U);
    check(active == 16.0 / 512.0, "one row touches top_k / experts");
    const auto shape = place::chain_shape(in, st, active, {}, hp::lookup_reads(t.records, f));
    double read = shape.head_bytes;
    for (const auto r : shape.read_bytes_per_token) read += r;
    check(near(read / 1e9, 10.40, 0.02), "decode bytes per token 10.40 GB");
    // Verify rows: the expert union grows sublinearly (upper bound under independent routes).
    double prev = active;
    for (std::uint32_t rows = 2; rows <= 6; ++rows) {
        const auto u = place::expert_union_fraction(f.top_k, f.experts, rows);
        check(u > prev && u < rows * active, "union grows, below rows x k/E");
        prev = u;
    }
    check(near(prev * f.experts, 88.8, 0.1), "6 rows touch <= 88.8 experts per layer");
    // One node cannot keep the Engram table resident under the GTT wall: the chain refuses it.
    const auto slices = hp::lookup_slices(in, t.records, f);
    const auto resident = place::chain_shape(in, st, active, slices, {});
    strix::pair::chain::Device d;
    d.name = "node";
    d.layer_ms_per_row.assign(f.layers, 0.01);
    d.capacity_bytes = 108.0 * GiB;
    d.read_bytes_per_ms = 213e6;
    const auto p = strix::pair::chain::evaluate({d}, {}, resident, 4096U, 0U, 0U, {0U, f.layers}, 4096U);
    check(!p.valid, "one node: Engram resident refused");
    const auto trips = hp::tp_trips(t.records, f, 2U);
    check(trips.size() == 86U, "86 exchanges per tensor-parallel token");
}

void test_plans(const Table& t, const hp::Facts& f) {
    const hp::Node node{213e6, 108.0 * GiB, 445.0};
    const strix::pair::tp::TripCost link{0.062, 4.58e6};
    const hp::Budget b{4.2, 131072.0, 2.0 * GiB, false};
    const auto two = hp::decode_plans(t.records, f, node, 2U, link, b);
    check(two.size() == 3U, "three plans for two nodes");
    for (const auto& r : two)
        std::printf("plan %-30s ranks=%u valid=%d decode_ms=%.2f tok_s=%.1f resident_GiB=%.2f %s\n", r.name.c_str(), r.ranks, r.valid ? 1 : 0,
                    r.decode_ms, 1000.0 / r.decode_ms, r.resident_bytes_per_rank / GiB, r.reason.c_str());
    check(two[0].valid && near(two[0].decode_ms, 53.0, 0.6), "one node ~53 ms/token");
    check(two[1].valid && two[1].ranks == 2U && two[1].decode_ms > two[0].decode_ms, "layer chain cannot beat one node in decode");
    check(two[2].valid && two[2].ranks == 2U && near(two[2].decode_ms, 36.8, 0.6), "tensor parallel x2 ~36.8 ms/token");
    check(two[2].resident_bytes_per_rank / GiB < 108.0 - 2.0, "tensor parallel x2 fits a node");
    // Four nodes: expert slices are 6 blocks of 256, so a 4th rank adds nothing over 3 (shares 2,2,1,1).
    const auto four = hp::decode_plans(t.records, f, node, 4U, link, b);
    check(four.back().ranks == 3U && four.back().decode_ms < two[2].decode_ms, "four nodes: 3 tensor-parallel ranks");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: strix-hy4-placement-tests FIXTURE.tsv\n"); return 2; }
    try {
        const auto t = load(argv[1]);
        const hp::Facts f{};
        test_owner_rule_and_qwen_identity();
        test_inputs(t, f);
        test_plans(t, f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::printf("PASS hy4 placement\n");
    return 0;
}
