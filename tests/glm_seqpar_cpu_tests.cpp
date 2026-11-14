#include "glm/xfer/glm_seqpar.hpp"
#include "xport/wire.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace strix::glm::xfer;
namespace {
void require(bool value) { if (!value) throw std::runtime_error("seqpar host test failed"); }
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
void split(unsigned rows) {
    const auto a = seqpar_rows(rows, 0), b = seqpar_rows(rows, 1);
    require(a.first == 0 && a.count == rows / 2 && b.first == a.count && b.count == a.count);
    require(b.first + b.count == rows);
    require(a.elements() == 0 && b.elements() == std::size_t{rows / 2} * 4096);
    require(b.elements(4 * 4096) == b.elements() * 4);
    require(seqpar_last_source(rows, 0) == 0 && seqpar_last_source(rows, 1) == rows - 1);
    require(a.bytes() == b.bytes() && a.bytes() + b.bytes() == GlmSchedule::row_bytes(rows));
    // RS sends the other half; AG sends our half. Payload per direction = AR.
    for (unsigned rank = 0; rank < 2; ++rank)
        require(seqpar_rows(rows, 1 - rank).bytes() + seqpar_rows(rows, rank).bytes() == GlmSchedule::row_bytes(rows));
}
// Simulate the exact production pointer policy at a small width, including
// signed zero and rounding-sensitive values. Poison non-owner residual/output
// storage so an incorrect gather source or destination is observable.
void roundtrip(unsigned rows, unsigned pipeline = 1) {
    constexpr unsigned width = 7;
    const std::array<float, 8> values{0.0F, -0.0F, 1.0F, -1.0F, 0x1p24F, -0x1p24F, 0x1p-24F, -0x1p-24F};
    std::array<std::vector<float>, 2> p, xn, wire;
    std::vector<float> expected(rows * width);
    for (unsigned rank = 0; rank < 2; ++rank) {
        p[rank].resize(rows * width);
        xn[rank].assign(rows * width, -999.0F);
        for (unsigned i = 0; i < rows * width; ++i) p[rank][i] = values[(i + rank * 3) % values.size()];
    }
    for (unsigned i = 0; i < rows * width; ++i) expected[i] = p[0][i] + p[1][i];
    for (unsigned tile_index = 0; tile_index < seqpar_tiles(rows, pipeline); ++tile_index) {
        const auto tile = seqpar_tile(rows, pipeline, tile_index);
        for (unsigned sender = 0; sender < 2; ++sender) {
            const auto destination = seqpar_owner(tile, 1U - sender);
            wire[sender].assign(p[sender].begin() + static_cast<std::ptrdiff_t>(destination.elements(width)),
                                p[sender].begin() + static_cast<std::ptrdiff_t>(destination.elements(width) + destination.count * width));
        }
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto own = seqpar_owner(tile, rank);
            for (unsigned r = 0; r < own.count; ++r)
                for (unsigned c = 0; c < width; ++c) {
                    const auto i = own.elements(width) + r * width + c;
                    const auto local = p[rank][i], peer = wire[1U - rank][r * width + c];
                    // Rank order is independent of owner: never local+peer on rank 1.
                    xn[rank][i] = rank == 0 ? local + peer : peer + local;
                }
        }
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto peer = seqpar_owner(tile, 1U - rank);
            for (unsigned r = 0; r < peer.count; ++r)
                for (unsigned c = 0; c < width; ++c) {
                    const auto i = peer.elements(width) + r * width + c;
                    xn[rank][i] = xn[1U - rank][i];
                }
        }
    }
    for (unsigned rank = 0; rank < 2; ++rank) {
        for (unsigned i = 0; i < rows * width; ++i)
            require(std::bit_cast<std::uint32_t>(xn[rank][i]) == std::bit_cast<std::uint32_t>(expected[i]));
    }
}
void tiled_plan(unsigned rows, unsigned pipeline) {
    const auto tiles = seqpar_tiles(rows, pipeline);
    require(tiles == (pipeline == 2 && rows % 4 == 0 ? 2U : 1U));
    std::vector<unsigned> owners(rows, 0);
    std::array<std::size_t, 2> payload{};
    unsigned next_column = 0;
    for (unsigned i = 0; i < tiles; ++i) {
        const auto tile = seqpar_tile(rows, pipeline, i);
        require(tile.first == next_column);
        next_column += tile.count;
        require(tile.count == rows / tiles);
        const auto a = seqpar_owner(tile, 0), b = seqpar_owner(tile, 1);
        require(a.first == tile.first && a.first + a.count == b.first);
        require(b.first + b.count == tile.first + tile.count && a.count == b.count);
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto own = seqpar_owner(tile, rank);
            // No row may be absent, duplicated, or owned on both ranks.
            for (unsigned r = own.first; r < own.first + own.count; ++r) {
                require(owners[r] == 0);
                owners[r] = rank + 1;
            }
            payload[rank] += 2 * own.bytes(); // one RS and one AG per tile
        }
    }
    require(next_column == rows);
    for (auto owner : owners) require(owner != 0);
    require(owners.front() == 1 && owners.back() == 2); // final-row handoff
    for (auto bytes : payload) require(bytes == GlmSchedule::row_bytes(rows));
    rejects([=] { (void)seqpar_tile(rows, pipeline, tiles); });
}

// A real direct-wire call with adjacent live owner bytes: the separate trailer
// must never land in the consumer's next row. Also check the exact payload and
// trailer bytes sent, independent of the serializer under test.
void direct_wire() {
    constexpr std::size_t n = 37;
    std::array<std::byte, n + 2> out{}, in{};
    out.fill(std::byte{0x61});
    in.fill(std::byte{0x7b});
    Trailer local{}, peer{};
    local.sequence = 19;
    local.plan = GlmSeqparTraits::kPlan;
    local.flags = GlmTraits::kDeferred | GlmTraits::kPrevious;
    local.prev_crc32c = 0xa1b2c3d4U;
    local.bytes = n;
    struct Transport {
        const Trailer& trailer;
        unsigned calls = 0;
        void exchange(std::span<const std::byte> tx, std::span<std::byte> rx) {
            require(tx.size() == rx.size());
            if (calls++ == 0) {
                require(tx.size() == n);
                for (std::size_t i = 0; i < n; ++i) {
                    require(tx[i] == std::byte{0x61});
                    rx[i] = std::byte{0x42};
                }
            } else {
                require(calls == 2 && tx.size() == sizeof(Trailer));
                require(std::memcmp(tx.data(), &trailer, sizeof(Trailer)) == 0);
                std::memcpy(rx.data(), &trailer, sizeof(Trailer));
            }
        }
    } transport{local};
    strix::xport::exchange_frame_direct(transport, out.data() + 1, in.data() + 1, n, local, peer);
    require(transport.calls == 2 && std::memcmp(&local, &peer, sizeof(Trailer)) == 0);
    require(in.front() == std::byte{0x7b} && in.back() == std::byte{0x7b});
    for (std::size_t i = 1; i <= n; ++i) require(in[i] == std::byte{0x42});
    for (auto b : out) require(b == std::byte{0x61});
}

void bandwidth_bound() {
    const auto half = seqpar_tile(8192, 2, 0);
    const auto message = seqpar_owner(half, 0).bytes();
    require(message == 33554432U); // RS and AG each, per direction
    require(2U * message == 67108864U);
    require(8U * message == 268435456U); // both halves and sublayers, per layer
    const double exchange = static_cast<double>(2U * message) / 4.6e9;
    const double compute = 14.2 / 45.0 / 2.0 / 2.0;
    require(exchange < compute);
}
}
// Exercise the production scheduler over consecutive layers. Lane state models
// publications/consumers, independently of how quickly the DMA queue progresses.
void single_stream_schedule(unsigned tiles) {
    using S = SeqparStep;
    struct Lane { unsigned gather = 0, reduce = 0, attention = 0, post = 0; };
    std::array<Lane, 2> lanes{};
    unsigned grouped = 0;
    for (unsigned layer = 0; layer < 3; ++layer) {
        unsigned attention_done = 0;
        seqpar_single_layer(tiles, [&](S step, unsigned i) {
            require(i < tiles);
            auto& lane = lanes[i];
            switch (step) {
            case S::pre_attn:
                require(lane.gather == 0 && lane.reduce == 0 && lane.post == 2 * layer);
                lane.gather = 1;
                break;
            case S::attn:
                require(lane.gather == 1 && i == attention_done++);
                lane.gather = 0;
                lane.attention = 1;
                break;
            case S::reduce_attn:
                require(lane.attention == 1 && lane.reduce == 0);
                lane.reduce = 1;
                break;
            case S::post_attn:
                require(lane.reduce == 1);
                lane.reduce = 0;
                ++lane.post;
                break;
            case S::pre_ffn:
                require(lane.post == 2 * layer + 1 && lane.gather == 0);
                lane.gather = 2;
                break;
            case S::ffn:
                require(attention_done == tiles && grouped++ == layer);
                for (unsigned j = 0; j < tiles; ++j) {
                    require(lanes[j].gather == 2 && lanes[j].reduce == 0);
                    lanes[j].gather = 0;
                }
                break;
            case S::reduce_ffn:
                require(grouped == layer + 1 && lane.reduce == 0);
                lane.reduce = 2;
                break;
            case S::post_ffn:
                require(lane.reduce == 2);
                lane.reduce = lane.attention = 0;
                ++lane.post;
                break;
            }
        });
    }
    require(grouped == 3);
    for (unsigned i = 0; i < tiles; ++i)
        require(lanes[i].post == 6 && lanes[i].gather == 0 && lanes[i].reduce == 0);
}
void moe_pieces(unsigned rows) {
    unsigned calls = 0, first = 0;
    while (first < rows) {
        const auto piece = seqpar_moe_piece(rows, first);
        require(piece.first == first && piece.count > 0 && piece.count <= 8192);
        first += piece.count;
        ++calls;
    }
    require(first == rows && calls == (rows + 8191U) / 8192U);
}
int main() {
    static_assert(seqpar_compute_streams() == 2);
    static_assert(seqpar_compute_streams("1") == 1);
    static_assert(seqpar_compute_streams("2") == 2);
    for (const char* value : {"", "0", "3", "auto", "01", "1 "})
        rejects([=] { (void)seqpar_compute_streams(value); });
    rejects([] { (void)seqpar_moe_piece(0, 0); });
    rejects([] { (void)seqpar_moe_piece(8192, 8192); });
    rejects([] { seqpar_single_layer(0, [](SeqparStep, unsigned) {}); });
    rejects([] { seqpar_single_layer(3, [](SeqparStep, unsigned) {}); });
    single_stream_schedule(1);
    single_stream_schedule(2);
    for (unsigned rows : {2U, 6U, 4096U, 8192U, 8194U, 16384U, 32768U, 65536U}) moe_pieces(rows);
    static_assert(GlmSeqparTraits::kPlan != GlmTraits::kPlan);
    static_assert(seqpar_rows(8192, 1).first == 4096);
    static_assert(seqpar_rows(8192, 0).bytes() == 64U * 1024U * 1024U);
    for (unsigned rows : {2U, 4U, 4096U, 8192U, 65536U}) split(rows);
    for (unsigned rows : {0U, 1U, 3U, 8191U}) {
        require(!seqpar_admits(rows));
        rejects([=] { (void)seqpar_rows(rows, 0); });
    }
    rejects([] { (void)seqpar_rows(8192, 2); });
    rejects([] { (void)seqpar_tile(3, 2, 0); });
    rejects([] { (void)seqpar_owner({0, 4}, 2); });
    for (unsigned rows : {2U, 6U, 128U}) roundtrip(rows);
    for (unsigned rows = 2; rows <= 256; rows += 2) {
        tiled_plan(rows, 1);
        tiled_plan(rows, 2);
        roundtrip(rows, 2);
    }
    for (unsigned rows : {1024U, 8192U, 65536U}) tiled_plan(rows, 2);
    direct_wire();
    bandwidth_bound();
    std::puts("glm seqpar host ownership/tile/payload/bitwise roundtrip/direct-wire/bandwidth/single-stream tests: PASS");
}
