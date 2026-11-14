#include "h3/dist/image_parallel.hpp"
#include "h3/dist/host_attn.hpp"
#include "h3/dist/kv_exchange.hpp"
#include "h3/vae/image.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <thread>

namespace d = strix::h3::dist;
namespace a = strix::h3::attn;
namespace v = strix::h3::vae;
namespace {
unsigned assertions = 0;
void require(bool ok) { ++assertions; if (!ok) throw std::runtime_error("pair host assertion " + std::to_string(assertions)); }
template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::invalid_argument&) { threw = true; }
    require(threw);
}
void ownership() {
    for (unsigned n : {1U, 2U, 3U, 8U}) {
        std::vector<unsigned> seen(n);
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto w = d::image_work(d::ImageParallel::automatic, n, 42, 2, rank);
            require(w.sequence == (n == 1));
            for (auto i : w.requests) { ++seen.at(i); require(n == 1 || i % 2 == rank); }
        }
        for (auto count : seen) require(count == (n == 1 ? 2U : 1U));
    }
    require(d::image_work(d::ImageParallel::request, 1, 42, 2, 1).requests.empty());
    require(d::image_work(d::ImageParallel::automatic, 3, 42, 1, 0).requests.size() == 3);
    require(d::image_work(d::ImageParallel::sequence, 3, 42, 2, 1).requests.size() == 3);
    rejects([] { d::image_work(d::ImageParallel::sequence, 1, 42, 1, 0); });
    for (auto count : {0U, 4097U}) rejects([&] { d::image_work(d::ImageParallel::automatic,count,42,2,0); });
    rejects([] { d::image_work(d::ImageParallel::automatic,2,UINT32_MAX,2,0); });
    rejects([] { d::image_work(d::ImageParallel::automatic,1,42,3,0); });
    rejects([] { d::image_work(d::ImageParallel::automatic,1,42,2,2); });
    rejects([] { d::image_parallel("cfg"); });
}
void kv_layout(unsigned S) {
    const auto sp = d::Split::of(S);
    require(sp.cut % 64 == 0 && sp.rows(0) + sp.rows(1) == S);
    require(sp.message_bytes() % 64 == 0 && sp.vt_pitch() >= sp.max_rows());
    const auto reference = a::make_qkv(2, 2, S, 4242, 0.2F);
    const std::array<unsigned,2> heads{3,55};
    std::vector<std::uint16_t> tx[2], rx[2];
    for (unsigned rank = 0; rank < 2; ++rank) {
        tx[rank].resize(sp.message_bytes()/2,0);
        rx[rank].resize(tx[rank].size());
        auto* k = tx[rank].data(); auto* vt = k + sp.k_bytes()/2;
        for (unsigned h = 0; h < heads.size(); ++h)
            for (unsigned r = 0; r < sp.rows(rank); ++r) for (unsigned c = 0; c < 128; ++c) {
                k[std::size_t(r)*d::kInner + heads[h]*128+c] = reference.kv(h,sp.first(rank)+r,c);
                vt[(std::size_t(heads[h])*128+c)*sp.vt_pitch()+r] = reference.vv(h,sp.first(rank)+r,c);
            }
        // Every padding word must be defined, including unequal K rows and V pitch.
        bool zero = true;
        for (auto r = sp.rows(rank); r < sp.max_rows(); ++r)
            for (unsigned c = 0; c < d::kInner; ++c) zero &= k[std::size_t(r)*d::kInner+c] == 0;
        for (unsigned c = 0; c < d::kInner; ++c)
            for (auto r = sp.rows(rank); r < sp.vt_pitch(); ++r) zero &= vt[std::size_t(c)*sp.vt_pitch()+r] == 0;
        require(zero);
    }
    d::MemoryKvPair pair(std::chrono::microseconds(40), 4);
    std::exception_ptr errors[2];
    std::thread ranks[2];
    for (unsigned rank = 0; rank < 2; ++rank) ranks[rank] = std::thread([&,rank] {
        try {
            pair.endpoint(rank).start(reinterpret_cast<const std::byte*>(tx[rank].data()),sp.message_bytes(),49);
            pair.endpoint(rank).finish(reinterpret_cast<std::byte*>(rx[rank].data()));
        } catch (...) { errors[rank]=std::current_exception(); pair.endpoint(rank).abort("CPU test failed"); }
    });
    for (auto& rank : ranks) rank.join();
    for (auto& error : errors) if (error) std::rethrow_exception(error);
    require(rx[0] == tx[1] && rx[1] == tx[0]);
    for (unsigned rank = 0; rank < 2; ++rank) for (unsigned h = 0; h < heads.size(); ++h)
        for (unsigned q = 0; q < 2; ++q) {
            a::Partial partials[2];
            // Local first, remote second; rank 1 still merges block 0 before block 1.
            for (unsigned block : {rank, 1U-rank}) {
                const auto& message = block == rank ? tx[rank] : rx[rank];
                const d::KeyView kv{message.data(),message.data()+sp.k_bytes()/2,sp.vt_pitch(),sp.rows(block)};
                partials[block] = d::host_partial(reference.q.data()+(h*2+q)*128,kv,heads[h]);
                const auto exact = a::partial_row_f32(reference,h,q,sp.first(block),sp.first(block)+sp.rows(block));
                require(partials[block].mc == exact.mc && partials[block].l == exact.l);
                require(std::memcmp(partials[block].o,exact.o,sizeof(exact.o)) == 0);
            }
            float merged[128]; double fp64[128];
            a::merge_f32(partials,2,merged); a::reference_row_f64(reference,h,q,fp64);
            require(a::row_error(merged,fp64) < 0.02);
        }
}
float tile_value(std::size_t tile, int c, int y, int x) {
    return float((tile*17 + std::size_t(c)*13 + y*3+x)%511)/256.0F;
}
void tile_layout(int H, int W) {
    const auto ty = v::split_tiles(H), tx = v::split_tiles(W);
    std::vector<std::size_t> sizes;
    std::vector<std::vector<float>> raw;
    for (auto h : ty.len) for (auto w : tx.len) {
        const auto index = raw.size();
        sizes.push_back(std::size_t(3)*h*w);
        std::vector<float> tile(sizes.back());
        for (int c = 0; c < 3; ++c) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
            tile[(std::size_t(c)*h+y)*w+x] = tile_value(index,c,y,x);
        raw.push_back(std::move(tile));
    }
    const d::ImageTileExchange layout(sizes);
    const auto rank0 = layout.pack(0,raw), rank1 = layout.pack(1,raw);
    const auto merged = layout.merge(rank0,rank1);
    require(merged == raw);
    const auto pixels = v::assemble_image_tiles(H,W,[&](std::size_t i,int,int) { return merged.at(i); });
    bool exact = true;
    // Independent pixel oracle: owner is the last tile origin <= coordinate.
    for (int c = 0; c < 3; ++c) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        const auto iy = std::size_t(std::upper_bound(ty.start.begin(),ty.start.end(),y)-ty.start.begin()-1);
        const auto ix = std::size_t(std::upper_bound(tx.start.begin(),tx.start.end(),x)-tx.start.begin()-1);
        const auto tile = iy*tx.start.size()+ix;
        const int yy = y-ty.start[iy], xx = x-tx.start[ix];
        float value = tile_value(tile,c,yy,xx);
        if (iy && yy < ty.overlap[iy-1]) {
            const float weight = float(yy)/float(ty.overlap[iy-1]);
            value = tile_value(tile-tx.start.size(),c,ty.len[iy-1]-ty.overlap[iy-1]+yy,xx)*(1-weight)+value*weight;
        }
        if (ix && xx < tx.overlap[ix-1]) {
            const float weight = float(xx)/float(tx.overlap[ix-1]);
            value = tile_value(tile-1,c,yy,tx.len[ix-1]-tx.overlap[ix-1]+xx)*(1-weight)+value*weight;
        }
        exact &= pixels[(std::size_t(c)*H+y)*W+x] == value;
    }
    require(exact);
    auto bad = rank1; bad.push_back(0);
    rejects([&] { layout.merge(rank0,bad); });
    auto bad_tiles = raw; bad_tiles.back().clear();
    rejects([&] { layout.pack(1,bad_tiles); });
}
}
int main() try {
    ownership();
    for (unsigned S : {129U,257U,300U}) kv_layout(S);
    require(d::Split::of(1280).message_bytes() == 18350080);
    require(d::Split::of(4352).message_bytes() == 62390272);
    for (const auto shape : {std::array{32,32},std::array{256,512},std::array{1024,1024},std::array{2048,2048}})
        tile_layout(shape[0],shape[1]);
    const d::ImageTileExchange empty({});
    require(empty.merge(empty.pack(0,{}),empty.pack(1,{})).empty());
    rejects([] { d::ImageTileExchange bad({0}); });
    rejects([] { d::ImageTileExchange bad({SIZE_MAX}); });
    std::cout << assertions << " pair host assertions passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
