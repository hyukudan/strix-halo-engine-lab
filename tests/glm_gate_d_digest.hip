// strix-glm5-gate-d-digest: gate D (b), device side. For every item of the CPU reference file written by
// tools/glm/gate_d_digest.py ('<prefix> <rank> <mode> <a> <b> <bits> <R> <C> <digest>'), stages the rank slice through
// the engine's own Lib::stage_out / stage_in (the loader's slicing path: 'out' -> stage_out(a, b - a), 'in' ->
// stage_in(a, b - a), 'replicate' -> stage_out(0, N) = Lib::whole), decodes the staged trellis on the GPU with strix-trellis'
// production primitive (trellis_device.hpp WaveTile + codebook_value, mul1; position/index mapping of the library's
// STGPU001 decoded_weights probe) and recomputes the same digest of the logical FP16 Q slice: band sums over 16-row tile
// rows of splitmix64(((r * C + c) << 16) | q), FNV-1a 64 over the bands. A single changed element always changes its
// band sum; the compared 64-bit digest can still collide (probabilistic evidence, not a proof). Every item's '.mul1' must
// hold 0x83DCD12D (mul1).
//   strix-glm5-gate-d-digest MODEL_DIR CPU.tsv GPU.tsv EXPECT [DEADLINE_S]
// Exit 0 only if the file has exactly EXPECT unique items, every item ran and every digest matches; 1 = mismatch or
// error, 3 = deadline reached (incomplete coverage).
#include "glm/engine/glm_lib.hpp"
#include "trellis_device.hpp"  // strix-trellis production decode primitive (STRIX_GLM5_TRELLIS_DIR/src, SYSTEM)

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace strix::glm::engine::detail;
using Clock = std::chrono::steady_clock;

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One wave32 per 16x16 tile of the staged [in_tiles][out_tiles][Bits * 8 words] slice; C = slice output width.
template <int Bits>
__global__ void tile_digest(const std::uint32_t* packed, std::uint64_t* tile_sum, std::uint64_t out_tiles) {
    const std::uint64_t tile = blockIdx.x;
    strix_trellis::WaveTile<Bits> data(packed + tile * Bits * 8);
    const int lane = static_cast<int>(threadIdx.x & 31U);
    const std::uint64_t C = out_tiles * 16, r_base = (tile / out_tiles) * 16, c_base = (tile % out_tiles) * 16;
    std::uint64_t acc = 0;
    for (int p = lane; p < 256; p += 32) {
        const int row = p / 16, col = p % 16;
        const int position = ((col & 7) << 5) | ((row & 6) << 2) | (row & 1) | ((row & 8) >> 2) | ((col & 8) >> 1);
        const float value = strix_trellis::codebook_value(data.state(position), 2);
        const std::uint64_t q = __half_as_ushort(__float2half_rn(value));
        const std::uint64_t index = (r_base + static_cast<std::uint64_t>(row)) * C + c_base + static_cast<std::uint64_t>(col);
        acc += splitmix64((index << 16) | q);
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor(acc, m, 32);
    if (lane == 0) tile_sum[tile] = acc;
}

template <int Bits>
void launch(const void* packed, std::uint64_t* sums, std::uint64_t tiles, std::uint64_t out_tiles, hipStream_t s) {
    hipLaunchKernelGGL((tile_digest<Bits>), dim3(static_cast<unsigned>(tiles)), dim3(32), 0, s,
                       static_cast<const std::uint32_t*>(packed), sums, out_tiles);
}

std::uint64_t fnv_bands(const std::vector<std::uint64_t>& bands) {
    std::uint64_t h = 14695981039346656037ULL;
    for (auto b : bands)
        for (int i = 0; i < 8; ++i) h = (h ^ ((b >> (8 * i)) & 0xFF)) * 1099511628211ULL;
    return h;
}

struct Item {
    std::string prefix, mode;
    unsigned rank = 0, bits = 0;
    std::uint64_t a = 0, b = 0, R = 0, C = 0, digest = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::fprintf(stderr, "usage: %s MODEL_DIR CPU.tsv GPU.tsv EXPECT [DEADLINE_S]\n", argv[0]);
            return 2;
        }
        const std::size_t expect = std::stoull(argv[4]);
        const double deadline = argc > 5 ? std::stod(argv[5]) : 400.0;
        std::vector<Item> items;
        {
            std::ifstream f(argv[2]);
            if (!f) throw std::runtime_error(std::string("open ") + argv[2]);
            std::string line;
            std::set<std::tuple<std::string, unsigned, std::uint64_t, std::uint64_t>> seen;
            while (std::getline(f, line)) {
                std::istringstream ls(line);
                Item it;
                std::string hex;
                if (!(ls >> it.prefix >> it.rank >> it.mode >> it.a >> it.b >> it.bits >> it.R >> it.C >> hex))
                    throw std::runtime_error("bad line: " + line);
                it.digest = std::stoull(hex, nullptr, 16);
                if (!seen.insert({it.prefix, it.rank, it.a, it.b}).second) throw std::runtime_error("duplicate item: " + line);
                items.push_back(it);
            }
            if (f.bad()) throw std::runtime_error("read error");
        }
        if (items.size() != expect) throw std::runtime_error("item count " + std::to_string(items.size()) + " != expected " + std::to_string(expect));

        PackReader pack(argv[1]);
        Lib lib(pack, false);
        std::ofstream out(argv[3]);
        if (!out) throw std::runtime_error(std::string("open ") + argv[3]);
        hipStream_t s{};
        hip_ok(hipStreamCreate(&s), "stream");
        void* dev = nullptr;
        std::uint64_t* sums = nullptr;
        std::uint64_t dev_bytes = 0, sum_tiles = 0;
        std::vector<std::uint64_t> h;
        const auto t0 = Clock::now();
        std::size_t done = 0, bad = 0;
        double gib = 0;
        for (const auto& it : items) {
            if (std::chrono::duration<double>(Clock::now() - t0).count() > deadline) break;
            if (it.b <= it.a) throw std::runtime_error("empty range " + it.prefix);
            {
                const auto& m = pack.info(it.prefix + ".mul1");
                if (m.bytes != 4 || m.dtype != "I32") throw std::runtime_error("bad mul1 " + it.prefix);
                std::int32_t mv = 0;
                pack.read(m, 0, 4, &mv);
                if (static_cast<std::uint32_t>(mv) != 0x83DCD12DU) throw std::runtime_error("codebook marker is not mul1: " + it.prefix);
            }
            const auto st = it.mode == "in" ? lib.stage_in(it.prefix, it.a, it.b - it.a)
                          : it.mode == "out" ? lib.stage_out(it.prefix, it.a, it.b - it.a)
                          : it.mode == "replicate" ? lib.stage_out(it.prefix, 0, pack.info(it.prefix + ".svh").shape[0])
                                                   : throw std::runtime_error("unknown mode " + it.mode);
            // Staged: rows = outputs, cols = inputs; the slice Q_s is [inputs][outputs] = [R][C].
            if (st.cols != it.R || st.rows != it.C || st.bits != it.bits || it.R % 16 || it.C % 16 || it.R == 0 || it.C == 0)
                throw std::runtime_error("slice shape/bits mismatch vs CPU " + it.prefix);
            const std::uint64_t in_tiles = it.R / 16, out_tiles = it.C / 16, tiles = in_tiles * out_tiles;
            if (st.tr_bytes != tiles * it.bits * 32 || it.R * it.C >= (std::uint64_t{1} << 47) || tiles >= (std::uint64_t{1} << 31))
                throw std::runtime_error("staged byte count / bounds " + it.prefix);
            if (st.tr_bytes > dev_bytes) {
                if (dev) hip_ok(hipFree(dev), "free");
                hip_ok(hipMalloc(&dev, st.tr_bytes), "malloc");
                dev_bytes = st.tr_bytes;
            }
            if (tiles > sum_tiles) {
                if (sums) hip_ok(hipFree(sums), "free");
                hip_ok(hipMalloc(reinterpret_cast<void**>(&sums), tiles * 8), "malloc sums");
                sum_tiles = tiles;
                h.resize(tiles);
            }
            hip_ok(hipMemcpyAsync(dev, st.tr, st.tr_bytes, hipMemcpyHostToDevice, s), "h2d");
            switch (it.bits) {
                case 2: launch<2>(dev, sums, tiles, out_tiles, s); break;
                case 3: launch<3>(dev, sums, tiles, out_tiles, s); break;
                case 4: launch<4>(dev, sums, tiles, out_tiles, s); break;
                case 5: launch<5>(dev, sums, tiles, out_tiles, s); break;
                case 6: launch<6>(dev, sums, tiles, out_tiles, s); break;
                case 8: launch<8>(dev, sums, tiles, out_tiles, s); break;
                default: throw std::runtime_error("unsupported bits " + it.prefix);
            }
            hip_ok(hipGetLastError(), "launch");
            hip_ok(hipMemcpyAsync(h.data(), sums, tiles * 8, hipMemcpyDeviceToHost, s), "d2h");
            hip_ok(hipStreamSynchronize(s), "sync");
            std::vector<std::uint64_t> bands(in_tiles, 0);
            for (std::uint64_t t = 0; t < tiles; ++t) bands[t / out_tiles] += h[t];
            const std::uint64_t d = fnv_bands(bands);
            char line[96];
            std::snprintf(line, sizeof line, " %u %s %llu %llu %u %llu %llu %016llx\n", it.rank, it.mode.c_str(),
                          static_cast<unsigned long long>(it.a), static_cast<unsigned long long>(it.b), it.bits,
                          static_cast<unsigned long long>(it.R), static_cast<unsigned long long>(it.C), static_cast<unsigned long long>(d));
            out << it.prefix << line;
            if (d != it.digest) {
                if (bad < 10) std::printf("MISMATCH %s rank %u %s %llu:%llu gpu %016llx cpu %016llx\n", it.prefix.c_str(), it.rank,
                                          it.mode.c_str(), static_cast<unsigned long long>(it.a), static_cast<unsigned long long>(it.b),
                                          static_cast<unsigned long long>(d), static_cast<unsigned long long>(it.digest));
                ++bad;
            }
            ++done;
            gib += static_cast<double>(st.tr_bytes) / 1073741824.0;
        }
        out.flush();
        if (!out) throw std::runtime_error("write error on GPU.tsv");
        lib.release_staging();
        if (sums) hip_ok(hipFree(sums), "free");
        if (dev) hip_ok(hipFree(dev), "free");
        const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        const bool complete = done == items.size();
        std::printf("gate-d-digest: %zu / %zu items, %zu mismatches, %.2f GiB staged trellis, %.1f s -> %s\n", done, items.size(), bad,
                    gib, sec, bad ? "FAIL" : complete ? "PASS" : "INCOMPLETE (deadline)");
        return bad ? 1 : complete ? 0 : 3;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
