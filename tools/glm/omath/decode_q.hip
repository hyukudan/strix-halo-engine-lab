// strix-glm5-decode-q (O-math, design notes §11): decodes whole EXL3 linears to their logical FP16 Q ("rotated",
// [inputs][outputs] row-major = strix_trellis.linear.dequantize_rotated) with the production decode primitive (the one gate
// D(b) checked against the codec), for the CPU's float64 Hadamard-domain products
// y = hadamard(hadamard(x * suh) @ Q) * svh. Per tensor: PREFIX.q16 in OUT_DIR (written to .tmp, then renamed) and one line
// 'PREFIX bits R C digest' on stdout, digest = gate D(b)'s band FNV of the same Q (comparable with gate_d_digest.py
// 'replicate' items).
//   strix-glm5-decode-q MODEL_DIR OUT_DIR PREFIX [PREFIX ...]      (or PREFIX = @FILE: one prefix per line)
#include "glm/engine/glm_lib.hpp"
#include "glm/engine/glm_pack.hpp"
#include "trellis_device.hpp"  // strix-trellis production decode primitive (STRIX_GLM5_TRELLIS_DIR/src, SYSTEM)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix::glm::engine::detail;

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One wave32 per 16x16 tile (gate D(b) tile_digest's decode and position map), writing Q[r][c] and the tile digest.
template <int Bits>
__global__ void tile_decode(const std::uint32_t* packed, std::uint16_t* q, std::uint64_t* tile_sum, std::uint64_t out_tiles) {
    const std::uint64_t tile = blockIdx.x;
    strix_trellis::WaveTile<Bits> data(packed + tile * Bits * 8);
    const int lane = static_cast<int>(threadIdx.x & 31U);
    const std::uint64_t C = out_tiles * 16, r_base = (tile / out_tiles) * 16, c_base = (tile % out_tiles) * 16;
    std::uint64_t acc = 0;
    for (int p = lane; p < 256; p += 32) {
        const int row = p / 16, col = p % 16;
        const int position = ((col & 7) << 5) | ((row & 6) << 2) | (row & 1) | ((row & 8) >> 2) | ((col & 8) >> 1);
        const float value = strix_trellis::codebook_value(data.state(position), 2);
        const std::uint16_t h = __half_as_ushort(__float2half_rn(value));
        const std::uint64_t index = (r_base + static_cast<std::uint64_t>(row)) * C + c_base + static_cast<std::uint64_t>(col);
        q[index] = h;
        acc += splitmix64((index << 16) | h);
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor(acc, m, 32);
    if (lane == 0) tile_sum[tile] = acc;
}

template <int Bits>
void launch(const void* packed, std::uint16_t* q, std::uint64_t* sums, std::uint64_t tiles, std::uint64_t out_tiles, hipStream_t s) {
    hipLaunchKernelGGL((tile_decode<Bits>), dim3(static_cast<unsigned>(tiles)), dim3(32), 0, s, static_cast<const std::uint32_t*>(packed), q,
                       sums, out_tiles);
}

std::uint64_t fnv_bands(const std::vector<std::uint64_t>& bands) {
    std::uint64_t h = 14695981039346656037ULL;
    for (auto b : bands)
        for (int i = 0; i < 8; ++i) h = (h ^ ((b >> (8 * i)) & 0xFF)) * 1099511628211ULL;
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::fprintf(stderr, "usage: %s MODEL_DIR OUT_DIR PREFIX [PREFIX ...] | @FILE\n", argv[0]);
            return 2;
        }
        std::vector<std::string> prefixes;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            if (a.size() > 1 && a[0] == '@') {
                std::ifstream f(a.substr(1));
                for (std::string l; std::getline(f, l);)
                    if (!l.empty()) prefixes.push_back(l);
            } else {
                prefixes.push_back(a);
            }
        }
        PackReader pack(argv[1]);
        Lib lib(pack, false);
        const std::string out_dir = argv[2];
        hipStream_t s{};
        hip_ok(hipStreamCreate(&s), "stream");
        void* dev = nullptr;
        std::uint16_t* dq = nullptr;
        std::uint64_t* sums = nullptr;
        std::uint64_t dev_bytes = 0, q_elems = 0, sum_tiles = 0;
        std::vector<std::uint64_t> h;
        std::vector<std::uint16_t> hq;
        for (const auto& prefix : prefixes) {
            const auto& m = pack.info(prefix + ".mul1");
            std::int32_t mv = 0;
            if (m.bytes != 4 || m.dtype != "I32") throw std::runtime_error("bad mul1 " + prefix);
            pack.read(m, 0, 4, &mv);
            if (static_cast<std::uint32_t>(mv) != 0x83DCD12DU) throw std::runtime_error("codebook marker is not mul1: " + prefix);
            const auto st = lib.stage_out(prefix, 0, pack.info(prefix + ".svh").shape[0]);  // whole tensor
            const std::uint64_t R = st.cols, C = st.rows, in_tiles = R / 16, out_tiles = C / 16, tiles = in_tiles * out_tiles;
            if (R % 16 || C % 16 || st.tr_bytes != tiles * st.bits * 32) throw std::runtime_error("staged shape / bytes " + prefix);
            if (st.tr_bytes > dev_bytes) {
                if (dev) hip_ok(hipFree(dev), "free");
                hip_ok(hipMalloc(&dev, st.tr_bytes), "malloc");
                dev_bytes = st.tr_bytes;
            }
            if (R * C > q_elems) {
                if (dq) hip_ok(hipFree(dq), "free");
                hip_ok(hipMalloc(reinterpret_cast<void**>(&dq), R * C * 2), "malloc q");
                q_elems = R * C;
            }
            if (tiles > sum_tiles) {
                if (sums) hip_ok(hipFree(sums), "free");
                hip_ok(hipMalloc(reinterpret_cast<void**>(&sums), tiles * 8), "malloc sums");
                sum_tiles = tiles;
            }
            h.resize(tiles);
            hq.resize(R * C);
            hip_ok(hipMemcpyAsync(dev, st.tr, st.tr_bytes, hipMemcpyHostToDevice, s), "h2d");
            switch (st.bits) {
                case 2: launch<2>(dev, dq, sums, tiles, out_tiles, s); break;
                case 3: launch<3>(dev, dq, sums, tiles, out_tiles, s); break;
                case 4: launch<4>(dev, dq, sums, tiles, out_tiles, s); break;
                case 5: launch<5>(dev, dq, sums, tiles, out_tiles, s); break;
                case 6: launch<6>(dev, dq, sums, tiles, out_tiles, s); break;
                case 8: launch<8>(dev, dq, sums, tiles, out_tiles, s); break;
                default: throw std::runtime_error("unsupported bits " + prefix);
            }
            hip_ok(hipGetLastError(), "launch");
            hip_ok(hipMemcpyAsync(h.data(), sums, tiles * 8, hipMemcpyDeviceToHost, s), "d2h sums");
            hip_ok(hipMemcpyAsync(hq.data(), dq, R * C * 2, hipMemcpyDeviceToHost, s), "d2h q");
            hip_ok(hipStreamSynchronize(s), "sync");
            std::vector<std::uint64_t> bands(in_tiles, 0);
            for (std::uint64_t t = 0; t < tiles; ++t) bands[t / out_tiles] += h[t];
            const std::string path = out_dir + "/" + prefix + ".q16";
            {
                std::ofstream f(path + ".tmp", std::ios::binary);
                f.write(reinterpret_cast<const char*>(hq.data()), static_cast<std::streamsize>(hq.size() * 2));
                if (!f) throw std::runtime_error("write " + path);
            }
            if (std::rename((path + ".tmp").c_str(), path.c_str()) != 0) throw std::runtime_error("rename " + path);
            std::printf("%s %u %llu %llu %016llx\n", prefix.c_str(), static_cast<unsigned>(st.bits), static_cast<unsigned long long>(R),
                        static_cast<unsigned long long>(C), static_cast<unsigned long long>(fnv_bands(bands)));
        }
        if (dev) hip_ok(hipFree(dev), "free");
        if (dq) hip_ok(hipFree(dq), "free");
        if (sums) hip_ok(hipFree(sums), "free");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
