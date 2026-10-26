// O-math M0 (design notes §4): fp64 GEMM rate on gfx1151, hipBLAS (rocBLAS Tensile) vs a plain tiled kernel.
// C[M][N] = A[M][K] @ B[K][N], row-major (hipBLAS called column-major on the transposed problem). Prints TFLOP/s per
// shape and the bitwise run-to-run identity of each path (determinism, plan §6.6); max |hipBLAS - tiled| / max |C|.
// usage: m0_dgemm M,N,K [M,N,K ...]
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { if ((x) != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", #x, __LINE__); std::exit(1); } } while (0)

constexpr int kT = 16;  // 16 x 16 threads, 64 x 64 tile, 4 x 4 per thread
__global__ void dgemm_tiled(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double As[64][kT + 1], Bs[kT][64 + 1];
    const int tx = threadIdx.x, ty = threadIdx.y, m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    double acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += kT) {
        for (int i = ty; i < 64; i += kT) {
            const int m = m0 + i, k = k0 + tx;
            As[i][tx] = m < M && k < K ? A[static_cast<size_t>(m) * K + k] : 0.0;
        }
        for (int i = ty; i < kT; i += kT)
            for (int j = tx; j < 64; j += kT) {
                const int k = k0 + i, n = n0 + j;
                Bs[i][j] = k < K && n < N ? B[static_cast<size_t>(k) * N + n] : 0.0;
            }
        __syncthreads();
        for (int k = 0; k < kT; ++k)
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) acc[a][b] = fma(As[ty + a * kT][k], Bs[k][tx + b * kT], acc[a][b]);
        __syncthreads();
    }
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            const int m = m0 + ty + a * kT, n = n0 + tx + b * kT;
            if (m < M && n < N) C[static_cast<size_t>(m) * N + n] = acc[a][b];
        }
}

int main(int argc, char** argv) {
    hipblasHandle_t h;
    if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "hipblasCreate failed\n"); return 1; }
    hipblasSetAtomicsMode(h, HIPBLAS_ATOMICS_NOT_ALLOWED);
    for (int a = 1; a < argc; ++a) {
        int M, N, K;
        if (std::sscanf(argv[a], "%d,%d,%d", &M, &N, &K) != 3) return 2;
        const size_t na = size_t(M) * K, nb = size_t(K) * N, nc = size_t(M) * N;
        std::vector<double> hA(na), hB(nb);
        std::mt19937_64 g(a);
        std::normal_distribution<double> d;
        for (auto& x : hA) x = d(g);
        for (auto& x : hB) x = d(g);
        double *A, *B, *C;
        CK(hipMalloc(&A, na * 8)); CK(hipMalloc(&B, nb * 8)); CK(hipMalloc(&C, nc * 8));
        CK(hipMemcpy(A, hA.data(), na * 8, hipMemcpyHostToDevice));
        CK(hipMemcpy(B, hB.data(), nb * 8, hipMemcpyHostToDevice));
        const double flop = 2.0 * M * N * K;
        std::vector<double> c1(nc), c2(nc), ct(nc);
        auto timed = [&](auto&& f, std::vector<double>& out) {
            f(); CK(hipDeviceSynchronize());  // warm-up
            const auto t0 = std::chrono::steady_clock::now();
            f(); CK(hipDeviceSynchronize());
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            CK(hipMemcpy(out.data(), C, nc * 8, hipMemcpyDeviceToHost));
            return s;
        };
        const double one = 1.0, zero = 0.0;
        auto blas = [&] {  // row-major C = A B  <=>  column-major C^T = B^T A^T
            if (hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N) != HIPBLAS_STATUS_SUCCESS)
                std::exit(3);
        };
        auto tiled = [&] {
            dgemm_tiled<<<dim3((N + 63) / 64, (M + 63) / 64), dim3(kT, kT)>>>(A, B, C, M, N, K);
        };
        const double sb = timed(blas, c1);
        timed(blas, c2);
        const bool det_b = std::memcmp(c1.data(), c2.data(), nc * 8) == 0;
        const double st = timed(tiled, ct);
        timed(tiled, c2);
        const bool det_t = std::memcmp(ct.data(), c2.data(), nc * 8) == 0;
        double md = 0, mc = 0;
        for (size_t i = 0; i < nc; ++i) { md = std::fmax(md, std::fabs(c1[i] - ct[i])); mc = std::fmax(mc, std::fabs(c1[i])); }
        std::printf("m0 gpu dgemm M %d N %d K %d: hipblas %.3f s %.3f TFLOP/s det %s | tiled %.3f s %.3f TFLOP/s det %s | max diff/max %.2e\n",
                    M, N, K, sb, flop / sb / 1e12, det_b ? "yes" : "NO", st, flop / st / 1e12, det_t ? "yes" : "NO", md / mc);
        std::fflush(stdout);
        CK(hipFree(A)); CK(hipFree(B)); CK(hipFree(C));
    }
    hipblasDestroy(h);
    return 0;
}
