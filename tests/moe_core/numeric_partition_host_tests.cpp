// Numeric partition policy v1: the shared constexpr functions (src/core/numeric_partition.hpp) against frozen scalar
// copies of the forms they replace: the Weights templates of the previous header, the rows-dependent split rule of
// linear_contract, and the MiniMax route policy (previous gemm_splits / gemm_tile_rows, mul1 codebook only).
#include "core/gemv_policy.hpp"
#include "core/numeric_partition.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace {

struct Matrix {
    int input, output, bits, codebook;
};

// Frozen: the previous Weights templates (signed form).
bool old_medium(const Matrix& w, int rows) {
    if (rows < 32 || w.codebook != 2) return false;
    if (rows >= 256) return (w.bits == 6 && w.input == 5120 && w.output == 1280) || (w.bits == 5 && w.input == 4096 && w.output == 1024);
    if (w.bits == 6) return w.input == 5120 && w.output == 1280;
    if (w.bits != 5) return false;
    return (w.input == 1280 && w.output == 32768) || (w.input == 8192 && w.output == 5120) ||
           (w.input == 5120 && w.output == 2304) || (w.input == 2304 && w.output == 5120) ||
           (w.input == 4096 && w.output == 1024);
}
int old_bucket(const Matrix& w, int rows) {
    return old_medium(w, rows) || (rows >= 256 && w.output >= 2304) ||
                   (rows >= 64 && w.output >= 32768 && w.bits == 5 && w.codebook == 2)
               ? 64
               : 16;
}
int old_vector(const Matrix& w, int rows) {
    if (w.bits == 5 && w.codebook == 2) {
        if (w.input == 4096 && w.output == 1024 && (rows == 1 || rows == 4 || rows == 8)) return 16;
        if (w.input == 5120 && w.output == 2304 && rows == 8) return 8;
    }
    return w.output <= 1024 ? 32 : (w.output <= 4096 && w.input >= 4096 ? 16 : 4);
}
// Frozen: previous src/core/linear_plan.hpp linear_contract split rule.
int old_gemm(const Matrix& w, int rows) {
    int requested = 1;
    if (rows > 8) {
        const std::size_t bucket_m = static_cast<std::size_t>(old_bucket(w, rows));
        const std::size_t buckets = ((static_cast<std::size_t>(rows) + bucket_m - 1) / bucket_m) *
                                    ((static_cast<std::size_t>(w.output) + 63) / 64);
        if (old_medium(w, rows)) {
            if (w.output >= 32768) requested = rows < 64 ? 4 : 1;
            else {
                const std::size_t target = rows > 64 && rows % 64 ? 512 : 256;
                while (buckets * static_cast<std::size_t>(requested) < target && requested < 16) requested *= 2;
            }
        } else {
            while (buckets * static_cast<std::size_t>(requested) < 160 && requested < 32) requested *= 2;
        }
    } else {
        requested = old_vector(w, rows);
    }
    return std::min(requested, w.input / 16);
}

// Frozen: the MiniMax private route policy (unsigned, mul1 only).
constexpr bool mm_medium(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    if (rows < 32) return false;
    if (rows >= 256) return (bits == 6 && input == 5120 && output == 1280) || (bits == 5 && input == 4096 && output == 1024);
    if (bits == 6) return input == 5120 && output == 1280;
    if (bits != 5) return false;
    return (input == 1280 && output == 32768) || (input == 8192 && output == 5120) || (input == 5120 && output == 2304) ||
           (input == 2304 && output == 5120) || (input == 4096 && output == 1024);
}
constexpr unsigned mm_vector(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    if (bits == 5) {
        if (input == 4096 && output == 1024 && (rows == 1 || rows == 4 || rows == 8)) return 16;
        if (input == 5120 && output == 2304 && rows == 8) return 8;
    }
    return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
}
constexpr unsigned mm_tile(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    return mm_medium(input, output, bits, rows) || (rows >= 256 && output >= 2304) ||
                   (rows >= 64 && output >= 32768 && bits == 5) ? 64U : 16U;
}
constexpr unsigned mm_gemm(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    std::size_t requested = 1;
    if (rows > 8) {
        const std::size_t tile_m = mm_tile(input, output, bits, rows);
        const std::size_t groups = ((rows + tile_m - 1) / tile_m) * ((output + 63U) / 64U);
        if (mm_medium(input, output, bits, rows)) {
            if (output >= 32768) {
                requested = rows < 64 ? 4U : 1U;
            } else {
                const std::size_t target = rows > 64 && rows % 64 ? 512U : 256U;
                while (groups * requested < target && requested < 16) requested *= 2;
            }
        } else {
            while (groups * requested < 160 && requested < 32) requested *= 2;
        }
    } else {
        requested = mm_vector(input, output, bits, rows);
    }
    return static_cast<unsigned>(std::min<std::size_t>(requested, input / 16));
}

namespace part = strix::core::partition;

// The policy is usable in constant expressions (static_asserts of device planners).
static_assert(part::gemm_splits(6144, 1536, 4, 2, 8) == 16);
static_assert(part::gemm_splits(6144, 1536, 4, 2, 9) == 8);
static_assert(part::bucket_rows(4096, 1024, 5, 2, 64) == 64);

int failures = 0;
void require(bool ok, const char* what, const Matrix& w, int rows) {
    if (ok) return;
    if (failures++ < 10)
        std::printf("MISMATCH %s in=%d out=%d bits=%d cb=%d rows=%d\n", what, w.input, w.output, w.bits, w.codebook, rows);
}

unsigned u(int v) { return static_cast<unsigned>(v); }

}  // namespace

int main() {
    const int inputs[] = {16, 128, 1280, 1536, 2304, 3072, 4096, 5120, 6144, 8192, 12288};
    const int outputs[] = {128, 1024, 1280, 1536, 2304, 3072, 4096, 5120, 6144, 32768, 100096};
    std::size_t cases = 0;
    for (int in : inputs)
        for (int out : outputs)
            for (int bits = 1; bits <= 8; ++bits)
                for (int cb = 0; cb <= 2; ++cb)
                    for (int rows = -1; rows <= 4200; ++rows) {
                        const Matrix w{in, out, bits, cb};
                        ++cases;
                        require(strix::core::numeric_partition_medium(w, rows) == old_medium(w, rows), "medium", w, rows);
                        require(strix::core::numeric_partition_bucket_rows(w, rows) == old_bucket(w, rows), "bucket", w, rows);
                        if (rows < 1) continue;
                        require(strix::core::measured_vector_splits(in, out, bits, cb, rows) == old_vector(w, rows), "vector", w, rows);
                        require(static_cast<int>(part::gemm_splits(u(in), u(out), u(bits), u(cb), u(rows))) == old_gemm(w, rows),
                                "gemm", w, rows);
                        if (cb != 2) continue;
                        require(part::gemm_splits(u(in), u(out), u(bits), 2U, u(rows)) == mm_gemm(u(in), u(out), u(bits), u(rows)),
                                "minimax gemm", w, rows);
                        require(part::bucket_rows(u(in), u(out), u(bits), 2U, u(rows)) == mm_tile(u(in), u(out), u(bits), u(rows)),
                                "minimax tile", w, rows);
                        require(part::vector_splits(u(in), u(out), u(bits), 2U, u(rows)) == mm_vector(u(in), u(out), u(bits), u(rows)),
                                "minimax vector", w, rows);
                    }
    if (failures != 0) {
        std::printf("numeric-partition: %d mismatches in %zu cases\n", failures, cases);
        return 1;
    }
    std::printf("numeric-partition: %zu cases identical (Weights templates, rows-dependent split, MiniMax route policy)\n", cases);
    return 0;
}
