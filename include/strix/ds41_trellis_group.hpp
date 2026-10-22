#pragma once

// Grouped trellis linears with column-strided input and output (identity ds41-trellis-k5-v1, o_a section):
//   y[r][g N + n] = Linear_g(x[r][g I + i])   for r < rows, g < groups (every group: I inputs, N outputs, same K),
// the wo_a groups of one layer (I 4096, N 1024; 4 per TP rank, 8 in one process). Default off: STRIX_DS41_TRELLIS_OA_GROUP
// (unset / 0 = the per-group enqueue_trellis_linear calls of the attention hook; 1 = this runner; 2 = this runner with the
// groups' library calls on concurrent streams). Every row gets the bits of the per-group path: the same library calls
// (same row count m per chunk, trellis_padded_rows, padding rows = copies of the chunk's last row), only the staging
// changes:
//   - a chunk of one row needs no staging at all (a single row is contiguous at any stride): the library reads x and
//     writes y in place;
//   - a chunk of n > 1 rows stages every group's input (and its padding rows) with ONE kernel into [G][m][I], runs the G
//     library calls into [G][m][N] and scatters the n real rows back with ONE kernel (instead of 1 + (m - n) input
//     copies and 1 output copy per group).
// When the library offers a grouped strided GEMV (strix_trellis.h defines STRIX_TRELLIS_HAS_GROUPED_GEMV), the runner
// uses it after a self-check at construction (bitwise == this staged path for R = 1..8 on the given sites); otherwise,
// or with STRIX_DS41_TRELLIS_OA_GROUP_LIB=0, the staged path.

#include "strix/ds41_trellis.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix::ds41 {

// STRIX_DS41_TRELLIS_OA_GROUP: 0 (unset) / 1 / 2, see above. Throws on another value.
[[nodiscard]] int trellis_oa_group_mode();

class TrellisGroupRunner {
public:
    // sites: one layer's groups (all the same I, N, K and context); the workspace is sized for groups x 8 rows and is
    // shared by every later call on `stream` with sites of the same shape (any layer). Self-checks the grouped library
    // entry on these sites when it exists.
    TrellisGroupRunner(const std::vector<const TrellisSite*>& sites, int mode, void* stream);
    ~TrellisGroupRunner();
    TrellisGroupRunner(const TrellisGroupRunner&) = delete;
    TrellisGroupRunner& operator=(const TrellisGroupRunner&) = delete;
    // y rows have stride y_stride >= G N, x rows x_stride >= G I; F32 device memory; any rows (chunks of the identity's
    // submission sizes).
    void enqueue(const std::vector<const TrellisSite*>& sites, const float* x, std::size_t x_stride, float* y,
                 std::size_t y_stride, std::uint32_t rows, void* stream) const;
    [[nodiscard]] std::string describe() const;  // mode and path (staged / grouped library), for the startup line
    // Device memory it allocated: x / y staging (groups x 8 rows) and the library workspaces (one per lane, plus the
    // grouped entry's when it passed the self-check).
    [[nodiscard]] std::uint64_t staging_bytes() const noexcept;
    [[nodiscard]] std::uint64_t library_workspace_bytes() const noexcept;
    // The staging the constructor allocates for G groups of I -> N.
    [[nodiscard]] static std::uint64_t staging_bytes_for(std::uint32_t groups, std::uint32_t columns, std::uint32_t rows) noexcept;
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::ds41
