#pragma once

// QTIP-style trellis weights for the deepseek41 dense decode sites (identity ds41-trellis-k5-v1, docs/models/deepseek-v4.1/trellis-k5-v1.md).
//
// A pack directory holds dense.L<l>.safetensors (layers.<l>.attn.{wq_a,wq_b,wkv,wo_b}, layers.<l>.attn.wo_a.slice.<g>,
// layers.<l>.ffn.shared_experts.{w1,w2,w3}) and dense.head.safetensors (head); each linear is four tensors:
// <name>.trellis I16 [I/16][N/16][16 K] (16 x 16 tiles, K bits per weight, each tile an independent bitstream),
// <name>.suh F16 [I], <name>.svh F16 [N], <name>.mul1 I32 (the codebook marker). W = diag(suh) H_I Q H_O diag(svh) with H
// the normalized 128-block Hadamard; the trellis library applies every transform itself and takes F32 X / Y in the
// original basis.
//
// Host side (this header, ds41_trellis.cpp): the pack index, the GGUF-name -> pack-name map and the host blob of a tensor
// or of a 128-aligned slice of it (output rows: H_O is block diagonal, so rows [a, a + n) are the tiles' columns
// [a/16, (a + n)/16) and svh[a, a + n); input columns: tiles' rows and suh). Device side (hip_ds41_trellis.hip): a
// context, prepared sites and enqueue_trellis_linear. Without the library (CMake STRIX_DS41_TRELLIS=OFF) the device
// side throws at context creation; the host side always builds.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strix::ds41 {

class SafetensorsCheckpoint;
// Ordered, expanded class=directory entries; preserves every directory opened by the loader.
[[nodiscard]] std::vector<std::pair<std::string, std::string>> parse_trellis_pack_spec(const std::string& spec);

// The name of the single-pack K5 identity (docs/models/deepseek-v4.1/trellis-k5-v1.md). Engines announce the identity of the packs they
// prepared (trellis_step_identity / trellis_config_identity below), never this constant alone.
inline constexpr const char* trellis_identity = "ds41-trellis-k5-v1";

// Site classes (same names as the native FP8 classes, plus head): q_a, q_b, kv, o_a, o_b, shexp, head.
// GGUF name -> class / pack name (blk.L.attn_q_b.weight -> layers.L.attn.wq_b; output.weight -> head; o_a has one pack
// tensor per group: layers.L.attn.wo_a.slice.<g>, see trellis_pack_name's group argument).
[[nodiscard]] std::optional<std::string> trellis_site_class(const std::string& model_name);
[[nodiscard]] std::optional<std::string> trellis_pack_name(const std::string& model_name, int group = -1);
// Comma-separated classes ("all" = every class) to the list; throws on an unknown class.
[[nodiscard]] std::vector<std::string> parse_trellis_site_classes(const std::string& list);
[[nodiscard]] bool trellis_class_enabled(const std::vector<std::string>& classes, const std::string& model_name);

// A 128-aligned part of a linear: none = the whole tensor; rows = output channels [first, first + count);
// columns = input channels [first, first + count) (a K part: its output is the partial product, summed by the caller).
struct TrellisSlice {
    enum class Kind : std::uint8_t { none, rows, columns };
    Kind kind = Kind::none;
    std::uint32_t first = 0, count = 0;
    [[nodiscard]] std::string describe() const;
};

// One linear of the pack (whole tensor): geometry and where its four tensors are.
struct TrellisTensorInfo {
    std::string name;                  // pack name, e.g. layers.0.ffn.shared_experts.w1
    std::uint32_t rows = 0;            // N, output channels
    std::uint32_t columns = 0;         // I, input channels
    std::uint32_t bits = 0;            // K
};

// Host transport of one (sliced) linear: blob = trellis | suh | svh, little endian, ready for the library's canonical
// trellis transport (STRIX_TRANSPORT_EXL3_CANONICAL in the library ABI; codebook mul1, marker 0x83DCD12D, no bias).
struct TrellisHostBlob {
    std::string name;
    std::uint32_t rows = 0, columns = 0, bits = 0, marker = 0;
    std::uint64_t trellis_offset = 0, trellis_bytes = 0, suh_offset = 0, suh_bytes = 0, svh_offset = 0, svh_bytes = 0;
    std::vector<std::uint8_t> blob;
};

// Mixed packs (one K per site class): the classes of trellis_mix_class (q_a, kv, q_b, o_a, o_b, shexp_w13 = shared gate
// and up, shexp_w2 = shared down, head) each read their own pack directory. A spec is a pack directory (every class), a
// list "class=dir,class=dir,...[,default=dir]" (shexp = shexp_w13 + shexp_w2), or a manifest file with one class=dir
// per line (# comments). trellis_env_spec(): STRIX_DS41_TRELLIS_MIX (list or manifest; STRIX_DS41_TRELLIS, if also set,
// is its default) else STRIX_DS41_TRELLIS; empty when neither is set.
[[nodiscard]] std::string trellis_mix_class(const std::string& pack_name);
[[nodiscard]] std::string trellis_env_spec();

class TrellisPack {
public:
    explicit TrellisPack(const std::filesystem::path& spec);
    ~TrellisPack();
    TrellisPack(const TrellisPack&) = delete;
    TrellisPack& operator=(const TrellisPack&) = delete;
    [[nodiscard]] const std::filesystem::path& dir() const noexcept { return dir_; }
    // "dir" for one pack, else "class=dir,..." (sorted, default last): the storage line of the engines.
    [[nodiscard]] const std::string& describe() const noexcept { return description_; }
    [[nodiscard]] bool mixed() const noexcept { return !by_class_.empty(); }
    // The pack directory a mix class (trellis_mix_class) reads: the single pack, the class's entry or the default;
    // empty if the mix has none.
    [[nodiscard]] std::filesystem::path class_dir(const std::string& mix_class) const;
    // nullopt if the pack has no such linear (pack name).
    [[nodiscard]] std::optional<TrellisTensorInfo> info(const std::string& pack_name);
    // The transport of the linear or of a slice; throws on a missing tensor, a bad geometry or an unaligned slice.
    [[nodiscard]] TrellisHostBlob load(const std::string& pack_name, const TrellisSlice& slice = {});

private:
    [[nodiscard]] SafetensorsCheckpoint& checkpoint(const std::string& pack_name);
    std::filesystem::path dir_;
    std::string description_;
    std::map<std::string, std::filesystem::path> dirs_;  // mix class -> directory (empty for one pack)
    std::filesystem::path default_dir_;                  // the single pack, or the mix's default (may be empty)
    std::shared_ptr<SafetensorsCheckpoint> checkpoint_;  // the single pack, or the mix's default (may be null)
    std::map<std::string, std::shared_ptr<SafetensorsCheckpoint>> by_class_;
};

// Device bytes of a prepared linear or slice (trellis + suh + svh as uploaded; TrellisSite::device_bytes): rows x
// columns x K / 8 + 2 (columns + rows), with the slice's rows or columns.
[[nodiscard]] std::uint64_t trellis_site_bytes(const TrellisTensorInfo& info, const TrellisSlice& slice = {});

// ---- identities announced under single packs and mixes (docs/models/deepseek-v4.1/trellis-k5-v1.md, "Announced identity") ----
// A step's identity: ds41-trellis-k<K>-v1:<site> with the K of the pack that step reads (k5 = ds41-trellis-k5-v1).
[[nodiscard]] std::string trellis_step_identity(std::uint32_t bits, const std::string& site);
// The shared expert's step: gate / up (shexp_w13) and down (shexp_w2) may come from packs of different K in a mix:
// ds41-trellis-k<K>-v1:shexp[-tp] when equal, else ds41-trellis-mix-v1{shexp_w13=K<a>,shexp_w2=K<b>}:shexp[-tp].
[[nodiscard]] std::string trellis_shexp_step_identity(std::uint32_t w13_bits, std::uint32_t w2_bits, bool shexp_tp);
// The identity of a configuration from the K of every enabled mix class (q_a, kv, q_b, o_a, o_b, shexp_w13, shexp_w2,
// head) and the enabled site classes (STRIX_DS41_TRELLIS_SITES), `shexp_tp` = STRIX_DS41_SHEXP_TP:
//   one K everywhere      -> ds41-trellis-k<K>-v1:<sites>
//   different K per class -> ds41-trellis-mix-v1{class=K<K>,...}:<sites>, plus " (= ds41-trellis-mixA-v1)" or
//                            " (= ds41-trellis-mixB-v1)" when the per-class K equal those declared mixes
// then ":shexp-tp" when the shared expert is split. Empty when no class is enabled. HEADOUT and OA_GROUP are adapter
// forms with the same bits and are not part of it.
[[nodiscard]] std::string trellis_config_identity(const std::map<std::string, std::uint32_t>& bits_by_mix_class,
                                                  const std::vector<std::string>& site_classes, bool shexp_tp);
// The K per enabled mix class read from the pack (layer `layer`'s tensors and the head); throws if an enabled class has
// no tensor in the pack.
[[nodiscard]] std::map<std::string, std::uint32_t> trellis_class_bits(TrellisPack& pack, const std::vector<std::string>& site_classes,
                                                                      int layer = 0);
// The weight configuration of the trellis environment (STRIX_DS41_TRELLIS / _MIX, _SITES, _SHEXP_TP) as one line for
// namespaces (e.g. the prefix cache's model id): "" when no trellis site is active, else
// "trellis <identity>; packs <class>=<dir>#<fingerprint>,..." with each directory's fingerprint the FNV-1a 64 of its
// SHA256 manifest <dir>.sha256 when present, else of its file names and sizes. Reads the pack headers only.
[[nodiscard]] std::string trellis_weights_namespace();

// Prepared plans created on first use, process-wide (every TrellisContext and TrellisGroupRunner): the count and the
// wall seconds of the creations (each one synchronises). A server warms them before READY; later creations are misses.
struct TrellisPlanStats {
    std::uint64_t created = 0;
    double seconds = 0.0;
};
[[nodiscard]] TrellisPlanStats trellis_plan_stats() noexcept;
void note_trellis_plan_created(double seconds) noexcept;

// ---- device side (hip_ds41_trellis.hip) -------------------------------------------------------------------------

class TrellisContext;

// A prepared linear on the device (whole tensor or slice). Owned by its context; stable address.
struct TrellisSite {
    void* weights = nullptr;           // strix_weights
    std::uint32_t rows = 0, columns = 0, bits = 0;
    std::uint64_t device_bytes = 0;    // trellis + suh + svh as uploaded
    std::string key;                   // pack name + slice description
    TrellisContext* context = nullptr;
};

// Library row counts of a site. The library's GEMV picks its kernel and K-split count from the row count, and for some
// shapes a row's bits depend on it (strix-hip-ds41-trellis-tests: "R classes"). The identity therefore fixes, per
// (I, N, K), the set of row counts a site is submitted with, all of one R class: bit k-1 set = k rows allowed (1..8).
// R rows go in chunks of the largest allowed count; a shorter tail is padded up to the smallest allowed count >= its
// rows with copies of its last row (the extra outputs are discarded; the weights are still read once per chunk).
[[nodiscard]] std::uint32_t trellis_submit_sizes(std::uint32_t columns, std::uint32_t rows, std::uint32_t bits) noexcept;
[[nodiscard]] std::uint32_t trellis_submit_sizes(const TrellisSite& site) noexcept;
// Largest allowed row count (the chunk) and the count a chunk of n <= chunk rows is submitted with.
[[nodiscard]] std::uint32_t trellis_chunk_rows(const TrellisSite& site) noexcept;
[[nodiscard]] std::uint32_t trellis_padded_rows(const TrellisSite& site, std::uint32_t n) noexcept;

class TrellisContext {
public:
    // Creates the library context on the current HIP device. Throws without the library.
    TrellisContext();
    ~TrellisContext();
    TrellisContext(const TrellisContext&) = delete;
    TrellisContext& operator=(const TrellisContext&) = delete;
    // Uploads the linear (or slice) once; a second call with the same pack name and slice returns the same site.
    const TrellisSite& prepare(TrellisPack& pack, const std::string& pack_name, const TrellisSlice& slice = {});
    [[nodiscard]] const TrellisSite* find(const std::string& pack_name, const TrellisSlice& slice = {}) const;
    // Allocates the workspace of `stream` for every prepared site (chunks of up to 8 rows; call after the last prepare
    // and before any graph capture; enqueue on an unreserved stream throws). Generic calls share scratch; split-K
    // counter banks are separate per stream, geometry, bits and submitted row count.
    void reserve(void* stream, std::uint32_t max_rows);
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;  // prepared weights
    // Reserved per-stream memory: x / y staging (8 rows of the widest site) and the library workspace.
    [[nodiscard]] std::uint64_t staging_bytes() const noexcept;
    [[nodiscard]] std::uint64_t library_workspace_bytes() const noexcept;
    // The staging one reserve() allocates for these prepared sites (8 x max columns + 8 x max rows floats; >= 128 each).
    [[nodiscard]] static std::uint64_t staging_bytes_for(std::uint32_t max_columns, std::uint32_t max_rows) noexcept;
    [[nodiscard]] std::size_t sites() const noexcept;
    // The identity of the prepared sites (trellis_config_identity of each prepared site's K per mix class); "" if none.
    [[nodiscard]] std::string announced_identity(const std::vector<std::string>& site_classes, bool shexp_tp) const;
    [[nodiscard]] static bool available() noexcept;             // built with the library
    [[nodiscard]] static const char* library_version() noexcept;
    [[nodiscard]] void* library_context() const noexcept;      // strix_context (tests and direct calls)
    // Internal adapter entry point; buffers must remain allocated until release_plans or context destruction.
    // The first call binds a prepared plan outside capture; subsequent calls only submit it.
    void submit(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, void* workspace,
                std::uint64_t workspace_bytes, void* stream);
    // The same with F32 row views: x rows x_stride apart, y rows y_stride apart (elements; 0 = contiguous rows).
    void submit(const TrellisSite& site, const float* x, std::size_t x_stride, float* y, std::size_t y_stride, std::uint32_t rows,
                void* workspace, std::uint64_t workspace_bytes, void* stream);
    void release_plans(void* workspace) noexcept;
    struct Impl;
    [[nodiscard]] Impl& impl() noexcept { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

// y[r][0, N) = Linear(x[r][0, I)) for r < rows (accumulate: y = y + Linear(x), one FP32 add per value).
// x rows have stride x_stride (>= I), y rows y_stride (>= N), F32 device memory, original basis. Rows are submitted in
// chunks / padded tails (trellis_submit_sizes); non-contiguous rows, padding and accumulate go through the stream's
// workspace.
void enqueue_trellis_linear(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, std::size_t x_stride,
                            std::size_t y_stride, bool accumulate, void* stream);

// STRIX_DS41_TRELLIS_HEADOUT=1 (default off; read once): the trellis LM head's adapter without staging copies.
// Also lets enqueue_trellis_linear submit a single-row chunk in place whatever its strides (one row is contiguous).
[[nodiscard]] bool trellis_headout_enabled() noexcept;
// Opt-in completion of the dense pack routes: q_a / kv and K4 shared gate/up row halves.
// Unset keeps the original engine routes, including in a v12 build.
[[nodiscard]] bool trellis_dense_full_enabled() noexcept;
// Only the explicit split-K build changes existing K4 shared gate/up call shapes.
[[nodiscard]] bool trellis_dense_row_slices_enabled() noexcept;
// Independently opt in to the q_a+kv pair. v12 submits the same two legacy linears;
// splitk04 uses the library pair plan. Inputs and outputs are contiguous F32 rows.
[[nodiscard]] bool trellis_qakv_pair_enabled() noexcept;
// STRIX_DS41_TRELLIS_ROW_VIEWS=1 (default off; read once): in a dense-r1 build, a TP-shape site's strided rows go to its
// plan as row views (the library takes row stride >= width) instead of through the workspace and hipMemcpy2DAsync staging
// copies; same kernel, schedule and products (the stride only moves addresses). Chunks with padding rows still stage;
// single-row chunks (R1) keep their existing plans, so R1 runs exactly the code path it runs without the flag.
[[nodiscard]] bool trellis_row_views_enabled() noexcept;
// The site's device bytes (the library's single allocation: trellis, suh, svh, ...), for read-ahead into the MALL;
// false when the linked library has no strix_weights_device_span (then nothing is read ahead for trellis sites).
[[nodiscard]] bool trellis_site_span(const TrellisSite& site, const void** data, std::uint64_t* bytes) noexcept;
void enqueue_trellis_qakv(const TrellisSite& qa, const TrellisSite& kv, const float* x, float* q, float* k,
                          std::uint32_t rows, void* stream);
// y[r][0, N) = Linear(x[r][0, I)) for r < rows, no staging where the call's layout allows it: x and y each hold at least
// `capacity` rows at their strides, rows in [rows, capacity) are scratch (read as padding inputs, overwritten by padding
// outputs). A chunk of n >= pad_from rows (pad_from 0 = never) is submitted as a full chunk (trellis_chunk_rows) when it
// fits the capacity; any chunk goes in place when its rows are contiguous (stride == I / N, or one row), else through the
// workspace as enqueue_trellis_linear. Exact only for sites whose rows are independent within a call and whose chunk
// size is in the same R class as the padded counts (the head: one class R1..R8).
void enqueue_trellis_linear_rows(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, std::size_t x_stride,
                                 std::size_t y_stride, std::uint32_t capacity, std::uint32_t pad_from, void* stream);

// act[r][i] = swiglu_clamped(gate[r][i], up[r][i], limit) in F32 (ds41_moe.hpp), rows x width, strides width.
void enqueue_swiglu_f32(const float* gate, const float* up, std::uint32_t rows, std::uint32_t width, float limit, float* act,
                        void* stream);

// STRIX_DS41_SHARED_GROUPED (read once; dense-r1 builds with the K4 shared row-halves route only): =1 on, =0 off,
// unset = on wherever the startup admission finds it admissible (else the old path, one log line):
// the shared expert's four gate/up row halves + enqueue_swiglu_f32 run as two library SwiGLU pair launches (one per
// 1152-row half: gate and up products of that half and the clamped declared SwiGLU in the completing block), writing
// act[r][h * 1152 + j] (row stride 2 * 1152). Same products (per-shape split-K order) and the same activation bits.
[[nodiscard]] bool trellis_shared_grouped_enabled() noexcept;
// Startup admission. Unset: enables the implicit default when admissible, otherwise keeps the old path for the whole
// process (one log line). =1: throws when the build cannot run it (no dense-r1
// or no declared SwiGLU pair in the linked library), with the shexp-tp route, without the K4 shared row-halves route
// (the flag would be ignored), or with more than 8 rows per step (STRIX_DS41_BATCH_ROWS16: not qualified yet).
void trellis_shared_grouped_admit(bool shexp_tp, bool halves_route, std::uint32_t row_limit);
// halves = {gate rows [0,1152), gate rows [1152,2304), up rows [0,1152), up rows [1152,2304)} (5120 -> 1152 each);
// x [rows][5120] contiguous, act [rows][2304] contiguous; rows chunked by 8.
void enqueue_trellis_shared_swiglu(const TrellisSite* const halves[4], const float* x, float* act, std::uint32_t rows,
                                   float limit, void* stream);

// STRIX_DS41_SHARED_DOWN_FUSED (read once): =1 on (throwing diagnostics), =0 off, unset = on wherever the startup
// admission finds it admissible (else the old path for the whole process, one log line). The shared expert's down
// (shexp_w2, mul1 K4 [N=5120, I=2304]) runs as one library launch (STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED: the input
// transform, the K16 product and the output transform folded with the three-launch route's bits) instead of three.
// The output stays the F32 [rows][5120] shared buffer. On only after trellis_shared_down_fused_admit admitted it.
[[nodiscard]] bool trellis_shared_down_fused_enabled() noexcept;
// Startup admission, after every shared down site is prepared and before any capture; the selection it makes is frozen
// once the route ran. `downs` = the trellis shared down site of every layer, `layers` = the shared expert steps of the
// plan: every one must have a K4 [5120, 2304] site of one context (mul1 is the pack format), else the old path. Then a
// startup probe of the linked library: for each site and 1..min(row_limit, 8) rows, the staged workspace query must
// succeed (else throws) and the declared one either succeeds (admitted) or returns STRIX_ERROR_INVALID_ARGUMENT
// (confirmed rejection: old path); any other status throws. Refused with STRIX_DS41_SHEXP_TP=1 (its down is a TP
// slice) and in builds whose library header has no declared route. =1 throws where the implicit default would refuse.
void trellis_shared_down_fused_admit(const std::vector<const TrellisSite*>& downs, std::size_t layers, bool shexp_tp,
                                     std::uint32_t row_limit);
// out[r][0, 5120) = down(act[r][0, 2304)) for r < rows (row strides act_stride / out_stride). The declared route when
// enabled (contiguous rows, chunks of 8, one prepared plan per site, buffers, stream and row count, created outside
// capture and counted in note_trellis_plan_created; the first one logs "shared down fused plan created"), else exactly
// enqueue_trellis_linear(site, act, out, rows, act_stride, out_stride, false, stream).
void enqueue_trellis_shared_down(const TrellisSite& site, const float* act, std::size_t act_stride, float* out,
                                 std::size_t out_stride, std::uint32_t rows, void* stream);

}  // namespace strix::ds41
