// Build locally. Run only on the native test node.
#include "cuda/core/head_emit.cuh"
#include "step_inputs_cases.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

// Incumbent reference bodies copied from src/qwen35/cuda/engine.cu:284-382.
namespace incumbent {
namespace cc { inline constexpr std::uint32_t kMaxRows = 32; }

struct InputPack {
    std::uint32_t pos0;
    std::uint32_t ids[cc::kMaxRows];
};
// Token-tree window descriptor (strix/spec/tree_window.hpp), absolute: tree[0] = leaf mask, tree[1 + r] = position of row r,
// tree[1 + 32 + r] = K/V index of row r. Written by the same input launch (tree == nullptr: linear window, nothing written).
struct TreePack {
    std::uint32_t leaf;
    std::uint32_t pos[cc::kMaxRows];
    std::uint32_t idx[cc::kMaxRows];
};
constexpr std::uint32_t kTreeWords = 1 + 2 * cc::kMaxRows;
__global__ void set_inputs_kernel(InputPack in, std::uint32_t* pos, std::uint32_t* ids) {
    if (threadIdx.x == 0) *pos = in.pos0;
    if (threadIdx.x < cc::kMaxRows) ids[threadIdx.x] = in.ids[threadIdx.x];
}
// Token-tree window: the same inputs + the tree descriptor (leaf mask, row positions, K/V indices) in one launch.
__global__ void set_inputs_tree_kernel(InputPack in, std::uint32_t* pos, std::uint32_t* ids, TreePack tp, std::uint32_t* tree) {
    if (threadIdx.x == 0) *pos = in.pos0;
    if (threadIdx.x < cc::kMaxRows) ids[threadIdx.x] = in.ids[threadIdx.x];
    if (threadIdx.x == 0) tree[0] = tp.leaf;
    if (threadIdx.x < cc::kMaxRows) {
        tree[1 + threadIdx.x] = tp.pos[threadIdx.x];
        tree[1 + cc::kMaxRows + threadIdx.x] = tp.idx[threadIdx.x];
    }
}
// Draft head output. Subset head (group_of set): compact row -> token id (slot group_of[row / group], row % group), in place
// on the argmax output. Device-fed MTP chain (feed_out set): the last row's token id is also stored in the chain's feed slot.
// top2 / feed_out2 (runner-up, token-tree siblings): the same map on the second id (0xFFFFFFFF = none stays none); the last
// row's logit gap (argmax2_rows) goes to the feed slot's gap word as its bits.
// TWO (runner-ups on) is a trait: the TWO = false instantiation is the plain map.
template <bool TWO = false>
__global__ void draft_map_kernel(std::uint32_t* top, const std::uint32_t* group_of, std::uint32_t rows, std::uint32_t group,
                                 std::uint32_t* feed_out, std::uint32_t* top2 = nullptr, std::uint32_t* feed_out2 = nullptr,
                                 const float* gap = nullptr, std::uint32_t* feed_gap = nullptr) {
    if constexpr (TWO) {
        if (gap && feed_gap && threadIdx.x + 1 == rows) *feed_gap = __float_as_uint(gap[threadIdx.x]);
    }
    if (threadIdx.x < rows) {
        std::uint32_t t = top[threadIdx.x];
        if (group_of) {
            t = group_of[t / group] * group + t % group;
            top[threadIdx.x] = t;
        }
        if (feed_out && threadIdx.x + 1 == rows) *feed_out = t;
        if (TWO && top2) {
            std::uint32_t u = top2[threadIdx.x];
            if (group_of && u != 0xFFFFFFFFu) {
                u = group_of[u / group] * group + u % group;
                top2[threadIdx.x] = u;
            }
            if (feed_out2 && threadIdx.x + 1 == rows) *feed_out2 = u;
        }
    }
}
// Device-fed MTP chain step: the input id is the previous step's draft, read from the feed on the device (no host round trip).
__global__ void fed_inputs_kernel(const std::uint32_t* feed, std::uint32_t slot, std::uint32_t pos0, std::uint32_t* pos,
                                  std::uint32_t* ids) {
    if (threadIdx.x == 0) {
        *pos = pos0;
        ids[0] = feed[slot];
    }
}
// Verify pre-launch: the verify rows' ids, host values or drafted ids still in the chain's feed slots (gathered on the
// device). Rows past the window get id 0, as set_inputs_kernel leaves them, so d_ids is the same array either way.
constexpr std::uint32_t kFeedHostRow = 0xFFFFFFFFU;  // = strix::spec::DraftFeed::kHost
struct FedPack {
    std::uint32_t pos0;
    std::uint32_t ids[cc::kMaxRows];
    std::uint32_t slot[cc::kMaxRows];
};
__global__ void fed_rows_kernel(FedPack in, const std::uint32_t* feed, std::uint32_t* pos, std::uint32_t* ids) {
    if (threadIdx.x == 0) *pos = in.pos0;
    if (threadIdx.x < cc::kMaxRows) {
        const std::uint32_t sl = in.slot[threadIdx.x];
        ids[threadIdx.x] = sl == kFeedHostRow ? in.ids[threadIdx.x] : feed[sl];
    }
}

// Multi-session batch tables (one small by-value pack, scattered to device arrays by one kernel before the graph).
struct SlotPack {
    std::uint32_t ids[cc::kMaxRows], pos[cc::kMaxRows], slot[cc::kMaxRows];
    std::uint32_t g_row0[cc::kMaxRows], g_rows[cc::kMaxRows], g_slot[cc::kMaxRows];
};
struct SlotDev {
    std::uint32_t *ids, *pos, *slot, *g_row0, *g_rows, *g_slot;
};
__global__ void set_slot_inputs_kernel(SlotPack in, SlotDev d) {
    const std::uint32_t i = threadIdx.x;
    if (i < cc::kMaxRows) {
        d.ids[i] = in.ids[i];
        d.pos[i] = in.pos[i];
        d.slot[i] = in.slot[i];
        d.g_row0[i] = in.g_row0[i];
        d.g_rows[i] = in.g_rows[i];
        d.g_slot[i] = in.g_slot[i];
    }
}
}  // namespace incumbent

namespace {
using namespace e2i;
namespace head = strix::cuda::core::head;
static_assert(incumbent::kTreeWords == scatter::kTreeWords);

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T>
struct Device {
    explicit Device(const T& input) {
        check(cudaMalloc(&ptr, sizeof(T)));
        const auto error = cudaMemcpy(ptr, &input, sizeof(T), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            (void)cudaFree(ptr);
            check(error);
        }
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)cudaFree(ptr); }

    T download() const {
        T out;
        check(cudaMemcpy(&out, ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }

    T* ptr = nullptr;
};

void launch_incumbent(Mode mode, const ScatterCase& c, const ScatterCase* device, ScatterOutput* out) {
    const auto pos = out->anchor.words + 1;
    const auto ids = out->ids.words + 1;
    switch (mode) {
        case Mode::linear: {
            incumbent::InputPack in{c.positions[0], {}};
            std::copy_n(c.ids, kRows, in.ids);
            incumbent::set_inputs_kernel<<<1, kRows>>>(in, pos, ids);
            break;
        }
        case Mode::tree: {
            incumbent::InputPack in{c.positions[0], {}};
            std::copy_n(c.ids, kRows, in.ids);
            incumbent::TreePack tree{c.tree.leaf, {}, {}};
            std::copy_n(c.positions, kRows, tree.pos);
            std::copy_n(c.tree.self_kv, kRows, tree.idx);
            incumbent::set_inputs_tree_kernel<<<1, kRows>>>(in, pos, ids, tree, out->tree.words + 1);
            break;
        }
        case Mode::fed_row:
            incumbent::fed_inputs_kernel<<<1, kRows>>>(device->feeds, c.indices[0], c.positions[0], pos, ids);
            break;
        case Mode::fed_rows: {
            incumbent::FedPack in{c.positions[0], {}, {}};
            std::copy_n(c.ids, kRows, in.ids);
            std::copy_n(c.indices, kRows, in.slot);
            incumbent::fed_rows_kernel<<<1, kRows>>>(in, device->feeds, pos, ids);
            break;
        }
        case Mode::slots: {
            incumbent::SlotPack in{};
            std::copy_n(c.ids, kRows, in.ids);
            std::copy_n(c.positions, kRows, in.pos);
            std::copy_n(c.slots, kRows, in.slot);
            std::copy_n(c.groups.row0, kRows, in.g_row0);
            std::copy_n(c.groups.rows, kRows, in.g_rows);
            std::copy_n(c.groups.slot, kRows, in.g_slot);
            const incumbent::SlotDev view{ids, out->positions.words + 1, out->slots.words + 1,
                out->group_row0.words + 1, out->group_rows.words + 1, out->group_slot.words + 1};
            incumbent::set_slot_inputs_kernel<<<1, kRows>>>(in, view);
            break;
        }
    }
    check(cudaGetLastError());
}

void check_scatter(Mode mode, const char* name) {
    unsigned count = 0;
    for (const auto shape : kScatterCases) {
        if (shape.mode != mode) continue;
        const auto c = make_scatter(shape);
        const Device<ScatterCase> input(c);
        const Device<exec::StepBlock> step(descriptor(shape, c));
        const ScatterOutput seed;
        const Device<ScatterOutput> reference(seed), shared(seed);
        launch_incumbent(mode, c, input.ptr, reference.ptr);
        with_scatter_view(mode, input.ptr, shared.ptr, [&](auto view) { check(scatter::enqueue(step.ptr, view, nullptr)); });
        const auto actual = shared.download();
        if (!equal_bytes(actual, reference.download()) || !equal_bytes(actual, scatter_oracle(mode, c))) {
            std::fprintf(stderr, "%s byte mismatch at %u rows, position %u\n", name, shape.rows, shape.first);
            throw std::runtime_error("step input scratch");
        }
        ++count;
    }
    std::printf("PASS E2i %s (%u cases)\n", name, count);
}

template <bool Two>
void launch_draft(DraftCase c, const DraftInput* input, DraftOutput* out, bool reference) {
    const auto map = c.has(mapped) ? input->map : nullptr;
    const auto top = out->top.words + 1;
    const auto top2 = c.has(second) ? out->top2.words + 1 : nullptr;
    const auto feed = c.has(feed_top) ? out->feed.words + 1 : nullptr;
    const auto feed2 = c.has(feed_second) ? out->feed2.words + 1 : nullptr;
    const auto gap = c.has(gap_input) ? input->gap : nullptr;
    const auto feed_gap = c.has(gap_output) ? out->gap.words + 1 : nullptr;
    if (reference) {
        incumbent::draft_map_kernel<Two><<<1, kRows>>>(top, map, c.rows, c.group, feed, top2, feed2, gap, feed_gap);
        check(cudaGetLastError());
    } else {
        check(head::enqueue_draft_remap<Two>(top, map, c.rows, c.group, feed, nullptr, top2, feed2, gap, feed_gap));
    }
}

void check_draft() {
    unsigned count = 0;
    for (const auto c : kDraftCases) {
        const auto in = make_draft_input(c);
        const auto seed = make_draft_output(c);
        const Device<DraftInput> input(in);
        const Device<DraftOutput> reference(seed), shared(seed);
        if (c.has(two)) {
            launch_draft<true>(c, input.ptr, reference.ptr, true);
            launch_draft<true>(c, input.ptr, shared.ptr, false);
        } else {
            launch_draft<false>(c, input.ptr, reference.ptr, true);
            launch_draft<false>(c, input.ptr, shared.ptr, false);
        }
        const auto actual = shared.download();
        if (!equal_bytes(actual, reference.download()) || !equal_bytes(actual, draft_oracle(c, in, seed))) {
            std::fprintf(stderr, "draft remap byte mismatch at %u rows, group %u, options %u\n", c.rows, c.group, c.options);
            throw std::runtime_error("draft remap output");
        }
        ++count;
    }
    std::printf("PASS E2i draft remap and last-row feeds (%u cases)\n", count);
}

}  // namespace

int main() {
    try {
        check_scatter(Mode::linear, "plain scatter");
        check_scatter(Mode::tree, "tree scatter");
        check_scatter(Mode::fed_row, "single fed-row scatter");
        check_scatter(Mode::fed_rows, "mixed fed-rows scatter");
        check_scatter(Mode::slots, "slot scatter");
        check_draft();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL E2i CUDA: %s\n", e.what());
        return 1;
    }
}
