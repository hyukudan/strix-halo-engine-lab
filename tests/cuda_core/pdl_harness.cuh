// Test-only harness for programmatic dependent launch (PDL) chains whose hazard detection is itself demonstrated.
//
// Part A (synthetic probes, run_probes): one-block kernels stamp %globaltimer, so every repetition states whether the
// successor really started before the predecessor's last write (overlap WITNESSED, not assumed). Cases, each in stream and
// graph form, serial (no PDL attribute) and PDL:
//   R1  the writer releases its successor BEFORE its last write; the reader reads after its wait  -> never stale
//   S1  the reader reads the writer's output BEFORE its wait (seeded bug)                          -> caught when overlapped
//   S2  the reader writes the writer's scratch BEFORE its wait (two overlapping kernels share it)  -> caught when overlapped
//   S3  a middle kernel releases at entry and never waits; the reader waits, then reads            -> stale: completion chain
//       broken (rule C2: every PDL kernel waits in at least one block of every launch)
//   T   the same chain with a waiting middle kernel                                                -> never stale
// A queue-holder kernel (no PDL) runs first, so every launch of a case is enqueued before the first probe starts.
//
// Part B helper (poison_gate): inserted between two kernels of a real chain. It waits (everything before it is complete),
// saves and poisons every live buffer, releases its successor, spins, restores. Whatever starts early behind it runs its
// pre-wait code while only poison is visible: a pre-wait read computes with NaN, a pre-wait write is undone by the restore.
// twin_copy is the seeded kernel of part B (mode 1 reads before its wait, mode 2 writes before its wait).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace pdlh {

#define PDLH_CK(x)                                                                                         \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

__device__ __forceinline__ void trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void spin(unsigned long long ns) {
    const unsigned long long t0 = now();
    while (now() - t0 < ns) {
    }
}

enum : int { kPostRead = 0, kPreRead = 1, kPreWrite = 2 };

// Stamps of one repetition: [0] writer last write done, [1] middle entry, [2] reader entry (before any access).
__global__ void probe_holder(unsigned long long ns) {
    if (threadIdx.x == 0) spin(ns);
}
// Waits for its predecessor, sets its own scratch, releases its successor, then works (spin), reads the scratch back into
// wout (1 unless someone else wrote it meanwhile) and writes buf LAST.
__global__ void probe_writer(float* buf, float* scratch, float* wout, unsigned long long* ts, unsigned long long ns) {
    wait();
    if (threadIdx.x == 0) {
        *static_cast<volatile float*>(scratch) = 1.f;
        __threadfence();
    }
    __syncthreads();
    trigger();
    if (threadIdx.x == 0) {
        spin(ns);
        wout[0] = *static_cast<volatile float*>(scratch);
        *static_cast<volatile float*>(buf) = 1.f;
        __threadfence();
        ts[0] = now();
    }
}
__global__ void probe_middle(int waits, unsigned long long* ts) {
    trigger();
    if (waits) wait();
    if (threadIdx.x == 0) ts[1] = now();
}
__global__ void probe_reader(const float* buf, float* scratch, float* out, int mode, unsigned long long* ts) {
    float v = -1.f;
    if (threadIdx.x == 0) {
        ts[2] = now();
        if (mode == kPreRead) v = *static_cast<const volatile float*>(buf);
        if (mode == kPreWrite) *static_cast<volatile float*>(scratch) = 2.f;
    }
    trigger();
    wait();
    if (threadIdx.x == 0) out[0] = mode == kPreRead ? v : *static_cast<const volatile float*>(buf);
}

// One launch with or without the programmatic-serialization attribute (the attribute goes on the dependent launch).
template <class... K, class... A>
cudaError_t launch(bool pdl, dim3 grid, dim3 block, cudaStream_t s, void (*k)(K...), A... a) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = pdl ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, k, static_cast<K>(a)...);
}

struct ProbeResult {
    int overlap = 0, caught = 0, reps = 0;
};

// Device block of one repetition: buf, scratch, wout, out (floats) then 3 stamps.
struct ProbeMem {
    float* f = nullptr;              // [0] buf [1] scratch [2] wout [3] out
    unsigned long long* t = nullptr; // [0] writer last write [1] middle entry [2] reader entry
    float* hf = nullptr;             // pinned host copies
    unsigned long long* ht = nullptr;
};

enum class Case : int { R1 = 0, S1 = 1, S2 = 2, S3 = 3, T = 4 };
inline const char* case_name(Case c) {
    switch (c) {
        case Case::R1: return "R1 release-before-last-write, read after wait";
        case Case::S1: return "S1 seeded read before wait";
        case Case::S2: return "S2 seeded scratch write before wait";
        case Case::S3: return "S3 middle kernel never waits";
        case Case::T: return "T  middle kernel waits";
    }
    return "?";
}

inline void enqueue_case(Case c, bool pdl, const ProbeMem& m, cudaStream_t s) {
    PDLH_CK(cudaMemsetAsync(m.f, 0, 2 * sizeof(float), s));           // buf = scratch = 0
    PDLH_CK(cudaMemsetAsync(m.f + 2, 0xFF, 2 * sizeof(float), s));    // wout = out = NaN
    PDLH_CK(cudaMemsetAsync(m.t, 0, 3 * sizeof(unsigned long long), s));
    PDLH_CK(launch(false, dim3(1), dim3(32), s, probe_holder, 200000ull));
    PDLH_CK(launch(pdl, dim3(1), dim3(32), s, probe_writer, m.f, m.f + 1, m.f + 2, m.t, 100000ull));
    if (c == Case::S3 || c == Case::T) PDLH_CK(launch(pdl, dim3(1), dim3(32), s, probe_middle, c == Case::T ? 1 : 0, m.t));
    const int mode = c == Case::S1 ? kPreRead : (c == Case::S2 ? kPreWrite : kPostRead);
    PDLH_CK(launch(pdl, dim3(1), dim3(32), s, probe_reader, static_cast<const float*>(m.f), m.f + 1, m.f + 3, mode, m.t));
}

// Runs one case `reps` times (graph: captured once, launched reps times). Every result is read after cudaStreamSynchronize
// into pinned memory. overlap: the reader started before the writer's last write; caught: the hazard is visible.
inline ProbeResult run_case(Case c, bool pdl, bool graph, int reps, const ProbeMem& m, cudaStream_t s) {
    ProbeResult r;
    cudaGraphExec_t ge = nullptr;
    if (graph) {
        cudaGraph_t g;
        PDLH_CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        enqueue_case(c, pdl, m, s);
        PDLH_CK(cudaStreamEndCapture(s, &g));
        PDLH_CK(cudaGraphInstantiate(&ge, g, 0));
        PDLH_CK(cudaGraphDestroy(g));
    }
    for (int i = 0; i < reps; ++i) {
        if (graph)
            PDLH_CK(cudaGraphLaunch(ge, s));
        else
            enqueue_case(c, pdl, m, s);
        PDLH_CK(cudaMemcpyAsync(m.hf, m.f, 4 * sizeof(float), cudaMemcpyDeviceToHost, s));
        PDLH_CK(cudaMemcpyAsync(m.ht, m.t, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, s));
        PDLH_CK(cudaStreamSynchronize(s));
        const bool ov = m.ht[2] != 0 && m.ht[2] < m.ht[0];
        bool hz = false;
        if (c == Case::S2) hz = m.hf[2] != 1.f;   // the writer's own scratch was overwritten
        else if (c == Case::S1) hz = m.hf[3] != 1.f;
        else hz = m.hf[3] != 1.f;                 // R1, S3, T: the post-wait read must see the last write
        r.overlap += ov ? 1 : 0;
        r.caught += hz ? 1 : 0;
        ++r.reps;
    }
    if (ge) PDLH_CK(cudaGraphExecDestroy(ge));
    return r;
}

// Part A. Returns the number of failed expectations; prints one line per (form, case, attribute) and one verdict per form.
// Expectations: serial = no overlap and no hazard in any case. PDL: R1 and T never stale; S1 / S2 caught in EVERY overlapped
// repetition; graph form (the engine's form) must overlap in every repetition of R1 / S1 / S2, else nothing is proven.
// S3 is a fact report (spec hazard): printed, gated only to "caught when overlapped" being reported.
inline int run_probes(int reps) {
    int fails = 0;
    cudaStream_t s;
    PDLH_CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    ProbeMem m;
    PDLH_CK(cudaMalloc(&m.f, 4 * sizeof(float)));
    PDLH_CK(cudaMalloc(&m.t, 3 * sizeof(unsigned long long)));
    PDLH_CK(cudaMallocHost(&m.hf, 4 * sizeof(float)));
    PDLH_CK(cudaMallocHost(&m.ht, 3 * sizeof(unsigned long long)));
    for (bool graph : {false, true}) {
        const char* form = graph ? "graph" : "stream";
        int form_fails = 0;
        for (Case c : {Case::R1, Case::S1, Case::S2, Case::S3, Case::T}) {
            for (bool pdl : {false, true}) {
                const ProbeResult r = run_case(c, pdl, graph, reps, m, s);
                std::printf("-- pdl probe %s %s %s: overlap %d/%d, hazard seen %d/%d\n", form, pdl ? "pdl   " : "serial",
                            case_name(c), r.overlap, r.reps, r.caught, r.reps);
                bool ok = true;
                if (!pdl) ok = r.overlap == 0 && r.caught == 0;
                else if (c == Case::R1 || c == Case::T) ok = r.caught == 0 && (!graph || c == Case::T || r.overlap == r.reps);
                else if (c == Case::S1 || c == Case::S2) {
                    int caught_overlapped = r.caught;  // a hazard can only be seen when the reader started early
                    ok = caught_overlapped >= r.overlap && (!graph || (r.overlap == r.reps && r.caught == r.reps));
                }
                if (!ok) {
                    ++form_fails;
                    std::printf("FAIL pdl probe %s %s %s\n", form, pdl ? "pdl" : "serial", case_name(c));
                }
            }
        }
        std::printf("%s pdl probes %s: %d failed expectations\n", form_fails ? "FAIL" : "PASS", form, form_fails);
        fails += form_fails;
    }
    PDLH_CK(cudaStreamSynchronize(s));
    PDLH_CK(cudaFree(m.f));
    PDLH_CK(cudaFree(m.t));
    PDLH_CK(cudaFreeHost(m.hf));
    PDLH_CK(cudaFreeHost(m.ht));
    PDLH_CK(cudaStreamDestroy(s));
    return fails;
}

// ---- Part B helpers -------------------------------------------------------------------------------------------------
constexpr int kMaxLive = 8;
struct Live {
    float* p[kMaxLive];
    unsigned long long n[kMaxLive];  // floats
    int count;
};
constexpr unsigned kPoisonBits = 0x7fbadbadu;  // quiet NaN with a payload: any use propagates NaN

// win[0] = latest release (max over blocks), win[1] = earliest restore (min over blocks): the poison window. One gate per
// slot; win must be preset to {0, ~0}.
__global__ void poison_gate(Live live, float* shadow, unsigned long long* win, unsigned long long ns) {
    wait();
    const unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long nt = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    unsigned long long off = 0;
    for (int b = 0; b < live.count; ++b) {
        for (unsigned long long i = tid; i < live.n[b]; i += nt) {
            shadow[off + i] = live.p[b][i];
            live.p[b][i] = __uint_as_float(kPoisonBits);
        }
        off += live.n[b];
    }
    __threadfence();
    __syncthreads();
    trigger();
    if (threadIdx.x == 0) atomicMax(&win[0], now());
    spin(ns);
    if (threadIdx.x == 0) atomicMin(&win[1], now());
    __syncthreads();
    off = 0;
    for (int b = 0; b < live.count; ++b) {
        for (unsigned long long i = tid; i < live.n[b]; i += nt) live.p[b][i] = shadow[off + i];
        off += live.n[b];
    }
}

// dst = src (mode 0, correct), mode 1 reads src BEFORE its wait (seeded), mode 2 writes dst BEFORE its wait, then adds src
// after it (seeded: its pre-wait write lands in a buffer the gate still owns). Serial results: modes 0 / 1: dst = src;
// mode 2: dst = 3 + src. entry[0] = earliest block entry (atomicMin, preset ~0).
__global__ void twin_copy(const float* src, float* dst, unsigned n, int mode, unsigned long long* entry) {
    if (threadIdx.x == 0) atomicMin(entry, now());
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (mode == 1 && i < n) v = src[i];
    if (mode == 2 && i < n) dst[i] = 3.f;
    trigger();
    wait();
    if (i >= n) return;
    if (mode == 0) dst[i] = src[i];
    if (mode == 1) dst[i] = v;
    if (mode == 2) dst[i] = __fadd_rn(dst[i], src[i]);
}

}  // namespace pdlh
