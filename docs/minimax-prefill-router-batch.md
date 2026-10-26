# Prefill row-batched router (draft)

STRIX_MM_PF_ROUTER_BATCH=1 enables a separate router launch for post-exchange
calls with more than eight rows. The default is off. STRIX_MM_PF_ROUTER_ROWS
selects 4 or 8 rows per weight pass (default 8); other nonempty values fail at
Engine construction. Engine::set_prefill_router_batch and prefill_router_batch
allow runtime selection without rereading the environment.

The common Engine::Impl::enqueue_post wrapper covers the communicator,
xfer (including receive-flag waits), and prefill module paths, including the
interleaved post_view schedule. It preserves the call's ids and weights pointers
(in particular wts4_d at route_d + n * kTopE), stream, prepare jobs and Ilp mode.
The original post kernel runs with router_w=nullptr, followed on that stream by
the batched router reading its global y. No post-exchange prefill path is excluded.
Calls that route inside MoE still do so; calls with n <= 8 keep the original
launch and routing behavior. step accepts at most eight rows.

Each lane still starts at positive zero, performs ascending j=lane+32*q FMAs,
and adds the same XOR-shuffle partners in order 16,8,4,2,1. Each shared weight
load feeds independent row accumulators. The global y store and original LDS ys
store receive the same float o. Sigmoid, bias, ascending expert scan, tie rule,
sum order, weight expression and row-major output layout are unchanged. Tail
rows neither read y nor write outputs. The post kernel itself is unchanged.

Run strix-hip-minimax-moe-tests for byte comparisons of h, y, ids and weights
for R=4 and R=8 at n=1,7,8,9,15,16,17,100,1024,4096. Cases include random data,
all-expert ties, signed zeros, subnormals and large finite inputs. All Ilp modes
are also checked at n=17, and output guards check masked tails. HIP events report
median-of-20 old and split post+router times at n=1024 and n=4096, excluding
input reset and transfer time.

Use --ab pfrouter:PROMPT.u32:BLOCK for A B A B A B A prefills, with mandatory
per-arm cache-state and final-row hashes and the existing token comparisons.
Run this on both ranks with representative complete workloads and both row
batch sizes. Compiler, native HIP tests, model fidelity and performance gates
have not been run on the development PC (no HIP toolchain or GPU). This switch
remains an unqualified draft until those gates pass.
