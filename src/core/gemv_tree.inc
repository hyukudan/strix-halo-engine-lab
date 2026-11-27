// Shared block-tree scalar GEMV body (see gemv_tree.hpp). Textual body: include it inside a kernel,
// like gemv_rows_direct.inc, for the same reason: the body written against callables (lambdas or
// binding members) compiled to a different instruction order for the large codebook decoders
// (IQ3_XXS) on gfx1151, while this body compiles to the instructions of a kernel that spells it.
// In scope at the include:
//   macros     GEMV_TREE_THREADS            lanes of the block = leaves of the tree (a power of two)
//              GEMV_TREE_SCRATCH            float[GEMV_TREE_THREADS] in LDS
//              GEMV_TREE_WIDTH              input width (features)
//              GEMV_TREE_ACTIVATION(f)      the activation of feature f as float
//              GEMV_TREE_WEIGHT(f)          the weight of feature f of this block's row as float
// Lane t accumulates features t, t + GEMV_TREE_THREADS, ... in ascending order as
// sum = sum + activation * weight from +0; the tree adds scratch[t] += scratch[t + stride] for
// stride = GEMV_TREE_THREADS / 2 .. 1 with a barrier after each stage. On exit scratch[0] holds the
// row's sum and every lane has passed the last barrier. The body undefines its macros.
{
    float gemv_tree_sum = 0.0F;
    for (std::uint32_t feature = static_cast<std::uint32_t>(threadIdx.x); feature < (GEMV_TREE_WIDTH);
         feature += (GEMV_TREE_THREADS)) {
        gemv_tree_sum += GEMV_TREE_ACTIVATION(feature) * GEMV_TREE_WEIGHT(feature);
    }
    GEMV_TREE_SCRATCH[threadIdx.x] = gemv_tree_sum;
    __syncthreads();
    for (std::uint32_t stride = (GEMV_TREE_THREADS) / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            GEMV_TREE_SCRATCH[threadIdx.x] += GEMV_TREE_SCRATCH[threadIdx.x + stride];
        }
        __syncthreads();
    }
}
#undef GEMV_TREE_THREADS
#undef GEMV_TREE_SCRATCH
#undef GEMV_TREE_WIDTH
#undef GEMV_TREE_ACTIVATION
#undef GEMV_TREE_WEIGHT
