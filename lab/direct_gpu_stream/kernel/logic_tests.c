/* SPDX-License-Identifier: GPL-2.0-only */
#include "gpu_stream_logic.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>

static void mapping_properties(void)
{
    unsigned seed;
    for (seed = 1; seed <= 4096; ++seed) {
        struct sgu_dma_segment segments[8];
        __u64 logical = 0, dma;
        __u32 cursor = 0, i, frames;
        unsigned random = seed;

        for (i = 0; i < 8; ++i) {
            random = random * 1664525U + 1013904223U;
            frames = random % 31U + 1U;
            segments[i].logical_offset = logical;
            segments[i].dma_address = (__u64)i * 0x100000U;
            segments[i].bytes = (__u64)frames * SGU_FRAME_BYTES;
            logical += segments[i].bytes;
        }
        for (__u64 offset = 0; offset < logical; offset += SGU_FRAME_BYTES) {
            __u32 oracle = 0;
            while (offset >= segments[oracle].logical_offset + segments[oracle].bytes)
                ++oracle;
            assert(sgu_resolve_frame(segments, 8, offset, &cursor, &dma));
            assert(cursor == oracle);
            assert(dma == segments[oracle].dma_address + offset - segments[oracle].logical_offset);
        }
        assert(!sgu_resolve_frame(segments, 8, logical, &cursor, &dma));
    }
    {
        struct sgu_dma_segment split[] = {{0, 0, 2048}, {2048, 0x1000, 4096}};
        __u32 cursor = 0;
        __u64 dma = 99;
        assert(!sgu_resolve_frame(split, 2, 0, &cursor, &dma));
        assert(sgu_resolve_frame(split, 2, 2048, &cursor, &dma) && dma == 0x1000);
        cursor = 0; split[0].bytes = 4096;
        assert(sgu_resolve_frame(split, 2, 0, &cursor, &dma) && dma == 0); /* DMA zero valid */
        split[0].dma_address = ~(__u64)0 - 2048;
        assert(!sgu_resolve_frame(split, 2, 0, &cursor, &dma));
        split[0].dma_address = 0; split[0].logical_offset = ~(__u64)0 - 2048;
        assert(!sgu_resolve_frame(split, 2, split[0].logical_offset, &cursor, &dma));
        assert(!sgu_resolve_frame(NULL, 2, 0, &cursor, &dma));
        assert(!sgu_resolve_frame(split, 0, 0, &cursor, &dma));
        cursor = 2;
        assert(!sgu_resolve_frame(split, 2, 0, &cursor, &dma));
    }
}

static void ownership_properties(void)
{
    unsigned queued, completed;
    for (queued = 0; queued <= 128; ++queued) {
        for (completed = 0; completed <= queued; ++completed) {
            struct sgu_controller_lifetime l = {0, 1, 0, 1};
            assert(sgu_controller_can_release(&l));
            assert(!sgu_controller_queue(&l)); /* stopped controller cannot enqueue */
            l.stop_confirmed = 0;
            l.paths_removed = 0;
            for (unsigned i = 0; i < queued; ++i) assert(sgu_controller_queue(&l));
            for (unsigned i = 0; i < completed; ++i) assert(sgu_controller_callback(&l));
            l.cancel_requested = 1; /* ACK neither drains nor releases */
            assert(l.pending_callbacks == queued - completed);
            assert(!sgu_controller_can_release(&l));
            if (completed != queued) assert(!sgu_controller_stopped(&l));
            while (l.pending_callbacks) assert(sgu_controller_callback(&l));
            assert(!sgu_controller_can_release(&l)); /* callbacks alone are not stop */
            assert(sgu_controller_stopped(&l)); /* actual synchronous stop boundary */
            assert(!sgu_controller_can_release(&l)); /* failed path teardown still retains */
            l.paths_removed = 1;
            assert(sgu_controller_can_release(&l));
            assert(!sgu_controller_callback(&l)); /* duplicate/late callback rejected */
        }
    }
    {
        struct sgu_controller_lifetime l = {0, 0, 0, 0};
        for (unsigned i = 0; i < 2U * SGU_MAX_FRAMES; ++i) assert(sgu_controller_queue(&l));
        assert(!sgu_controller_queue(&l));
        assert(!sgu_controller_can_release(&l));
    }
}

static void geometry_and_fences(void)
{
    struct sgu_fence a = {0}, b;
    struct sgu_prepare operation = {0};
    __u32 frames;

    a.abi_major = SGU_ABI_MAJOR; a.abi_minor = SGU_ABI_MINOR; a.abi_cookie = SGU_ABI_COOKIE;
    a.device_instance[0] = 1; a.local_uuid[0] = 2; a.peer_uuid[0] = 3;
    a.route = 4; a.in_hopid = 5; a.out_hopid = 6;
    a.connection_epoch = 7; a.pool_binding_identity = 8; a.import_generation = 9;
    b = a;
    assert(sgu_fence_equal(&a, &b));
    for (unsigned mutation = 0; mutation < 12; ++mutation) {
        b = a;
        switch (mutation) {
        case 0: ++b.abi_major; break;
        case 1: ++b.abi_minor; break;
        case 2: ++b.abi_cookie; break;
        case 3: ++b.device_instance[0]; break;
        case 4: ++b.local_uuid[0]; break;
        case 5: ++b.peer_uuid[0]; break;
        case 6: ++b.route; break;
        case 7: ++b.in_hopid; break;
        case 8: ++b.out_hopid; break;
        case 9: ++b.connection_epoch; break;
        case 10: ++b.pool_binding_identity; break;
        case 11: ++b.import_generation; break;
        }
        assert(!sgu_fence_equal(&a, &b));
    }
    b = a; b.import_generation = 0;
    assert(!sgu_fence_equal(&a, &b));
    assert(sgu_window_valid(64ULL * SGU_FRAME_BYTES, SGU_FRAME_BYTES, 32ULL * SGU_FRAME_BYTES, 32));
    assert(!sgu_window_valid(32ULL * SGU_FRAME_BYTES, 1, 32ULL * SGU_FRAME_BYTES, 32));
    assert(!sgu_window_valid(32ULL * SGU_FRAME_BYTES, SGU_FRAME_BYTES, 32ULL * SGU_FRAME_BYTES, 32));
    assert(!sgu_window_valid(~(__u64)0, 0, 32ULL * SGU_FRAME_BYTES, 32));
    operation.handle = 1; operation.sequence = 1; operation.payload_bytes = 5120;
    operation.terminal_bytes = 32;
    assert(sgu_prepare_valid(&operation, 32, &frames) && frames == 2);
    operation.payload_bytes = 31ULL * SGU_FRAME_BYTES;
    assert(sgu_prepare_valid(&operation, 32, &frames) && frames == 31);
    ++operation.payload_bytes;
    assert(!sgu_prepare_valid(&operation, 32, &frames));
    operation.payload_bytes = 1; operation.pool_offset = 31ULL * SGU_FRAME_BYTES;
    assert(!sgu_prepare_valid(&operation, 32, &frames));
    operation.pool_offset = 0; operation.terminal_bytes = SGU_FRAME_BYTES + 1;
    assert(!sgu_prepare_valid(&operation, 32, &frames));
    operation.terminal_bytes = 32; operation.direction = 2;
    assert(!sgu_prepare_valid(&operation, 32, &frames));
    assert(!sgu_prepare_valid(NULL, 32, &frames));
    assert(!sgu_prepare_valid(&operation, 32, NULL));
}

int main(void)
{
    mapping_properties(); ownership_properties(); geometry_and_fences();
    puts("direct GPU stream kernel logic properties passed");
    return 0;
}
