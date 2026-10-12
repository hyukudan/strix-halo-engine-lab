/* SPDX-License-Identifier: GPL-2.0-only */
/* Shared by the actual kernel extension and hardware-free property tests. */
#ifndef STRIX_GPU_STREAM_LOGIC_H
#define STRIX_GPU_STREAM_LOGIC_H
#ifdef __KERNEL__
#include <uapi/linux/strix_gpu_stream.h>
#include <linux/string.h>
#else
#include "strix/linux_direct_gpu_stream_uapi.h"
#include <string.h>
#endif

struct sgu_dma_segment {
    __u64 logical_offset;
    __u64 dma_address;
    __u64 bytes;
};
struct sgu_controller_lifetime {
    __u32 pending_callbacks;
    __u32 stop_confirmed;
    __u32 cancel_requested;
    __u32 paths_removed;
};

static inline int sgu_window_valid(__u64 full, __u64 offset, __u64 bytes, __u32 frames)
{
    return full && full <= SGU_MAX_IMPORTED_BYTES && frames >= 32 && frames <= SGU_MAX_FRAMES &&
        bytes == (__u64)frames * SGU_FRAME_BYTES && !(offset % SGU_FRAME_BYTES) &&
        offset <= full && bytes <= full - offset;
}

/* cursor advances monotonically: import mapping is O(segments + frames).
 * Every RX frame reserves a full 4096-byte DMA range, including the terminal.
 * Zero is a valid DMA bus address; physical/virtual address substitution is not. */
static inline int sgu_resolve_frame(const struct sgu_dma_segment *segments, __u32 count,
                                   __u64 offset, __u32 *cursor, __u64 *dma)
{
    const __u64 maximum = ~(__u64)0;
    __u32 i;

    if (!segments || !count || count > SGU_MAX_SG_ENTRIES || !cursor || !dma || *cursor >= count)
        return 0;
    for (i = *cursor; i < count; ++i) {
        const struct sgu_dma_segment *s = &segments[i];
        __u64 inside;

        if (!s->bytes || s->bytes > maximum - s->logical_offset || s->bytes > maximum - s->dma_address)
            return 0;
        if (offset < s->logical_offset)
            return 0;
        inside = offset - s->logical_offset;
        if (inside >= s->bytes)
            continue;
        if (SGU_FRAME_BYTES > s->bytes - inside)
            return 0;
        *cursor = i;
        *dma = s->dma_address + inside;
        return 1;
    }
    return 0;
}

static inline int sgu_endpoint_equal(const struct sgu_fence *a, const struct sgu_fence *b)
{
    return a->abi_major == SGU_ABI_MAJOR && a->abi_minor == SGU_ABI_MINOR &&
        a->abi_cookie == SGU_ABI_COOKIE && a->abi_major == b->abi_major &&
        a->abi_minor == b->abi_minor && a->abi_cookie == b->abi_cookie &&
        !memcmp(a->device_instance, b->device_instance, 16) &&
        !memcmp(a->local_uuid, b->local_uuid, 16) && !memcmp(a->peer_uuid, b->peer_uuid, 16) &&
        a->route == b->route && a->in_hopid == b->in_hopid && a->out_hopid == b->out_hopid;
}

static inline int sgu_fence_equal(const struct sgu_fence *a, const struct sgu_fence *b)
{
    return sgu_endpoint_equal(a, b) && a->connection_epoch && a->pool_binding_identity &&
        a->import_generation && a->connection_epoch == b->connection_epoch &&
        a->pool_binding_identity == b->pool_binding_identity && a->import_generation == b->import_generation;
}

static inline int sgu_prepare_valid(const struct sgu_prepare *p, __u32 frames, __u32 *payload_frames)
{
    const __u64 terminal = (__u64)(frames - 1U) * SGU_FRAME_BYTES;
    __u64 count;

    if (!p || !payload_frames || frames < 32 || frames > SGU_MAX_FRAMES || !p->handle || !p->sequence || p->direction > SGU_DIRECTION_RX ||
        p->reserved || !p->payload_bytes || !p->terminal_bytes || p->terminal_bytes > SGU_FRAME_BYTES ||
        p->pool_offset % SGU_FRAME_BYTES || p->pool_offset >= terminal || p->payload_bytes > terminal - p->pool_offset)
        return 0;
    count = (p->payload_bytes - 1U) / SGU_FRAME_BYTES + 1U;
    if (count >= frames)
        return 0;
    *payload_frames = (__u32)count;
    return 1;
}

static inline int sgu_controller_queue(struct sgu_controller_lifetime *l)
{
    if (l->stop_confirmed || l->pending_callbacks >= 2U * SGU_MAX_FRAMES)
        return 0;
    ++l->pending_callbacks;
    return 1;
}
static inline int sgu_controller_callback(struct sgu_controller_lifetime *l)
{
    if (!l->pending_callbacks)
        return 0;
    --l->pending_callbacks;
    return 1;
}
static inline int sgu_controller_stopped(struct sgu_controller_lifetime *l)
{
    if (l->pending_callbacks)
        return 0;
    l->stop_confirmed = 1;
    return 1;
}
static inline int sgu_controller_can_release(const struct sgu_controller_lifetime *l)
{
    return l->stop_confirmed && l->paths_removed && !l->pending_callbacks;
}

#endif
