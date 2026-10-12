// SPDX-License-Identifier: GPL-2.0-only
/* Project-owned experimental DMA-BUF extension to the GPL-2.0 USB4STREAM
 * driver. Included by stream.c; upstream Intel notices remain unchanged there.
 * Controller ownership only: no GPU visibility or remote completion proof.
 */
#include <linux/dma-buf.h>
#include <linux/dma-fence.h>
#include <linux/dma-resv.h>
#include <linux/kdev_t.h>
#include <linux/random.h>
#include <linux/scatterlist.h>
#include <linux/vmalloc.h>
#include "tb.h"
#include "gpu_stream_logic.h"

MODULE_IMPORT_NS("DMA_BUF");

enum sgu_op_phase { SGU_OP_EMPTY, SGU_OP_PREPARED, SGU_OP_PENDING,
                    SGU_OP_COMPLETE, SGU_OP_TAKEN, SGU_OP_STANDBY };
struct sgu_hw_operation;
struct sgu_hw_frame {
    struct ring_frame frame;
    struct tbstream_gpu_context *context;
    struct sgu_hw_operation *operation;
    struct sgu_frame_result result;
    bool queued;
};
struct sgu_window {
    struct dma_buf *buffer;
    struct dma_buf_attachment *attachment;
    struct sg_table *mapping;
    struct device *device;
    enum dma_data_direction direction;
    struct sgu_hw_frame *frames;
    struct sgu_buffer binding;
};
struct sgu_dma_fence {
    struct dma_fence base;
    spinlock_t lock;
};
struct sgu_hw_operation {
    struct sgu_prepare request;
    struct dma_fence *fence;
    enum sgu_op_phase phase;
    u32 payload_frames, total_frames, remaining, event_cursor, error_flags;
    u64 completion_ns;
    bool event_pending, retired, canceled;
};
struct tbstream_gpu_context {
    struct tbstream_dev *sdev;
    spinlock_t lock; /* callbacks never acquire sdev->lock */
    struct sgu_fence identity;
    struct sgu_window windows[2];
    struct sgu_hw_operation operations[2];
    struct sgu_controller_lifetime lifetime;
    struct tbstream *stream_owner;
    struct device *service_owner, *dma_owner, *xdomain_owner, *domain_owner;
    u64 import_counter, last_epoch, last_handle, last_sequence[2], fence_context, fence_sequence[2];
    bool imported, started, armed, stopped, failed, rearming_rx, quarantined;
};

static const char *sgu_fence_driver(struct dma_fence *fence) { return "strix-usb4stream"; }
static const char *sgu_fence_timeline(struct dma_fence *fence) { return "controller-dma"; }
static void sgu_fence_free(struct dma_fence *fence)
{
    kfree(container_of(fence, struct sgu_dma_fence, base));
}
static const struct dma_fence_ops sgu_fence_ops = {
    .get_driver_name = sgu_fence_driver, .get_timeline_name = sgu_fence_timeline,
    .release = sgu_fence_free,
};

static bool sgu_ring_running(struct tb_ring *ring)
{
    unsigned long flags;
    bool running;

    spin_lock_irqsave(&ring->lock, flags);
    running = ring->running; /* public API field is a bit-field */
    spin_unlock_irqrestore(&ring->lock, flags);
    return running;
}

static struct sgu_hw_frame *sgu_frame_at(struct tbstream_gpu_context *ctx,
                                      struct sgu_hw_operation *op, u32 index)
{
    u32 slot = index < op->payload_frames ? op->request.pool_offset / SGU_FRAME_BYTES + index :
               ctx->sdev->ring_size - 1U;
    return &ctx->windows[op->request.direction].frames[slot];
}

static void sgu_signal(struct dma_fence *fence, int error)
{
    if (!fence)
        return;
    if (error)
        dma_fence_set_error(fence, error);
    dma_fence_signal(fence);
    dma_fence_put(fence);
}

static void sgu_complete_locked(struct tbstream_gpu_context *ctx, struct sgu_hw_operation *op,
                               struct dma_fence **signal, int *error)
{
    if (op->remaining)
        return;
    op->completion_ns = ktime_get_ns();
    if (ctx->rearming_rx && op->request.direction == SGU_DIRECTION_RX && op->phase == SGU_OP_STANDBY) {
        *signal = dma_fence_get(op->fence);
        *error = op->error_flags ? -EIO : 0; /* unused standby canceled by explicit reconfiguration */
        return;
    }
    op->phase = SGU_OP_COMPLETE;
    op->event_cursor = 0;
    op->event_pending = true;
    *signal = dma_fence_get(op->fence);
    *error = op->error_flags ? -EIO : op->canceled ? -ECANCELED : 0;
}

static void sgu_callback(struct tb_ring *ring, struct ring_frame *frame, bool canceled)
{
    struct sgu_hw_frame *f = container_of(frame, struct sgu_hw_frame, frame);
    struct tbstream_gpu_context *ctx = f->context;
    struct sgu_hw_operation *op = f->operation;
    struct dma_fence *signal = NULL;
    unsigned long flags;
    int error = 0;

    spin_lock_irqsave(&ctx->lock, flags);
    if (!f->queued || !op || !op->remaining || !sgu_controller_callback(&ctx->lifetime)) {
        WRITE_ONCE(ctx->failed, true);
        goto unlock;
    }
    f->queued = false;
    --op->remaining;
    if (canceled) {
        f->result.actual_bytes = 0;
        f->result.error_flags = SGU_FRAME_ERROR_CANCELED;
        if (!(ctx->rearming_rx && op->phase == SGU_OP_STANDBY))
            op->canceled = true;
    } else {
        f->result.actual_bytes = tb_ring_frame_size(frame);
        if (ring->is_tx) {
            /* TX descriptors retain submitted size; the callback is local DMA
             * completion only, never an acknowledgement from the other host. */
            f->result.actual_bytes = f->result.expected_bytes;
        } else {
            if (frame->flags & RING_DESC_CRC_ERROR)
                f->result.error_flags |= SGU_FRAME_ERROR_CRC;
            if (frame->flags & RING_DESC_BUFFER_OVERRUN)
                f->result.error_flags |= SGU_FRAME_ERROR_OVERRUN;
            if (f->result.actual_bytes != f->result.expected_bytes)
                f->result.error_flags |= SGU_FRAME_ERROR_LENGTH;
            if (frame->sof != TBSTREAM_FRAME_START || frame->eof != TBSTREAM_DATA)
                f->result.error_flags |= SGU_FRAME_ERROR_PROTOCOL;
            if (op->phase == SGU_OP_STANDBY)
                f->result.error_flags |= SGU_FRAME_ERROR_STANDBY_ARRIVAL;
        }
        op->error_flags |= f->result.error_flags;
        if (op->error_flags)
            WRITE_ONCE(ctx->failed, true);
    }
    sgu_complete_locked(ctx, op, &signal, &error);
unlock:
    spin_unlock_irqrestore(&ctx->lock, flags);
    sgu_signal(signal, error);
    wake_up_interruptible_poll(&ctx->sdev->wait, EPOLLIN | EPOLLRDNORM | EPOLLERR);
}

static void sgu_window_free(struct sgu_window *w)
{
    /* Only reached after tb_ring_stop returned and all callbacks drained. */
    kvfree(w->frames);
    w->frames = NULL;
    if (w->mapping)
        dma_buf_unmap_attachment_unlocked(w->attachment, w->mapping, w->direction);
    w->mapping = NULL;
    if (w->attachment)
        dma_buf_detach(w->buffer, w->attachment);
    w->attachment = NULL;
    if (w->buffer)
        dma_buf_put(w->buffer);
    w->buffer = NULL;
}

static int sgu_window_map(struct tbstream_gpu_context *ctx, struct sgu_window *w,
                          const struct sgu_buffer *b, u32 direction)
{
    struct sgu_dma_segment *segments = NULL;
    struct scatterlist *sg;
    struct inode *inode;
    u64 logical = 0, dma = 0;
    u32 cursor = 0, i;
    int ret = -EINVAL;

    if (b->fd < 0 || b->reserved || !b->allocation_generation ||
        !sgu_window_valid(b->full_bytes, b->window_offset, b->window_bytes, ctx->sdev->ring_size))
        return -EINVAL;
    w->buffer = dma_buf_get(b->fd);
    if (IS_ERR(w->buffer)) {
        ret = PTR_ERR(w->buffer);
        w->buffer = NULL;
        return ret;
    }
    inode = file_inode(w->buffer->file);
    if (w->buffer->size != b->full_bytes || huge_encode_dev(inode->i_sb->s_dev) != b->fd_device ||
        inode->i_ino != b->fd_inode)
        goto fail;
    if (direction == SGU_DIRECTION_RX && ctx->windows[SGU_DIRECTION_TX].buffer &&
        (w->buffer == ctx->windows[SGU_DIRECTION_TX].buffer ||
         w->buffer->resv == ctx->windows[SGU_DIRECTION_TX].buffer->resv))
        goto fail;
    w->device = tb_ring_dma_device(direction == SGU_DIRECTION_TX ? ctx->sdev->tx_ring.ring : ctx->sdev->rx_ring.ring);
    w->direction = direction == SGU_DIRECTION_TX ? DMA_TO_DEVICE : DMA_FROM_DEVICE;
    w->attachment = dma_buf_attach(w->buffer, w->device);
    if (IS_ERR(w->attachment)) {
        ret = PTR_ERR(w->attachment);
        w->attachment = NULL;
        goto fail;
    }
    w->mapping = dma_buf_map_attachment_unlocked(w->attachment, w->direction);
    if (IS_ERR(w->mapping)) {
        ret = PTR_ERR(w->mapping);
        w->mapping = NULL;
        goto fail;
    }
    if (!w->mapping->nents || w->mapping->nents > SGU_MAX_SG_ENTRIES)
        goto fail;
    segments = kvcalloc(w->mapping->nents, sizeof(*segments), GFP_KERNEL);
    w->frames = kvcalloc(ctx->sdev->ring_size, sizeof(*w->frames), GFP_KERNEL);
    if (!segments || !w->frames) {
        ret = -ENOMEM;
        goto fail;
    }
    for_each_sg(w->mapping->sgl, sg, w->mapping->nents, i) {
        u64 bytes = sg_dma_len(sg);

        /* A bounced mapping would add CPU payload copies in DMA sync. */
        if (!bytes || bytes > U64_MAX - logical || bytes > U64_MAX - (u64)sg_dma_address(sg) ||
            sg_dma_is_swiotlb(sg))
            goto fail;
        segments[i].logical_offset = logical;
        segments[i].dma_address = sg_dma_address(sg);
        segments[i].bytes = bytes;
        logical += bytes;
    }
    if (logical < b->full_bytes)
        goto fail;
    for (i = 0; i < ctx->sdev->ring_size; ++i) {
        struct sgu_hw_frame *f = &w->frames[i];

        if (!sgu_resolve_frame(segments, w->mapping->nents,
                               b->window_offset + (u64)i * SGU_FRAME_BYTES, &cursor, &dma) ||
            dma != (u64)(dma_addr_t)dma || dma > dma_get_mask(w->device) ||
            SGU_FRAME_BYTES - 1U > dma_get_mask(w->device) - dma)
            goto fail;
        f->context = ctx;
        f->frame.buffer_phy = dma;
        f->frame.callback = sgu_callback;
        INIT_LIST_HEAD(&f->frame.list);
        f->result.pool_offset = (u64)i * SGU_FRAME_BYTES;
        f->result.capacity = SGU_FRAME_BYTES;
        f->result.physical_slot = i;
    }
    w->binding = *b;
    kvfree(segments);
    return 0;
fail:
    kvfree(segments);
    sgu_window_free(w);
    return ret;
}

static void sgu_operation_clear(struct tbstream_gpu_context *ctx, struct sgu_hw_operation *op)
{
    struct dma_fence *fence;
    unsigned long flags;
    u32 i;

    spin_lock_irqsave(&ctx->lock, flags);
    for (i = 0; i < op->total_frames; ++i) {
        struct sgu_hw_frame *f = sgu_frame_at(ctx, op, i);

        f->operation = NULL;
        f->queued = false;
    }
    fence = op->fence;
    memset(op, 0, sizeof(*op));
    spin_unlock_irqrestore(&ctx->lock, flags);
    dma_fence_put(fence);
}

static int sgu_publish_fence(struct tbstream_gpu_context *ctx, struct sgu_hw_operation *op)
{
    struct sgu_window *w = &ctx->windows[op->request.direction];
    struct sgu_dma_fence *f;
    enum dma_resv_usage wait_usage = op->request.direction == SGU_DIRECTION_TX ? DMA_RESV_USAGE_WRITE : DMA_RESV_USAGE_READ;
    enum dma_resv_usage own_usage = op->request.direction == SGU_DIRECTION_TX ? DMA_RESV_USAGE_READ : DMA_RESV_USAGE_WRITE;
    long waited;
    struct dma_fence *previous;
    unsigned long flags;
    int ret;

    waited = dma_resv_wait_timeout(w->buffer->resv, wait_usage, true, msecs_to_jiffies(1000));
    if (waited < 0)
        return waited;
    if (!waited)
        return -ETIMEDOUT;
    f = kzalloc(sizeof(*f), GFP_KERNEL);
    if (!f)
        return -ENOMEM;
    if (ctx->fence_sequence[op->request.direction] == U64_MAX) {
        kfree(f);
        return -EOVERFLOW;
    }
    spin_lock_init(&f->lock);
    dma_fence_init(&f->base, &sgu_fence_ops, &f->lock,
        ctx->fence_context + op->request.direction, ++ctx->fence_sequence[op->request.direction]);
    ret = dma_resv_lock_interruptible(w->buffer->resv, NULL);
    if (ret)
        goto put;
    if (!dma_resv_test_signaled(w->buffer->resv, wait_usage)) {
        ret = -EAGAIN;
        goto unlock;
    }
    ret = dma_resv_reserve_fences(w->buffer->resv, 1);
    if (ret)
        goto unlock;
    dma_resv_add_fence(w->buffer->resv, &f->base, own_usage);
    dma_resv_unlock(w->buffer->resv);
    spin_lock_irqsave(&ctx->lock, flags);
    previous = op->fence;
    op->fence = &f->base;
    spin_unlock_irqrestore(&ctx->lock, flags);
    dma_fence_put(previous);
    return 0;
unlock:
    dma_resv_unlock(w->buffer->resv);
put:
    dma_fence_put(&f->base);
    return ret;
}

static int sgu_enqueue(struct tbstream_gpu_context *ctx, struct sgu_hw_operation *op, bool standby)
{
    struct sgu_window *w = &ctx->windows[op->request.direction];
    struct tb_ring *ring = op->request.direction == SGU_DIRECTION_TX ? ctx->sdev->tx_ring.ring : ctx->sdev->rx_ring.ring;
    unsigned long flags;
    struct dma_fence *signal = NULL;
    int ret, error = 0;
    u32 i, j;

    ret = sgu_publish_fence(ctx, op);
    if (ret)
        return ret;
    dma_sync_sgtable_for_device(w->device, w->mapping, w->direction);
    spin_lock_irqsave(&ctx->lock, flags);
    op->phase = standby ? SGU_OP_STANDBY : SGU_OP_PENDING;
    op->remaining = op->total_frames;
    op->error_flags = 0;
    op->canceled = false;
    op->event_pending = false;
    op->event_cursor = 0;
    spin_unlock_irqrestore(&ctx->lock, flags);
    for (i = 0; i < op->total_frames; ++i) {
        struct sgu_hw_frame *f = sgu_frame_at(ctx, op, i);

        f->result.actual_bytes = 0;
        f->result.error_flags = 0;
        f->frame.flags = 0;
        f->frame.sof = TBSTREAM_FRAME_START;
        f->frame.eof = TBSTREAM_DATA;
        f->frame.size = op->request.direction == SGU_DIRECTION_RX ? 0 :
                        f->result.expected_bytes == SGU_FRAME_BYTES ? 0 : f->result.expected_bytes;
        spin_lock_irqsave(&ctx->lock, flags);
        f->queued = sgu_controller_queue(&ctx->lifetime);
        spin_unlock_irqrestore(&ctx->lock, flags);
        if (!f->queued) {
            ret = -EIO;
            break;
        }
        ret = op->request.direction == SGU_DIRECTION_TX ? tb_ring_tx(ring, &f->frame) : tb_ring_rx(ring, &f->frame);
        if (ret) {
            spin_lock_irqsave(&ctx->lock, flags);
            f->queued = false;
            if (!sgu_controller_callback(&ctx->lifetime))
                WRITE_ONCE(ctx->failed, true);
            spin_unlock_irqrestore(&ctx->lock, flags);
            break;
        }
    }
    if (ret) {
        /* Some frames may already be owned by hardware. Never unwind imports
         * or discard their callbacks when a later enqueue fails. */
        spin_lock_irqsave(&ctx->lock, flags);
        WRITE_ONCE(ctx->failed, true);
        op->canceled = true;
        for (j = i; j < op->total_frames; ++j) {
            struct sgu_hw_frame *f = sgu_frame_at(ctx, op, j);

            f->result.actual_bytes = 0;
            f->result.error_flags = SGU_FRAME_ERROR_CANCELED;
            --op->remaining;
        }
        sgu_complete_locked(ctx, op, &signal, &error);
        spin_unlock_irqrestore(&ctx->lock, flags);
        sgu_signal(signal, error);
    }
    return ret;
}

static int sgu_live(struct tbstream_gpu_context *ctx, const struct sgu_fence *fence, bool cleanup)
{
    if (!sgu_fence_equal(fence, &ctx->identity) || !ctx->imported)
        return -ESTALE;
    if (!cleanup && (READ_ONCE(ctx->failed) || ctx->stopped || tbstream_dev_valid(ctx->sdev)))
        return -ESHUTDOWN;
    return 0;
}

static int tbstream_gpu_open(struct tbstream_dev *sdev)
{
    struct tbstream_gpu_context *ctx;
    struct tb_xdomain *xd = tbstream_dev_xdomain(sdev);
    uuid_t instance;
    struct tb_ring *ring;

    if (sdev->busy_poll || !xd || !xd->local_uuid || !xd->remote_uuid ||
        !xd->tb->cm_ops || xd->tb->cm_ops->driver_ready)
        return -EOPNOTSUPP;
    if (sdev->gpu_context)
        return -EBUSY;
    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return -ENOMEM;
    ctx->sdev = sdev;
    spin_lock_init(&ctx->lock);
    ctx->lifetime.stop_confirmed = 1;
    ctx->lifetime.paths_removed = 1;
    ctx->fence_context = dma_fence_context_alloc(2); /* independent TX/RX ordered timelines */
    ctx->stream_owner = tbstream_get(sdev->stream);
    ctx->service_owner = get_device(&sdev->stream->svc->dev);
    ctx->xdomain_owner = get_device(&xd->dev);
    ctx->domain_owner = get_device(&xd->tb->dev);
    uuid_gen(&instance);
    ctx->identity.abi_major = SGU_ABI_MAJOR;
    ctx->identity.abi_minor = SGU_ABI_MINOR;
    ctx->identity.abi_cookie = SGU_ABI_COOKIE;
    memcpy(ctx->identity.device_instance, instance.b, 16);
    memcpy(ctx->identity.local_uuid, xd->local_uuid->b, 16);
    memcpy(ctx->identity.peer_uuid, xd->remote_uuid->b, 16);
    ctx->identity.route = xd->route;
    ctx->identity.in_hopid = sdev->in_hopid;
    ctx->identity.out_hopid = sdev->out_hopid;
    ring = tb_ring_alloc_tx(xd->tb->nhi, -1, sdev->ring_size, RING_FLAG_FRAME | RING_FLAG_E2E);
    if (!ring)
        goto fail;
    sdev->tx_ring.ring = ring;
    ring = tb_ring_alloc_rx(xd->tb->nhi, -1, sdev->ring_size, RING_FLAG_FRAME | RING_FLAG_E2E,
                            sdev->tx_ring.ring->hop, BIT(TBSTREAM_FRAME_START), BIT(TBSTREAM_DATA), NULL, NULL);
    if (!ring) {
        tb_ring_free(sdev->tx_ring.ring);
        sdev->tx_ring.ring = NULL;
        goto fail;
    }
    sdev->rx_ring.ring = ring;
    ctx->dma_owner = get_device(tb_ring_dma_device(ring));
    sdev->gpu_context = ctx;
    return 0; /* allocation only: no path, ring start or DMA enqueue */
fail:
    put_device(ctx->domain_owner);
    put_device(ctx->xdomain_owner);
    put_device(ctx->service_owner);
    tbstream_put(ctx->stream_owner);
    kfree(ctx);
    return -ENOMEM;
}

static void tbstream_gpu_stop(struct tbstream_dev *sdev)
{
    struct tbstream_gpu_context *ctx = sdev->gpu_context;
    struct tb_xdomain *xd;
    unsigned long flags;

    if (!ctx)
        return;
    if (sdev->tx_ring.ring && sgu_ring_running(sdev->tx_ring.ring))
        tb_ring_stop(sdev->tx_ring.ring);
    if (sdev->rx_ring.ring && sgu_ring_running(sdev->rx_ring.ring))
        tb_ring_stop(sdev->rx_ring.ring);
    xd = tbstream_dev_xdomain(sdev);
    if (ctx->armed) {
        if (!xd || tb_xdomain_disable_paths(xd, sdev->out_hopid, sdev->tx_ring.ring->hop,
                                            sdev->in_hopid, sdev->rx_ring.ring->hop))
            WRITE_ONCE(ctx->failed, true); /* keep path/ring identities retained */
        else
            ctx->armed = false;
    }
    ctx->started = false;
    ctx->stopped = true;
    spin_lock_irqsave(&ctx->lock, flags);
    ctx->lifetime.paths_removed = !ctx->armed;
    if (!sgu_controller_stopped(&ctx->lifetime))
        WRITE_ONCE(ctx->failed, true);
    spin_unlock_irqrestore(&ctx->lock, flags);
    wake_up_interruptible_poll(&sdev->wait, EPOLLIN | EPOLLERR);
}

static bool sgu_can_release(struct tbstream_gpu_context *ctx)
{
    unsigned long flags;
    bool result;

    spin_lock_irqsave(&ctx->lock, flags);
    result = !ctx->armed && sgu_controller_can_release(&ctx->lifetime);
    spin_unlock_irqrestore(&ctx->lock, flags);
    return result;
}

static void tbstream_gpu_disconnect(struct tbstream_dev *sdev)
{
    if (!sdev->gpu_context)
        return;
    WRITE_ONCE(sdev->gpu_context->failed, true);
    tbstream_gpu_stop(sdev);
}

static bool tbstream_gpu_paths_pending(struct tbstream_dev *sdev)
{
    return sdev->gpu_context && sdev->gpu_context->armed;
}

static void sgu_release_windows(struct tbstream_gpu_context *ctx)
{
    sgu_operation_clear(ctx, &ctx->operations[SGU_DIRECTION_TX]);
    sgu_operation_clear(ctx, &ctx->operations[SGU_DIRECTION_RX]);
    sgu_window_free(&ctx->windows[SGU_DIRECTION_RX]);
    sgu_window_free(&ctx->windows[SGU_DIRECTION_TX]);
    ctx->imported = false;
    ctx->identity.import_generation = 0;
    ctx->identity.pool_binding_identity = 0;
    ctx->identity.connection_epoch = 0;
}

static void tbstream_gpu_close(struct tbstream_dev *sdev)
{
    struct tbstream_gpu_context *ctx = sdev->gpu_context;

    if (!ctx)
        return;
    tbstream_gpu_stop(sdev);
    if (WARN_ON_ONCE(!sgu_can_release(ctx))) {
        if (!ctx->quarantined) {
            tbstream_dev_get(sdev);
            __module_get(THIS_MODULE);
            ctx->quarantined = true;
        }
        return; /* retain device, stream, module, imports and callback sources */
    }
    sgu_release_windows(ctx);
    tb_ring_free(sdev->rx_ring.ring);
    tb_ring_free(sdev->tx_ring.ring);
    sdev->rx_ring.ring = NULL;
    sdev->tx_ring.ring = NULL;
    sdev->gpu_context = NULL;
    put_device(ctx->dma_owner);
    put_device(ctx->domain_owner);
    put_device(ctx->xdomain_owner);
    put_device(ctx->service_owner);
    tbstream_put(ctx->stream_owner);
    kfree(ctx);
}

static int sgu_import_windows(struct tbstream_gpu_context *ctx, struct sgu_import *request)
{
    int ret;

    if (ctx->imported || ctx->started || READ_ONCE(ctx->failed) ||
        !sgu_endpoint_equal(&request->fence, &ctx->identity) || request->fence.import_generation ||
        !request->fence.pool_binding_identity || !request->fence.connection_epoch ||
        request->fence.connection_epoch < ctx->last_epoch ||
        (request->fence.connection_epoch != ctx->last_epoch &&
         request->fence.connection_epoch <= ctx->sdev->gpu_last_epoch) ||
        request->reserved[0] || request->reserved[1] ||
        request->tx.full_bytes > SGU_MAX_IMPORTED_BYTES ||
        request->rx.full_bytes > SGU_MAX_IMPORTED_BYTES - request->tx.full_bytes ||
        ctx->import_counter == U64_MAX)
        return -EINVAL;
    ret = sgu_window_map(ctx, &ctx->windows[SGU_DIRECTION_TX], &request->tx, SGU_DIRECTION_TX);
    if (ret)
        return ret;
    ret = sgu_window_map(ctx, &ctx->windows[SGU_DIRECTION_RX], &request->rx, SGU_DIRECTION_RX);
    if (ret)
        goto free_tx;
    if (ctx->windows[0].buffer == ctx->windows[1].buffer ||
        ctx->windows[0].buffer->resv == ctx->windows[1].buffer->resv) {
        ret = -EINVAL;
        goto free_rx;
    }
    if (request->fence.connection_epoch != ctx->last_epoch)
        ctx->last_sequence[0] = ctx->last_sequence[1] = 0;
    ctx->last_epoch = request->fence.connection_epoch;
    ctx->sdev->gpu_last_epoch = request->fence.connection_epoch;
    ctx->identity.connection_epoch = request->fence.connection_epoch;
    ctx->identity.pool_binding_identity = request->fence.pool_binding_identity;
    ctx->identity.import_generation = ++ctx->import_counter;
    request->fence = ctx->identity;
    ctx->imported = true;
    ctx->stopped = false;
    return 0;
free_rx:
    sgu_window_free(&ctx->windows[SGU_DIRECTION_RX]);
free_tx:
    sgu_window_free(&ctx->windows[SGU_DIRECTION_TX]);
    return ret;
}

static int sgu_arm(struct tbstream_gpu_context *ctx)
{
    struct tb_xdomain *xd = tbstream_dev_xdomain(ctx->sdev);
    int ret;
    unsigned long flags;

    if (ctx->started || !xd)
        return -EBUSY;
    tb_ring_throttling(ctx->sdev->tx_ring.ring, ctx->sdev->throttling);
    tb_ring_throttling(ctx->sdev->rx_ring.ring, ctx->sdev->throttling);
    tb_ring_start(ctx->sdev->tx_ring.ring);
    tb_ring_start(ctx->sdev->rx_ring.ring);
    ctx->started = true;
    spin_lock_irqsave(&ctx->lock, flags);
    ctx->lifetime.stop_confirmed = 0;
    spin_unlock_irqrestore(&ctx->lock, flags);
    if (!sgu_ring_running(ctx->sdev->tx_ring.ring) || !sgu_ring_running(ctx->sdev->rx_ring.ring)) {
        WRITE_ONCE(ctx->failed, true);
        tbstream_gpu_stop(ctx->sdev);
        return -ENXIO;
    }
    ret = tb_xdomain_enable_paths(xd, ctx->sdev->out_hopid, ctx->sdev->tx_ring.ring->hop,
                                 ctx->sdev->in_hopid, ctx->sdev->rx_ring.ring->hop);
    if (ret) {
        WRITE_ONCE(ctx->failed, true);
        tbstream_gpu_stop(ctx->sdev);
        return ret;
    }
    ctx->armed = true;
    spin_lock_irqsave(&ctx->lock, flags);
    ctx->lifetime.paths_removed = 0;
    spin_unlock_irqrestore(&ctx->lock, flags);
    return 0;
}

static int sgu_prepare_operation(struct tbstream_gpu_context *ctx, const struct sgu_prepare *request)
{
    struct sgu_hw_operation *op;
    unsigned long flags;
    u32 payload_frames, i;
    int ret = sgu_live(ctx, &request->fence, false);

    if (ret)
        return ret;
    if (!sgu_prepare_valid(request, ctx->sdev->ring_size, &payload_frames) ||
        request->handle <= ctx->last_handle || request->sequence <= ctx->last_sequence[request->direction])
        return -EINVAL;
    op = &ctx->operations[request->direction];
    spin_lock_irqsave(&ctx->lock, flags);
    if (op->phase == SGU_OP_STANDBY && op->retired && request->direction == SGU_DIRECTION_RX) {
        if (op->remaining != op->total_frames || READ_ONCE(ctx->failed)) {
            spin_unlock_irqrestore(&ctx->lock, flags);
            return -ESHUTDOWN;
        }
        ctx->rearming_rx = true;
        spin_unlock_irqrestore(&ctx->lock, flags);
        tb_ring_stop(ctx->sdev->rx_ring.ring); /* callback lock does not invert sdev mutex */
        spin_lock_irqsave(&ctx->lock, flags);
        ctx->rearming_rx = false;
        if (op->remaining || READ_ONCE(ctx->failed)) {
            WRITE_ONCE(ctx->failed, true);
            spin_unlock_irqrestore(&ctx->lock, flags);
            return -EIO;
        }
        spin_unlock_irqrestore(&ctx->lock, flags);
        sgu_operation_clear(ctx, op);
        tb_ring_start(ctx->sdev->rx_ring.ring);
        if (!sgu_ring_running(ctx->sdev->rx_ring.ring)) {
            WRITE_ONCE(ctx->failed, true);
            return -ENXIO;
        }
        spin_lock_irqsave(&ctx->lock, flags);
    }
    if (op->phase != SGU_OP_EMPTY) {
        spin_unlock_irqrestore(&ctx->lock, flags);
        return -EBUSY;
    }
    op->request = *request;
    op->payload_frames = payload_frames;
    op->total_frames = payload_frames + 1U;
    op->phase = SGU_OP_PREPARED;
    for (i = 0; i < op->total_frames; ++i) {
        struct sgu_hw_frame *f = sgu_frame_at(ctx, op, i);
        u64 remaining = i < payload_frames ? request->payload_bytes - (u64)i * SGU_FRAME_BYTES : request->terminal_bytes;

        f->operation = op;
        f->result.expected_bytes = min_t(u64, remaining, SGU_FRAME_BYTES);
        f->result.actual_bytes = 0;
        f->result.error_flags = 0;
    }
    spin_unlock_irqrestore(&ctx->lock, flags);
    ctx->last_handle = request->handle;
    ctx->last_sequence[request->direction] = request->sequence;
    return 0;
}

static long sgu_event_ioctl(struct tbstream_gpu_context *ctx, void __user *user)
{
    struct sgu_event *out;
    struct sgu_hw_operation *op = NULL;
    unsigned long flags;
    u32 direction, i;
    int ret;

    out = kzalloc(sizeof(*out), GFP_KERNEL);
    if (!out)
        return -ENOMEM;
    if (copy_from_user(out, user, sizeof(*out))) {
        ret = -EFAULT;
        goto free;
    }
    ret = sgu_live(ctx, &out->fence, true);
    if (ret)
        goto free;
    if (memchr_inv((u8 *)out + sizeof(out->fence), 0, sizeof(*out) - sizeof(out->fence))) {
        ret = -EINVAL;
        goto free;
    }
    spin_lock_irqsave(&ctx->lock, flags);
    for (direction = 0; direction < 2; ++direction) {
        struct sgu_hw_operation *candidate = &ctx->operations[direction];

        if (candidate->event_pending && (!op || candidate->completion_ns < op->completion_ns))
            op = candidate;
    }
    if (!op) {
        spin_unlock_irqrestore(&ctx->lock, flags);
        ret = -EAGAIN;
        goto free;
    }
    out->handle = op->request.handle;
    out->sequence = op->request.sequence;
    out->payload_bytes = op->request.payload_bytes;
    out->monotonic_ns = op->completion_ns;
    out->direction = op->request.direction;
    out->status = op->error_flags ? SGU_EVENT_FAILED : op->canceled ? SGU_EVENT_CANCELED : SGU_EVENT_COMPLETE;
    out->first_frame = op->event_cursor;
    out->total_frames = op->total_frames;
    out->count = min_t(u32, SGU_EVENT_BATCH, op->total_frames - op->event_cursor);
    for (i = 0; i < out->count; ++i)
        out->frames[i] = sgu_frame_at(ctx, op, op->event_cursor + i)->result;
    spin_unlock_irqrestore(&ctx->lock, flags);
    if (copy_to_user(user, out, sizeof(*out))) {
        ret = -EFAULT; /* do not consume the batch on failed copy */
        goto free;
    }
    spin_lock_irqsave(&ctx->lock, flags);
    op->event_cursor += out->count;
    if (op->event_cursor == op->total_frames) {
        op->event_pending = false;
        op->phase = SGU_OP_TAKEN;
    }
    spin_unlock_irqrestore(&ctx->lock, flags);
    ret = 0;
free:
    kfree(out);
    return ret;
}

static int sgu_action_execute(struct tbstream_gpu_context *ctx, unsigned int command, struct sgu_action *a)
{
    struct sgu_hw_operation *op;
    unsigned long flags;
    bool allowed;
    int ret = sgu_live(ctx, &a->fence, command == SGU_IOC_STOP || command == SGU_IOC_RELEASE || command == SGU_IOC_CANCEL);

    if (ret)
        return ret;
    if (a->reserved || a->result || a->queued_frames || a->direction > SGU_DIRECTION_RX)
        return -EINVAL;
    if (command == SGU_IOC_ARM || command == SGU_IOC_STOP || command == SGU_IOC_RELEASE) {
        if (a->handle || a->sequence || a->direction)
            return -EINVAL;
        if (command == SGU_IOC_ARM)
            return sgu_arm(ctx);
        if (command == SGU_IOC_STOP) {
            tbstream_gpu_stop(ctx->sdev);
            if (!sgu_can_release(ctx))
                return -EIO;
            a->result = SGU_RESULT_CONTROLLER_STOPPED;
            return 0;
        }
        if (!ctx->stopped || !sgu_can_release(ctx))
            return -EBUSY;
        sgu_release_windows(ctx);
        return 0;
    }
    op = &ctx->operations[a->direction];
    spin_lock_irqsave(&ctx->lock, flags);
    allowed = a->handle && a->sequence && op->request.handle == a->handle && op->request.sequence == a->sequence;
    spin_unlock_irqrestore(&ctx->lock, flags);
    if (!allowed)
        return -ESTALE;
    switch (command) {
    case SGU_IOC_SUBMIT:
        spin_lock_irqsave(&ctx->lock, flags);
        allowed = ctx->armed && !READ_ONCE(ctx->failed) && op->phase == SGU_OP_PREPARED;
        spin_unlock_irqrestore(&ctx->lock, flags);
        if (!allowed)
            return -EINVAL;
        ret = sgu_enqueue(ctx, op, false);
        if (!ret)
            a->queued_frames = op->total_frames;
        return ret;
    case SGU_IOC_CANCEL:
        spin_lock_irqsave(&ctx->lock, flags);
        WRITE_ONCE(ctx->failed, true);
        op->canceled = true;
        ctx->lifetime.cancel_requested = 1;
        spin_unlock_irqrestore(&ctx->lock, flags);
        a->result = SGU_RESULT_CANCEL_REQUESTED;
        return 0; /* no pending count changes: cancellation is not a stop */
    case SGU_IOC_RX_REPOST:
        spin_lock_irqsave(&ctx->lock, flags);
        allowed = a->direction == SGU_DIRECTION_RX && op->phase == SGU_OP_TAKEN && !op->remaining &&
                  !op->error_flags && !op->canceled && !READ_ONCE(ctx->failed);
        spin_unlock_irqrestore(&ctx->lock, flags);
        if (!allowed)
            return -EINVAL;
        ret = sgu_enqueue(ctx, op, true);
        if (!ret) {
            a->result = SGU_RESULT_RX_REPOSTED;
            a->queued_frames = op->total_frames;
        }
        return ret;
    case SGU_IOC_RETIRE:
        spin_lock_irqsave(&ctx->lock, flags);
        if (op->retired) {
            spin_unlock_irqrestore(&ctx->lock, flags);
            return -ESTALE;
        }
        if (a->direction == SGU_DIRECTION_RX) {
            if (op->phase != SGU_OP_STANDBY || READ_ONCE(ctx->failed)) {
                spin_unlock_irqrestore(&ctx->lock, flags);
                return -EINVAL;
            }
            op->retired = true; /* hardware remains owner of standby frames */
            spin_unlock_irqrestore(&ctx->lock, flags);
            return 0;
        }
        allowed = op->phase == SGU_OP_TAKEN && !op->remaining;
        spin_unlock_irqrestore(&ctx->lock, flags);
        if (!allowed)
            return -EBUSY;
        sgu_operation_clear(ctx, op);
        return 0;
    default:
        return -ENOTTY;
    }
}

static long tbstream_gpu_ioctl(struct file *file, unsigned int command, unsigned long argument)
{
    struct tbstream_dev *sdev = to_tbstream_dev(file->private_data);
    struct tbstream_gpu_context *ctx;
    void __user *user = (void __user *)argument;
    int ret;

    if (mutex_lock_interruptible(&sdev->lock))
        return -ERESTARTSYS;
    ctx = sdev->gpu_context;
    if (!ctx) {
        ret = -ENXIO;
        goto unlock;
    }
    if (command == SGU_IOC_QUERY) {
        struct sgu_query query;

        if (copy_from_user(&query, user, sizeof(query))) {
            ret = -EFAULT;
            goto unlock;
        }
        if (query.fence.abi_major != SGU_ABI_MAJOR || query.fence.abi_minor != SGU_ABI_MINOR ||
            memchr_inv((u8 *)&query + 8, 0, sizeof(query) - 8)) {
            ret = -EINVAL;
            goto unlock;
        }
        memset(&query, 0, sizeof(query));
        query.fence = ctx->identity;
        query.state = tbstream_dev_valid(sdev) ? SGU_STATE_DISCONNECTED : READ_ONCE(ctx->failed) ? SGU_STATE_FAILED :
                      ctx->stopped ? SGU_STATE_STOPPED : ctx->armed ? SGU_STATE_ARMED : ctx->imported ? SGU_STATE_IMPORTED : SGU_STATE_OPEN;
        query.frame_bytes = SGU_FRAME_BYTES;
        query.ring_frames = sdev->ring_size;
        query.maximum_sg_entries = SGU_MAX_SG_ENTRIES;
        query.maximum_window_bytes = (u64)sdev->ring_size * SGU_FRAME_BYTES;
        query.maximum_imported_bytes = SGU_MAX_IMPORTED_BYTES;
        query.maximum_event_batch = SGU_EVENT_BATCH;
        ret = copy_to_user(user, &query, sizeof(query)) ? -EFAULT : 0;
    } else if (command == SGU_IOC_IMPORT) {
        struct sgu_import request;

        if (copy_from_user(&request, user, sizeof(request))) {
            ret = -EFAULT;
            goto unlock;
        }
        if (tbstream_dev_valid(sdev)) {
            ret = -ENXIO;
            goto unlock;
        }
        ret = sgu_import_windows(ctx, &request);
        if (!ret && copy_to_user(user, &request, sizeof(request))) {
            WRITE_ONCE(ctx->failed, true); /* retained mapping/generation despite lost reply */
            ret = -EFAULT;
        }
    } else if (command == SGU_IOC_PREPARE) {
        struct sgu_prepare request;

        ret = copy_from_user(&request, user, sizeof(request)) ? -EFAULT : sgu_prepare_operation(ctx, &request);
    } else if (command == SGU_IOC_EVENT) {
        ret = sgu_event_ioctl(ctx, user);
    } else if (command == SGU_IOC_ARM || command == SGU_IOC_SUBMIT || command == SGU_IOC_CANCEL ||
               command == SGU_IOC_RX_REPOST || command == SGU_IOC_RETIRE || command == SGU_IOC_STOP || command == SGU_IOC_RELEASE) {
        struct sgu_action action;

        if (copy_from_user(&action, user, sizeof(action))) {
            ret = -EFAULT;
            goto unlock;
        }
        ret = sgu_action_execute(ctx, command, &action);
        if (!ret && copy_to_user(user, &action, sizeof(action))) {
            WRITE_ONCE(ctx->failed, true); /* never treat a lost side-effect reply as rollback */
            ret = -EFAULT;
        }
    } else {
        ret = -ENOTTY;
    }
unlock:
    mutex_unlock(&sdev->lock);
    return ret;
}

static __poll_t tbstream_gpu_poll(struct file *file, poll_table *wait)
{
    struct tbstream_dev *sdev = to_tbstream_dev(file->private_data);
    struct tbstream_gpu_context *ctx = READ_ONCE(sdev->gpu_context);
    unsigned long flags;
    __poll_t mask = 0;

    poll_wait(file, &sdev->wait, wait);
    if (!ctx)
        return EPOLLERR | EPOLLHUP;
    spin_lock_irqsave(&ctx->lock, flags);
    if (ctx->operations[0].event_pending || ctx->operations[1].event_pending)
        mask |= EPOLLIN | EPOLLRDNORM;
    if (READ_ONCE(ctx->failed) || READ_ONCE(sdev->stream) != ctx->stream_owner || READ_ONCE(sdev->removed))
        mask |= EPOLLERR | EPOLLHUP;
    spin_unlock_irqrestore(&ctx->lock, flags);
    return mask;
}
