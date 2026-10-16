/* SPDX-License-Identifier: GPL-2.0-only */
/* Included after the complete materialized kernel implementation. API/DMA
 * shims are not a kernel or cache emulator. No function body is rewritten. */
#define main inherited_kernel_suite_main
#include "sgu_queue_cases.h"
#undef main

void* queue_fixture_create(unsigned frames) {
    struct fixture *f=calloc(1,sizeof(*f));assert(f);
    fixture_open(f,2,frames,false);
    memset(f->sdev.gpu_context->identity.local_uuid,0x11,16);
    memset(f->sdev.gpu_context->identity.peer_uuid,0x22,16);
    f->file.f_flags=O_NONBLOCK;
    f->file.f_op=&tbstream_dev_fops;
    f->sdev.gpu_direct=true; f->sdev.users=1;
    for(int fd=80;fd<240;++fd) if(!fd_table[fd]) { fd_table[fd]=&f->file; break; }
    return f;
}
void queue_fixture_reverse_endpoints(void *value) {
    struct fixture *f=value;
    memset(f->sdev.gpu_context->identity.local_uuid,0x22,16);
    memset(f->sdev.gpu_context->identity.peer_uuid,0x11,16);
    f->sdev.gpu_context->identity.device_instance[0]^=0x80;
}
void queue_fixture_destroy(void *value) {
    struct fixture *f=value;
    for(int fd=80;fd<240;++fd) if(fd_table[fd]==&f->file)fd_table[fd]=NULL;
    fixture_close(f);free(f);
    assert(atomic_load(&module_refs)==0);
}
int queue_fixture_ioctl(void *value,unsigned long command,void *argument) {
    struct fixture *f=value;
    return (int)tbstream_gpu_ioctl(&f->file,(unsigned)command,(unsigned long)argument);
}
void queue_fixture_pool(void *value,struct sgu_buffer *tx,struct sgu_buffer *rx) {
    const struct sgu_import p=import_request(value,1);*tx=p.tx;*rx=p.rx;
}
void queue_fixture_complete(void *value,unsigned direction,unsigned slot) {
    struct fixture *f=value;struct tbstream_gpu_context *ctx=f->sdev.gpu_context;
    assert(direction<=1&&slot<ctx->queue_depth);
    struct sgu_hw_operation *op=&ctx->operations[direction][slot];
    struct tb_ring *ring=direction?f->sdev.rx_ring.ring:f->sdev.tx_ring.ring;
    assert(op->phase==SGU_OP_PENDING);
    for(unsigned n=0;n<op->total_frames;++n){
        struct sgu_hw_frame *sf=sgu_frame_at(ctx,op,n);struct ring_frame *rf=&sf->frame;
        rf->size=sf->result.expected_bytes==4096?0:sf->result.expected_bytes;
        rf->sof=sf->result.expected_bytes<=256?0:1;rf->eof=2;
        list_del_init(&rf->list);rf->callback(ring,rf,false);
    }
}
unsigned queue_fixture_pending(void *value) {
    struct fixture *f=value;return f->sdev.gpu_context->lifetime.pending_callbacks;
}
void queue_fixture_dependency(void *value,unsigned direction,int status) {
    struct fixture *f=value;struct dma_fence *pending=foreign_fence(&f->resv[direction]);
    if(status){dma_fence_set_error(pending,status<0?status:0);dma_fence_signal(pending);}
    dma_fence_put(pending);
}
void queue_fixture_inject_stop_failure(int enabled){stop_fails=enabled;}
void queue_fixture_identity_change(void *value) {
    struct fixture *f=value;f->sdev.gpu_context->identity.peer_uuid[0]^=1;
}
void queue_fixture_complete_pending(void *value) {
    struct fixture *f=value;struct tbstream_gpu_context *ctx=f->sdev.gpu_context;
    for(unsigned d=0;d<2;++d)for(unsigned s=0;s<ctx->queue_depth;++s)
        if(ctx->operations[d][s].phase==SGU_OP_PENDING)queue_fixture_complete(value,d,s);
}

int queue_fixture_fd(void *value) {
    struct fixture *f=value;
    for(int fd=80;fd<240;++fd)if(fd_table[fd]==&f->file)return fd;
    return -1;
}
void queue_fixture_inject_reset_failure(int code) { reset_error=code; }
