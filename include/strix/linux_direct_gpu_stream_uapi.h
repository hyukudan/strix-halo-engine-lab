/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/* Project-owned experimental UAPI. Not an upstream Linux ioctl contract. */
#ifndef STRIX_LINUX_DIRECT_GPU_STREAM_UAPI_H
#define STRIX_LINUX_DIRECT_GPU_STREAM_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define SGU_ABI_MAJOR 1U
#define SGU_ABI_MINOR 0U
#define SGU_ABI_COOKIE 0x5354475055303031ULL
#define SGU_FRAME_BYTES 4096U
#define SGU_EVENT_BATCH 64U
#define SGU_MAX_FRAMES 4096U
#define SGU_MAX_SG_ENTRIES 65536U
#define SGU_MAX_IMPORTED_BYTES (128ULL * 1024ULL * 1024ULL)

#define SGU_DIRECTION_TX 0U
#define SGU_DIRECTION_RX 1U
#define SGU_STATE_OPEN 0U
#define SGU_STATE_IMPORTED 1U
#define SGU_STATE_ARMED 2U
#define SGU_STATE_STOPPED 3U
#define SGU_STATE_FAILED 4U
#define SGU_STATE_DISCONNECTED 5U

#define SGU_EVENT_COMPLETE 0U
#define SGU_EVENT_CANCELED 1U
#define SGU_EVENT_FAILED 2U
#define SGU_FRAME_ERROR_CRC 1U
#define SGU_FRAME_ERROR_OVERRUN 2U
#define SGU_FRAME_ERROR_LENGTH 4U
#define SGU_FRAME_ERROR_PROTOCOL 8U
#define SGU_FRAME_ERROR_CANCELED 16U
#define SGU_FRAME_ERROR_STANDBY_ARRIVAL 32U

#define SGU_RESULT_NONE 0U
#define SGU_RESULT_CANCEL_REQUESTED 1U
#define SGU_RESULT_CONTROLLER_STOPPED 2U
#define SGU_RESULT_RX_REPOSTED 3U

/* Observed endpoint identity; import generation zero means no active import. */
struct sgu_fence {
    __u32 abi_major;
    __u32 abi_minor;
    __aligned_u64 abi_cookie;
    __u8 device_instance[16];
    __u8 local_uuid[16];
    __u8 peer_uuid[16];
    __aligned_u64 route;
    __u32 in_hopid;
    __u32 out_hopid;
    __aligned_u64 connection_epoch;
    __aligned_u64 pool_binding_identity;
    __aligned_u64 import_generation;
};

/* QUERY input: ABI major/minor; every other byte zero. No qualification bits. */
struct sgu_query {
    struct sgu_fence fence;
    __u32 state;
    __u32 frame_bytes;
    __u32 ring_frames;
    __u32 maximum_sg_entries;
    __aligned_u64 maximum_window_bytes;
    __aligned_u64 maximum_imported_bytes;
    __u32 maximum_event_batch;
    __u32 reserved;
};

struct sgu_buffer {
    __s32 fd;
    __u32 reserved;
    __aligned_u64 full_bytes;
    __aligned_u64 window_offset;
    __aligned_u64 window_bytes;
    __aligned_u64 allocation_generation;
    __aligned_u64 fd_device;
    __aligned_u64 fd_inode;
};

/* IMPORT: fence has generation zero and positive peer-agreed epoch/pool ID.
 * Kernel assigns import_generation on success. Full extents/FD identities are
 * exact; TX/RX must name distinct backing reservation objects. */
struct sgu_import {
    struct sgu_fence fence;
    struct sgu_buffer tx;
    struct sgu_buffer rx;
    __aligned_u64 reserved[2];
};

/* PREPARE reserves one C1 operation per direction. Its start offset is frame-aligned;
 * the final payload frame may be partial and has an exact expected wire size.
 * contiguous and cannot overlap the dedicated last-frame terminal region.
 * terminal_bytes is wire metadata size, 1..4096; the kernel never parses it. */
struct sgu_prepare {
    struct sgu_fence fence;
    __aligned_u64 handle;
    __aligned_u64 sequence;
    __aligned_u64 pool_offset;
    __aligned_u64 payload_bytes;
    __u32 terminal_bytes;
    __u32 direction;
    __aligned_u64 reserved;
};

/* ARM/STOP/RELEASE use zero handle/sequence/direction. Other actions identify
 * the exact prepared operation. result/queued_frames/reserved are zero input.
 * CANCEL acknowledges a request; only STOP proves controller quiescence.
 * RX_REPOST physically enqueues the slots; RETIRE keeps standby DMA retained.
 * The next RX PREPARE drains/rearms standby slots before changing geometry. */
struct sgu_action {
    struct sgu_fence fence;
    __aligned_u64 handle;
    __aligned_u64 sequence;
    __u32 direction;
    __u32 result;
    __u32 queued_frames;
    __u32 reserved;
};

struct sgu_frame_result {
    __aligned_u64 pool_offset;
    __u32 capacity;
    __u32 expected_bytes;
    __u32 actual_bytes;
    __u32 error_flags;
    __u32 physical_slot;
    __u32 reserved;
};

/* EVENT input is only fence; all remaining bytes zero. Batches are emitted
 * after every controller frame completes/cancels. first_frame/count cover
 * payload frames followed by the dedicated terminal frame. No GPU, commit
 * validation, remote ACK or consumer-credit proof is encoded here. */
struct sgu_event {
    struct sgu_fence fence;
    __aligned_u64 handle;
    __aligned_u64 sequence;
    __aligned_u64 payload_bytes;
    __aligned_u64 monotonic_ns;
    __u32 direction;
    __u32 status;
    __u32 first_frame;
    __u32 count;
    __u32 total_frames;
    __u32 reserved;
    struct sgu_frame_result frames[SGU_EVENT_BATCH];
};

/* Experimental numbers local to this opt-in device mode, not allocated stock
 * USB4STREAM ioctls. Exact command/size is checked; unknown commands fail. */
#define SGU_IOC_QUERY _IOWR(0xb7, 0x40, struct sgu_query)
#define SGU_IOC_IMPORT _IOWR(0xb7, 0x41, struct sgu_import)
#define SGU_IOC_ARM _IOWR(0xb7, 0x42, struct sgu_action)
#define SGU_IOC_PREPARE _IOW(0xb7, 0x43, struct sgu_prepare)
#define SGU_IOC_SUBMIT _IOWR(0xb7, 0x44, struct sgu_action)
#define SGU_IOC_EVENT _IOWR(0xb7, 0x45, struct sgu_event)
#define SGU_IOC_CANCEL _IOWR(0xb7, 0x46, struct sgu_action)
#define SGU_IOC_RX_REPOST _IOWR(0xb7, 0x47, struct sgu_action)
#define SGU_IOC_RETIRE _IOWR(0xb7, 0x48, struct sgu_action)
#define SGU_IOC_STOP _IOWR(0xb7, 0x49, struct sgu_action)
#define SGU_IOC_RELEASE _IOWR(0xb7, 0x4a, struct sgu_action)

#endif
