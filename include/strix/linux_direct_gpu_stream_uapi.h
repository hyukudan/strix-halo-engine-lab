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
#define SGU_MAX_QUEUE_DEPTH 8U
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
#define SGU_FRAME_ERROR_TIMEOUT 64U

#define SGU_RESULT_NONE 0U
#define SGU_RESULT_CANCEL_REQUESTED 1U
#define SGU_RESULT_CONTROLLER_STOPPED 2U
#define SGU_RESULT_RX_REPOSTED 3U
#define SGU_RESULT_SESSION_RECOVERED 4U

#define SGU_CAP_MULTI_OP 1U
#define SGU_CAP_NONBLOCK_SUBMIT 2U
#define SGU_CAP_SYNC_FILE 4U
#define SGU_CAP_SESSION_RECOVERY 8U
#define SGU_CAP_DMA_TIMEOUT 16U
#define SGU_CAP_STATS 32U
#define SGU_CAP_IDLE_RESET 64U

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
 * terminal_bytes is wire metadata size, 1..4096. Without flags the kernel never
 * parses it. flags (formerly reserved, zero = legacy behaviour) may hold
 * SGU_PREPARE_TERMINAL_HEADER, see struct sgu_terminal_header below. */
struct sgu_prepare {
    struct sgu_fence fence;
    __aligned_u64 handle;
    __aligned_u64 sequence;
    __aligned_u64 pool_offset;
    __aligned_u64 payload_bytes;
    __u32 terminal_bytes;
    __u32 direction;
    union {
        __aligned_u64 reserved;
        __aligned_u64 flags;
    };
};

/* ARM/STOP/RELEASE use zero handle/sequence/direction. Other actions identify
 * the exact prepared operation. result/queued_frames/reserved are zero input.
 * CANCEL acknowledges a request; only STOP proves controller quiescence.
 * RX_REPOST physically enqueues the slots; RETIRE keeps standby DMA retained.
 * The next RX PREPARE must preserve pool_offset and payload frame count.
 * Changed slot geometry requires STOP/RELEASE and a coordinated fresh import. */
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

/* Optional extensions: old command sizes and the default depth=1 are unchanged.
 * QUEUE input: exact QUERY identity, depth=0 to query or 1..maximum_depth to
 * configure before IMPORT; maximum_depth/features/reserved must be zero.
 * Depth > 1 reserves the final `depth` frames as per-slot terminal storage.
 * DMA mappings requiring CPU cache sync cannot use the multi-operation mode.
 * Features describe software interfaces, NOT hardware qualification. */
struct sgu_queue_config {
    struct sgu_fence fence;
    __u32 depth;
    __u32 maximum_depth;
    __u32 features;
    __u32 reserved;
};

/* Slot must be < configured depth. Payload reserves whole frame slots and
 * cannot overlap another unretired operation or any terminal slot.
 * Terminal offset = (ring_frames - depth + slot) * SGU_FRAME_BYTES.
 * SUBMIT order must have increasing sequence numbers in each direction.
 * In depth>1 mode RX_REPOST is unsupported: RETIRE after consuming EVENT,
 * then PREPARE_SLOT/SUBMIT another operation without stopping either ring.
 * Coordinate the same slot geometry and submission order with the peer. */
struct sgu_prepare_slot {
    struct sgu_prepare operation;
    __u32 slot;
    __u32 reserved;
};

/* Export a submitted operation's local controller fence (not a remote ACK).
 * Input fd=-1, reserved=0. Success installs an O_CLOEXEC sync_file descriptor.
 * An exported fence remains valid after RETIRE/RELEASE/close.
 * Never substitute one operation's fence for all accesses to the DMA-BUF:
 * other slots can still be using the allocation. */
struct sgu_sync_file {
    struct sgu_fence fence;
    __aligned_u64 handle;
    __aligned_u64 sequence;
    __u32 direction;
    __s32 fd;
    __aligned_u64 reserved;
};

/* RECOVER uses sgu_action with zero handle/sequence/direction. It stops only
 * this session, requires proven local quiescence, releases its imports and
 * clears its failure state. A fresh strictly larger peer-agreed epoch is
 * mandatory. It does NOT reset the NHI, retrain the cable, or certify the peer.
 * Success returns the unimported identity in fence and SESSION_RECOVERED.
 * Blocking SUBMIT/RX_REPOST waits for foreign fences without holding the
 * device mutex; O_NONBLOCK returns EAGAIN without publishing/enqueuing. */

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
#define SGU_IOC_QUEUE _IOWR(0xb7, 0x4b, struct sgu_queue_config)
#define SGU_IOC_EXPORT_SYNC_FILE _IOWR(0xb7, 0x4c, struct sgu_sync_file)
#define SGU_IOC_RECOVER _IOWR(0xb7, 0x4d, struct sgu_action)
#define SGU_IOC_PREPARE_SLOT _IOW(0xb7, 0x4e, struct sgu_prepare_slot)

/* STATS input: exact current QUERY identity; all following bytes zero.
 * Counters are cumulative for the open context, including recovery/reimports.
 * A completion counts a DMA batch (including legacy standby), not a peer ACK.
 * Durations include local software queueing; byte counters exclude the terminal.
 */
struct sgu_stats {
    struct sgu_fence fence;
    __aligned_u64 monotonic_ns;
    __aligned_u64 submissions[2];
    __aligned_u64 completions[2];
    __aligned_u64 errors[2];
    __aligned_u64 payload_bytes[2];
    __aligned_u64 last_dma_ns[2];
    __aligned_u64 max_dma_ns[2];
    __aligned_u64 dependency_retries;
    __aligned_u64 timeouts;
    __aligned_u64 dma_stops;
    __aligned_u64 resets;
    __aligned_u64 reset_failures;
    __u32 pending_callbacks;
    __u32 peak_callbacks;
    __u32 queue_depth;
    __u32 quarantined;
    __aligned_u64 reserved[2];
};
#define SGU_IOC_STATS _IOWR(0xb7, 0x4f, struct sgu_stats)
/* Explicit CAP_SYS_ADMIN operation. Requires STOP and zero local ownership;
 * rejects any other allocated non-control ring on this NHI, even if idle.
 * Does not stop other users, force PCI reset, or retrain the physical link.
 * On success, drops imports and requires a newer peer-agreed connection epoch.
 * Input/output use the existing sgu_action layout with zero operation fields.
 */
#define SGU_RESULT_CONTROLLER_RESET 5U
#define SGU_IOC_RESET _IOWR(0xb7, 0x50, struct sgu_action)

/* All members explicitly lend one live GPU-stream FD with its exact current
 * identity. Every session must already be stopped with its DMA paths removed.
 * Duplicate FDs/devices, mixed controllers or any unknown data ring are refused.
 * Lock contention is EAGAIN without reset; no member is stopped implicitly.
 * On success every member is invalidated and must RECOVER/IMPORT using a fresh,
 * strictly larger peer-agreed epoch. Old imports never resume automatically.
 * This is a LOCAL controller operation: coordinate every peer independently.
 * It is forbidden to use the affected USB4NET link as the only control channel.
 * result/reserved and all unused member bytes must be zero on input. */
#define SGU_MAX_RESET_MEMBERS 16U
#define SGU_CAP_GROUP_RESET 128U
#define SGU_RESET_GROUP_NETWORK 1U
#define SGU_RESULT_GROUP_RESET 6U
struct sgu_reset_member {
    __s32 fd;
    __u32 reserved;
    struct sgu_fence fence;
};
struct sgu_reset_group {
    __u32 count;
    __u32 result;
    /* NETWORK explicitly authorizes local USB4NET down/up under RTNL. This
     * can drop packets: use an independently authenticated control channel. */
    __u32 flags;
    __u32 reserved;
    struct sgu_reset_member members[SGU_MAX_RESET_MEMBERS];
};
#define SGU_IOC_RESET_GROUP _IOWR(0xb7, 0x51, struct sgu_reset_group)

/* Kernel-validated terminal frame (optional, additive extension).
 *
 * Without it, RX frames land in descriptor posting order and the kernel only
 * checks each frame's length: if the peers disagree on payload_bytes or on
 * SUBMIT order and the sizes happen to match, data shifts silently.
 *
 * With SGU_PREPARE_TERMINAL_HEADER in sgu_prepare.flags (terminal_bytes must
 * then be SGU_TERMINAL_HEADER_BYTES), the terminal frame no longer uses the
 * DMA-BUF terminal slot. On TX the kernel itself sends this header, built from
 * the operation. On RX the terminal lands in kernel memory and is checked
 * against the local operation: length, magic, version, the import's
 * connection_epoch, sequence, payload_bytes, payload frame count and slot. A
 * mismatch sets SGU_FRAME_ERROR_TERMINAL on that frame, EVENT reports
 * SGU_EVENT_FAILED and the session fails (RECOVER with a fresh epoch).
 *
 * Contract: both peers set the flag on matching operations, and an RX
 * operation uses the same sequence number as the peer's TX operation it
 * receives. The kernel validates framing, not payload integrity (link CRC
 * only); an end-to-end payload checksum remains the application's job.
 *
 * Discovery: SGU_IOC_QUEUE reports SGU_CAP_TERMINAL_HEADER. Kernels without
 * it reject a non-zero flags field with EINVAL. ABI versioning: abi_minor is
 * part of every identity and QUERY requires an exact match, so additive
 * extensions are advertised as capability bits and abi_minor stays 0; it will
 * only change together with an exact-match-free QUERY. */
#define SGU_CAP_TERMINAL_HEADER 256U
#define SGU_PREPARE_TERMINAL_HEADER 1ULL
#define SGU_FRAME_ERROR_TERMINAL 128U
#define SGU_TERMINAL_MAGIC 0x54554753U /* "SGUT" as little-endian bytes */
#define SGU_TERMINAL_VERSION 1U
#define SGU_TERMINAL_HEADER_BYTES 64U
/* Little-endian on the wire. */
struct sgu_terminal_header {
    __le32 magic;
    __le16 version;
    __le16 header_bytes;
    __le64 connection_epoch;
    __le64 sequence;
    __le64 payload_bytes;
    __le32 payload_frames;
    __le32 slot;
    __le64 reserved[3];
};

#endif
