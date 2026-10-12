#include "strix/linux_direct_gpu_stream_uapi.h"
#include <stddef.h>

// These constants define ABI v1, independently of the compiler's data model.
// Compile this same file for both LP64 and ILP32; it needs no C++ runtime.
static_assert(SGU_ABI_MAJOR == 1U && SGU_ABI_MINOR == 0U);
static_assert(sizeof(sgu_fence) == 104 && alignof(sgu_fence) == 8);
static_assert(offsetof(sgu_fence, abi_cookie) == 8);
static_assert(offsetof(sgu_fence, device_instance) == 16);
static_assert(offsetof(sgu_fence, local_uuid) == 32);
static_assert(offsetof(sgu_fence, peer_uuid) == 48);
static_assert(offsetof(sgu_fence, route) == 64);
static_assert(offsetof(sgu_fence, connection_epoch) == 80);
static_assert(offsetof(sgu_fence, pool_binding_identity) == 88);
static_assert(offsetof(sgu_fence, import_generation) == 96);
static_assert(sizeof(sgu_query) == 144 && alignof(sgu_query) == 8);
static_assert(offsetof(sgu_query, maximum_window_bytes) == 120);
static_assert(sizeof(sgu_buffer) == 56 && alignof(sgu_buffer) == 8);
static_assert(offsetof(sgu_buffer, fd) == 0);
static_assert(offsetof(sgu_buffer, full_bytes) == 8);
static_assert(offsetof(sgu_buffer, fd_inode) == 48);
static_assert(sizeof(sgu_import) == 232 && alignof(sgu_import) == 8);
static_assert(offsetof(sgu_import, tx) == 104);
static_assert(offsetof(sgu_import, rx) == 160);
static_assert(offsetof(sgu_import, reserved) == 216);
static_assert(sizeof(sgu_prepare) == 152 && alignof(sgu_prepare) == 8);
static_assert(offsetof(sgu_prepare, handle) == 104);
static_assert(offsetof(sgu_prepare, pool_offset) == 120);
static_assert(offsetof(sgu_prepare, terminal_bytes) == 136);
static_assert(sizeof(sgu_action) == 136 && alignof(sgu_action) == 8);
static_assert(offsetof(sgu_action, direction) == 120);
static_assert(sizeof(sgu_frame_result) == 32 && alignof(sgu_frame_result) == 8);
static_assert(offsetof(sgu_frame_result, actual_bytes) == 16);
static_assert(sizeof(sgu_event) == 2208 && alignof(sgu_event) == 8);
static_assert(offsetof(sgu_event, handle) == 104);
static_assert(offsetof(sgu_event, direction) == 136);
static_assert(offsetof(sgu_event, frames) == 160);
static_assert(SGU_EVENT_BATCH == 64 && SGU_FRAME_BYTES == 4096);

#define CHECK_IOCTL(command, number, bytes, directions) \
    static_assert(_IOC_TYPE(command) == 0xb7); \
    static_assert(_IOC_NR(command) == number); \
    static_assert(_IOC_SIZE(command) == bytes); \
    static_assert(_IOC_DIR(command) == (directions))
CHECK_IOCTL(SGU_IOC_QUERY, 0x40, 144, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_IMPORT, 0x41, 232, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_ARM, 0x42, 136, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_PREPARE, 0x43, 152, _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_SUBMIT, 0x44, 136, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_EVENT, 0x45, 2208, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_CANCEL, 0x46, 136, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_RX_REPOST, 0x47, 136, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_RETIRE, 0x48, 136, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_STOP, 0x49, 136, _IOC_READ | _IOC_WRITE);
CHECK_IOCTL(SGU_IOC_RELEASE, 0x4a, 136, _IOC_READ | _IOC_WRITE);
#undef CHECK_IOCTL

int main() { return 0; }
