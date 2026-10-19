#pragma once
#include "strix/linux_direct_gpu_stream_uapi.h"
#ifdef __cplusplus
extern "C" {
#endif
void* queue_fixture_create(unsigned frames);
void queue_fixture_reverse_endpoints(void*);
void queue_fixture_destroy(void*);
int queue_fixture_ioctl(void*, unsigned long, void*);
void queue_fixture_pool(void*, struct sgu_buffer*, struct sgu_buffer*);
void queue_fixture_complete(void*, unsigned direction, unsigned slot);
void queue_fixture_complete_pending(void*);
unsigned queue_fixture_pending(void*);
void queue_fixture_dependency(void*, unsigned direction, int status);
void queue_fixture_inject_stop_failure(int enabled);
void queue_fixture_identity_change(void*);
int queue_fixture_fd(void *value);
void queue_fixture_inject_reset_failure(int code);
/* Next RX kernel-header terminal delivered with a wrong sequence (peer mismatch). */
void queue_fixture_corrupt_terminal(int enabled);
#ifdef __cplusplus
}
#endif
