/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef STRIX_GPU_STREAM_H
#define STRIX_GPU_STREAM_H
struct tbstream_dev;
struct tbstream_gpu_context;
static int tbstream_gpu_open(struct tbstream_dev *sdev);
static void tbstream_gpu_stop(struct tbstream_dev *sdev);
static void tbstream_gpu_close(struct tbstream_dev *sdev);
static void tbstream_gpu_disconnect(struct tbstream_dev *sdev);
static bool tbstream_gpu_paths_pending(struct tbstream_dev *sdev);
static long tbstream_gpu_ioctl(struct file *file, unsigned int command, unsigned long argument);
static __poll_t tbstream_gpu_poll(struct file *file, poll_table *wait);
#endif
