"""Reject SDK dependencies in neutral resource, transport and kernel-policy headers."""

import re
import sys
from pathlib import Path


SHARED_ROOTS = ("include/strix", "src/core", "src/moe_core", "src/attn_core", "src/xport")
HEADER_SUFFIXES = {".h", ".hpp", ".cuh", ".hh", ".hxx", ".inl"}
NATIVE_PREFIXES = ("include/strix/cuda/", "include/strix/hip_")
# Native bindings and kernel bodies sharing directories with neutral contracts.
NATIVE_HEADERS = {
    "include/strix/cuda_core/attention.hpp",
    "include/strix/cuda_core/attention_step_lanes.hpp",
    "include/strix/cuda_core/dense.hpp",
    "include/strix/cuda_core/dense_big.hpp",
    "include/strix/cuda_core/gdn.hpp",
    "include/strix/cuda_core/norms.hpp",
    "include/strix/cuda_core/nvfp4.hpp",
    "include/strix/cuda_core/rope.hpp",
    "include/strix/cuda_core/submit_window.hpp",
    "include/strix/cuda_core/types.hpp",
    "include/strix/cuda_logprob_partials.hpp",
    "include/strix/device/cuda_enumerator.hpp",
    "include/strix/device/hip_enumerator.hpp",
    "include/strix/device/intel_enumerator.hpp",
    "include/strix/device/runtime.hpp",
    "include/strix/device/width_graphs.hpp",
    "include/strix/dist/communicator_hip.hpp",
    "include/strix/dist/gpu_direct.hpp",
    "include/strix/dist/gpu_direct_copy.hpp",
    "include/strix/dist/host_copy.hpp",
    "include/strix/dist/staged_ready_hip.hpp",
    "include/strix/dist/staged_ready_kernels.hpp",
    "include/strix/gdn_recurrent_layout_diagnostic.hpp",
    "include/strix/glm/engine.hpp",
    "include/strix/hip/legacy_intrinsics.hpp",
    "include/strix/hip/legacy_launch.hpp",
    "include/strix/minimax/engine.hpp",
    "include/strix/moe_batched_activation_diagnostic.hpp",
    "include/strix/pair/producer_publish_hip.hpp",
    "include/strix/pinned_host.hpp",
    "include/strix/qwen35/cuda_engine.hpp",
    "include/strix/qwen35/cuda_spec.hpp",
    "src/attn_core/cache_rows.hpp",
    "src/attn_core/cache_shadow.hpp",
    "src/attn_core/chunk_decode.hpp",
    "src/attn_core/chunk_exact_combine.hpp",
    "src/attn_core/dense_decode.hpp",
    "src/attn_core/dense_decode_flash.hpp",
    "src/attn_core/dense_flash_prefill.hpp",
    "src/attn_core/eager_selected.hpp",
    "src/attn_core/eager_softmax.hpp",
    "src/attn_core/group_pool.hpp",
    "src/attn_core/group_score_scan.hpp",
    "src/attn_core/group_select_kernel.hpp",
    "src/attn_core/index_score_band.hpp",
    "src/attn_core/index_score_wmma.hpp",
    "src/attn_core/masks.hpp",
    "src/attn_core/prepared_bf16_attention.hpp",
    "src/attn_core/select.hpp",
    "src/attn_core/select_radix3.hpp",
    "src/attn_core/select_threshold.hpp",
    "src/attn_core/selected_exact_mla.hpp",
    "src/attn_core/sparse_flash_keysplit.hpp",
    "src/attn_core/sparse_flash_rowgroup.hpp",
    "src/attn_core/sparse_flash_wmma.hpp",
    "src/attn_core/wmma_types.hpp",
    "src/core/a4_tiles.hpp",
    "src/core/backend_hip.hpp",
    "src/core/dense_linear.hpp",
    "src/core/dense_linear_iu4.hpp",
    "src/core/f16_dense_iu4.hpp",
    "src/core/f16_dense_q8a8.hpp",
    "src/core/f16_dense_tall.hpp",
    "src/core/f16_dense_tall_hilo.hpp",
    "src/core/f16_tile_gemm.hpp",
    "src/core/fma_bf16_rows.hpp",
    "src/core/fma_rows.hpp",
    "src/core/gemv_rows.hpp",
    "src/core/hadamard.hpp",
    "src/core/hc_core.hpp",
    "src/core/hc_expand_rows.hpp",
    "src/core/hc_lane_chain.hpp",
    "src/core/hip_host_ready.hpp",
    "src/core/hip_screened_head.hpp",
    "src/core/hip_weight_uploader.hpp",
    "src/core/i4_shadow.hpp",
    "src/core/iu4a4_dense.hpp",
    "src/core/moe_i4_chain.hpp",
    "src/core/ordered_reduce.hpp",
    "src/core/pointwise.hpp",
    "src/core/q8_f16_staging.hpp",
    "src/core/residual_mix.hpp",
    "src/core/resident_greedy.hpp",  # Native HIP/CUDA buffers, pinned replies and selector binding.
    "src/core/short_convolution.hpp",
    "src/core/tp_join.hpp",
    "src/core/trellis_codec.hpp",
    "src/core/token_sampling_device_body.inl",
    "src/core/w4a16.hpp",
    "src/core/weight_decode.hpp",
    "src/core/wmma_tiled.hpp",
    "src/moe_core/decode.hpp",
    "src/moe_core/device_grouped_plan.hpp",
    "src/moe_core/device_plan.hpp",
    "src/moe_core/exl3_trellis.hpp",
    "src/moe_core/expert_activation.hpp",
    "src/moe_core/grouped_a8_f16.hpp",
    "src/moe_core/grouped_dispatch.hpp",
    "src/moe_core/grouped_integer_decode.hpp",
    "src/moe_core/grouped_iu4a4.hpp",  # Native launch binding; grouped_iu4a4_contract.hpp and plan remain checked.
    "src/moe_core/grouped_microscaled.hpp",
    "src/moe_core/grouped_rows.hpp",
    "src/moe_core/microscaled_r1.hpp",
    "src/moe_core/route.hpp",
    "src/moe_core/route_prune.hpp",
    "src/moe_core/route_tap_hip.hpp",
    "src/moe_core/route_project_kernel.hpp",  # HIP-only route projection body; the neutral route_project.hpp remains SDK-free.
    "src/moe_core/route_wave_kernel.hpp",
    "src/xport/bf16_wire_hip.hpp",
    "src/xport/channel_device.hpp",
    "src/xport/channel_hip.hpp",
    "src/xport/device_crc32c.hpp",
    "src/xport/exchange_format_hip.hpp",
    "src/xport/hip_channel_runtime.hpp",
    "src/xport/hip_pipe_staging.hpp",
    "src/xport/kvshard_channel_hip.hpp",
    "src/xport/pair_device_link.hpp",
    "src/xport/tp_gather_hip.hpp",
    "src/xport/tp_join_bf16_hip.hpp",
    "src/xport/tp_join_exec_hip.hpp",
    "src/xport/tp_join_map_hip.hpp",
}
COMMENT_OR_LITERAL = re.compile(
    r"""(?P<comment>//[^\r\n]*|/\*[\s\S]*?\*/)|"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'"""
)
SDK_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*[<"](?:hip/|cuda[/_.]|driver_types\.h|vector_types\.h|'
    r'device_launch_parameters\.h|(?:CL/)?sycl[/_.]|hipblas|rocblas|cublas|cufft|'
    r'nvrtc|hiprtc|level_zero/)[^>"\r\n]*[>"]',
    re.MULTILINE,
)
SDK_TYPE = re.compile(
    r"\b(?:hip[A-Z]\w*_t|hipDeviceptr_t|hipMemcpyKind|hipStreamCaptureMode|"
    r"hipMemAllocation\w*|hipMemLocation|cuda[A-Z]\w*_t|cudaDeviceProp|cudaMemcpyKind|"
    r"CU(?:device|stream|event|context|result|module|function|deviceptr|graph\w*|"
    r"uuid|linkState|mem\w+)|dim3|__half2?|__hip_bfloat16|__nv_bfloat162?)\b|\bsycl[ \t]*::"
)


def blank(match):
    return re.sub(r"[^\r\n]", " ", match[0])


def main(root):
    violations = []
    for directory in SHARED_ROOTS:
        tree = root / directory
        if not tree.is_dir():
            raise ValueError(f"Missing header directory {tree}")
        for path in sorted(tree.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.suffix not in HEADER_SUFFIXES or relative.startswith(NATIVE_PREFIXES) or relative in NATIVE_HEADERS:
                continue
            source = path.read_text(encoding="utf-8-sig")
            without_comments = COMMENT_OR_LITERAL.sub(
                lambda match: blank(match) if match["comment"] else match[0], source
            )
            without_literals = COMMENT_OR_LITERAL.sub(blank, source)
            for pattern, code in ((SDK_INCLUDE, without_comments), (SDK_TYPE, without_literals)):
                for match in pattern.finditer(code):
                    line = source.count("\n", 0, match.start()) + 1
                    violations.append(f"{relative}:{line}: {match[0].strip()}")
    if violations:
        print("SDK declarations in shared headers:", *violations, sep="\n", file=sys.stderr)
    return bool(violations)


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
