"""Compile the real host admission and launcher with recording kernel submissions."""

from pathlib import Path
import re
import sys


core = Path(__file__).resolve().parents[2] / "src/core"
namespace = "namespace strix::core::controller_rms_rows {"
namespace_end = "}  // namespace strix::core::controller_rms_rows"

admission = (core / "controller_rms_rows.hpp").read_text(encoding="utf-8")
start = admission.index(namespace)
device = admission.index("namespace detail {", start)
host = admission.index("inline bool admits(", device)
end = admission.index(namespace_end, host)

launch = (core / "fma_bf16_rows.hpp").read_text(encoding="utf-8")
start_launch = launch.index("inline hipError_t enqueue(const Args&", launch.index(namespace))
end_launch = launch.index(namespace_end, start_launch)
launch = re.sub(r"controller_rms_kernel<(\dU)><<<grid, kThreads, 0, stream>>>\(a\)",
                r"submit<\1, kThreads>(grid, stream, a)", launch[start_launch:end_launch])

Path(sys.argv[1]).write_text(admission[start:device] + admission[host:end] + launch + namespace_end + "\n",
                             encoding="utf-8")
