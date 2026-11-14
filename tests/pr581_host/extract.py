"""Compile production host control flow with deterministic OS/transport doubles.

No algorithm is reimplemented: extract the reader constructor/read methods, the
shared chunk planner and the complete ready branch. Only platform APIs are shims.
An optional original revision proves the regressions fail before the fix.
"""
import pathlib
import subprocess
import sys

root, output = map(pathlib.Path, sys.argv[1:3])
baseline = sys.argv[3] if len(sys.argv) > 3 else ""
here = pathlib.Path(__file__).parent


def between(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def linux(text):
    # Select the Linux branches without redefining _WIN32 in the host standard library.
    active, stack, result = True, [], []
    for line in text.splitlines():
        if line.startswith("#if defined("):
            stack.append((active, "__linux__" in line))
            active = active and stack[-1][1]
        elif line == "#else":
            active = stack[-1][0] and not stack[-1][1]
        elif line == "#endif":
            active = stack.pop()[0]
        elif active:
            result.append(line)
    assert not stack
    return "\n".join(result)


for version in (["fixed", "original"] if baseline else ["fixed"]):
    def read(path):
        if version == "fixed":
            return (root / path).read_text(encoding="utf-8")
        return subprocess.check_output(["git", "show", f"{baseline}:{path}"], cwd=root).decode("utf-8")

    src = read("src/ds41_engram.cpp")
    header = read("include/strix/ds41_engram.hpp")
    ctor = linux(between(src, "EngramRowReader::EngramRowReader(", "EngramRowReader::~EngramRowReader()"))
    ctor = ctor.replace("::open(", "fake_open(").replace("struct stat st", "FakeStat st").replace("::fstat(", "fake_fstat(")
    batch = between(src, "bool EngramRowReader::read_batched(", "#else\nbool EngramRowReader::read_batched")
    one = between(src, "void EngramRowReader::read_one(", "#if defined(__linux__)\n// STRIX_DS41_ENGRAM_AIO")
    aio_member = between(header, "    struct AioContext" if "struct AioContext" in header else "    unsigned long aio_", "    std::uint8_t* aio_pages_")
    value = "aio_.value" if "struct AioContext" in header else "aio_"
    template = (here / "aio.cpp.in").read_text(encoding="utf-8")
    for key, val in {"CTOR": ctor, "BATCH": batch, "ONE": one, "AIO_MEMBER": aio_member, "AIO_VALUE": value}.items():
        template = template.replace(f"@{key}@", val)
    (output / f"aio-{version}.cpp").write_text(template, encoding="utf-8")

    comm = read("include/strix/dist/communicator.hpp")
    capacity = between(comm, "    [[nodiscard]] std::size_t staged_ready_capacity()", "    void start_staged_ready(")
    forward = read("src/hip_deepseek41_batch_forward.hip")
    branch = between(forward, "                const std::size_t ready_cap" if "const std::size_t ready_cap" in forward else "                const std::uint32_t ready_chunk", "                if (exchanged) {")
    serve = read("apps/deepseek41_serve_main.cpp")
    digest = between(serve, "struct StepTokenDigest {", "StepTokenDigest g_step_tokens;")
    add = "d.add(rows, tokens);" if "void add(const Rows&" in digest else "(void)rows; d.add(tokens);"
    template = (here / "ready.cpp.in").read_text(encoding="utf-8")
    for key, val in {"CAPACITY": capacity, "BRANCH": branch, "DIGEST": digest, "ADD": add}.items():
        template = template.replace(f"@{key}@", val)
    (output / f"ready-{version}.cpp").write_text(template, encoding="utf-8")
