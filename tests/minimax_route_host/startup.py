"""Check rank and serve effective-config startup in isolated processes."""
import os
import subprocess
import sys


def run(command, args, **settings):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRIX_")}
    env.update(settings)
    return subprocess.run([*command, *args, "--print-effective-config"], env=env,
                          text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=15)


def check(result, expected, refused=False, tail="0"):
    needle = f"STRIX_MMX_PF_ORDER.effective={expected}"
    if refused:
        ok = result.returncode == 2 and expected in result.stdout
    else:
        lines = result.stdout.splitlines()
        ok = result.returncode == 0 and needle in lines and f"STRIX_MM_PF_DENSE_TAIL_FIXED={tail}" in lines
    if not ok:
        raise SystemExit(f"{result.args}: expected {'refusal' if refused else expected}; exit {result.returncode}\n{result.stdout}")


def main(rank, serve):
    rank = [rank]
    serve = [serve]
    rank_args = ["MODEL_DIR", "0", "usb4-auto", "OUT_DIR"]
    serve_args = ["MODEL_DIR", "--rank", "0", "--transport", "usb4-auto"]
    check(run(rank, rank_args), "invariant")
    check(run(serve, serve_args), "invariant")
    tail_needs_order = "STRIX_MM_PF_DENSE_TAIL_FIXED=1 needs STRIX_MMX_PF_ORDER=invariant"
    for options, env in [(["--baseline"], {}), ([], {"STRIX_MM_BASELINE": "1"})]:
        # The historical baseline runs on both entry points (adaptive order), as on the landing base.
        check(run(rank, rank_args + options, **env), "adaptive")
        check(run(serve, serve_args + options, **env), "adaptive")
        check(run(rank, rank_args + options, STRIX_MM_PF_DENSE_TAIL_FIXED="1", **env), tail_needs_order, refused=True)
        check(run(serve, serve_args + options, STRIX_MM_PF_DENSE_TAIL_FIXED="1", **env), tail_needs_order, refused=True)
        env["STRIX_MMX_PF_ORDER"] = "invariant"
        check(run(rank, rank_args + options, **env), "invariant")
        check(run(serve, serve_args + options, **env), "invariant")
    for command, args in [(rank, rank_args), (serve, serve_args)]:
        check(run(command, args, STRIX_MMX_PF_ORDER="adaptive"), "adaptive")
        check(run(command, args, STRIX_MM_PF_DENSE_TAIL_FIXED="1"), "invariant", tail="1")
        check(run(command, args, STRIX_MM_PF_DENSE_TAIL_FIXED="1", STRIX_MMX_PF_ORDER="adaptive"), tail_needs_order,
              refused=True)
        check(run(command, args, STRIX_MM_PF_DENSE_TAIL_FIXED="1", STRIX_MM_PREFILL_IDENTITY="fp32"),
              "STRIX_MM_PF_DENSE_TAIL_FIXED=1 conflicts with STRIX_MM_PREFILL_IDENTITY=fp32", refused=True)
    print("rank/serve startup: default, baseline, explicit orders and dense tail conflicts passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
