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


def check(result, expected, refused=False):
    needle = f"STRIX_MMX_PF_ORDER.effective={expected}"
    if refused:
        ok = result.returncode == 2 and "STRIX_MMX_PF_ORDER=adaptive" in result.stdout
    else:
        ok = result.returncode == 0 and needle in result.stdout.splitlines()
    if not ok:
        raise SystemExit(f"{result.args}: expected {'refusal' if refused else expected}; exit {result.returncode}\n{result.stdout}")


def main(rank, serve):
    rank = [rank]
    serve = [serve]
    rank_args = ["MODEL_DIR", "0", "usb4-auto", "OUT_DIR"]
    serve_args = ["MODEL_DIR", "--rank", "0", "--transport", "usb4-auto"]
    check(run(rank, rank_args), "invariant")
    check(run(serve, serve_args), "invariant")
    for options, env in [(["--baseline"], {}), ([], {"STRIX_MM_BASELINE": "1"})]:
        check(run(rank, rank_args + options, **env), "adaptive")
        check(run(serve, serve_args + options, **env), "adaptive", refused=True)
        env["STRIX_MMX_PF_ORDER"] = "invariant"
        check(run(rank, rank_args + options, **env), "invariant")
        check(run(serve, serve_args + options, **env), "invariant")
    check(run(rank, rank_args, STRIX_MMX_PF_ORDER="adaptive"), "adaptive")
    check(run(serve, serve_args, STRIX_MMX_PF_ORDER="adaptive"), "adaptive", refused=True)
    print("rank/serve startup: default, baseline flag/environment and explicit orders passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
