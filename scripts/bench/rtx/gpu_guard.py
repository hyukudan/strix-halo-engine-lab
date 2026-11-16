#!/usr/bin/env python3
"""gpu_guard.py: run one GPU job under a 1 s telemetry guard that fails closed.

    python scripts/bench/rtx/gpu_guard.py --gpu-name "RTX PRO 6000" --out DIR [--latch FILE] [--timeout-s 300] -- CMD ...

What it does (read-only on the machine: it never changes driver, power, registry or clock settings):
  * resolves the GPU by NAME (and optional --pci) to its UUID and runs CMD with CUDA_VISIBLE_DEVICES=<UUID>, so a
    renumbered device can never be picked by index;
  * before the launch: refuses when the latch file exists, when the GPU is not visible, when its memory is above
    --max-idle-mib, or when a guarded Windows event happened in the last --quiet-before-s seconds;
  * while CMD runs: `nvidia-smi -i UUID --query-gpu=... -l 1` to DIR/smi.csv, a persistent System-log watcher
    (winevents.ps1: WHEA-Logger any id, nvlddmkm any id, Display 4101) to DIR/events.log, PCIe replay counter every
    --replay-every-s, child stdout/stderr to DIR/child.log (and echoed);
  * aborts on the FIRST: guarded event, telemetry silence > --stale-s (GPU gone), GPU missing from nvidia-smi, temperature
    margin to the slowdown limit (temperature.gpu.tlimit) < --tlimit-min-c or temperature >= --temp-max-c, PCIe replay
    increase, a child output line matching --fail-pattern (CUDA "unknown error", illegal address, launch failure), or the
    wall limit. Abort = kill the whole child tree, write the latch (reason + time) so no next run starts, exit non-zero;
  * after CMD: --post-s more seconds of telemetry, GPU must still be visible; a non-zero child rc is a failure.
  * DIR/guard.json + one 'GUARD ...' summary line (peak W, max C, min margin, max MiB, largest 1 s power step up/down).

Exit codes: child rc when only the child failed; 90 event, 91 GPU lost / telemetry silent, 92 temperature, 93 PCIe
replay, 94 wall limit, 95 fail pattern, 96 refused before launch (nothing ran). 0 only when everything was clean.
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, threading, time
from datetime import datetime

FIELDS = ("timestamp,uuid,pstate,temperature.gpu,temperature.gpu.tlimit,power.draw,enforced.power.limit,utilization.gpu,"
          "memory.used,memory.total,clocks.sm,clocks.mem,clocks_event_reasons.active,pcie.link.gen.current,"
          "pcie.link.width.current")
FAIL_DEFAULT = (r"failed: unknown error|unspecified launch failure|illegal memory access|illegal instruction|"
                r"No devices were found|GPU is lost|Xid|cudaErrorDevicesUnavailable|device-side assert")
WIN = os.name == "nt"


def now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def smi(args, timeout=15):
    try:
        r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception as e:  # nvidia-smi hung or missing: treat as not visible
        return 255, str(e)


def resolve(name, pci):
    rc, out = smi(["--query-gpu=index,uuid,name,pci.bus_id", "--format=csv,noheader"])
    if rc:
        return None, out.strip()
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and name in p[2] and (not pci or p[3].lower().endswith(pci.lower())):
            return p[1], line.strip()
    return None, out.strip()


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def replays(uuid):
    rc, out = smi(["-i", uuid, "-q"])  # no -d section carries the replay counter
    m = re.search(r"Replays Since Reset\s*:\s*(\d+)", out)
    return int(m.group(1)) if (rc == 0 and m) else None


class Guard:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.trip = None          # (code, reason)
        self.last_smi = time.time()
        self.last_hb = time.time()
        self.samples = []
        self.events = []
        self.child = None

    def fire(self, code, reason):
        with self.lock:
            if self.trip:
                return
            self.trip = (code, reason)
        self.log(f"TRIP {code} {reason}")
        self.kill_child()
        if self.a.latch:
            try:
                with open(self.a.latch, "a", encoding="utf-8") as f:
                    f.write(f"{now()} gpu_guard STOP code {code}: {reason} (out {self.a.out})\n")
            except OSError as e:
                self.log(f"latch write failed: {e}")

    def log(self, msg):
        line = f"{now()} {msg}"
        with open(os.path.join(self.a.out, "guard.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print("[gpu_guard] " + line, file=sys.stderr, flush=True)

    def kill_child(self):
        c = self.child
        if c is None or c.poll() is not None:
            return
        t = time.time()
        if WIN:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(c.pid)], capture_output=True)
            try:
                c.wait(timeout=2)
            except subprocess.TimeoutExpired:
                c.kill()
            self.log(f"child tree killed in {time.time() - t:.2f} s")
        else:
            try:
                os.killpg(c.pid, signal.SIGKILL)
            except OSError:
                pass

    # ---- watchers -------------------------------------------------------------------------------------------
    def smi_reader(self, proc):
        out = open(os.path.join(self.a.out, "smi.csv"), "a", encoding="utf-8")
        out.write(FIELDS + "\n")
        for line in proc.stdout:
            out.write(line)
            out.flush()
            p = [x.strip() for x in line.split(",")]
            if len(p) < 15 or not p[1].startswith("GPU-"):
                if "No devices" in line or "Unable" in line or "lost" in line:
                    self.fire(91, f"nvidia-smi: {line.strip()[:120]}")
                continue
            self.last_smi = time.time()
            s = {"t": time.time(), "temp": num(p[3]), "tlimit": num(p[4]), "w": num(p[5]), "plim": num(p[6]),
                 "util": num(p[7]), "mib": num(p[8]), "sm": num(p[10]), "gen": num(p[13]), "width": num(p[14])}
            self.samples.append(s)
            if s["tlimit"] is not None and s["tlimit"] < self.a.tlimit_min_c:
                self.fire(92, f"temperature margin {s['tlimit']} C < {self.a.tlimit_min_c} C (temp {s['temp']} C)")
            if s["temp"] is not None and s["temp"] >= self.a.temp_max_c:
                self.fire(92, f"temperature {s['temp']} C >= {self.a.temp_max_c} C")

    def event_reader(self, proc):
        out = open(os.path.join(self.a.out, "events.log"), "a", encoding="utf-8")
        for line in proc.stdout:
            if line.startswith("HB "):
                self.last_hb = time.time()
                continue
            out.write(line)
            out.flush()
            if line.startswith("EVT "):
                self.events.append(line.strip()[:300])
                self.fire(90, "windows event: " + line.strip()[4:180])

    def child_reader(self, proc, pat):
        out = open(os.path.join(self.a.out, "child.log"), "ab")
        for raw in proc.stdout:
            out.write(raw)
            out.flush()
            sys.stdout.buffer.write(raw)
            sys.stdout.flush()
            if pat.search(raw.decode("utf-8", "replace")):
                self.fire(95, "child output: " + raw.decode("utf-8", "replace").strip()[:160])

    def summary(self, child_rc, code, t0, t1):
        S = self.samples
        def mx(k):
            v = [s[k] for s in S if s[k] is not None]
            return max(v) if v else None
        def mn(k):
            v = [s[k] for s in S if s[k] is not None]
            return min(v) if v else None
        steps = [b["w"] - a["w"] for a, b in zip(S, S[1:]) if a["w"] is not None and b["w"] is not None]
        busy = [s for s in S if (s["util"] or 0) >= 90]
        d = {"status": "ok" if code == 0 else "FAIL", "exit": code, "child_rc": child_rc,
             "reason": self.trip[1] if self.trip else ("child rc %s" % child_rc if code else ""),
             "start": t0, "end": t1, "samples": len(S), "peak_w": mx("w"), "power_limit_w": mx("plim"),
             "max_temp_c": mx("temp"), "min_tlimit_c": mn("tlimit"), "max_mib": mx("mib"),
             "max_step_up_w": round(max(steps), 1) if steps else None,
             "max_step_down_w": round(min(steps), 1) if steps else None,
             "busy_s": len(busy), "busy_mean_w": round(sum(s["w"] for s in busy) / len(busy), 1) if busy else None,
             "busy_mean_sm_mhz": round(sum(s["sm"] for s in busy) / len(busy)) if busy else None,
             "pcie_gen_min": mn("gen"), "pcie_width_min": mn("width"), "events": self.events,
             "cmd": self.a.cmd}
        with open(os.path.join(self.a.out, "guard.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        print(f"GUARD {d['status']} exit {code} child_rc {child_rc} reason '{d['reason']}' peak {d['peak_w']} W "
              f"(limit {d['power_limit_w']}) maxT {d['max_temp_c']} C min margin {d['min_tlimit_c']} C "
              f"max {d['max_mib']} MiB step +{d['max_step_up_w']}/{d['max_step_down_w']} W busy {d['busy_s']} s "
              f"@ {d['busy_mean_w']} W {d['busy_mean_sm_mhz']} MHz events {len(self.events)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-name", required=True, help="substring of the nvidia-smi name, e.g. 'RTX PRO 6000'")
    ap.add_argument("--pci", default="", help="optional PCI bus id suffix, e.g. 01:00.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latch", default=os.environ.get("GPU_GUARD_LATCH", ""),
                    help="STOP file: refuse to start while it exists; written on any trip")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--tlimit-min-c", type=float, default=5.0)
    ap.add_argument("--temp-max-c", type=float, default=88.0)
    ap.add_argument("--stale-s", type=float, default=5.0)
    ap.add_argument("--max-idle-mib", type=float, default=2048)
    ap.add_argument("--quiet-before-s", type=int, default=60)
    ap.add_argument("--replay-every-s", type=float, default=10.0)
    ap.add_argument("--post-s", type=float, default=5.0)
    ap.add_argument("--fail-pattern", default=FAIL_DEFAULT)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    if not a.cmd:
        ap.error("no command after --")
    # resolve through PATH: on Windows a bare 'bash' would otherwise start System32\bash.exe (WSL, ~10 s, other OS)
    a.cmd[0] = shutil.which(a.cmd[0]) or a.cmd[0]
    os.makedirs(a.out, exist_ok=True)
    g = Guard(a)
    t0 = now()

    def refuse(msg):
        g.log("REFUSED " + msg)
        g.trip = (96, "refused: " + msg[:200])
        g.summary(None, 96, t0, now())
        sys.exit(96)

    if a.latch and os.path.exists(a.latch):
        refuse(f"latch {a.latch} exists: " + open(a.latch, encoding="utf-8", errors="replace").read()[-300:].strip())
    uuid, info = resolve(a.gpu_name, a.pci)
    if not uuid:
        refuse(f"no GPU named '{a.gpu_name}' {a.pci}: {info[:200]}")
    g.log(f"GPU {info}")
    rc, out = smi(["-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if rc or (num(out.strip()) or 1e9) > a.max_idle_mib:
        refuse(f"GPU busy before launch: {out.strip()[:80]} MiB > {a.max_idle_mib}")

    here = os.path.dirname(os.path.abspath(__file__))
    procs = []
    if WIN:
        since = datetime.fromtimestamp(time.time() - a.quiet_before_s).isoformat(timespec="seconds")
        ev = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                               os.path.join(here, "winevents.ps1"), "-Since", since],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        procs.append(ev)
        threading.Thread(target=g.event_reader, args=(ev,), daemon=True).start()
    else:
        g.log("not Windows: no system-event watcher (telemetry, child rc and fail patterns still guard)")
    sm = subprocess.Popen(["nvidia-smi", "-i", uuid, "--query-gpu=" + FIELDS, "--format=csv,noheader,nounits", "-l", "1"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs.append(sm)
    threading.Thread(target=g.smi_reader, args=(sm,), daemon=True).start()
    rp0 = replays(uuid)
    g.log(f"PCIe replays before {rp0}")
    # pre-launch quiet window: 3 s of telemetry and event heartbeats, no trip (an event inside --quiet-before-s trips here)
    time.sleep(3.0)
    if g.trip:
        for p in procs:
            p.kill()
        g.summary(None, g.trip[0], t0, now())
        sys.exit(g.trip[0])

    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=uuid, GPU_GUARD_UUID=uuid)
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
    g.log("LAUNCH " + " ".join(a.cmd))
    tl = time.time()
    g.child = subprocess.Popen(a.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kw)
    cr = threading.Thread(target=g.child_reader, args=(g.child, re.compile(a.fail_pattern)), daemon=True)
    cr.start()
    next_rp = time.time() + a.replay_every_s
    while g.child.poll() is None:
        time.sleep(0.25)
        t = time.time()
        if t - g.last_smi > a.stale_s:
            g.fire(91, f"no telemetry for {t - g.last_smi:.1f} s (GPU not answering)")
        if WIN and t - g.last_hb > 3 * a.stale_s:
            g.fire(91, f"event watcher silent for {t - g.last_hb:.1f} s")
        if t - tl > a.timeout_s:
            g.fire(94, f"wall limit {a.timeout_s} s")
        if t >= next_rp:
            next_rp = t + a.replay_every_s
            rp = replays(uuid)
            if rp is not None and rp0 is not None and rp > rp0:
                g.fire(93, f"PCIe replays {rp0} -> {rp}")
    child_rc = g.child.wait()
    cr.join(timeout=5)
    g.log(f"child rc {child_rc} after {time.time() - tl:.1f} s")
    end = time.time() + a.post_s
    while time.time() < end and not g.trip:
        time.sleep(0.25)
        if time.time() - g.last_smi > a.stale_s:
            g.fire(91, "no telemetry after the run (GPU not answering)")
    u2, info2 = resolve(a.gpu_name, a.pci)
    if u2 != uuid:
        g.fire(91, f"GPU not visible after the run: {info2[:160]}")
    rp1 = replays(uuid) if u2 else None
    if rp1 is not None and rp0 is not None and rp1 > rp0:
        g.fire(93, f"PCIe replays {rp0} -> {rp1}")
    for p in procs:
        p.kill()
    code = g.trip[0] if g.trip else (child_rc if child_rc else 0)
    if code == 0 and child_rc:
        code = child_rc
    g.summary(child_rc, code, t0, now())
    sys.exit(code)


if __name__ == "__main__":
    main()
