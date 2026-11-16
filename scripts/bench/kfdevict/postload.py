#!/usr/bin/env python3
# postload.py RUNDIR: rank-0 userptr allocations made after the weight load ended (load time from r0.log), i.e. during
# the timed prefills: count, MB, the >= 1 MB ones (size@t). Clock: kfdevmon monotonic seconds, t0 = first allocation.
import glob, re, sys
d = sys.argv[1]; log = open(d + '/r0.log', errors='replace').read()
load = float(re.search(r'rank 0: load ([\d.]+) s', log).group(1)); m = sorted(glob.glob(d + '/mon.*'))[0]
t0 = None; n = 0; tot = 0; big = []
for l in open(m, errors='replace'):
    x = re.match(r'([\d.]+) ALLOC USERPTR size=(\d+)', l)
    if not x: continue
    t = float(x.group(1)); t0 = t if t0 is None else t0
    if t - t0 < load + 0.5: continue
    n += 1; s = int(x.group(2)); tot += s
    if s >= 1 << 20: big.append('%.1fMB@%.0f' % (s / 1e6, t - t0))
print('%s: load %.1f s; after load %d userptr allocs %.1f MB; >=1MB: %s' % (d.split('/')[-1], load, n, tot / 1e6, ' '.join(big[:12])))
