# Power and energy receipt

Energy is measured once, in the shared base, and reported by every engine in the same block. The code is
[include/strix/power/sampler.hpp](../../include/strix/power/sampler.hpp) and
[src/power/sampler.cpp](../../src/power/sampler.cpp): a host library (`strix_power`) with no device runtime
dependency.

## What it does

- A sampler thread reads the registered sources every 100 ms (`STRIX_POWER_SAMPLE_MS`, 10..10000) into a fixed
  ring (65,536 samples, about 1.8 hours at the default period).
- It needs no elevated rights and changes no system setting. A source that is missing or unreadable is reported as
  unavailable with the reason and skipped. A run never fails because of the sampler.
- The device gate starts it when a binary binds its device, and the HTTP server starts it at server start, so an
  engine adds nothing to be sampled.
- A receipt is a time window of the ring. The window edges are interpolated between the neighbouring samples, so the
  caller only passes timestamps it already has.

## Sources

| source | reads | channels |
|---|---|---|
| `amdgpu-hwmon` | `hwmon*/power1_average` (or `power1_input`), `freq1_input`, `temp1_input`, `device/gpu_busy_percent` of every hwmon device named `amdgpu` | power, GPU clock, temperature, GPU load |
| `rapl` | `powercap/intel-rapl:N/energy_uj` of the package zones, with `max_energy_range_uj` for the wrap | package energy counter |
| `cpufreq` | `cpufreq/policy*/scaling_cur_freq`, at most once per second | mean CPU clock |

On an integrated AMD device the hwmon power value is the package power tracking value of the whole SoC (label
`PPT`), so it covers CPU and GPU together. On a discrete card it is the card.

Energy comes from the package counter when it is readable (`energy_source: "counter"`), otherwise from the
trapezoid integral of the power samples (`energy_source: "power-integral"`).

### Package energy counter permission

Many kernels expose `energy_uj` to root only (mode 0400). The sampler does not work around that: the `rapl`
source reports `permission denied: <path>` and the receipt uses the power integral. To let the sampler read the
counter, the machine owner grants read access once per boot:

```sh
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj
```

## The receipt block

Schema `strix-power-receipt-v1`, one object:

```json
{"schema":"strix-power-receipt-v1","enabled":true,"available":true,
 "sources":["amdgpu-hwmon","cpufreq"],
 "source_detail":[{"name":"rapl","available":false,"reason":"permission denied: ..."},
                  {"name":"amdgpu-hwmon","available":true,"detail":"hwmon5: power1_average(PPT) freq1_input temp1_input gpu_busy_percent"},
                  {"name":"cpufreq","available":true,"detail":"32 policies: scaling_cur_freq"}],
 "request":{...},"prefill":{...},"decode":{...}}
```

Each of `request`, `prefill` and `decode` is either `{"available":false,"reason":"..."}` or:

| field | meaning |
|---|---|
| `duration_ms` | length of the window |
| `energy_j` | energy in the window |
| `avg_w`, `peak_w` | mean power (`energy_j / duration`) and the highest power sample |
| `tokens`, `tok_per_j` | tokens of the phase and tokens per joule (prefill: prompt tokens computed; decode: tokens emitted) |
| `gpu_clock_mhz_median` | median GPU clock over the samples of the window |
| `temp_c_max` | highest temperature sample |
| `cpu_mhz_median`, `gpu_busy_pct_mean` | CPU clock and GPU load |
| `samples` | samples inside the window |
| `energy_source` | `counter` or `power-integral` |

The energy of a window is the energy of the machine during that window. Requests that overlap in time each
report the whole machine; the block does not apportion energy between concurrent requests.

## Where it appears

- **Engine CLI** (`strix-hip-decode-token`): member `"power"` of the request receipt on stdout. With one generated
  token the request is one prefill phase (the first token is inside, as in the request timing). With more, the
  phases split at the first emitted token; a route that does not mark its first token reports the request total only.
- **Every binary that binds a device**: one line `strix power: {...}` on stderr at exit with the process total.
- **HTTP server**: `timings.power` of a completion that carries timings, `power` on each entry of
  `GET /api/status` `requests`, and a top-level `power` object in `GET /api/status` (schema
  `strix-power-status-v1`: sources, the latest reading, energy and mean power since start). The request windows
  are counted back from completion with the request's own `prefill_ms` and `decode_ms`, so a family reports
  nothing extra.

## Switch

`STRIX_POWER_SAMPLER=0` turns the sampler, the receipt members and the exit line off. The sampler does not touch
the device, the model or any arithmetic: token ids and logits are identical with it on or off.

## Measured

Qwen, 8,192-token real-text prompt, one node, default exact route, package energy counter not readable (power
integral, 10 Hz, 46 samples per prefill):

| run | tok/s | energy | mean / peak power | tok/J | median GPU clock | max temperature |
|---|---|---|---|---|---|---|
| prefill, sampler on (1) | 1,755.9 | 576.6 J | 123.6 / 150.0 W | 14.21 | 2,514 MHz | 84 C |
| prefill, sampler off (1) | 1,752.9 | - | - | - | - | - |
| prefill, sampler on (2) | 1,744.5 | 580.0 J | 123.5 / 150.7 W | 14.12 | 2,524 MHz | 89 C |
| prefill, sampler off (2) | 1,733.5 | - | - | - | - | - |
| prefill of the decode run | 1,747.8 | 587.8 J | 125.4 / 151.7 W | 13.94 | 2,512 MHz | 92 C |
| decode, 128 tokens, serial greedy | 22.5 | 543.8 J | 95.7 / 150.5 W | 0.235 | 2,873 MHz | 74 C |

The runs alternate on / off in one session. The sampler costs nothing measurable (on 1,750.2 tok/s mean, off
1,743.2; the spread between repeats of one arm is larger than the difference between arms), and the logits file and
the generated tokens are byte-identical in all four prefill runs.

## Adding a device

A new device (a discrete card, an accelerator, another operating system) is one source and one registration; no
caller changes:

```cpp
class MyCard final : public strix::power::Source {
    strix::power::SourceStatus probe() override;        // open what is readable; say why when it is not
    void read(strix::power::Sample& sample) noexcept override;  // add power or energy, set clock, raise temperature
};
strix::power::register_source("my-card", [](const std::string& root) { return std::make_unique<MyCard>(root); });
```

`read()` adds its power (or cumulative energy) to the sample, takes the maximum for temperature and sets a clock
that is still absent. Sources must cover disjoint power domains, because their power is added. Register before the
first device bind. Tests pass a directory that holds a fake `sys/` tree as `root`
([tests/power_sampler_tests.cpp](../../tests/power_sampler_tests.cpp)).
