#!/usr/bin/env python3
"""Generate the README speed chart (docs/media/speed-*.svg) from measured numbers.

Writes two static SVGs, one for light and one for dark backgrounds; the README picks
one with <picture> and prefers-color-scheme. Edit MEASUREMENTS after a benchmark
campaign and rerun:

    python3 tools/gen_readme_chart.py

Every value must come from a recorded measurement; put its commit and date in SOURCE.
"""

from pathlib import Path

SOURCE = "One Bosgame M5 (Ryzen AI Max+ 395, 128 GB) · UD-Q4_K_XL · commit dff1dbe · 2026-10-18"

PROFILES = ("Exact (FP32 dense)", "Dense A8 (recommended)")

MEASUREMENTS = {
    # Decode with MTP windows {2,4,8}, code benchmark, 256 generated tokens, replay error 0.
    "decode": (53.1, 54.9),
    # Tiled prefill request throughput (tok/s) per prompt length.
    "prefill": {
        "1K": (401, 703),
        "4K": (388, 694),
        "16K": (375, 668),
        "32K": (371, 656),
    },
}

THEMES = {
    "light": {
        "surface": "#ffffff", "border": "#d0d7de", "ink": "#1f2328", "ink2": "#59636e",
        "grid": "#e6e8eb", "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#0d1117", "border": "#30363d", "ink": "#f0f6fc", "ink2": "#9198a1",
        "grid": "#21262d", "series": ("#3987e5", "#d95926"),
    },
}

W, H = 920, 380
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def bar(x, y_top, w, base, color):
    """Bar with a 4px rounded top anchored square on the baseline."""
    h = base - y_top
    r = min(4, h / 2, w / 2)
    return (f'<path d="M{x:.1f},{base:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} V{base:.1f} Z" fill="{color}"/>')


def panel(out, t, x0, x1, title, subtitle, groups, vmax, ticks):
    top, base = 128, 318
    out.append(f'<text x="{x0}" y="96" font-size="15" font-weight="600" fill="{t["ink"]}">{title}</text>')
    out.append(f'<text x="{x0}" y="114" font-size="12" fill="{t["ink2"]}">{subtitle}</text>')
    plot_x0 = x0 + 34
    for tick in ticks:
        y = base - (base - top) * tick / vmax
        out.append(f'<line x1="{plot_x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{plot_x0 - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{t["ink2"]}">{tick}</text>')
    n = len(groups)
    slot = (x1 - plot_x0) / n
    bw = min(34, slot * 0.3)
    gap = 2
    for i, (label, values) in enumerate(groups):
        cx = plot_x0 + slot * (i + 0.5)
        for k, v in enumerate(values):
            bx = cx - bw - gap / 2 if k == 0 else cx + gap / 2
            y = base - (base - top) * v / vmax
            out.append(bar(bx, y, bw, base, t["series"][k]))
            txt = f"{v:.1f}" if isinstance(v, float) else f"{v}"
            out.append(f'<text x="{bx + bw / 2:.1f}" y="{y - 6:.1f}" font-size="11" text-anchor="middle" '
                       f'fill="{t["ink"]}">{txt}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base + 18}" font-size="12" text-anchor="middle" fill="{t["ink2"]}">{label}</text>')
    out.append(f'<line x1="{plot_x0}" x2="{x1}" y1="{base}" y2="{base}" stroke="{t["ink2"]}" stroke-width="1"/>')


def render(mode):
    t = THEMES[mode]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="{FONT}" role="img" aria-labelledby="title desc">',
         '<title id="title">Measured speed: writing answers and reading prompts</title>',
         '<desc id="desc">Decode with MTP: exact profile {0} tok/s, dense A8 {1} tok/s. Prefill tok/s by prompt length, '
         'exact / dense A8: {2}. {3}.</desc>'.format(
             *MEASUREMENTS["decode"],
             ", ".join(f"{k} {a}/{b}" for k, (a, b) in MEASUREMENTS["prefill"].items()), SOURCE),
         f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="{t["surface"]}" stroke="{t["border"]}"/>',
         f'<text x="28" y="40" font-size="19" font-weight="700" fill="{t["ink"]}">Tokens per second, one Strix Halo box</text>']
    # Legend
    lx = 28
    for k, name in enumerate(PROFILES):
        o.append(f'<rect x="{lx}" y="56" width="12" height="12" rx="3" fill="{t["series"][k]}"/>')
        o.append(f'<text x="{lx + 18}" y="66.5" font-size="12.5" fill="{t["ink"]}">{name}</text>')
        lx += 18 + 8 * len(name) + 24
    panel(o, t, 28, 268, "Writes answers", "decode, MTP on, 256 tokens",
          [("code prompt", MEASUREMENTS["decode"])], 70, (0, 20, 40, 60))
    panel(o, t, 330, 892, "Reads your prompt", "prefill, by prompt length (tokens)",
          list(MEASUREMENTS["prefill"].items()), 800, (0, 200, 400, 600, 800))
    o.append(f'<text x="28" y="{H - 16}" font-size="11" fill="{t["ink2"]}">{SOURCE}. Higher is better.</text>')
    o.append('</svg>')
    return "\n".join(o) + "\n"


def main():
    media = Path(__file__).resolve().parent.parent / "docs" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        (media / f"speed-{mode}.svg").write_text(render(mode), encoding="utf-8")
        print(media / f"speed-{mode}.svg")


if __name__ == "__main__":
    main()
