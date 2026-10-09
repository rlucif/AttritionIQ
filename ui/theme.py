"""Visual system for AttritionIQ: colour tokens, page accents and the CSS for custom blocks.

Streamlit's own widgets are themed in .streamlit/config.toml. This file only styles the
HTML blocks rendered by ui/blocks.py plus a few layout touches.
"""
from __future__ import annotations

import streamlit as st

# Colour tokens (same values as config.toml)
INK = "#0e100f"
SURFACE = "#1a1c1b"
LINE = "#2b2e2c"
CREAM = "#fffce1"
MUTED = "#a9a797"
GREEN = "#0ae448"
LILAC = "#9d95ff"
ORANGE = "#ff8709"
PINK = "#fec5fb"
SKY = "#00bae2"
RISK = "#ff6b5e"      # "raises risk"
SAFE = "#0ae448"      # "lowers risk"

# One accent per page, the way gsap.com gives each tool its own colour.
ACCENT = {"overview": GREEN, "employee": LILAC, "workforce": ORANGE, "trust": PINK}

CSS = f"""
:root {{
  --ink:{INK}; --surface:{SURFACE}; --line:{LINE}; --cream:{CREAM}; --muted:{MUTED};
  --green:{GREEN}; --lilac:{LILAC}; --orange:{ORANGE}; --pink:{PINK}; --sky:{SKY};
  --risk:{RISK}; --safe:{SAFE};
  --display:"Bricolage Grotesque", system-ui, sans-serif;
  --ease: cubic-bezier(.22,1,.36,1);
}}

/* ---------- layout ---------- */
[data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-top: 4.5rem; padding-bottom: 6rem; }}
[data-testid="stHeader"] {{ background: rgba(14,16,15,.82); backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--line); }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
[data-testid="stCaptionContainer"] {{ color: var(--muted); }}
[data-testid="stTabs"] button[role="tab"] p {{ font-weight: 600; }}

/* ---------- context bar ---------- */
.aiq-context {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; color:var(--muted);
  font-size:.86rem; }}
.aiq-chip {{ display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .7rem;
  border:1px solid var(--line); border-radius:999px; background:var(--surface); color:var(--cream);
  font-size:.82rem; white-space:nowrap; }}
.aiq-chip b {{ font-weight:600; color:var(--accent, var(--green)); }}
.aiq-chip--quiet {{ background:transparent; color:var(--muted); }}

/* ---------- hero ---------- */
.aiq-hero {{ position:relative; display:grid; grid-template-columns: minmax(0,1.05fr) minmax(0,.95fr);
  gap:2.5rem; align-items:center; padding: .5rem 0 1.5rem; }}
.aiq-hero--compact {{ grid-template-columns: minmax(0,1fr) auto; padding-bottom:.5rem; }}
.aiq-title {{ font-family:var(--display); font-weight:800; font-size:clamp(2.3rem, 4.4vw, 3.7rem);
  line-height:.98; letter-spacing:-.035em; margin:0 0 1rem; color:var(--cream); }}
.aiq-hero--compact .aiq-title {{ font-size:clamp(2.1rem, 4.4vw, 3.4rem); }}
.aiq-title .ln {{ display:block; overflow:hidden; padding-bottom:.06em; }}
.aiq-title .ln > span {{ display:inline-block; }}
.aiq-title .ln--accent > span {{ color: var(--accent); }}
.aiq-lede {{ font-size:1.1rem; line-height:1.55; color:var(--muted); max-width:52ch; margin:0 0 1.1rem; }}
.aiq-tags {{ display:flex; flex-wrap:wrap; gap:.4rem; }}
.aiq-tag {{ font-size:.78rem; padding:.22rem .65rem; border-radius:999px;
  border:1px solid color-mix(in srgb, var(--accent) 45%, transparent);
  color: color-mix(in srgb, var(--accent) 80%, var(--cream)); }}
.aiq-glyph {{ width:120px; height:120px; }}

/* ---------- workforce dot field (Overview hero) ---------- */
.aiq-field {{ background:var(--surface); border:1px solid var(--line); border-radius:1.4rem; padding:1.3rem 1.3rem 1rem; }}
.aiq-dots {{ display:grid; gap:clamp(3px, .55vw, 7px); }}
.aiq-dots .dot {{ display:block; aspect-ratio:1; border-radius:50%; background:var(--cream); opacity:.12; }}
.aiq-dots .dot.mid {{ opacity:.5; }}
.aiq-dots .dot.hot {{ background:var(--accent); opacity:1; box-shadow:0 0 12px color-mix(in srgb, var(--accent) 55%, transparent); }}
.aiq-legend {{ display:flex; flex-wrap:wrap; gap:1rem; margin-top:.9rem; font-size:.82rem; color:var(--muted); }}
.aiq-legend i {{ display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.35rem;
  vertical-align:middle; background:var(--cream); opacity:.45; }}
.aiq-legend i.hot {{ background:var(--accent); opacity:1; }}
.aiq-legend i.low {{ opacity:.14; }}

/* ---------- page glyph (compact heroes) ---------- */
.aiq-glyph {{ position:relative; width:128px; height:128px; }}
.aiq-glyph > span {{ position:absolute; display:block; }}
.aiq-glyph .g-star {{ inset:14px; background:var(--accent);
  clip-path: polygon(50% 0, 60% 40%, 100% 50%, 60% 60%, 50% 100%, 40% 60%, 0 50%, 40% 40%); }}
.aiq-glyph .g-ring {{ left:0; top:4px; width:26px; height:26px; border-radius:50%; border:3px solid var(--cream); }}
.aiq-glyph .g-square {{ right:0; bottom:0; width:24px; height:24px; border-radius:6px; background:var(--cream); opacity:.85; }}

/* ---------- KPI strip ---------- */
.aiq-kpis {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); border-top:1px solid var(--line);
  border-bottom:1px solid var(--line); margin: .5rem 0 1rem; }}
.aiq-kpi {{ padding:1.1rem 1.2rem 1.2rem 0; }}
.aiq-kpi + .aiq-kpi {{ padding-left:1.2rem; border-left:1px solid var(--line); }}
.aiq-kpi__v {{ font-family:var(--display); font-weight:700; font-size:clamp(1.7rem,3vw,2.5rem);
  letter-spacing:-.03em; line-height:1.05; color:var(--cream); font-variant-numeric: tabular-nums; }}
.aiq-kpi__l {{ color:var(--muted); font-size:.9rem; margin-top:.3rem; }}
.aiq-kpi__d {{ display:inline-block; margin-top:.45rem; font-size:.78rem; padding:.12rem .55rem; border-radius:999px;
  background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); }}
.aiq-kpi--hero .aiq-kpi__v {{ color: var(--accent); }}
.aiq-kpis--stack {{ grid-template-columns: 1fr; margin-top:0; }}
.aiq-kpis--stack .aiq-kpi + .aiq-kpi {{ padding-left:0; border-left:0; border-top:1px solid var(--line); }}

/* ---------- section + step headers ---------- */
.aiq-sec {{ margin: 2.4rem 0 .6rem; }}
.aiq-sec h2 {{ font-family:var(--display); font-size:1.75rem; font-weight:700; margin:0; letter-spacing:-.025em; }}
.aiq-sec p {{ color:var(--muted); margin:.35rem 0 0; max-width:70ch; }}
.aiq-step {{ display:grid; grid-template-columns:auto 1fr; gap:1rem; align-items:start; margin: 3rem 0 .9rem; }}
.aiq-step__n {{ width:2.6rem; height:2.6rem; border-radius:50%; display:grid; place-items:center;
  font-family:var(--display); font-weight:800; font-size:1.15rem; color:var(--ink); background:var(--accent); }}
.aiq-step h2 {{ font-family:var(--display); font-size:1.65rem; font-weight:700; margin:.15rem 0 0; letter-spacing:-.025em; }}
.aiq-step p {{ color:var(--muted); margin:.3rem 0 0; max-width:68ch; }}

/* ---------- employee card + gauge ---------- */
.aiq-person {{ display:grid; grid-template-columns:minmax(0,1.2fr) minmax(0,.8fr); gap:2rem; align-items:center;
  background:var(--surface); border:1px solid var(--line); border-radius:1.4rem; padding:1.6rem 1.8rem; margin-top:.4rem; }}
.aiq-person__id {{ font-family:var(--display); font-weight:800; font-size:clamp(2rem,4vw,3rem); letter-spacing:-.035em; line-height:1; }}
.aiq-person__role {{ font-size:1.15rem; margin:.5rem 0 1.1rem; color:var(--cream); }}
.aiq-facts {{ margin:0; padding:0; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; }}
.aiq-facts dt {{ color:var(--muted); font-size:.82rem; }}
.aiq-facts dd {{ margin:.15rem 0 0; font-weight:600; font-size:1.05rem; font-variant-numeric: tabular-nums; }}
.aiq-gauge {{ text-align:center; }}
.aiq-dial-wrap {{ position:relative; width:min(290px, 100%); margin: 1.6rem auto 0; }}
.aiq-dial {{ position:relative; width:100%; aspect-ratio:2/1; overflow:hidden; }}
.aiq-dial__arc {{ position:absolute; left:0; right:0; top:0; height:200%; border-radius:50%;
  background: conic-gradient(from 270deg, var(--gauge) calc(var(--p) * 180deg), var(--line) 0 180deg, transparent 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 20px), #000 calc(100% - 19px));
          mask: radial-gradient(farthest-side, transparent calc(100% - 20px), #000 calc(100% - 19px)); }}
.aiq-dial__tick {{ position:absolute; left:calc(50% - 1.5px); bottom:0; width:3px; height:100%; transform-origin:50% 100%;
  transform: rotate(calc(var(--t) * 180deg - 90deg));
  background: linear-gradient(to top, transparent calc(100% - 28px), var(--cream) 0); }}
.aiq-dial__v {{ position:absolute; left:0; right:0; bottom:-.1em; font-family:var(--display); font-weight:800;
  font-size:clamp(2.4rem, 5vw, 3.3rem); letter-spacing:-.04em; line-height:1; color:var(--cream); font-variant-numeric: tabular-nums; }}
.aiq-dial__tl {{ position:absolute; transform:translate(-50%, 50%); font-size:.75rem; color:var(--cream); white-space:nowrap; }}
.aiq-dial__scale {{ display:flex; justify-content:space-between; width:min(290px,100%); margin:.35rem auto 0; font-size:.75rem; color:var(--muted); }}
.aiq-dial__l {{ font-size:.85rem; color:var(--muted); margin:.4rem 0 .6rem; }}
.aiq-verdict {{ display:inline-block; margin-top:.2rem; padding:.3rem .8rem; border-radius:999px; font-weight:600; font-size:.9rem; }}
.aiq-verdict.hot {{ background: color-mix(in srgb, var(--risk) 18%, transparent); color: var(--risk); }}
.aiq-verdict.cool {{ background: color-mix(in srgb, var(--safe) 15%, transparent); color: var(--safe); }}

/* ---------- SHAP driver bars ---------- */
.aiq-drivers {{ display:grid; gap:.15rem; }}
.aiq-drv {{ display:grid; grid-template-columns:minmax(0,15rem) minmax(0,1fr) 4.2rem; gap:1rem; align-items:center;
  padding:.45rem 0; border-bottom:1px solid var(--line); }}
.aiq-drv__name {{ font-weight:600; }}
.aiq-drv__val {{ display:block; color:var(--muted); font-size:.8rem; font-weight:400; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.aiq-drv__track {{ position:relative; height:.75rem; }}
.aiq-drv__track::before {{ content:""; position:absolute; left:50%; top:-.3rem; bottom:-.3rem; width:1px; background:var(--line); }}
.aiq-drv__bar {{ position:absolute; top:0; height:100%; border-radius:999px; }}
.aiq-drv__bar.up {{ left:50%; background:var(--risk); transform-origin:left center; }}
.aiq-drv__bar.down {{ right:50%; background:var(--safe); transform-origin:right center; }}
.aiq-drv__num {{ text-align:right; font-variant-numeric: tabular-nums; font-size:.9rem; color:var(--muted); }}
.aiq-drv-legend {{ display:flex; justify-content:space-between; color:var(--muted); font-size:.8rem; margin:.3rem 0 .2rem; }}
.aiq-drv-legend span:first-child {{ color:var(--safe); }} .aiq-drv-legend span:last-child {{ color:var(--risk); }}

/* ---------- what-if result ---------- */
.aiq-shift {{ display:flex; align-items:baseline; gap:1rem; flex-wrap:wrap; background:var(--surface);
  border:1px solid var(--line); border-radius:1.2rem; padding:1.1rem 1.4rem; }}
.aiq-shift__from {{ font-family:var(--display); font-size:2rem; font-weight:700; color:var(--muted); }}
.aiq-shift__arrow {{ color: var(--muted); font-size:1.4rem; }}
.aiq-shift__to {{ font-family:var(--display); font-size:2.6rem; font-weight:800; letter-spacing:-.03em; }}
.aiq-shift__d {{ padding:.2rem .7rem; border-radius:999px; font-weight:600; font-size:.9rem; }}
.aiq-shift__d.down {{ background: color-mix(in srgb, var(--safe) 15%, transparent); color:var(--safe); }}
.aiq-shift__d.up {{ background: color-mix(in srgb, var(--risk) 18%, transparent); color:var(--risk); }}
.aiq-shift__d.flat {{ background: var(--line); color:var(--muted); }}

/* ---------- recommendation cards ---------- */
.aiq-recs {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; }}
.aiq-rec {{ background:var(--surface); border:1px solid var(--line); border-radius:1.2rem; padding:1.2rem 1.3rem;
  transition: transform .35s var(--ease), border-color .35s var(--ease); }}
.aiq-rec:first-child {{ border-color: color-mix(in srgb, var(--accent) 60%, transparent); }}
.aiq-rec__rank {{ color:var(--accent); font-weight:600; font-size:.85rem; }}
.aiq-rec h4 {{ font-family:var(--display); font-size:1.3rem; margin:.3rem 0 .8rem; letter-spacing:-.02em; }}
.aiq-meter {{ display:grid; grid-template-columns:6.5rem 1fr 2.6rem; gap:.6rem; align-items:center; font-size:.82rem; color:var(--muted); margin:.3rem 0; }}
.aiq-meter__t {{ height:.4rem; background:var(--line); border-radius:999px; overflow:hidden; }}
.aiq-meter__f {{ height:100%; background:var(--accent); border-radius:999px; transform-origin:left center; }}
.aiq-meter__f.alt {{ background: var(--cream); opacity:.55; }}
.aiq-rec__cost {{ margin-top:.9rem; padding-top:.8rem; border-top:1px solid var(--line); font-size:.9rem; color:var(--muted); }}
.aiq-rec__cost b {{ color:var(--cream); font-weight:600; }}

/* ---------- notes ---------- */
.aiq-note {{ border-left:3px solid var(--accent); padding:.55rem .9rem; color:var(--muted); font-size:.9rem;
  background: color-mix(in srgb, var(--accent) 6%, transparent); border-radius:0 .6rem .6rem 0; margin:.6rem 0; }}

/* ---------- responsive ---------- */
@media (max-width: 900px) {{
  .aiq-hero, .aiq-person {{ grid-template-columns: 1fr; }}
  .aiq-hero--compact {{ grid-template-columns: 1fr; }}
  .aiq-glyph {{ display:none; }}
  .aiq-chip--quiet {{ display:none; }}
  .aiq-kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .aiq-kpi:nth-child(3) {{ padding-left:0; border-left:0; }}
  .aiq-kpi:nth-child(n+3) {{ border-top:1px solid var(--line); }}
  .aiq-recs {{ grid-template-columns: 1fr; }}
  .aiq-drv {{ grid-template-columns: minmax(0,9rem) minmax(0,1fr) 3.4rem; }}
  .aiq-facts {{ grid-template-columns: repeat(2,minmax(0,1fr)); }}
}}

/* Hover answers the pointer, and only on cards you can read more from. */
@media (hover:hover) and (prefers-reduced-motion: no-preference) {{
  .aiq-rec:hover {{ transform: translateY(-4px); border-color: var(--accent); }}
}}
"""


def inject() -> None:
    """Add the stylesheet once per run (st.html sends style-only HTML to the event container)."""
    st.html(f"<style>{CSS}</style>")
