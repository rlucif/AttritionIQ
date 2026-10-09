"""HTML building blocks. Each one renders its final values in plain HTML; ui/motion.py
animates them when GSAP is available."""
from __future__ import annotations

import math
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from ui import theme

CUR = config.CURRENCY_SYMBOL


def money(x: float) -> str:
    return f"{CUR}{x:,.0f}"


def nice(name: str) -> str:
    """'YearsSinceLastPromotion' -> 'Years since last promotion' (keeps acronyms like ID intact)."""
    out, prev = [], ""
    for ch in str(name):
        if ch.isupper() and prev and (prev.islower() or prev.isdigit()):
            out.append(" ")
        out.append(ch)
        prev = ch
    words = "".join(out).split(" ")
    return " ".join([words[0]] + [w if w.isupper() and len(w) > 1 else w.lower() for w in words[1:]])


def _fmt(v: float, fmt: str) -> str:
    return {"int": f"{v:,.0f}", "money": money(v), "pct": f"{v:.0%}", "pct1": f"{v:.1%}",
            "dec2": f"{v:.2f}"}[fmt]


def count(v: float, fmt: str, key: str, delay: float = 0) -> str:
    """A number that counts up from its previous value."""
    return (f'<span data-aiq-count data-to="{v}" data-fmt="{fmt}" data-key="{escape(key)}" '
            f'data-delay="{delay}">{_fmt(v, fmt)}</span>')


class Blocks:
    """Renders blocks in one page's accent colour."""

    def __init__(self, page: str):
        self.page = page
        self.accent = theme.ACCENT[page]

    def _html(self, body: str) -> None:
        st.html(f'<div style="--accent:{self.accent}">{body}</div>')

    # ---------------------------------------------------------------- hero
    def hero(self, lines: list[str], lede: str, tags: list[str], right: str = "", compact: bool = False,
             cols: int = 1) -> None:
        title = "".join(f'<span class="ln"><span>{escape(l)}</span></span>' for l in lines)
        tag_html = "".join(f'<span class="aiq-tag">{escape(t)}</span>' for t in tags)
        right = right or (self.glyph() if compact else "")
        self._html(
            f'<section class="aiq-hero{" aiq-hero--compact" if compact else ""}" '
            f'data-aiq-hero="{self.page}" data-cols="{cols}">'
            f'<div><h1 class="aiq-title">{title}</h1><p class="aiq-lede">{escape(lede)}</p>'
            f'<div class="aiq-tags">{tag_html}</div></div>{right}</section>')

    def glyph(self) -> str:
        """A small shape cluster per page, echoing gsap.com's header shapes (plain HTML: st.html strips SVG)."""
        return ('<div class="aiq-glyph" aria-hidden="true"><span class="g-star spin"></span>'
                '<span class="g-ring"></span><span class="g-square"></span></div>')

    @staticmethod
    def dot_field(ids, p, threshold: float) -> tuple[str, int]:
        """One dot per employee, lit when the model flags them. Returns (html, columns)."""
        n = len(p)
        cols = max(1, round(math.sqrt(n * 1.6)))
        dots = []
        for eid, pi in zip(ids, p):
            cls = "hot" if pi >= threshold else ("mid" if pi >= threshold / 2 else "low")
            dots.append(f'<i class="dot {cls}" title="#{eid}: {pi:.0%}"></i>')
        flagged = int((np.asarray(p) >= threshold).sum())
        grid = (f'<div class="aiq-dots" role="img" aria-label="{n} employees, {flagged} flagged at risk" '
                f'style="grid-template-columns:repeat({cols},minmax(0,1fr))">{"".join(dots)}</div>')
        legend = (f'<div class="aiq-legend"><span><i class="hot"></i>Flagged at risk ({flagged})</span>'
                  f'<span><i></i>Watch: above half the threshold</span><span><i class="low"></i>Low risk</span></div>')
        return f'<div class="aiq-field">{grid}{legend}</div>', cols

    # ---------------------------------------------------------------- numbers
    def kpis(self, items: list[dict], stack: bool = False) -> None:
        """items: value, fmt, key, label, optional delta (text) and hero (bool). stack=True for a narrow column."""
        cells = []
        for i, it in enumerate(items):
            d = f'<div class="aiq-kpi__d">{escape(it["delta"])}</div>' if it.get("delta") else ""
            cells.append(f'<div class="aiq-kpi{" aiq-kpi--hero" if it.get("hero") else ""}">'
                         f'<div class="aiq-kpi__v">{count(it["value"], it["fmt"], self.page + it["key"], i * .08)}</div>'
                         f'<div class="aiq-kpi__l">{escape(it["label"])}</div>{d}</div>')
        self._html(f'<div class="aiq-kpis{" aiq-kpis--stack" if stack else ""}">{"".join(cells)}</div>')

    # ---------------------------------------------------------------- headings
    def section(self, title: str, text: str = "") -> None:
        p = f"<p>{escape(text)}</p>" if text else ""
        self._html(f'<div class="aiq-sec"><h2>{escape(title)}</h2>{p}</div>')

    def step(self, n: int, title: str, text: str = "") -> None:
        p = f"<p>{escape(text)}</p>" if text else ""
        self._html(f'<div class="aiq-step" data-aiq-reveal><div class="aiq-step__n">{n}</div>'
                   f'<div><h2>{escape(title)}</h2>{p}</div></div>')

    def note(self, text: str) -> None:
        self._html(f'<div class="aiq-note">{escape(text)}</div>')

    # ---------------------------------------------------------------- employee
    def person(self, emp: pd.Series, p: float, threshold: float) -> None:
        flagged = p >= threshold
        gauge_col = theme.RISK if flagged else self.accent
        ang = math.pi * (1 - threshold)            # where the threshold tick sits on the half-dial
        lx, ly = 50 + 50 * 1.2 * math.cos(ang), 100 * 1.2 * math.sin(ang)
        verdict = ('<span class="aiq-verdict hot">Flagged at risk</span>' if flagged
                   else '<span class="aiq-verdict cool">Below the threshold</span>')
        facts = [("Department", emp.Department), ("Monthly income", money(emp.MonthlyIncome)),
                 ("Years at company", int(emp.YearsAtCompany)), ("Overtime", "Yes" if emp.OverTime else "No"),
                 ("Job level", int(emp.JobLevel)), ("Work-life balance", f"{int(emp.WorkLifeBalance)} of 4")]
        dl = "".join(f"<div><dt>{escape(k)}</dt><dd>{escape(str(v))}</dd></div>" for k, v in facts)
        gauge = (
            f'<div class="aiq-gauge" data-aiq-gauge data-key="emp-gauge" data-to="{p:.4f}" '
            f'style="--p:{p:.4f};--t:{threshold:.4f};--gauge:{gauge_col}" role="img" '
            f'aria-label="Chance of leaving {p:.0%}, threshold {threshold:.0%}">'
            f'<div class="aiq-dial-wrap"><div class="aiq-dial"><div class="aiq-dial__arc"></div>'
            f'<div class="aiq-dial__tick"></div><div class="aiq-dial__v">{p:.0%}</div></div>'
            f'<span class="aiq-dial__tl" style="left:{lx:.1f}%;bottom:{ly:.1f}%">threshold {threshold:.0%}</span></div>'
            f'<div class="aiq-dial__scale"><span>0%</span><span>100%</span></div>'
            f'<div class="aiq-dial__l">chance of leaving within 12 months</div>{verdict}</div>')
        self._html(
            f'<div class="aiq-person"><div><div class="aiq-person__id">Employee #{escape(str(emp[config.ID_COL]))}</div>'
            f'<div class="aiq-person__role">{escape(str(emp.JobRole))}</div><dl class="aiq-facts">{dl}</dl></div>'
            f'{gauge}</div>')

    def drivers(self, d: pd.DataFrame) -> None:
        """Diverging SHAP bars: right/red raises risk, left/green lowers it."""
        top = d.head(10)
        m = float(top.shap.abs().max()) or 1.0
        rows = []
        for _, r in top.iterrows():
            w = abs(r.shap) / m * 50
            side = "up" if r.shap > 0 else "down"
            val = r.value if isinstance(r.value, str) else (f"{r.value:,.2f}".rstrip("0").rstrip(".")
                                                            if isinstance(r.value, (float, np.floating)) else r.value)
            rows.append(f'<div class="aiq-drv"><div class="aiq-drv__name">{escape(nice(r.feature))}'
                        f'<span class="aiq-drv__val">value: {escape(str(val))}</span></div>'
                        f'<div class="aiq-drv__track"><div class="aiq-drv__bar {side}" style="width:{w:.1f}%"></div></div>'
                        f'<div class="aiq-drv__num">{r.shap:+.2f}</div></div>')
        self._html('<div class="aiq-drv-legend"><span>Lowers risk</span><span>Raises risk</span></div>'
                   f'<div class="aiq-drivers" data-aiq-bars>{"".join(rows)}</div>')

    def shift(self, p_from: float, p_to: float) -> None:
        d = (p_to - p_from) * 100
        cls = "down" if d < -0.05 else ("up" if d > 0.05 else "flat")
        txt = f"{d:+.1f} pts" if cls != "flat" else "no change"
        self._html(f'<div class="aiq-shift"><span class="aiq-shift__from">{p_from:.0%}</span>'
                   f'<span class="aiq-shift__arrow" aria-hidden="true">&#8594;</span>'
                   f'<span class="aiq-shift__to">{count(p_to, "pct", "whatif")}</span>'
                   f'<span class="aiq-shift__d {cls}">{txt}</span>'
                   f'<span style="color:var(--muted);font-size:.9rem">simulated chance of leaving</span></div>')

    def recs(self, recs: pd.DataFrame) -> None:
        cards = []
        for i, (_, r) in enumerate(recs.iterrows()):
            meters = "".join(
                f'<div class="aiq-meter"><span>{lab}</span><div class="aiq-meter__t">'
                f'<div class="aiq-meter__f{alt}" style="width:{max(v, 0) * 100:.0f}%"></div></div><span>{v:.2f}</span></div>'
                for lab, v, alt in [("Overall fit", r.hybrid_score, ""), ("Matches drivers", r.content_score, " alt"),
                                    ("Similar staff", r.cf_score, " alt")])
            cards.append(f'<article class="aiq-rec"><div class="aiq-rec__rank">Option {i + 1}</div>'
                         f'<h4>{escape(r["name"])}</h4>{meters}'
                         f'<div class="aiq-rec__cost">Estimated cost <b>{money(r.est_cost)}</b> '
                         f'({r.cost_months:g} months of salary)</div></article>')
        self._html(f'<div class="aiq-recs" data-aiq-bars>{"".join(cards)}</div>')

    def context(self, chips: list[tuple[str, str]]) -> None:
        html = "".join(f'<span class="aiq-chip">{escape(k)} <b>{escape(v)}</b></span>' for k, v in chips)
        self._html(f'<div class="aiq-context">{html}</div>')
