"""Motion layer: loads GSAP into the page once and animates the custom blocks.

How it works
------------
* `inject()` adds one small script (st.html with JavaScript enabled). It loads GSAP (self-hosted in
  static/), then watches the page with a MutationObserver. Streamlit swaps an HTML block's
  nodes whenever its content changes, so every new or updated block is picked up.
* Blocks opt in with data attributes (see ui/blocks.py):
    data-aiq-hero    headline lines rise in; the workforce dot field staggers in from the centre
    data-aiq-count   numbers count up from their previous value (so changing an assumption
                     visibly moves the KPI instead of snapping)
    data-aiq-gauge   the risk dial sweeps from the last employee's score to the new one
    data-aiq-bars    bars grow from zero, staggered
    data-aiq-reveal  step headers animate as they scroll into view
* Text is always rendered with its final value in HTML first, so the page reads correctly
  if GSAP is blocked or the browser asks for reduced motion (then nothing animates).
"""
from __future__ import annotations

import streamlit as st

# Self-hosted copy of GSAP 3.12.5 (served from static/ because config.toml enables static serving),
# so the motion works offline and on any host without a CDN.
GSAP_URL = "app/static/gsap.min.js"

_JS = r"""
<script>
(function () {
  if (window.__aiq) return;
  const A = window.__aiq = { vals: {}, seen: {} };
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce) return;   // final values are already in the HTML

  const fmt = {
    int:   v => Math.round(v).toLocaleString('en-US'),
    money: v => '__CUR__' + Math.round(v).toLocaleString('en-US'),
    pct:   v => Math.round(v * 100) + '%',
    pct1:  v => (v * 100).toFixed(1) + '%',
    dec2:  v => v.toFixed(2),
  };

  function pick(root, sel) {
    const out = root.querySelectorAll ? Array.from(root.querySelectorAll(sel)) : [];
    if (root.matches && root.matches(sel)) out.unshift(root);
    return out.filter(el => { if (el.__aiq) return false; el.__aiq = 1; return true; });
  }

  function count(el, g) {
    const to = parseFloat(el.dataset.to), key = el.dataset.key || '';
    const from = key in A.vals ? A.vals[key] : (el.dataset.from ? parseFloat(el.dataset.from) : 0);
    A.vals[key] = to;
    if (!isFinite(to) || from === to) return;
    const f = fmt[el.dataset.fmt || 'int'] || fmt.int, o = { v: from };
    el.textContent = f(from);
    g.to(o, { v: to, duration: parseFloat(el.dataset.dur || 1.3), ease: 'power3.out',
              delay: parseFloat(el.dataset.delay || 0),
              onUpdate: () => { el.textContent = f(o.v); } });
  }

  function hero(el, g) {
    const key = el.dataset.aiqHero, first = !A.seen[key];
    A.seen[key] = 1;
    const dots = el.querySelectorAll('.dot'), hot = el.querySelectorAll('.dot.hot');
    // flagged dots keep "breathing" so the at-risk group stays easy to spot
    const breathe = d => hot.length && g.to(hot, { scale: .72, duration: 1.4, ease: 'sine.inOut', repeat: -1, yoyo: true,
                                                   stagger: { each: .07, from: 'random' }, delay: d });
    if (!first) {   // assumptions changed: re-flag only the at-risk dots
      if (hot.length) g.from(hot, { scale: 0, duration: .5, ease: 'back.out(3)', stagger: { amount: .4, from: 'random' } });
      breathe(.9);
      return;
    }
    const tl = g.timeline({ defaults: { ease: 'power4.out' } });
    tl.from(el.querySelectorAll('.aiq-title .ln > span'), { yPercent: 110, duration: 1.05, stagger: .09 })
      .from(el.querySelectorAll('.aiq-lede, .aiq-tags'), { y: 18, autoAlpha: 0, duration: .8, stagger: .08 }, '-=.7');
    if (dots.length) {
      const cols = parseInt(el.dataset.cols || '1'), rows = Math.ceil(dots.length / cols);
      tl.from(dots, { scale: 0, autoAlpha: 0, duration: .5, ease: 'power2.out',
                      stagger: { grid: [rows, cols], from: 'center', amount: 1.1 } }, .15)
        .to(hot, { scale: 1.9, duration: .22, ease: 'power2.out', yoyo: true, repeat: 1,
                   stagger: { amount: .5, from: 'random' } }, '-=.2');
      breathe(2.6);
    }
    const glyph = el.querySelectorAll('.aiq-glyph > *');
    if (glyph.length) {
      tl.from(glyph, { scale: 0, rotation: -120, duration: 1.1, ease: 'back.out(1.8)', stagger: .12 }, .1);
      g.to(el.querySelector('.aiq-glyph .spin'), { rotation: '+=360', duration: 24, ease: 'none', repeat: -1, delay: 1.3 });
    }
  }

  function gauge(el, g) {
    const key = el.dataset.key || 'gauge', to = parseFloat(el.dataset.to);
    const from = key in A.vals ? A.vals[key] : 0;
    A.vals[key] = to;
    const txt = el.querySelector('.aiq-dial__v'), o = { v: from };
    const paint = () => { el.style.setProperty('--p', o.v.toFixed(4));
                          if (txt) txt.textContent = Math.round(o.v * 100) + '%'; };
    paint();
    g.to(o, { v: to, duration: 1.4, ease: 'power3.inOut', onUpdate: paint });
  }

  function bars(el, g) {
    g.from(el.querySelectorAll('.aiq-drv__bar, .aiq-meter__f'),
           { scaleX: 0, duration: .9, ease: 'power3.out', stagger: .045, delay: .1 });
  }

  const io = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      const g = window.gsap, t = e.target;
      g.timeline({ defaults: { ease: 'power3.out' } })
        .to(t.querySelector('.aiq-step__n'), { scale: 1, rotation: 0, duration: .7, ease: 'back.out(2)' })
        .to(t.querySelectorAll('h2, p'), { y: 0, autoAlpha: 1, duration: .6, stagger: .07 }, '-=.45');
    });
  }, { threshold: .4 }) : null;

  function reveal(el, g) {
    if (!io) return;
    g.set(el.querySelector('.aiq-step__n'), { scale: 0, rotation: -90 });
    g.set(el.querySelectorAll('h2, p'), { y: 16, autoAlpha: 0 });
    io.observe(el);
  }

  function scan(root) {
    const g = window.gsap;
    pick(root, '[data-aiq-hero]').forEach(el => hero(el, g));
    pick(root, '[data-aiq-count]').forEach(el => count(el, g));
    pick(root, '[data-aiq-gauge]').forEach(el => gauge(el, g));
    pick(root, '[data-aiq-bars]').forEach(el => bars(el, g));
    pick(root, '[data-aiq-reveal]').forEach(el => reveal(el, g));
  }

  const s = document.createElement('script');
  s.src = '__GSAP__';
  s.onload = () => {
    scan(document.body);
    new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(n => { if (n.nodeType === 1) scan(n); })))
      .observe(document.body, { childList: true, subtree: true });
  };
  document.head.appendChild(s);
})();
</script>
"""


def inject(currency: str = "$") -> None:
    st.html(_JS.replace("__GSAP__", GSAP_URL).replace("__CUR__", currency), unsafe_allow_javascript=True)
