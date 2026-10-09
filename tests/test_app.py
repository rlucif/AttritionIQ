"""App smoke tests: every page renders without an exception (headless, Streamlit AppTest).

Needs a trained bundle, so run after `python train.py`. Re-run these on the
deployment host (Phase 6) before opening the site.

`AppTest.switch_page` only handles file-based pages, and app.py registers its pages
as callables with `st.navigation`. So each test runs app.py with navigation stubbed
out, then calls that page's `render(ctx)` directly: the same code path the
navigation would run.
"""
from pathlib import Path

import pytest

from src import config

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not config.BUNDLE_PATH.exists(), reason="run `python train.py` first")

PAGE_SCRIPT = """
import sys, runpy, importlib
sys.path.insert(0, {root!r})
import streamlit as st

class _NoNav:
    def run(self):
        pass

st.navigation = lambda *a, **k: _NoNav()
g = runpy.run_path({root!r} + "/app.py")
importlib.import_module("views.{page}").render(g["ctx"])
"""


@pytest.mark.parametrize("page", ["overview", "employee", "workforce", "trust"])
def test_page_renders(page):
    at = st_testing.AppTest.from_string(PAGE_SCRIPT.format(root=str(ROOT), page=page),
                                        default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_classic_app_renders():
    at = st_testing.AppTest.from_file(str(ROOT / "app_classic.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 7
