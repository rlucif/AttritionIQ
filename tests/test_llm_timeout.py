"""The retention brief must fall back to the template when the LLM never answers
(seen on the live site, 2026-10-09: the page spun on "Drafting the brief" for 4+ minutes).
No dataset or API key needed: a local socket accepts the request and stays silent."""
import socket
import threading
import time

import pytest

from src import rag


@pytest.fixture
def silent_server():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    held = []
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                held.append(srv.accept()[0])   # accept, then never reply
            except OSError:
                pass

    threading.Thread(target=accept, daemon=True).start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    stop.set()
    for c in held:
        c.close()
    srv.close()


def test_gemini_stall_falls_back_to_template(monkeypatch, silent_server):
    from google import genai

    real_client = genai.Client

    def client_pointing_at_silent_server(*args, http_options=None, **kwargs):
        assert http_options is not None and http_options.timeout, "Gemini call has no timeout"
        http_options.base_url = silent_server
        return real_client(*args, http_options=http_options, **kwargs)

    monkeypatch.setattr(genai, "Client", client_pointing_at_silent_server)
    monkeypatch.setattr(rag, "LLM_TIMEOUT_S", 2.0)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    t0 = time.time()
    text, provider = rag.generate("Brief for HR.\nEmployee: #1, P(leave) 0.50")
    elapsed = time.time() - t0

    assert provider == "template"
    assert "LLM call failed" in text and "Employee:" in text
    assert elapsed < 15, f"fallback took {elapsed:.1f}s; the timeout is not applied"
