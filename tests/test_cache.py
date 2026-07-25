"""Raw cache: roundtrip and key sensitivity to every component of the spec's
cache-key tuple."""

from __future__ import annotations

from pathlib import Path

from langbench.cache import RawCache, cache_key


def test_roundtrip(tmp_path: Path) -> None:
    c = RawCache(tmp_path / "raw.sqlite")
    key = cache_key("groq", "m1", "v1", {"temperature": 0.0}, "hello")
    assert c.get(key) is None
    c.put(key, "groq", "m1", "v1", {"text": "hi", "latency_ms": 5.0})
    got = c.get(key)
    assert got is not None and got["text"] == "hi"
    assert c.count() == 1


def test_put_is_idempotent(tmp_path: Path) -> None:
    c = RawCache(tmp_path / "raw.sqlite")
    key = cache_key("groq", "m1", "v1", {}, "x")
    c.put(key, "groq", "m1", "v1", {"text": "a"})
    c.put(key, "groq", "m1", "v1", {"text": "b"})
    got = c.get(key)
    assert got is not None and got["text"] == "b"
    assert c.count() == 1


def test_key_sensitivity() -> None:
    base = cache_key("groq", "m1", "v1", {"temperature": 0.0}, "text")
    assert base != cache_key("mistral", "m1", "v1", {"temperature": 0.0}, "text")
    assert base != cache_key("groq", "m2", "v1", {"temperature": 0.0}, "text")
    assert base != cache_key("groq", "m1", "v2", {"temperature": 0.0}, "text")
    assert base != cache_key("groq", "m1", "v1", {"temperature": 0.7}, "text")
    assert base != cache_key("groq", "m1", "v1", {"temperature": 0.0}, "TEXT")
    # And it must be stable across param dict ordering:
    assert cache_key("g", "m", "v", {"a": 1, "b": 2}, "t") == cache_key(
        "g", "m", "v", {"b": 2, "a": 1}, "t"
    )
