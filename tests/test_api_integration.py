"""
Integration tests — require the server to be running.

    python run.py   # in another terminal
    pytest tests/test_api_integration.py -v -s
"""
import json
import time

import pytest
import requests

BASE_URL = "http://localhost:4000"


def _call(method: str, path: str, payload: dict | None = None) -> tuple[requests.Response, float]:
    t0 = time.perf_counter()
    if method == "GET":
        resp = requests.get(f"{BASE_URL}{path}")
    else:
        resp = requests.post(f"{BASE_URL}{path}", json=payload)
    elapsed = time.perf_counter() - t0
    return resp, elapsed


def _print(label: str, payload: dict | None, resp: requests.Response, elapsed: float):
    print(f"\n{'─' * 60}")
    print(f"[{label}]")
    if payload is not None:
        print(f"  input : {json.dumps(payload, ensure_ascii=False)}")
    print(f"  output: {json.dumps(resp.json(), ensure_ascii=False)}")
    print(f"  status: {resp.status_code}  time: {elapsed * 1000:.1f} ms")


def test_health():
    resp, elapsed = _call("GET", "/health")
    _print("health", None, resp, elapsed)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_extract_person_and_org():
    payload = {
        "text": "Elon Musk founded SpaceX in 2002.",
        "labels": ["person", "organization"],
    }
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract person & org", payload, resp, elapsed)

    assert resp.status_code == 200
    entities = resp.json()["entities"]
    labels = {e["label"] for e in entities}
    texts  = {e["text"]  for e in entities}
    assert "person"       in labels
    assert "organization" in labels
    assert "Elon Musk"    in texts
    assert "SpaceX"       in texts


def test_extract_with_high_threshold():
    payload = {
        "text": "Barack Obama visited Paris.",
        "labels": ["person", "location"],
        "threshold": 0.9,
    }
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract high threshold", payload, resp, elapsed)

    assert resp.status_code == 200
    for e in resp.json()["entities"]:
        assert e["score"] >= 0.9


def test_extract_empty_text_returns_empty():
    payload = {"text": "", "labels": ["person"]}
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract empty text", payload, resp, elapsed)

    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_empty_labels_returns_empty():
    payload = {"text": "Apple is great.", "labels": []}
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract empty labels", payload, resp, elapsed)

    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_invalid_threshold_rejected():
    payload = {"text": "Hello", "labels": ["person"], "threshold": 2.0}
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract invalid threshold", payload, resp, elapsed)

    assert resp.status_code == 422


def test_entity_fields_present():
    payload = {
        "text": "Tim Cook leads Apple.",
        "labels": ["person", "organization"],
    }
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract field check", payload, resp, elapsed)

    assert resp.status_code == 200
    for e in resp.json()["entities"]:
        assert {"text", "label", "score", "start", "end"} <= e.keys()
        assert 0.0 <= e["score"] <= 1.0
        assert e["start"] < e["end"]
