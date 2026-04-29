"""
Integration tests — require the server to be running.

    python run.py   # in another terminal
    pytest tests/test_api_integration.py -v -s
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import requests

import os
BASE_URL = os.getenv("NER_BASE_URL", "http://localhost:4000/api/v1")
HF_BASE_URL = "https://robinwu-nerserver.hf.space/api/v1"


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
        "text": (
            "Elon Musk, the CEO of Tesla and founder of SpaceX, announced a new partnership "
            "with NASA last Tuesday. The deal, signed at Kennedy Space Center in Florida, "
            "will see SpaceX supply rockets for upcoming lunar missions planned by NASA over the next decade."
        ),
        "labels": ["person", "organization", "location"],
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
        "text": (
            "Former US President Barack Obama delivered a keynote speech at the United Nations "
            "headquarters in New York City on Monday, addressing climate change alongside French "
            "President Emmanuel Macron and German Chancellor Olaf Scholz. The event drew leaders "
            "from over fifty countries including Japan, Brazil, and South Africa."
        ),
        "labels": ["person", "location", "organization"],
        "threshold": 0.9,
    }
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract high threshold", payload, resp, elapsed)

    assert resp.status_code == 200
    for e in resp.json()["entities"]:
        assert e["score"] >= 0.9


def test_extract_empty_text_returns_empty():
    payload = {"text": "", "labels": ["person", "organization", "location"]}
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract empty text", payload, resp, elapsed)

    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_empty_labels_returns_empty():
    payload = {
        "text": (
            "Apple Inc. reported record quarterly earnings on Thursday, with CEO Tim Cook "
            "crediting strong iPhone sales in markets across Europe and Southeast Asia. "
            "The company also announced plans to expand its research center in Austin, Texas."
        ),
        "labels": [],
    }
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract empty labels", payload, resp, elapsed)

    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_invalid_threshold_rejected():
    payload = {"text": "Hello world, this is a simple test sentence.", "labels": ["person"], "threshold": 2.0}
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract invalid threshold", payload, resp, elapsed)

    assert resp.status_code == 422


def test_entity_fields_present():
    payload = {
        "text": (
            "Tim Cook, CEO of Apple, met with Sundar Pichai from Google and Satya Nadella "
            "from Microsoft at a technology summit held in San Francisco last week. The three "
            "executives discussed artificial intelligence regulation and data privacy policies "
            "being proposed by the European Union and the US Congress."
        ),
        "labels": ["person", "organization", "location"],
    }
    resp, elapsed = _call("POST", "/extract", payload)
    _print("extract field check", payload, resp, elapsed)

    assert resp.status_code == 200
    for e in resp.json()["entities"]:
        assert {"text", "label", "score", "start", "end"} <= e.keys()
        assert 0.0 <= e["score"] <= 1.0
        assert e["start"] < e["end"]


def test_concurrent_two_requests():
    payloads = [
        {
            "text": (
                "Jeff Bezos founded Amazon in a garage in Bellevue, Washington in 1994. "
                "The company started as an online bookstore before expanding into cloud computing "
                "through AWS, making Amazon one of the most valuable companies in the world."
            ),
            "labels": ["person", "organization", "location"],
        },
        {
            "text": (
                "The World Health Organization declared a new health advisory after researchers "
                "at Johns Hopkins University and the University of Oxford published findings on "
                "antibiotic resistance. Dr. Maria Chen led the study, which was funded by the "
                "Bill & Melinda Gates Foundation."
            ),
            "labels": ["person", "organization", "location"],
        },
    ]

    results = {}
    t0 = time.perf_counter()

    def fetch(idx: int, payload: dict):
        t = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/extract", json=payload)  # BASE_URL already includes /api/v1
        return idx, payload, resp, time.perf_counter() - t

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, i, p) for i, p in enumerate(payloads)]
        for f in as_completed(futures):
            idx, payload, resp, elapsed = f.result()
            results[idx] = (payload, resp, elapsed)

    total = time.perf_counter() - t0

    print(f"\n{'─' * 60}")
    print("[concurrent 2 requests]")
    for idx in sorted(results):
        payload, resp, elapsed = results[idx]
        print(f"  --- request {idx + 1} ---")
        print(f"  input : {json.dumps(payload, ensure_ascii=False)}")
        print(f"  output: {json.dumps(resp.json(), ensure_ascii=False)}")
        print(f"  time  : {elapsed * 1000:.1f} ms")
    print(f"  total wall time: {total * 1000:.1f} ms")

    for idx in sorted(results):
        _, resp, _ = results[idx]
        assert resp.status_code == 200
        assert len(resp.json()["entities"]) > 0
