"""
Integration tests — require the server to be running.

    python run.py   # in another terminal
    pytest tests/test_api_integration.py -v
"""
import requests
import pytest

BASE_URL = "http://localhost:4000"


def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_extract_person_and_org():
    resp = requests.post(
        f"{BASE_URL}/extract",
        json={
            "text": "Elon Musk founded SpaceX in 2002.",
            "labels": ["person", "organization"],
        },
    )
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    labels = {e["label"] for e in entities}
    texts  = {e["text"]  for e in entities}
    assert "person"       in labels
    assert "organization" in labels
    assert "Elon Musk"    in texts
    assert "SpaceX"       in texts


def test_extract_with_high_threshold():
    resp = requests.post(
        f"{BASE_URL}/extract",
        json={
            "text": "Barack Obama visited Paris.",
            "labels": ["person", "location"],
            "threshold": 0.9,
        },
    )
    assert resp.status_code == 200
    for e in resp.json()["entities"]:
        assert e["score"] >= 0.9


def test_extract_empty_text_returns_empty():
    resp = requests.post(
        f"{BASE_URL}/extract",
        json={"text": "", "labels": ["person"]},
    )
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_empty_labels_returns_empty():
    resp = requests.post(
        f"{BASE_URL}/extract",
        json={"text": "Apple is great.", "labels": []},
    )
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_invalid_threshold_rejected():
    resp = requests.post(
        f"{BASE_URL}/extract",
        json={"text": "Hello", "labels": ["person"], "threshold": 2.0},
    )
    assert resp.status_code == 422


def test_entity_fields_present():
    resp = requests.post(
        f"{BASE_URL}/extract",
        json={
            "text": "Tim Cook leads Apple.",
            "labels": ["person", "organization"],
        },
    )
    assert resp.status_code == 200
    for e in resp.json()["entities"]:
        assert {"text", "label", "score", "start", "end"} <= e.keys()
        assert 0.0 <= e["score"] <= 1.0
        assert e["start"] < e["end"]
