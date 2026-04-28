"""
conftest.py stubs gliner/torch before these imports run,
so no real model is loaded during tests.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.models import Entity


@pytest.fixture()
def client():
    mock_ner = MagicMock()
    # Patch NERService so lifespan assigns our mock instead of a real model
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.NERService", lambda *_: mock_ner)
        with TestClient(app) as c:
            yield c, mock_ner


def test_health(client):
    c, _ = client
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_returns_entities(client):
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="Apple", label="organization", score=0.95, start=0, end=5)
    ]

    resp = c.post(
        "/api/v1/extract",
        json={"text": "Apple is a tech company.", "labels": ["organization", "person"]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entities"]) == 1
    assert data["entities"][0]["text"] == "Apple"
    assert data["entities"][0]["label"] == "organization"


def test_extract_empty_text(client):
    c, mock_ner = client
    mock_ner.extract.return_value = []

    resp = c.post("/api/v1/extract", json={"text": "", "labels": ["person"]})

    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_empty_labels(client):
    c, mock_ner = client
    mock_ner.extract.return_value = []

    resp = c.post("/api/v1/extract", json={"text": "Some text.", "labels": []})

    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_threshold_forwarded(client):
    c, mock_ner = client
    mock_ner.extract.return_value = []

    c.post(
        "/api/v1/extract",
        json={"text": "Hello world", "labels": ["person"], "threshold": 0.8},
    )

    mock_ner.extract.assert_called_once_with("Hello world", ["person"], 0.8)


def test_extract_invalid_threshold(client):
    c, _ = client
    resp = c.post(
        "/api/v1/extract",
        json={"text": "Hello", "labels": ["person"], "threshold": 1.5},
    )
    assert resp.status_code == 422
