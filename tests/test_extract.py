"""
Unit tests — no real model loaded (GLiNER/torch stubbed in conftest.py).
Covers:
  - API contract (health, validation, threshold forwarding)
  - New v2 features: optional labels, bilingual expansion, labels_used echo
  - English / Chinese / Arabic / mixed-language text handling
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.models import Entity
from app.labels import DEFAULT_LABELS, expand_bilingual


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    mock_ner = MagicMock()
    # Default: extract() returns ([], [])
    mock_ner.extract.return_value = ([], [])
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.NERService", lambda *_: mock_ner)
        with TestClient(app) as c:
            yield c, mock_ner


def _ents(*args) -> tuple[list[Entity], list[str]]:
    """Helper: wrap Entity list in the (entities, labels_used) tuple."""
    entities = list(args)
    labels = [e.label for e in entities]
    return entities, labels


# ── System / API contract ─────────────────────────────────────────────────────

def test_health(client):
    c, _ = client
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_empty_text(client):
    c, mock_ner = client
    resp = c.post("/api/v1/extract", json={"text": "", "labels": ["person"]})
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_empty_labels_uses_defaults(client):
    """labels 为空时服务端应自动使用默认双语标签集，不报错。"""
    c, mock_ner = client
    mock_ner.extract.return_value = ([], DEFAULT_LABELS)
    resp = c.post("/api/v1/extract", json={"text": "Apple Inc. is in Cupertino."})
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
    assert "labels_used" in data
    assert len(data["labels_used"]) > 0


def test_extract_omit_labels_entirely(client):
    """labels 字段完全不传也应该正常工作。"""
    c, mock_ner = client
    mock_ner.extract.return_value = ([], DEFAULT_LABELS)
    resp = c.post("/api/v1/extract", json={"text": "Some text."})
    assert resp.status_code == 200


def test_extract_threshold_forwarded(client):
    c, mock_ner = client
    c.post("/api/v1/extract",
           json={"text": "Hello world", "labels": ["person"], "threshold": 0.8})
    mock_ner.extract.assert_called_once_with("Hello world", ["person"], 0.8, language="auto")


def test_extract_invalid_threshold(client):
    c, _ = client
    resp = c.post("/api/v1/extract",
                  json={"text": "Hello", "labels": ["person"], "threshold": 1.5})
    assert resp.status_code == 422


def test_extract_language_field_forwarded(client):
    c, mock_ner = client
    c.post("/api/v1/extract",
           json={"text": "北京协和医院", "labels": ["医院名称"], "language": "zh"})
    mock_ner.extract.assert_called_once_with("北京协和医院", ["医院名称"], 0.4, language="zh")


def test_extract_invalid_language(client):
    """不支持的 language 值应返回 422。"""
    c, _ = client
    resp = c.post("/api/v1/extract",
                  json={"text": "Hello", "language": "jp"})
    assert resp.status_code == 422


def test_entity_response_fields(client):
    """每个实体包含全部必填字段且值合法。"""
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="Apple", label="organization", score=0.95, start=0, end=5)
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "Apple is great.", "labels": ["organization"]})
    assert resp.status_code == 200
    e = resp.json()["entities"][0]
    assert set(e.keys()) >= {"text", "label", "score", "start", "end"}
    assert 0.0 <= e["score"] <= 1.0
    assert e["start"] < e["end"]


def test_labels_used_echoed(client):
    """响应中 labels_used 应回传实际使用的标签列表。"""
    c, mock_ner = client
    used = ["person", "organization"]
    mock_ner.extract.return_value = ([], used)
    resp = c.post("/api/v1/extract",
                  json={"text": "Elon Musk works at Tesla.", "labels": ["person"]})
    assert resp.status_code == 200
    assert resp.json()["labels_used"] == used


# ── Bilingual label expansion (unit-level, no HTTP) ───────────────────────────

def test_expand_bilingual_adds_english_for_chinese():
    result = expand_bilingual(["人名或姓名"])
    assert "人名或姓名" in result
    assert "full name of a person" in result


def test_expand_bilingual_adds_chinese_for_english():
    result = expand_bilingual(["company or organization name"])
    assert "company or organization name" in result
    assert "公司或组织机构名称" in result


def test_expand_bilingual_no_duplicate():
    labels = ["人名或姓名", "full name of a person"]
    result = expand_bilingual(labels)
    assert result.count("人名或姓名") == 1
    assert result.count("full name of a person") == 1


def test_expand_bilingual_custom_label_preserved():
    """自定义标签（不在对照表中）原样保留。"""
    result = expand_bilingual(["my custom label"])
    assert "my custom label" in result


def test_default_labels_nonempty():
    assert len(DEFAULT_LABELS) > 0
    # 必须包含中英文各至少一个
    has_en = any(all(ord(c) < 128 for c in lbl) for lbl in DEFAULT_LABELS)
    has_zh = any(any('一' <= c <= '鿿' for c in lbl) for lbl in DEFAULT_LABELS)
    assert has_en and has_zh


# ── English ───────────────────────────────────────────────────────────────────

def test_english_person_org(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="Elon Musk",  label="person",       score=0.98, start=0,  end=9),
        Entity(text="Tesla",      label="organization", score=0.96, start=18, end=23),
        Entity(text="SpaceX",     label="organization", score=0.97, start=28, end=34),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "Elon Musk is the CEO of Tesla and founded SpaceX.",
                        "labels": ["full name of a person", "company or organization name"]})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"Elon Musk", "Tesla", "SpaceX"} <= texts


def test_english_location_date(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="Paris",  label="location", score=0.94, start=20, end=25),
        Entity(text="2024",   label="date",     score=0.91, start=29, end=33),
        Entity(text="France", label="location", score=0.93, start=38, end=44),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "The summit was held in Paris in 2024, in France.",
                        "labels": ["geographical location", "date or year"]})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"Paris", "France", "2024"} <= texts


def test_english_threshold_filters(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="NASA", label="organization", score=0.95, start=0, end=4),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "NASA explored the Moon.",
                        "labels": ["company or organization name"],
                        "threshold": 0.8})
    assert resp.status_code == 200
    mock_ner.extract.assert_called_once_with(
        "NASA explored the Moon.", ["company or organization name"], 0.8, language="auto"
    )


# ── Chinese ───────────────────────────────────────────────────────────────────

def test_chinese_person_org(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="马云",     label="人名或姓名",         score=0.96, start=8,  end=10),
        Entity(text="张勇",     label="人名或姓名",         score=0.94, start=25, end=27),
        Entity(text="阿里巴巴", label="公司或组织机构名称", score=0.97, start=0,  end=4),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "阿里巴巴集团创始人马云卸任，由张勇接任。",
                        "labels": ["人名或姓名", "公司或组织机构名称"],
                        "language": "zh"})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"马云", "张勇", "阿里巴巴"} <= texts


def test_chinese_entity_boundary(client):
    """实体边界不应包含动词 — '尤氏来请' 应只取 '尤氏'。"""
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="尤氏",   label="人名或姓名", score=0.82, start=0,  end=2),
        Entity(text="王熙凤", label="人名或姓名", score=0.95, start=8, end=11),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "尤氏来请，王熙凤笑道：'你来了。'",
                        "labels": ["人名或姓名"]})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert "尤氏"       in texts
    assert "王熙凤"     in texts
    assert "尤氏来请"   not in texts
    assert "王熙凤笑道" not in texts


def test_chinese_location_product(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="杭州",  label="地名或城市",    score=0.93, start=17, end=19),
        Entity(text="淘宝",  label="产品或品牌名称", score=0.91, start=22, end=24),
        Entity(text="天猫",  label="产品或品牌名称", score=0.92, start=25, end=27),
        Entity(text="支付宝", label="产品或品牌名称", score=0.90, start=28, end=31),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "阿里巴巴总部位于杭州，旗下有淘宝、天猫、支付宝。",
                        "labels": ["地名或城市", "产品或品牌名称"]})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"杭州", "淘宝", "天猫", "支付宝"} <= texts


# ── Arabic ────────────────────────────────────────────────────────────────────

def test_arabic_person_location(client):
    """阿拉伯语：识别人名与地名。"""
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="محمد بن سلمان",           label="full name of a person", score=0.82, start=12, end=26),
        Entity(text="المملكة العربية السعودية", label="geographical location",  score=0.85, start=44, end=68),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "أعلن الرئيس محمد بن سلمان عن مشروع نيوم في المملكة العربية السعودية.",
                        "labels": ["full name of a person", "geographical location"],
                        "language": "ar"})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert "محمد بن سلمان"           in texts
    assert "المملكة العربية السعودية" in texts


# ── Mixed Chinese-English ─────────────────────────────────────────────────────

def test_mixed_entities_both_scripts(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="张伟",    label="person",       score=0.95, start=0,  end=2),
        Entity(text="Google",  label="organization", score=0.97, start=9,  end=15),
        Entity(text="北京",    label="location",     score=0.93, start=25, end=27),
        Entity(text="Android", label="product",      score=0.91, start=33, end=40),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "张伟入职了 Google，驻扎在北京，负责 Android 研发。",
                        "labels": ["full name of a person", "人名或姓名",
                                   "company or organization name", "公司或组织机构名称",
                                   "geographical location", "地名或城市",
                                   "product or technology name"]})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"张伟", "Google", "北京", "Android"} <= texts


def test_mixed_labels_chinese_and_english(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="李明",  label="人名或姓名", score=0.94, start=0,  end=2),
        Entity(text="Tesla", label="人名或姓名", score=0.96, start=10, end=15),
        Entity(text="上海",  label="地名或城市", score=0.92, start=22, end=24),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "李明在上海加入了 Tesla。",
                        "labels": ["人名或姓名", "full name of a person",
                                   "地名或城市", "geographical location",
                                   "company or organization name"]})
    assert resp.status_code == 200
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"李明", "Tesla", "上海"} <= texts


def test_mixed_no_cross_language_contamination(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="OpenAI", label="organization", score=0.97, start=5,  end=11),
        Entity(text="王芳",   label="person",       score=0.93, start=15, end=17),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "他在 OpenAI 工作，同事王芳也在同一部门。",
                        "labels": ["person", "organization"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert any(e["text"] == "OpenAI" and e["label"] == "organization" for e in entities)
    assert any(e["text"] == "王芳"   and e["label"] == "person"       for e in entities)
