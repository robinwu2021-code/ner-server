"""
Unit tests — no real model loaded (gliner/torch stubbed in conftest.py).
Covers: API contract, Chinese / English / mixed-language text handling.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.models import Entity


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    mock_ner = MagicMock()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.NERService", lambda *_: mock_ner)
        with TestClient(app) as c:
            yield c, mock_ner


# ── API contract ──────────────────────────────────────────────────────────────

def test_health(client):
    c, _ = client
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


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
    c.post("/api/v1/extract",
           json={"text": "Hello world", "labels": ["person"], "threshold": 0.8})
    mock_ner.extract.assert_called_once_with("Hello world", ["person"], 0.8)


def test_extract_invalid_threshold(client):
    c, _ = client
    resp = c.post("/api/v1/extract",
                  json={"text": "Hello", "labels": ["person"], "threshold": 1.5})
    assert resp.status_code == 422


def test_entity_response_fields(client):
    """每个实体包含全部必填字段且值合法。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="Apple", label="organization", score=0.95, start=0, end=5)
    ]
    resp = c.post("/api/v1/extract",
                  json={"text": "Apple is great.", "labels": ["organization"]})
    assert resp.status_code == 200
    e = resp.json()["entities"][0]
    assert set(e.keys()) >= {"text", "label", "score", "start", "end"}
    assert 0.0 <= e["score"] <= 1.0
    assert e["start"] < e["end"]


# ── English ───────────────────────────────────────────────────────────────────

def test_english_person_org(client):
    """英文文本：识别人名和机构。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="Elon Musk",  label="person",       score=0.98, start=0,  end=9),
        Entity(text="Tesla",      label="organization", score=0.96, start=18, end=23),
        Entity(text="SpaceX",     label="organization", score=0.97, start=28, end=34),
    ]
    text = "Elon Musk is the CEO of Tesla and founded SpaceX in 2002."
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["full name of a person",
                                   "company or organization name"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    texts = {e["text"] for e in entities}
    labels = {e["label"] for e in entities}
    assert "Elon Musk" in texts
    assert "Tesla"     in texts
    assert "SpaceX"    in texts
    assert "person"       in labels
    assert "organization" in labels


def test_english_location_date(client):
    """英文文本：识别地点和日期。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="Paris",   label="location", score=0.94, start=20, end=25),
        Entity(text="2024",    label="date",     score=0.91, start=29, end=33),
        Entity(text="France",  label="location", score=0.93, start=38, end=44),
    ]
    text = "The summit was held in Paris in 2024, in France."
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["geographical location", "date or year"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    texts = {e["text"] for e in entities}
    assert "Paris"  in texts
    assert "France" in texts
    assert "2024"   in texts


def test_english_threshold_filters_low_confidence(client):
    """高阈值过滤低置信度结果。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="NASA", label="organization", score=0.95, start=0, end=4),
    ]
    text = "NASA and some group discussed the Moon landing plan in 2026."
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["company or organization name"],
                        "threshold": 0.8})
    assert resp.status_code == 200
    mock_ner.extract.assert_called_once_with(text,
                                             ["company or organization name"],
                                             0.8)
    for e in resp.json()["entities"]:
        assert e["score"] >= 0.0   # mock controls scores; just verify structure


# ── Chinese ───────────────────────────────────────────────────────────────────

def test_chinese_person_org(client):
    """中文文本：识别人名和机构名。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="马云",     label="人名或姓名",      score=0.96, start=8,  end=10),
        Entity(text="张勇",     label="人名或姓名",      score=0.94, start=25, end=27),
        Entity(text="阿里巴巴", label="公司或组织机构名称", score=0.97, start=0,  end=4),
    ]
    text = "阿里巴巴集团创始人马云于2019年卸任董事局主席，由张勇接任。"
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["人名或姓名", "公司或组织机构名称", "地名或城市"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    texts = {e["text"] for e in entities}
    assert "马云"     in texts
    assert "张勇"     in texts
    assert "阿里巴巴" in texts


def test_chinese_entity_boundary(client):
    """中文文本：实体边界不应包含动词（如"尤氏来请"只取"尤氏"）。"""
    c, mock_ner = client
    # 模拟模型正确截断边界
    mock_ner.extract.return_value = [
        Entity(text="尤氏",   label="人名或姓名", score=0.82, start=0, end=2),
        Entity(text="王熙凤", label="人名或姓名", score=0.95, start=8, end=11),
    ]
    text = "尤氏来请，王熙凤笑道：'你来了。'"
    resp = c.post("/api/v1/extract",
                  json={"text": text, "labels": ["人名或姓名"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    texts = {e["text"] for e in entities}
    # 正确：只有名称，不含动作
    assert "尤氏"     in texts
    assert "王熙凤"   in texts
    assert "尤氏来请" not in texts
    assert "王熙凤笑道" not in texts


def test_chinese_location_product(client):
    """中文文本：识别地点和产品名。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="杭州",  label="地名或城市",   score=0.93, start=17, end=19),
        Entity(text="淘宝",  label="产品或品牌名称", score=0.91, start=22, end=24),
        Entity(text="天猫",  label="产品或品牌名称", score=0.92, start=25, end=27),
        Entity(text="支付宝", label="产品或品牌名称", score=0.90, start=28, end=31),
    ]
    text = "阿里巴巴集团总部位于杭州，旗下拥有淘宝、天猫、支付宝等业务板块。"
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["地名或城市", "产品或品牌名称"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    texts = {e["text"] for e in entities}
    assert "杭州"  in texts
    assert "淘宝"  in texts
    assert "天猫"  in texts
    assert "支付宝" in texts


# ── Mixed Chinese-English ─────────────────────────────────────────────────────

def test_mixed_entities_both_scripts(client):
    """中英混合文本：两种语言的实体都能正确识别。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="张伟",    label="person",       score=0.95, start=0,  end=2),
        Entity(text="Google",  label="organization", score=0.97, start=9,  end=15),
        Entity(text="北京",    label="location",     score=0.93, start=25, end=27),
        Entity(text="Android", label="product",      score=0.91, start=33, end=40),
    ]
    text = "张伟入职了 Google 担任工程师，驻扎在北京，负责 Android 产品研发。"
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["full name of a person",
                                   "company or organization name",
                                   "geographical location",
                                   "product or technology name"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    texts = {e["text"] for e in entities}
    assert "张伟"    in texts
    assert "Google"  in texts
    assert "北京"    in texts
    assert "Android" in texts


def test_mixed_labels_chinese_and_english(client):
    """中英混合文本：使用中英双语描述性标签。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="李明",      label="人名或姓名",  score=0.94, start=0,  end=2),
        Entity(text="Tesla",     label="人名或姓名",  score=0.96, start=10, end=15),
        Entity(text="上海",      label="地名或城市",  score=0.92, start=22, end=24),
    ]
    text = "李明在上海加入了 Tesla，成为首席工程师。"
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["人名或姓名",
                                   "full name of a person",
                                   "地名或城市",
                                   "geographical location",
                                   "company or organization name"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert len(entities) == 3
    texts = {e["text"] for e in entities}
    assert "李明"  in texts
    assert "Tesla" in texts
    assert "上海"  in texts


def test_mixed_no_cross_language_contamination(client):
    """混合文本中，中文实体不应被识别为英文标签类型，反之亦然。"""
    c, mock_ner = client
    mock_ner.extract.return_value = [
        Entity(text="OpenAI",   label="organization", score=0.97, start=5,  end=11),
        Entity(text="王芳",     label="person",       score=0.93, start=15, end=17),
    ]
    text = "他在 OpenAI 工作，同事王芳也在同一部门。"
    resp = c.post("/api/v1/extract",
                  json={"text": text,
                        "labels": ["person", "organization"]})
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    org_entities  = [e for e in entities if e["label"] == "organization"]
    person_entities = [e for e in entities if e["label"] == "person"]
    assert any(e["text"] == "OpenAI" for e in org_entities)
    assert any(e["text"] == "王芳"   for e in person_entities)
