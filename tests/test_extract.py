"""
Unit tests — no real model loaded (GLiNER/torch stubbed via conftest.py).
Covers:
  - API contract (health, validation, threshold, language field)
  - Dual-model routing (ZH → BERT backend, EN/AR/mixed → GLiNER backend)
  - Optional labels + bilingual auto-expansion
  - English / Chinese / Arabic / mixed-language scenarios
  - labels_used echo in response
"""
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Entity
from app.labels import DEFAULT_LABELS, expand_bilingual, labels_to_bert_types
from app.ner import detect_language


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ents(*args: Entity) -> tuple[list[Entity], list[str]]:
    """Wrap entities in (entities, labels_used) tuple expected by NERService."""
    return list(args), [e.label for e in args]


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """
    Patch NERService so no model is actually loaded.
    mock_ner.extract() returns ([], []) by default.
    """
    mock_ner = MagicMock()
    mock_ner.extract.return_value = ([], [])
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.NERService", lambda *_: mock_ner)
        with TestClient(app) as c:
            yield c, mock_ner


# ── System / API contract ─────────────────────────────────────────────────────

def test_health(client):
    c, _ = client
    assert c.get("/api/v1/health").json() == {"status": "ok"}


def test_extract_empty_text_returns_empty(client):
    c, _ = client
    resp = c.post("/api/v1/extract", json={"text": "", "labels": ["person"]})
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_labels_optional(client):
    """labels 字段完全不传应正常返回 200。"""
    c, mock_ner = client
    mock_ner.extract.return_value = ([], DEFAULT_LABELS)
    resp = c.post("/api/v1/extract", json={"text": "Some text."})
    assert resp.status_code == 200
    assert len(resp.json()["labels_used"]) > 0


def test_extract_empty_labels_uses_defaults(client):
    """labels=[] 时应使用默认双语标签集。"""
    c, mock_ner = client
    mock_ner.extract.return_value = ([], DEFAULT_LABELS)
    resp = c.post("/api/v1/extract", json={"text": "Hello world.", "labels": []})
    assert resp.status_code == 200
    assert resp.json()["labels_used"] == DEFAULT_LABELS


def test_extract_threshold_forwarded(client):
    c, mock_ner = client
    c.post("/api/v1/extract",
           json={"text": "Hello", "labels": ["person"], "threshold": 0.8})
    mock_ner.extract.assert_called_once_with(
        "Hello", ["person"], 0.8, language="auto", min_entities=None
    )


def test_extract_invalid_threshold(client):
    c, _ = client
    assert c.post("/api/v1/extract",
                  json={"text": "x", "threshold": 1.5}).status_code == 422


def test_extract_language_field_forwarded(client):
    c, mock_ner = client
    c.post("/api/v1/extract",
           json={"text": "北京协和医院", "labels": ["医院名称"], "language": "zh"})
    mock_ner.extract.assert_called_once_with(
        "北京协和医院", ["医院名称"], 0.4, language="zh", min_entities=None
    )


def test_extract_min_entities_forwarded(client):
    c, mock_ner = client
    c.post("/api/v1/extract",
           json={"text": "马云在杭州。", "language": "zh", "min_entities": 5})
    mock_ner.extract.assert_called_once_with(
        "马云在杭州。", [], 0.4, language="zh", min_entities=5
    )


def test_extract_negative_min_entities_rejected(client):
    c, _ = client
    resp = c.post("/api/v1/extract",
                  json={"text": "x", "min_entities": -1})
    assert resp.status_code == 422


def test_extract_invalid_language(client):
    c, _ = client
    assert c.post("/api/v1/extract",
                  json={"text": "x", "language": "jp"}).status_code == 422


def test_labels_used_echoed(client):
    c, mock_ner = client
    used = ["人名或姓名", "地名或城市"]
    mock_ner.extract.return_value = ([], used)
    resp = c.post("/api/v1/extract", json={"text": "马云在杭州。", "labels": ["人名或姓名"]})
    assert resp.json()["labels_used"] == used


def test_entity_response_fields(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="Apple", label="organization", score=0.95, start=0, end=5)
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "Apple is great.", "labels": ["organization"]})
    e = resp.json()["entities"][0]
    assert {"text", "label", "score", "start", "end"} <= e.keys()
    assert 0.0 <= e["score"] <= 1.0
    assert e["start"] < e["end"]


# ── Language detection ────────────────────────────────────────────────────────

def test_detect_language_english():
    assert detect_language("Elon Musk founded SpaceX in California.") == "en"


def test_detect_language_chinese():
    assert detect_language("阿里巴巴集团创始人马云于杭州卸任。") == "zh"


def test_detect_language_arabic():
    assert detect_language("أعلن الرئيس محمد بن سلمان عن مشروع نيوم.") == "ar"


def test_detect_language_mixed():
    assert detect_language("张伟加入了 Google 北京研发中心，负责 Android 优化。") == "mixed"


def test_detect_language_empty():
    assert detect_language("") == "en"


# ── Bilingual label expansion ─────────────────────────────────────────────────

def test_expand_bilingual_adds_english_for_chinese():
    result = expand_bilingual(["人名或姓名"])
    assert "full name of a person" in result


def test_expand_bilingual_adds_chinese_for_english():
    result = expand_bilingual(["company or organization name"])
    assert "公司或组织机构名称" in result


def test_expand_bilingual_no_duplicate():
    result = expand_bilingual(["人名或姓名", "full name of a person"])
    assert result.count("人名或姓名") == 1
    assert result.count("full name of a person") == 1


def test_expand_bilingual_custom_label_preserved():
    result = expand_bilingual(["my custom label"])
    assert "my custom label" in result


def test_default_labels_bilingual():
    has_en = any(all(ord(c) < 128 for c in lbl) for lbl in DEFAULT_LABELS)
    has_zh = any(any('一' <= c <= '鿿' for c in lbl) for lbl in DEFAULT_LABELS)
    assert has_en and has_zh


# ── BERT label mapping ────────────────────────────────────────────────────────

def test_labels_to_bert_types_chinese_label():
    types = labels_to_bert_types(["人名或姓名"])
    assert "PER" in types


def test_labels_to_bert_types_english_label():
    types = labels_to_bert_types(["geographical location"])
    assert "LOC" in types


def test_labels_to_bert_types_empty_returns_none():
    assert labels_to_bert_types([]) is None


def test_labels_to_bert_types_unmapped_returns_none():
    # 无法映射的标签 → 不过滤（返回 None）
    assert labels_to_bert_types(["some unknown label"]) is None


# ── English ───────────────────────────────────────────────────────────────────

def test_english_person_org(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="Elon Musk", label="person",       score=0.98, start=0,  end=9),
        Entity(text="Tesla",     label="organization", score=0.96, start=18, end=23),
        Entity(text="SpaceX",    label="organization", score=0.97, start=28, end=34),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "Elon Musk is the CEO of Tesla and founded SpaceX.",
                        "labels": ["full name of a person", "company or organization name"],
                        "language": "en"})
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
                        "labels": ["geographical location", "date or year"],
                        "language": "en"})
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"Paris", "France", "2024"} <= texts


def test_english_threshold_forwarded(client):
    c, mock_ner = client
    c.post("/api/v1/extract",
           json={"text": "NASA explored the Moon.",
                 "labels": ["company or organization name"],
                 "threshold": 0.8, "language": "en"})
    mock_ner.extract.assert_called_once_with(
        "NASA explored the Moon.", ["company or organization name"], 0.8,
        language="en", min_entities=None,
    )


# ── Chinese (BERT backend) ────────────────────────────────────────────────────

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
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"马云", "张勇", "阿里巴巴"} <= texts


def test_chinese_entity_boundary(client):
    """BERT NER 应精确截断实体边界，不含动词。"""
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="尤氏",   label="人名或姓名", score=0.82, start=0,  end=2),
        Entity(text="王熙凤", label="人名或姓名", score=0.95, start=8, end=11),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "尤氏来请，王熙凤笑道：'你来了。'",
                        "labels": ["人名或姓名"], "language": "zh"})
    texts = {e["text"] for e in resp.json()["entities"]}
    assert "尤氏"       in texts
    assert "王熙凤"     in texts
    assert "尤氏来请"   not in texts
    assert "王熙凤笑道" not in texts


def test_chinese_location_product(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="杭州",  label="地名或城市",    score=0.93, start=9,  end=11),
        Entity(text="淘宝",  label="产品或品牌名称", score=0.91, start=14, end=16),
        Entity(text="天猫",  label="产品或品牌名称", score=0.92, start=17, end=19),
        Entity(text="支付宝", label="产品或品牌名称", score=0.90, start=20, end=23),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "阿里巴巴总部位于杭州，旗下有淘宝、天猫、支付宝。",
                        "labels": ["地名或城市", "产品或品牌名称"],
                        "language": "zh"})
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"杭州", "淘宝", "天猫", "支付宝"} <= texts


def test_chinese_auto_routes_to_zh(client):
    """auto 检测到中文应路由到 ZH 模型（language 透传为 'auto'，内部检测为 zh）。"""
    c, mock_ner = client
    c.post("/api/v1/extract",
           json={"text": "马云创立了阿里巴巴。"})
    # NERService.extract 被调用时 language='auto'，路由逻辑在 ner.py 内部处理
    mock_ner.extract.assert_called_once()
    call_kwargs = mock_ner.extract.call_args
    assert call_kwargs[1].get("language") == "auto" or call_kwargs[0][3] == "auto"


# ── Arabic ────────────────────────────────────────────────────────────────────

def test_arabic_person_location(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="محمد بن سلمان",           label="full name of a person", score=0.82, start=12, end=26),
        Entity(text="المملكة العربية السعودية", label="geographical location",  score=0.85, start=44, end=68),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "أعلن الرئيس محمد بن سلمان مشروع نيوم في المملكة العربية السعودية.",
                        "labels": ["full name of a person", "geographical location"],
                        "language": "ar"})
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
                                   "product or technology name"],
                        "language": "mixed"})
    texts = {e["text"] for e in resp.json()["entities"]}
    assert {"张伟", "Google", "北京", "Android"} <= texts


def test_mixed_no_cross_language_contamination(client):
    c, mock_ner = client
    mock_ner.extract.return_value = _ents(
        Entity(text="OpenAI", label="organization", score=0.97, start=5,  end=11),
        Entity(text="王芳",   label="person",       score=0.93, start=15, end=17),
    )
    resp = c.post("/api/v1/extract",
                  json={"text": "他在 OpenAI 工作，同事王芳也在同一部门。",
                        "labels": ["person", "organization"]})
    entities = resp.json()["entities"]
    assert any(e["text"] == "OpenAI" and e["label"] == "organization" for e in entities)
    assert any(e["text"] == "王芳"   and e["label"] == "person"       for e in entities)


# ── Fallback & merge (NERService unit tests, no HTTP) ────────────────────────

def _build_svc():
    """Construct a bare NERService with mocked backends and locks."""
    import threading
    from app.ner import NERService
    svc = NERService.__new__(NERService)
    svc._en_lock = threading.Lock()
    svc._zh_lock = threading.Lock()
    svc._en_backend = MagicMock()
    svc._zh_backend = MagicMock()
    return svc


# ── Sufficiency heuristic ────────────────────────────────────────────────────

def test_expected_min_short_text():
    from app.ner import NERService
    assert NERService._expected_min("马云", []) == 1


def test_expected_min_medium_text():
    from app.ner import NERService
    text = "x" * 50
    assert NERService._expected_min(text, []) == 2


def test_expected_min_long_text():
    from app.ner import NERService
    text = "x" * 350
    assert NERService._expected_min(text, []) == 4


def test_expected_min_label_floor_takes_over():
    """9 个标签 → ⌈9/3⌉=3，超过短文本的 length_floor=1，最终取 3。"""
    from app.ner import NERService
    short_text = "马云"
    labels = [f"l{i}" for i in range(9)]
    assert NERService._expected_min(short_text, labels) == 3


# ── ZH branch fallback ───────────────────────────────────────────────────────

def test_zh_empty_triggers_fallback_and_adds():
    """ZH 主模型 0 个 → 触发兜底 → 返回兜底结果。"""
    svc = _build_svc()
    svc._zh_backend.predict.return_value = ([], [])
    svc._en_backend.predict.return_value = _ents(
        Entity(text="马云", label="person", score=0.75, start=0, end=2)
    )
    entities, _ = svc.extract("马云", [], 0.4, language="zh")
    assert any(e.text == "马云" for e in entities)
    svc._zh_backend.predict.assert_called_once()
    svc._en_backend.predict.assert_called_once()


def test_zh_sufficient_no_fallback():
    """ZH 主模型实体数 ≥ expected_min(=1 短文本) → 不调用兜底。"""
    svc = _build_svc()
    svc._zh_backend.predict.return_value = _ents(
        Entity(text="马云", label="person", score=0.92, start=0, end=2)
    )
    svc.extract("马云", [], 0.4, language="zh")
    svc._en_backend.predict.assert_not_called()


def test_zh_insufficient_triggers_fallback_and_results_added():
    """
    关键测试：ZH 返回 1 个，但文本长 → expected_min=4，不充分 →
    触发兜底，主结果 + 兜底结果一并返回（相加，不替换）。
    """
    svc = _build_svc()
    long_text = "马云" + "x" * 350     # length_floor = 4
    svc._zh_backend.predict.return_value = _ents(
        Entity(text="马云", label="人名或姓名", score=0.95, start=0, end=2),
    )
    svc._en_backend.predict.return_value = _ents(
        Entity(text="Tesla", label="organization", score=0.90, start=10, end=15),
        Entity(text="2024",  label="date",         score=0.88, start=20, end=24),
    )
    entities, _ = svc.extract(long_text, [], 0.4, language="zh")

    texts = {e.text for e in entities}
    # 主模型的"马云"必须保留，同时兜底的 Tesla / 2024 也加进来
    assert "马云"  in texts
    assert "Tesla" in texts
    assert "2024"  in texts
    assert len(entities) == 3      # 1 + 2 = 3，确实是相加


def test_user_min_entities_overrides_heuristic():
    """请求里传 min_entities=5 时应覆盖启发式，主模型 3 个仍触发兜底。"""
    svc = _build_svc()
    svc._zh_backend.predict.return_value = _ents(
        Entity(text="马云", label="person", score=0.95, start=0, end=2),
        Entity(text="张勇", label="person", score=0.93, start=4, end=6),
        Entity(text="杭州", label="location", score=0.91, start=8, end=10),
    )
    svc._en_backend.predict.return_value = _ents(
        Entity(text="Tesla", label="organization", score=0.85, start=15, end=20),
    )
    entities, _ = svc.extract("马云、张勇、杭州 Tesla", [], 0.4,
                              language="zh", min_entities=5)

    # 3 < 5 → 触发兜底；最终 3 + 1 = 4
    assert len(entities) == 4
    svc._en_backend.predict.assert_called_once()


def test_user_min_entities_zero_disables_fallback():
    """min_entities=0 时主模型即使返回空也不触发兜底。"""
    svc = _build_svc()
    svc._zh_backend.predict.return_value = ([], [])
    svc.extract("马云", [], 0.4, language="zh", min_entities=0)
    svc._en_backend.predict.assert_not_called()


# ── EN branch fallback (symmetric) ───────────────────────────────────────────

def test_en_insufficient_triggers_zh_fallback_and_adds():
    """EN 主模型不充分 → 调 ZH 兜底 → 结果相加。"""
    svc = _build_svc()
    long_text = "Tesla" + "x" * 350
    svc._en_backend.predict.return_value = _ents(
        Entity(text="Tesla", label="organization", score=0.95, start=0, end=5),
    )
    svc._zh_backend.predict.return_value = _ents(
        Entity(text="马云", label="人名或姓名", score=0.91, start=10, end=12),
    )
    entities, _ = svc.extract(long_text, [], 0.4, language="en")

    texts = {e.text for e in entities}
    assert "Tesla" in texts and "马云" in texts
    assert len(entities) == 2


# ── Mixed: always merge both ─────────────────────────────────────────────────

def test_mixed_always_runs_both_models():
    """mixed 语言无视充分性，永远跑两个模型并合并。"""
    svc = _build_svc()
    svc._en_backend.predict.return_value = _ents(
        Entity(text="Google", label="organization", score=0.95, start=5, end=11)
    )
    svc._zh_backend.predict.return_value = _ents(
        Entity(text="张伟", label="人名或姓名", score=0.91, start=0, end=2)
    )
    entities, _ = svc.extract("张伟加入 Google。", [], 0.4, language="mixed")
    texts = {e.text for e in entities}
    assert {"Google", "张伟"} <= texts
    svc._en_backend.predict.assert_called_once()
    svc._zh_backend.predict.assert_called_once()


def test_merge_deduplicates_overlapping_spans():
    """两个模型对同一 span 都命中 → 保留得分最高的那条。"""
    svc = _build_svc()
    svc._en_backend.predict.return_value = (
        [Entity(text="张伟", label="person", score=0.70, start=0, end=2)],
        ["person"],
    )
    svc._zh_backend.predict.return_value = (
        [Entity(text="张伟", label="人名或姓名", score=0.92, start=0, end=2)],
        ["人名或姓名"],
    )
    entities, _ = svc.extract("张伟", [], 0.4, language="mixed")
    matches = [e for e in entities if e.text == "张伟"]
    assert len(matches) == 1
    assert matches[0].score == 0.92          # 高分胜出
