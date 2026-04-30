"""
NER 服务层 — 双模型路由 + 兜底策略
──────────────────────────────────────────────────────────────────────────────
语言检测（两层）：
  1. Unicode 脚本比例：快速，适合中文 / 阿拉伯文等脚本明显的语言
  2. langdetect 库兜底：覆盖纯英文及边界文本

路由 & 兜底规则：
  ┌──────────┬──────────────────┬──────────────────────────────┐
  │ language │ 主模型           │ 兜底条件                     │
  ├──────────┼──────────────────┼──────────────────────────────┤
  │ zh       │ ChineseBERT      │ 实体数=0 → 补充 GLiNER 结果  │
  │ en / ar  │ GLiNER           │ 实体数=0 → 补充 BERT 结果    │
  │ mixed    │ GLiNER + BERT    │ 同时运行两个模型，结果合并   │
  │ auto     │ 先检测语言再路由 │                              │
  └──────────┴──────────────────┴──────────────────────────────┘
"""

import threading
import unicodedata
from abc import ABC, abstractmethod

from gliner import GLiNER

from app.labels import (
    DEFAULT_LABELS,
    BERT_TYPE_TO_LABEL,
    expand_bilingual,
    labels_to_bert_types,
)
from app.models import Entity


# ── 语言检测 ──────────────────────────────────────────────────────────────────
#
# 两层策略：
#   Layer-1  Unicode 脚本比例
#     · 遍历文本中所有字母字符，统计 CJK / Arabic 脚本占比
#     · 优点：零依赖、极快；缺点：对极短或纯拉丁文本判断力弱
#
#   Layer-2  langdetect（仅 Layer-1 返回 'en' 时作为校验）
#     · 基于 n-gram 概率模型，原理同 Google CLD2
#     · 对短文本（<20 字）仍有一定误判率，以 Layer-1 为主
#     · 若 langdetect 检测到中文/日文/韩文 → 返回 'zh'
#     · 失败时静默回退到 Layer-1 结果

def _unicode_script_ratio(text: str) -> str:
    """Layer-1：基于 Unicode 脚本比例的语言分类。"""
    cjk = arabic = letters = 0
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        letters += 1
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF):
            cjk += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            arabic += 1
    if not letters:
        return "en"
    cjk_r = cjk / letters
    ar_r  = arabic / letters
    latin_r = (letters - cjk - arabic) / letters

    # 中文+拉丁都显著 → mixed（优先级高于单纯 zh 判断）
    if cjk_r >= 0.08 and latin_r >= 0.10:
        return "mixed"
    # 阿拉伯+拉丁都显著 → mixed
    if ar_r >= 0.08 and latin_r >= 0.10:
        return "mixed"
    # 单脚本主导
    if cjk_r >= 0.20:
        return "zh"
    if ar_r >= 0.20:
        return "ar"
    return "en"


def detect_language(text: str) -> str:
    """
    两层语言检测，返回 'zh' | 'ar' | 'mixed' | 'en'。

    Layer-1 优先（Unicode 脚本比例）；Layer-1 返回 'en' 时，
    用 langdetect 做一次二次确认，防止把中文误判为英文。
    """
    if not text:
        return "en"

    layer1 = _unicode_script_ratio(text)
    if layer1 != "en":          # 已明确是非英文，直接返回
        return layer1

    # Layer-2：langdetect 校验（仅对 Layer-1='en' 的文本）
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0    # 保证结果稳定
        lang_code = detect(text)    # e.g. 'zh-cn', 'ar', 'en', 'ja' …
        if lang_code.startswith("zh") or lang_code in ("ja", "ko"):
            return "zh"
        if lang_code == "ar":
            return "ar"
    except Exception:
        pass                        # langdetect 失败时静默回退

    return "en"


# ── Span 去重 ─────────────────────────────────────────────────────────────────

def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """
    双语标签或模型合并时可能产生同一 (start, end) 的重复结果，
    保留置信度最高的那条，并按起始位置排序。
    """
    best: dict[tuple[int, int], Entity] = {}
    for e in entities:
        key = (e.start, e.end)
        if key not in best or e.score > best[key].score:
            best[key] = e
    return sorted(best.values(), key=lambda x: x.start)


# ── 后端基类 ──────────────────────────────────────────────────────────────────

class _Backend(ABC):
    @abstractmethod
    def predict(
        self, text: str, labels: list[str], threshold: float
    ) -> tuple[list[Entity], list[str]]:
        """返回 (entities, labels_used)"""


# ── GLiNER 后端（英文 / 阿拉伯文 / 混合） ─────────────────────────────────────

class GLiNERBackend(_Backend):
    """
    零样本 NER：urchade/gliner_multi-v2.1
    • 支持英文、阿拉伯文及混合文本
    • 自动做双语标签扩展，提升召回率
    """

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model = GLiNER.from_pretrained(model_name, cache_dir=cache_dir)

    def predict(
        self, text: str, labels: list[str], threshold: float
    ) -> tuple[list[Entity], list[str]]:
        eff_labels = expand_bilingual(labels) if labels else DEFAULT_LABELS
        raw = self._model.predict_entities(text, eff_labels, threshold=threshold)
        entities = [
            Entity(
                text=e["text"],
                label=e["label"],
                score=round(e["score"], 4),
                start=e["start"],
                end=e["end"],
            )
            for e in raw
        ]
        return _deduplicate(entities), eff_labels


# ── 中文 BERT 后端 ─────────────────────────────────────────────────────────────

class ChineseBERTBackend(_Backend):
    """
    专用中文 NER：shibing624/bert4ner-base-chinese
    • 模型大小：~400 MB（BERT-base）
    • 推理速度：~100 ms
    • 固定实体类型：PER / LOC / ORG / TIME → 映射为双语标签
    • 用户传入标签时按标签类型过滤；无法映射的自定义标签不过滤（返回全部）
    """

    def __init__(self, model_name: str, cache_dir: str) -> None:
        # 延迟导入：避免顶层 import 在测试收集阶段触发 torch.__spec__ 检测
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            "token-classification",
            model=model_name,
            model_kwargs={"cache_dir": cache_dir},
            aggregation_strategy="simple",
        )

    def predict(
        self, text: str, labels: list[str], threshold: float
    ) -> tuple[list[Entity], list[str]]:
        raw = self._pipe(text)
        allowed_types = labels_to_bert_types(labels)   # None = 不过滤

        entities: list[Entity] = []
        labels_seen: set[str] = set()

        for r in raw:
            score = float(r["score"])
            if score < threshold:
                continue

            bert_type = r.get("entity_group", r.get("entity", ""))
            bert_type = bert_type.lstrip("BI-").strip()  # 去掉可能的 B-/I- 前缀

            if allowed_types is not None and bert_type not in allowed_types:
                continue

            std_label = BERT_TYPE_TO_LABEL.get(bert_type, bert_type)
            labels_seen.add(std_label)
            entities.append(Entity(
                text=r["word"],
                label=std_label,
                score=round(score, 4),
                start=r["start"],
                end=r["end"],
            ))

        used = list(labels_seen) if labels_seen else list(BERT_TYPE_TO_LABEL.values())
        return entities, used


# ── NER 服务（路由 + 兜底） ────────────────────────────────────────────────────

class NERService:
    """
    持有两个后端，按检测到的语言分发请求。

    兜底规则（召回为空时）：
      zh   主模型 BERT 无结果 → 用 GLiNER 补充
      en/ar 主模型 GLiNER 无结果 → 用 BERT 补充
      mixed 同时运行两个模型，合并去重后返回
    """

    def __init__(self, en_model_name: str, zh_model_name: str, cache_dir: str) -> None:
        self._en_name = en_model_name
        self._zh_name = zh_model_name
        self._cache_dir = cache_dir

        self._en_backend: GLiNERBackend | None = None
        self._zh_backend: ChineseBERTBackend | None = None
        self._en_lock = threading.Lock()
        self._zh_lock = threading.Lock()

    # ── 懒加载 ────────────────────────────────────────────────────────────────

    def _en(self) -> GLiNERBackend:
        if self._en_backend is None:
            with self._en_lock:
                if self._en_backend is None:
                    self._en_backend = GLiNERBackend(self._en_name, self._cache_dir)
        return self._en_backend

    def _zh(self) -> ChineseBERTBackend:
        if self._zh_backend is None:
            with self._zh_lock:
                if self._zh_backend is None:
                    self._zh_backend = ChineseBERTBackend(self._zh_name, self._cache_dir)
        return self._zh_backend

    # ── 兜底合并 ──────────────────────────────────────────────────────────────

    def _merge(
        self,
        primary: tuple[list[Entity], list[str]],
        fallback: tuple[list[Entity], list[str]],
    ) -> tuple[list[Entity], list[str]]:
        """合并两个模型的结果，去重后按位置排序。"""
        p_ents, p_labels = primary
        f_ents, f_labels = fallback
        merged = _deduplicate(p_ents + f_ents)
        used = list(dict.fromkeys(p_labels + f_labels))  # 保序去重
        return merged, used

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        labels: list[str],
        threshold: float,
        language: str = "auto",
    ) -> tuple[list[Entity], list[str]]:
        """
        返回 (entities, labels_used)。

        路由逻辑：
          auto  → 检测语言 → 路由
          zh    → BERT 主，GLiNER 兜底（主模型无结果时补充）
          en/ar → GLiNER 主，BERT 兜底（主模型无结果时补充）
          mixed → 两模型同时运行，结果合并去重
        """
        if not text:
            return [], labels

        lang = language if language != "auto" else detect_language(text)

        if lang == "mixed":
            # 同时运行两个模型，合并结果
            en_result = self._en().predict(text, labels, threshold)
            zh_result = self._zh().predict(text, labels, threshold)
            return self._merge(en_result, zh_result)

        if lang == "zh":
            primary_result = self._zh().predict(text, labels, threshold)
            if not primary_result[0]:       # 主模型无结果 → GLiNER 兜底
                fallback_result = self._en().predict(text, labels, threshold)
                if fallback_result[0]:
                    return fallback_result
            return primary_result

        # en / ar / 其他
        primary_result = self._en().predict(text, labels, threshold)
        if not primary_result[0]:           # 主模型无结果 → BERT 兜底
            fallback_result = self._zh().predict(text, labels, threshold)
            if fallback_result[0]:
                return fallback_result
        return primary_result

    def warmup(self) -> None:
        """启动时预热两个模型，首个请求无需等待。"""
        self._en()
        self._zh()
