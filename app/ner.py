import unicodedata

from gliner import GLiNER

from app.labels import DEFAULT_LABELS, expand_bilingual
from app.models import Entity


# ── 语言检测 ──────────────────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """
    通过 Unicode 脚本比例判断文本语言。
    返回: 'zh' | 'ar' | 'mixed' | 'en'
    """
    if not text:
        return "en"

    cjk = arabic = letters = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            letters += 1
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or   # CJK Unified
                    0x3400 <= cp <= 0x4DBF or
                    0xF900 <= cp <= 0xFAFF or
                    0x20000 <= cp <= 0x2A6DF):
                cjk += 1
            elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
                arabic += 1

    if not letters:
        return "en"
    cjk_r = cjk / letters
    ar_r  = arabic / letters
    if cjk_r >= 0.20 and ar_r < 0.08:
        return "zh"
    if ar_r >= 0.20 and cjk_r < 0.08:
        return "ar"
    if cjk_r >= 0.08 or ar_r >= 0.08:
        return "mixed"
    return "en"


# ── Span 去重 ─────────────────────────────────────────────────────────────────

def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """
    双语标签可能对同一 (start, end) 跨度产生两条结果，
    保留置信度最高的那条，并按位置排序。
    """
    best: dict[tuple[int, int], Entity] = {}
    for e in entities:
        key = (e.start, e.end)
        if key not in best or e.score > best[key].score:
            best[key] = e
    return sorted(best.values(), key=lambda x: x.start)


# ── NER 服务 ──────────────────────────────────────────────────────────────────

class NERService:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model = GLiNER.from_pretrained(model_name, cache_dir=cache_dir)

    def extract(
        self,
        text: str,
        labels: list[str],
        threshold: float,
        language: str = "auto",
    ) -> tuple[list[Entity], list[str]]:
        """
        返回 (entities, labels_used)。

        labels 处理逻辑：
          1. labels 为空 → 使用内置双语默认标签集
          2. labels 非空 → 自动补充双语对等标签（提升中文召回）

        threshold 处理逻辑：
          - language='auto' 时自动检测
          - 中文 / 混合文本若传入默认 threshold(0.4) 则不调整（已足够低）
        """
        if not text:
            return [], labels

        # 确定有效语言
        eff_lang = language if language != "auto" else _detect_language(text)

        # 确定标签集
        if not labels:
            eff_labels = DEFAULT_LABELS
        else:
            eff_labels = expand_bilingual(labels)

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
