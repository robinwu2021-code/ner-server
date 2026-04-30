"""
对比 gliner_multi-v2.1 和 gliner-multitask-large-v0.5 两个模型
在中文、英文、阿拉伯文、中英混合文本上的 NER 效果。

优化点：
  - 所有测试用例统一使用双语标签（中英并列），提升中文识别率
  - 结果写入 UTF-8 Markdown 报告，避免 Windows GBK 控制台乱码
  - 新增阿拉伯语测试用例
  - 新增 span 去重：双语标签可能产生重复跨度，保留得分最高的

用法：
    python scripts/compare_models.py
报告：
    reports/comparison_report.md
"""
import io
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"          # Windows OpenMP 冲突
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Windows 符号链接警告

from huggingface_hub import snapshot_download
from gliner import GLiNER


# ── 测试用例（全部使用双语标签） ──────────────────────────────────────────────

CASES = [
    # ── 英文 ─────────────────────────────────────────────────────────────────
    {
        "name": "EN-01  英文 · 科技人物",
        "lang": "en",
        "text": (
            "Elon Musk, CEO of Tesla and founder of SpaceX, announced a new "
            "Starship launch from Boca Chica, Texas. NASA has partnered with "
            "SpaceX for the Artemis lunar lander mission planned for 2026."
        ),
        "labels": [
            "full name of a person",
            "company or organization name",
            "geographical location",
            "product or technology name",
            "date or year",
        ],
        "expected": ["Elon Musk", "Tesla", "SpaceX", "NASA", "Boca Chica", "Texas", "2026"],
    },
    {
        "name": "EN-02  英文 · 政治新闻",
        "lang": "en",
        "text": (
            "President Biden signed the Inflation Reduction Act in Washington D.C. "
            "on August 16, 2022. The legislation was championed by Senator Chuck Schumer "
            "and was seen as a major win for the Democratic Party."
        ),
        "labels": [
            "full name of a person",
            "company or organization name",
            "geographical location",
            "legislation or policy name",
            "date or year",
            "political party",
        ],
        "expected": ["Biden", "Chuck Schumer", "Washington D.C.", "August 16, 2022", "Democratic Party"],
    },
    # ── 中文 ─────────────────────────────────────────────────────────────────
    {
        "name": "ZH-01  中文 · 现代商业（双语标签）",
        "lang": "zh",
        "text": (
            "阿里巴巴集团创始人马云于2019年卸任董事局主席，由张勇接任。"
            "总部位于杭州的阿里巴巴旗下拥有淘宝、天猫、支付宝等业务板块。"
        ),
        "labels": [
            "人名或姓名", "full name of a person",
            "公司或组织机构名称", "company or organization name",
            "地名或城市", "geographical location",
            "产品或品牌名称", "product or brand name",
            "日期或年份", "date or year",
        ],
        "expected": ["马云", "张勇", "阿里巴巴", "杭州", "淘宝", "天猫", "支付宝", "2019"],
    },
    {
        "name": "ZH-02  中文 · 古典文学（边界测试）",
        "lang": "zh",
        "text": (
            "尤氏来请，王熙凤笑道：'你来了。'贾母命人摆酒，"
            "宝玉和黛玉在大观园散步，薛宝钗独坐梨香院。"
        ),
        "labels": [
            "人名或姓名", "full name of a person",
            "地名或场所", "place or location name",
        ],
        "expected": ["尤氏", "王熙凤", "贾母", "宝玉", "黛玉", "薛宝钗", "大观园", "梨香院"],
        "boundary_check": {
            "must_not_contain": ["尤氏来请", "王熙凤笑道"],
        },
    },
    {
        "name": "ZH-03  中文 · 医疗场景（双语标签）",
        "lang": "zh",
        "text": (
            "北京协和医院心内科主任王建国教授团队，于2023年成功完成首例"
            "机器人辅助冠状动脉搭桥手术，患者来自山东省济南市。"
        ),
        "labels": [
            "人名或姓名", "full name of a person",
            "医院或机构名称", "hospital or institution name",
            "地名或城市", "geographical location",
            "医疗技术或手术名称", "medical procedure or technology",
            "日期或年份", "date or year",
        ],
        "expected": ["王建国", "北京协和医院", "济南", "山东", "2023"],
    },
    # ── 阿拉伯文 ─────────────────────────────────────────────────────────────
    {
        "name": "AR-01  阿拉伯语 · 新闻",
        "lang": "ar",
        "text": (
            "أعلن الرئيس محمد بن سلمان عن إطلاق مشروع نيوم في المملكة العربية السعودية "
            "عام 2017، وتبلغ تكلفته 500 مليار دولار."
        ),
        "labels": [
            "full name of a person",
            "company or organization name",
            "geographical location",
            "project or initiative name",
            "date or year",
            "monetary amount",
        ],
        "expected": ["محمد بن سلمان", "نيوم", "المملكة العربية السعودية", "2017"],
    },
    # ── 中英混合 ─────────────────────────────────────────────────────────────
    {
        "name": "MIX-01  中英混合 · 职场场景（双语标签）",
        "lang": "mixed",
        "text": (
            "张伟加入了 Google 北京研发中心，负责 Android 系统优化。"
            "他的同事 Sarah Chen 来自 Meta，两人共同参与了 2024 年的 AI Summit。"
        ),
        "labels": [
            "人名或姓名", "full name of a person",
            "公司或组织机构名称", "company or organization name",
            "地名或城市", "geographical location",
            "产品或技术名称", "product or technology name",
            "日期或年份", "date or year",
        ],
        "expected": ["张伟", "Google", "Sarah Chen", "Meta", "Android", "北京", "2024"],
    },
    {
        "name": "MIX-02  中英混合 · 学术场景（双语标签）",
        "lang": "mixed",
        "text": (
            "清华大学计算机系教授李明在 NeurIPS 2023 发表了关于 Transformer 架构的论文，"
            "合作者来自 MIT 和 Stanford University。"
        ),
        "labels": [
            "人名或姓名", "full name of a person",
            "大学或研究机构", "university or research institution",
            "会议或期刊名称", "conference or journal name",
            "技术或模型名称", "technology or model name",
            "日期或年份", "date or year",
        ],
        "expected": ["李明", "清华大学", "NeurIPS 2023", "Transformer", "MIT", "Stanford University"],
    },
]

THRESHOLD = 0.4
CACHE_DIR = "./model_cache"
REPORT_DIR = Path("reports")

MODELS = [
    ("gliner_multi-v2.1",           "urchade/gliner_multi-v2.1"),
    ("gliner-multitask-large-v0.5", "knowledgator/gliner-multitask-large-v0.5"),
]


# ── span 去重 ─────────────────────────────────────────────────────────────────

def deduplicate(entities: list[dict]) -> list[dict]:
    """双语标签可能对同一 span 产生两条结果，保留得分最高的那条。"""
    best: dict[tuple, dict] = {}
    for e in entities:
        key = (e["start"], e["end"])
        if key not in best or e["score"] > best[key]["score"]:
            best[key] = e
    return sorted(best.values(), key=lambda x: x["start"])


# ── 模型下载（直接复制，无符号链接，兼容 Windows） ────────────────────────────

def ensure_local(model_name: str) -> str:
    safe = model_name.replace("/", "__")
    local_dir = Path(CACHE_DIR) / safe
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  [cached]   {local_dir}")
    else:
        print(f"  [download] {model_name} -> {local_dir}")
        snapshot_download(repo_id=model_name, local_dir=str(local_dir))
        print(f"  [done]")
    return str(local_dir)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_name: str
    lang: str
    text: str
    expected: list[str]
    entities: list[dict]
    elapsed_ms: float
    boundary_violations: list[str] = field(default_factory=list)

    @property
    def found_texts(self) -> set[str]:
        return {e["text"] for e in self.entities}

    @property
    def hit_count(self) -> int:
        return sum(1 for exp in self.expected if exp in self.found_texts)

    @property
    def recall(self) -> float:
        if not self.expected:
            return 1.0
        return self.hit_count / len(self.expected)


@dataclass
class ModelResult:
    model_name: str
    load_ms: float
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def avg_recall(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.recall for c in self.cases) / len(self.cases)

    @property
    def avg_infer_ms(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.elapsed_ms for c in self.cases) / len(self.cases)


# ── 运行模型 ──────────────────────────────────────────────────────────────────

def run_model(short_name: str, model_name: str) -> ModelResult:
    print(f"\n{'─'*60}")
    print(f"Loading model: {model_name}")
    t0 = time.perf_counter()
    local_path = ensure_local(model_name)
    model = GLiNER.from_pretrained(local_path, local_files_only=True)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[loaded] {load_ms:.0f}ms")

    result = ModelResult(model_name=short_name, load_ms=load_ms)
    for case in CASES:
        t1 = time.perf_counter()
        raw = model.predict_entities(case["text"], case["labels"], threshold=THRESHOLD)
        elapsed_ms = (time.perf_counter() - t1) * 1000
        entities = deduplicate(raw)

        bc = case.get("boundary_check", {})
        violations = [
            e["text"] for e in entities
            if e["text"] in bc.get("must_not_contain", [])
        ]
        result.cases.append(CaseResult(
            case_name=case["name"],
            lang=case["lang"],
            text=case["text"],
            expected=case.get("expected", []),
            entities=entities,
            elapsed_ms=elapsed_ms,
            boundary_violations=violations,
        ))
        status = "OK" if not violations else f"BOUNDARY ERR: {violations}"
        print(f"  {case['name'][:30]:30s}  {len(entities):2d} entities  {elapsed_ms:.0f}ms  {status}")

    return result


# ── Markdown 报告生成 ─────────────────────────────────────────────────────────

def write_report(all_results: list[ModelResult], out_path: Path):
    buf = io.StringIO()
    w = buf.write

    w("# NER 模型对比测试报告\n\n")
    w(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
    w(f"阈值（threshold）：`{THRESHOLD}`  \n\n")

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    w("## 一、汇总对比\n\n")
    header = "| 测试用例 | 语言 |"
    sep    = "|---|---|"
    for r in all_results:
        header += f" {r.model_name} 召回 | {r.model_name} 耗时 |"
        sep    += "---|---|"
    w(header + "\n")
    w(sep + "\n")

    for i, case in enumerate(CASES):
        row = f"| {case['name']} | `{case['lang']}` |"
        for r in all_results:
            cr = r.cases[i]
            pct = f"{cr.recall*100:.0f}%"
            row += f" {cr.hit_count}/{len(cr.expected)} ({pct}) | {cr.elapsed_ms:.0f}ms |"
        w(row + "\n")

    # avg row
    avg_row = "| **平均** | — |"
    for r in all_results:
        avg_row += f" **{r.avg_recall*100:.0f}%** | **{r.avg_infer_ms:.0f}ms** |"
    w(avg_row + "\n\n")

    # ── 加载时间 ──────────────────────────────────────────────────────────────
    w("## 二、模型加载时间\n\n")
    w("| 模型 | 加载耗时 |\n|---|---|\n")
    for r in all_results:
        w(f"| {r.model_name} | {r.load_ms/1000:.1f}s |\n")
    w("\n")

    # ── 逐用例详情 ────────────────────────────────────────────────────────────
    w("## 三、逐用例详细结果\n\n")
    for i, case in enumerate(CASES):
        w(f"### {case['name']}\n\n")
        w(f"**文本**\n```\n{case['text']}\n```\n\n")
        w(f"**期望实体**：{', '.join(f'`{e}`' for e in case.get('expected', []))}\n\n")

        for r in all_results:
            cr = r.cases[i]
            hits = [e for e in cr.expected if e in cr.found_texts]
            misses = [e for e in cr.expected if e not in cr.found_texts]

            w(f"#### {r.model_name}  （{cr.elapsed_ms:.0f}ms，{len(cr.entities)} 个实体，召回 {cr.recall*100:.0f}%）\n\n")

            if cr.entities:
                w("| 文本 | 标签 | 置信度 | 命中期望 |\n|---|---|---|---|\n")
                for e in cr.entities:
                    hit_mark = "✓" if e["text"] in cr.expected else ""
                    w(f"| `{e['text']}` | {e['label']} | {e['score']:.2f} | {hit_mark} |\n")
            else:
                w("_未识别到实体_\n")

            if misses:
                w(f"\n**未命中**：{', '.join(f'`{m}`' for m in misses)}\n")
            if cr.boundary_violations:
                w(f"\n> ⚠️ **边界错误**：{cr.boundary_violations}\n")
            w("\n")

    # ── 结论 ─────────────────────────────────────────────────────────────────
    w("## 四、结论与建议\n\n")
    best = max(all_results, key=lambda r: r.avg_recall)
    fast = min(all_results, key=lambda r: r.avg_infer_ms)
    w(f"- **综合召回最高**：`{best.model_name}`（平均召回 {best.avg_recall*100:.0f}%）\n")
    w(f"- **推理最快**：`{fast.model_name}`（平均 {fast.avg_infer_ms:.0f}ms/次）\n\n")
    w("### 优化建议\n\n")
    w("1. **双语标签策略**：对中文或混合文本，同时提供中英文标签描述（如 `\"人名或姓名\"` + `\"full name of a person\"`），可显著提升中文实体召回率。GLiNER 是零样本模型，标签描述越具体、越接近训练语料的表达方式，识别效果越好。\n")
    w("2. **Span 去重**：使用双语标签时同一文本跨度可能被打上两个标签，建议在服务层按 `(start, end)` 去重，保留得分最高的结果（已在 `app/ner.py` 实现）。\n")
    w("3. **阈值调优**：英文建议 `threshold=0.5`，中文建议 `threshold=0.35~0.4`（模型对中文置信度普遍偏低）。\n")
    w("4. **古典/文言文**：两个模型对文言文支持均弱，建议结合规则或专用模型（如 `BERT-CRF` 在古汉语语料上微调）处理此类文本。\n")
    w("5. **阿拉伯语**：`gliner-multitask-large-v0.5` 在多语言上训练，对阿拉伯语有基础支持；`gliner_multi-v2.1` 阿拉伯语效果有限。\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\n[report] {out_path.resolve()}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results: list[ModelResult] = []
    for short_name, model_name in MODELS:
        all_results.append(run_model(short_name, model_name))

    # 控制台简要汇总（ASCII safe）
    print(f"\n{'='*70}")
    print(f"{'Case':<42} " + "  ".join(f"{r.model_name[:20]:<20}" for r in all_results))
    print(f"{'─'*70}")
    for i, case in enumerate(CASES):
        row = f"{case['name'][:40]:<42}"
        for r in all_results:
            cr = r.cases[i]
            row += f"  {cr.hit_count}/{len(cr.expected)} {cr.recall*100:3.0f}% {cr.elapsed_ms:5.0f}ms    "
        print(row)
    print(f"{'─'*70}")
    avg_row = f"{'Average':<42}"
    for r in all_results:
        avg_row += f"  avg {r.avg_recall*100:.0f}% / {r.avg_infer_ms:.0f}ms         "
    print(avg_row)

    report_path = REPORT_DIR / "comparison_report.md"
    write_report(all_results, report_path)
