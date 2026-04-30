"""
对远端 HF Spaces 上部署的 NER API 做端到端测试，覆盖所有路由分支与边界情况。
为每个用例记录：HTTP 状态、识别到的实体、调用耗时、自动检测的语言（如有）。
最终输出 Markdown 报告：reports/remote_api_test_report.md
"""
import io
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://robinwu-nerserver.hf.space"
EXTRACT  = f"{BASE_URL}/api/v1/extract"
HEALTH   = f"{BASE_URL}/api/v1/health"
REPORT   = Path("reports/remote_api_test_report.md")


# ── 用例定义 ──────────────────────────────────────────────────────────────────
#
# 每个用例字段：
#   id          短编号
#   group       分组（用于报告分类）
#   description 中文描述
#   payload     传给 /api/v1/extract 的 JSON
#   expected    期望命中的实体文本（用于召回率统计；可为空集合表示不校验）

CASES: list[dict] = [
    # ── EN 路由 ──
    {
        "id": "EN-01", "group": "EN — GLiNER 主路径",
        "description": "英文短句，显式 language=en，自定义标签",
        "payload": {
            "text": "Elon Musk founded SpaceX in Hawthorne, California in 2002.",
            "labels": ["full name of a person", "company or organization name",
                       "geographical location", "date or year"],
            "language": "en",
        },
        "expected": {"Elon Musk", "SpaceX", "Hawthorne", "California", "2002"},
    },
    {
        "id": "EN-02", "group": "EN — GLiNER 主路径",
        "description": "英文长段，labels 留空触发默认双语标签集",
        "payload": {
            "text": ("President Biden signed the Inflation Reduction Act in "
                     "Washington D.C. on August 16, 2022. The legislation was "
                     "championed by Senator Chuck Schumer and was seen as a major "
                     "win for the Democratic Party."),
            "language": "en",
        },
        "expected": {"Biden", "Chuck Schumer", "Washington D.C.", "Democratic Party"},
    },
    # ── ZH 路由 ──
    {
        "id": "ZH-01", "group": "ZH — BERT 主路径",
        "description": "中文现代商业文本，显式 language=zh",
        "payload": {
            "text": "阿里巴巴集团创始人马云于2019年卸任董事局主席，由张勇接任。"
                    "总部位于杭州的阿里巴巴旗下拥有淘宝、天猫、支付宝等业务板块。",
            "language": "zh",
        },
        "expected": {"马云", "张勇", "阿里巴巴", "杭州"},
    },
    {
        "id": "ZH-02", "group": "ZH — BERT 主路径",
        "description": "中文医疗场景，自定义双语标签",
        "payload": {
            "text": "北京协和医院心内科主任王建国教授团队，于2023年成功完成首例"
                    "机器人辅助冠状动脉搭桥手术，患者来自山东省济南市。",
            "labels": ["人名或姓名", "医院或医疗机构名称", "地名或城市", "日期或年份"],
            "language": "zh",
        },
        "expected": {"王建国", "北京协和医院", "济南"},
    },
    {
        "id": "ZH-03", "group": "ZH — BERT 边界识别",
        "description": "古典文学边界测试 — 「尤氏来请」应只取「尤氏」",
        "payload": {
            "text": "尤氏来请，王熙凤笑道：你来了。贾母命人摆酒，宝玉和黛玉在大观园散步。",
            "language": "zh",
        },
        "expected": {"尤氏", "王熙凤", "贾母", "宝玉", "黛玉", "大观园"},
        "must_not_contain": {"尤氏来请", "王熙凤笑道"},
    },
    # ── AR 路由 ──
    {
        "id": "AR-01", "group": "AR — GLiNER 主路径",
        "description": "阿拉伯语新闻",
        "payload": {
            "text": ("أعلن الرئيس محمد بن سلمان عن إطلاق مشروع نيوم في المملكة "
                     "العربية السعودية عام 2017، وتبلغ تكلفته 500 مليار دولار."),
            "labels": ["full name of a person", "geographical location",
                       "project or initiative name", "date or year"],
            "language": "ar",
        },
        "expected": {"محمد بن سلمان", "المملكة العربية السعودية"},
    },
    # ── Mixed 路由（双跑合并） ──
    {
        "id": "MIX-01", "group": "Mixed — 双模型合并",
        "description": "中英混合 · 职场场景，language=mixed 强制双跑",
        "payload": {
            "text": "张伟加入了 Google 北京研发中心，负责 Android 系统优化。"
                    "他的同事 Sarah Chen 来自 Meta，两人共同参与了 2024 年的 AI Summit。",
            "language": "mixed",
        },
        "expected": {"张伟", "Google", "Sarah Chen", "Meta", "Android", "北京", "2024"},
    },
    {
        "id": "MIX-02", "group": "Mixed — 双模型合并",
        "description": "学术场景，labels 留空",
        "payload": {
            "text": "清华大学计算机系教授李明在 NeurIPS 2023 发表了关于 "
                    "Transformer 架构的论文，合作者来自 MIT 和 Stanford University。",
            "language": "mixed",
        },
        "expected": {"李明", "清华大学", "MIT", "Stanford University", "Transformer"},
    },
    # ── auto 自动检测 ──
    {
        "id": "AUTO-01", "group": "auto — 自动语言检测",
        "description": "纯中文文本，应被检测为 zh",
        "payload": {
            "text": "马云创立了阿里巴巴，总部在杭州。",
        },
        "expected": {"马云", "阿里巴巴", "杭州"},
    },
    {
        "id": "AUTO-02", "group": "auto — 自动语言检测",
        "description": "纯英文文本，应被检测为 en",
        "payload": {
            "text": "Tim Cook is the CEO of Apple in Cupertino.",
        },
        "expected": {"Tim Cook", "Apple", "Cupertino"},
    },
    {
        "id": "AUTO-03", "group": "auto — 自动语言检测",
        "description": "中英混合，应被检测为 mixed 并双跑合并",
        "payload": {
            "text": "李华在 Microsoft 担任工程师，常驻 Seattle 办公室。",
        },
        "expected": {"李华", "Microsoft", "Seattle"},
    },
    # ── min_entities 覆盖 ──
    {
        "id": "MIN-01", "group": "min_entities 覆盖启发式",
        "description": "min_entities=10 强制兜底（短文本启发式只期望 1 个）",
        "payload": {
            "text": "马云",
            "language": "zh",
            "min_entities": 10,
        },
        "expected": {"马云"},
    },
    {
        "id": "MIN-02", "group": "min_entities 覆盖启发式",
        "description": "min_entities=0 关闭兜底",
        "payload": {
            "text": "马云",
            "language": "zh",
            "min_entities": 0,
        },
        "expected": {"马云"},
    },
    # ── 阈值变化 ──
    {
        "id": "THR-01", "group": "Threshold 变化",
        "description": "高阈值 0.8 - 期望返回更少但更高置信度的实体",
        "payload": {
            "text": "Tesla and SpaceX are companies founded by Elon Musk.",
            "language": "en",
            "threshold": 0.8,
        },
        "expected": {"Tesla", "SpaceX", "Elon Musk"},
    },
    # ── 边界请求 ──
    {
        "id": "EDGE-01", "group": "Edge cases",
        "description": "空文本",
        "payload": {"text": ""},
        "expected": set(),
    },
]


# ── HTTP 调用 + 计时 ──────────────────────────────────────────────────────────

@dataclass
class CallResult:
    case_id: str
    status: int
    elapsed_ms: float
    entities: list[dict] = field(default_factory=list)
    labels_used: list[str] = field(default_factory=list)
    error: str | None = None


def post_extract(payload: dict, timeout: int = 60) -> CallResult:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        EXTRACT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = (time.perf_counter() - t0) * 1000
            data = json.loads(resp.read().decode())
            return CallResult(
                case_id="",
                status=resp.status,
                elapsed_ms=elapsed,
                entities=data.get("entities", []),
                labels_used=data.get("labels_used", []),
            )
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return CallResult(case_id="", status=e.code, elapsed_ms=elapsed,
                          error=e.read().decode("utf-8", errors="replace"))
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return CallResult(case_id="", status=0, elapsed_ms=elapsed, error=str(e))


# ── 健康检查 ──────────────────────────────────────────────────────────────────

def check_health() -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(HEALTH, timeout=30) as resp:
            elapsed = (time.perf_counter() - t0) * 1000
            return resp.status == 200, elapsed, resp.read().decode()
    except Exception as e:
        return False, (time.perf_counter() - t0) * 1000, str(e)


# ── 三档匹配 ──────────────────────────────────────────────────────────────────
#
#   exact   完全相等                 e.g. "Biden" == "Biden"
#   partial 一方包含另一方           e.g. "President Biden" 包含 "Biden"
#                                         "济南" 被 "济南市" 包含
#   miss    都不满足
#
# 这是为了解决严格相等带来的"假阴性"——模型边界差一点也算未命中。

def match_one(expected: str, returned_texts: list[str]) -> tuple[str, str]:
    """返回 (level, matched_text)。level ∈ {'exact','partial','miss'}"""
    for r in returned_texts:
        if r == expected:
            return "exact", r
    for r in returned_texts:
        if expected in r or r in expected:
            return "partial", r
    return "miss", ""


@dataclass
class CaseMetrics:
    expected_n: int
    returned_n: int
    tp_exact: int = 0
    tp_partial: int = 0
    miss_list: list[str] = field(default_factory=list)
    matched_pairs: list[tuple[str, str, str]] = field(default_factory=list)  # (expected, level, returned)

    @property
    def tp(self) -> int:
        return self.tp_exact + self.tp_partial

    @property
    def precision(self) -> float:
        return self.tp / self.returned_n if self.returned_n else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.expected_n if self.expected_n else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(case: dict, res: CallResult) -> CaseMetrics | None:
    expected = case.get("expected", set())
    if not expected:
        return None
    returned_texts = [e["text"] for e in res.entities]
    m = CaseMetrics(expected_n=len(expected), returned_n=len(returned_texts))
    for exp in expected:
        level, found = match_one(exp, returned_texts)
        if level == "exact":
            m.tp_exact += 1
            m.matched_pairs.append((exp, "exact", found))
        elif level == "partial":
            m.tp_partial += 1
            m.matched_pairs.append((exp, "partial", found))
        else:
            m.miss_list.append(exp)
    return m


# ── 报告生成 ──────────────────────────────────────────────────────────────────

def write_report(results: list[tuple[dict, CallResult]], health: tuple[bool, float, str]):
    buf = io.StringIO()
    w = buf.write

    w("# 远端 API 测试报告\n\n")
    w(f"- 服务地址：`{BASE_URL}`\n")
    w(f"- 测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    ok, hms, hbody = health
    w(f"- 健康检查：{'✓ OK' if ok else '✗ FAIL'} ({hms:.0f}ms) — {hbody}\n")
    w(f"- 用例总数：{len(results)}\n\n")

    # ── 评测说明 ──────────────────────────────────────────────────────────────
    w("## 评测口径\n\n")
    w("- **返回数**：API 实际返回的实体个数（输出量）\n")
    w("- **期望数**：用例作者预先列出的正确答案个数（标准答案）\n")
    w("- **命中**：把每个期望项分到三档之一\n")
    w("    - `exact` ：完全相等（如 `Biden` == `Biden`）\n")
    w("    - `partial`：一方包含另一方（如 `President Biden` 包含 `Biden`、`济南` 被 `济南市` 包含），算半对，**仍计入 TP**\n")
    w("    - `miss`  ：两种都不满足\n")
    w("- **指标公式**\n")
    w("    - Precision = TP / 返回数  （模型说的话有多少是有效的）\n")
    w("    - Recall    = TP / 期望数  （应该说的话说了多少）\n")
    w("    - F1        = 2·P·R / (P+R)\n\n")

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    w("## 一、汇总\n\n")
    w("| 用例 | 描述 | HTTP | 返回数 | 期望数 | TP(精/部) | P | R | F1 | 耗时 |\n")
    w("|---|---|---|---:|---:|---|---:|---:|---:|---:|\n")
    total_ms = 0.0
    pass_n = 0
    aggregate = {"tp_exact": 0, "tp_partial": 0, "returned": 0, "expected": 0}
    for case, res in results:
        ok_mark = "✓" if res.status == 200 else "✗"
        m = evaluate(case, res)
        if m is None:
            tp_cell = "—"
            p_cell = r_cell = f1_cell = "—"
            ret_n = len(res.entities)
            exp_n = "—"
        else:
            tp_cell = f"{m.tp} ({m.tp_exact}/{m.tp_partial})"
            p_cell  = f"{m.precision*100:.0f}%"
            r_cell  = f"{m.recall*100:.0f}%"
            f1_cell = f"{m.f1*100:.0f}%"
            ret_n = m.returned_n
            exp_n = m.expected_n
            aggregate["tp_exact"]   += m.tp_exact
            aggregate["tp_partial"] += m.tp_partial
            aggregate["returned"]   += m.returned_n
            aggregate["expected"]   += m.expected_n
        w(f"| **{case['id']}** | {case['description']} | {ok_mark} {res.status} | "
          f"{ret_n} | {exp_n} | {tp_cell} | {p_cell} | {r_cell} | {f1_cell} | "
          f"{res.elapsed_ms:.0f}ms |\n")
        if res.status == 200:
            pass_n += 1
        total_ms += res.elapsed_ms

    # 整体微平均
    tp_total = aggregate["tp_exact"] + aggregate["tp_partial"]
    micro_p  = tp_total / aggregate["returned"] if aggregate["returned"] else 0.0
    micro_r  = tp_total / aggregate["expected"] if aggregate["expected"] else 0.0
    micro_f1 = 2*micro_p*micro_r / (micro_p+micro_r) if (micro_p+micro_r) else 0.0
    w(f"\n- 通过率：**{pass_n}/{len(results)}**（HTTP 200）\n")
    w(f"- 累计耗时：**{total_ms:.0f}ms**（平均 {total_ms/len(results):.0f}ms/请求）\n")
    w(f"- 整体微平均：**P={micro_p*100:.0f}%  R={micro_r*100:.0f}%  F1={micro_f1*100:.0f}%**\n")
    w(f"  （TP={tp_total}（精确 {aggregate['tp_exact']} + 部分 {aggregate['tp_partial']}），"
      f"返回总 {aggregate['returned']}，期望总 {aggregate['expected']}）\n\n")

    # ── 分组详情 ──────────────────────────────────────────────────────────────
    groups: dict[str, list] = {}
    for case, res in results:
        groups.setdefault(case["group"], []).append((case, res))

    w("## 二、分组详细结果\n\n")
    for group_name, items in groups.items():
        w(f"### {group_name}\n\n")
        for case, res in items:
            w(f"#### {case['id']} · {case['description']}\n\n")
            w("**请求**\n```json\n")
            w(json.dumps(case["payload"], ensure_ascii=False, indent=2))
            w("\n```\n\n")

            w(f"**响应**：HTTP {res.status} · {res.elapsed_ms:.0f}ms · "
              f"{len(res.entities)} 个实体\n\n")

            if res.error:
                w(f"```\nERROR: {res.error}\n```\n\n")
                continue

            if res.entities:
                w("| 文本 | 标签 | 置信度 | 起止 |\n|---|---|---|---|\n")
                for e in res.entities:
                    w(f"| `{e['text']}` | {e['label']} | {e['score']:.2f} | "
                      f"{e['start']}–{e['end']} |\n")
            else:
                w("_未识别到实体_\n")

            m = evaluate(case, res)
            if m is not None:
                w(f"\n**指标**：返回 {m.returned_n}，期望 {m.expected_n}，"
                  f"TP={m.tp}（exact={m.tp_exact}，partial={m.tp_partial}）  \n")
                w(f"**P / R / F1** = {m.precision*100:.0f}% / "
                  f"{m.recall*100:.0f}% / {m.f1*100:.0f}%  \n\n")

                if m.matched_pairs:
                    w("**命中明细**\n\n| 期望 | 档位 | 实际命中 |\n|---|---|---|\n")
                    for exp, level, found in m.matched_pairs:
                        icon = "✓" if level == "exact" else "≈"
                        w(f"| `{exp}` | {icon} {level} | `{found}` |\n")
                if m.miss_list:
                    w(f"\n**未命中**：{', '.join(f'`{x}`' for x in m.miss_list)}  \n")

            mnc = case.get("must_not_contain", set())
            if mnc:
                bad = {e["text"] for e in res.entities} & mnc
                if bad:
                    w(f"\n> ⚠️ **边界错误**：{bad}\n")
                else:
                    w(f"\n> ✓ 边界正确（未出现 {mnc}）\n")
            w("\n")

    # ── 性能聚合 ──────────────────────────────────────────────────────────────
    w("## 三、按路由分组性能\n\n")
    by_group: dict[str, list[float]] = {}
    for case, res in results:
        if res.status == 200:
            by_group.setdefault(case["group"], []).append(res.elapsed_ms)
    w("| 分组 | 用例数 | 最快 | 最慢 | 平均 |\n|---|---|---|---|---|\n")
    for g, times in by_group.items():
        w(f"| {g} | {len(times)} | {min(times):.0f}ms | "
          f"{max(times):.0f}ms | {sum(times)/len(times):.0f}ms |\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nReport: {REPORT.resolve()}")


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    print(f"Target: {BASE_URL}")
    health = check_health()
    print(f"Health: {'OK' if health[0] else 'FAIL'} ({health[1]:.0f}ms)")
    if not health[0]:
        print(f"  -> {health[2]}")
        return

    results: list[tuple[dict, CallResult]] = []
    for case in CASES:
        print(f"  {case['id']:8s}  ", end="", flush=True)
        res = post_extract(case["payload"])
        res.case_id = case["id"]
        results.append((case, res))
        status = "OK" if res.status == 200 else f"FAIL({res.status})"
        print(f"{status:8s}  {res.elapsed_ms:6.0f}ms  {len(res.entities)} entities")

    write_report(results, health)


if __name__ == "__main__":
    main()
