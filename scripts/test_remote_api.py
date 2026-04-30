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

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    w("## 一、汇总\n\n")
    w("| 用例 | 描述 | HTTP | 实体数 | 召回 | 耗时 |\n")
    w("|---|---|---|---|---|---|\n")
    total_ms = 0.0
    pass_n = 0
    for case, res in results:
        expected = case.get("expected", set())
        found = {e["text"] for e in res.entities}
        hit = len(expected & found)
        recall = f"{hit}/{len(expected)}" if expected else "—"
        ok_mark = "✓" if res.status == 200 else "✗"
        w(f"| **{case['id']}** | {case['description']} | {ok_mark} {res.status} | "
          f"{len(res.entities)} | {recall} | {res.elapsed_ms:.0f}ms |\n")
        if res.status == 200:
            pass_n += 1
        total_ms += res.elapsed_ms
    w(f"\n- 通过率：**{pass_n}/{len(results)}**\n")
    w(f"- 累计耗时：**{total_ms:.0f}ms**（平均 {total_ms/len(results):.0f}ms/请求）\n\n")

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

            expected = case.get("expected", set())
            if expected:
                found = {e["text"] for e in res.entities}
                hits   = expected & found
                misses = expected - found
                w(f"\n**期望命中** {len(hits)}/{len(expected)}：")
                w(", ".join(f"`{x}`" for x in expected) + "  \n")
                if misses:
                    w(f"**未命中**：{', '.join(f'`{x}`' for x in misses)}  \n")

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
