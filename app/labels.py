"""
双语标签管理模块
────────────────
* DEFAULT_LABELS   — 内置通用双语标签集（labels 为空时使用）
* expand_bilingual — 自动为已有标签补充对等的另一语言版本
"""

# 英文 ↔ 中文 对照表（顺序决定输出标签的语言偏好）
_PAIRS: list[tuple[str, str]] = [
    ("full name of a person",             "人名或姓名"),
    ("company or organization name",      "公司或组织机构名称"),
    ("geographical location",             "地名或城市"),
    ("product or technology name",        "产品或技术名称"),
    ("date or year",                      "日期或年份"),
    ("hospital or medical institution",   "医院或医疗机构名称"),
    ("university or research institution","大学或研究机构"),
    ("project or initiative name",        "项目或计划名称"),
    ("legislation or policy name",        "法规或政策名称"),
    ("monetary amount",                   "金额或货币"),
    ("job title or position",             "职位或头衔"),
    ("event name",                        "事件或活动名称"),
]

# 默认标签集：英中并列，涵盖最常用实体类型
DEFAULT_LABELS: list[str] = [item for pair in _PAIRS for item in pair]

# 快速查找：任意一种语言的标签 → 对等标签
_EN_TO_ZH: dict[str, str] = {en: zh for en, zh in _PAIRS}
_ZH_TO_EN: dict[str, str] = {zh: en for en, zh in _PAIRS}


def expand_bilingual(labels: list[str]) -> list[str]:
    """
    为调用者传入的标签自动补充另一语言的对等描述。

    例如：
        ["人名或姓名", "company or organization name"]
        →  ["人名或姓名", "full name of a person",
            "company or organization name", "公司或组织机构名称"]

    规则：
    * 已有标签保持原位不变
    * 对等标签紧随其后插入（若已存在则跳过）
    * 未在对照表中的自定义标签原样保留，不做处理
    """
    seen: set[str] = set(labels)
    result: list[str] = []
    for lbl in labels:
        result.append(lbl)
        counterpart = _EN_TO_ZH.get(lbl) or _ZH_TO_EN.get(lbl)
        if counterpart and counterpart not in seen:
            result.append(counterpart)
            seen.add(counterpart)
    return result
