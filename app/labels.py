"""
双语标签管理模块
────────────────────────────────────────────────────────────────
* DEFAULT_LABELS       — 内置通用双语标签集（labels 为空时使用）
* expand_bilingual     — 自动为已有标签补充对等的另一语言版本
* BERT_TYPE_TO_LABEL   — 中文 BERT 模型固定实体类型 → 标准双语标签
* LABEL_TO_BERT_TYPES  — 标准标签 → 对应 BERT 实体类型列表（用于过滤）
"""

# ── 英中对照表 ─────────────────────────────────────────────────────────────────
_PAIRS: list[tuple[str, str]] = [
    ("full name of a person",              "人名或姓名"),
    ("company or organization name",       "公司或组织机构名称"),
    ("geographical location",              "地名或城市"),
    ("product or technology name",         "产品或技术名称"),
    ("date or year",                       "日期或年份"),
    ("hospital or medical institution",    "医院或医疗机构名称"),
    ("university or research institution", "大学或研究机构"),
    ("project or initiative name",         "项目或计划名称"),
    ("legislation or policy name",         "法规或政策名称"),
    ("monetary amount",                    "金额或货币"),
    ("job title or position",              "职位或头衔"),
    ("event name",                         "事件或活动名称"),
]

# 默认标签集：英中并列
DEFAULT_LABELS: list[str] = [item for pair in _PAIRS for item in pair]

# 快速查找
_EN_TO_ZH: dict[str, str] = {en: zh for en, zh in _PAIRS}
_ZH_TO_EN: dict[str, str] = {zh: en for en, zh in _PAIRS}


def expand_bilingual(labels: list[str]) -> list[str]:
    """
    为调用者传入的标签自动补充另一语言的对等描述。
    已有标签保持原位不变，对等标签紧随其后插入（若已存在则跳过）。
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


# ── 中文 BERT NER 固定类型映射 ────────────────────────────────────────────────
# shibing624/bert4ner-base-chinese 输出的实体类型
BERT_TYPE_TO_LABEL: dict[str, str] = {
    "PER":  "人名或姓名",
    "LOC":  "地名或城市",
    "ORG":  "公司或组织机构名称",
    "TIME": "日期或年份",
    "GPE":  "地名或城市",    # 部分模型区分 GPE（地缘政治实体）
}

# 标准标签 → BERT 类型列表（用于用户自定义标签过滤）
LABEL_TO_BERT_TYPES: dict[str, list[str]] = {
    # 人名
    "人名或姓名":                   ["PER"],
    "full name of a person":        ["PER"],
    # 地名
    "地名或城市":                   ["LOC", "GPE"],
    "geographical location":        ["LOC", "GPE"],
    # 机构
    "公司或组织机构名称":           ["ORG"],
    "company or organization name": ["ORG"],
    "医院或医疗机构名称":           ["ORG"],
    "hospital or medical institution": ["ORG"],
    "大学或研究机构":               ["ORG"],
    "university or research institution": ["ORG"],
    # 时间
    "日期或年份":                   ["TIME"],
    "date or year":                 ["TIME"],
}


def labels_to_bert_types(labels: list[str]) -> set[str] | None:
    """
    将用户标签列表转换为 BERT 实体类型集合。
    返回 None 表示"接受所有类型"（labels 为空或无法映射时）。
    """
    if not labels:
        return None   # 无限制，返回全部
    types: set[str] = set()
    for lbl in labels:
        mapped = LABEL_TO_BERT_TYPES.get(lbl)
        if mapped:
            types.update(mapped)
    return types if types else None  # 无映射 → 不过滤
