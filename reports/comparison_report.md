# NER 模型对比测试报告

生成时间：2026-04-30 08:12:29  
阈值（threshold）：`0.4`  

## 一、汇总对比

| 测试用例 | 语言 | gliner_multi-v2.1 召回 | gliner_multi-v2.1 耗时 | gliner-multitask-large-v0.5 召回 | gliner-multitask-large-v0.5 耗时 |
|---|---|---|---|---|---|
| EN-01  英文 · 科技人物 | `en` | 7/7 (100%) | 851ms | 7/7 (100%) | 1478ms |
| EN-02  英文 · 政治新闻 | `en` | 3/5 (60%) | 424ms | 5/5 (100%) | 1804ms |
| ZH-01  中文 · 现代商业（双语标签） | `zh` | 0/8 (0%) | 485ms | 0/8 (0%) | 3195ms |
| ZH-02  中文 · 古典文学（边界测试） | `zh` | 0/8 (0%) | 738ms | 0/8 (0%) | 1937ms |
| ZH-03  中文 · 医疗场景（双语标签） | `zh` | 0/5 (0%) | 774ms | 0/5 (0%) | 2137ms |
| AR-01  阿拉伯语 · 新闻 | `ar` | 2/4 (50%) | 423ms | 2/4 (50%) | 1944ms |
| MIX-01  中英混合 · 职场场景（双语标签） | `mixed` | 4/7 (57%) | 582ms | 4/7 (57%) | 2436ms |
| MIX-02  中英混合 · 学术场景（双语标签） | `mixed` | 2/6 (33%) | 698ms | 2/6 (33%) | 2396ms |
| **平均** | — | **38%** | **622ms** | **43%** | **2166ms** |

## 二、模型加载时间

| 模型 | 加载耗时 |
|---|---|
| gliner_multi-v2.1 | 23.4s |
| gliner-multitask-large-v0.5 | 10.9s |

## 三、逐用例详细结果

### EN-01  英文 · 科技人物

**文本**
```
Elon Musk, CEO of Tesla and founder of SpaceX, announced a new Starship launch from Boca Chica, Texas. NASA has partnered with SpaceX for the Artemis lunar lander mission planned for 2026.
```

**期望实体**：`Elon Musk`, `Tesla`, `SpaceX`, `NASA`, `Boca Chica`, `Texas`, `2026`

#### gliner_multi-v2.1  （851ms，8 个实体，召回 100%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `Elon Musk` | full name of a person | 0.94 | ✓ |
| `Tesla` | company or organization name | 0.77 | ✓ |
| `SpaceX` | company or organization name | 0.94 | ✓ |
| `Boca Chica` | geographical location | 0.89 | ✓ |
| `Texas` | geographical location | 0.78 | ✓ |
| `NASA` | company or organization name | 0.79 | ✓ |
| `SpaceX` | company or organization name | 0.93 | ✓ |
| `2026` | date or year | 0.93 | ✓ |

#### gliner-multitask-large-v0.5  （1478ms，10 个实体，召回 100%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `Elon Musk` | full name of a person | 0.99 | ✓ |
| `Tesla` | company or organization name | 1.00 | ✓ |
| `SpaceX` | company or organization name | 1.00 | ✓ |
| `Starship` | product or technology name | 0.98 |  |
| `Boca Chica` | geographical location | 0.99 | ✓ |
| `Texas` | geographical location | 0.95 | ✓ |
| `NASA` | company or organization name | 1.00 | ✓ |
| `SpaceX` | company or organization name | 1.00 | ✓ |
| `Artemis` | product or technology name | 0.75 |  |
| `2026` | date or year | 0.99 | ✓ |

### EN-02  英文 · 政治新闻

**文本**
```
President Biden signed the Inflation Reduction Act in Washington D.C. on August 16, 2022. The legislation was championed by Senator Chuck Schumer and was seen as a major win for the Democratic Party.
```

**期望实体**：`Biden`, `Chuck Schumer`, `Washington D.C.`, `August 16, 2022`, `Democratic Party`

#### gliner_multi-v2.1  （424ms，6 个实体，召回 60%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `President Biden` | full name of a person | 0.61 |  |
| `Inflation Reduction Act` | legislation or policy name | 0.88 |  |
| `Washington D.C.` | geographical location | 0.68 | ✓ |
| `August 16, 2022` | date or year | 0.96 | ✓ |
| `Senator Chuck Schumer` | full name of a person | 0.56 |  |
| `Democratic Party` | political party | 0.99 | ✓ |

**未命中**：`Biden`, `Chuck Schumer`

#### gliner-multitask-large-v0.5  （1804ms，7 个实体，召回 100%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `Biden` | full name of a person | 0.64 | ✓ |
| `Inflation Reduction Act` | legislation or policy name | 0.98 |  |
| `Washington D.C.` | geographical location | 0.94 | ✓ |
| `August 16, 2022` | date or year | 0.99 | ✓ |
| `Senator` | company or organization name | 0.51 |  |
| `Chuck Schumer` | full name of a person | 0.80 | ✓ |
| `Democratic Party` | political party | 0.99 | ✓ |

### ZH-01  中文 · 现代商业（双语标签）

**文本**
```
阿里巴巴集团创始人马云于2019年卸任董事局主席，由张勇接任。总部位于杭州的阿里巴巴旗下拥有淘宝、天猫、支付宝等业务板块。
```

**期望实体**：`马云`, `张勇`, `阿里巴巴`, `杭州`, `淘宝`, `天猫`, `支付宝`, `2019`

#### gliner_multi-v2.1  （485ms，2 个实体，召回 0%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `阿里巴巴集团创始人马云于2019年卸任董事局主席` | 公司或组织机构名称 | 0.60 |  |
| `支付宝等业务板块` | 产品或品牌名称 | 0.58 |  |

**未命中**：`马云`, `张勇`, `阿里巴巴`, `杭州`, `淘宝`, `天猫`, `支付宝`, `2019`

#### gliner-multitask-large-v0.5  （3195ms，0 个实体，召回 0%）

_未识别到实体_

**未命中**：`马云`, `张勇`, `阿里巴巴`, `杭州`, `淘宝`, `天猫`, `支付宝`, `2019`

### ZH-02  中文 · 古典文学（边界测试）

**文本**
```
尤氏来请，王熙凤笑道：'你来了。'贾母命人摆酒，宝玉和黛玉在大观园散步，薛宝钗独坐梨香院。
```

**期望实体**：`尤氏`, `王熙凤`, `贾母`, `宝玉`, `黛玉`, `薛宝钗`, `大观园`, `梨香院`

#### gliner_multi-v2.1  （738ms，0 个实体，召回 0%）

_未识别到实体_

**未命中**：`尤氏`, `王熙凤`, `贾母`, `宝玉`, `黛玉`, `薛宝钗`, `大观园`, `梨香院`

#### gliner-multitask-large-v0.5  （1937ms，2 个实体，召回 0%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `王熙凤笑道` | 人名或姓名 | 0.64 |  |
| `贾母命人摆酒` | 人名或姓名 | 0.42 |  |

**未命中**：`尤氏`, `王熙凤`, `贾母`, `宝玉`, `黛玉`, `薛宝钗`, `大观园`, `梨香院`

> ⚠️ **边界错误**：['王熙凤笑道']

### ZH-03  中文 · 医疗场景（双语标签）

**文本**
```
北京协和医院心内科主任王建国教授团队，于2023年成功完成首例机器人辅助冠状动脉搭桥手术，患者来自山东省济南市。
```

**期望实体**：`王建国`, `北京协和医院`, `济南`, `山东`, `2023`

#### gliner_multi-v2.1  （774ms，1 个实体，召回 0%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `北京协和医院心内科主任王建国教授团队` | hospital or institution name | 0.65 |  |

**未命中**：`王建国`, `北京协和医院`, `济南`, `山东`, `2023`

#### gliner-multitask-large-v0.5  （2137ms，0 个实体，召回 0%）

_未识别到实体_

**未命中**：`王建国`, `北京协和医院`, `济南`, `山东`, `2023`

### AR-01  阿拉伯语 · 新闻

**文本**
```
أعلن الرئيس محمد بن سلمان عن إطلاق مشروع نيوم في المملكة العربية السعودية عام 2017، وتبلغ تكلفته 500 مليار دولار.
```

**期望实体**：`محمد بن سلمان`, `نيوم`, `المملكة العربية السعودية`, `2017`

#### gliner_multi-v2.1  （423ms，5 个实体，召回 50%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `محمد بن سلمان` | full name of a person | 0.82 | ✓ |
| `مشروع نيوم` | project or initiative name | 0.72 |  |
| `المملكة العربية السعودية` | geographical location | 0.85 | ✓ |
| `عام 2017` | date or year | 0.91 |  |
| `500 مليار دولار` | monetary amount | 0.94 |  |

**未命中**：`نيوم`, `2017`

#### gliner-multitask-large-v0.5  （1944ms，5 个实体，召回 50%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `محمد بن سلمان` | full name of a person | 0.72 | ✓ |
| `مشروع نيوم` | project or initiative name | 0.76 |  |
| `المملكة العربية السعودية` | geographical location | 0.98 | ✓ |
| `عام 2017` | date or year | 0.99 |  |
| `500 مليار دولار` | monetary amount | 0.97 |  |

**未命中**：`نيوم`, `2017`

### MIX-01  中英混合 · 职场场景（双语标签）

**文本**
```
张伟加入了 Google 北京研发中心，负责 Android 系统优化。他的同事 Sarah Chen 来自 Meta，两人共同参与了 2024 年的 AI Summit。
```

**期望实体**：`张伟`, `Google`, `Sarah Chen`, `Meta`, `Android`, `北京`, `2024`

#### gliner_multi-v2.1  （582ms，5 个实体，召回 57%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `Google` | 公司或组织机构名称 | 0.68 | ✓ |
| `北京研发中心` | 地名或城市 | 0.49 |  |
| `Sarah Chen` | 人名或姓名 | 0.88 | ✓ |
| `Meta` | 公司或组织机构名称 | 0.67 | ✓ |
| `2024` | 日期或年份 | 0.41 | ✓ |

**未命中**：`张伟`, `Android`, `北京`

#### gliner-multitask-large-v0.5  （2436ms，6 个实体，召回 57%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `张伟加入了` | 人名或姓名 | 0.40 |  |
| `Google` | company or organization name | 0.89 | ✓ |
| `Android` | 产品或技术名称 | 0.68 | ✓ |
| `Sarah Chen` | 人名或姓名 | 0.94 | ✓ |
| `Meta` | company or organization name | 0.84 | ✓ |
| `2024 年的` | date or year | 0.93 |  |

**未命中**：`张伟`, `北京`, `2024`

### MIX-02  中英混合 · 学术场景（双语标签）

**文本**
```
清华大学计算机系教授李明在 NeurIPS 2023 发表了关于 Transformer 架构的论文，合作者来自 MIT 和 Stanford University。
```

**期望实体**：`李明`, `清华大学`, `NeurIPS 2023`, `Transformer`, `MIT`, `Stanford University`

#### gliner_multi-v2.1  （698ms，2 个实体，召回 33%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `MIT` | 大学或研究机构 | 0.70 | ✓ |
| `Stanford University` | 大学或研究机构 | 0.85 | ✓ |

**未命中**：`李明`, `清华大学`, `NeurIPS 2023`, `Transformer`

#### gliner-multitask-large-v0.5  （2396ms，5 个实体，召回 33%）

| 文本 | 标签 | 置信度 | 命中期望 |
|---|---|---|---|
| `NeurIPS` | conference or journal name | 0.90 |  |
| `2023` | date or year | 0.91 |  |
| `Transformer 架构的论文` | technology or model name | 0.74 |  |
| `MIT` | university or research institution | 0.93 | ✓ |
| `Stanford University` | university or research institution | 0.93 | ✓ |

**未命中**：`李明`, `清华大学`, `NeurIPS 2023`, `Transformer`

## 四、结论与建议

- **综合召回最高**：`gliner-multitask-large-v0.5`（平均召回 43%）
- **推理最快**：`gliner_multi-v2.1`（平均 622ms/次）

### 优化建议

1. **双语标签策略**：对中文或混合文本，同时提供中英文标签描述（如 `"人名或姓名"` + `"full name of a person"`），可显著提升中文实体召回率。GLiNER 是零样本模型，标签描述越具体、越接近训练语料的表达方式，识别效果越好。
2. **Span 去重**：使用双语标签时同一文本跨度可能被打上两个标签，建议在服务层按 `(start, end)` 去重，保留得分最高的结果（已在 `app/ner.py` 实现）。
3. **阈值调优**：英文建议 `threshold=0.5`，中文建议 `threshold=0.35~0.4`（模型对中文置信度普遍偏低）。
4. **古典/文言文**：两个模型对文言文支持均弱，建议结合规则或专用模型（如 `BERT-CRF` 在古汉语语料上微调）处理此类文本。
5. **阿拉伯语**：`gliner-multitask-large-v0.5` 在多语言上训练，对阿拉伯语有基础支持；`gliner_multi-v2.1` 阿拉伯语效果有限。
