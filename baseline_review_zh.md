# Baseline 论文精读与对比综述

## 1. 文档定位

本文档严格以 `2026CIKM.pdf` 为主线，服务于当前 `Unimiss` 工作区的 baseline 代码整理、论文对比和后续实验组织。这里不把 `papers/` 中所有论文都视为主实验 baseline，而是只围绕当前 CIKM 主表实际要对打的方法展开。

## 2. 当前 CIKM 主实验范围

根据 `2026CIKM.pdf` 当前稿中的实验设定，主线范围已经比较明确：

- 数据集：`ETT` 与 `IAQ`
- 缺失机制：`MAR`、`MNAR`、`Mix`
- 缺失率：`20% / 30% / 40%`
- `Mix` 场景：先施加 `MNAR`，再施加 `MAR`，两者对半
- 随机种子：`3407 / 3408 / 3409`
- 指标：`RMSE / MAE / MRE / NRMSE`
- 主表 baseline：`Mean / BRITS / SPIN / TimesNet / SAITS / ImputeFormer / MTSCI / FGTI / Ours`

同时，稿件还明确提示了若干收敛方向：

- 主表应重点突出 `Mix` 场景
- `MAR / MNAR` 可以作为补充结果或附录
- 应删去与当前主线无关的旧配置，例如 `physionet`、`pems_traffic` 等历史遗留部分

## 3. CIKM 中的 baseline 重新分组方式

`2026CIKM.pdf` 并不是简单把 prior methods 横向并列，而是把它们重新解释为三类缺失依赖建模方式的对照：

- `OO`：observed-to-observed，上下文内部的稳定结构建模
- `OM`：observed-to-missing，由观测信息恢复缺失位置
- `MM`：missing-to-missing，显式建模缺失位置之间的结构与耦合

按这个视角，现有 baseline 的价值不只是“强弱比较”，而是回答一个更细的问题：每种方法主要强化了哪类依赖，又忽略了哪类依赖。

## 4. 核心 baseline 精读

### 4.1 Mean

`Mean.py` 对应的是启发式统计基线，不依赖单独论文。它的意义主要在于提供一个最低复杂度、最低表达能力的参考下界。它不会建模 `OO`、`OM` 或 `MM` 中的任何一类结构，更多是为了确认复杂模型带来的真实收益。

### 4.2 BRITS

- 论文：`papers/brits_neurips2018.pdf`
- 代码：`Baseline/Brits.py`

BRITS 的核心思想是把缺失值视为递归计算图中的变量，在双向时序传播中持续更新，而不是先静态填补再做序列建模。它通过 temporal decay 和双向一致性，强化了基于观测历史的信息恢复。

按 `2026CIKM.pdf` 的解读，BRITS 是非常典型的 `OM` 主导 baseline。它确实利用了 observed context，但 missing positions 之间并没有被显式建模成独立结构，因此 `MM` 基本缺位。

对我们的启发是：BRITS 很适合说明“只依赖 observed history 的恢复逻辑”能够做到什么，但它不足以覆盖复杂 missing-side pattern。

### 4.3 SAITS

- 论文：`papers/saits_2023_arxiv.pdf`
- 代码：`Baseline/SAITS.py`

SAITS 用自注意力替代严格自回归链路，把全局上下文建模引入缺失恢复任务。它的优势在于并行建模与全局依赖表达更强，能更稳定地把 observed token 的信息传给 missing token。

在当前 CIKM 视角下，SAITS 依然属于典型的 `OM` 融合方法，同时具有一定 `OO` 支撑，因为 observed tokens 之间的关系会通过 self-attention 被强化。但 `MM` 依旧只是共享 backbone 内的隐式副产物，不是一个独立建模目标。

### 4.4 ImputeFormer

- 论文：`papers/imputeformer_kdd2024_arxiv2312.01728.pdf`
- 代码：`Baseline/ImputeFormer.py`

ImputeFormer 进一步强化了时序维和变量维的联合建模，把插补任务直接融入 Transformer 表征。相较 BRITS 或更早的 attention baseline，它对 observed context 的统一表达更强，也更强调掩码相关信息在 backbone 中的作用。

在 `2026CIKM.pdf` 中，ImputeFormer 被视为 `OO + OM` 都较强的方法：它对 observed-observed 的结构表达比 BRITS 更完整，对 observed-to-missing 的恢复也更自然。但 missing positions 之间的耦合仍然没有被提升为显式分支，`MM` 仍是隐式的。

### 4.5 SPIN

- 论文：`papers/spin_2022_arxiv.pdf`
- 代码：`Baseline/SPIN.py`，共享实现位于 `models/spin_model.py`

SPIN 强调图结构与时空传播，把变量之间的连接关系和时间依赖统一考虑，并辅以 mask-aware initialization。它关注的是如何更充分地传播 observed context，而不是如何单独抽取 missing-side pattern。

因此，SPIN 在当前文稿中最适合被归入 `OO` 强 baseline。它对 `OM` 也有帮助，因为 missing node 仍会从 observed node 获取信息；但它并没有把 `MM` 写成显式主线。

### 4.6 TimesNet

- 论文：`papers/timesnet_2023_arxiv.pdf`
- 代码：`Baseline/Timesnet.py`

TimesNet 的代表性在于多周期模式建模。它更擅长从时间序列中提炼周期性和多尺度时间结构，而不是专门为缺失建模设计恢复结构。

在当前比较框架中，TimesNet 更像“提供周期模式先验”的 baseline。它可以增强 `OO` 以及部分间接的 `OM` 恢复能力，但并不显式回答 missing-specific structure 应如何建模。因此它不是 `MM` baseline，而是周期结构先验的代表。

### 4.7 FGTI

- 论文：`papers/fgti_neurips2024.pdf`
- 代码：`Baseline/FGTI.py`

FGTI 引入了生成式恢复思路，不再把缺失值恢复视为单点确定性映射，而是把恢复建模为条件生成 / 去噪过程。它比传统 `OM` 方法更接近 missing-side structure 的建模动机，因为多个缺失位置可以在同一个生成链条中被共同约束。

按照 `2026CIKM.pdf` 的判断，FGTI 是当前 baseline 中最接近 `MM` 动机的生成式方法，但它并没有把 `MM` 做成显式的 expert、branch 或 router。它更像是“隐式触及 MM”，而不是“显式写出 MM”。

这也是它对当前工作的关键价值：FGTI 为 `MM` 提供了最强启发来源，但还没有真正给出可解释、可路由的 missing-structure 专家结构。

### 4.8 MTSCI

- 论文：`papers/mtsci_2024_arxiv.pdf`
- 代码：`Baseline/MTSCI.py`

MTSCI 同样属于生成式 / 一致性约束视角的方法。它通过 diffusion 或 denoising 式恢复路径，把多步修正和恢复一致性纳入建模过程，强调恢复过程的全局稳定性。

从 CIKM 稿件的分类看，MTSCI 对 `OM` 有明显支撑，对 `MM` 也有启发，因为多个缺失位置会在共同恢复过程中被约束；但这种 missing coupling 仍然没有被拆成显式 missing-side module。

因此，MTSCI 是“生成式一致性约束视角”的重要 baseline，却仍然不是显式 `MM` 方法。

## 5. 与我们工作的关键边界

当前 `2026CIKM.pdf` 对我们工作的主张，不是“在统一 backbone 上再堆一个更强模型”，而是重新划分恢复过程中的职责边界：

- `OO` 不再承担所有恢复职责，而是保留为稳定的 foundational context path
- `OM` 被提升为 interaction-oriented expert branch，专门决定如何调用 observed evidence
- `MM` 被提升为 missing-structure expert branch，显式建模 co-missing topology、周期性缺失证据、极值相关线索等 missing-side 结构
- 在此基础上再加入 `Stage-II mechanism-aware MAR-vs-MNAR soft gate`，动态调节 `OM` 与 `MM` 的相对权重

换句话说，prior work 通常是“统一主干 + 统一恢复逻辑”；而当前工作要强调的是“结构分工 + 机制感知 + 动态路由”。

## 6. 为什么 `MM` 必须被显式提升

按当前稿件的主张，现实中的缺失并不总是随机噪声。尤其在 `MNAR` 和 `Mix` 场景下，不同缺失位置的恢复依赖并不一致：

- `MAR-like` 位置更适合依赖 `OO + OM`
- `MNAR-like` 位置更需要依赖 `OO + MM`

如果仍然让所有位置共享同一条恢复链路，那么模型即使偶尔“碰到” missing-side pattern，也很难做到显式、稳定、可解释地调用它。`MM` 被单独写成 expert 的意义，正在于把这条依赖从共享 backbone 的隐式副产物，提升成一个明确的建模对象。

## 7. `papers/` 中哪些不是当前主实验 baseline

以下论文更适合作为方法设计灵感，而不是当前 CIKM 主表 baseline：

- `flex_moe_neurips2024.pdf`
- `i2moe_icml2025.pdf`
- `lingual_smoe_iclr2024.pdf`
- `moepp_iclr2025.pdf`
- `moe_x_icml2025.pdf`
- `moirai_moe_icml2025.pdf`
- `mole_iclr2024_arxiv.pdf`
- `roe_iclr2025.pdf`
- `soft_moe_iclr2024.pdf`
- `time_moe_iclr2025_arxiv.pdf`

这些论文的主要作用，是为 expert、router、MoE specialization、routing interpretability 等设计提供参考，而不是在当前主表里逐个对打。

## 8. 当前代码侧对应关系

当前工作区的 baseline runner 已经统一整理到 `Baseline/`：

- `Baseline/Brits.py`
- `Baseline/Mean.py`
- `Baseline/Timesnet.py`
- `Baseline/SAITS.py`
- `Baseline/ImputeFormer.py`
- `Baseline/FGTI.py`
- `Baseline/MTSCI.py`
- `Baseline/SPIN.py`

共享实现保留在：

- `models/`
- `layers/`

其中：

- `SPIN` 直接依赖 `models/spin_model.py`
- `FGTI` 依赖 `models/main_model.py`
- `MTSCI` 依赖 `models/model.py` 等共享实现

## 9. 后续写论文或汇报时的推荐叙事顺序

建议严格按下面顺序组织口径：

1. 先说明 prior work 普遍把恢复逻辑写成统一 backbone，`MM` 很少被显式建模。
2. 再按 `OO / OM / MM` 重组 baseline 家族，而不是按年份简单罗列。
3. 逐篇说明 `BRITS / SAITS / ImputeFormer / SPIN / TimesNet / FGTI / MTSCI` 各自强化了哪类依赖、缺了哪类依赖。
4. 最后引出我们的结构性差异：`OO foundational context + OM expert + MM expert + Stage-II MAR-vs-MNAR soft gate`。

只有这样，当前工作与 baseline 的边界才会清晰，也更符合 `2026CIKM.pdf` 现有草稿的论证路径。
