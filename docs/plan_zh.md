# UniMiss 仓库梳理与 Ours 解读文档实施计划

## Goal Description
围绕 `docs/draft.md` 的要求，完成一次以静态分析与文档整理为主的仓库梳理工作，不以跑实验为目标，而是以“读清代码、理顺结构、补全文档、标出风险”为目标产出两类核心成果：

1. 一份面向整个仓库的中文总览 README，用于说明代码目录关系、文档关系、可安全整理的文件布局，以及 Baseline / Ours 的运行命令入口。
2. 一份专门面向 `Unimiss`（Ours）的中文深度解读文档，用于说明模型模块组成、参数配置、主实验/消融/可视化/超参数实验入口，以及面向论文投稿还需要关注的实现与改进点。

本计划默认以“不运行完整训练、不伪造实验结果、不破坏现有导入路径”为约束，只做静态阅读、必要的小范围代码修复、文档重写与结构化整理。

## Acceptance Criteria

以下验收标准遵循可确定性验证原则，每条标准都包含正向与反向检查。

- AC-1: 仓库总览文档准确覆盖当前代码库的主要目录、入口脚本、文档与论文材料之间的关系。
  - Positive Tests (expected to PASS):
    - 总览文档明确描述 `Baseline/`、`Ours/`、`models/`、`layers/`、`common/`、`paper/`、`papers/`、`docs/` 的职责边界。
    - 文档中列出的关键入口文件路径均真实存在，例如 `Ours/run.py`、`Baseline/*.py`、`pixi.toml`。
    - 文档中能说明 Baseline 与 Ours 的命令来源是脚本入口或 `pixi` 任务，而不是凭空编造命令。
  - Negative Tests (expected to FAIL):
    - 文档遗漏任一核心目录，导致用户无法根据文档理解仓库主结构。
    - 文档引用不存在的脚本、错误的目录名或错误的命令参数。
    - 文档把论文材料、参考论文 PDF、代码实现混为一谈，未说明它们之间的边界。

- AC-2: Ours 专项解读文档完整说明 `Unimiss` 的实现链路、模块分工与实验入口。
  - Positive Tests (expected to PASS):
    - 专项文档能追踪 `Ours/run.py -> models/unimiss_model.py -> layers/unimiss_modules.py` 的主依赖链。
    - 文档明确说明 `OO / OM / MM / Stage-II gate / decoder` 等核心模块的角色。
    - 文档列出当前 CLI 暴露的主要超参数、结构开关、实验分组以及对应用途。
    - 文档能给出主实验、消融实验、可视化实验、参数统计等任务的入口方式或定位方式。
  - Negative Tests (expected to FAIL):
    - 文档只复述论文口号，不对应到真实代码文件与参数。
    - 文档遗漏主模型模块或把旧草稿链路误写成官方链路。
    - 文档描述的超参数或实验分组在代码中不存在。
  - AC-2.1: 专项文档包含“为了做顶会论文还需要关注的工程/实验缺口”小节。
    - Positive:
      - 明确区分“代码里已经有的内容”和“后续建议补强的内容”。
      - 建议项可追溯到当前仓库现状，例如测试缺失、结果汇总口径、可视化产物、消融覆盖面。
    - Negative:
      - 建议项与仓库无关，或直接捏造尚未存在的实验结论。

- AC-3: 对代码问题的检查采用静态审阅与必要修复的方式完成，且修复范围可解释、可回溯。
  - Positive Tests (expected to PASS):
    - 至少检查 `Ours/`、`models/`、`layers/`、`common/` 与 Baseline 入口的导入链、参数链与文档一致性。
    - 若发现明显问题，修复应局限于不改变研究设定的代码问题，例如路径错误、文档误导、命令失配、显式的静态 bug。
    - 所有修改都能在最终说明中对应到具体文件。
  - Negative Tests (expected to FAIL):
    - 在没有证据的情况下重构模型逻辑或擅自改变实验设计。
    - 为了“看起来完整”而修改大范围代码，但无法解释修改原因。
    - 不检查代码问题，直接宣称“模型没有问题”。

- AC-4: 交付物命名与落点清晰，且默认采用非破坏式文档更新策略。
  - Positive Tests (expected to PASS):
    - 计划执行后，至少有一份仓库总览文档和一份 Ours 专项中文解读文档，且路径在计划中提前定义。
    - 如需补充索引文档、重命名说明文档或整理非代码文件，必须保持代码导入与脚本命令不受影响。
    - 若发生文件移动，文档中同步记录移动前后关系与不移动代码文件的理由。
  - Negative Tests (expected to FAIL):
    - 直接覆盖关键脚本或随意移动 Python 模块导致导入路径失效。
    - 产出文件名与用途不清，用户无法区分“全库说明”和“Ours 解读”。

- AC-5: 输出内容必须是中文，并且术语统一、边界清晰。
  - Positive Tests (expected to PASS):
    - 两份核心交付物主体内容均为中文，英文术语仅作为代码名、参数名、模块名保留。
    - 对“README”“解读文档”“论文草稿”“参考论文 PDF”“运行入口”使用统一称谓。
  - Negative Tests (expected to FAIL):
    - 中英文混写严重，导致结构不一致或术语前后冲突。
    - 文档中使用“已经验证有效”“实验结果优于”之类未在本轮静态工作中证实的表述。

## Path Boundaries

路径边界用于约束本任务的合理实施范围。

### Upper Bound (Maximum Acceptable Scope)
在完整阅读仓库主要代码与文档后，产出一份高质量的全库中文 README 与一份 Ours 深度解读文档，并在必要时完成小范围静态问题修复、文档路径整理、命令清单统一、论文相关材料索引补全；如果存在明显的说明文件散乱问题，可对非代码文档做安全重组并同步更新引用关系。

### Lower Bound (Minimum Acceptable Scope)
不移动任何代码文件，在保持现有目录结构不变的前提下，完成两份中文文档的新增或重写；同时做一次 Ours 主链路静态检查，并记录发现的问题与建议，但只修复明显且低风险的问题。

### Allowed Choices
- Can use: `README.md` 重写、`docs/` 下新增中文说明文档、静态阅读代码、提取 `pixi.toml` 与脚本参数、必要的低风险代码修补、补充文档索引。
- Can use: 将“整体说明”放在根目录 `README.md`，将“Ours 详解”放在 `docs/unimiss_code_guide_zh.md` 或同类路径，以避免覆盖现有实验入口文档。
- Cannot use: 运行完整训练、伪造实验结果、无依据修改模型算法、移动 Python 模块并破坏导入路径、删除用户现有材料。
- Cannot use: 仅凭论文题目或文件名推断实现细节而不核对代码。

> **Note on Deterministic Designs**: 本任务的核心约束较强。最终必须围绕真实文件、真实命令、真实参数写文档，因此“允许选择”的空间主要在文档落点与组织方式，而不在技术事实本身。

## Feasibility Hints and Suggestions

> **Note**: 本节是实施建议，不是强制唯一做法。

### Conceptual Approach
建议按“先建索引，再写文档，最后回查问题”的顺序执行：

1. 先扫描仓库目录、代码入口、环境配置与现有 README，建立“目录职责表”和“运行入口表”。
2. 再沿 Ours 主链路做静态阅读，整理模块图、参数表、实验映射表、输出目录约定。
3. 然后起草两份中文文档：
   - 根 README 强调仓库视角、目录关系、运行入口、文件整理建议。
   - Ours 专项文档强调模型视角、模块分解、超参数、实验设计、后续论文改进点。
4. 最后回看代码与文档是否存在不一致；若有明显静态问题，再做小修并在文档中说明。

### Relevant References
- `docs/draft.md` - 用户原始任务描述与交付要求。
- `README.md` - 当前仓库总览说明，可作为重写起点。
- `Ours/README.md` - 当前 Ours 说明，可作为专项文档的事实来源之一。
- `Ours/run.py` - Ours 官方实验入口，决定命令与参数说明口径。
- `Ours/train.py` - 兼容入口，需要确认其与 `run.py` 的关系。
- `models/unimiss_model.py` - Ours 主模型实现入口。
- `layers/unimiss_modules.py` - `OO / OM / MM / gate / decoder` 等模块拆分位置。
- `common/experiment_utils.py` - 缺失机制、实验辅助逻辑与标签构造来源。
- `Baseline/*.py` - 各 baseline 的统一入口脚本。
- `Baseline/README.md` - baseline 侧说明来源。
- `pixi.toml` - 批量实验任务、命令模板与环境约束。
- `paper/main.tex` 与 `paper/sections/` - 论文草稿材料，帮助统一方法描述口径。

## Dependencies and Sequence

### Milestones
1. 里程碑 1：建立仓库事实基线
   - Phase A: 扫描目录、脚本入口、现有说明文档与环境配置。
   - Phase B: 形成“目录职责表”“命令来源表”“文档落点方案”。
2. 里程碑 2：完成 Ours 主链路静态分析
   - Phase A: 追踪 `run.py`、`train.py`、`models/`、`layers/`、`common/` 的调用关系。
   - Phase B: 汇总模块职责、参数清单、实验分组、输出产物与可能风险点。
3. 里程碑 3：产出两份核心中文文档
   - Phase A: 编写根 README，总结全库结构、文件关系、运行命令、整理建议。
   - Phase B: 编写 Ours 专项解读文档，总结模块、超参数、实验映射、改进方向。
4. 里程碑 4：回查一致性并做必要修复
   - Phase A: 校对文档与代码、命令与路径、实验名与参数是否一致。
   - Phase B: 修复低风险静态问题，并在最终说明中列出修改与未覆盖风险。

依赖顺序上，里程碑 1 是所有后续工作的前提；里程碑 2 依赖仓库事实基线；里程碑 3 依赖前两者的分析结果；里程碑 4 最后执行，用于收口。

## Task Breakdown

每个任务只包含一个路由标签。

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 扫描仓库目录、现有 README、`pixi.toml`、核心脚本入口，建立全库事实清单 | AC-1, AC-4 | analyze | - |
| task2 | 梳理 Baseline 与 Ours 的运行入口、参数来源、输出目录与命令映射关系 | AC-1, AC-5 | analyze | task1 |
| task3 | 追踪 Ours 主模型调用链，整理 `OO / OM / MM / gate / decoder` 模块与相关超参数 | AC-2, AC-2.1 | analyze | task1 |
| task4 | 对 `Ours/`、`models/`、`layers/`、`common/` 做静态代码问题审阅，记录可修问题列表 | AC-3 | analyze | task3 |
| task5 | 编写或重写根目录总览 README，明确仓库结构、命令入口、文件关系与整理建议 | AC-1, AC-4, AC-5 | coding | task2 |
| task6 | 编写 Ours 中文专项解读文档，覆盖模块、参数、实验、可视化、论文改进点 | AC-2, AC-2.1, AC-5 | coding | task3 |
| task7 | 仅在确认低风险且必要时，修复静态代码/文档不一致问题 | AC-3, AC-4 | coding | task4 |
| task8 | 交叉校验两份文档与代码事实是否一致，补充最终风险说明 | AC-1, AC-2, AC-3, AC-5 | analyze | task5, task6, task7 |

## Claude-Codex Deliberation

### Agreements
- 本任务应以“仓库梳理与文档产出”为主，而不是运行实验。
- Ours 的事实口径应以 `Ours/run.py`、`models/unimiss_model.py`、`layers/unimiss_modules.py` 与 `pixi.toml` 为准。
- 文档更新应优先采用非破坏式策略，避免因整理文件而引入导入错误。

### Resolved Disagreements
- 输出文档落点：直接覆盖 `Ours/README.md` 与新增 `docs/` 文档之间，优先选择“根 README 负责全库说明，Ours 详解放在 `docs/` 下新增文件”的方案，因为这样更稳妥，也更容易保留现有 Ours 简版说明作为入口索引。

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

- DEC-1: Ours 详细解读文档的最终落点
  - Claude Position: 放在 `docs/unimiss_code_guide_zh.md`，保留 `Ours/README.md` 作为轻量入口。
  - Codex Position: 同意该方案，因为它最不容易破坏已有引用，并能容纳更长的中文说明。
  - Tradeoff Summary: 覆盖 `Ours/README.md` 更直接，但新增 `docs/` 文档更安全、可扩展、也更适合长篇代码解读。
  - Decision Status: `采用 docs/unimiss_code_guide_zh.md 作为默认方案`

## Implementation Notes

### Code Style Requirements
- 后续如果需要修代码，代码与注释中不要引入 `AC-`、`Milestone`、`Phase` 等计划术语。
- 文档中可保留 `OO`、`OM`、`MM`、`Stage-II gate`、`lightweight_level` 等真实代码术语。
- 文档应显式区分“仓库现状”“静态推断”“后续建议”，避免把建议写成既成事实。

## Output File Convention

本次 `gen-plan` 的主输出文件为：

- `docs/plan_zh.md`

后续执行本计划时，默认交付物命名约定为：

- `README.md`：仓库级中文总览说明。
- `docs/unimiss_code_guide_zh.md`：UniMiss（Ours）中文深度解读文档。
- 如需补充整理说明，可新增 `docs/repo_structure_zh.md` 或同类辅助文件，但不能替代上述两份核心交付物。
