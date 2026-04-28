# UniMiss 工作区

## 1. 项目概览

本仓库围绕 [`2026CIKM.pdf`](/D:/codex/Unimiss/2026CIKM.pdf) 的当前稿件组织，包含两条主线：

- `Baseline/`：当前 CIKM 主表中使用的 baseline 运行脚本与结果写出逻辑。
- `Ours/`：严格对齐文档方法论的 UniMiss 实现、消融、超参数、可视化、轻量化与参数量统计入口。

当前统一实验范围如下：

- 数据集：`electricity_transformer_temperature`、`italy_air_quality`
- 缺失机制：`MAR`、`MNAR`、`Mix`
- 缺失率：`0.2 / 0.3 / 0.4`
- 随机种子：`3407 / 3408 / 3409`
- 指标：`MAE / RMSE / MRE / MAPE`

## 2. 目录结构

- `Baseline/`
  - 各 baseline 的统一 runner
  - `result_writer.py`
  - `Baseline/README.md`
- `Ours/`
  - `run.py`：唯一官方实验入口
  - `train.py`：兼容包装层，内部转发到 `run.py`
  - `Ours/README.md`
- `models/`
  - `unimiss_model.py`：`UniMissModel`
- `layers/`
  - `unimiss_modules.py`：`OO / OM / MM / Stage-II gate` 显式模块
- `docs/`
  - `baseline_review_zh.md`：baseline 精读综述与对比口径
- `paper/`
  - ACM CIKM 风格的 LaTeX 论文草稿
- `.codex-task/`
  - Humanize 任务计划、状态与交接记录

## 3. Ours 方法映射

当前 `Ours` 主线与文档方法论一一对应：

- `OO foundational context path`
  - [`models/unimiss_model.py`](/D:/codex/Unimiss/models/unimiss_model.py)
  - [`layers/unimiss_modules.py`](/D:/codex/Unimiss/layers/unimiss_modules.py) 中的 `OOFoundationPath`
- `OM interaction-oriented expert branch`
  - [`layers/unimiss_modules.py`](/D:/codex/Unimiss/layers/unimiss_modules.py) 中的 `OMBranch`
- `MM missing-structure expert branch`
  - [`layers/unimiss_modules.py`](/D:/codex/Unimiss/layers/unimiss_modules.py) 中的 `MMBranch`
- `Stage-II mechanism-aware MAR-vs-MNAR soft gate`
  - [`layers/unimiss_modules.py`](/D:/codex/Unimiss/layers/unimiss_modules.py) 中的 `StageIIGate`
- 轻量解码器
  - [`layers/unimiss_modules.py`](/D:/codex/Unimiss/layers/unimiss_modules.py) 中的 `LightweightDecoder`

训练目标现在收敛为文档要求的三项主损失：

- `L_rec`
- `L_sep`
- `L_gate`

其中 `L_gate` 使用由缺失机制构造出的弱监督标签，约束 Stage-II gate 学到 `OM-vs-MM` 的 `MAR-vs-MNAR` 动态偏好。

## 4. 统一实验入口

官方入口是 [`Ours/run.py`](/D:/codex/Unimiss/Ours/run.py)。

常用参数：

- 基础实验参数
  - `--dataset`
  - `--mask_type`
  - `--missing_rate`
  - `--prep_n_steps`
  - `--epochs`
  - `--batch_size`
  - `--lr`
  - `--weight_decay`
- 结构参数
  - `--d_model`
  - `--n_heads`
  - `--n_layers`
  - `--d_ff`
  - `--period_len`
  - `--dropout`
  - `--lambda_sep`
  - `--lambda_gate`
  - `--gate_temperature`
- 核心消融开关
  - `--no-use-oo`
  - `--no-use-om`
  - `--no-use-mm`
  - `--no-use-stage2-gate`
  - `--no-use-topology-expert`
  - `--no-use-periodic-expert`
  - `--no-use-extreme-expert`
- 实验分组
  - `--experiment_group main`
  - `--experiment_group ablation`
  - `--experiment_group hyperparameter`
  - `--experiment_group visualization`
  - `--experiment_group scaling`
  - `--experiment_group param_count`

说明：

- `Ours/train.py` 不是独立实现，只是兼容入口，避免旧命令失效。
- `layers/unimiss_layers.py` 保留为早期草稿，不是当前官方依赖链。

## 5. 输出目录约定

`Ours` 的默认输出目录统一写入：

```text
outputs/unimiss/<experiment_group>/<dataset>/<mask_type>/mr_<missing_rate>/<run_name>/seed_<seed>/
```

典型输出文件：

- `config.json`
- `best.pt`
- `metrics.json`
- `metrics_avg.json`
- `result.md`
- `visualization.json`
- `visualization.png`（若本地存在 `matplotlib`）

可视化分组下会额外保存：

- 样本级插补对比
- gate 权重分布摘要
- `MAR-like` / `MNAR-like` 位置对应的 gate 行为

## 6. Pixi 任务

`pixi.toml` 中已经整理出：

- baseline 主实验矩阵
- `run-ours-*` 主实验矩阵
- `run-ours-ablation-*` 四个主消融
- `run-ours-hparam-*` 关键超参数实验
- `run-ours-vis-*` 可视化实验
- `run-ours-lite-*` 轻量化/伸缩性实验
- `count-ours-params` 参数量统计

建议优先直接使用这些任务名，而不是继续新增平行 runner。

## 7. 论文草稿

论文草稿位于 [`paper/`](/D:/codex/Unimiss/paper)，采用 ACM CIKM 风格 `acmart` 组织。当前已包含：

- `main.tex`
- `sections/*.tex`
- `refs.bib`

写作口径应严格参考：

- [`2026CIKM.pdf`](/D:/codex/Unimiss/2026CIKM.pdf)
- [`tmp/pdfs/2026CIKM.txt`](/D:/codex/Unimiss/tmp/pdfs/2026CIKM.txt)
- [`docs/baseline_review_zh.md`](/D:/codex/Unimiss/docs/baseline_review_zh.md)

## 8. 当前限制

- 当前 shell 中未发现可直接使用的 `pixi` 命令，因此本轮主要完成静态整理与代码收口。
- 当前 `py -3` 环境缺少可用的 `torch._C`，无法在此 shell 内完成真实训练验证。
- 因此最终运行效果仍需以用户自己的 Pixi / Conda 实验环境为准。


pixi.toml 现在包含 222 个任务，覆盖了 CIKM 2026 的完整实验需求：

类别	任务数	说明
Baseline 个体任务	152	8 个模型 × 2 数据集 × 3 机制 × 3 缺失率 + 8 个快捷入口
Ours 主实验	19	2 数据集 × 3 机制 × 3 缺失率 + 1 默认入口
消融实验	24	ETT+IAQ × mix × {0.2,0.3,0.4} × {no_oo, no_om, no_mm, no_gate}
超参数	11	d_model / gate_temp / period_len 变体，覆盖两个数据集
可视化	2	ETT + IAQ
缩放/轻量化	6	lite_s / lite_m / tiny × 两个数据集
参数统计	1	
批量编排	7	一键跑整组实验