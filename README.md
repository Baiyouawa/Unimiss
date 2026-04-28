# UniMiss 仓库说明

## 1. 项目定位

这个仓库围绕 `UniMiss` 的缺失值插补实验整理，当前工作区同时包含：

- `Baseline/`：用于对比实验的 baseline 入口脚本。
- `Ours/`：`UniMiss` 的官方实验入口。
- `models/`、`layers/`、`common/`：`Ours` 与部分 baseline 共用的模型、层与实验辅助逻辑。
- `paper/`、`papers/`：论文草稿与参考论文 PDF。
- `docs/`：任务草稿、计划文档，以及后续生成的中文说明文档。

当前文档基于静态阅读整理，不声称已经在本地完成完整训练或复现实验结果。

## 2. 目录关系

### 核心代码目录

- `Baseline/`
  - 8 个 baseline 的可运行脚本：`Brits.py`、`Mean.py`、`Timesnet.py`、`SAITS.py`、`ImputeFormer.py`、`MTSCI.py`、`FGTI.py`、`SPIN.py`
  - `result_writer.py` 负责生成 baseline 结果摘要 `result.md`
- `Ours/`
  - `run.py` 是 `UniMiss` 的唯一官方实验入口
  - `train.py` 只是兼容包装层，内部直接调用 `Ours.run.main()`
  - `README.md` 提供 Ours 的简版入口说明
- `models/`
  - `unimiss_model.py`：`UniMissModel` 主模型
  - 其他文件主要服务于 FGTI、MTSCI、SPIN、扩散模型等对比方法
- `layers/`
  - `unimiss_modules.py`：`OO / OM / MM / Stage-II gate / decoder / sep loss`
  - `umag_layers.py`：`MaskAwareTemporalEncoder`、RoPE 编码层、SRNE 相关逻辑
  - `spin_*`、`Diff_layers.py`、`Embed.py` 等文件服务于其他模型或早期实现
- `common/`
  - `experiment_utils.py`：数据缓存、缺失机制、弱监督标签、指标汇总、结果追加等公共函数

### 文档与论文目录

- `docs/`
  - `draft.md`：用户原始任务草稿
  - `plan_zh.md`：本轮生成的中文实施计划
  - `unimiss_code_guide_zh.md`：`UniMiss` 中文深度解读文档
- `paper/`
  - `main.tex` 与 `sections/*.tex`：当前论文草稿
  - `refs.bib`：参考文献
- `papers/`
  - baseline、MoE、时序插补相关参考论文 PDF
- `tmp/pdfs/`
  - 若干 PDF 的文本抽取结果，便于快速检索内容

### 代码关系摘要

- `Ours/run.py -> models/unimiss_model.py -> layers/unimiss_modules.py -> layers/umag_layers.py`
- `Baseline/*.py` 直接作为运行入口，部分 baseline 会复用 `models/`、`layers/`、`Baseline/result_writer.py`
- `pixi.toml` 是批量实验命令和任务编排的事实来源

## 3. 当前实验范围

从 `pixi.toml`、`Ours/run.py` 与 baseline 脚本看，当前主实验口径是：

- 数据集
  - `electricity_transformer_temperature`
  - `italy_air_quality`
- 缺失机制
  - ETT：`mar`、`mnar_t`、`mix`
  - IAQ：`mar`、`mnar_x`、`mix`
- 缺失率
  - `0.2`、`0.3`、`0.4`
- 默认随机种子
  - `3407,3408,3409`

`mix` 场景还支持 `--mar_ratio`，用于控制 MAR/MNAR 的混合比例。

## 4. 如何运行 Baseline

### 直接运行脚本

所有 baseline 都可以直接从仓库根目录执行，例如：

```bash
python Baseline/Brits.py --dataset electricity_transformer_temperature --mask_type mix --missing_rate 0.2
python Baseline/Mean.py --dataset italy_air_quality --mask_type mar --missing_rate 0.3
python Baseline/FGTI.py --dataset italy_air_quality --mask_type mix --missing_rate 0.2 --mar_ratio 0.8
```

当前 baseline 脚本普遍支持这组核心参数：

- `--dataset`
- `--mask_type`
- `--missing_rate`
- `--mar_ratio`（`mix` 场景）
- `--prep_n_steps`
- 多数深度模型还支持 `--epochs`、`--batch_size`、`--lr` 等训练参数

### 通过 Pixi 运行

`pixi.toml` 中为 baseline 预定义了完整任务矩阵。示例：

```bash
pixi run run-brits-ett-mix-20
pixi run run-mean-italy-mar-20
pixi run run-saits-ett-mnar_t-40
pixi run run-fgti-italy-mix-30
```

批量任务：

```bash
pixi run batch-main-table
pixi run batch-supplementary-ett
pixi run batch-supplementary-italy
```

### Baseline 输出

baseline 默认把结果写到各自目录，例如：

- `outputs/brits/`
- `outputs/mean/`
- `outputs/saits/`
- `outputs/timesnet/`
- `outputs/imputeformer/`
- `outputs/mtsci/`
- `outputs/fgti/`
- `outputs/spin/`

典型产物：

- `seed_<seed>/metrics.json`
- `metrics_avg.json`
- `result.md`

## 5. 如何运行 Ours（UniMiss）

### 官方入口

官方入口是：

```bash
python Ours/run.py
```

兼容旧命令的入口是：

```bash
python Ours/train.py
```

`Ours/train.py` 不包含独立训练逻辑，只是把调用转发给 `Ours/run.py`。

### 常用直接命令

```bash
python Ours/run.py --dataset electricity_transformer_temperature --mask_type mix --missing_rate 0.2 --experiment_group main --run_name full
python Ours/run.py --dataset italy_air_quality --mask_type mix --missing_rate 0.2 --experiment_group ablation --run_name no_mm --no-use-mm
python Ours/run.py --dataset electricity_transformer_temperature --mask_type mix --missing_rate 0.2 --experiment_group visualization --run_name vis_ett_mix20
python Ours/run.py --dataset electricity_transformer_temperature --mask_type mar --missing_rate 0.2 --experiment_group param_count --run_name params
```

### 常用 Pixi 任务

主实验：

```bash
pixi run run-ours-ett-mix-20
pixi run run-ours-italy-mix-30
```

消融：

```bash
pixi run abl-ett-20-no-oo
pixi run abl-ett-20-no-mm
pixi run abl-italy-40-no-gate
```

超参数：

```bash
pixi run hparam-ett-dmodel-128
pixi run hparam-ett-gate-temp-05
pixi run hparam-italy-period-48
```

可视化：

```bash
pixi run vis-ett-mix-20
pixi run vis-italy-mix-20
```

轻量化/缩放：

```bash
pixi run scale-ett-lite-s
pixi run scale-ett-lite-m
pixi run scale-italy-tiny
```

参数量与比例敏感性：

```bash
pixi run count-ours-params
pixi run ratio-ours-ett-mar20
pixi run ratio-ours-italy-mar80
```

批量编排：

```bash
pixi run batch-ablation
pixi run batch-hyperparameter
pixi run batch-scaling
pixi run batch-ratio
pixi run batch-all
```

### Ours 输出目录

默认输出路径由 `Ours/run.py` 动态构造：

```text
outputs/unimiss/<experiment_group>/<dataset>/<mask_type_or_mix_ratio>/mr_<missing_rate>/<run_name>/
```

其中：

- 平衡 `mix` 场景使用 `mix`
- 非默认比例会写成 `mix_mar20`、`mix_mar80`

典型产物：

- `run_manifest.json`
- `seed_<seed>/config.json`
- `seed_<seed>/best.pt`
- `seed_<seed>/metrics.json`
- `seed_<seed>/visualization.json`
- `seed_<seed>/visualization.png`（若本地安装了 `matplotlib`）
- `seed_<seed>/param_count.json`（仅 `param_count` 分组）
- `metrics_avg.json`
- `result.md`

## 6. 推荐阅读顺序

如果你想先读懂仓库，再决定怎么跑实验，建议按这个顺序：

1. `README.md`
2. `Baseline/README.md`
3. `Ours/README.md`
4. `docs/unimiss_code_guide_zh.md`
5. `Ours/run.py`
6. `models/unimiss_model.py`
7. `layers/unimiss_modules.py`
8. `common/experiment_utils.py`

## 7. 文件整理建议

当前目录已经基本按“入口脚本 / 共用模块 / 文档 / 论文材料”分开。为了不破坏导入路径，这一轮不建议移动 Python 代码文件。

可以继续保持：

- 根目录只放总览说明、环境文件与高层文档
- `docs/` 放中文解读、计划与说明文档
- `paper/` 放正在写的论文草稿
- `papers/` 放外部参考论文

## 8. 相关说明文档

- `Baseline/README.md`：baseline 侧说明
- `Ours/README.md`：Ours 快速入口说明
- `docs/unimiss_code_guide_zh.md`：Ours 深度解读
- `baseline_review_zh.md`：baseline 综述性中文文档

如果后续要做主实验、消融、可视化或论文写作，优先以 `pixi.toml`、`Ours/run.py` 和 `docs/unimiss_code_guide_zh.md` 为准。
