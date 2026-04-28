# UniMiss Ours 实验说明

## 1. 官方入口

当前 `Ours` 的唯一官方入口是 [`Ours/run.py`](/D:/codex/Unimiss/Ours/run.py)。

补充说明：

- [`Ours/train.py`](/D:/codex/Unimiss/Ours/train.py) 只是兼容包装层，不是独立的官方实现。
- 当前有效模型链路是：
  - [`models/unimiss_model.py`](/D:/codex/Unimiss/models/unimiss_model.py)
  - [`layers/unimiss_modules.py`](/D:/codex/Unimiss/layers/unimiss_modules.py)

## 2. 方法结构

当前实现严格按照 `2026CIKM` 文档中的分工组织：

- `OO foundational context path`
- `OM interaction-oriented expert branch`
- `MM missing-structure expert branch`
- `Stage-II mechanism-aware MAR-vs-MNAR soft gate`
- `Lightweight decoder`

训练目标与文档保持一致，只保留三项主损失：

- `L_rec`
- `L_sep`
- `L_gate`

其中：

- `L_sep` 用于维持 `OO / OM / MM` 的角色分离；
- `L_gate` 用于约束 Stage-II gate 真正学到 `MAR-vs-MNAR` 的动态调节偏好。

## 3. 统一实验协议

`Ours/run.py` 复用和 baseline 一致的实验主协议：

- 数据集：`electricity_transformer_temperature`、`italy_air_quality`
- 缺失机制：
  - `ETT`：`mar / mnar_t / mix`
  - `IAQ`：`mar / mnar_x / mix`
- 缺失率：`0.2 / 0.3 / 0.4`
- Mix 场景 MAR 比例：`--mar_ratio`，默认 `0.5`（即 50/50），支持 `0.2`（MNAR 主导）/ `0.8`（MAR 主导）
- 随机种子：默认 `3407,3408,3409`
- 指标：`MAE / RMSE / MRE / MAPE`（含 `mape_capped / mape_trimmed / smape` 鲁棒变体）

## 4. 主要参数

基础参数：

- `--dataset`
- `--mask_type`
- `--missing_rate`
- `--mar_ratio`（仅 mix 场景有效，默认 0.5）
- `--prep_n_steps`
- `--epochs`
- `--batch_size`
- `--lr`
- `--weight_decay`

模型参数：

- `--d_model`
- `--n_heads`
- `--n_layers`
- `--d_ff`
- `--dropout`
- `--period_len`
- `--lambda_sep`
- `--lambda_gate`
- `--gate_temperature`

结构开关：

- `--use-oo / --no-use-oo`
- `--use-om / --no-use-om`
- `--use-mm / --no-use-mm`
- `--use-stage2-gate / --no-use-stage2-gate`
- `--use-topology-expert / --no-use-topology-expert`
- `--use-periodic-expert / --no-use-periodic-expert`
- `--use-extreme-expert / --no-use-extreme-expert`
- `--use-srne / --no-use-srne`

实验分组：

- `--experiment_group main`
- `--experiment_group ablation`
- `--experiment_group hyperparameter`
- `--experiment_group visualization`
- `--experiment_group scaling`
- `--experiment_group param_count`

轻量化配置：

- `--lightweight_level base`
- `--lightweight_level lite_s`
- `--lightweight_level lite_m`
- `--lightweight_level tiny`

## 5. 推荐命令（Pixi）

所有实验均通过 `pixi.toml` 管理，在项目根目录运行：

单个实验：

```bash
pixi run run-ours-ett-mix-20          # ETT / mix / 0.2 主实验
pixi run abl-ett-20-no-mm             # ETT / mix / 0.2 消融：去掉 MM
pixi run hparam-ett-gate-temp-05      # ETT 超参数：gate_temperature=0.5
pixi run vis-ett-mix-20               # ETT 可视化
pixi run scale-ett-lite-s             # ETT 轻量化 lite_s
pixi run count-ours-params            # 参数量统计
pixi run ratio-ours-ett-mar20         # ETT / mix / MNAR 主导（MAR:MNAR = 20:80）
pixi run ratio-ours-ett-mar80         # ETT / mix / MAR 主导（MAR:MNAR = 80:20）
```

批量编排（一键跑一组实验）：

```bash
pixi run batch-main-table             # 主表：所有 baseline + Ours × Mix × 全部 rate × 两个数据集
pixi run batch-ablation               # 全部消融实验（ETT+IAQ × mix × 3 rates × 4 ablations）
pixi run batch-hyperparameter         # 全部超参数敏感性实验
pixi run batch-scaling                # 全部轻量化/伸缩性实验
pixi run batch-ratio                  # Mix 比例敏感性（Ours + FGTI/MTSCI/ImputeFormer × 两档比例）
pixi run batch-supplementary-ett      # ETT 补充实验（MAR + MNAR_T 场景）
pixi run batch-supplementary-italy    # IAQ 补充实验（MAR + MNAR_X 场景）
pixi run batch-all                    # 跑完 CIKM 2026 全部实验
```

查看所有可用任务：

```bash
pixi task list
```

## 6. 输出目录

默认输出路径：

```text
outputs/unimiss/<experiment_group>/<dataset>/<mask_type>/mr_<missing_rate>/<run_name>/
```

当 `--mar_ratio` 不是默认的 0.5 时，`<mask_type>` 部分会带上比例标识：

```text
outputs/unimiss/main/electricity_transformer_temperature/mix_mar20/mr_0.2/ratio_mar20/
```

每个 seed 会写到：

```text
outputs/unimiss/<experiment_group>/<dataset>/<mask_type>/mr_<missing_rate>/<run_name>/seed_<seed>/
```

主要产物包括：

- `config.json`
- `best.pt`
- `metrics.json`
- `metrics_avg.json`
- `result.md`

在 `visualization` 分组下，还会导出：

- 样本级预测曲线
- `beta_om / beta_mm` 分布摘要
- `MAR-like / MNAR-like` 标签对应的 gate 统计

## 7. 实验映射建议

主表：

- 使用 `main`
- 保持 `OO / OM / MM / Stage-II gate` 全部开启

主消融：

- `w/o OO foundation`：`--no-use-oo`
- `w/o OM interaction routing`：`--no-use-om`
- `w/o MM experts`：`--no-use-mm`
- `w/o Stage-II mechanism-aware gate`：`--no-use-stage2-gate`

补充实验：

- 超参数：`d_model`、`gate_temperature`、`period_len`
- 可视化：`visualization`
- 轻量化与伸缩性：`scaling`
- 参数量：`param_count`

## 8. Pixi 环境说明

- 所有实验任务已统一配置在项目根目录 `pixi.toml` 中。
- 首次运行 `pixi run <task>` 时会自动创建隔离环境并安装依赖。
- 需要 CUDA 12.0 + Python 3.9 环境。
