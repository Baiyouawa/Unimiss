# UniMiss Ours 快速说明

## 官方入口

- 官方入口：`Ours/run.py`
- 兼容入口：`Ours/train.py`

`Ours/train.py` 只是兼容包装层，内部直接转发到 `Ours.run.main()`。

## 真实实现链路

当前 `UniMiss` 的主链路是：

```text
Ours/run.py
  -> models/unimiss_model.py
    -> layers/unimiss_modules.py
      -> layers/umag_layers.py
```

如果你想看完整中文解读，请直接阅读：

- `docs/unimiss_code_guide_zh.md`

## 常用运行方式

直接运行：

```bash
python Ours/run.py --dataset electricity_transformer_temperature --mask_type mix --missing_rate 0.2 --experiment_group main --run_name full
python Ours/run.py --dataset italy_air_quality --mask_type mix --missing_rate 0.2 --experiment_group visualization --run_name vis_italy_mix20
python Ours/run.py --dataset electricity_transformer_temperature --mask_type mar --missing_rate 0.2 --experiment_group param_count --run_name params
```

通过 `pixi.toml` 运行：

```bash
pixi run run-ours-ett-mix-20
pixi run abl-ett-20-no-mm
pixi run hparam-ett-gate-temp-05
pixi run vis-ett-mix-20
pixi run scale-ett-lite-s
pixi run count-ours-params
```

## 当前 CLI 要点

### 基础参数

- `--dataset`
- `--mask_type`
- `--missing_rate`
- `--mar_ratio`
- `--prep_n_steps`
- `--epochs`
- `--batch_size`
- `--lr`
- `--weight_decay`

### 结构与损失参数

- `--d_model`
- `--n_heads`
- `--n_layers`
- `--d_ff`
- `--dropout`
- `--period_len`
- `--lambda_sep`
- `--lambda_gate`
- `--gate_temperature`

### 结构开关

- `--use-oo / --no-use-oo`
- `--use-om / --no-use-om`
- `--use-mm / --no-use-mm`
- `--use-stage2-gate / --no-use-stage2-gate`
- `--use-sep-loss / --no-use-sep-loss`
- `--use-srne / --no-use-srne`
- `--use-topology-expert / --no-use-topology-expert`
- `--use-periodic-expert / --no-use-periodic-expert`
- `--use-extreme-expert / --no-use-extreme-expert`

### 实验分组

- `main`
- `ablation`
- `hyperparameter`
- `visualization`
- `scaling`
- `param_count`

### 轻量化等级

- `base`
- `small`
- `lite_s`
- `lite_m`
- `tiny`

## 输出目录

默认输出目录：

```text
outputs/unimiss/<experiment_group>/<dataset>/<mask_type_or_mix_ratio>/mr_<missing_rate>/<run_name>/
```

典型产物：

- `run_manifest.json`
- `seed_<seed>/config.json`
- `seed_<seed>/best.pt`
- `seed_<seed>/metrics.json`
- `metrics_avg.json`
- `result.md`
- `seed_<seed>/visualization.json`
- `seed_<seed>/visualization.png`

## 进一步阅读

- 仓库总览：`README.md`
- Ours 深度解读：`docs/unimiss_code_guide_zh.md`
- baseline 入口说明：`Baseline/README.md`
