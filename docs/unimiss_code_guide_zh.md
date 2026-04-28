# UniMiss 代码详解

## 1. 文档目的

这份文档只解释当前仓库中的 `UniMiss` 实现，不替代论文，也不声称已经在本地完成训练验证。它的目标是帮助你回答四类问题：

- `UniMiss` 现在到底由哪些代码文件组成。
- 模型从数据到输出的真实调用链是什么。
- 现有超参数、实验分组、消融与可视化入口分别在哪里。
- 如果要继续做顶会论文，当前工程还缺什么、下一步该补什么。

## 2. 官方入口与调用链

### 官方入口

- 官方入口：`Ours/run.py`
- 兼容入口：`Ours/train.py`

`Ours/train.py` 只有一件事：导入 `Ours.run.main()` 并执行它，所以后续分析都以 `Ours/run.py` 为准。

### 主调用链

`UniMiss` 当前的真实主链路是：

```text
Ours/run.py
  -> UniMissModel (models/unimiss_model.py)
    -> OOFoundationPath / OMBranch / MMBranch / StageIIGate / LightweightDecoder
       (layers/unimiss_modules.py)
         -> MaskAwareTemporalEncoder / RoPE layers / SRNE support
            (layers/umag_layers.py)
```

这条链路也是后续文档、实验与消融说明的唯一事实来源。

## 3. Ours/run.py 在做什么

`Ours/run.py` 同时承担了四件事情：

1. 定义命令行参数和实验分组。
2. 准备数据与缺失机制标签。
3. 调用 `UniMissModel` 训练、验证、测试。
4. 输出 `run_manifest.json`、`metrics.json`、`metrics_avg.json`、`result.md`、可视化文件等实验产物。

### 数据与缺失机制

`Ours/run.py` 通过 `common.experiment_utils.apply_mask_with_labels()` 构造缺失位置和弱监督标签：

- `MISSING_LABEL_MAR = 1`
- `MISSING_LABEL_MNAR = 2`

这意味着 `UniMiss` 不只是做插补，还会把缺失机制标签用于 `Stage-II gate` 的弱监督训练。

### 训练流程

每个 seed 的流程大致是：

1. 读取数据集。
2. 在 train/val/test 上按当前缺失机制重新造缺失。
3. 构建 `SequenceDataset`。
4. 用 `UniMissModel` 前向。
5. 计算重建损失、分支分离损失、gate 约束损失。
6. 按验证集 `rmse` 保存最优权重。
7. 在测试集汇总指标并写文件。

### 输出目录

默认输出目录是：

```text
outputs/unimiss/<experiment_group>/<dataset>/<mask_type_or_mix_ratio>/mr_<missing_rate>/<run_name>/
```

每个 seed 会在下面再展开一层：

```text
seed_<seed>/
```

常见产物：

- `run_manifest.json`
- `seed_<seed>/config.json`
- `seed_<seed>/best.pt`
- `seed_<seed>/metrics.json`
- `metrics_avg.json`
- `result.md`
- `seed_<seed>/visualization.json`
- `seed_<seed>/visualization.png`
- `seed_<seed>/param_count.json`（仅 `param_count`）

## 4. UniMissModel 的结构

`models/unimiss_model.py` 里的 `UniMissModel` 非常直接，它把主模型拆成 5 个可解释模块：

1. `OOFoundationPath`
2. `OMBranch`
3. `MMBranch`
4. `StageIIGate`
5. `LightweightDecoder`

另外还定义了一个辅助损失：

- `BranchDecouplingLoss`

前向逻辑可以概括为：

1. `OOFoundationPath` 先得到基础上下文表征 `z_oo`
2. `OMBranch` 基于 `z_oo` 和局部统计量得到交互分支 `z_om`
3. `MMBranch` 基于 `z_oo` 和缺失结构相关特征得到缺失结构分支 `z_mm`
4. `StageIIGate` 根据 `MM` prompt、局部缺失密度、全局缺失率生成 `beta_om / beta_mm`
5. `LightweightDecoder` 用 `z_oo`、`z_om`、`z_mm` 与 gate 权重重建 `x_hat`

## 5. 各模块的真实职责

### 5.1 OOFoundationPath

文件位置：`layers/unimiss_modules.py`

职责：

- 建立全局的基础上下文表示。
- 把值本身、缺失情况、相位信息、密度信息融合进一个统一的 latent 表征。

关键细节：

- 核心编码器是 `MaskAwareTemporalEncoder`
- prompt 输入由这些量拼接得到：
  - 当前与前一时刻的差值 `delta`
  - 当前时刻观测率 `obs_rate_t`
  - 密度特征 `density`
  - 周期相位 `phase`

输出：

- `z_oo`
- `oo_prompt`
- `mu`
- `logvar`

### 5.2 OMBranch

职责：

- 建模“交互模式”而不是“缺失结构”本身。
- 通过 3 个 expert 让模型分别关注不同层面的依赖关系。

它的 3 个 expert 是：

- `temporal_expert`
- `feature_expert`
- `global_expert`

router 会输出 3 路 softmax 权重，决定每个位置更偏向哪类交互表征。

### 5.3 MMBranch

职责：

- 专门建模缺失结构相关信号。
- 对应当前实现里的 3 类 expert：
  - `topology_expert`
  - `periodic_expert`
  - `extreme_expert`

关键输入：

- 局部缺失密度 `topo`
- 特征均值与时刻均值差形成的 `extreme`
- 基于 `period_len` 构造的 `period_bank`
- 周期相位 `phase`

输出除了 `z_mm` 之外，还会额外返回：

- `weights`
- `amplitude`
- `topology`
- `extreme`
- `periodic`

其中 `amplitude` 会直接送进解码器。

### 5.4 StageIIGate

职责：

- 让模型在 `OM` 与 `MM` 之间做机制感知的动态加权。
- 本质上输出两条权重：
  - `beta_om`
  - `beta_mm`

gate 的输入不是原始 `x`，而是：

- `MM` 分支的 prompt
- 局部缺失密度
- 全局缺失率

这说明当前实现更像是“根据缺失结构与缺失强度，决定交互分支和缺失结构分支谁更该被信任”。

### 5.5 LightweightDecoder

职责：

- 用尽量轻的 MLP 把三路 latent 与 gate 权重变回插补值。

输入拼接包括：

- `z_oo`
- `z_om * beta_om`
- `z_mm * beta_mm`
- `beta_om`
- `beta_mm`
- `amplitude`

这意味着当前 decoder 不做复杂自回归，而是把大部分建模能力留在前面的表示学习与 routing 上。

## 6. MaskAwareTemporalEncoder 与 SRNE

`OOFoundationPath` 依赖的 `MaskAwareTemporalEncoder` 在 `layers/umag_layers.py` 里。

它的关键点：

- 用 `value_proj` 把标量时间序列映射到 `d_model`
- 对缺失位置使用 `missing_embed`
- 用多层 `RoPETransformerLayer` 做时序编码
- 生成 `mu / logvar / z`，带有轻量 VAE 风格的随机采样形式

### SRNE 是什么

当前代码里的 `use_srne` 开关会在 `MaskAwareTemporalEncoder.forward()` 中启用一段基于 `density` 的低观测率补偿逻辑：

- 先构造 `inv_density`
- 过 `sra_mlp`
- 再加到 fused 表征上

因此，这里的 SRNE 更像“对低观测率位置做额外表征补偿”的机制开关，而不是完全独立的模型分支。

## 7. 当前训练目标

`Ours/run.py` 实际用到 3 项训练信号：

### 7.1 重建损失 `L_rec`

由：

- `reconstruction_loss(outputs["x_hat"], batch["raw_x"], batch["target_mask"])`

负责主插补目标。

### 7.2 分支分离损失 `L_sep`

由：

- `outputs["sep_loss"]`

对应 `BranchDecouplingLoss`，实现方式是让：

- `z_oo`
- `z_om`
- `z_mm`

在目标缺失位置上尽量少对齐，降低分支角色塌缩的风险。

是否启用由：

- `--use-sep-loss`

控制，损失权重由：

- `--lambda_sep`

控制。

### 7.3 Gate 约束损失 `L_gate`

由：

- `gate_regulation_loss(beta_om, beta_mm, mech_labels, target_mask)`

实现。

它利用 `MAR / MNAR` 的弱监督标签，要求：

- MAR 位置更偏向 `beta_om`
- MNAR 位置更偏向 `beta_mm`

损失权重由：

- `--lambda_gate`

控制。

## 8. 现有 CLI 面与默认参数

下面按当前代码的真实默认值整理。

### 数据与训练参数

- `--dataset`：`electricity_transformer_temperature` 或 `italy_air_quality`
- `--mask_type`：`mar`、`mnar_x`、`mnar_t`、`mix`
- `--missing_rate`：`0.2 / 0.3 / 0.4`
- `--mar_ratio`：默认 `0.5`
- `--prep_n_steps`：默认 `48`
- `--cuda_device`：默认 `None`
- `--epochs`：默认 `20`
- `--batch_size`：默认 `16`
- `--lr`：默认 `1e-3`
- `--weight_decay`：默认 `1e-5`
- `--grad_clip`：默认 `5.0`

### 模型结构参数

- `--d_model`：默认 `192`
- `--n_heads`：默认 `8`
- `--n_layers`：默认 `4`
- `--d_ff`：默认 `256`
- `--dropout`：默认 `0.1`
- `--period_len`：默认 `24`
- `--gate_temperature`：默认 `1.0`

### 损失权重

- `--lambda_sep`：默认 `0.05`
- `--lambda_gate`：默认 `0.1`

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

### 运行控制参数

- `--experiment_group`：`main / ablation / hyperparameter / visualization / scaling / param_count`
- `--run_name`：默认 `default`
- `--seeds`：默认 `3407,3408,3409`
- `--output_dir`
- `--visualization_samples`：默认 `2`

### 轻量化等级

当前代码支持：

- `base`
- `small`
- `lite_s`
- `lite_m`
- `tiny`

其中 `small` 与 `lite_s` 在 `resolve_model_scale()` 中走同一条缩放逻辑。

## 9. 实验分组与 Pixi 任务

### 主实验

对应：

- `run-ours-ett-mar-*`
- `run-ours-ett-mnar_t-*`
- `run-ours-ett-mix-*`
- `run-ours-italy-mar-*`
- `run-ours-italy-mnar_x-*`
- `run-ours-italy-mix-*`

### 主消融

对应：

- `abl-ett-*-no-oo`
- `abl-ett-*-no-om`
- `abl-ett-*-no-mm`
- `abl-ett-*-no-gate`
- `abl-italy-*-no-oo`
- `abl-italy-*-no-om`
- `abl-italy-*-no-mm`
- `abl-italy-*-no-gate`

这些开关分别对应：

- 去掉 `OO foundation`
- 去掉 `OM branch`
- 去掉 `MM branch`
- 去掉 `Stage-II gate`

### 超参数敏感性

当前任务集中在：

- `d_model`
- `gate_temperature`
- `period_len`

对应任务名：

- `hparam-ett-*`
- `hparam-italy-*`

### 可视化

可视化任务：

- `vis-ett-mix-20`
- `vis-italy-mix-20`

保存内容包括：

- 样本级 `ground_truth / prediction / observed`
- `target_mask`
- `mechanism_label`
- `beta_om`
- `beta_mm`
- `gate_prompt_norm`
- 汇总的 gate 分布统计

### 轻量化/缩放

当前 `pixi.toml` 给出的任务是：

- `scale-ett-lite-s`
- `scale-ett-lite-m`
- `scale-ett-tiny`
- `scale-italy-lite-s`
- `scale-italy-lite-m`
- `scale-italy-tiny`

### 参数量

任务：

- `count-ours-params`

它不会跑完整训练，而是写出参数统计结果。

### 比例敏感性

当前额外支持 `mix` 场景下的 MAR/MNAR 比例敏感性：

- `ratio-ours-ett-mar20`
- `ratio-ours-ett-mar80`
- `ratio-ours-italy-mar20`
- `ratio-ours-italy-mar80`

## 10. 当前实现适合怎么做主实验、消融、可视化

### 主实验怎么跑

建议直接使用 `pixi.toml` 里的主任务矩阵，而不是再造新 runner。

最核心的主表任务是：

```bash
pixi run batch-main-table
```

如果只想单点检查：

```bash
pixi run run-ours-ett-mix-20
pixi run run-ours-italy-mix-20
```

### 消融怎么跑

如果要做论文主消融，当前最直接的是 `mix` 场景下的 4 个开关：

- `w/o OO`
- `w/o OM`
- `w/o MM`
- `w/o Stage-II gate`

示例：

```bash
pixi run abl-ett-20-no-mm
pixi run abl-italy-30-no-gate
```

### 可视化怎么跑

```bash
pixi run vis-ett-mix-20
pixi run vis-italy-mix-20
```

重点看：

- `beta_om / beta_mm` 是否和 `MAR-like / MNAR-like` 标签方向一致
- 可视化样本上观测值、预测值、真实值是否具有一致性

### 超参数实验怎么跑

```bash
pixi run batch-hyperparameter
```

当前代码最值得先看的是：

- `d_model`
- `gate_temperature`
- `period_len`

### 轻量化怎么跑

```bash
pixi run batch-scaling
```

如果你想继续写论文中的“效率-效果折中”部分，这一组任务就是当前仓库已经准备好的起点。

## 11. 本轮静态审阅结论

### 已确认的问题

1. 旧版 `README.md` 的目录、链接和 `pixi` 任务名存在明显失真，不能继续作为仓库总览。
2. 原来的 `Ours/README.md` 没覆盖当前 CLI 的全部开关，尤其遗漏了 `--use-sep-loss` 和 `lightweight_level=small`。
3. `Ours/run.py` 中保留了一组与 `common.experiment_utils.py` 重复的 masking helper，当前主流程并不使用它们，属于低风险死代码候选。

### 尚未发现需要大改的模型逻辑错误

基于静态阅读，当前没有发现“必须立即修改研究设定”的硬错误，例如：

- 明显的参数维度断裂
- 主链路导入错误
- `pixi.toml` 与 `Ours/run.py` 的实验分组完全不匹配

但这不等于已经证明训练结果正确，只能说明从静态结构上看，主链路是可解释且基本自洽的。

## 12. 如果要继续做顶会论文，下一步建议

这部分只谈“建议”，不把它写成已经完成的事实。

### 工程层面

- 增加最小化的自动检查，至少覆盖 CLI 参数、输出目录命名和可视化产物存在性。
- 把 `Ours/run.py` 中重复但未使用的 masking 辅助函数清理掉，避免后续文档和主流程出现双事实源。
- 给关键实验产物增加更稳定的汇总脚本，而不是只依赖人工读 `metrics_avg.json`。

### 实验层面

- 扩展比率敏感性实验，把 `MAR:MNAR` 混合比例分析写成独立补充实验段落。
- 在消融中加入 `use_sep_loss`、`use_srne`、`topology / periodic / extreme expert` 的更细粒度拆分。
- 对 `small / lite_s / lite_m / tiny` 做统一的效率表，包括参数量、训练时间、推理时间与效果下降幅度。

### 可视化层面

- 增加不同缺失机制下的 gate 热力图，而不只看均值统计。
- 增加失败样例分析，特别是 `MNAR` 主导和极端值区域。

### 论文表达层面

- 明确区分“交互模式分支”和“缺失结构分支”的职责，避免只写成抽象的多专家。
- 把 `Stage-II gate` 的监督来源讲清楚：它不是完全无监督，而是利用了机制标签的弱监督约束。
- 解释 `small` 与 `lite_s` 的关系，避免读者误以为这是两个完全不同的缩放策略。
