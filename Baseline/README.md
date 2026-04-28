# Baseline Directory

This directory contains the runnable baseline entry scripts used by the current `Unimiss` workspace.

## Included Baselines

| Script | Method | Paper / Source | Notes |
| --- | --- | --- | --- |
| `Brits.py` | BRITS | `papers/brits_neurips2018.pdf` | Typical observed-to-missing baseline |
| `Mean.py` | Mean | heuristic baseline | No paper in `papers/` |
| `Timesnet.py` | TimesNet | `papers/timesnet_2023_arxiv.pdf` | Periodicity-oriented baseline |
| `SAITS.py` | SAITS | `papers/saits_2023_arxiv.pdf` | Attention-based imputation baseline |
| `ImputeFormer.py` | ImputeFormer | `papers/imputeformer_kdd2024_arxiv2312.01728.pdf` | Transformer baseline |
| `FGTI.py` | FGTI | `papers/fgti_neurips2024.pdf` | Generative baseline with shared `models/` implementation |
| `MTSCI.py` | MTSCI | `papers/mtsci_2024_arxiv.pdf` | Diffusion / consistency-style baseline |
| `SPIN.py` | SPIN | `papers/spin_2022_arxiv.pdf` | Graph-aware spatio-temporal baseline |

## Shared Components

- `result_writer.py`: shared result markdown writer for baseline runners.
- `../models/`: shared model implementations, including FGTI / MTSCI / SPIN related code.
- `../layers/`: shared low-level layers used by project and baseline code.

## Current CIKM-Oriented Scope

The current CIKM draft narrows the main baseline comparison to:

- Datasets: `electricity_transformer_temperature` and `italy_air_quality`
- Missing mechanisms: `mar`, `mnar_t` or `mnar_x`, and `mix`
- Missing rates: `0.2`, `0.3`, `0.4`
- Seeds: `3407`, `3408`, `3409`

`pixi.toml` is aligned to this cleaned baseline scope.

