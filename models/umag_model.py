from typing import Tuple

import torch
import torch.nn as nn

from layers.umag_layers import (
    MaskAwareTemporalEncoder,
    SpatialMechanismCoupledDecoder,
    TailSensitiveContrastiveModule,
)


class UMAGModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_heads: int = 16,
        n_layers: int = 6,
        d_ff: int = 256,
        phase_dim: int = 2,
        period_len: int = 24,
        adj_lambda: float = 0.1,
        corr_min: float = 0.05,
        dropout: float = 0.1,
        temperature: float = 0.1,
        tail_q: float = 0.1,
        max_samples: int = 1024,
        tscl_period_weight: float = 0.0,
    ):
        super().__init__()
        self.period_len = period_len
        self.mate = MaskAwareTemporalEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.smcd = SpatialMechanismCoupledDecoder(
            n_features=n_features,
            d_model=d_model,
            phase_dim=phase_dim,
            period_len=period_len,
            adj_lambda=adj_lambda,
            corr_min=corr_min,
            dropout=dropout,
        )
        self.tscl = TailSensitiveContrastiveModule(
            temperature=temperature,
            tail_q=tail_q,
            max_samples=max_samples,
            period_weight=tscl_period_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        phase: torch.Tensor,
        density: torch.Tensor,
        raw_x: torch.Tensor,
        obs_mask: torch.Tensor,
        time_index: torch.Tensor = None,
        use_gate: bool = True,
        use_mech: bool = True,
        use_rca: bool = True,
        use_pmm: bool = True,
        use_srne: bool = True,
        use_tscl: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.mate(x, mask, density=density, use_srne=use_srne)
        x_hat, m_hat, g = self.smcd(
            z,
            phase,
            density,
            use_gate=use_gate,
            use_mech=use_mech,
            use_rca=use_rca,
            use_pmm=use_pmm,
            time_index=time_index,
        )
        if use_tscl:
            l_contrast = self.tscl(
                z,
                raw_x,
                obs_mask,
                time_index=time_index,
                period_len=self.period_len,
            )
        else:
            l_contrast = torch.zeros((), device=z.device)
        return x_hat, m_hat, l_contrast, mu, logvar
