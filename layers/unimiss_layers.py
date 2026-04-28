import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.umag_layers import MaskAwareTemporalEncoder


class FeedForwardExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OOFoundationContextPath(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MaskAwareTemporalEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.value_proj = nn.Linear(1, d_model)
        self.missing_embed = nn.Parameter(torch.randn(n_features, d_model))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        density: torch.Tensor,
        use_oo_foundation: bool = True,
        use_srne: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_oo_foundation:
            return self.encoder(x, mask, density=density, use_srne=use_srne)

        x_proj = self.value_proj(x.unsqueeze(-1))
        missing = self.missing_embed.view(1, 1, x.shape[-1], -1)
        fused = x_proj * mask.unsqueeze(-1) + missing * (1.0 - mask.unsqueeze(-1))
        zero = torch.zeros_like(fused)
        return fused, zero, zero


class OMInteractionBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int = 3,
        expert_hidden: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = max(3, n_experts)
        self.temperature = temperature
        expert_hidden = expert_hidden or (2 * d_model)

        prompt_dim = 2 * d_model + 3
        self.prompt_mlp = nn.Sequential(
            nn.Linear(prompt_dim, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, d_model),
        )
        self.router = nn.Sequential(
            nn.Linear(3 * d_model, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, self.n_experts),
        )

        self.local_expert = FeedForwardExpert(2 * d_model, expert_hidden, d_model, dropout)
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)
        self.cross_out = nn.Linear(d_model, d_model)
        self.reliability_expert = FeedForwardExpert(2 * d_model, expert_hidden, d_model, dropout)
        self.extra_experts = nn.ModuleList(
            [FeedForwardExpert(2 * d_model, expert_hidden, d_model, dropout) for _ in range(self.n_experts - 3)]
        )

    def forward(
        self,
        z_oo: torch.Tensor,
        observed_summary: torch.Tensor,
        local_missingness: torch.Tensor,
        block_stats: torch.Tensor,
        reliability: torch.Tensor,
        use_om_branch: bool = True,
        use_om_routing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not use_om_branch:
            zeros = torch.zeros_like(z_oo)
            weights = torch.zeros(*z_oo.shape[:-1], self.n_experts, device=z_oo.device)
            return zeros, zeros, weights

        prompt_input = torch.cat(
            [z_oo, observed_summary, local_missingness, block_stats, reliability],
            dim=-1,
        )
        om_prompt = self.prompt_mlp(prompt_input)
        router_input = torch.cat([om_prompt, z_oo, om_prompt * z_oo], dim=-1)
        logits = self.router(router_input) / max(self.temperature, 1e-6)
        if use_om_routing:
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = torch.full_like(logits, 1.0 / logits.shape[-1])

        local_out = self.local_expert(torch.cat([z_oo, observed_summary], dim=-1))

        q = self.cross_q(z_oo)
        k = self.cross_k(z_oo)
        v = self.cross_v(z_oo)
        scores = torch.einsum("btdh,btfh->btdf", q, k) / math.sqrt(self.d_model)
        attn = torch.softmax(scores, dim=-1)
        cross_out = torch.einsum("btdf,btfh->btdh", attn, v)
        cross_out = self.cross_out(cross_out)

        reliability_scaled = observed_summary * reliability
        reliability_out = self.reliability_expert(torch.cat([z_oo, reliability_scaled], dim=-1))

        expert_outputs = [local_out, cross_out, reliability_out]
        for expert in self.extra_experts:
            expert_outputs.append(expert(torch.cat([z_oo, om_prompt], dim=-1)))
        stacked = torch.stack(expert_outputs, dim=-2)
        z_om = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return z_om, om_prompt, weights


class MMMissingStructureBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        phase_dim: int = 2,
        period_len: int = 24,
        n_experts: int = 3,
        expert_hidden: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.phase_dim = phase_dim
        self.period_len = max(int(period_len), 1)
        self.n_experts = max(3, n_experts)
        self.temperature = temperature
        expert_hidden = expert_hidden or (2 * d_model)

        self.period_memory = nn.Parameter(torch.randn(self.period_len, d_model))
        self.topology_proj = nn.Linear(d_model, d_model)
        self.periodic_proj = nn.Sequential(
            nn.Linear(phase_dim + d_model, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, d_model),
        )
        self.extreme_proj = nn.Sequential(
            nn.Linear(d_model + 1, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, d_model),
        )
        self.prompt_mlp = nn.Sequential(
            nn.Linear(4 * d_model + 2, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, d_model),
        )
        self.router = nn.Sequential(
            nn.Linear(3 * d_model, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, self.n_experts),
        )
        self.topology_expert = FeedForwardExpert(2 * d_model, expert_hidden, d_model, dropout)
        self.periodic_expert = FeedForwardExpert(2 * d_model, expert_hidden, d_model, dropout)
        self.extreme_expert = FeedForwardExpert(2 * d_model, expert_hidden, d_model, dropout)
        self.extra_experts = nn.ModuleList(
            [FeedForwardExpert(2 * d_model, expert_hidden, d_model, dropout) for _ in range(self.n_experts - 3)]
        )
        self.amplitude_head = nn.Linear(d_model, 1)

    def forward(
        self,
        z_oo: torch.Tensor,
        mask: torch.Tensor,
        phase: torch.Tensor,
        raw_x: torch.Tensor,
        density: torch.Tensor,
        time_index: Optional[torch.Tensor] = None,
        use_mm_branch: bool = True,
        use_topology_prior: bool = True,
        use_periodic_evidence: bool = True,
        use_amplitude_calibration: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        bsz, seq_len, n_features, _ = z_oo.shape
        zeros = torch.zeros_like(z_oo)
        zero_scalar = torch.zeros(*z_oo.shape[:-1], 1, device=z_oo.device)
        if not use_mm_branch:
            weights = torch.zeros(*z_oo.shape[:-1], self.n_experts, device=z_oo.device)
            summaries = {
                "topology_summary": zeros,
                "periodic_summary": zeros,
                "extreme_summary": zero_scalar,
            }
            return zeros, zeros, weights, torch.zeros_like(mask), summaries

        miss = 1.0 - mask
        if use_topology_prior:
            co_missing = torch.einsum("btu,btv->buv", miss, miss) / max(seq_len, 1)
            co_missing = co_missing / (co_missing.sum(dim=-1, keepdim=True) + 1e-8)
            topology_summary = torch.einsum("buv,btvh->btuh", co_missing, z_oo)
            topology_summary = self.topology_proj(topology_summary)
        else:
            topology_summary = torch.zeros_like(z_oo)

        if time_index is None:
            time_index = torch.arange(seq_len, device=z_oo.device)
        period_idx = time_index % self.period_len
        period_memory = self.period_memory[period_idx].unsqueeze(0).unsqueeze(2).expand(bsz, seq_len, n_features, -1)
        if use_periodic_evidence:
            phase_feat = phase.unsqueeze(2).expand(bsz, seq_len, n_features, phase.shape[-1])
            periodic_summary = self.periodic_proj(torch.cat([phase_feat, period_memory], dim=-1))
        else:
            periodic_summary = torch.zeros_like(z_oo)

        feat_mean = (raw_x * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        feat_var = ((raw_x - feat_mean) ** 2 * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        feat_std = torch.sqrt(feat_var + 1e-6)
        extreme_score = torch.abs(raw_x - feat_mean) / feat_std
        extreme_summary = self.extreme_proj(torch.cat([z_oo, extreme_score.unsqueeze(-1)], dim=-1))

        global_missing = (1.0 - density).unsqueeze(1).unsqueeze(-1).expand(bsz, seq_len, n_features, 1)
        prompt_input = torch.cat(
            [z_oo, topology_summary, periodic_summary, extreme_summary, global_missing, miss.unsqueeze(-1)],
            dim=-1,
        )
        mm_prompt = self.prompt_mlp(prompt_input)
        router_input = torch.cat([mm_prompt, z_oo, mm_prompt * z_oo], dim=-1)
        weights = torch.softmax(self.router(router_input) / max(self.temperature, 1e-6), dim=-1)

        topology_out = self.topology_expert(torch.cat([z_oo, topology_summary], dim=-1))
        periodic_out = self.periodic_expert(torch.cat([z_oo, periodic_summary], dim=-1))
        extreme_out = self.extreme_expert(torch.cat([z_oo, extreme_summary], dim=-1))

        expert_outputs = [topology_out, periodic_out, extreme_out]
        for expert in self.extra_experts:
            expert_outputs.append(expert(torch.cat([z_oo, mm_prompt], dim=-1)))
        stacked = torch.stack(expert_outputs, dim=-2)
        z_mm = (weights.unsqueeze(-1) * stacked).sum(dim=-2)

        if use_amplitude_calibration:
            amplitude = torch.sigmoid(self.amplitude_head(z_mm)).squeeze(-1)
        else:
            amplitude = torch.zeros_like(mask)
        summaries = {
            "topology_summary": topology_summary,
            "periodic_summary": periodic_summary,
            "extreme_summary": extreme_score.unsqueeze(-1),
        }
        return z_mm, mm_prompt, weights, amplitude, summaries


class MechanismAwareGate(nn.Module):
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        hidden_dim = hidden_dim or (2 * d_model)
        self.temperature = temperature
        self.prompt_proj = nn.Sequential(
            nn.Linear(d_model + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.score_head = nn.Linear(d_model, 2)

    def forward(
        self,
        mm_prompt: torch.Tensor,
        local_missingness: torch.Tensor,
        global_missingness: torch.Tensor,
        use_stage2_gate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([mm_prompt, local_missingness, global_missingness], dim=-1)
        gate_prompt = self.prompt_proj(gate_input)
        if use_stage2_gate:
            logits = self.score_head(gate_prompt) / max(self.temperature, 1e-6)
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = torch.full(
                (*mm_prompt.shape[:-1], 2),
                0.5,
                device=mm_prompt.device,
                dtype=mm_prompt.dtype,
            )
        return gate_prompt, weights


class UniMissLightweightDecoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or (2 * d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_oo: torch.Tensor,
        z_om: torch.Tensor,
        z_mm: torch.Tensor,
        gate_weights: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_om = gate_weights[..., 0:1]
        beta_mm = gate_weights[..., 1:2]
        z_fused = z_oo + beta_om * z_om + beta_mm * z_mm
        decoder_input = torch.cat([z_fused, beta_om, beta_mm, amplitude.unsqueeze(-1)], dim=-1)
        x_hat = self.decoder(decoder_input).squeeze(-1)
        return x_hat, z_fused
