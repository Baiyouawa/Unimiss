from typing import Optional

import torch
from torch import Tensor, nn


class SPINModel(nn.Module):
    """SPIN-style model without external tsl/pyg dependency.

    It keeps the key design choices from original SPIN:
    - mask-aware initialization
    - different embeddings for observed/missing states
    - layered spatio-temporal message passing
    - deep supervision-compatible outputs
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_nodes: int,
        u_size: Optional[int] = None,
        output_size: Optional[int] = None,
        temporal_self_attention: bool = True,
        reweight: Optional[str] = "softmax",
        n_layers: int = 4,
        eta: int = 3,
        message_layers: int = 1,
    ):
        super().__init__()
        del reweight, message_layers  # kept for API compatibility
        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention
        # Keep both trunks; only reduce spatial contribution.
        self.use_spatial_branch = True
        self.spatial_scale = 0.5
        self.temporal_scale = 0.5

        self.x_proj = nn.Linear(input_size, hidden_size)
        self.u_proj = nn.Linear(u_size, hidden_size)
        self.node_emb = nn.Embedding(n_nodes, hidden_size)
        self.valid_emb = nn.Embedding(n_nodes, hidden_size)
        self.mask_emb = nn.Embedding(n_nodes, hidden_size)
        bottleneck = max(hidden_size * 3 // 4, 8)

        # Keep core temporal branch, but with small dropout for mild regularization.
        head_num = 4 if hidden_size % 4 == 0 else 1
        self.temporal_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=head_num,
                    batch_first=True,
                    dropout=0.1,
                )
                for _ in range(n_layers)
            ]
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, bottleneck),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(bottleneck, hidden_size),
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
        self.x_skip = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(n_layers)])
        self.readouts = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(n_layers)])

    def _edge_index_to_adj(self, edge_index: Tensor, device: torch.device) -> Tensor:
        adj = torch.eye(self.n_nodes, device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0].long(), edge_index[1].long()
            adj[src, dst] = 1.0
            # Force symmetric connectivity to match original SPIN setting.
            adj[dst, src] = 1.0
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adj / deg

    def _temporal_message(self, h: Tensor, layer_idx: int) -> Tensor:
        # h: [B, L, N, D] -> attention on each node sequence [B*N, L, D]
        if not self.temporal_self_attention:
            return torch.zeros_like(h)
        b, l, n, d = h.shape
        hn = h.permute(0, 2, 1, 3).reshape(b * n, l, d)
        out, _ = self.temporal_attn[layer_idx](hn, hn, hn, need_weights=False)
        out = out.reshape(b, n, l, d).permute(0, 2, 1, 3)
        return out

    def forward(
        self,
        x: Tensor,
        u: Tensor,
        mask: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        node_index: Optional[Tensor] = None,
        target_nodes: Optional[Tensor] = None,
    ):
        del edge_weight, node_index
        if target_nodes is None:
            target_nodes = slice(None)

        b, l, n, _ = x.shape
        if n != self.n_nodes:
            raise ValueError(f"n_nodes mismatch: got {n}, expected {self.n_nodes}")

        device = x.device
        node_ids = torch.arange(n, device=device)
        node_bias = self.node_emb(node_ids).view(1, 1, n, -1)

        h = self.x_proj(x) + self.u_proj(u).unsqueeze(2) + node_bias
        h = torch.where(mask.bool(), h, self.u_proj(u).unsqueeze(2) + node_bias)

        imputations = []
        for layer_idx, (layer, norm, out_head) in enumerate(zip(self.layers, self.norms, self.readouts)):
            if layer_idx == self.eta:
                valid = self.valid_emb(node_ids).view(1, 1, n, -1)
                masked = self.mask_emb(node_ids).view(1, 1, n, -1)
                h = torch.where(mask.bool(), h + 0.7 * valid, h + 0.7 * masked)

            # Skip only on observed points (same spirit as original implementation).
            h = h + self.x_skip[layer_idx](x) * mask

            if self.use_spatial_branch:
                adj = self._edge_index_to_adj(edge_index.to(device), device)
                spatial = torch.einsum("ij,bljd->blid", adj, h)
            else:
                spatial = torch.zeros_like(h)
            temporal = self._temporal_message(h, layer_idx)
            fused = self.spatial_scale * spatial + self.temporal_scale * temporal
            h = norm(h + layer(fused))
            imputations.append(out_head(h[..., target_nodes, :]))

        x_hat = imputations.pop(-1)
        return x_hat, imputations
