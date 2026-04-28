from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm, functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter
from torch_scatter.utils import broadcast
from tsl.nn.blocks.encoders import MLP
from tsl.nn.functional import sparse_softmax


class AdditiveAttention(MessagePassing):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        output_size: int,
        msg_size: Optional[int] = None,
        msg_layers: int = 1,
        root_weight: bool = True,
        reweight: Optional[str] = None,
        norm: bool = True,
        dropout: float = 0.0,
        dim: int = -2,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=dim, **kwargs)
        self.output_size = output_size
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size

        self.msg_size = msg_size or self.output_size
        self.msg_layers = msg_layers
        assert reweight in ["softmax", "l1", None]
        self.reweight = reweight
        self.root_weight = root_weight
        self.dropout = dropout

        self.lin_src = Linear(
            self.src_size, self.output_size, weight_initializer="glorot", bias_initializer="zeros"
        )
        self.lin_tgt = Linear(self.tgt_size, self.output_size, weight_initializer="glorot", bias=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size, bias=False)
        else:
            self.register_parameter("lin_skip", None)

        self.msg_nn = nn.Sequential(
            nn.PReLU(init=0.2),
            MLP(
                self.output_size,
                self.msg_size,
                self.output_size,
                n_layers=self.msg_layers,
                dropout=self.dropout,
                activation="prelu",
            ),
        )

        if self.reweight == "softmax":
            self.msg_gate = nn.Linear(self.output_size, 1, bias=False)
        else:
            self.msg_gate = nn.Sequential(nn.Linear(self.output_size, 1), nn.Sigmoid())

        if norm:
            self.norm = LayerNorm(self.output_size)
        else:
            self.register_parameter("norm", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_tgt.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()

    def forward(self, x: PairTensor, edge_index: Adj, mask: OptTensor = None):
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src
        n_src, n_tgt = x_src.size(self.node_dim), x_tgt.size(self.node_dim)
        msg_src = self.lin_src(x_src)
        msg_tgt = self.lin_tgt(x_tgt)
        out = self.propagate(edge_index, msg=(msg_src, msg_tgt), mask=mask, size=(n_src, n_tgt))
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)
        if self.norm is not None:
            out = self.norm(out)
        return out

    def normalize_weights(self, weights, index, num_nodes, mask=None):
        if mask is not None:
            fill_value = float("-inf") if self.reweight == "softmax" else 0.0
            weights = weights.masked_fill(torch.logical_not(mask), fill_value)
        if self.reweight == "l1":
            expanded_index = broadcast(index, weights, self.node_dim)
            weights_sum = scatter(weights, expanded_index, self.node_dim, dim_size=num_nodes, reduce="sum")
            weights_sum = weights_sum.index_select(self.node_dim, index)
            weights = weights / (weights_sum + 1e-5)
        elif self.reweight == "softmax":
            weights = sparse_softmax(weights, index, num_nodes=num_nodes, dim=self.node_dim)
        return weights

    def message(self, msg_j: Tensor, msg_i: Tensor, index, size_i, mask_j: OptTensor = None) -> Tensor:
        msg = self.msg_nn(msg_j + msg_i)
        gate = self.msg_gate(msg)
        alpha = self.normalize_weights(gate, index, size_i, mask_j)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha * msg


class TemporalAdditiveAttention(AdditiveAttention):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        output_size: int,
        msg_size: Optional[int] = None,
        msg_layers: int = 1,
        root_weight: bool = True,
        reweight: Optional[str] = None,
        norm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("dim", 1)
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            msg_size=msg_size,
            msg_layers=msg_layers,
            root_weight=root_weight,
            reweight=reweight,
            dropout=dropout,
            norm=norm,
            **kwargs,
        )

    def forward(
        self,
        x: PairTensor,
        mask: OptTensor = None,
        temporal_mask: OptTensor = None,
        causal_lag: Optional[int] = None,
    ):
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        l, s = x_tgt.size(self.node_dim), x_src.size(self.node_dim)
        i = torch.arange(l, dtype=torch.long, device=x_src.device)
        j = torch.arange(s, dtype=torch.long, device=x_src.device)
        if temporal_mask is None and isinstance(causal_lag, int):
            temporal_mask = tuple(torch.tril_indices(l, l, offset=-causal_lag, device=x_src.device))
        if temporal_mask is not None:
            assert temporal_mask.size() == (l, s)
            i, j = torch.meshgrid(i, j)
            edge_index = torch.stack((j[temporal_mask], i[temporal_mask]))
        else:
            edge_index = torch.cartesian_prod(j, i).T

        return super().forward(x, edge_index, mask=mask)
