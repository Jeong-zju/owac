"""dNRI encoder/decoder port (R06).

Upstream (verbatim copies kept next to this file)::

    components/upstream/dnri/dnri/models/dnri.py
    components/upstream/dnri/dnri/models/encoders.py
    components/upstream/dnri/dnri/models/decoders.py
    components/upstream/dnri/dnri/models/model_utils.py   (supplementary, see README)
    repository cgraber/cvpr_dNRI @ 96c94694608fcf2a2b372699a74e5c57be3c89f9

Port differences (also in ``third_party.lock.json``):

* ``params`` stays a plain dict with the upstream key names, and every submodule keeps the
  upstream attribute name (``mlp1``..``mlp4``, ``forward_rnn``, ``reverse_rnn``,
  ``prior_fc_out``, ``encoder_fc_out``, ``msg_fc1``/``msg_fc2``, ``hidden_r``/``hidden_i``/
  ``hidden_n``, ...), so an upstream ``state_dict`` loads with ``strict=True``.
* ``RefNRIMLP`` (upstream ``models/model_utils.py``) is re-implemented here because the file was
  not part of the 52-file audit cache; the implementation follows the pinned upstream file kept
  in ``components/upstream/dnri/dnri/models/model_utils.py``.  It is exercised directly by the
  parity suite, so any divergence shows up as a numerical mismatch.
* The upstream ``encoder_save_eval_memory`` branch calls ``.cuda()`` unconditionally in the
  reverse pass, so it cannot run on CPU; the port always uses the batched path and documents
  that (the two paths are the same computation).
* **No time leakage**: :meth:`DNRIEncoder.online_step` / :meth:`DNRIEncoder.online_sequence`
  use only the forward RNN and the prior head.  The bidirectional
  :meth:`DNRIEncoder.forward_bidirectional` is the *training* posterior and must never be used
  online; the parity suite asserts that perturbing future frames leaves every already-produced
  online output bit-identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_REPOSITORY = "cgraber/cvpr_dNRI"
UPSTREAM_COMMIT = "96c94694608fcf2a2b372699a74e5c57be3c89f9"

DEFAULT_PARAMS: dict[str, Any] = {
    "num_vars": 4,
    "num_edge_types": 2,
    "input_size": 3,
    "encoder_hidden": 16,
    "encoder_rnn_hidden": 16,
    "encoder_rnn_type": "lstm",
    "encoder_dropout": 0.0,
    "encoder_mlp_num_layers": 1,
    "prior_num_layers": 1,
    "decoder_hidden": 16,
    "decoder_dropout": 0.0,
    "skip_first": True,
    "gpu": False,
}


def encode_onehot(labels) -> np.ndarray:
    """Upstream ``model_utils.encode_onehot`` (one row per label, one column per class)."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    return np.array(list(map(classes_dict.get, labels)), dtype=np.int32)


class RefNRIMLP(nn.Module):
    """Upstream ``model_utils.RefNRIMLP``: two-layer ELU net with optional BatchNorm1d."""

    def __init__(
        self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.0, no_bn: bool = False
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True),
        )
        self.bn = None if no_bn else nn.BatchNorm1d(n_out)
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs: torch.Tensor) -> torch.Tensor:
        orig_shape = inputs.shape
        x = inputs.reshape(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        return x


class DNRIEncoder(nn.Module):
    """Upstream ``dnri.DNRI_Encoder``: prior (online) and posterior (bidirectional) heads."""

    def __init__(self, params: dict[str, Any]):
        super().__init__()
        num_vars = params["num_vars"]
        self.num_edges = params["num_edge_types"]
        self.separate_prior_encoder = params.get("separate_prior_encoder", False)
        dropout = params["encoder_dropout"]
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(
            torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False
        )
        self.save_eval_memory = params.get("encoder_save_eval_memory", False)

        hidden_size = params["encoder_hidden"]
        rnn_hidden_size = params["encoder_rnn_hidden"] or hidden_size
        rnn_type = params["encoder_rnn_type"]
        inp_size = params["input_size"]
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout)

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.forward_rnn = rnn_cls(hidden_size, rnn_hidden_size, batch_first=True)
        self.reverse_rnn = rnn_cls(hidden_size, rnn_hidden_size, batch_first=True)
        self.rnn_type = rnn_type
        out_hidden_size = 2 * rnn_hidden_size
        self.encoder_fc_out = _fc_stack(
            out_hidden_size,
            params["encoder_mlp_num_layers"],
            params.get("encoder_mlp_hidden"),
            self.num_edges,
        )
        self.prior_fc_out = _fc_stack(
            rnn_hidden_size,
            params["prior_num_layers"],
            params.get("prior_hidden_size"),
            self.num_edges,
        )
        self.num_vars = num_vars
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if node_embeddings.dim() == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        if edge_embeddings.dim() == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(
                old_shape[0], -1, old_shape[2], old_shape[3]
            )
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming / (self.num_vars - 1)

    def copy_states(self, prior_state):
        if isinstance(prior_state, (tuple, list)):
            return (prior_state[0].clone(), prior_state[1].clone())
        return prior_state.clone()

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], (tuple, list)):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        return torch.cat(hidden, dim=0)

    def _node_mlp_stack(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)
        return self.mlp4(x)

    def forward_bidirectional(self, inputs: torch.Tensor):
        """**Training posterior** (uses the whole sequence, including future frames)."""
        x = inputs.transpose(2, 1).contiguous()  # [B, num_vars, T, features]
        x = self._node_mlp_stack(x)
        old_shape = x.shape
        x = x.contiguous().view(-1, old_shape[2], old_shape[3])
        forward_x, prior_state = self.forward_rnn(x)
        timesteps = old_shape[2]
        reverse_x = x.flip(1)
        reverse_x, _ = self.reverse_rnn(reverse_x)
        reverse_x = reverse_x.flip(1)
        prior_result = (
            self.prior_fc_out(forward_x)
            .view(old_shape[0], old_shape[1], timesteps, self.num_edges)
            .transpose(1, 2)
            .contiguous()
        )
        combined_x = torch.cat([forward_x, reverse_x], dim=-1)
        encoder_result = (
            self.encoder_fc_out(combined_x)
            .view(old_shape[0], old_shape[1], timesteps, self.num_edges)
            .transpose(1, 2)
            .contiguous()
        )
        return prior_result, encoder_result, prior_state

    def single_step_forward(self, inputs: torch.Tensor, prior_state):
        """Upstream ``single_step_forward``: forward RNN + prior head only (no future frames)."""
        x = self._node_mlp_stack(inputs)
        old_shape = x.shape
        x = x.contiguous().view(-1, 1, old_shape[-1])
        old_prior_shape = prior_state[0].shape
        prior_state = (
            prior_state[0].view(1, old_prior_shape[0] * old_prior_shape[1], old_prior_shape[2]),
            prior_state[1].view(1, old_prior_shape[0] * old_prior_shape[1], old_prior_shape[2]),
        )
        x, prior_state = self.forward_rnn(x, prior_state)
        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        prior_state = (
            prior_state[0].view(old_prior_shape),
            prior_state[1].view(old_prior_shape),
        )
        return prior_result, prior_state

    def init_prior_state(self, batch_size: int, device=None):
        """Zero RNN state of the shape ``single_step_forward`` expects.

        The forward RNN runs over ``batch * num_graph_edges`` sequences, where the graph edges are
        the ``num_vars * (num_vars - 1)`` directed pairs (not the ``num_edge_types`` categories).
        """
        hidden = self.forward_rnn.hidden_size
        layers = self.forward_rnn.num_layers
        num_graph_edges = len(self.send_edges)
        zeros = torch.zeros(layers, batch_size * num_graph_edges, hidden, device=device)
        if self.rnn_type == "lstm":
            return (zeros.clone(), zeros.clone())
        return zeros

    def online_step(self, inputs: torch.Tensor, prior_state):
        """One online step: prior head only. ``inputs`` is ``[B, num_vars, features]``."""
        return self.single_step_forward(inputs, prior_state)

    def online_sequence(self, inputs: torch.Tensor):
        """Online rollout over ``[B, T, num_vars, features]`` using the prior head only."""
        batch, timesteps = inputs.shape[0], inputs.shape[1]
        state = self.init_prior_state(batch, device=inputs.device)
        priors = []
        for t in range(timesteps):
            prior_result, state = self.online_step(inputs[:, t], state)
            priors.append(prior_result)
        return torch.stack(priors, dim=1), state


class DNRIDecoder(nn.Module):
    """Upstream ``dnri.DNRI_Decoder`` (one recurrent step, GRU-style gated aggregation)."""

    def __init__(self, params: dict[str, Any]):
        super().__init__()
        self.num_vars = params["num_vars"]
        input_size = params["input_size"]
        n_hid = params["decoder_hidden"]
        edge_types = params["num_edge_types"]
        skip_first = params["skip_first"]
        out_size = params["input_size"]

        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)
        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)
        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        self.dropout_prob = params["decoder_dropout"]
        edges = np.ones(self.num_vars) - np.eye(self.num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(
            torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False
        )

    def get_initial_hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """Upstream verbatim: ``inputs`` is ``[batch, num_timesteps, num_vars, input_size]``.

        The initial hidden state is ``[batch, num_vars, msg_out_shape]``; upstream reads
        ``inputs.size(2)`` for the node axis, so the 4-D layout is the one that must be passed.
        """
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, edges: torch.Tensor):
        dropout_prob = self.dropout_prob if self.training else 0.0
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device
        )
        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i : i + 1]
            all_msgs = all_msgs + msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / (self.num_vars - 1)

        inp_r = self.input_r(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)
        pred = self.out_fc3(pred)
        return inputs + pred, hidden


def _fc_stack(in_size: int, num_layers: int, tmp_hidden: int | None, out_size: int) -> nn.Module:
    if num_layers == 1:
        return nn.Linear(in_size, out_size)
    layers: list[nn.Module] = [nn.Linear(in_size, tmp_hidden), nn.ELU(inplace=True)]
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(tmp_hidden, tmp_hidden))
        layers.append(nn.ELU(inplace=True))
    layers.append(nn.Linear(tmp_hidden, out_size))
    return nn.Sequential(*layers)


__all__ = [
    "DEFAULT_PARAMS",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "DNRIDecoder",
    "DNRIEncoder",
    "RefNRIMLP",
    "encode_onehot",
]
