"""
Neural fields ported from GeoINR (Hillier et al., 2023): coordinate-based INR networks
(no Fourier-feature encoding; coordinates are expected pre-normalised to a ~unit range)
paired with GeoINR's **gradient-normalised** loss terms.

`GeoINR` is the base of the family: it builds a plain multi-layer perceptron and carries
the shared GeoINR loss terms (see `GeoINR.loss`); subclasses swap only the network
architecture — `Siren` replaces the MLP with a sine-activation stack. Fit is inherited
from `curlew.fields.BaseNF` unchanged, and all GeoINR loss terms are **disabled by
default**, so these fields behave exactly like any other `BaseNF` (generic
`curlew.core.HSet`-driven losses) unless the extra weights are set.
"""

import curlew
import numpy as np
import torch
import torch.nn as nn
from curlew.fields import BaseNF


class GeoINR(BaseNF):
    """
    GeoINR-inspired neural field for interpolation of geological structures.

    The base class of the GeoINR family: a plain multi-layer perceptron mapping
    coordinates to a scalar potential (hidden layers use ``Softplus`` by default;
    the final layer is linear; Kaiming-uniform init, mirroring GeoINR's
    ``Perceptron``). Subclasses swap the architecture by overriding ``initField``
    (see `Siren`); the loss terms below are shared by the whole family.

    In addition to the generic `curlew.fields.BaseSF.loss` terms, `loss` adds
    GeoINR's **gradient-normalised** terms, each gated by a constructor weight
    (NOT by `curlew.core.HSet`, which stays generic) and disabled (0) by default:

    - ``eq_norm_weight`` — **interface** loss over the traces in ``CSet.eq``: the
      pairwise residual ``|f(P) - f(P_ref)| / ‖∇f(P)‖`` between points of the *same*
      interface (GeoINR's ``interface_loss_using_pairs``). Behaves like a distance
      from the interface, so the field cannot shrink its output range to satisfy
      the traces trivially. When used, leave ``HSet.eq_loss`` at 0.
    - ``iq_norm_weight`` — **inequality** loss over the pairs in ``CSet.iq``: the
      residual ``(f(P1) - f(P2)) / ‖∇f(P1)‖`` hinged by each pair's relation
      (GeoINR's above/below losses) and averaged over **active violations only**,
      so satisfied pairs do not dilute coherent violation clusters. When used,
      leave ``HSet.iq_loss`` at 0.
    - ``overturn_weight`` — **no-overturn** regularizer sampled on ``CSet.grid``:
      penalises the field decreasing in the younging direction (``CSet.trend``, or
      +last axis). The penalty is the *magnitude* of the downward gradient component
      (NOT normalised to a cosine — the gradient norm is what drives polarity
      correction and keeps the field monotone in the younging direction). Because it
      is magnitude-based it couples to the field's output ``scale``; keep that ~1
      (as the unit-only strat builder does) so the term stays O(1).

    ``norm_samples`` sets how many points are drawn per ``eq`` trace each step for
    the interface loss (``iq`` pairs use the count stored in ``CSet.iq``).

    See Hillier et al., 2023 for further details:

    `Hillier, Michael, et al. "GeoINR 1.0: an implicit neural network approach to three-dimensional geological modelling." Geoscientific Model Development 16.23 (2023): 6987-7012.`
    """

    def __init__(self, *args, eq_norm_weight: float = 0.0, iq_norm_weight: float = 0.0,
                 overturn_weight: float = 0.0, norm_samples: int = 256, **kwargs):
        self.eq_norm_weight = float(eq_norm_weight)
        self.iq_norm_weight = float(iq_norm_weight)
        self.overturn_weight = float(overturn_weight)
        self.norm_samples = int(norm_samples)
        super().__init__(*args, **kwargs)

    def initField(self,
                  hidden_dim: int = 256,
                  num_hidden_layers: int = 3,
                  activation: nn.Module = None,
                  learning_rate: float = 1e-3,
                  **kwargs):
        """
        Initialise and build this neural field.

        Parameters
        ----------
        hidden_dim : int, optional
            Width of each hidden layer. Default 256.
        num_hidden_layers : int, optional
            Number of hidden layers *in addition* to the input projection, so
            the MLP has ``num_hidden_layers + 2`` linear layers in total
            (input projection, ``num_hidden_layers`` hidden, linear output).
            Default 3.
        activation : nn.Module, optional
            Activation applied between hidden layers. Default ``nn.Softplus(beta=20)``.
        learning_rate : float, optional
            Learning rate of the Adam optimiser. Default 1e-3.
        """
        if activation is None:
            activation = nn.Softplus(beta=20)
        self.activation = activation

        # Linear stack: input_dim -> hidden_dim -> ... -> output_dim.
        # Activation between hidden layers; final layer linear (mirrors GeoINR `INR`).
        layers = []
        in_d = self.input_dim
        for _ in range(num_hidden_layers + 1):  # input projection + hidden layers
            lin = nn.Linear(in_d, hidden_dim, device=curlew.device, dtype=curlew.dtype)
            nn.init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')
            layers.append(lin)
            layers.append(self.activation)
            in_d = hidden_dim
        final = nn.Linear(in_d, self.output_dim, device=curlew.device, dtype=curlew.dtype)
        nn.init.kaiming_uniform_(final.weight, mode='fan_in', nonlinearity='relu')
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

        # push onto device and initialise the optimiser
        self.to(curlew.device)
        self.init_optim(lr=learning_rate)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network to create a scalar value or property estimate.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, output_dim), representing the scalar potential.
        """
        return self.scale * self.mlp(x)

    def loss(self, transform=True):
        """
        Compute the loss pebble for this field: the generic `curlew.fields.BaseSF.loss`
        terms plus (when their weights are non-zero) the GeoINR normalised interface,
        normalised inequality, and no-overturn terms described on `GeoINR`.
        """
        pebble = super().loss(transform=transform)
        C = self.C
        if C is None:
            return pebble

        # gradient-normalised interface loss over CSet.eq traces: pairwise |Δf|/‖∇f‖
        # between points of the same interface (GeoINR's interface_loss_using_pairs)
        if (C.eq is not None) and (self.eq_norm_weight > 0):
            ns = self.norm_samples
            p_list, r_list = [], []
            for trace in C.eq:
                n = trace.shape[0]
                if n < 2:
                    continue
                pi = torch.randint(0, n, (ns,), device=curlew.device)
                ri = torch.randint(0, n, (ns,), device=curlew.device)
                p_list.append(trace[pi])
                r_list.append(trace[ri])
            if p_list:
                grad, v = self.gradient(torch.cat(p_list, dim=0), normalize=False,
                                        transform=transform, return_value=True,
                                        retain_graph=True, create_graph=True,
                                        accumulate=False)
                vr = self(torch.cat(r_list, dim=0), transform=transform).flatten()
                r = (v.flatten() - vr).abs() / (torch.norm(grad, dim=-1) + 1e-6)
                pebble.push(self.name, 'eq_norm_loss', r.mean(),
                            weight=self.eq_norm_weight, optim=self.optim)

        # gradient-normalised inequality over CSet.iq pairs
        if (C.iq is not None) and (self.iq_norm_weight > 0):
            ns = int(C.iq[0])
            p1_list, p2_list, rels = [], [], []
            for start, end, rel in C.iq[1]:
                if (start.shape[0] == 0) or (end.shape[0] == 0):
                    continue
                six = torch.randint(0, start.shape[0], (ns,), device=curlew.device)
                eix = torch.randint(0, end.shape[0], (ns,), device=curlew.device)
                p1_list.append(start[six])
                p2_list.append(end[eix])
                rels.append(rel if isinstance(rel, str) else str(rel))
            if p1_list:
                grad, v1 = self.gradient(torch.cat(p1_list, dim=0), normalize=False,
                                         transform=transform, return_value=True,
                                         retain_graph=True, create_graph=True,
                                         accumulate=False)
                v2 = self(torch.cat(p2_list, dim=0), transform=transform).flatten()
                r = (v1.flatten() - v2) / (torch.norm(grad, dim=-1) + 1e-6)
                viols = []
                for i, rel in enumerate(rels):
                    ri = r[i * ns:(i + 1) * ns]
                    if '>' in rel:   # f(P1) should exceed f(P2) → penalise negative residuals
                        viols.append(torch.clamp_max(ri, 0).abs())
                    else:            # '<' → penalise positive residuals
                        viols.append(torch.clamp_min(ri, 0))
                e = torch.cat(viols)
                active = e[e > 0]
                if active.numel() > 0:
                    pebble.push(self.name, 'iq_norm_loss', active.mean(),
                                weight=self.iq_norm_weight, optim=self.optim)

        # no-overturn regularizer on grid samples
        if (C.grid is not None) and (self.overturn_weight > 0):
            pts = C.grid.draw(self.transform if transform else None)
            g = self.gradient(pts, normalize=False, transform=transform,
                              retain_graph=True, create_graph=True, accumulate=False)
            if C.trend is not None:
                younging = C.trend / (torch.norm(C.trend) + 1e-8)
                gy = (g * younging[None, :]).sum(dim=-1)
            else:
                gy = g[:, -1]
            pebble.push(self.name, 'overturn_loss', torch.clamp_max(gy, 0).abs().mean(),
                        weight=self.overturn_weight, optim=self.optim)

        return pebble


class _SineLayer(nn.Module):
    """
    A single sine-activation layer (ports GeoINR's ``SineLayer``):
    ``forward = sin(omega_0 * linear(x))``.

    Weight init: first layer ``U(-1/in, 1/in)``; subsequent layers
    ``U(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0)``.
    """

    def __init__(self, in_features: int, out_features: int,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features,
                                device=curlew.device, dtype=curlew.dtype)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = np.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(GeoINR):
    """
    Sine-activation member of the GeoINR family (ports GeoINR's ``Siren``;
    Sitzmann et al., 2020).

    Replaces the base `GeoINR` MLP with a stack of `_SineLayer`s
    (``sin(omega_0 * Wx + b)``) followed by a linear output layer. The first layer
    is scaled by ``omega0`` and subsequent layers by ``omega``; weight bounds follow
    the standard SIREN initialisation so activations stay well-conditioned through
    depth. Losses (including the GeoINR gradient-normalised terms) and evaluation
    are inherited from `GeoINR`.
    """

    def initField(self,
                  hidden_dim: int = 256,
                  num_hidden_layers: int = 3,
                  omega0: float = 2.0,
                  omega: float = 30.0,
                  learning_rate: float = 1e-4,
                  **kwargs):
        """
        Initialise and build this neural field.

        Parameters
        ----------
        hidden_dim : int, optional
            Width of each hidden layer. Default 256.
        num_hidden_layers : int, optional
            Number of hidden sine layers after the first. Default 3.
        omega0 : float, optional
            Frequency scale of the first sine layer. Default 2.0.
        omega : float, optional
            Frequency scale of subsequent sine layers (and the bound on the
            final linear init). Default 30.0.
        learning_rate : float, optional
            Learning rate of the Adam optimiser. Default 1e-4.
        """
        layers = [_SineLayer(self.input_dim, hidden_dim, is_first=True, omega_0=omega0)]
        for _ in range(num_hidden_layers):
            layers.append(_SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega))

        final = nn.Linear(hidden_dim, self.output_dim,
                          device=curlew.device, dtype=curlew.dtype)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

        # push onto device and initialise the optimiser
        self.to(curlew.device)
        self.init_optim(lr=learning_rate)
