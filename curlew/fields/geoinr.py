"""
Neural fields ported from GeoINR: a plain multi-layer perceptron (`GeoINR`) and
a sine-activation network (`Siren`), both implemented as `curlew.fields.BaseNF`
subclasses so they slot into Curlew's events/constraints/fit machinery.

Loss and fit are inherited from `curlew.fields.BaseNF` unchanged — these classes
only build the network (`initField`) and define the forward evaluation
(`evaluate`).
"""

import curlew
import numpy as np
import torch
import torch.nn as nn
from curlew.fields import BaseNF


class GeoINR(BaseNF):
    """
    GeoINR-inspired neural field for interpolation of geological structures.

    A plain multi-layer perceptron mapping coordinates to a scalar potential.
    Hidden layers use a smooth activation (``Softplus`` by default); the final
    layer is linear. Weights use Kaiming-uniform initialisation (``fan_in``,
    ReLU gain), mirroring GeoINR's ``Perceptron``.

    See Hillier et al., 2023 for further details:

    `Hillier, Michael, et al. "GeoINR 1.0: an implicit neural network approach to three-dimensional geological modelling." Geoscientific Model Development 16.23 (2023): 6987-7012.`
    """

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


class Siren(BaseNF):
    """
    Sine-activation neural field (ports GeoINR's ``Siren``; Sitzmann et al., 2020).

    A stack of `_SineLayer`s (``sin(omega_0 * Wx + b)``) followed by a linear
    output layer. The first layer is scaled by ``omega0`` and subsequent layers
    by ``omega``; weight bounds follow the standard SIREN initialisation so
    activations stay well-conditioned through depth.
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
