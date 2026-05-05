"""
quantum_model.py — Hybrid Quantum-Classical Model
Input: 13 raw heart disease features → binary classification (0/1)

Architecture:
  1. Classical encoder  : 13 → num_qubits  (compress + regularise with Dropout)
  2. Quantum circuit    : ZZFeatureMap + RealAmplitudes ansatz → scalar output
  3. Classical head     : (q_out + enc) → sigmoid probability

The residual concat of (q_out, enc) in the head ensures the classical
gradient path is never fully blocked, preventing barren-plateau collapse.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "config.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import config

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class HybridModel(nn.Module):

    def __init__(self, num_qubits: int | None = None):
        super().__init__()
        nq = num_qubits if num_qubits is not None else config.NUM_QUBITS
        self.num_qubits = nq

        # ── 1. Classical encoder (13 → nq) ────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, nq),
            nn.Tanh(),          # bound to [-1,1] → safe angle encoding
        )

        # ── 2. Quantum circuit ─────────────────────────────────────────────
        # ZZFeatureMap captures pairwise feature entanglement (better than ZFeatureMap)
        feature_map = ZZFeatureMap(nq, reps=config.QUANTUM_REPS)
        ansatz      = RealAmplitudes(nq, reps=config.QUANTUM_REPS, entanglement="full")

        qc = QuantumCircuit(nq)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz,      inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=Estimator(),
        )
        self.qnn = TorchConnector(qnn, initial_weights=None)

        # ── 3. Classical head with residual skip ───────────────────────────
        # Input = quantum output (1) + encoded features (nq)
        self.head = nn.Sequential(
            nn.Linear(1 + nq, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode raw features to qubit space
        enc = self.encoder(x)                               # (B, nq)  in [-1,1]

        # Map Tanh output [-1,1] → [0, π] for stable rotation angles
        q_in = (enc + 1.0) * (np.pi / 2.0)                 # (B, nq)

        # Quantum forward pass
        q_out = self.qnn(q_in)                              # (B, 1)

        # Residual concat: quantum signal + classical encoding
        combined = torch.cat([q_out, enc], dim=1)           # (B, 1+nq)

        return self.head(combined)                           # (B, 1)