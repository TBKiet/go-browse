"""
Frontier scoring module for Go-Browse exploration.

Implements the composite scoring function:
    S = α·U + β·V + θ·D

Where:
    U (Uncertainty) = σ / (μ + ε)  — variance/mean of task success rates
    V (Value)       = SR · (1 / log(n + 2))  — success rate with visit penalty
    D (Diversity)   = mean cosine distance between consecutive state embeddings
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
import math
import logging

if TYPE_CHECKING:
    from .node import Node

logger = logging.getLogger(__name__)


class FrontierScorer:
    """Computes frontier scores for unexplored nodes.

    Usage:
        scorer = FrontierScorer(alpha=1.0, beta=1.0, theta=1.0)
        score = scorer.compute(node)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        theta: float = 1.0,
        epsilon: float = 1e-5,
    ):
        """
        Args:
            alpha: Weight for Uncertainty (U).
            beta:   Weight for Value (V).
            theta:  Weight for Diversity (D).
            epsilon: Small constant to avoid division by zero.
        """
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, node: Node) -> float:
        """Compute composite frontier score S = α·U + β·V + θ·D for a node."""
        U = self.uncertainty(node)
        V = self.value(node)
        D = self.diversity(node)
        return self.alpha * U + self.beta * V + self.theta * D

    def breakdown(self, node: Node) -> dict:
        """Return a dict with the score and its components (for logging/debugging)."""
        U = self.uncertainty(node)
        V = self.value(node)
        D = self.diversity(node)
        S = self.alpha * U + self.beta * V + self.theta * D
        return {
            "score": S,
            "U": U,
            "V": V,
            "D": D,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
        }

    # ------------------------------------------------------------------
    # Component: Uncertainty  U = σ / (μ + ε)
    # ------------------------------------------------------------------

    def uncertainty(self, node: Node) -> float:
        """Uncertainty — higher means more potential for novel tasks.

        U = σ / (μ + ε)

        σ: variance of task success rates at this node.
        μ: mean of those success rates.
        ε: small constant.

        Fresh nodes (no exploration tasks) get U = 1.0 to encourage exploration.
        """
        if not node.exploration_tasks:
            return 1.0

        task_scores = []
        for task in node.exploration_tasks.values():
            total = len(task.positive_trajs) + len(task.negative_trajs)
            if total > 0:
                task_scores.append(len(task.positive_trajs) / total)
            else:
                task_scores.append(0.5)

        if not task_scores:
            return 0.5

        n = len(task_scores)
        mu = sum(task_scores) / n
        sigma = sum((s - mu) ** 2 for s in task_scores) / n  # variance
        return sigma / (mu + self.epsilon)

    # ------------------------------------------------------------------
    # Component: Value  V = SR · (1 / log(n + 2))
    # ------------------------------------------------------------------

    def value(self, node: Node) -> float:
        """Value (exploitation) — higher means this node has been fruitful.

        V = SR · (1 / log(n + 2))

        SR: success rate of trajectories from this node.
        n:  exploration count (how many times visited/sampled).
        The log penalty reduces value for over-exploited nodes.
        """
        sr = node.success_rate if node.total_trajs > 0 else 0.5
        n = max(node.exploration_count, 0)
        penalty = 1.0 / math.log(n + 2)
        return sr * penalty

    # ------------------------------------------------------------------
    # Component: Diversity  D = mean cosine distance between embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_distance(a: List[float], b: List[float], eps: float = 1e-10) -> float:
        """Cosine distance between two vectors: 1 - cos_sim(a, b)."""
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        cos_sim = dot / (norm_a * norm_b + eps)
        return 1.0 - cos_sim

    def diversity(self, node: Node) -> float:
        """Diversity — higher means more UI/DOM state change during interaction.

        D = mean cosine distance between consecutive embedding vectors.

        Returns 0.5 when no embeddings are available (neutral tiebreaker).
        """
        if node.embedding is None or len(node.embedding) < 2:
            return 0.5

        distances = []
        for i in range(len(node.embedding) - 1):
            dist = self.cosine_distance(node.embedding[i], node.embedding[i + 1])
            distances.append(dist)
        return sum(distances) / len(distances) if distances else 0.5

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrontierScorer":
        return cls(
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 1.0),
            theta=d.get("theta", 1.0),
            epsilon=d.get("epsilon", 1e-5),
        )
