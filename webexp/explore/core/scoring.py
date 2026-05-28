"""
Frontier scoring module for Go-Browse exploration.

Implements the composite scoring function:
    S = α·U + β·V + θ·D

Where:
    U (Uncertainty) = σ / (μ + ε)  — variance/mean of lookahead confidences
    V (Value)       = SR · (1 / log(n + 1))  — parent success rate with sibling penalty
    D (Diversity)   = mean cosine distance between consecutive state embeddings
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List
import math
import logging
from textwrap import dedent
from openai import OpenAI
import os

if TYPE_CHECKING:
    from .node import Node

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lightweight LLM-based confidence predictor for task candidates
# ------------------------------------------------------------------

class LookaheadPredictor:
    """Zero-shot lookahead: uses a small/cheap LLM to propose plausible
    exploration tasks on a page *before* any agent runs.

    Each candidate has a task description, type, and confidence score (0-1).
    The *variance* of these confidence scores becomes the Uncertainty (U)
    component of the frontier score.

    This is the key enabler for scoring nodes in the frontier without
    running expensive PageExplorer/NavExplorer episodes.
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        max_tokens: int = 256,
        num_candidates: int = 6,
    ):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", None),
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.num_candidates = num_candidates

    def propose(self, axtree_snippet: str) -> list[dict]:
        """Propose {num_candidates} useful frontier exploration tasks with
        feasibility confidence scores, based on the page's accessibility tree.

        Returns:
            list[dict]: Each dict has keys "task" (str), "type" (str),
            and "confidence" (float).
        """
        prompt = dedent(f"""\
            You are evaluating a web page state for autonomous web exploration.

            Given the accessibility tree below, propose {self.num_candidates} diverse candidate tasks
            that an agent could attempt from this exact page state.

            Each task should be:
            - local to the current page state, or a direct navigation action visible from this page
            - concrete and executable
            - useful for discovering website functionality or new states
            - not dependent on hidden information

            For each task, provide:
            - "task": a short natural-language goal
            - "type": one of ["local", "navigation", "search", "form", "unknown"]
            - "confidence": a number from 0.0 to 1.0 estimating how likely the task is feasible from this page state

            Use confidence as FEASIBILITY confidence, not user-intent probability.

            Page accessibility tree:
            {axtree_snippet[:3500]}

            Respond with ONLY valid JSON in this exact format:
            {{
              "candidates": [
                {{"task": "...", "type": "local", "confidence": 0.80}}
              ]
            }}
            """)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            # Parse JSON — handle both array and {"candidates": [...]} wrappers
            import json
            data = json.loads(raw)
            if isinstance(data, dict):
                # Try common wrapper keys
                for key in ("candidates", "actions", "predictions"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                logger.warning(f"Lookahead: unexpected response format (not a list): {raw[:200]}")
                return self._fallback()
            # Validate and cap
            results = []
            for item in data[:self.num_candidates]:
                if isinstance(item, dict) and ("task" in item or "action" in item):
                    conf = float(item.get("confidence", 0.5))
                    task = str(item.get("task") or item.get("action"))
                    task_type = str(item.get("type", "unknown"))
                    if task_type not in {"local", "navigation", "search", "form", "unknown"}:
                        task_type = "unknown"
                    results.append({
                        "task": task,
                        "type": task_type,
                        "confidence": max(0.0, min(1.0, conf)),
                    })
            if results:
                return results
        except Exception as e:
            logger.warning(f"Lookahead proposal failed: {e}")

        return self._fallback()

    def _fallback(self) -> list[dict]:
        """Return neutral candidates when prediction fails."""
        return [
            {"task": f"generic_interaction_{i}", "type": "unknown", "confidence": 0.5}
            for i in range(self.num_candidates)
        ]


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
        V_sr, V_sr_source = self._estimate_parent_success_rate_with_source(node)
        V_n, V_n_source = self._parent_selected_child_count_with_source(node)
        V = V_sr * (1.0 / math.log(max(V_n, 1) + 1))
        D = self.diversity(node)
        S = self.alpha * U + self.beta * V + self.theta * D
        return {
            "score": S,
            "U": U,
            "V": V,
            "V_sr": V_sr,
            "V_sr_source": V_sr_source,
            "V_n": V_n,
            "V_n_source": V_n_source,
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

        σ: variance of lookahead confidence scores at this node.
        μ: mean of those confidence scores.
        ε: small constant.

        Data source: `node.lookahead_candidates` — a list of
        {"task": str, "type": str, "confidence": float} produced by the zero-shot
        LookaheadPredictor when the node first entered the frontier.

        Interpretation:
          - High σ (e.g. confidences = [0.1, 0.5, 0.9]): the model is
            uncertain about what interactions are possible → high exploration
            potential.
          - Low σ (e.g. confidences = [0.95, 0.98, 0.99]): the page is
            predictable → low exploration potential.

        Fresh nodes (no lookahead data yet) get U = 1.0 to encourage
        initial exploration.
        """
        candidates = node.lookahead_candidates
        if not candidates:
            return 1.0

        confidences = [c["confidence"] for c in candidates if isinstance(c, dict)]
        if not confidences:
            return 1.0

        n = len(confidences)
        mu = sum(confidences) / n
        sigma = sum((c - mu) ** 2 for c in confidences) / n  # variance
        return sigma / (mu + self.epsilon)

    # ------------------------------------------------------------------
    # Component: Value  V = SR · (1 / log(n + 1))
    # ------------------------------------------------------------------

    def value(self, node: Node) -> float:
        """Value (exploitation) — higher means the parent branch is fruitful.

        V = SR · (1 / log(n + 1))

        SR: success rate of the parent node.
        n:  number of child URLs selected for exploration from that parent.
        Sibling URLs share the same SR and n, so they share the same V.
        """
        sr, _ = self._estimate_parent_success_rate_with_source(node)
        n, _ = self._parent_selected_child_count_with_source(node)
        n = max(n, 1)
        penalty = 1.0 / math.log(n + 1)
        return sr * penalty

    def _estimate_parent_success_rate_with_source(self, node: Node) -> tuple[float, str]:
        """Estimate parent SR and report its source for value scoring.

        Child nodes use their direct parent's observed success_rate. Root nodes
        use their own observed success_rate. If the required node has no
        trajectories yet, use the neutral prior 0.5.
        """
        parent = getattr(node, "parent", None)
        if parent is not None:
            if getattr(parent, "total_trajs", 0) > 0:
                return parent.success_rate, f"parent:{getattr(parent, 'url', None)}"
            return 0.5, f"parent_prior:{getattr(parent, 'url', None)}"

        if getattr(node, "total_trajs", 0) > 0:
            return node.success_rate, "self"

        return 0.5, "prior"

    def _parent_selected_child_count_with_source(self, node: Node) -> tuple[int, str]:
        """Return n for V: selected child count on the direct parent."""
        parent = getattr(node, "parent", None)
        if parent is not None:
            return max(getattr(parent, "selected_child_count", 0), 0), f"parent:{getattr(parent, 'url', None)}"
        return max(getattr(node, "exploration_count", 0), 0), "self"

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
