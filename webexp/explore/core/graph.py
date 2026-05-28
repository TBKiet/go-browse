from __future__ import annotations
from .node import Node
from .trace import Trace
from .trajectory import TrajectoryStep
from .scoring import FrontierScorer, LookaheadPredictor
from typing import Sequence, Optional
import json
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Graph:
    def __init__(
            self, 
            root_url: str, 
            exp_dir: str, 
            allowlist_patterns: Sequence[str] = tuple(), 
            denylist_patterns: Sequence[str] = tuple(), 
            resume: bool=False,
            # Frontier scoring addition: hyperparameters for S = alpha*U + beta*V + theta*D.
            alpha: float = 1.0,   # Weight for Uncertainty
            beta: float = 1.0,    # Weight for Value (exploitation)
            theta: float = 1.0,   # Weight for Diversity
            epsilon: float = 1e-5, # Small constant to avoid division by zero
            # Frontier scoring addition: optional zero-shot U prior.
            lookahead_model: Optional[str] = None,
        ):

        self.nodes = {}
        self.explored_nodes = []
        self.unexplored_nodes = []
        self.exp_dir = os.path.join(exp_dir, "graph")
        self.allowlist_patterns = allowlist_patterns
        self.denylist_patterns = denylist_patterns
        # Frontier scoring addition: replaces origin FIFO frontier selection.
        self.scorer = FrontierScorer(
            alpha=alpha, beta=beta, theta=theta, epsilon=epsilon,
        )
        # Frontier scoring addition: lazy-init predictor for U.
        self._lookahead_predictor = None
        self._lookahead_model = lookahead_model

        if not resume:
            self.root = self.add_url(root_url, None, [])

            # Save graph info
            graph_info = {
                "root_url": self.root.url,
                "allowlist_patterns": self.allowlist_patterns,
                "denylist_patterns": self.denylist_patterns,
                "frontier_scoring": self.scorer.to_dict(),
            }
            with open(os.path.join(self.exp_dir, "graph_info.json"), "w") as f:
                json.dump(graph_info, f, indent=4)
        
    def get_node(self, url: str) -> Node | None:
        return self.nodes.get(url, None)
    
    def add_url(self, url: str, parent: Node, prefixes: list[Trace], node_misc: dict = None) -> Node:
        
        if url in self.nodes:
            logger.warning(f"In Graph.add_url: Node {url} already exists in the graph.")
            return self.nodes[url]
        
        node_exp_dir = os.path.join(self.exp_dir, f"node_{len(self.nodes)}")
        # Frontier scoring addition: keep parent links so sibling URLs share V.
        parent_url = parent.url if parent else None
        node = Node(
            url, {}, {}, [], "", prefixes, False, node_exp_dir,
            misc=node_misc,
            parent_url=parent_url,
            parent=parent,
        )
        if parent:
            parent.children.append(node.url)
            parent.update_save(save_prefix=False)
        self.nodes[url] = node
        self.unexplored_nodes.append(node)

        # Frontier scoring addition: collect lookahead confidences for U.
        self._run_lookahead(node)

        return node

    def _run_lookahead(self, node: Node):
        """Run the zero-shot LookaheadPredictor on a newly discovered node.

        This populates node.lookahead_candidates so that FrontierScorer
        can compute Uncertainty (U) without running expensive agents.

        Silently skips if no lookahead model is configured or if the
        predictor fails (node stays scorable with U=1.0).
        """
        if not self._lookahead_model:
            return

        if node.lookahead_candidates is not None:
            return  # Already scored

        # Lazy-init predictor
        if self._lookahead_predictor is None:
            self._lookahead_predictor = LookaheadPredictor(model_name=self._lookahead_model)

        try:
            # We need the axtree for this node. The graph doesn't own the
            # browser env, so we rely on the prefix traces which contain
            # observation snapshots. Use the most recent prefix's last step.
            axtree = None
            if node.prefixes:
                last_prefix = node.prefixes[-1]
                if last_prefix.steps and last_prefix.steps[-1].observation:
                    axtree = last_prefix.steps[-1].observation.get("axtree_txt")

            if not axtree:
                logger.debug(f"Lookahead: no axtree available for {node.url[:80]}, skipping.")
                return

            candidates = self._lookahead_predictor.propose(axtree)
            node.lookahead_candidates = candidates
            node.update_save(save_prefix=False, save_info=True)
            confs = [c["confidence"] for c in candidates]
            logger.info(
                f"Lookahead for '{node.url[:80]}': {len(candidates)} candidates, "
                f"confidences=[{', '.join(f'{c:.2f}' for c in confs)}]"
            )
        except Exception as e:
            logger.warning(f"Lookahead failed for '{node.url[:80]}': {e}")
    
    def add_to_explored(self, node: Node):
        was_unexplored = node in self.unexplored_nodes
        if node not in self.explored_nodes:
            self.explored_nodes.append(node)
        if node in self.unexplored_nodes:
            self.unexplored_nodes.remove(node)
        parent = getattr(node, "parent", None)
        if was_unexplored and parent is not None:
            # Frontier scoring addition: n for child V is counted on the parent.
            parent.selected_child_count = max(getattr(parent, "selected_child_count", 0), 0) + 1
            parent.update_save(save_prefix=False)
        node.visited = True
        node.update_save(save_prefix=False)
        logger.info(f"Node {node.url} has been explored.")

    def get_next_node(self) -> Node | None:
        if len(self.unexplored_nodes) == 0:
            logger.info("No nodes left to explore.")
            return None

        # Frontier scoring addition: rank frontier nodes instead of origin FIFO.
        scored_nodes = [
            (node, self.scorer.compute(node))
            for node in self.unexplored_nodes
        ]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        best_node, best_score = scored_nodes[0]
        breakdown = self.scorer.breakdown(best_node)
        logger.info(
            f"Frontier scoring: selected '{best_node.url[:80]}' with score={best_score:.4f} "
            f"(U={breakdown['U']:.4f}, V={breakdown['V']:.4f}, "
            f"V_sr={breakdown['V_sr']:.4f}, V_sr_source={breakdown['V_sr_source']}, "
            f"V_n={breakdown['V_n']}, V_n_source={breakdown['V_n_source']}, "
            f"D={breakdown['D']:.4f})"
        )

        # Frontier scoring addition: persist selected score for debugging/resume analysis.
        best_node.frontier_score = best_score
        best_node.frontier_breakdown = breakdown
        best_node.update_save(save_prefix=False, save_info=True)

        # Frontier scoring addition: persist full ranking for analysis/debugging.
        self._save_frontier_snapshot(scored_nodes)

        return best_node

    def _save_frontier_snapshot(self, scored_nodes: list):
        """Save a snapshot of the current frontier ranking to the graph directory."""
        snapshot = []
        for node, score in scored_nodes:
            bd = self.scorer.breakdown(node)
            lookahead_confs = (
                [c["confidence"] for c in node.lookahead_candidates]
                if node.lookahead_candidates else None
            )
            snapshot.append({
                "url": node.url[:120],
                "parent_url": getattr(node, "parent_url", None),
                "score": round(score, 6),
                "U": round(bd["U"], 6),
                "V": round(bd["V"], 6),
                "V_sr": round(bd["V_sr"], 6),
                "V_sr_source": bd["V_sr_source"],
                "V_n": bd["V_n"],
                "V_n_source": bd["V_n_source"],
                "D": round(bd["D"], 6),
                "exploration_count": node.exploration_count,
                "selected_child_count": node.selected_child_count,
                "success_rate": node.success_rate,
                "total_trajs": node.total_trajs,
                "num_tasks": len(node.tasks),
                "num_exploration_tasks": len(node.exploration_tasks),
                "lookahead_confidences": lookahead_confs,
            })
        frontier_file = os.path.join(self.exp_dir, "frontier_snapshot.json")
        with open(frontier_file, "w") as f:
            json.dump(snapshot, f, indent=4)

    
    def check_if_url_allowed(self, url: str) -> bool:
        for pattern in self.allowlist_patterns:
            if re.match(pattern, url):
                return True
        for pattern in self.denylist_patterns:
            if re.match(pattern, url):
                return False
        return True


    @staticmethod
    def load(
        path: str,
        load_steps: bool=True,
        load_prefixes: bool=True,
        load_images: bool=True,
        max_nodes=-1,
        lookahead_model: Optional[str] = None,
    ) -> Graph:
        nodes = {}
        explored_nodes = []
        unexplored_nodes = []

        logger.info(f"Loading graph from {path}")

        node_dirs = []
        for name in os.listdir(path):
            match = re.fullmatch(r"node_(\d+)", name)
            if match and os.path.isdir(os.path.join(path, name)):
                node_dirs.append((int(match.group(1)), name))
        node_dirs.sort(key=lambda item: item[0])
        if max_nodes != -1:
            node_dirs = node_dirs[:max_nodes]

        for node_index, node_dir_name in node_dirs:
            logger.info(f"Loading node {node_index} from {path}")
            node_load_dir = os.path.join(path, node_dir_name)
            node = Node.load(node_load_dir, load_steps=load_steps, load_prefix=load_prefixes, load_images=load_images)
            nodes[node.url] = node
            if node.visited:
                explored_nodes.append(node)
            else:
                unexplored_nodes.append(node)
        
        graph_info = {}
        with open(os.path.join(path, "graph_info.json"), "r") as f:
            graph_info = json.load(f)
        
        # Frontier scoring addition: restore saved scorer weights when resuming.
        scoring_dict = graph_info.get("frontier_scoring", {})
        restored = FrontierScorer.from_dict(scoring_dict)
        graph = Graph(
            graph_info["root_url"], path,
            graph_info["allowlist_patterns"], graph_info["denylist_patterns"],
            resume=True,
            alpha=restored.alpha, beta=restored.beta,
            theta=restored.theta, epsilon=restored.epsilon,
        )
        graph.root = nodes[graph_info["root_url"]]
        graph.nodes = nodes
        graph.explored_nodes = explored_nodes
        graph.unexplored_nodes = unexplored_nodes
        graph.exp_dir = path
        graph._lookahead_model = lookahead_model
        for node in graph.nodes.values():
            # Frontier scoring addition: restore runtime parent links from parent_url.
            parent_url = getattr(node, "parent_url", None)
            node.parent = graph.nodes.get(parent_url) if parent_url else None
            node.selected_child_count = 0
        for node in graph.nodes.values():
            if getattr(node, "parent", None) is not None and node.visited:
                # Frontier scoring addition: rebuild parent n from visited children.
                parent = node.parent
                parent.selected_child_count = max(getattr(parent, "selected_child_count", 0), 0) + 1

        logger.info(f"Loaded graph with {len(nodes)} nodes, {len(explored_nodes)} explored nodes, and {len(unexplored_nodes)} unexplored nodes.")
        
        return graph
