"""Unit tests cho FrontierScorer."""
import math
import pytest
from webexp.explore.core.scoring import FrontierScorer
from webexp.explore.core.node import Node
from webexp.explore.core.task import Task
from webexp.explore.core.trajectory import Trajectory


def _make_task(goal: str, n_positive: int, n_negative: int) -> Task:
    """Helper: tạo Task với số lượng trajectory chỉ định."""
    pos = [
        Trajectory(
            steps=[], goal=goal, success=True, reward=1.0,
            action="", observation={}, agent_info={}, misc={},
        )
        for _ in range(n_positive)
    ]
    neg = [
        Trajectory(
            steps=[], goal=goal, success=False, reward=0.0,
            action="", observation={}, agent_info={}, misc={},
        )
        for _ in range(n_negative)
    ]
    return Task(goal=goal, positive_trajs=pos, negative_trajs=neg,
                exp_dir="/tmp/test", misc={})


def _make_node(
    url: str = "http://test.com",
    exploration_tasks: dict = None,
    total_trajs: int = 0,
    successful_trajs: int = 0,
    success_rate: float = 0.0,
    exploration_count: int = 0,
    embedding: list = None,
    parent_url: str = None,
    parent: Node = None,
) -> Node:
    return Node(
        url=url,
        tasks={},
        exploration_tasks=exploration_tasks or {},
        children=[],
        description="",
        prefixes=[],
        visited=False,
        exp_dir="/tmp/test",
        misc={},
        exploration_count=exploration_count,
        success_rate=success_rate,
        total_trajs=total_trajs,
        successful_trajs=successful_trajs,
        embedding=embedding,
        parent_url=parent_url,
        parent=parent,
    )


class TestUncertainty:
    def test_fresh_node_returns_1(self):
        """Node không có exploration tasks → U = 1.0"""
        scorer = FrontierScorer()
        node = _make_node()
        assert scorer.uncertainty(node) == 1.0

    def test_one_task_one_traj(self):
        """1 task, 1 positive → SR=1.0 → mu=1.0, sigma=0 → U=0"""
        scorer = FrontierScorer()
        node = _make_node(exploration_tasks={
            "t1": _make_task("t1", n_positive=1, n_negative=0),
        })
        assert scorer.uncertainty(node) == 0.0

    def test_two_tasks_split(self):
        """2 tasks: SR=1.0, SR=0.0 → mu=0.5, sigma=0.25 → U=0.5"""
        scorer = FrontierScorer()
        node = _make_node(exploration_tasks={
            "t1": _make_task("t1", n_positive=1, n_negative=0),
            "t2": _make_task("t2", n_positive=0, n_negative=1),
        })
        assert abs(scorer.uncertainty(node) - 0.5) < 0.01

    def test_two_tasks_equal(self):
        """2 tasks both SR=0.5 → sigma=0 → U=0"""
        scorer = FrontierScorer()
        node = _make_node(exploration_tasks={
            "t1": _make_task("t1", n_positive=1, n_negative=1),
            "t2": _make_task("t2", n_positive=1, n_negative=1),
        })
        assert scorer.uncertainty(node) == 0.0

    def test_three_tasks_high_variance(self):
        """3 tasks: SR=1.0, 0.5, 0.0 → mu=0.5, sigma=0.1667, U≈0.333"""
        scorer = FrontierScorer()
        node = _make_node(exploration_tasks={
            "t1": _make_task("t1", n_positive=2, n_negative=0),
            "t2": _make_task("t2", n_positive=1, n_negative=1),
            "t3": _make_task("t3", n_positive=0, n_negative=2),
        })
        U = scorer.uncertainty(node)
        assert abs(U - 0.333) < 0.01, f"Expected ~0.333, got {U}"


class TestValue:
    def test_fresh_node(self):
        """Node mới → SR=0.5 (default), n=0 → V = 0.5/log(2)"""
        scorer = FrontierScorer()
        node = _make_node()
        expected = 0.5 / math.log(2)
        assert abs(scorer.value(node) - expected) < 0.01

    def test_with_history(self):
        """SR=0.8, n=10 → V = 0.8/log(12)"""
        scorer = FrontierScorer()
        node = _make_node(total_trajs=10, successful_trajs=8,
                          success_rate=0.8, exploration_count=10)
        expected = 0.8 / math.log(12)
        assert abs(scorer.value(node) - expected) < 0.01

    def test_decay_over_time(self):
        """Càng khai thác nhiều, V càng giảm"""
        scorer = FrontierScorer()
        node_low = _make_node(total_trajs=1, successful_trajs=1,
                              success_rate=1.0, exploration_count=1)
        node_high = _make_node(total_trajs=100, successful_trajs=100,
                               success_rate=1.0, exploration_count=100)
        assert scorer.value(node_low) > scorer.value(node_high)


    def test_fresh_node_inherits_nearest_parent_success_rate(self):
        scorer = FrontierScorer()
        parent = _make_node(
            url="http://parent.com",
            total_trajs=10,
            successful_trajs=8,
            success_rate=0.8,
        )
        child = _make_node(
            url="http://child.com",
            parent_url=parent.url,
            parent=parent,
        )

        expected = 0.8 / math.log(2)
        assert abs(scorer.value(child) - expected) < 0.01
        sr, source = scorer._estimate_success_rate_with_source(child)
        assert sr == 0.8
        assert source == "ancestor:http://parent.com"

    def test_fresh_node_inherits_nearest_observed_ancestor(self):
        scorer = FrontierScorer()
        grandparent = _make_node(
            url="http://grandparent.com",
            total_trajs=5,
            successful_trajs=3,
            success_rate=0.6,
        )
        parent = _make_node(
            url="http://parent.com",
            parent_url=grandparent.url,
            parent=grandparent,
        )
        child = _make_node(
            url="http://child.com",
            parent_url=parent.url,
            parent=parent,
        )

        sr, source = scorer._estimate_success_rate_with_source(child)
        assert sr == 0.6
        assert source == "ancestor:http://grandparent.com"

    def test_fresh_node_without_parent_reference_uses_prior(self):
        scorer = FrontierScorer()
        node = _make_node(parent_url="http://missing-parent.com")
        sr, source = scorer._estimate_success_rate_with_source(node)
        assert sr == 0.5
        assert source == "prior"


class TestDiversity:
    def test_no_embedding_returns_05(self):
        scorer = FrontierScorer()
        node = _make_node()
        assert scorer.diversity(node) == 0.5

    def test_orthogonal_vectors(self):
        """2 vector trực giao → cos_dist = 1.0"""
        scorer = FrontierScorer()
        node = _make_node(embedding=[[1, 0, 0], [0, 1, 0]])
        assert abs(scorer.diversity(node) - 1.0) < 0.01

    def test_identical_vectors(self):
        """2 vector giống hệt → cos_dist = 0.0"""
        scorer = FrontierScorer()
        node = _make_node(embedding=[[1, 0, 0], [1, 0, 0]])
        assert abs(scorer.diversity(node) - 0.0) < 0.01

    def test_three_vectors_mixed(self):
        """[1,0], [0,1], [1,0] → dists: 1.0, 1.0 → mean=1.0"""
        scorer = FrontierScorer()
        node = _make_node(embedding=[[1, 0], [0, 1], [1, 0]])
        assert abs(scorer.diversity(node) - 1.0) < 0.01

    def test_single_embedding_returns_05(self):
        """Chỉ 1 embedding → không tính được D → 0.5"""
        scorer = FrontierScorer()
        node = _make_node(embedding=[[1, 0, 0]])
        assert scorer.diversity(node) == 0.5

    def test_empty_embedding_list_returns_05(self):
        scorer = FrontierScorer()
        node = _make_node(embedding=[])
        assert scorer.diversity(node) == 0.5


class TestComposite:
    def test_breakdown_structure(self):
        scorer = FrontierScorer(alpha=1.0, beta=2.0, theta=0.5)
        node = _make_node()
        b = scorer.breakdown(node)
        assert all(k in b for k in (
            "score", "U", "V", "V_sr", "V_sr_source", "D", "alpha", "beta", "theta"
        ))

    def test_breakdown_score_matches_compute(self):
        scorer = FrontierScorer(alpha=1.0, beta=2.0, theta=0.5)
        node = _make_node()
        assert abs(scorer.compute(node) - scorer.breakdown(node)["score"]) < 1e-10

    def test_zero_weights(self):
        """Tất cả weights = 0 → score = 0"""
        scorer = FrontierScorer(alpha=0.0, beta=0.0, theta=0.0)
        node = _make_node()
        assert scorer.compute(node) == 0.0

    def test_serialization_roundtrip(self):
        orig = FrontierScorer(alpha=2.0, beta=0.5, theta=0.0, epsilon=1e-6)
        restored = FrontierScorer.from_dict(orig.to_dict())
        assert restored.alpha == 2.0
        assert restored.beta == 0.5
        assert restored.theta == 0.0
        assert restored.epsilon == 1e-6
