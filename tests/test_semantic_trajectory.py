"""Tests for semantic trajectory improvements (Phase 1-3)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from webexp.explore.core.trajectory import Trajectory, TrajectoryStep
from webexp.agents.solver_agent import (
    extract_action_and_thought,
    _extract_full_response,
    _extract_element_metadata,
    sanitize_action,
)


def test_trajectory_step_new_fields():
    obs = {"axtree_txt": "test", "screenshot": np.zeros((100, 100, 3), dtype=np.uint8)}
    step = TrajectoryStep(
        action="click('42')", parsed_action="click('42')",
        thought="I need to click", observation=obs,
        action_nl="Click on the Add to Cart button",
        refined_goal="Purchase a Sony WH-1000XM5 under $350 on Amazon",
        action_reasoning="Right product at the right price",
        bounding_box={"x": 100, "y": 200, "width": 80, "height": 30},
        element_metadata={"bid": "42", "tag": "button", "text": "Add to Cart"},
        page_url_before="https://amazon.com/search",
        page_url_after="https://amazon.com/product",
    )
    assert step.action_nl == "Click on the Add to Cart button"
    assert step.refined_goal == "Purchase a Sony WH-1000XM5 under $350 on Amazon"
    assert step.bounding_box == {"x": 100, "y": 200, "width": 80, "height": 30}
    assert step.element_metadata == {"bid": "42", "tag": "button", "text": "Add to Cart"}
    print("OK: TrajectoryStep new fields")


def test_trajectory_step_roundtrip():
    obs = {"axtree_txt": "test", "screenshot": np.zeros((100, 100, 3), dtype=np.uint8)}
    step = TrajectoryStep(
        action="click('42')", parsed_action="click('42')", thought="test",
        observation=obs, action_nl="Click the button",
        refined_goal="Complete the purchase on Amazon",
        action_reasoning="Makes sense",
        bounding_box={"x": 10, "y": 20, "width": 50, "height": 25},
        element_metadata={"bid": "42", "tag": "button", "text": "Buy Now"},
        page_url_before="https://a.com", page_url_after="https://b.com",
        misc={"key": "value"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        step.save(tmpdir, keep_image_in_memory=True, save_image=True)
        loaded = TrajectoryStep.load(tmpdir, load_image=True)
        assert loaded.action_nl == "Click the button"
        assert loaded.refined_goal == "Complete the purchase on Amazon"
        assert loaded.action_reasoning == "Makes sense"
        assert loaded.bounding_box == {"x": 10, "y": 20, "width": 50, "height": 25}
        assert loaded.element_metadata == {"bid": "42", "tag": "button", "text": "Buy Now"}
        assert loaded.page_url_before == "https://a.com"
        assert loaded.page_url_after == "https://b.com"
    print("OK: TrajectoryStep save/load roundtrip")


def test_trajectory_step_backward_compat():
    old_step_info = {
        "action": "click('42')", "parsed_action": "click('42')",
        "thought": "old thought", "observation": {}, "misc": {"old_key": "old_value"},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "step_info.json"), "w") as f:
            json.dump(old_step_info, f)
        Image.new("RGB", (10, 10)).save(os.path.join(tmpdir, "screenshot.png"))
        loaded = TrajectoryStep.load(tmpdir, load_image=False)
        assert loaded.action == "click('42')"
        assert loaded.action_nl is None
        assert loaded.refined_goal is None
    print("OK: TrajectoryStep backward compat")


def test_trajectory_add_step_semantic():
    traj = Trajectory.from_goal("Test goal")
    traj.add_step(
        action="click('42')", parsed_action="click('42')", thought="test",
        observation={"test": "obs"}, action_nl="Click the button",
        refined_goal="Buy a laptop on Amazon", action_reasoning="Good",
        bounding_box={"x": 1}, element_metadata={"bid": "42"},
        page_url_before="https://a.com", page_url_after="https://b.com",
    )
    step = traj.steps[0]
    assert step.action_nl == "Click the button"
    assert step.refined_goal == "Buy a laptop on Amazon"
    assert step.element_metadata == {"bid": "42"}
    print("OK: Trajectory.add_step semantic fields")


def test_trajectory_save_info():
    traj = Trajectory.from_goal("Test goal")
    traj.misc = {"initial": "data"}
    with tempfile.TemporaryDirectory() as tmpdir:
        traj.save(tmpdir)
        traj.misc["summarized_goal"] = "Summarized task on example.com"
        traj.save_info()
        with open(os.path.join(tmpdir, "traj_info.json"), "r") as f:
            info = json.load(f)
            assert info["misc"]["summarized_goal"] == "Summarized task on example.com"
    print("OK: Trajectory.save_info")


def test_extract_full_response_all_fields():
    raw = '{"thought": "I will click", "action": "click(\'42\')", "action_in_natural_language": "Click on the button", "refined_goal": "Buy a Sony headphone on Amazon"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('42')"
    assert r["action_nl"] == "Click on the button"
    assert r["refined_goal"] == "Buy a Sony headphone on Amazon"
    print("OK: _extract_full_response all fields")


def test_extract_full_response_minimal():
    raw = '{"thought": "test", "action": "click(\'42\')"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('42')"
    assert r["action_nl"] is None
    assert r["refined_goal"] is None
    print("OK: _extract_full_response minimal")


def test_extract_full_response_qwen_think():
    raw = '<think>I should click the button</think>{"thought": "clicking button", "action": "click(\'99\')", "action_in_natural_language": "Click on the submit button", "refined_goal": "Submit the form on the website"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('99')"
    assert r["action_nl"] == "Click on the submit button"
    print("OK: _extract_full_response Qwen think tags")


def test_extract_full_response_json_fence():
    raw = 'Some verbose text```json\n{"thought": "test", "action": "click(\'99\')"}\n```'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('99')"
    print("OK: _extract_full_response JSON fence")


def test_extract_element_metadata():
    axtree = "[42] [LINK] [Sony WH-1000XM5 Wireless Headphones]\n[43] [BUTTON] [Add to Cart]\n[44] [TEXTBOX] [Search]"
    m = _extract_element_metadata(axtree, "click('42')")
    assert m is not None
    assert m["bid"] == "42"
    assert m["tag"] == "LINK"
    assert "Sony" in m["text"]
    print("OK: _extract_element_metadata")


def test_extract_element_metadata_type_action():
    axtree = "[7] [TEXTBOX] [Search Amazon]"
    m = _extract_element_metadata(axtree, "type('7', 'laptop')")
    assert m is not None
    assert m["bid"] == "7"
    assert m["tag"] == "TEXTBOX"
    print("OK: _extract_element_metadata type action")


def test_extract_element_metadata_not_found():
    axtree = "[1] [LINK] [Home]"
    m = _extract_element_metadata(axtree, "scroll('down')")
    assert m is None
    print("OK: _extract_element_metadata no bid in action")


def test_extract_action_and_thought_still_works():
    raw = '{"thought": "I should click", "action": "click(\'42\')", "action_in_natural_language": "Click the button", "refined_goal": "Buy the item on Amazon"}'
    action, thought = extract_action_and_thought(raw)
    assert action == "click('42')"
    assert thought == "I should click"
    print("OK: extract_action_and_thought still works")


def test_sanitize_action():
    assert sanitize_action("click('42')\n") == "click('42')"
    assert sanitize_action("​click('42')") == "click('42')"
    assert sanitize_action("  click('42')  ") == "click('42')"
    print("OK: sanitize_action")


if __name__ == "__main__":
    tests = [
        test_trajectory_step_new_fields,
        test_trajectory_step_roundtrip,
        test_trajectory_step_backward_compat,
        test_trajectory_add_step_semantic,
        test_trajectory_save_info,
        test_extract_full_response_all_fields,
        test_extract_full_response_minimal,
        test_extract_full_response_qwen_think,
        test_extract_full_response_json_fence,
        test_extract_element_metadata,
        test_extract_element_metadata_type_action,
        test_extract_element_metadata_not_found,
        test_extract_action_and_thought_still_works,
        test_sanitize_action,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAILED: {test.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(0 if failed == 0 else 1)
