"""Pure-Python tests for semantic trajectory parsing functions."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webexp.agents.solver_agent import (
    extract_action_and_thought,
    _extract_full_response,
    _extract_element_metadata,
    sanitize_action,
)
from webexp.explore.core.trajectory import Trajectory, TrajectoryStep


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


def test_extract_full_response_escaped_quotes():
    raw = '{"thought": "He said \\"hello\\"", "action": "click(\'42\')", "action_in_natural_language": "Click the \\"Submit\\" button"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('42')"
    print("OK: _extract_full_response escaped quotes")


def test_extract_full_response_newlines():
    raw = '  {\n  "thought": "test",\n  "action": "click(\'99\')",\n  "action_in_natural_language": "Click the button",\n  "refined_goal": "Complete the task on Amazon"\n}  '
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('99')"
    assert r["action_nl"] == "Click the button"
    print("OK: _extract_full_response newlines/whitespace")


def test_extract_full_response_truncated():
    raw = '{"thought": "long reasoning", "action": "click(\'4'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('4"
    print("OK: _extract_full_response truncated JSON")


def test_extract_full_response_empty():
    assert _extract_full_response("") is None
    assert _extract_full_response(None) is None
    print("OK: _extract_full_response empty input")


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


def test_extract_element_metadata_no_bid():
    axtree = "[1] [LINK] [Home]"
    m = _extract_element_metadata(axtree, "scroll('down')")
    assert m is None
    print("OK: _extract_element_metadata no bid in action")


def test_extract_element_metadata_bid_not_in_tree():
    axtree = "[1] [LINK] [Home]"
    m = _extract_element_metadata(axtree, "click('99')")
    assert m is not None
    assert m["bid"] == "99"
    assert m["tag"] is None
    assert m["text"] is None
    print("OK: _extract_element_metadata bid not in tree")


def test_extract_action_and_thought_with_semantic():
    raw = '{"thought": "I should click", "action": "click(\'42\')", "action_in_natural_language": "Click the button", "refined_goal": "Buy the item on Amazon"}'
    action, thought = extract_action_and_thought(raw)
    assert action == "click('42')"
    assert thought == "I should click"
    print("OK: extract_action_and_thought with semantic fields")


def test_extract_action_and_thought_old_format():
    raw = '{"thought": "old format", "action": "click(\'1\')"}'
    action, thought = extract_action_and_thought(raw)
    assert action == "click('1')"
    assert thought == "old format"
    print("OK: extract_action_and_thought old format")


def test_sanitize_action():
    assert sanitize_action("click('42')\n") == "click('42')"
    assert sanitize_action("​click('42')") == "click('42')"  # zero-width space
    assert sanitize_action("  click('42')  ") == "click('42')"
    assert sanitize_action("click('42')\r\n") == "click('42')"
    print("OK: sanitize_action")


def test_trajectory_step_load_old_format():
    """Backward compat: loading old step_info without new fields."""
    old_info = {
        "action": "click('42')", "parsed_action": "click('42')",
        "thought": "old", "observation": {}, "misc": {},
    }
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "step_info.json"), "w") as f:
            json.dump(old_info, f)
        # Create a dummy PNG
        with open(os.path.join(d, "screenshot.png"), "wb") as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82')
        step = TrajectoryStep.load(d, load_image=False)
        assert step.action == "click('42')"
        assert step.action_nl is None
        assert step.refined_goal is None
        assert step.bounding_box is None
    print("OK: TrajectoryStep.load old format backward compat")


def test_trajectory_add_step_defaults():
    """Old calls to add_step without new fields still work."""
    traj = Trajectory.from_goal("test")
    traj.add_step("click('42')", "click('42')", "thought", {"test": "obs"})
    assert traj.steps[0].action_nl is None
    assert traj.steps[0].refined_goal is None
    print("OK: Trajectory.add_step backward compat")


def test_trajectory_save_info():
    traj = Trajectory.from_goal("Test goal")
    traj.misc = {"initial": "data"}
    with tempfile.TemporaryDirectory() as d:
        traj.save(d)
        traj.misc["summarized_goal"] = "Buy a laptop on Amazon"
        traj.save_info()
        with open(os.path.join(d, "traj_info.json")) as f:
            info = json.load(f)
        assert info["misc"]["summarized_goal"] == "Buy a laptop on Amazon"
    print("OK: Trajectory.save_info")


if __name__ == "__main__":
    tests = [
        test_extract_full_response_all_fields,
        test_extract_full_response_minimal,
        test_extract_full_response_qwen_think,
        test_extract_full_response_escaped_quotes,
        test_extract_full_response_newlines,
        test_extract_full_response_truncated,
        test_extract_full_response_empty,
        test_extract_element_metadata,
        test_extract_element_metadata_type_action,
        test_extract_element_metadata_no_bid,
        test_extract_element_metadata_bid_not_in_tree,
        test_extract_action_and_thought_with_semantic,
        test_extract_action_and_thought_old_format,
        test_sanitize_action,
        test_trajectory_step_load_old_format,
        test_trajectory_add_step_defaults,
        test_trajectory_save_info,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"\nFAILED: {test.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'='*50}")
    print(f"{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(0 if failed == 0 else 1)
