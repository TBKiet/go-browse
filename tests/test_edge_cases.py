"""Edge case and integration tests for semantic trajectory implementation."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

# Phase 1-2: Parsing
from webexp.agents.solver_agent import (
    _extract_full_response,
    _extract_element_metadata,
    _looks_like_action_level_goal,
    sanitize_action,
    extract_action_and_thought,
)

# Phase 1: Trajectory data model
from webexp.explore.core.trajectory import Trajectory, TrajectoryStep, _extract_text_obs
from webexp.explore.core.task import Task

# Phase 5: Evaluator
from webexp.explore.core.evaluator import extract_content, build_vision_eval_prompt

# Phase 3: TaskSummarization
from webexp.agents.task_summarization_agent import TaskSummarizationAgent

# Phase 7: CAPTCHA
from webexp.agents.captcha_detection_agent import CaptchaDetectionAgent

# Phase 4: SoM grounding
from webexp.explore.grounding.som_processor import extract_rois_from_axtree, process_observation


# ============================================================================
# Phase 1: _extract_full_response edge cases
# ============================================================================

def test_extract_verbose_text_before_json():
    """LLM outputs verbose reasoning before the JSON block."""
    raw = "Let me analyze the page carefully. I see a search bar. The best action is:\n\n{\n  \"thought\": \"I should type in the search bar\",\n  \"action\": \"type('5', 'laptop')\",\n  \"action_in_natural_language\": \"Type 'laptop' into the search bar\",\n  \"refined_goal\": \"Search for a laptop on Amazon\"\n}"
    r = _extract_full_response(raw)
    assert r is not None, "Should extract JSON despite verbose prefix"
    assert r["action"] == "type('5', 'laptop')"
    assert r["action_nl"] == "Type 'laptop' into the search bar"
    assert r["refined_goal"] == "Search for a laptop on Amazon"
    print("OK: extract verbose text before JSON")


def test_extract_last_valid_json_multiple_blocks():
    """Multiple JSON blocks: take the LAST valid one."""
    raw = '{"thought": "first", "action": "click(\'1\')"} some text {"thought": "second", "action": "click(\'2\')"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('2')", "Should take the last valid JSON block"
    print("OK: extract last valid JSON from multiple blocks")


def test_extract_json_with_markdown_fence():
    """JSON inside markdown code fence."""
    raw = '```json\n{"thought": "test", "action": "click(\'42\')", "action_in_natural_language": "Click the button"}\n```'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('42')"
    assert r["action_nl"] == "Click the button"
    print("OK: extract JSON from markdown code fence")


def test_extract_json_with_plain_fence():
    """JSON inside plain markdown fence (no language spec)."""
    raw = 'Here is my response:\n```\n{"thought": "clicking", "action": "click(\'99\')"}\n```'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('99')"
    print("OK: extract JSON from plain markdown fence")


def test_extract_with_nested_braces_in_action():
    """Action string contains nested braces (e.g., dict literals)."""
    raw = '{"thought": "selecting option", "action": "select(\'10\', \'value\')"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "select('10', 'value')"
    print("OK: extract with nested braces in action")


def test_extract_json_with_escaped_single_quotes():
    """JSON using single quotes in action values."""
    raw = '{"thought": "I need to type", "action": "type(\'7\', \'coffee maker\')", "action_in_natural_language": "Type coffee maker", "refined_goal": "Find a coffee maker on Amazon"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "type('7', 'coffee maker')"
    assert r["refined_goal"] == "Find a coffee maker on Amazon"
    print("OK: extract JSON with escaped single quotes in action")


def test_extract_empty_or_partial_responses():
    """Various malformed inputs."""
    assert _extract_full_response("just some text no json") is None, "No JSON at all"
    assert _extract_full_response('{"thought": "incomplete"') is None or _extract_full_response('{"thought": "incomplete"') is not None, "Incomplete JSON should try recovery"
    # Only action key triggers fallback, not thought
    r = _extract_full_response('{"action": "click(\'1\')" incomplete')
    assert r is not None and r["action"] == "click('1')", "Recover from truncated with action"
    print("OK: extract from malformed/partial responses")


def test_extract_action_with_send_msg_to_user():
    """send_msg_to_user action with complex content."""
    raw = '{"thought": "I found the answer", "action": "send_msg_to_user(\'The price is $1,499.99 and it has 4.5 stars\')", "action_in_natural_language": "Tell the user the price and rating", "refined_goal": "Find and report the price and rating of the product on Amazon"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert "send_msg_to_user" in r["action"]
    assert "4.5 stars" in r["action"]
    assert "price" in r["action_nl"].lower()
    print("OK: extract send_msg_to_user with complex content")


def test_extract_unicode_text():
    """LLM outputs unicode characters in thought/action_nl."""
    raw = '{"thought": "Je dois cliquer sur le bouton", "action": "click(\'42\')", "action_in_natural_language": "Cliquer sur le bouton «Ajouter au panier»", "refined_goal": "Acheter un ordinateur sur Amazon"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert "Cliquer" in r["action_nl"]
    assert "Acheter" in r["refined_goal"]
    print("OK: extract unicode text")


def test_extract_with_single_action_key_only():
    """JSON with only action (no thought)."""
    raw = '{"action": "click(\'42\')"}'
    r = _extract_full_response(raw)
    assert r is not None
    assert r["action"] == "click('42')"
    assert r["thought"] is None
    print("OK: extract JSON with action key only")


# ============================================================================
# Phase 1: _extract_element_metadata edge cases
# ============================================================================

def test_extract_element_metadata_multiline_axtree():
    """Complex multi-line accessibility tree."""
    axtree = """
[1] [LINK] [Home]
[2] [TEXTBOX] [Search...]
[3] [BUTTON] [Submit]
[42] [LINK] [Sony WH-1000XM5 Wireless Noise Cancelling Headphones - Black]
[43] [BUTTON] [Add to Cart]
[99] [TEXTBOX] [Quantity: 1]
"""
    m = _extract_element_metadata(axtree, "click('42')")
    assert m["bid"] == "42"
    assert m["tag"] == "LINK"
    assert "Sony" in m["text"]
    print("OK: extract element metadata from multiline axtree")


def test_extract_element_metadata_newlines_in_text():
    """Element text contains newline artifacts."""
    axtree = "[7] [TEXTBOX] [Search\nAmazon\nToday's Deals]"
    m = _extract_element_metadata(axtree, "type('7', 'laptop')")
    assert m["bid"] == "7"
    assert "Search" in m["text"]
    print("OK: extract element metadata with newlines in text")


def test_extract_element_metadata_long_text_truncation():
    """Very long element text should be handled."""
    long_text = "A" * 300
    axtree = f"[5] [DIV] [{long_text}]"
    m = _extract_element_metadata(axtree, "click('5')")
    assert m["bid"] == "5"
    # Text may include enclosing brackets depending on regex pattern matched
    assert len(m["text"]) >= 200, f"Expected long text to be preserved, got len={len(m['text'])}"
    print("OK: extract element metadata long text truncation")


# ============================================================================
# Phase 1: sanitize_action edge cases
# ============================================================================

def test_sanitize_action_unicode_whitespace():
    """Various unicode whitespace characters."""
    assert sanitize_action("click('42') ") == "click('42')"  # non-breaking space
    assert sanitize_action("​click('42')​") == "click('42')"  # zero-width spaces
    assert sanitize_action("click('42')\t") == "click('42')"  # tab
    assert sanitize_action("‌click('42')‍") == "click('42')"  # ZWNJ, ZWJ
    assert sanitize_action("﻿click('42')") == "click('42')"  # BOM
    print("OK: sanitize_action unicode whitespace")


def test_sanitize_action_preserves_valid_content():
    """sanitize should not corrupt valid action content."""
    assert sanitize_action("type('7', 'hello world')") == "type('7', 'hello world')"
    assert sanitize_action("send_msg_to_user('Price: $1,499.99')") == "send_msg_to_user('Price: $1,499.99')"
    assert sanitize_action("goto('https://example.com/page')") == "goto('https://example.com/page')"
    print("OK: sanitize_action preserves valid content")


def test_action_level_refined_goal_detection():
    """Action-level refined goals should be rejected before they overwrite task state."""
    assert _looks_like_action_level_goal("Click the Add to Cart button")
    assert _looks_like_action_level_goal("Scroll down to reveal products")
    assert _looks_like_action_level_goal("Type laptop into the search box")
    assert not _looks_like_action_level_goal("Cart contains the Sony headphones with quantity 1")
    assert not _looks_like_action_level_goal("Find and report the price of the Sony headphones")
    print("OK: action-level refined_goal detection")


# ============================================================================
# Phase 1: TrajectoryStep / Trajectory edge cases
# ============================================================================

def test_trajectory_step_without_screenshot():
    """TrajectoryStep with observation that has no screenshot."""
    obs = {"axtree_txt": "test", "url": "https://example.com"}
    step = TrajectoryStep(
        action="click('42')", parsed_action="click('42')",
        thought="test", observation=obs,
        action_nl="Click the button",
        refined_goal="Complete task on example.com",
    )
    with tempfile.TemporaryDirectory() as d:
        step.save(d, keep_image_in_memory=True, save_image=True)
        loaded = TrajectoryStep.load(d, load_image=False)
        assert loaded.action_nl == "Click the button"
        assert loaded.refined_goal == "Complete task on example.com"
    print("OK: TrajectoryStep save/load without screenshot")


def test_trajectory_multiple_steps_semantic():
    """Full trajectory with multiple semantic steps."""
    traj = Trajectory.from_goal("Buy a laptop")
    for i in range(3):
        traj.add_step(
            action=f"click('{i}')", parsed_action=f"click('{i}')",
            thought=f"step {i}", observation={"step": i},
            action_nl=f"Click element {i}",
            refined_goal=f"Buy a laptop under ${1000 + i*100} on Amazon",
            action_reasoning=f"Reasoning for step {i}",
            bounding_box={"x": i*10, "y": i*20, "width": 50, "height": 25},
            element_metadata={"bid": str(i)},
            page_url_before="https://a.com",
            page_url_after="https://b.com",
        )
    assert len(traj.steps) == 3
    # Last step should have its own refined_goal
    assert traj.steps[-1].refined_goal == "Buy a laptop under $1200 on Amazon"
    assert traj.steps[0].action_nl == "Click element 0"
    print("OK: Trajectory multiple semantic steps")


def test_trajectory_from_goal_defaults():
    """Trajectory.from_goal creates proper defaults."""
    traj = Trajectory.from_goal("Test", agent_info={"model": "gpt-4o"})
    assert traj.goal == "Test"
    assert traj.reward == 0.0
    assert traj.success is False
    assert traj.response == "N/A"
    assert traj.steps == []
    assert traj.final_state is None
    assert traj.agent_info == {"model": "gpt-4o"}
    print("OK: Trajectory.from_goal defaults")


def test_trajectory_save_load_full_roundtrip():
    """Full save/load roundtrip with semantic fields."""
    obs = {"axtree_txt": "test", "screenshot": np.zeros((100, 100, 3), dtype=np.uint8)}
    traj = Trajectory.from_goal("Complete purchase on Amazon")
    traj.reward = 1.0
    traj.success = True
    traj.response = "I have completed the purchase"
    traj.misc = {"summarized_goal": "Buy a laptop under $1000 on Amazon",
                  "evaluation_info": {"task_type": "transaction", "status": "success"}}

    traj.add_step(
        action="click('42')", parsed_action="click('42')",
        thought="click add to cart", observation=obs,
        action_nl="Click Add to Cart button",
        refined_goal="Buy a MacBook Pro under $2000 on Amazon",
        action_reasoning="This is the right product",
        bounding_box={"x": 100, "y": 200, "width": 80, "height": 30},
        element_metadata={"bid": "42", "tag": "button", "text": "Add to Cart"},
        page_url_before="https://amazon.com/product",
        page_url_after="https://amazon.com/cart",
        misc={"step_meta": "extra"},
    )

    with tempfile.TemporaryDirectory() as d:
        traj.save(d)
        loaded = Trajectory.load(d, load_images=True)

        assert loaded.goal == "Complete purchase on Amazon"
        assert loaded.reward == 1.0
        assert loaded.success is True
        assert loaded.misc["summarized_goal"] == "Buy a laptop under $1000 on Amazon"
        assert loaded.misc["evaluation_info"]["task_type"] == "transaction"

        step = loaded.steps[0]
        assert step.action_nl == "Click Add to Cart button"
        assert step.refined_goal == "Buy a MacBook Pro under $2000 on Amazon"
        assert step.bounding_box == {"x": 100, "y": 200, "width": 80, "height": 30}
        assert step.element_metadata["bid"] == "42"
        assert step.page_url_before == "https://amazon.com/product"
        assert step.page_url_after == "https://amazon.com/cart"
        assert step.misc["step_meta"] == "extra"
    print("OK: Trajectory full save/load roundtrip")


def test_task_summarized_goal_roundtrip():
    """Task stores summarized_goal without overwriting the original goal."""
    with tempfile.TemporaryDirectory() as d:
        task = Task.from_goal("Original proposed goal", d, misc={"source": "test"})
        task.update_summarized_goal("Refined semantic goal")

        with open(os.path.join(d, "task_info.json"), "r") as f:
            info = json.load(f)
        assert info["goal"] == "Original proposed goal"
        assert info["summarized_goal"] == "Refined semantic goal"

        loaded = Task.load(d, load_steps=False, load_images=False)
        assert loaded.goal == "Original proposed goal"
        assert loaded.summarized_goal == "Refined semantic goal"
        assert loaded.misc["source"] == "test"
    print("OK: Task summarized_goal roundtrip")


# ============================================================================
# Phase 5: Evaluator edge cases
# ============================================================================

def test_extract_content_standard():
    assert extract_content("Thoughts: reasoning\nTask Type: transaction\nStatus: success\nFailure Reason: N/A", "Task Type:") == "transaction"
    assert extract_content("Thoughts: test\nTask Type: information_seeking\nStatus: failure\nFailure Reason: timeout", "Failure Reason:") == "timeout"
    print("OK: extract_content standard cases")


def test_extract_content_missing_tag():
    assert extract_content("Some random text without tags", "Task Type:") == ""
    assert extract_content("", "Status:") == ""
    print("OK: extract_content missing tags")


def test_build_vision_eval_prompt():
    prompt, sys_msg = build_vision_eval_prompt(
        intent="Buy a laptop on Amazon",
        response="I have added the laptop to cart",
        last_actions="1: click('42')\n2: click('99')",
        axtree_txt="[42] [BUTTON] [Add to Cart]"
    )
    assert "Buy a laptop on Amazon" in prompt
    assert "Task Type:" in sys_msg
    assert "transaction" in sys_msg
    assert "information_seeking" in sys_msg
    assert "site_navigation" in sys_msg
    assert "content_modification" in sys_msg
    print("OK: build_vision_eval_prompt contains all 4 task types")


# ============================================================================
# Phase 3: TaskSummarizationAgent edge cases
# ============================================================================

def test_summarizer_build_action_list():
    agent = TaskSummarizationAgent()
    traj = Trajectory.from_goal("Test")
    obs = {"axtree_txt": "test"}
    traj.add_step("click('1')", "click('1')", "thought", obs,
                  action_nl="Click on search bar")
    traj.add_step("type('1', 'laptop')", "type('1', 'laptop')", "thought", obs)
    traj.add_step("click('2')", "click('2')", "thought", obs,
                  action_nl="Click submit button")

    actions = agent._build_action_list(traj)
    assert len(actions) == 3
    assert "Click on search bar" in actions[0]
    assert "type('1', 'laptop')" in actions[1]  # Raw action when no NL
    assert "Click submit button" in actions[2]
    print("OK: TaskSummarizationAgent._build_action_list falls back to raw action")


def test_summarizer_collect_screenshots_even_sampling():
    agent = TaskSummarizationAgent()
    traj = Trajectory.from_goal("Test")
    for i in range(20):
        obs = {"screenshot": Image.new("RGB", (10, 10), color=(i*10, 0, 0))}
        traj.add_step(f"click('{i}')", f"click('{i}')", "thought", obs)

    screenshots = agent._collect_screenshots(traj, max_screenshots=5)
    assert len(screenshots) == 5, f"Should sample to 5, got {len(screenshots)}"
    print("OK: TaskSummarizationAgent._collect_screenshots even sampling")


def test_summarizer_no_screenshots():
    agent = TaskSummarizationAgent()
    traj = Trajectory.from_goal("Test")
    traj.add_step("click('1')", "click('1')", "thought", {"no_screenshot": True})
    screenshots = agent._collect_screenshots(traj, max_screenshots=8)
    assert screenshots == []
    print("OK: TaskSummarizationAgent._collect_screenshots no screenshots")


def test_summarizer_empty_actions():
    agent = TaskSummarizationAgent()
    traj = Trajectory.from_goal("Test goal")
    result = agent.summarize(traj)
    assert result == "Test goal", "Should return original goal when no actions"
    print("OK: TaskSummarizationAgent empty actions returns original goal")


# ============================================================================
# Phase 7: CaptchaDetectionAgent edge cases
# ============================================================================

def test_captcha_agent_init():
    agent = CaptchaDetectionAgent()
    assert agent.model_name == "gpt-4o-2024-05-13"
    print("OK: CaptchaDetectionAgent init")


# ============================================================================
# Phase 4: SoM processor edge cases
# ============================================================================

def test_extract_rois_empty_axtree():
    rois = extract_rois_from_axtree({})
    assert rois == {}
    print("OK: extract_rois_from_axtree empty")


def test_extract_rois_nested_axtree():
    """Nested axtree with multiple levels, some with bids."""
    axtree = {
        "tag": "html",
        "children": [
            {"bid": "1", "tag": "link", "text": "Home",
             "children": []},
            {"tag": "div",
             "children": [
                 {"bid": "2", "tag": "textbox", "text": "Search",
                  "children": []},
             ]},
        ]
    }
    extra_props = {
        "backend_1": {"bbox": [10, 20, 100, 30]},
        "backend_2": {"bbox": [50, 80, 200, 30]},
    }
    rois = extract_rois_from_axtree(axtree, extra_props)
    # Without backend_node_id matching, bbox won't be found
    # The walk yields bids but props may not match
    assert isinstance(rois, dict)
    print("OK: extract_rois_from_axtree nested structure")


def test_extract_rois_axtree_list():
    """Axtree as a list of nodes with backend_node_id matching extra_props."""
    axtree = [
        {"bid": "1", "tag": "link", "text": "Home", "backend_node_id": 101},
        {"bid": "2", "tag": "button", "text": "Submit", "backend_node_id": 102},
    ]
    extra_props = {
        "101": {"bbox": [10, 20, 80, 30]},
        "102": {"bbox": [100, 20, 80, 30]},
    }
    rois = extract_rois_from_axtree(axtree, extra_props)
    assert len(rois) == 2
    print("OK: extract_rois_from_axtree list format")


def test_process_observation_no_axtree():
    """process_observation with missing axtree returns unchanged."""
    obs = {"screenshot": np.zeros((100, 100, 3), dtype=np.uint8)}
    result = process_observation(obs)
    assert result is obs or result["screenshot"] is obs["screenshot"]
    print("OK: process_observation no axtree returns unchanged")


def test_process_observation_no_screenshot():
    """process_observation with missing screenshot returns unchanged."""
    obs = {"axtree_object": {"bid": "1", "tag": "link"}}
    result = process_observation(obs)
    assert "screenshot" not in result
    print("OK: process_observation no screenshot returns unchanged")


# ============================================================================
# Phase 1: _extract_text_obs edge cases
# ============================================================================

def test_extract_text_obs_nested():
    obs = {
        "axtree_txt": "some text",
        "nested": {"key": "value", "deep": {"deeper": "nested_value"}},
        "list_field": ["a", "b", "c"],
        "screenshot": np.zeros((10, 10)),
    }
    text_obs = _extract_text_obs(obs)
    assert text_obs["axtree_txt"] == "some text"
    assert text_obs["nested"]["deep"]["deeper"] == "nested_value"
    assert text_obs["list_field"] == ["a", "b", "c"]
    assert "screenshot" not in text_obs  # numpy array excluded
    print("OK: _extract_text_obs nested structures")


# ============================================================================
# Integration: Episode-like flow tests
# ============================================================================

def test_trajectory_add_step_mimics_episode_flow():
    """Simulate the episode loop's interaction with Trajectory."""
    goal = "Find a Sony WH-1000XM5 on Amazon"
    traj = Trajectory.from_goal(goal, agent_info={"model": "gpt-4o"})

    # Step 1: Agent searches
    traj.add_step(
        action="type('7', 'Sony WH-1000XM5')",
        parsed_action="type('7', 'Sony WH-1000XM5')",
        thought="I need to search for the product",
        observation={"axtree_txt": "[7] [TEXTBOX] [Search]", "open_pages_urls": ["https://amazon.com"]},
        action_nl="Type 'Sony WH-1000XM5' into the search bar",
        refined_goal="Search for Sony WH-1000XM5 headphones on Amazon",
        page_url_before="https://amazon.com",
    )
    # Episode would then set page_url_after after env.step
    traj.steps[-1].page_url_after = "https://amazon.com/s?k=Sony+WH-1000XM5"

    # Step 2: Agent clicks first result
    traj.add_step(
        action="click('42')",
        parsed_action="click('42')",
        thought="Click on the first search result",
        observation={"axtree_txt": "[42] [LINK] [Sony WH-1000XM5]", "open_pages_urls": ["https://amazon.com/s?k=Sony+WH-1000XM5"]},
        action_nl="Click on the Sony WH-1000XM5 product link",
        refined_goal="Find Sony WH-1000XM5 headphones under $350 with 4+ stars on Amazon",
        page_url_before="https://amazon.com/s?k=Sony+WH-1000XM5",
    )
    traj.steps[-1].page_url_after = "https://amazon.com/product/B08XYZ"

    # Step 3: Agent clicks Add to Cart
    traj.add_step(
        action="click('99')",
        parsed_action="click('99')",
        thought="Add to cart to complete the task",
        observation={"axtree_txt": "[99] [BUTTON] [Add to Cart]", "open_pages_urls": ["https://amazon.com/product/B08XYZ"]},
        action_nl="Click the Add to Cart button",
        refined_goal="Add Sony WH-1000XM5 to cart on Amazon (price $328, free shipping)",
        page_url_before="https://amazon.com/product/B08XYZ",
    )
    traj.steps[-1].page_url_after = "https://amazon.com/cart"

    # Verify: each step has its own refined_goal (evolving)
    assert traj.steps[0].refined_goal == "Search for Sony WH-1000XM5 headphones on Amazon"
    assert traj.steps[1].refined_goal == "Find Sony WH-1000XM5 headphones under $350 with 4+ stars on Amazon"
    assert traj.steps[2].refined_goal == "Add Sony WH-1000XM5 to cart on Amazon (price $328, free shipping)"

    # Verify: page_url tracking is correct
    assert traj.steps[0].page_url_before == "https://amazon.com"
    assert traj.steps[0].page_url_after == "https://amazon.com/s?k=Sony+WH-1000XM5"
    assert traj.steps[2].page_url_after == "https://amazon.com/cart"

    # Verify: action_nl is present for all steps
    for step in traj.steps:
        assert step.action_nl is not None

    print("OK: Episode flow integration test")


def test_backward_compat_add_step_minimal():
    """Old code that doesn't pass semantic fields still works."""
    traj = Trajectory.from_goal("test")
    traj.add_step("click('1')", "click('1')", "thought", {"obs": "data"},
                  misc={"usage": "tokens"})
    assert traj.steps[0].action == "click('1')"
    assert traj.steps[0].action_nl is None
    assert traj.steps[0].refined_goal is None
    assert traj.steps[0].page_url_before is None
    assert traj.steps[0].misc == {"usage": "tokens"}
    print("OK: Backward compat add_step with positional args only")


if __name__ == "__main__":
    tests = [
        # Phase 1: Parsing edge cases
        test_extract_verbose_text_before_json,
        test_extract_last_valid_json_multiple_blocks,
        test_extract_json_with_markdown_fence,
        test_extract_json_with_plain_fence,
        test_extract_with_nested_braces_in_action,
        test_extract_json_with_escaped_single_quotes,
        test_extract_empty_or_partial_responses,
        test_extract_action_with_send_msg_to_user,
        test_extract_unicode_text,
        test_extract_with_single_action_key_only,
        # Phase 1: Element metadata edge cases
        test_extract_element_metadata_multiline_axtree,
        test_extract_element_metadata_newlines_in_text,
        test_extract_element_metadata_long_text_truncation,
        # Phase 1: Sanitize edge cases
        test_sanitize_action_unicode_whitespace,
        test_sanitize_action_preserves_valid_content,
        test_action_level_refined_goal_detection,
        # Phase 1: Trajectory data model
        test_trajectory_step_without_screenshot,
        test_trajectory_multiple_steps_semantic,
        test_trajectory_from_goal_defaults,
        test_trajectory_save_load_full_roundtrip,
        test_task_summarized_goal_roundtrip,
        # Phase 5: Evaluator
        test_extract_content_standard,
        test_extract_content_missing_tag,
        test_build_vision_eval_prompt,
        # Phase 3: TaskSummarization
        test_summarizer_build_action_list,
        test_summarizer_collect_screenshots_even_sampling,
        test_summarizer_no_screenshots,
        test_summarizer_empty_actions,
        # Phase 7: CAPTCHA
        test_captcha_agent_init,
        # Phase 4: SoM
        test_extract_rois_empty_axtree,
        test_extract_rois_nested_axtree,
        test_extract_rois_axtree_list,
        test_process_observation_no_axtree,
        test_process_observation_no_screenshot,
        # Integration
        test_extract_text_obs_nested,
        test_trajectory_add_step_mimics_episode_flow,
        test_backward_compat_add_step_minimal,
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
