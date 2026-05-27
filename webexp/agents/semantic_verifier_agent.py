"""Semantic trajectory verification.

This verifier checks whether the semantic labels attached to a trajectory are
supported by the action history and final page state. It is intentionally kept
separate from the reward evaluator: the reward evaluator decides task success,
while this verifier decides whether the trajectory is good semantic training
data.
"""

from __future__ import annotations

from ..explore.core.trajectory import Trajectory
from .task_summarization_agent import _pil_to_b64
from PIL import Image
from openai import OpenAI
import json
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticVerifierAgent:
    """LLM-based critic for semantic trajectory metadata."""

    SYSTEM_PROMPT = """You are a strict data-quality verifier for web agent trajectories.

Given an original task, an accomplished task summary, the action history, the final page state, and screenshots, decide whether the semantic labels are suitable for training data.

Check three things:
1. is_aligned: the accomplished task and trajectory remain aligned with the original task intent.
2. is_grounded: natural-language actions are consistent with the raw grounded actions and visible target elements.
3. is_complete: the final page state and action history support that the accomplished task was completed.

Return only a JSON object with:
{
  "is_aligned": true/false,
  "is_grounded": true/false,
  "is_complete": true/false,
  "failure_reason": null or "<short reason>",
  "evidence": "<short evidence from action history or final state>"
}
"""

    def __init__(self, model_name: str = "gpt-4o-2024-05-13"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", None),
        )

    def verify(
        self,
        trajectory: Trajectory,
        accomplished_goal: str | None = None,
        max_screenshots: int = 6,
    ) -> dict | None:
        """Verify semantic alignment, grounding, and completion."""
        try:
            if not trajectory.steps:
                return {
                    "is_aligned": False,
                    "is_grounded": False,
                    "is_complete": False,
                    "failure_reason": "Trajectory has no steps.",
                    "evidence": None,
                }

            messages = self._build_messages(trajectory, accomplished_goal, max_screenshots)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=384,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            result = self._parse_response(raw)
            if result is not None:
                result["model_usage"] = response.usage.to_dict()
            return result
        except Exception as e:
            logger.error(f"Semantic verification failed: {e}")
            return None

    def _build_messages(
        self,
        trajectory: Trajectory,
        accomplished_goal: str | None,
        max_screenshots: int,
    ) -> list[dict]:
        action_lines = []
        task_state_lines = []
        for idx, step in enumerate(trajectory.steps):
            action_nl = step.action_nl or (step.misc or {}).get("action_nl")
            raw_action = step.parsed_action or step.action
            grounding_valid = (step.misc or {}).get("grounding_valid")
            element_metadata = step.element_metadata or (step.misc or {}).get("element_metadata")
            action_lines.append(
                f"{idx + 1}. action_nl={action_nl!r}; raw_action={raw_action!r}; "
                f"grounding_valid={grounding_valid!r}; element={element_metadata!r}"
            )
            task_state = (step.misc or {}).get("task_state")
            if task_state:
                task_state_lines.append(f"{idx + 1}. {task_state}")

        final_step = trajectory.steps[-1]
        final_obs = final_step.observation or {}
        final_text = final_obs.get("axtree_txt") or final_obs.get("axtree_visible_only_txt") or ""
        final_url = final_step.page_url_after or final_step.page_url_before
        if not final_url:
            urls = final_obs.get("open_pages_urls") or []
            final_url = urls[0] if urls else None

        user_text = (
            f"Original task:\n{trajectory.goal}\n\n"
            f"Accomplished task summary:\n{accomplished_goal or (trajectory.misc or {}).get('summarized_goal') or trajectory.goal}\n\n"
            f"Agent response:\n{trajectory.response}\n\n"
            f"Final URL:\n{final_url}\n\n"
            "Action history:\n"
            + "\n".join(action_lines)
            + "\n\nTask state history:\n"
            + ("\n".join(task_state_lines) if task_state_lines else "None")
            + "\n\nFinal page accessibility text:\n"
            + final_text[:12000]
        )

        content = [{"type": "text", "text": user_text}]
        for img in self._collect_screenshots(trajectory, max_screenshots):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(img)}"},
            })

        return [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

    def _collect_screenshots(self, trajectory: Trajectory, max_screenshots: int) -> list[Image.Image]:
        screenshots = []
        for step in trajectory.steps:
            img = (step.observation or {}).get("screenshot")
            if img is None:
                continue
            screenshots.append(img if isinstance(img, Image.Image) else Image.fromarray(img))
        if len(screenshots) <= max_screenshots:
            return screenshots
        stride = len(screenshots) / max_screenshots
        return [screenshots[min(int(i * stride), len(screenshots) - 1)] for i in range(max_screenshots)]

    def _parse_response(self, raw: str) -> dict | None:
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = match.group(0) if match else raw
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return {
            "is_aligned": bool(parsed.get("is_aligned")),
            "is_grounded": bool(parsed.get("is_grounded")),
            "is_complete": bool(parsed.get("is_complete")),
            "failure_reason": parsed.get("failure_reason"),
            "evidence": parsed.get("evidence"),
        }
