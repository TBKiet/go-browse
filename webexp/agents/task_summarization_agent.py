"""Task Summarization Agent.

After a trajectory is collected, this agent produces a clean, coherent task
description from the full action history and screenshots. This is a key part
of making trajectories "semantic" — the summarized goal captures what was
actually accomplished, not just what was initially proposed.
"""

from ..explore.core.trajectory import Trajectory
from dataclasses import dataclass
from PIL import Image
from openai import OpenAI
import base64
import io
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


def _pil_to_b64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 data URL string."""
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()


@dataclass
class TaskSummary:
    """Structured post-hoc description of what a trajectory accomplished."""

    accomplished_goal: str
    completion_evidence: str | None = None
    summary_differs_from_original: bool = False

    def to_dict(self) -> dict:
        return {
            "accomplished_goal": self.accomplished_goal,
            "completion_evidence": self.completion_evidence,
            "summary_differs_from_original": self.summary_differs_from_original,
        }


class TaskSummarizationAgent:
    """Produces a summarized task description from a completed trajectory.

    Uses GPT-4V to look at the action history (natural language descriptions)
    and screenshots, then outputs a concise task description in the standard
    format: "<action verb> <specific details> on <website>".
    """

    SYSTEM_PROMPT = """Given a list of actions performed on a website and the corresponding screenshots, your task is to infer the user task that was actually accomplished by performing these actions in the given sequence on the website.

*IMPORTANT*
0. Infer the accomplished task from the action history and screenshots first. The original task proposal is only a weak hint for resolving ambiguity.
1. Do not copy the original task proposal unless the action history and screenshots support exactly that task.
2. The task must contain an action verb: "Buy, Book, Find, Check, Choose, show me, search, browse, get, compare, view, give me, add to cart, ...", ideally involving transactions or finding information on a specific product or service.
3. You should propose a task that is clear and specific.
4. The task description should provide all the necessary information to complete the task.
5. The task description must indicate the domain of the website at the end with format: "... on <website_name>", for instance, "Purchase a laptop on Amazon", "Book a hair appointment on Yelp", etc.
6. The task should be feasible to complete by a real user and should not require any additional information not specified.
7. The task description should specify constraints like given budget, product features, and other specifications that can narrow down the search.
8. The task description should describe what was accomplished, not how it was accomplished. Do not mention procedural details like clicking, waiting, scrolling, or opening a menu unless that is the user's actual task.
9. Do NOT use any quotation marks (either single or double) in the task description.

*OUTPUT FORMAT*: Return only a JSON object within ``` ``` with these keys:
{
  "accomplished_goal": "<TASK_DESCRIPTION>",
  "completion_evidence": "<short evidence from the action history or final state>",
  "summary_differs_from_original": <true or false>
}
"""

    def __init__(self, model_name: str = "gpt-4o-2024-05-13"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", None),
        )

    def summarize(self, trajectory: Trajectory, max_screenshots: int = 8) -> str | None:
        """Summarize the task from a completed trajectory.

        Args:
            trajectory: The completed trajectory to summarize.
            max_screenshots: Max number of screenshots to include (to control cost).

        Returns:
            Summarized task description string, or None if summarization fails.
        """
        summary = self.summarize_details(trajectory, max_screenshots=max_screenshots)
        return summary.accomplished_goal if summary else None

    def summarize_details(self, trajectory: Trajectory, max_screenshots: int = 8) -> TaskSummary | None:
        """Return a structured summary for a completed trajectory."""
        try:
            # Build action list from trajectory steps, preferring NL descriptions
            action_list = self._build_action_list(trajectory)

            # Collect screenshots
            screenshots = self._collect_screenshots(trajectory, max_screenshots)

            if not action_list:
                logger.warning("No actions to summarize, using original goal")
                return TaskSummary(
                    accomplished_goal=trajectory.goal,
                    completion_evidence="No actions were available; fell back to the original task proposal.",
                    summary_differs_from_original=False,
                )

            messages = self._build_messages(action_list, screenshots, trajectory.goal)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.01,
            )

            raw_response = response.choices[0].message.content
            logger.info(f"Summarization response: {raw_response[:500]}")

            parsed = self._parse_summary_response(raw_response, trajectory.goal)
            if parsed:
                logger.info(f"Accomplished goal: {parsed.accomplished_goal}")
                return parsed

            # Fallback: take the last non-empty line
            lines = [l.strip() for l in raw_response.split("\n") if l.strip()]
            if lines:
                fallback = lines[-1]
                logger.info(f"Fallback summary: {fallback}")
                return TaskSummary(
                    accomplished_goal=fallback,
                    completion_evidence="Fallback summary parsed from the model response.",
                    summary_differs_from_original=fallback != trajectory.goal,
                )

            return TaskSummary(
                accomplished_goal=trajectory.goal,
                completion_evidence="Empty summarizer response; fell back to the original task proposal.",
                summary_differs_from_original=False,
            )

        except Exception as e:
            logger.error(f"Task summarization failed: {e}")
            return None

    def _parse_summary_response(self, raw_response: str, original_goal: str) -> TaskSummary | None:
        """Parse structured summary JSON, with backward-compatible fenced text fallback."""
        if not raw_response:
            return None

        match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
        candidate = match.group(1).strip() if match else raw_response.strip()

        try:
            parsed = json.loads(candidate)
            accomplished_goal = str(parsed.get("accomplished_goal") or parsed.get("summarized_goal") or "").strip()
            if not accomplished_goal:
                return None
            completion_evidence = parsed.get("completion_evidence")
            differs = parsed.get("summary_differs_from_original")
            if differs is None:
                differs = accomplished_goal != original_goal
            return TaskSummary(
                accomplished_goal=accomplished_goal,
                completion_evidence=str(completion_evidence).strip() if completion_evidence else None,
                summary_differs_from_original=bool(differs),
            )
        except json.JSONDecodeError:
            pass

        if match:
            text_summary = candidate.strip()
            if text_summary:
                return TaskSummary(
                    accomplished_goal=text_summary,
                    completion_evidence="Legacy fenced text summary parsed from the model response.",
                    summary_differs_from_original=text_summary != original_goal,
                )
        return None

    def _build_action_list(self, trajectory: Trajectory) -> list[str]:
        """Build a list of human-readable action descriptions."""
        actions = []
        for i, step in enumerate(trajectory.steps):
            if step.action_nl:
                actions.append(f"Step {i+1}: {step.action_nl}")
            elif step.action:
                actions.append(f"Step {i+1}: {step.action}")
        return actions

    def _collect_screenshots(self, trajectory: Trajectory, max_screenshots: int) -> list[Image.Image]:
        """Collect screenshots from trajectory steps, evenly sampled."""
        screenshots = []
        for step in trajectory.steps:
            if "screenshot" in step.observation:
                img = step.observation["screenshot"]
                if isinstance(img, Image.Image):
                    screenshots.append(img)
                else:
                    # numpy array
                    screenshots.append(Image.fromarray(img))

        if not screenshots:
            return []

        # Evenly sample if too many
        if len(screenshots) > max_screenshots:
            step = len(screenshots) / max_screenshots
            sampled = []
            for i in range(max_screenshots):
                idx = min(int(i * step), len(screenshots) - 1)
                sampled.append(screenshots[idx])
            return sampled

        return screenshots

    def _build_messages(self, action_list: list[str], screenshots: list[Image.Image], original_goal: str) -> list[dict]:
        """Build GPT-4V messages for task summarization."""
        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
        }

        user_content = [
            {
                "type": "text",
                "text": (
                    f"ACTIONS PERFORMED:\n" + "\n".join(action_list) +
                    f"\n\nOriginal task proposal: {original_goal}\n\n"
                    "The screenshots below show the web pages at key steps of the trajectory."
                ),
            }
        ]

        for img in screenshots:
            b64 = _pil_to_b64(img)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        return [system_msg, {"role": "user", "content": user_content}]
