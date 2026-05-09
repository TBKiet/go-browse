from . import BasePromptBuilder, flatten_messages
from ..trajectory_data import BrowserGymAgentStepData, BrowserGymAgentTrajectoryData
from browsergym.core.action.base import AbstractActionSet
from dataclasses import dataclass
from textwrap import dedent
import json

class SolverPromptBuilder(BasePromptBuilder):

    def __init__(self, action_set: AbstractActionSet):
        self.action_set = action_set
        self.action_set_description = action_set.describe(with_long_description=True, with_examples=True)


    def build_messages(self, obs: dict):
        messages = []
        if "message" in obs:
            messages.append({"text": obs["message"]})
        return messages

    def format_thought_and_action(self, thought: str, action: str) -> str:
        d = {}
        if thought:
            d['thought'] = thought
        if action:
            d['action'] = action
        return json.dumps(d)

    def trim_axtree(self, axtree: str, num_chars_overflow: int) -> str:
        trim_str = "...trimmed due to context size limit"
        return axtree[:-num_chars_overflow - len(trim_str)] + trim_str


    def build_trajectory_messages(self, trajectory_data: BrowserGymAgentTrajectoryData, char_limit: int=-1) -> list[dict]:
        messages = []
        for i, step in enumerate(trajectory_data.steps):
            messages.append(self.build_messages(trajectory_data.goal, step, trajectory_data.steps[:i], char_limit))
        return messages


    def build_messages(self, goal: str, current_step: BrowserGymAgentStepData, history: list[BrowserGymAgentStepData], char_limit: int=-1) -> dict:
        past_thoughts = [step.thought for step in history]
        past_actions = [step.misc['parsed_action'] if 'parsed_action' in step.misc else step.action for step in history]
        past_errors = [step.last_action_error for step in history]

        axtree = current_step.axtree
        last_action_error = current_step.last_action_error
        completion_thought = current_step.thought
        completion_action = current_step.misc['parsed_action'] if current_step.misc and 'parsed_action' in current_step.misc else current_step.action

        add_completion = completion_thought or completion_action

        messages = self._build_messages(
            goal,
            past_thoughts,
            past_actions,
            past_errors,
            axtree,
            last_action_error,
            completion_thought,
            completion_action
        )
        curr_char_count = self.count_message_chars(messages['prompt'] + (messages['completion'] if add_completion else []))
        if char_limit > 0 and curr_char_count > char_limit:
            past_thoughts, past_actions, past_errors = self.trim_past_thoughts_and_actions(past_thoughts, past_actions, past_errors, max_allowed=8)
            messages = self._build_messages(
                goal,
                past_thoughts,
                past_actions,
                past_errors,
                axtree,
                last_action_error,
                completion_thought,
                completion_action
            )

            curr_char_count = self.count_message_chars(messages['prompt'] + (messages['completion'] if add_completion else []))
            remaining_overflow = curr_char_count - char_limit
            if remaining_overflow > 0:
                axtree = self.trim_axtree(axtree, remaining_overflow)
                messages = self._build_messages(
                    goal,
                    past_thoughts,
                    past_actions,
                    past_errors,
                    axtree,
                    last_action_error,
                    completion_thought,
                    completion_action
                )

        return {k : flatten_messages(v) for k, v in messages.items() if v}


    def count_message_chars(self, messages: list[dict]) -> int:
        return sum([len(m['text']) for message in messages for m in message['content']])

    def trim_past_thoughts_and_actions(self, past_thoughts: list[str | None], past_actions: list[str], past_errors: list[str | None] = None, max_allowed: int=3) -> tuple[list[str | None], list[str], list[str | None]]:
        if past_errors is None:
            past_errors = [None] * len(past_thoughts)
        if len(past_thoughts) > max_allowed:
            past_thoughts = past_thoughts[-max_allowed:]
            past_actions = past_actions[-max_allowed:]
            past_errors = past_errors[-max_allowed:]
        return past_thoughts, past_actions, past_errors


    def _build_messages(
        self,
        goal: str,
        thoughts: list[str | None],
        actions: list[str | None],
        past_errors: list[str | None] = None,
        axtree: str = "",
        last_action_error: str | None = None,
        completion_thought: str | None = None,
        completion_action: str | None = None
    ):
        if past_errors is None:
            past_errors = [None] * len(thoughts)

        system_messages = {"role": "system", "content": [self.system_message()]}

        # Build action history with error annotations
        action_history = self.action_history_with_errors(thoughts, actions, past_errors)

        user_messages = {
            "role": "user",
            "content": [
                self.goal_message(goal),
                self.axtree_message(axtree),
                self.action_space_message(self.action_set),
                action_history,
            ]
        }

        # Detect repeated failures and inject escalated warnings
        last_action = actions[-1] if actions else None
        consecutive_same_failures = self._count_consecutive_same_failures(actions, past_errors)
        if consecutive_same_failures >= 2:
            user_messages["content"].append(self.strategy_change_message(consecutive_same_failures, last_action, last_action_error))
        else:
            # Detect silent stuck: same action repeated without errors (e.g., infinite scrolling)
            silent_same_count = self._count_consecutive_same_actions(actions)
            if silent_same_count >= 4:
                user_messages["content"].append(self.silent_stuck_message(silent_same_count, last_action))

        if last_action_error:
            user_messages["content"].append(self.last_action_error_message(last_action_error))

        user_messages["content"].append(self.next_action_request_message(has_errors=bool(last_action_error)))

        output = { "prompt": [system_messages, user_messages] }

        if completion_thought or completion_action:
            assistant_messages = {
                "role": "assistant",
                "content": [self.completion_message(completion_thought, completion_action)]
            }
            output["completion"] = [assistant_messages]

        return output


    def action_history_with_errors(self, thoughts: list[str | None], actions: list[str], errors: list[str | None]):
        """Build action history with error annotations for past failures."""
        if not actions:
            return {"type": "text", "text": "# History of past actions\n(no actions yet)"}
        newline = "\n"
        lines = []
        for i, (thought, action, error) in enumerate(zip(thoughts, actions, errors)):
            entry = self.format_thought_and_action(thought, action)
            if error:
                # Truncate long error messages
                error_short = error[:200] + "..." if len(error) > 200 else error
                lines.append(f"{entry}  <-- FAILED: {error_short}")
            else:
                lines.append(entry)
        return {
            "type": "text",
            "text": "# History of past actions\n" + newline.join(lines)
        }

    def _count_consecutive_same_failures(self, actions: list[str | None], errors: list[str | None]) -> int:
        """Count how many times the most recent action has failed consecutively with any error."""
        if not actions or not errors:
            return 0
        last_action = actions[-1]
        if last_action is None:
            return 0
        count = 0
        for a, e in zip(reversed(actions), reversed(errors)):
            if a == last_action and e:
                count += 1
            else:
                break
        return count

    def _count_consecutive_same_actions(self, actions: list[str | None]) -> int:
        """Count how many times the most recent action appears consecutively (regardless of errors)."""
        if not actions:
            return 0
        last_action = actions[-1]
        if last_action is None:
            return 0
        count = 0
        for a in reversed(actions):
            if a == last_action:
                count += 1
            else:
                break
        return count

    def silent_stuck_message(self, repeat_count: int, action: str):
        """Warning when the same action is repeated without errors (silent loop)."""
        action_short = action[:100] + "..." if len(action) > 100 else action
        return {
            "type": "text",
            "text": (
                f"# WARNING: YOU APPEAR TO BE STUCK\n"
                f"You have repeated `{action_short}` {repeat_count} times in a row "
                f"without making progress. The page state may not be changing.\n\n"
                f"You MUST change your approach NOW:\n"
                f"- If scrolling is revealing no new content, you may have reached the end of the page\n"
                f"- Try a different action: click a specific element, use goto() to navigate elsewhere\n"
                f"- If you have already found the answer, use send_msg_to_user() to finish\n"
                f"- If the task is impossible, use report_infeasible('reason')"
            )
        }

    def strategy_change_message(self, failure_count: int, failed_action: str, last_error: str | None):
        """Escalated warning when the same action has failed repeatedly."""
        error_detail = ""
        error_category = None
        if last_error:
            error_short = last_error[:200] + "..." if len(last_error) > 200 else last_error
            error_detail = f" The error was: {error_short}"
            error_category = self._classify_error(last_error)
        action_short = failed_action[:100] + "..." if len(failed_action) > 100 else failed_action

        # Targeted advice based on error category
        targeted_advice = self._error_recovery_advice(error_category, failed_action)

        return {
            "type": "text",
            "text": (
                f"# WARNING: REPEATED ACTION FAILURE\n"
                f"Your action `{action_short}` has FAILED {failure_count} times in a row.{error_detail}\n\n"
                f"You MUST try a DIFFERENT strategy or action. Do NOT repeat the same action.\n"
                f"{targeted_advice}"
            )
        }

    def _classify_error(self, error_msg: str) -> str:
        """Classify error message into a category for targeted recovery advice."""
        error_lower = error_msg.lower()
        if "multi-action" in error_lower or "only single-actions" in error_lower:
            return "multi_action"
        if "empty action" in error_lower:
            return "empty_action"
        if "timeout" in error_lower or "timeout" in error_lower:
            return "timeout"
        if "locator" in error_lower or "element" in error_lower or "selector" in error_lower:
            return "element_not_found"
        if "navigation" in error_lower or "page" in error_lower or "content" in error_lower:
            return "page_error"
        if "rate limit" in error_lower or "api" in error_lower:
            return "api_error"
        return "unknown"

    def _error_recovery_advice(self, error_category: str | None, failed_action: str) -> str:
        """Return targeted recovery advice based on the error category and action type."""
        if error_category == "multi_action":
            return (
                "The parser detected multiple actions in your response. This usually means "
                "the action parser scanned your 'thought' text. Make sure your action string "
                "is a SINGLE function call like click('1234') with no extra text or characters.\n"
                "Valid alternatives: try a different action type, or simplify your response."
            )
        elif error_category == "empty_action":
            return (
                "The environment received an empty or unparseable action. Check that your "
                "action string is formatted correctly: function_name('arguments').\n"
                "Do NOT use send_msg_to_user if the environment is rejecting it — instead "
                "navigate to a results page or use a different mechanism to complete the task."
            )
        elif error_category == "timeout":
            return (
                "The action timed out — the page may be slow to load or the element may not exist. "
                "Wait briefly with noop(1000) then try scrolling or looking for alternative elements. "
                "If the page is still loading, wait longer before acting."
            )
        elif error_category == "element_not_found":
            return (
                "The target element was not found on the page. It may have a different bid/identifier, "
                "be hidden, or not exist yet. Scroll to reveal more of the page, or use a different "
                "element with similar function. Check the accessibility tree for alternative bids."
            )
        elif error_category == "page_error":
            return (
                "A page/navigation error occurred. The page may have reloaded or the content may have "
                "changed. Re-read the accessibility tree carefully — element bids may have changed. "
                "If this persists, try navigating back or to a different page."
            )
        elif error_category == "api_error":
            return (
                "An API error occurred (rate limit or server error). Wait briefly (noop) then retry "
                "with a different approach. Reduce unnecessary actions."
            )
        else:
            return (
                "Valid alternatives:\n"
                "- Scroll up/down to reveal more of the page\n"
                "- Try clicking a different element with a similar label/function\n"
                "- Use keyboard navigation (press('Enter'), press('Tab'), etc.)\n"
                "- Navigate to a different page or go back (goto(...))\n"
                "- If the element is not interactable, inspect nearby elements instead"
            )

    def system_message(self):
        return  {
                "type": "text",
                "text": dedent("""\
                    # Instructions
                    You are a UI Assistant, your goal is to help the user perform tasks using a web browser.
                    Review the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

                    FORMAT: Your entire response must be a single JSON object: {"thought": "...", "action": "..."}
                    - Do NOT output any text, explanation, or markdown outside the JSON.
                    - All reasoning goes inside the "thought" key.
                    - Only ONE action in the "action" key.
                    """
                )
        }

    def goal_message(self, goal: str):
        return  {
                "type": "text",
                "text": (
                    "# Goal\n"
                    f"{goal}"
                )
        }


    def action_space_message(self, action_set: AbstractActionSet):
        newline = "\n"
        return  {
                "type": "text",
                "text": ("# Action Space"
                    f"{self.action_set_description}\n\n"
                    "Here are examples of actions with chain-of-thought reasoning:\n\n"
                    f"{newline.join(newline + json.dumps(cot_example) for cot_example in self.cot_examples())}\n\n\n"
                )
        }

    def cot_examples(self) -> list[dict]:
        return [
            {"thought": "I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.", "action": "click('12')"},
            {"thought": "I found the information requested by the user, I will send it to the chat.", "action": "send_msg_to_user('The price for a 15 inch laptop is 1499 USD.')"},
            {"thought": "I have finished navigating to the Products page. I will inform the user that I have completed the task.", "action": "send_msg_to_user('I have finished navigating to the Products page.')"},
        ]


    def axtree_message(self, axtree: str):
        return  {
                "type": "text",
                "text": (
                    "# Current page Accessibility Tree\n"
                    f"{axtree}"
                )
        }

    def last_action_error_message(self, last_action_error: str):
        return  {
                "type": "text",
                "text": (
                    "# Error message from last action\n"
                    f"{last_action_error}\n\n"
                    "Your action was rejected by the browser environment. "
                    "Read the error carefully — it explains what went wrong. "
                    "Then choose a DIFFERENT action that addresses the error."
                )
        }

    def action_history_messages(self, thoughts: list[str | None], actions: list[str]):
        newline = "\n"
        return  {
                "type": "text",
                "text": (
                    "# History of past actions\n"
                    f"{newline.join(self.format_thought_and_action(thought, action) for thought, action in zip(thoughts, actions))}"
                )
        }


    def next_action_request_message(self, has_errors: bool = False):
        base_text = (
            "# Next action\n\n"
            "You will now think step by step and produce your next best action. "
            "Reflect on your past actions, any resulting error message, and the current "
            "state of the page before deciding on your next action.\n\n"
        )
        if has_errors:
            base_text += (
                "Your last action FAILED. In your thought, analyze WHY it failed and "
                "choose a DIFFERENT approach. Do NOT repeat the failed action.\n\n"
            )
        base_text += (
            "CRITICAL: Your ENTIRE response must be ONLY a single JSON object — no other "
            "text before or after. Do NOT include any reasoning, explanation, markdown "
            "code fences, or thinking tags outside the JSON. All your reasoning must be "
            "contained within the \"thought\" key of the JSON.\n\n"
            "Format: {\"thought\": \"<your reasoning>\", \"action\": \"<single action>\"}\n\n"
            "IMPORTANT: Provide exactly ONE action in the \"action\" key. "
            "If you have finished the task, use send_msg_to_user(\"<answer>\")."
        )
        return {"type": "text", "text": base_text}


    def completion_message(self, completion_thought: str, completion_action: str):
        return  {
                "type": "text",
                "text": f"{self.format_thought_and_action(completion_thought, completion_action)}"
        }


