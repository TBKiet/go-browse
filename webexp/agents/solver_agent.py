from .base_agent import AgentFactory, BaseAgent
from .prompt_builders.solver_prompt_builder import SolverPromptBuilder
from .trajectory_data import BrowserGymAgentStepData
from ..explore.grounding.som_processor import process_observation
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from openai import OpenAI
from tenacity import retry, before_sleep_log, stop_after_attempt, wait_exponential, wait_random
import ast
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# All valid action prefixes, dynamically built from the action set.
# We use a regex approach: any token matching `function_name(` is valid.
# This avoids hardcoding and having to update when action sets change.
import re as _re

def _is_action_valid(action: str) -> bool:
    """Check if action starts with a valid function call pattern (e.g. `click(...)`, `go_back()`)."""
    return bool(_re.match(r'[a-zA-Z_][a-zA-Z0-9_]*\(', action))

VALID_ACTION_PREFIXES = None  # kept for backwards compatibility, use _is_action_valid instead

def messages_to_string(messages: list[dict]) -> str:
    prompt_text_strings = []
    for message in messages:
        prompt_text_strings.append(message["content"])
    full_prompt_txt = "\n".join(prompt_text_strings)
    return full_prompt_txt


def extract_action_and_thought(raw_string):
    """Extract thought and action from raw LLM output that may contain
    verbose reasoning text before the JSON block.

    Handles:
    - Text before </think> marker (Qwen/DeepSeek thinking format)
    - Verbose reasoning text before the JSON
    - Multiple JSON blocks (takes the last valid one)
    - Malformed escape sequences

    Args:
        raw_string (str): Raw string containing thought and action

    Returns:
        tuple: (action, thought) or (None, None) if extraction fails
    """
    parsed = _extract_full_response(raw_string)
    if parsed is None:
        return None, None
    return parsed.get("action"), parsed.get("thought")


def _extract_full_response(raw_string: str) -> dict | None:
    """Extract the full JSON response dict from raw LLM output.

    Parses thought, action, action_in_natural_language, and refined_goal
    from the LLM JSON response. Handles all the same edge cases as
    extract_action_and_thought.

    Args:
        raw_string (str): Raw string from LLM

    Returns:
        dict with keys: action, thought, action_nl, refined_goal
        or None if extraction fails
    """
    if not raw_string:
        return None

    # Step 1: Strip text before </think> marker
    think_end = raw_string.rfind("</think>")
    if think_end != -1:
        raw_string = raw_string[think_end + len("</think>"):]

    # Step 2: Find and parse the LAST valid JSON object
    json_start = -1
    for pattern in [r'\{\s*"thought"', r'\{\s*"action"']:
        for m in re.finditer(pattern, raw_string):
            json_start = max(json_start, m.start())
    if json_start != -1:
        depth = 0
        json_end = -1
        for i in range(json_start, len(raw_string)):
            if raw_string[i] == '{':
                depth += 1
            elif raw_string[i] == '}':
                depth -= 1
                if depth == 0:
                    json_end = i
                    break
        if json_end != -1:
            json_str = raw_string[json_start:json_end + 1]
            try:
                parsed = json.loads(json_str)
                if parsed.get("action"):
                    return {
                        "action": parsed.get("action"),
                        "thought": parsed.get("thought"),
                        "action_nl": parsed.get("action_in_natural_language"),
                        "refined_goal": parsed.get("refined_goal"),
                    }
            except json.JSONDecodeError:
                pass

    # Step 3: Fallback to regex extraction
    try:
        result = {}
        thought_match = re.search(r'"thought"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).replace('\\"', '"')

        action_match = re.search(r'"action"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if action_match:
            result["action"] = action_match.group(1).replace('\\"', '"')

        action_nl_match = re.search(r'"action_in_natural_language"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if action_nl_match:
            result["action_nl"] = action_nl_match.group(1).replace('\\"', '"')

        refined_goal_match = re.search(r'"refined_goal"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if refined_goal_match:
            result["refined_goal"] = refined_goal_match.group(1).replace('\\"', '"')

        if result.get("action"):
            return result
    except Exception:
        pass

    # Step 4: Truncated JSON recovery — try to extract action even if JSON is cut off
    action_match = re.search(r'"action"\s*:\s*"([^"]*)', raw_string, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        if action:
            logger.warning(f"Recovered action from truncated JSON: {action}")
            return {"action": action, "thought": None, "action_nl": None, "refined_goal": None}

    return None


def _extract_element_metadata(axtree: str, action: str) -> dict | None:
    """Extract element metadata from axtree based on the bid referenced in action.

    Args:
        axtree: Accessibility tree string with format like '[42] [LINK] [Sony ...]'
        action: Action string like "click('42')" or "type('42', 'text')"

    Returns:
        dict with keys: bid, tag, text or None if not found
    """
    if not axtree or not action:
        return None
    # Extract bid from action: click('42') -> 42, type('42', ...) -> 42
    bid_match = re.search(r"\(['\"](\d+)['\"]", action)
    if not bid_match:
        return None
    bid = bid_match.group(1)

    # Find the element line in axtree: [42] [LINK] [Sony WH-1000XM5 ...]
    line_pattern = re.compile(rf'\[{bid}\]\s+\[(\w+)\]\s+\[(.*?)\]')
    m = line_pattern.search(axtree)
    if m:
        return {"bid": bid, "tag": m.group(1), "text": m.group(2)}
    # Try simpler pattern: just [42] followed by anything
    line_pattern2 = re.compile(rf'\[{bid}\]\s+\[(\w+)\]\s*(.*)')
    m2 = line_pattern2.search(axtree)
    if m2:
        text = m2.group(2).strip()
        if len(text) > 200:
            text = text[:200] + "..."
        return {"bid": bid, "tag": m2.group(1), "text": text}
    return {"bid": bid, "tag": None, "text": None}


def sanitize_action(action: str) -> str:
    """Strip hidden whitespace, zero-width characters, and normalize newlines
    from an action string. Critical for preventing parser mismatches where
    invisible characters cause multi-action detection or malformed parsing.

    Handles:
    - Zero-width space (U+200B), zero-width non-joiner (U+200C), zero-width joiner (U+200D)
    - Zero-width no-break space (U+FEFF / BOM)
    - Various Unicode space characters
    - Carriage returns
    - Leading/trailing whitespace
    """
    if not action:
        return action
    # Strip zero-width and invisible Unicode characters
    action = action.replace('​', '')  # zero-width space
    action = action.replace('‌', '')  # zero-width non-joiner
    action = action.replace('‍', '')  # zero-width joiner
    action = action.replace('﻿', '')  # zero-width no-break space / BOM
    action = action.replace('\r\n', '\n')  # normalize Windows newlines
    action = action.replace('\r', '\n')    # normalize old Mac newlines
    action = action.replace('\n', '')      # remove newlines entirely (actions are single-line)
    # Strip other common invisible characters
    action = action.replace(' ', ' ')  # non-breaking space -> normal space
    action = action.replace('\t', ' ')      # tab -> space
    return action.strip()


@AgentFactory.register
class SolverAgent(BaseAgent):
    """
    Agent used to fulfill/solve user requests.
    """

    def __init__(
            self,
            model_id: str,
            model_id_2: str | None = None,
            base_url: str | None = None,
            base_url_2: str | None = None,
            api_key: str | None = None,
            temperature: float = 1.0,
            char_limit: int = -1,
            demo_mode: str = 'off',
            use_som: bool = False,
    ):
        """
        Initialize the agent.

        Args:
            model_name (str): The name of the model to use.
            temperature (float): The temperature to use for sampling.
            demo_mode (bool): Whether to run in demo mode.
            use_som (bool): Whether to use Set-of-Mark visual grounding on screenshots.
        """

        # These are args that will be specified in the config.
        super().__init__(model_id=model_id, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode)

        self.model_id = model_id
        self.model_id_2 = model_id_2 or model_id
        self.temperature = temperature
        self.char_limit = char_limit
        self.demo_mode = demo_mode
        self.use_som = use_som


        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        base_url_2 = base_url_2 or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY", "Unspecified!")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.client_long = OpenAI(base_url=base_url_2, api_key=api_key)

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode
        )

        self.prompt_builder = SolverPromptBuilder(self.action_set)

        self.history: list[BrowserGymAgentStepData] = []
        self._recent_action_errors: list[tuple[str, str]] = []  # (action, error) for retry-loop detection
        self._max_same_action_retries = 3
        self._refined_goal: str | None = None  # evolving task description

    def reset(self):
        self.history.clear()
        self._recent_action_errors.clear()
        self._refined_goal = None

    def obs_preprocessor(self, obs: dict) -> dict:

        processed = {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"], filter_visible_only=False, extra_properties=obs["extra_element_properties"]),
            "axtree_visible_only_txt": flatten_axtree_to_str(obs["axtree_object"], filter_visible_only=True, extra_properties=obs["extra_element_properties"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
            "extra_element_properties": obs["extra_element_properties"],
            "axtree_object": obs["axtree_object"],  # Keep raw axtree for SoM processing
        }

        # Phase 4: Apply Set-of-Mark visual grounding if enabled
        if self.use_som:
            try:
                processed = process_observation(processed)
                # SoM changes the screenshot to annotated version
            except Exception as e:
                logger.warning(f"SoM processing failed: {e}")

        return processed

    def action_processor(self, action: str) -> str:
        """
        Process the action before it is passed to the environment.

        Args:
            action (str): The action to process.

        Returns:
            str: The processed action.
        """
        parsed_action, thought = extract_action_and_thought(action)
        return self.action_set.to_python_code(parsed_action if parsed_action else action)


    def get_action(self, obs: dict, oracle_action:tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        """
        Get the action for the given observation.

        Args:
            obs (dict): The observation from the environment.
            oracle_action tuple[str, str]: Tuple of (action, thought) to use if available instead of generating a new one.

        Returns:
            str: The action to take.
        """

        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs["axtree_visible_only_txt"],  # visible-only saves ~33% tokens vs full tree
            last_action_error=obs.get("last_action_error"),
            misc={}
        )

        action_nl = None
        refined_goal = None
        element_metadata = None
        page_url_before = obs.get("open_pages_urls", [None])[0] if obs.get("open_pages_urls") else None

        if oracle_action is None:
            # Use adaptive retry mechanism with character limit reduction
            response = self.make_llm_call_with_adaptive_retry(obs, current_step)

            raw_action = response.choices[0].message.content or ""
            logger.info(f"Raw LLM response (first 500 chars): {raw_action[:500]}")
            logger.info(f"Raw LLM response repr (first 200 chars): {repr(raw_action[:200])}")
            parsed_response = _extract_full_response(raw_action) if raw_action else None
            if parsed_response is None:
                raise ValueError(f"Could not parse action from LLM response. Raw (first 500 chars): {raw_action[:500]}")
            action = parsed_response.get("action")
            thought = parsed_response.get("thought")
            action_nl = parsed_response.get("action_nl")
            refined_goal = parsed_response.get("refined_goal")
            if action is None:
                raise ValueError(f"Could not parse action from LLM response. Raw (first 500 chars): {raw_action[:500]}")
            current_step.misc["model_usage"] = response.usage.to_dict()

        else:
            action, thought = oracle_action
            raw_action = f'{{"thought": "{thought}", "action": "{action}"}}'

        # Sanitize: strip hidden Unicode, zero-width chars, newlines in action
        action = sanitize_action(action)
        logger.info(f"Parsed action: {action}")
        logger.info(f"Parsed action repr: {repr(action)}")

        # Validate the action against known prefixes
        if not _is_action_valid(action):
            raise ValueError(
                f"Action '{action}' does not start with a valid prefix. "
                f"Valid prefixes: any function call pattern, e.g. click(...), go_back(...)"
            )

        # Extract element metadata from axtree based on the action's bid
        element_metadata = _extract_element_metadata(current_step.axtree, action)

        # Update refined_goal tracking
        if refined_goal:
            self._refined_goal = refined_goal
            logger.info(f"Refined goal updated: {refined_goal[:200]}")
        if action_nl:
            logger.info(f"Action (NL): {action_nl[:200]}")

        # Detect repeated action+error loops
        last_error = obs.get("last_action_error")
        if last_error:
            self._recent_action_errors.append((action, last_error))
            # Keep only last 10 entries
            if len(self._recent_action_errors) > 10:
                self._recent_action_errors = self._recent_action_errors[-10:]
            # Count consecutive occurrences of the same (action, error) pair
            same_count = 0
            for a, e in reversed(self._recent_action_errors):
                if a == action and e == last_error:
                    same_count += 1
                else:
                    break
            if same_count >= self._max_same_action_retries:
                raise ValueError(
                    f"Same action '{action}' failed {same_count} times with error: {last_error}. "
                    f"Forcing replan to break retry loop."
                )

        # Stuck detection: check if agent is cycling through same small action set
        if len(self.history) >= 5:
            recent_actions = [step.action for step in self.history[-5:]]
            unique_actions = set(recent_actions)
            if len(unique_actions) <= 2 and len(recent_actions) >= 4:
                logger.warning(
                    f"Stuck detection: only {len(unique_actions)} unique action(s) "
                    f"({unique_actions}) in last 5 steps. Model may be in a retry loop."
                )

        # Hard stuck detection: same action N times consecutively (with or without errors)
        same_action_threshold = 6
        if len(self.history) >= same_action_threshold:
            recent_actions = [step.action for step in self.history[-same_action_threshold:]]
            if len(set(recent_actions)) == 1:
                raise ValueError(
                    f"Same action '{recent_actions[0]}' executed {same_action_threshold} "
                    f"consecutive times. Task is stuck — forcing stop to prevent wasting steps."
                )

        logger.info(f"Raw Action:\n {raw_action}")

        current_step.action = action
        current_step.thought = thought
        current_step.last_action_error = obs.get("last_action_error")
        current_step.misc.update({
            "thought": thought,
            "parsed_action": action,
            "raw_action": raw_action,
            "action_nl": action_nl,
            "refined_goal": refined_goal or self._refined_goal,
            "element_metadata": element_metadata,
            "page_url_before": page_url_before,
        })

        self.history.append(current_step)

        logger.info(f"Action passed to env.step(): {action}")
        logger.info(f"Action passed to env.step() repr: {repr(action)}")
        return action, current_step.misc

    def make_llm_call_with_adaptive_retry(self, obs: dict, current_step: BrowserGymAgentStepData) -> dict:
        """
        Make a call to the LLM with adaptive retry that reduces character limit on failures.

        Args:
            obs (dict): The observation from the environment.
            current_step (BrowserGymAgentStepData): The current step data.

        Returns:
            dict: The response from the LLM.
        """
        max_attempts = 5
        attempt = 0
        current_char_limit = self.char_limit
        use_json_format = True  # Will be set to False if API doesn't support it

        while attempt < max_attempts:
            try:
                # Build messages with current character limit
                effective_char_limit = current_char_limit
                messages = self.prompt_builder.build_messages(
                    goal=obs["goal_object"][0]["text"],
                    current_step=current_step,
                    history=self.history,
                    char_limit=effective_char_limit,
                    refined_goal=self._refined_goal,
                    use_som=self.use_som,
                )['prompt']

                print(f"Attempt {attempt+1}: Using char_limit={current_char_limit}, json_mode={use_json_format}")

                client = self.client if attempt == 0 else self.client_long
                model = self.model_id if attempt == 0 else self.model_id_2

                # Make the API call
                kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1024,  # Qwen models emit verbose thoughts; 256 caused JSON truncation
                )
                if use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)

                # Check for empty/null content (some APIs return success with null content
                # when json_object format is not supported)
                content = response.choices[0].message.content
                if not content:
                    if use_json_format:
                        logger.warning("LLM returned empty content with json_object format, retrying without JSON mode")
                        use_json_format = False
                        attempt += 1
                        if attempt >= max_attempts:
                            raise ValueError("LLM returned empty content after all retries")
                        current_char_limit = int(current_char_limit * 0.95)
                        continue
                    else:
                        raise ValueError("LLM returned empty content even without JSON mode")

                return response

            except Exception as e:
                err_msg = str(e).lower()
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                    raise

                # If response_format json_object is not supported, disable it
                if "response_format" in err_msg or "json_object" in err_msg:
                    logger.warning("response_format not supported by API, disabling JSON mode")
                    use_json_format = False

                if attempt > 1:
                    current_char_limit = int(current_char_limit * 0.95)
                logger.warning(f"Retrying with {current_char_limit} character limit after error: {str(e)}")

                if attempt > 1:  # Skip delay for first retry
                    wait_time = 1.5 * (2 ** (attempt-1)) + (0.1 * attempt)
                    logger.info(f"Waiting {wait_time:.2f} seconds before retry")
                    time.sleep(wait_time)
                else:
                    logger.info("Retrying immediately")
