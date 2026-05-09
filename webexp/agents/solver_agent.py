from .base_agent import AgentFactory, BaseAgent
from .prompt_builders.solver_prompt_builder import SolverPromptBuilder
from .trajectory_data import BrowserGymAgentStepData
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from openai import OpenAI
from tenacity import retry, before_sleep_log, stop_after_attempt, wait_exponential, wait_random
import ast
import logging
import os
import re
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def messages_to_string(messages: list[dict]) -> str:
    prompt_text_strings = []
    for message in messages:
        prompt_text_strings.append(message["content"])
    full_prompt_txt = "\n".join(prompt_text_strings)
    return full_prompt_txt
        

def extract_action_and_thought(raw_string):
    """Extract thought and action from potentially malformed JSON string.
    
    Args:
        raw_string (str): Raw string containing thought and action
        
    Returns:
        tuple: (action, thought) or (None, None) if extraction fails
    """
    # Initialize defaults
    thought = None
    action = None
    
    try:
        # Look for thought pattern using non-greedy match
        thought_match = re.search(r'"thought"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1)
            # Clean up escaped quotes
            thought = thought.replace('\\"', '"')
            
        # Look for action pattern using non-greedy match    
        action_match = re.search(r'"action"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if action_match:
            action = action_match.group(1)
            # Clean up escaped quotes
            action = action.replace('\\"', '"')
            
    except Exception as e:
        print(f"Error parsing string: {e}")
        return None, None
        
    return action, thought


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
    ):
        """
        Initialize the agent.

        Args:
            model_name (str): The name of the model to use.
            temperature (float): The temperature to use for sampling.
            demo_mode (bool): Whether to run in demo mode.
        """
        
        # These are args that will be specified in the config.
        super().__init__(model_id=model_id, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode)
        
        self.model_id = model_id
        self.model_id_2 = model_id_2 or model_id
        self.temperature = temperature
        self.char_limit = char_limit
        self.demo_mode = demo_mode
        

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

    def reset(self):
        self.history.clear()

    def obs_preprocessor(self, obs: dict) -> dict:

        return {
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
        }
    
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
            axtree=obs["axtree_txt"],
            last_action_error=obs.get("last_action_error"),
            misc={}
        )

        if oracle_action is None:
            # Use adaptive retry mechanism with character limit reduction
            response = self.make_llm_call_with_adaptive_retry(obs, current_step)
            
            raw_action = response.choices[0].message.content
            action, thought = extract_action_and_thought(raw_action)
            current_step.misc["model_usage"] = response.usage.to_dict()
        
        else:
            action, thought = oracle_action
            raw_action = f'{{"thought": "{thought}", "action": "{action}"}}'
            
        print(f"Raw Action:\n {raw_action}")

        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({"thought": thought, "parsed_action": action})
        
        self.history.append(current_step)

        return raw_action, current_step.misc
        
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
        
        while attempt < max_attempts:
            try:
                # Build messages with current character limit
                messages = self.prompt_builder.build_messages(
                    goal=obs["goal_object"][0]["text"],
                    current_step=current_step,
                    history=self.history,
                    char_limit=current_char_limit if (attempt == 0) or (current_char_limit < 0) else current_char_limit * 2 # TODO: Ad-hoc!
                )['prompt']
                
                print(f"Attempt {attempt+1}: Using char_limit={current_char_limit}")
                
                if attempt == 0:
                    # Make the actual API call
                    return self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=self.temperature
                    )
                else:
                    return self.client_long.chat.completions.create(
                        model=self.model_id_2,
                        messages=messages,
                        temperature=self.temperature
                    )
                
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                    raise
                    
                if attempt > 1:
                    current_char_limit = int(current_char_limit * 0.95)
                logger.warning(f"Retrying with {current_char_limit} character limit after error: {str(e)}")
                
                if attempt > 1:  # Skip delay for first retry
                    wait_time = 1.5 * (2 ** (attempt-1)) + (0.1 * attempt)
                    logger.info(f"Waiting {wait_time:.2f} seconds before retry")
                    time.sleep(wait_time)
                else:
                    logger.info("Retrying immediately")
