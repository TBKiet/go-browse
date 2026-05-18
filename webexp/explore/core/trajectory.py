from dataclasses import dataclass
import logging
import numpy as np
from PIL import Image
import json
import os

logger = logging.getLogger(__name__)

def _extract_text_obs(observation):
    """
    Recursively extract all textual observations from a nested structure.
    """
    if isinstance(observation, str):
        return observation
    elif isinstance(observation, dict):
        return {k: _extract_text_obs(v) for k, v in observation.items() if isinstance(v, (str, dict, list))}
    elif isinstance(observation, list):
        return [_extract_text_obs(item) for item in observation if isinstance(item, (str, dict, list))]
    return None

@dataclass
class TrajectoryStep:
    action: str | None
    parsed_action: str | None
    thought: str | None
    observation: dict
    # Semantic fields for richer trajectory representation
    action_nl: str | None = None
    refined_goal: str | None = None
    action_reasoning: str | None = None
    bounding_box: dict | None = None
    element_metadata: dict | None = None
    page_url_before: str | None = None
    page_url_after: str | None = None
    misc: dict | None = None

    def __post_init__(self):
        self._last_saved_dir = None

    def save(self, save_dir: str, keep_image_in_memory: bool=False, save_image: bool=True):
        # Extract all textual observations, including nested collections
        text_obs = _extract_text_obs(self.observation)
        step_info = {
            "action": self.action,
            "parsed_action": self.parsed_action,
            "thought": self.thought,
            "observation": text_obs,
            "action_nl": self.action_nl,
            "refined_goal": self.refined_goal,
            "action_reasoning": self.action_reasoning,
            "bounding_box": self.bounding_box,
            "element_metadata": self.element_metadata,
            "page_url_before": self.page_url_before,
            "page_url_after": self.page_url_after,
            "misc": self.misc
        }
        
        with open(os.path.join(save_dir, "step_info.json"), "w") as f:
            json.dump(step_info, f, indent=4)
        
        if 'screenshot' in self.observation:
            if save_image:
                # Save screenshot
                screenshot = self.observation["screenshot"]
                img = Image.fromarray(screenshot)
                img.save(os.path.join(save_dir, "screenshot.png"))

            if not keep_image_in_memory:
                # Remove the screenshot from memory to save space
                del self.observation["screenshot"]
                self.observation = {k: v for k, v in self.observation.items() if k != "screenshot"}
            
        self._last_saved_dir = save_dir

    @staticmethod
    def load(load_dir: str, load_image: bool=True):
        with open(os.path.join(load_dir, "step_info.json"), "r") as f:
            step_info = json.load(f)

        if load_image:
            screenshot = np.asarray(Image.open(os.path.join(load_dir, "screenshot.png")))
            step_info["observation"]["screenshot"] = screenshot

        return TrajectoryStep(
            step_info["action"],
            step_info["parsed_action"],
            step_info["thought"],
            step_info["observation"],
            step_info.get("action_nl"),
            step_info.get("refined_goal"),
            step_info.get("action_reasoning"),
            step_info.get("bounding_box"),
            step_info.get("element_metadata"),
            step_info.get("page_url_before"),
            step_info.get("page_url_after"),
            step_info.get("misc"),
        )
    
    @property
    def last_saved_dir(self) -> str | None:
        return self._last_saved_dir


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    final_state: TrajectoryStep | None
    goal: str
    reward: float
    success: bool
    response: str
    agent_info: dict
    misc: dict

    def __post_init__(self):
        self._save_dir: str | None = None
    
    def add_step(self, action: str, parsed_action: str | None, thought: str | None, observation: dict, misc: dict = None,
                 action_nl: str | None = None, refined_goal: str | None = None, action_reasoning: str | None = None,
                 bounding_box: dict | None = None, element_metadata: dict | None = None,
                 page_url_before: str | None = None, page_url_after: str | None = None):
        self.steps.append(TrajectoryStep(
            action, parsed_action, thought, observation,
            action_nl=action_nl, refined_goal=refined_goal, action_reasoning=action_reasoning,
            bounding_box=bounding_box, element_metadata=element_metadata,
            page_url_before=page_url_before, page_url_after=page_url_after,
            misc=misc
        ))
    
    def extract_response(self, env):
        chat_messages = env.chat.messages
        if chat_messages and chat_messages[-1]["role"] == "assistant":
            self.response = chat_messages[-1]["message"]
        elif chat_messages and chat_messages[-1]["role"] == "infeasible":
            self.response = "User goal/request is infeasible."
            
        return self.response
    
    def save(self, save_dir: str):
        self._save_dir = save_dir
        traj_info = {
            "goal": self.goal,
            "reward": self.reward,
            "success": self.success,
            "response": self.response,
            "agent_info": self.agent_info,
            "misc": self.misc
        }

        with open(os.path.join(save_dir, "traj_info.json"), "w") as f:
            json.dump(traj_info, f, indent=4)
        
        for i, step in enumerate(self.steps):
            
            step_save_dir = os.path.join(save_dir, f"step_{i}")
            os.makedirs(step_save_dir, exist_ok=True)
            
            step.save(step_save_dir)

        final_state_save_dir = os.path.join(save_dir, "final_state")
        os.makedirs(final_state_save_dir, exist_ok=True)
        if self.final_state is not None:
            self.final_state.save(final_state_save_dir)

    def save_info(self):
        """Re-save only the trajectory info JSON (not steps/screenshots).

        Useful for updating metadata like summarized_goal after the trajectory
        has already been saved to disk.
        """
        if self._save_dir is None:
            logger.warning("Cannot save_info: trajectory was never saved to disk")
            return
        traj_info = {
            "goal": self.goal,
            "reward": self.reward,
            "success": self.success,
            "response": self.response,
            "agent_info": self.agent_info,
            "misc": self.misc
        }
        with open(os.path.join(self._save_dir, "traj_info.json"), "w") as f:
            json.dump(traj_info, f, indent=4)

    @staticmethod
    def load(load_dir: str, load_steps: bool=True, load_images: bool=True):
        with open(os.path.join(load_dir, "traj_info.json"), "r") as f:
            traj_info = json.load(f)
        
        steps = []
        if load_steps:
            i = 0
            while os.path.exists(os.path.join(load_dir, f"step_{i}")):
                step_load_dir = os.path.join(load_dir, f"step_{i}")
                steps.append(TrajectoryStep.load(step_load_dir, load_image=load_images))
                i += 1

        final_state_load_dir = os.path.join(load_dir, "final_state")
        if os.path.exists(os.path.join(final_state_load_dir, "step_info.json")):
            final_state = TrajectoryStep.load(final_state_load_dir, load_image=load_images)
        else:
            final_state = None
        
        return Trajectory(steps, final_state, traj_info["goal"], traj_info["reward"], traj_info["success"], traj_info["response"], traj_info["agent_info"], traj_info["misc"])
        
    def __len__(self):
        return len(self.steps)
    
    @staticmethod    
    def from_goal(goal: str, agent_info: dict = None, misc: dict = None):
        if agent_info is None:
            agent_info = {}
        if misc is None:
            misc = {}
        return Trajectory([], None, goal, 0.0, False, "N/A", agent_info=agent_info, misc=misc)
