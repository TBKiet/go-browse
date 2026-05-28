from .trajectory import TrajectoryStep
from dataclasses import dataclass
import json
import os

@dataclass
class Trace:
    """A trace of a sequence of steps"""

    steps: list[TrajectoryStep]
    start_url: str
    end_url: str
    misc: dict | None = None

    @classmethod
    def from_trajectory_steps(
        cls,
        steps: list[TrajectoryStep| str],
        start_url: str,
        end_url: str,
        misc: dict | None = None,
    ):
        return cls(steps, start_url, end_url, misc)
    
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        trace_info = {
            "start_url": self.start_url,
            "end_url": self.end_url,
            "misc": self.misc
        }
        
        with open(os.path.join(save_dir, "trace_info.json"), "w") as f:
            json.dump(trace_info, f, indent=4)
        
        for i, step in enumerate(self.steps):
            step_save_dir = os.path.join(save_dir, f"step_{i}")
            os.makedirs(step_save_dir, exist_ok=True)
            step.save(step_save_dir, keep_image_in_memory=True, save_image=False)

    def load(load_dir: str, load_steps: bool=True, load_images: bool=False):
        with open(os.path.join(load_dir, "trace_info.json"), "r") as f:
            trace_info = json.load(f)
        
        steps = []
        if load_steps:
            for i in range(len(os.listdir(load_dir)) - 1):
                step_load_dir = os.path.join(load_dir, f"step_{i}")
                steps.append(TrajectoryStep.load(step_load_dir, load_image=load_images))
        else: 
            steps = os.listdir(load_dir)
        
        return Trace(steps, trace_info["start_url"], trace_info["end_url"], trace_info["misc"])
    
    def __len__(self):
        return len(self.steps)