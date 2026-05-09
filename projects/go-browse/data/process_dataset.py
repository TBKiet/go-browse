import os
import json
import re
from glob import glob
from tqdm import tqdm
from browsergym.core.action.highlevel import HighLevelActionSet


from webexp.agents.solver_agent import SolverAgent
from webexp.agents.prompt_builders.solver_prompt_builder import SolverPromptBuilder
from webexp.agents.trajectory_data import BrowserGymAgentStepData, BrowserGymAgentTrajectoryData

# Note: This script assumes input data is ordered by trajectory and starts with traj_num 0

INCLUDE_PREFIX = False

INPUT_DATA_FILE = "<PLACEHOLDER_FOR_INPUT_FILE.jsonl>"  # Replace with your desired input file path
OUTPUT_DATA_FILE = "<PLACEHOLDER_FOR_OUTPUT_FILE.jsonl>" # Replace with your desired output file path

PREFIX_PROB = 0

if __name__ == "__main__":

    action_set = HighLevelActionSet(
                subsets=["chat", "bid", "infeas", "nav"],
                strict=False,
                multiaction=False,
                demo_mode=False
            )

    prompt_builder = SolverPromptBuilder(action_set)

    total_trajs = 0
    total_steps = 0
    prev_traj_data = None
    
    
    with open(INPUT_DATA_FILE, "r") as f:
        with open(OUTPUT_DATA_FILE, "w") as out_f:

            steps = []
            curr_traj = 0
            traj_goal = ""
            for line in tqdm(f, desc="Processing lines", position=0, leave=False):
                
                line_data = json.loads(line) 
                traj_data = line_data['traj_data']
                
                if curr_traj == traj_data['traj_num'] - 1:
                    
                    curr_traj += 1

                    traj = BrowserGymAgentTrajectoryData(steps, prev_traj_data['goal'], prev_traj_data['reward'], prev_traj_data['misc']).process_for_dataset()

                    steps = []
                    
                    if 'skip' in traj.misc and traj.misc['skip']:
                        continue
                    
                    total_trajs += 1

                    step_data_points = prompt_builder.build_trajectory_messages(traj, char_limit=80000)
                    for i, d in enumerate(step_data_points):
                        d_wrap = {'step_idx': total_steps, 'step_data': d, 'traj_reward': traj.reward, 'next_step_idx': (total_steps + 1) if i != (len(step_data_points) - 1) else -1, 'traj_length': len(step_data_points), 'step_number': i}
                        out_f.write(json.dumps(d_wrap) + "\n")
                        total_steps += 1
                
                prev_traj_data = traj_data
                
                step_data = line_data['step_data']
                if INCLUDE_PREFIX or not ('is_prefix_step' in step_data['misc'] and step_data['misc']['is_prefix_step']):
                    steps.append(BrowserGymAgentStepData(
                        action=step_data['parsed_action'],
                        thought=step_data['thought'],
                        axtree=step_data['obs']['axtree_txt'],
                        last_action_error=step_data['obs']['last_action_error'],
                        misc=step_data['misc'],
                    ))
            
            if steps:  # If there are steps collected for the last trajectory
                traj = BrowserGymAgentTrajectoryData(steps, prev_traj_data['goal'], prev_traj_data['reward'], prev_traj_data['misc']).process_for_dataset()
                
                step_data_points = prompt_builder.build_trajectory_messages(traj, char_limit=80000)
                for i, d in enumerate(step_data_points):
                    d_wrap = {'step_idx': total_steps, 'step_data': d, 'traj_reward': traj.reward, 'next_step_idx': (total_steps + 1) if i != (len(step_data_points) - 1) else -1, 'traj_length': len(step_data_points), 'step_number': i}
                    out_f.write(json.dumps(d_wrap) + "\n")
                    total_steps += 1
                total_trajs += 1
            

    print("Total Trajs: ", total_trajs)
    print("Total Steps: ", total_steps)
