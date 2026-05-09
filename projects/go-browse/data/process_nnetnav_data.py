#%%
import os
import json
from datasets import load_dataset
from tqdm import tqdm
from browsergym.core.action.highlevel import HighLevelActionSet
from webexp.agents.solver_agent import SolverAgent
from webexp.agents.prompt_builders.solver_prompt_builder import SolverPromptBuilder
from webexp.agents.trajectory_data import BrowserGymAgentStepData, BrowserGymAgentTrajectoryData

OUTPUT_DATA_FILE = "<PLACEHOLDER_FOR_OUTPUT_FILE.jsonl>"  # Replace with your desired output file path

#%%
def action_convert(action):
    if 'type' in action:
        action = action.replace("type", "fill")
        id = action.split("[")[1].split("]")[0]
        content = action.split("[")[2].split("]")[0]
        if action.count("[") > 3:
            press_enter_after = action.split("[")[3].split("]")[0]
        else:
            press_enter_after = '0'
        action = f"fill(\'{id}\', \'{content}\')"
        if press_enter_after == "1":
            action += f"\npress(\'{id}\', 'Enter')"
    elif 'press' in action:
        key_comb = action.split("[")[1].split("]")[0]
        action = f"press(None, \'{key_comb}\')"
    elif 'scroll' in action:
        direction = action.split("[")[1].split("]")[0]
        if direction == 'down':
            action = f"scroll(0, 500)"
        elif direction == 'up':
            action = f"scroll(0, -500)"
    elif 'close_tab' in action:
        action = action.replace("close_tab", "tab_close")
        action = f"tab_close()"
    elif 'stop' in action:
        action = action.replace("stop", "send_msg_to_user")
        text = action.split("[")[1].split("]")[0]
        action = f"send_msg_to_user(\'{text}\')"
    elif 'click' in action:
        id = action.split("[")[1].split("]")[0]
        action = f"click(\'{id}\')"
    elif 'hover' in action:
        id = action.split("[")[1].split("]")[0]
        action = f"hover(\'{id}\')"
    elif 'new_tab' in action:
        action = f"new_tab()"
    elif 'tab_focus' in action:
        id = action.split("[")[1].split("]")[0]
        action = f"tab_focus(\'{id}\')"
    elif 'goto' in action:
        url = action.split("[")[1].split("]")[0]
        action = f"goto(\'{url}\')"
    elif 'go_back' in action:
        action = f"go_back()"
    elif 'go_forward' in action:
        action = f"go_forward()"
    return action

#%%
def parse_example(example):
    parts = example["output"].split("In summary, ", 1)
    think = parts[0].strip()
    original_output = example["output"]
    
    # Remove assistant header from think section
    think = think.replace("<|start_header_id|>assistant<|end_header_id|>", "")\
        .replace("Let's think step-by-step.", "")\
        .replace("Let's think step by step.", "")\
        .strip()
    
    if len(parts) > 1:
        action = parts[1].split("<|eot_id|>")[0].strip()
        if "``````" in action:
            action_parts = action.split("```")
            if len(action_parts) > 2:
                action = action_parts[2].split("``````")[0]
                action = action_convert(action)
            else:
                action = ""
        elif "```" in action:
            # Split by triple backticks and process each action
            action_parts = action.split("```")
            # Extract only the action part after "In summary, my next action should be"
            if len(action_parts) > 1:
                action = action_parts[1].strip()
                action = action_convert(action)
            else:
                action = ""
    else:
        action = ""
    
    prompt = example['prompt']
    
    # Extract content between OBJECTIVE and PREVIOUS ACTIONS
    objective_start = prompt.find("OBJECTIVE:")
    previous_actions_start = prompt.find("PREVIOUS ACTIONS:")
    
    if objective_start != -1 and previous_actions_start != -1:
        # Start after "OBJECTIVE:" and remove any leading/trailing whitespace
        objective_content = prompt[objective_start + len("OBJECTIVE:"):previous_actions_start].strip()
    else:
        objective_content = ""
        
    axtree = prompt.split("OBSERVATION:")[1].split("\nURL: ")[0].strip()
    
    step_data = BrowserGymAgentStepData(
                        action=action,
                        thought=think,
                        axtree=axtree,
                        last_action_error=None,
                        misc={'goal': objective_content},
                    )
    
    return step_data


#%%
dataset = load_dataset("stanfordnlp/nnetnav-wa", split='train')
    
# %%
unique_task_names = sorted(set(dataset['task_name']))


#%%
action_set = HighLevelActionSet(
                subsets=["chat", "bid", "infeas", "nav"],
                strict=False,
                multiaction=False,
                demo_mode=False
            )

prompt_builder = SolverPromptBuilder(action_set)

# %% Process each task group separately

total_trajs = len(unique_task_names)
total_steps = 0
skipped_steps = 0


processed_data = []

with open(OUTPUT_DATA_FILE, "w") as out_f:
    
    curr_task_name = dataset[0]['task_name']    
    steps = []
    
    for t, example in tqdm(enumerate(dataset), desc="Processing examples", position=0, leave=False):
        if example['task_name'] != curr_task_name:
            curr_task_name = example['task_name']
            
            traj = BrowserGymAgentTrajectoryData(steps, steps[0].misc['goal'], 1, {}).process_for_dataset()
            
            step_data_points = prompt_builder.build_trajectory_messages(traj, char_limit=80000)
            for i, d in enumerate(step_data_points):
                d_wrap = {'step_idx': total_steps, 'step_data': d, 'traj_reward': traj.reward, 'next_step_idx': (total_steps + 1) if i != (len(step_data_points) - 1) else -1, 'traj_length': len(step_data_points), 'step_number': i}
                out_f.write(json.dumps(d_wrap) + "\n")
                total_steps += 1
            
            steps = []
        
        try:
            steps.append(parse_example(example))
        except Exception as e:
            print(f"Error processing example {t}: {e}")
            skipped_steps += 1
            continue

print(f"Total steps: {total_steps}, Skipped steps: {skipped_steps}")
print(f"Total trajectories: {total_trajs}")
