import os
import json
import re
from glob import glob
from tqdm import tqdm
from webexp.explore.core.trajectory import Trajectory
from webexp.explore.core.node import Node
from webexp.explore.core.graph import Graph
import networkx as nx
import random

OUTPUT_FILE_NAME = "<PLACEHOLDER_FOR_OUTPUT_FILE.jsonl>"  # Replace with your desired output file path


DIRS_TO_LOAD = [
    "<PLACEHOLDER_PATH_FOR_GRAPH_DIR_1>",  # Replace with your desired graph directory path
    "<PLACEHOLDER_PATH_FOR_GRAPH_DIR_2>",  # Replace with your desired graph directory path
    ...
]

MAX_NODES_PER_DIR = 20

TRAJ_PATTERNS = [
    'tasks/*/*_trajs/*/',
]

def extract_domain(path):
    """
    Extract domain name from a run directory path.
    
    Example path: "data/runs/4432878-map/graph"
    Returns: "map"
    """
    # Use regex to extract the domain name between the last hyphen and /graph
    match = re.search(r'/runs/\d+-([^/]+)/graph', path)
    if match:
        return match.group(1)
    
    # Alternative approach using string manipulation if regex fails
    try:
        # Split the path and locate the directory containing 'graph'
        parts = path.split('/')
        for i, part in enumerate(parts):
            if part == 'graph' and i > 0:
                # Get the parent directory name and extract domain after hyphen
                parent_dir = parts[i-1]
                if '-' in parent_dir:
                    return parent_dir.split('-', 1)[1]
    except:
        pass
    
    # Return None or the original path if parsing fails
    return None


def build_nx_graph(g: Graph):
    
    netG = nx.DiGraph()
    
    node_urls = list(g.nodes.keys())
    for i, node in enumerate(g.nodes.values()):
        for prefix in node.prefixes:
            prefix_misc = prefix.misc
            if prefix_misc and prefix.start_url in node_urls:
                task_misc = prefix_misc.get("task_misc", None)
                sourced_from_nav_explore_task = task_misc is not None and 'agent_info' in task_misc and task_misc['agent_info']['name'] == 'NavExplorerAgent'
                
                tags = prefix_misc.get("tags", None)
                task_tagged_as_nav = "NAV" in tags if tags else False
                
                if task_tagged_as_nav or sourced_from_nav_explore_task:
                    if netG.has_edge(prefix.start_url, prefix.end_url):
                        old_weight = netG[prefix.start_url][prefix.end_url]['weight']
                        if old_weight > len(prefix):
                            netG[prefix.start_url][prefix.end_url]['weight'] = len(prefix)
                            netG[prefix.start_url][prefix.end_url]['trace'] = prefix
                    else:
                        netG.add_edge(prefix.start_url, prefix.end_url, weight=len(prefix), trace=prefix)
                        
    return netG


def get_prefixes(g: Graph, netG: nx.DiGraph):
    shortest_paths = dict(nx.single_source_all_shortest_paths(netG, source=g.root.url, weight='weight'))
    
    prefixes = {}
    
    for k in shortest_paths.keys():
        paths = shortest_paths[k]
        
        prefixes[k] = []
        
        for path in paths:
            
            curr_prefix = []
            
            for i in range(len(path) - 1):
                start_url = path[i]
                end_url = path[i + 1]
                
                curr_prefix.extend(netG[start_url][end_url]['trace'].steps)
                
            for step in curr_prefix:
                step.misc['is_prefix_step'] = True

            prefixes[k].append(curr_prefix)
    
    return prefixes
        
        
if __name__ == "__main__":

    total_trajs = 0
    total_steps = 0


    with open(OUTPUT_FILE_NAME, "w") as f:
        for directory in tqdm(DIRS_TO_LOAD, desc="Domains", position=0, leave=False):
            
            g = Graph.load(directory, load_steps=True, load_prefixes=True, load_images=False, max_nodes=MAX_NODES_PER_DIR)
            
            netG = build_nx_graph(g)
            
            prefixes = get_prefixes(g, netG)
                
            for _, node in tqdm(enumerate(g.nodes.values()), desc="Nodes", position=1, leave=False):
                for task in tqdm(node.tasks.values(), desc="Tasks", position=2, leave=False):
                    for traj in task.positive_trajs + task.negative_trajs:
                        sampled_prefix = random.choice(prefixes[node.url]) if node.url in prefixes else []
                        traj_steps = traj.steps
                        
                        is_traj_prefixed = False
                        
                        if (not ("needs_prefix" in traj.misc and not traj.misc["needs_prefix"])) and traj.reward > 0 and sampled_prefix:
                            traj_steps = sampled_prefix + traj_steps
                            is_traj_prefixed = True
                            
                        for i, step in enumerate(traj_steps):
                            
                            processed_obs = {k: v for k, v in step.observation.items() if k != "extra_element_properties"}
                            
                            step_data = {
                                "action": step.action,
                                "parsed_action": step.parsed_action,
                                "thought": step.thought, 
                                "obs": processed_obs,
                                "misc": step.misc,
                                "step_number": i,
                            }
                            
                            traj_data = {
                                "goal": traj.goal,
                                "reward": traj.reward,
                                "success": traj.success,
                                "response": traj.response,
                                "misc": traj.misc,
                                "traj_num": total_trajs,
                                "traj_length": len(traj_steps),
                                "is_traj_prefixed": is_traj_prefixed,
                            }
                            
                            node_data = {
                                "node_url": node.url
                            }
                            
                            graph_data = {
                                "domain": extract_domain(directory),
                                "root_url": g.root.url,
                            }
                            
                            row_data = {
                                "step_data": step_data,
                                "traj_data": traj_data,
                                "node_data": node_data,
                                "graph_data": graph_data,
                            }
                            
                            f.write(json.dumps(row_data) + "\n")
                            
                            total_steps += 1
                        total_trajs += 1

    print("Total Trajs: ", total_trajs)
    print("Total Steps: ", total_steps)