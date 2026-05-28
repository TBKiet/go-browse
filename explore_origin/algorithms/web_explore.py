from ..core.agent import AgentWithExplorationCallbacks, ExplorerAgentWithExplorationCallbacks, wrap_agent_for_callback_protocol
from ..core.evaluator import Evaluator
from ..core.episode import run_episode, get_action, perform_env_step
from ..core.graph import Graph
from ..core.node import Node
from ..core.task import Task
from ..core.trace import Trace
from ..core.trajectory import Trajectory
from ...agents.base_agent import AgentFactory
from browsergym.core.env import BrowserEnv
from browsergym.experiments.loop import EnvArgs
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
from pathlib import Path
from typing import Sequence, List, Dict, Optional
import argparse
import logging
import os
import random
import requests
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class WebExploreAgentConfig:
    """
    Configuration for the Explorer agents.

    Attributes:
        agent_name (str): Name of the agent.
        agent_factory_args (Dict): Arguments for the agent factory.
        max_steps (int): Maximum steps for the agent.
        retries (int): Number of retries for the agent.
    """
    agent_factory_args: Dict
    max_steps: int
    retries: int

@dataclass
class WebExploreConfig:
    """
    Configuration for the WebExplore algorithm.

    Attributes:
        env (Dict): Environment configuration.
        evaluator (Dict): Evaluator configuration.
        max_nodes (int): Maximum number of nodes to explore.
        resume_from (Optional[str]): Path to resume from.
        page_explorers (List[WebExploreAgentConfig]): List of page explorer agent configurations.
        nav_explorers (List[WebExploreAgentConfig]): List of navigation explorer agent configurations.
        feasibility_checkers (List[WebExploreAgentConfig]): List of feasibility checker agent configurations.
        solvers (List[WebExploreAgentConfig]): List of solver agent configurations.
        allowlist_patterns (List[str]): List of URL patterns to allow.
        denylist_patterns (List[str]): List of URL patterns to block/deny.
        max_feasible_page_explorer_tasks_per_node (int): Maximum feasible tasks per node for page explorers.
        max_feasible_nav_explorer_tasks_per_node (int): Maximum feasible tasks per node for navigation explorers.
        exp_dir (str): Directory for saving exploration data.
        full_reset_url (Optional[str]): URL for full reset.
    """
    env: Dict
    evaluator: Dict
    max_nodes: int
    resume_from: Optional[str]
    page_explorers: List[WebExploreAgentConfig]
    nav_explorers: List[WebExploreAgentConfig]
    feasibility_checkers: List[WebExploreAgentConfig]
    solvers: List[WebExploreAgentConfig]
    allowlist_patterns: List[str]
    denylist_patterns: List[str]
    exp_dir: str
    max_feasible_page_explorer_tasks_per_node: int
    max_feasible_nav_explorer_tasks_per_node: int
    full_reset_url: Optional[str]


def perform_full_reset(full_reset_url: str, num_retries: int = 3):
    """
    Perform a full reset of the environment by sending a POST request to the specified URL.
    """
    for _ in range(num_retries):
        try:
            response = requests.post(full_reset_url)
            if response.status_code == 200:
                logger.info(f"Full reset successful: {response.text}")
                return
            else:
                logger.error(f"Full reset failed: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during full reset: {e}")

    logger.error("Failed to perform full reset after multiple attempts.")

def backtrack_if_needed(
    agent, step_num: int, goal: str, env: BrowserEnv, graph: Graph, node: Node, traj: Trajectory, obs: dict,
        reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict
):
    """
    Callback to check if we are on a blocked page and backtrack if needed.
    """
    open_urls = obs['open_pages_urls']
    for url in open_urls:
        if not graph.check_if_url_allowed(url):

            logger.info(f"Blocked page detected: {url}")

            oracle_action = (
                "go_back()",
                "I am not permitted to view this page as it is on a blocklist,\
                    I will return back to the previous page and try something else."
            )
            
            action = get_action(
                env=env,
                agent=agent,
                obs=obs,
                traj=traj,
                oracle_action=oracle_action
            )

            obs, reward, terminated, truncated, env_info = perform_env_step(
                env=env,
                agent=agent,
                action=action,
            )

            logger.info(f"Backtracked to {obs['open_pages_urls'][-1]}")

    return step_num, obs, reward, terminated, truncated, env_info, goal, callback_context

def prestep_store_url(
    agent, step_num: int, goal: str, env: BrowserEnv, graph: Graph, node: Node, traj: Trajectory, obs: dict,
        reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict
):
    """
    Callback to log the active url before the step.
    """
    callback_context['pre_step_url'] = env.page.url
    return step_num, obs, reward, terminated, truncated, env_info, goal, callback_context

def backtrack_when_new_page_found(
    agent, step_num: int, goal: str, env: BrowserEnv, graph: Graph, node: Node, traj: Trajectory, obs: dict,
        reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict
    ):
    """
    Callback to check if we are on a new page and backtrack if needed.
    """

    open_urls = obs['open_pages_urls']
    if len(open_urls) > 1:
        for i in range(len(open_urls) - 1):
            oracle_action = (
                "close_tab()",
                "I have opened a new tab. It is better to just use a single tab when exploring. \
                    I will close tab and return to the original tab to resume exploring."
            )

            action = get_action(
                env=env,
                agent=agent,
                obs=obs,
                traj=traj,
                oracle_action=oracle_action
            )

            obs, reward, terminated, truncated, env_info = perform_env_step(
                env=env,
                agent=agent,
                action=action,
            )

            logger.info(f"Closed tab {open_urls[i]}")

    open_urls = obs['open_pages_urls']

    if open_urls[0] != callback_context['pre_step_url']:
        oracle_action = (
            "go_back()",
            "I was successfully able to navigate to the new page. Since I was able to successfully navigate to a new page, \
                I should add a corresponding navigation task to the dataset next. But first, I will navigate back to the previous page."
        )

        action = get_action(
            env=env,
            agent=agent,
            obs=obs,
            traj=traj,
            oracle_action=oracle_action
        )
        obs, reward, terminated, truncated, env_info = perform_env_step(
            env=env,
            agent=agent,
            action=action,
        )
        logger.info(f"Backtracked to {obs['open_pages_urls'][-1]}")
        
    return step_num, obs, reward, terminated, truncated, env_info, goal, callback_context


def sample_task_candidates_for_node(
    env: BrowserEnv,
    explorer: ExplorerAgentWithExplorationCallbacks,
    evaluator: Evaluator,
    graph: Graph,
    node: Node,
    max_steps: int,
    max_retries: int = 3
) -> tuple[Task]:
    
    goal = explorer.goal_str
    tasks = []

    logger.info(f"Sampling tasks for node {node.url} with agent config:\n{explorer.get_config()}")

    retry = 0
    while not tasks and retry < max_retries:
        logger.info(f"Sampling tasks for node {node.url}. On Retry {retry}/{max_retries}.")

        traj = run_episode(
            goal=goal,
            node=node,
            env=env,
            agent=explorer,
            evaluator=evaluator,
            graph=graph,
            max_steps=max_steps
        )

        node.add_exploration_traj(traj)

        tasks.extend(explorer.get_proposed_tasks())

        logger.info(f"On Retry {retry}. Sampled tasks for node {node.url}:\n{tasks}.")

        retry += 1


    return node.add_tasks(tasks, task_misc={'agent_info': explorer.get_config()})


def filter_to_feasible_tasks_for_node(
    tasks: List[Task],
    env: BrowserEnv,
    feasibility_checker: AgentWithExplorationCallbacks,
    evaluator: Evaluator,
    graph: Graph,
    node: Node,
    max_steps: int | Sequence[int] = 10,
    max_retries: int = 3,
    max_feasible_tasks: Optional[int] = None,
):
    
    # Shuffle tasks if max_feasible_tasks is provided to ensure diversity
    if max_feasible_tasks is not None:
        random.shuffle(tasks)
    
    # TODO: We may want to account for the more general case where we can have multiple feasibility checkers.
    # In this case, we would need to initialize this count to the number of feasible tasks found so far for filtered to tasks with similar agent_configs to input tasks.
    feasible_count = 0
    
    for i, task in enumerate(tasks):
        trajs = []
        for r in range(max_retries):
            try:
                traj = run_episode(
                    goal=task.goal,
                    node=node,
                    env=env,
                    agent=feasibility_checker,
                    evaluator=evaluator,
                    graph=graph,
                    max_steps=max_steps,
                    callback_context={"task_misc": task.misc}  # Pass task misc to the callback context
                )

                trajs.append(traj)

                if traj.success:
                    feasible_count += 1
                    break

            except Exception as e:
                logger.error(f"Error checking feasibility for node {node.url} and task {task} on retry {r}: {e}")
                logger.error(traceback.format_exc())

        node.add_trajectories(trajs)
        
        # Early termination if we've found enough feasible tasks
        if max_feasible_tasks is not None and feasible_count >= max_feasible_tasks:
            logger.info(f"Found {feasible_count} feasible tasks (max: {max_feasible_tasks}). Stopping feasibility checking early.")
            break


def sample_task_solving_trajectories_for_node(
    node: Node,
    env: BrowserEnv,
    agent: AgentWithExplorationCallbacks,
    evaluator: Evaluator,
    graph: Graph,
    max_steps: int,
    num_trajs_per_task: int,
):
    tasks = node.get_feasible_tasks()

    logger.info(f"Sampling trajectories for node {node.url} with agent config:\n{agent.get_config()}")
    logger.info(f"Node has {len(tasks)} feasible tasks.")

    for task in tasks:

        logger.info(f"Sampling prefixed trajectories for node {node.url} and task {task.goal}.")

        for _ in range(num_trajs_per_task):

            try:
                traj = run_episode(
                    goal=task.goal,
                    node=node,
                    env=env,
                    agent=agent,
                    evaluator=evaluator,
                    graph=graph,
                    max_steps=max_steps,
                    callback_context={"task_misc": task.misc}
                )

                traj.misc["needs_prefix"] = True

                node.add_trajectory(traj)

            except Exception as e:
                logger.error(f"Error sampling trajectories for node {node.url} and task {task.goal}: {e}")
                logger.error(traceback.format_exc())


        for _ in range(num_trajs_per_task):

            try:
                traj = run_episode(
                    goal=task.goal,
                    node=graph.root,
                    env=env,
                    agent=agent,
                    evaluator=evaluator,
                    graph=graph,
                    max_steps=max_steps,
                    callback_context={**task.misc}
                )

                traj.misc["needs_prefix"] = False

                node.add_trajectory(traj)

            except Exception as e:
                logger.error(f"Error sampling trajectories for node {node.url} and task {task.goal}: {e}")
                logger.error(traceback.format_exc())

def process_open_urls_callback(
    agent: AgentWithExplorationCallbacks, step_num: int, goal: str, env: BrowserEnv, graph: Graph, node: Node, traj: Trajectory, obs: dict,
        reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict
):
    """
    Callback to process the open urls after each step.
    """
    open_urls = obs['open_pages_urls']

    for url in open_urls:
        curr_prefix = Trace.from_trajectory_steps(
            steps=traj.steps,
            start_url=node.url,
            end_url=url,
            misc={'agent_info': agent.get_config(), 'goal': goal, 'task_misc': callback_context.get('task_misc', {})}
        )

        if graph.check_if_url_allowed(url):

            update_prefix = url != node.url # No self-edges

            url_node = graph.get_node(url)
            if url_node:
                if update_prefix:
                    url_node.add_prefix(curr_prefix)
            else:
                graph.add_url(
                    url=url,
                    parent=node,
                    prefixes=[curr_prefix] if update_prefix else [],
                    node_misc={'discovered_by': agent.get_config(), 'goal': goal, 'task_misc': callback_context.get('task_misc', {})}
                )
            
            if url not in node.children:
                node.children.append(url)
                node.update_save(save_prefix=False, save_info=True)
    
    return step_num, obs, reward, terminated, truncated, env_info, goal, callback_context


def web_explore_loop():

    parser = argparse.ArgumentParser(description="Run an episode with a browser gym agent.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config: WebExploreConfig = oc.load(args.config)
    oc.resolve(config)
    config_dict = oc.to_container(config)

    logger.info(f"WebExploreConfig:\n{config}")

    os.makedirs(config.exp_dir, exist_ok=True)

    page_explorers = [
        wrap_agent_for_callback_protocol(
            AgentFactory.create_agent(**explorer['agent_factory_args']),
            pre_step_callbacks=[prestep_store_url, ],
            post_step_callbacks=[backtrack_if_needed, process_open_urls_callback],
        )
        for explorer in config_dict['page_explorers']
    ]

    nav_explorers = [
        wrap_agent_for_callback_protocol(
            AgentFactory.create_agent(**explorer['agent_factory_args']),
            pre_step_callbacks=[prestep_store_url,],
            post_step_callbacks=[backtrack_if_needed, process_open_urls_callback, backtrack_when_new_page_found],
        )
        for explorer in config_dict['nav_explorers']
    ]

    feasibility_checkers = [
        wrap_agent_for_callback_protocol(
            AgentFactory.create_agent(**feasibility_checker['agent_factory_args']),
            pre_step_callbacks=[prestep_store_url],
            post_step_callbacks=[backtrack_if_needed, process_open_urls_callback],
        )
        for feasibility_checker in config_dict['feasibility_checkers']
    ]

    solvers = [
        wrap_agent_for_callback_protocol(
            AgentFactory.create_agent(**solver['agent_factory_args']),
            pre_step_callbacks=[prestep_store_url,],
            post_step_callbacks=[backtrack_if_needed, process_open_urls_callback],
        )
        for solver in config_dict['solvers']
    ]

    env: BrowserEnv = EnvArgs(**config_dict['env_args']).make_env(
        action_mapping=lambda x: x,
        exp_dir=config.exp_dir
    )
    env = env.unwrapped
    env.reset()
    root_url = env.page.url

    evaluator = Evaluator(**config.evaluator)

    if config.resume_from:
        graph = Graph.load(os.path.join(config.resume_from, "graph"), load_images=False)
    else:
        graph = Graph(
            root_url=root_url,
            exp_dir=config.exp_dir,
            denylist_patterns=config_dict['denylist_patterns'], 
            allowlist_patterns=config_dict['allowlist_patterns']
        )
    
    try:
        curr_node = graph.get_next_node()
        exploration_count = len(graph.explored_nodes)

        while curr_node and exploration_count < config.max_nodes:
            
            logger.info(f"Exploring node {curr_node.url} ...")

            if hasattr(config, 'full_reset_url') and config.full_reset_url:
                logger.info(f"Performing full env reset with url: {config.full_reset_url}")
                perform_full_reset(config.full_reset_url)

            if not len(curr_node.tasks):
                
                page_explorer_tasks = []
                for i, page_explorer in enumerate(page_explorers):
                    page_explorer_tasks.extend(sample_task_candidates_for_node(
                        env=env,
                        explorer=page_explorer,
                        evaluator=evaluator,
                        graph=graph,
                        node=curr_node,
                        max_steps=config.page_explorers[i].max_steps,
                        max_retries=config.page_explorers[i].retries,
                    ))

                nav_explorer_tasks = []
                for i, nav_explorer in enumerate(nav_explorers):
                    nav_explorer_tasks.extend(sample_task_candidates_for_node(
                        env=env,
                        explorer=nav_explorer,
                        evaluator=evaluator,
                        graph=graph,
                        node=curr_node,
                        max_steps=config.nav_explorers[i].max_steps,
                        max_retries=config.nav_explorers[i].retries,
                    ))


            for i, feasibility_checker in enumerate(feasibility_checkers):
                filter_to_feasible_tasks_for_node(
                    tasks=page_explorer_tasks,
                    env=env,
                    feasibility_checker=feasibility_checker,
                    evaluator=evaluator,
                    graph=graph,
                    node=curr_node,
                    max_steps=config.feasibility_checkers[i].max_steps,
                    max_retries=config.feasibility_checkers[i].retries,
                    max_feasible_tasks=config.max_feasible_page_explorer_tasks_per_node
                )

                filter_to_feasible_tasks_for_node(
                    tasks=nav_explorer_tasks,
                    env=env,
                    feasibility_checker=feasibility_checker,
                    evaluator=evaluator,
                    graph=graph,
                    node=curr_node,
                    max_steps=config.feasibility_checkers[i].max_steps,
                    max_retries=config.feasibility_checkers[i].retries,
                    max_feasible_tasks=config.max_feasible_nav_explorer_tasks_per_node
                )
            
            for i, solver in enumerate(solvers):
                sample_task_solving_trajectories_for_node(
                    node=curr_node,
                    env=env,
                    agent=solver,
                    evaluator=evaluator,
                    graph=graph,
                    max_steps=config.solvers[i].max_steps,
                    num_trajs_per_task=config.solvers[i].retries
                )  

            graph.add_to_explored(curr_node)
            exploration_count += 1
            curr_node = graph.get_next_node()

            if exploration_count == config.max_nodes:
                logger.info(f"Max nodes to explore reached: {config.max_nodes}")
            else:
                logger.info(f"We will now explore the next node: {curr_node.url if curr_node else 'No nodes left to explore!'}")

    except Exception as e:
        logger.error(f"Error during exploration: {e}")
        logger.error(traceback.format_exc())
        raise e

    finally:
        env.close()

if __name__ == "__main__":
    web_explore_loop()
