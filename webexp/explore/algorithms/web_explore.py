from ..core.agent import AgentWithExplorationCallbacks, ExplorerAgentWithExplorationCallbacks, wrap_agent_for_callback_protocol
from ..core.evaluator import Evaluator
from ..core.episode import run_episode, get_action, perform_env_step, get_fresh_obs
from ..core.graph import Graph
from ..core.node import Node
from ..core.task import Task
from ..core.trace import Trace
from ..core.trajectory import Trajectory

from ...agents.base_agent import AgentFactory
from ...agents.task_summarization_agent import TaskSummarizationAgent
from ...agents.semantic_verifier_agent import SemanticVerifierAgent
from ...agents.captcha_detection_agent import CaptchaDetectionAgent
from browsergym.core.env import BrowserEnv
from browsergym.experiments.loop import EnvArgs
from dataclasses import dataclass
from dotenv import load_dotenv
from omegaconf import OmegaConf as oc
from pathlib import Path
from typing import Sequence, List, Dict, Optional
import argparse
import logging
import os
import random
import requests
import sys
import traceback

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Track per-node file handlers so we can cleanly remove them between nodes
_node_log_handlers: dict[str, logging.Handler] = {}


def _graph_dir_for_resume(path: str) -> str:
    """Return a graph directory from either an exp_dir or graph dir path."""
    if os.path.isfile(os.path.join(path, "graph_info.json")):
        return path
    return os.path.join(path, "graph")


def _has_saved_graph(exp_dir: str) -> bool:
    return os.path.isfile(os.path.join(exp_dir, "graph", "graph_info.json"))


def _redact_secrets(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(secret_key in str(key).lower() for secret_key in ("api_key", "apikey", "token", "secret")):
                redacted[key] = "<REDACTED>"
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value

def _setup_node_logging(node: Node):
    """Add a file handler that writes all logs to a per-node file.

    The log file is saved at: <node.exp_dir>/explore.log
    Captures logs from ALL modules (web_explore, episode, solver_agent, etc.)
    at INFO level and above.
    """
    global _node_log_handlers

    # Remove previous node's file handler if any
    for existing_handler in _node_log_handlers.values():
        existing_handler.close()
        logging.getLogger().removeHandler(existing_handler)
    _node_log_handlers.clear()

    log_path = os.path.join(node.exp_dir, "explore.log")
    os.makedirs(node.exp_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Attach to the root logger so ALL module logs are captured
    logging.getLogger().addHandler(file_handler)
    _node_log_handlers[node.exp_dir] = file_handler

    logger.info(f"Per-node logging enabled → {log_path}")

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
        frontier_alpha (float): Frontier scoring addition; weight for Uncertainty (U).
        frontier_beta (float): Frontier scoring addition; weight for Value (V).
        frontier_theta (float): Frontier scoring addition; weight for Diversity (D).
        captcha_detection_enabled (bool): Whether to run vision CAPTCHA checks before exploring nodes.
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
    # Frontier scoring addition: Graph uses these weights when ranking the frontier.
    frontier_alpha: float = 1.0
    frontier_beta: float = 1.0
    frontier_theta: float = 1.0
    lookahead_model: Optional[str] = None
    captcha_detection_enabled: bool = True


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


    task_misc = {'agent_info': explorer.get_config()}
    return node.add_tasks(tasks, task_misc=task_misc)


def _tasks_created_by_agent(node: Node, agent_name: str) -> list[Task]:
    tasks = []
    for task in node.tasks.values():
        agent_info = (task.misc or {}).get("agent_info", {})
        if agent_info.get("name") == agent_name:
            tasks.append(task)
    return tasks


def _pending_feasibility_tasks(tasks: list[Task]) -> list[Task]:
    """Only feasibility-check tasks that have no saved outcome yet."""
    return [
        task for task in tasks
        if not task.positive_trajs and not task.negative_trajs
    ]


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

                # Frontier scoring addition: update node SR for future child V.
                node.record_trajectory_outcome(traj.success)

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

        existing_prefixed = sum(
            1 for traj in [*task.positive_trajs, *task.negative_trajs]
            if traj.misc and traj.misc.get("needs_prefix") is True
        )
        for _ in range(max(num_trajs_per_task - existing_prefixed, 0)):

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
                # Frontier scoring addition: update node SR for future child V.
                node.record_trajectory_outcome(traj.success)

            except Exception as e:
                logger.error(f"Error sampling trajectories for node {node.url} and task {task.goal}: {e}")
                logger.error(traceback.format_exc())


        existing_unprefixed = sum(
            1 for traj in [*task.positive_trajs, *task.negative_trajs]
            if traj.misc and traj.misc.get("needs_prefix") is False
        )
        for _ in range(max(num_trajs_per_task - existing_unprefixed, 0)):

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
                # Frontier scoring addition: update node SR for future child V.
                node.record_trajectory_outcome(traj.success)

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


def _summarize_node_trajectories(node: Node, model_name: str):
    """Summarize trajectories for all feasible tasks on a node.

    Uses TaskSummarizationAgent to produce clean task descriptions from
    the full action history and screenshots of successful trajectories.
    """
    summarizer = TaskSummarizationAgent(model_name=model_name)
    verifier = SemanticVerifierAgent(model_name=model_name)
    feasible_tasks = node.get_feasible_tasks()

    for task in feasible_tasks:
        for traj in task.positive_trajs:
            if traj.misc is None:
                traj.misc = {}
            if not traj.misc.get("semantic_summary"):
                summary = summarizer.summarize_details(traj)
                if summary:
                    traj.misc["semantic_summary"] = summary.to_dict()
                    traj.misc["summarized_goal"] = summary.accomplished_goal
                    logger.info(f"Summarized goal for task '{task.goal[:100]}': {summary.accomplished_goal[:200]}")
            if traj.misc.get("semantic_summary") and not traj.misc.get("semantic_verification"):
                accomplished_goal = traj.misc["semantic_summary"].get("accomplished_goal")
                verification = verifier.verify(traj, accomplished_goal=accomplished_goal)
                if verification:
                    traj.misc["semantic_verification"] = verification
            if traj.misc.get("semantic_summary") or traj.misc.get("semantic_verification"):
                traj.save_info()

        for traj in task.negative_trajs:
            if traj.misc is None:
                traj.misc = {}
            if not traj.misc.get("semantic_summary"):
                summary = summarizer.summarize_details(traj)
                if summary:
                    traj.misc["semantic_summary"] = summary.to_dict()
                    traj.misc["summarized_goal"] = summary.accomplished_goal
            if traj.misc.get("semantic_summary") and not traj.misc.get("semantic_verification"):
                accomplished_goal = traj.misc["semantic_summary"].get("accomplished_goal")
                verification = verifier.verify(traj, accomplished_goal=accomplished_goal)
                if verification:
                    traj.misc["semantic_verification"] = verification
            if traj.misc.get("semantic_summary") or traj.misc.get("semantic_verification"):
                traj.save_info()

        best_summary = _select_best_summarized_goal(task)
        if best_summary:
            if task.misc is None:
                task.misc = {}
            task.misc["semantic_summary_selected"] = best_summary
            task.update_summarized_goal(best_summary)
            logger.info(f"Task-level summarized goal saved for '{task.goal[:100]}': {best_summary[:200]}")


def _select_best_summarized_goal(task: Task) -> str | None:
    """Prefer verified successful trajectory summaries with clear evidence."""
    def candidate_for(traj):
        if not traj.misc:
            return None
        summary = traj.misc.get("semantic_summary") or {}
        accomplished_goal = summary.get("accomplished_goal") or traj.misc.get("summarized_goal")
        if not accomplished_goal:
            return None
        verification = traj.misc.get("semantic_verification") or {}
        verified = bool(
            verification.get("is_aligned")
            and verification.get("is_grounded")
            and verification.get("is_complete")
        )
        evidence = bool(summary.get("completion_evidence") or verification.get("evidence"))
        differs = accomplished_goal.strip() != task.goal.strip()
        return {
            "goal": accomplished_goal,
            "verified": verified,
            "evidence": evidence,
            "differs": differs,
            "success": bool(traj.success),
            "length": len(accomplished_goal),
        }

    candidates = [
        candidate
        for candidate in [candidate_for(traj) for traj in [*task.positive_trajs, *task.negative_trajs]]
        if candidate
    ]
    if candidates:
        best = max(
            candidates,
            key=lambda c: (c["success"], c["verified"], c["evidence"], c["differs"], c["length"]),
        )
        return best["goal"]
    return None


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

    logger.info(f"WebExploreConfig:\n{_redact_secrets(config_dict)}")

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
    # Add extra HTTP headers to bypass ngrok warning page
    env.pw_context_kwargs.setdefault("extra_http_headers", {}).update({
        "ngrok-skip-browser-warning": "true",
    })
    env.reset()
    root_url = env.page.url

    evaluator = Evaluator(**config.evaluator)
    captcha_detection_enabled = getattr(config, "captcha_detection_enabled", True)
    captcha_detector = (
        CaptchaDetectionAgent(model_name=config.evaluator.get("model_name", "gpt-4o-2024-05-13"))
        if captcha_detection_enabled
        else None
    )
    if not captcha_detection_enabled:
        logger.info("CAPTCHA detection disabled by config.")

    if config.resume_from:
        graph_dir = _graph_dir_for_resume(config.resume_from)
        logger.info(f"Resuming exploration from configured graph: {graph_dir}")
        graph = Graph.load(
            graph_dir,
            load_images=False,
            # Frontier scoring addition: restore lookahead behavior while scorer weights come from graph_info.
            lookahead_model=getattr(config, 'lookahead_model', None),
        )
    elif _has_saved_graph(config.exp_dir):
        graph_dir = _graph_dir_for_resume(config.exp_dir)
        logger.info(f"Found existing graph in exp_dir; auto-resuming from: {graph_dir}")
        graph = Graph.load(
            graph_dir,
            load_images=False,
            # Frontier scoring addition: restore lookahead behavior while scorer weights come from graph_info.
            lookahead_model=getattr(config, 'lookahead_model', None),
        )
    else:
        graph = Graph(
            root_url=root_url,
            exp_dir=config.exp_dir,
            denylist_patterns=config_dict['denylist_patterns'], 
            allowlist_patterns=config_dict['allowlist_patterns'],
            # Frontier scoring addition: pass ranking weights and lookahead model.
            alpha=config.frontier_alpha,
            beta=config.frontier_beta,
            theta=config.frontier_theta,
            lookahead_model=getattr(config, 'lookahead_model', None),
        )
    
    try:
        curr_node = graph.get_next_node()
        exploration_count = len(graph.explored_nodes)

        while curr_node and exploration_count < config.max_nodes:
            
            logger.info(f"Exploring node {curr_node.url} ...")

            # Set up per-node log file to capture all logs for this node
            _setup_node_logging(curr_node)

            # Phase 7: Check for CAPTCHA before exploring
            if captcha_detector is not None:
                obs = get_fresh_obs(env)
                if "screenshot" in obs and captcha_detector.is_captcha(obs["screenshot"]):
                    logger.warning(f"CAPTCHA detected at {curr_node.url}, skipping node.")
                    graph.add_to_explored(curr_node)
                    exploration_count += 1
                    curr_node = graph.get_next_node()
                    continue

            if hasattr(config, 'full_reset_url') and config.full_reset_url:
                logger.info(f"Performing full env reset with url: {config.full_reset_url}")
                perform_full_reset(config.full_reset_url)

            page_explorer_tasks = _tasks_created_by_agent(curr_node, "PageExplorerAgent")
            nav_explorer_tasks = _tasks_created_by_agent(curr_node, "NavExplorerAgent")
            if curr_node.tasks and not page_explorer_tasks and not nav_explorer_tasks:
                logger.info(
                    "Loaded tasks without explorer source metadata; reusing them as page explorer tasks."
                )
                page_explorer_tasks = list(curr_node.tasks.values())

            if not page_explorer_tasks:
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

            if not nav_explorer_tasks:
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

            page_explorer_tasks = _pending_feasibility_tasks(page_explorer_tasks)
            nav_explorer_tasks = _pending_feasibility_tasks(nav_explorer_tasks)

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

            # Phase 3: Summarize trajectories to produce clean semantic task descriptions
            _summarize_node_trajectories(curr_node, evaluator.model_name)

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
