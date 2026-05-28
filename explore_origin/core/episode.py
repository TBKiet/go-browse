from .agent import AgentWithExplorationCallbacks, ExplorerAgentWithExplorationCallbacks, wrap_agent_for_callback_protocol
from .evaluator import Evaluator
from .graph import Graph
from .node import Node
from .task import Task
from .trajectory import Trajectory, TrajectoryStep
from ...agents.base_agent import Agent, ExplorerAgent
from browsergym.core.env import BrowserEnv
from browsergym.experiments.loop import _send_chat_info
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=20),
)
def reset_env_to_node(
        env: BrowserEnv,
        node: Node,
        agent: Agent,
        goal: str,
):
    """
    Reset the environment to a given node.
    
    Args:
        env: The environment to reset.
        node (Node): The node to reset to.
        agent (BaseAgent): The agent to reset.
        goal (str): The goal for the episode.
    """
    
    logger.info(f"Resetting environment to node: {node.url}...")
    
    env.reset()
    nav_action = f"goto('{node.url}')"
    env.action_mapping = agent.action_processor
    env.goal_object = [{"type": "text", "text": goal}]
    env.chat.add_message(role="user", msg=goal)
    obs, _, _, _, _ = env.step(nav_action)
    return agent.obs_preprocessor(obs)


def get_fresh_obs(env: BrowserEnv):
    """
    Get a fresh observation from the environment.
    
    Args:
        env: The environment to get the observation from.
    
    Returns:
        dict: The observation from the environment.
    """
    # TODO: We can make an ExplorationBrowserEnv that has a more reliable api.
    obs = env._get_obs()
    return obs

# TODO: Can move this into a ExplorationBrowserEnv class as part of the is_done logic.
def has_new_assistant_message(env: BrowserEnv):
    """
    Check if there is a new assistant message in the environment.
    
    Args:
        env: The environment to check.
    
    Returns:
        bool: True if there is a new assistant message, False otherwise.
    """
    chat_messages = env.chat.messages
    if chat_messages[-1]["role"] == "assistant" or chat_messages[-1]["role"] == "infeasible":
        return True
    return False

def get_action(
    env: BrowserEnv,
    agent: Agent,
    obs: dict,
    traj: Trajectory,
    oracle_action: str = None
) -> tuple[str, dict]:
    """
    Get the action from the agent.
    
    Args:
        env: The environment to get the action from.
        agent (BaseAgent): The agent to get the action from.
        obs (dict): The observation from the environment.
        traj (Trajectory): The trajectory of the episode.
        oracle_action (str, optional): The oracle action to use if available.
    
    Returns:
        tuple: The action and action extras dict from the agent.
    """
    action, action_extras = agent.get_action(obs, oracle_action=oracle_action)
    thought = action_extras.get("thought", None)
    parsed_action = action_extras.get("parsed_action", None)

    if thought and "think" not in action_extras:
        action_extras["think"] = thought

    logger.info(f"Agent chose action: \n{action}")
    
    traj.add_step(action, parsed_action, thought, obs, {'model_usage': action_extras.get("model_usage", None), 'agent_config': agent.get_config()})

    # TODO: Need a more stable api for modifying the chat pane. Perhaps we can create an env wrapper that exposes such as an api.
    _send_chat_info(env.chat, action, action_extras)
    
    return action, action_extras


def perform_env_step(
    env: BrowserEnv,
    agent: Agent,
    action: str,
) -> tuple:
    """
    Perform a step in the environment.
    
    Args:
        env: The environment to perform the step in.
        agent (BaseAgent): The agent to perform the step with.
        action (str): The action to perform.
        traj (Trajectory): The trajectory of the episode.
        oracle_action (str, optional): The oracle action to use if available.
    
    Returns:
        tuple: The observation, reward, terminated, truncated, and env_info from the environment.
    """
    obs, reward, terminated, truncated, env_info = env.step(action)
    obs = agent.obs_preprocessor(obs)
    return obs, reward, terminated, truncated, env_info



def run_episode(
    goal: str,
    node: Node,
    env: BrowserEnv,
    agent: AgentWithExplorationCallbacks,
    evaluator: Evaluator,
    graph: Graph,
    max_steps: int,
    callback_context: dict = None,
) -> Trajectory:
    """
    Run an episode with an agent in the environment.
    
    Args:
        goal (str): The goal for the episode.
        node (Node): The current node in the graph.
        env: The environment.
        agent (BaseAgent): The agent to run the episode.
        evaluator (Evaluator): The evaluator to evaluate the episode.
        graph (Graph): The graph of nodes.
        max_steps (int): The maximum number of steps in the episode.
    
    Returns:
    
        Trajectory: The trajectory of the episode.
    """

    logger.info(f"Running episode for goal: {goal}, for node {node.url}...")

    obs = reset_env_to_node(
        env=env,
        node=node,
        agent=agent,
        goal=goal,
    )
    
    agent.reset()

    traj = Trajectory.from_goal(goal, agent.get_config())
    
    num_steps = 0
    done = False

    callback_context_seed = callback_context if callback_context else {}

    while not done and num_steps < max_steps:
        
        logger.info(f"Step {num_steps} for goal {goal}.")
        num_steps += 1

        callback_context = {**callback_context_seed}

        num_steps, obs, reward, terminated, truncated, env_info, goal, callback_context = agent.run_pre_step_callbacks(
            num_steps, goal, env, graph, node, traj, obs, 0.0, False, False, {}, callback_context
        )

        action, action_extras = get_action(
            env=env,
            agent=agent,
            obs=obs,
            traj=traj
        )

        if action is None:
            logger.info("Agent returned None action. Ending episode.")
            break

        obs, reward, terminated, truncated, env_info = perform_env_step(
            env=env,
            agent=agent,
            action=action,
        )
        
        if has_new_assistant_message(env):
            logger.info("New assistant message received.")
            terminated = True

            # We only need to evaluate if the when we are not exploring.
            if not isinstance(agent, ExplorerAgent):
                    logger.info("Evaluating episode...")
                    evaluator.evaluate(traj)
                    logger.info(f"Episode evaluated and received reward {traj.reward}.")

        num_steps, obs, reward, terminated, truncated, env_info, goal, callback_context = agent.run_post_step_callbacks(
            num_steps, goal, env, graph, node, traj, obs, reward, terminated, truncated, env_info, callback_context
        )

        done = terminated or truncated

    traj.extract_response(env)
    traj.final_state = TrajectoryStep(
        action=None,
        parsed_action=None,
        thought=None,
        observation=obs,
        misc=None,
    )

    return traj
