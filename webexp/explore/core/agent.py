from __future__ import annotations
from .graph import Graph
from .node import Node
from .trajectory import Trajectory
from ...agents.base_agent import Agent, ExplorerAgent
from browsergym.core.env import BrowserEnv
from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentWithExplorationCallbacks(Agent, Protocol):
    """
    Protocol for an agent that supports exploration callbacks.
    """

    def register_pre_step_callbacks(self, callbacks: list[callable]) -> AgentWithExplorationCallbacks:
        """Register a callback to be called before each step."""
        ...
        
    def register_post_step_callbacks(self, callbacks: list[callable]) -> AgentWithExplorationCallbacks:
        """Register a callback to be called after each step."""
        ...
    
    def run_pre_step_callbacks(self, step_num: int, goal: str, env: BrowserEnv, graph: Graph, traj: Trajectory, obs: dict, reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict) -> tuple:
        """Run all registered pre-step callbacks and return potentially modified versions of the inputs."""
        ...
    
    def run_post_step_callbacks(self, step_num: int, goal: str, env: BrowserEnv, graph: Graph, traj: Trajectory, obs: dict, reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict) -> tuple:
        """Run all registered post-step callbacks and return potentially modified versions of the inputs."""
        ...


def wrap_agent_for_callback_protocol(agent: Agent, pre_step_callbacks: list[callable]= None, post_step_callbacks: list[callable]=None) -> AgentWithExplorationCallbacks:
    """
    Wrap an agent to implement the AgentWithExplorationCallbacks protocol.
    """
    if isinstance(agent, AgentWithExplorationCallbacks):
        return agent
    
    class CallbackProtocolAgentWrapper(type(agent)):
        def __init__(self, agent):
            self._agent = agent
            self._pre_step_callbacks = []
            self._post_step_callbacks = []
            
        def register_pre_step_callbacks(self, callbacks):
            self._pre_step_callbacks.extend(callbacks)
            return self
            
        def register_post_step_callbacks(self, callbacks):
            self._post_step_callbacks.extend(callbacks)
            return self

        def run_pre_step_callbacks(self, step_num: int, goal: str, env: BrowserEnv, graph: Graph, node: Node, traj: Trajectory, obs: dict, reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict) -> tuple:
            for callback in self._pre_step_callbacks:
                step_num, obs, reward, terminated, truncated, env_info, goal, callback_context = callback(self, step_num, goal, env, graph, node, traj, obs, reward, terminated, truncated, env_info, callback_context)
            return step_num, obs, reward, terminated, truncated, env_info, goal, callback_context

        def run_post_step_callbacks(self, step_num: int, goal: str, env: BrowserEnv, graph: Graph, node: Node, traj: Trajectory, obs: dict, reward: float, terminated: bool, truncated: bool, env_info: dict, callback_context: dict) -> tuple:
            for callback in self._post_step_callbacks:
                step_num, obs, reward, terminated, truncated, env_info, goal, callback_context = callback(self, step_num, goal, env, graph, node, traj, obs, reward, terminated, truncated, env_info, callback_context)
            return step_num, obs, reward, terminated, truncated, env_info, goal, callback_context

        def __getattr__(self, name):
            return getattr(self._agent, name)
    
    wrapped_agent = CallbackProtocolAgentWrapper(agent)

    if pre_step_callbacks:
        wrapped_agent.register_pre_step_callbacks(pre_step_callbacks)
    
    if post_step_callbacks:
        wrapped_agent.register_post_step_callbacks(post_step_callbacks)

    return wrapped_agent

class ExplorerAgentWithExplorationCallbacks(ExplorerAgent, AgentWithExplorationCallbacks, Protocol):
    """
    Intersection type for an ExplorerAgent that supports exploration callbacks.
    """
    ...