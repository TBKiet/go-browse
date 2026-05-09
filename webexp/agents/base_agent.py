from typing import Protocol, runtime_checkable

@runtime_checkable
class Agent(Protocol):
    """
    Protocol for an agent.
    """
    
    def get_action(self, obs: dict, oracle_action: tuple[str, str]=None, **kwargs) -> tuple[str, dict]:
        """
        Get the action for the given observation.

        Args:
            obs (dict): The observation from the environment.
            oracle_action (tuple[str, str], optional): Tuple of (action, reason) to use if available instead of generating a new one.

        Returns:
            str: The action to take.
        """
        ...
    
    def reset(self):
        """
        Reset the agent's state.
        """
        ...
    
    def get_config(self) -> dict:
        """
        Get the agent's configuration.

        Returns:
            dict: The agent's configuration.
        """
        ...
    
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess the observation before it is passed to get_action.

        Args:
            obs (dict): The observation from the environment.

        Returns:
            dict: The preprocessed observation.
        """
        ...
    
    def action_processor(self, action: str) -> str:
        """
        Process the action before it is passed to the environment.

        Args:
            action (str): The action to process.

        Returns:
            str: The processed action.
        """
        ...

@runtime_checkable
class ExplorerAgent(Agent, Protocol):
    """
    Agent used to propose and collect exploration tasks.
    """
    
    def get_proposed_tasks(self) -> list[str]:
        ...

    @property
    def goal_str(self) -> str:
        """
        Get the exploration goal/task string for the agent.

        Returns:
            str: The goal string.
        """
        ...


class BaseAgent:
    def __init__(self, **kwargs):
        """
        Initialize the agent.

        Args:
            *args: Positional arguments for the agent.
            **kwargs: Keyword arguments for the agent.
        """
        self.config = {
            "name": self.__class__.__name__,
            **kwargs
        }

    def reset(self):
        """
        Reset the agent's state.
        """
        raise NotImplementedError

    def get_action(self, obs: dict, oracle_action: tuple[str, str]=None, **kwargs) -> tuple[str, dict]:
        """
        Get the action for the given observation.

        Args:
            obs (dict): The observation from the environment.
            oracle_action (str, optional): Tuple of (action, thought) to use if available instead of generating a new one.

        Returns:
            str: The action to take.
        """
        raise NotImplementedError
    
    def get_config(self) -> dict:
        """
        Get the agent's configuration.

        Returns:
            dict: The agent's configuration.
        """
        return self.config
    
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess the observation before it is passed to get_action.

        Args:
            obs (dict): The observation from the environment.

        Returns:
            dict: The preprocessed observation.
        """
        return obs
    
    def action_processor(self, action: str) -> str:
        """
        Process the action before it is passed to the environment.

        Args:
            action (str): The action to process.

        Returns:
            str: The processed action.
        """
        return action
    
    @classmethod
    def create_agent(cls, *args, **kwargs):
        """
        Create an agent instance.

        Args:
            *args: Positional arguments for the agent.
            **kwargs: Keyword arguments for the agent.

        Returns:
            BaseAgent: An instance of a class derived from BaseAgent.
        """
        return cls(*args, **kwargs)
    

AGENT_FACTORY_REGISTRY: dict[str, BaseAgent] = {}

class AgentFactory:
    
    @staticmethod
    def create_agent(name: str, *args, **kwargs):
        """
        Create an agent instance.

        Args:
            *args: Positional arguments for the agent.
            **kwargs: Keyword arguments for the agent.

        Returns:
            BaseAgent: An instance of a class derived from BaseAgent.
        """
        if name not in AGENT_FACTORY_REGISTRY:
            raise ValueError(f"Unknown agent: {name}")
        return AGENT_FACTORY_REGISTRY[name].create_agent(*args, **kwargs)

    
    @staticmethod
    def register(cls, aliases: str | tuple[str] = tuple()):
        """
        Register an agent class.

        Args:
            cls (type): The agent class to register.

        Returns:
            type: The agent class that was registered.
        """
        AGENT_FACTORY_REGISTRY[cls.__name__] = cls
        
        if isinstance(aliases, str):
            aliases = (aliases,)
        
        for name in aliases:
            AGENT_FACTORY_REGISTRY[name] = cls
        
        return cls
