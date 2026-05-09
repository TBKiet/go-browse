from .base_agent import AgentFactory
from browsergym.experiments import AbstractAgentArgs, EnvArgs, ExpArgs, get_exp_result
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
import argparse

@dataclass
class RunEpisodeConfig:
    """
    Configuration for running an agent for an episode.
    
    Attributes:
        agent_factory_args (dict): Arguments for the agent factory.
        env_args (dict): Arguments for the environment.
        exp_dir (str): Directory for storing experiment results. Default is "./results".
    """
    agent_factory_args: dict
    env_args: dict
    exp_dir: str


class BrowserGymAgentArgsWrapper(AbstractAgentArgs):
    def __init__(self, agent_factory_args: dict):
        super().__init__()
        self.agent_factory_args = agent_factory_args

    def make_agent(self):
        return AgentFactory.create_agent(**self.agent_factory_args)

def main():
    
    # Need to get config file from command line
    parser = argparse.ArgumentParser(description="Run an episode with a browser gym agent.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config: RunEpisodeConfig = oc.load(args.config)
    oc.resolve(config)
    config_dict = oc.to_container(config)

    agent_args = BrowserGymAgentArgsWrapper(config.agent_factory_args)
    env_args = EnvArgs(**config_dict['env_args'])

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    exp_args.prepare(config.exp_dir)
    exp_args.run()

    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
