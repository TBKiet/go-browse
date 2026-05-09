from agentlab.experiments.study import make_study, AgentArgs, Study
from ..agents.base_agent import AgentFactory
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
import argparse
import os

@dataclass
class RunBenchmarkConfig:
    """
    Configuration for running an agent for an episode.
    
    Attributes:
        agent_factory_args (dict): Arguments for the agent factory.
        exp_dir (str): Directory to save the experiment results.
        n_jobs (int): Number of workers to use.
    """
    agent_factory_args: dict
    exp_dir: str
    n_jobs: int
    resume_dir: str | None = None


class AgentLabAgentArgsWrapper(AgentArgs):
    def __init__(self, agent_factory_args: dict):
        super().__init__()
        self.agent_factory_args = agent_factory_args

    def make_agent(self):
        return AgentFactory.create_agent(**self.agent_factory_args)


def run():

    parser = argparse.ArgumentParser(description="Run webarena benchmark.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config: RunBenchmarkConfig = oc.load(args.config)
    oc.resolve(config)
    config_dict = oc.to_container(config)

    agent_args = AgentLabAgentArgsWrapper(config_dict['agent_factory_args'])
    
    if config.resume_dir is None:
    
        study = make_study(
            benchmark="webarena",
            agent_args=[agent_args],
            comment="WebArena benchmark run",

        )
    else:
        study = Study.load(config.resume_dir)
        study.find_incomplete(include_errors=True)

    study.run(
        n_jobs=config.n_jobs,
        exp_root=config.exp_dir,
        n_relaunch=8
    )
    

if __name__ == "__main__":
    run()
