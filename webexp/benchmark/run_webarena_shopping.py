"""
Custom benchmark runner that filters WebArena tasks to a specific domain only.
Usage:
    python -m webexp.benchmark.run_webarena_shopping -c configs/benchmark_qwen.yaml

Environment variables required:
    WA_SHOPPING=<your_local_shopping_url>   (e.g., http://localhost:7770)
"""
from agentlab.experiments.study import make_study, AgentArgs, Study
from ..agents.base_agent import AgentFactory
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
import argparse
import json
import importlib.resources
import webarena
import os


# Shopping task IDs extracted from webarena test.raw.json
def get_shopping_task_ids():
    """Return sorted list of task IDs that belong to the Shopping domain."""
    all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
    all_configs = json.loads(all_configs_str)
    shopping_ids = sorted(set(
        conf["task_id"] for conf in all_configs
        if "__SHOPPING__" in json.dumps(conf) and "__SHOPPING_ADMIN__" not in json.dumps(conf)
    ))
    return shopping_ids


@dataclass
class RunBenchmarkConfig:
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
    # Validate env vars
    if not os.environ.get("WA_SHOPPING"):
        print("WARNING: WA_SHOPPING env var not set! Set it to your local shopping URL.")
        print("Example: export WA_SHOPPING=http://localhost:7770")
        print("Continuing anyway...")

    parser = argparse.ArgumentParser(description="Run webarena benchmark - shopping only.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()

    config: RunBenchmarkConfig = oc.load(args.config)
    oc.resolve(config)
    config_dict = oc.to_container(config)

    agent_args = AgentLabAgentArgsWrapper(config_dict['agent_factory_args'])

    shopping_task_ids = get_shopping_task_ids()
    print(f"Found {len(shopping_task_ids)} shopping task IDs: {shopping_task_ids[:10]}...")

    # Build the list of browsergym task names for shopping
    shopping_task_names = [f"webarena.{tid}" for tid in shopping_task_ids]

    if config.resume_dir is None:
        study = make_study(
            benchmark="webarena",
            agent_args=[agent_args],
            comment="WebArena - shopping only",
        )
        # Filter study to only shopping tasks by task_name
        before = len(study.study_df) if hasattr(study, 'study_df') else '?'
        try:
            mask = study.study_df['task_name'].isin(shopping_task_names)
            study.study_df = study.study_df[mask].copy()
            after = len(study.study_df)
            print(f"Filtered study: {before} → {after} shopping tasks")
        except Exception as e:
            print(f"Could not filter study dataframe: {e}")
            print(f"Will run all tasks (shopping tasks have URLs in WA_SHOPPING)")
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
