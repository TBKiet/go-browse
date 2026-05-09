from .base_agent import AgentFactory
from .solver_agent import SolverAgent
from .prompt_builders.nav_explorer_prompt_builder import NavExplorerPromptBuilder
from browsergym.core.action.highlevel import HighLevelActionSet
from textwrap import dedent

TASK_COLLECTOR = []

def add_tasks_to_dataset(*tasks: str):
    """Given one or more navigation task strings, add them to the dataset we are collecting.
    You should add tags to the start of the task string to indicate the type of task it is.
    Since your job is to find navigation tasks, you should use the [NAV] tag to indicate that the task is a navigation task.
    
    Examples:
        add_tasks_to_dataset('[NAV] Navigate to the Recommendations page.', '[NAV] Visit the Home page.')
        add_tasks_to_dataset('[NAV] Take me to my cart.')
    """
    TASK_COLLECTOR.extend(tasks)


@AgentFactory.register
class NavExplorerAgent(SolverAgent):
    """
    Agent used to propose exploration tasks for a web page.
    """
    
    def __init__(
            self,
            model_id: str,
            base_url: str | None = None,
            api_key: str | None = None,
            temperature: float = 1.0,
            char_limit: int = -1,
            demo_mode: str = 'off',
    ):
        """
        Initialize the agent.
        """
        super().__init__(model_id=model_id, base_url=base_url, api_key=api_key, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode)

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav", "tab", "custom"],
            custom_actions=[add_tasks_to_dataset],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode,
        )

        self.action_set.python_includes = "from webexp.agents.nav_explorer_agent import TASK_COLLECTOR, add_tasks_to_dataset\n" + self.action_set.python_includes

        self.prompt_builder = NavExplorerPromptBuilder(action_set=self.action_set)

        self.config['goal_str'] = self.goal_str

    def reset(self):
        super().reset()
        TASK_COLLECTOR.clear()

    def get_proposed_tasks(self) -> list[str]:
        return TASK_COLLECTOR.copy()
    
    @property
    def goal_str(self) -> str:
        return dedent("""\
            I am trying to collect a dataset to train a better web browser agent that can perform actions for users in a web browser. For this, we are particularly interested to collect **navigation tasks** that are feasible to perform from the current web page.
            Navigation tasks are tasks requiring navigating to a specific page.

            Collect navigation tasks that require navigating to another webpage from this current page. You may click on links to try finding other interesting pages to collect tasks from. But if you do navigate to another page, instead of collecting tasks on that page, make sure to navigate back to the previous page using `go_back` or `goto`. We will collect tasks from these new pages later. When collecting navigation tasks, prioritize those that would likely have interesting/useful tasks on them over ones that likely won't give many useful tasks to collect.

            As you are exploring, you can add navigation tasks to the dataset using the `add_tasks_to_dataset` function.

            When you are done exploring the current page, send a message to the user using `send_msg_to_user` confirming this.
                      
            Be sure to prioritize adding navigation tasks to pages that a typical user of this web page would most often want to navigate to, over niche pages that the typical user would rarely frequent.

            **Important**
            Remember that if you are successful at navigating to a new page, you should add a corresponding task to the dataset as your next action before finding new pages."""
        )
