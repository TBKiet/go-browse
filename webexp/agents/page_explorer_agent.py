from .base_agent import AgentFactory
from .solver_agent import SolverAgent
from .prompt_builders.page_explorer_prompt_builder import PageExplorerPromptBuilder
from browsergym.core.action.highlevel import HighLevelActionSet
from textwrap import dedent

TASK_COLLECTOR = []

def add_tasks_to_dataset(*tasks: str):
    """Given one or more task strings, add them to the dataset we are collecting.
    You should add tags to the start of the task string to indicate the type of task it is.
    For example, you can use the following tags:
    [INFO] for information seeking tasks,
    [NAV] for navigation tasks, and
    [MOD] for content modification tasks, configuration changes, or anything that modifies the state of the webpage.
    You can add multiple tags to a single task string if it is a combination of different types of tasks.
    For example, you can use [INFO][NAV] for a task that requires both information seeking and navigation.
    
    Examples:
        add_tasks_to_dataset('[MOD] Add the Apple iPhone 13 to the cart.', '[MOD] Leave a review for iPhone 13 saying that I loved it.')
        add_tasks_to_dataset('[INFO] List the best-selling product for first quarter of 2023.')
        add_tasks_to_dataset('[INFO] Compare the driving and walking times from University of Washington to Amazon's headquarters in Seattle.')
        add_tasks_to_dataset('[NAV] Navigate to the product page for the Apple iPhone 13.')
    """
    TASK_COLLECTOR.extend(tasks)


@AgentFactory.register
class PageExplorerAgent(SolverAgent):
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

        self.action_set.python_includes = "from webexp.agents.page_explorer_agent import TASK_COLLECTOR, add_tasks_to_dataset\n" + self.action_set.python_includes

        self.prompt_builder = PageExplorerPromptBuilder(action_set=self.action_set)

    def reset(self):
        super().reset()
        TASK_COLLECTOR.clear()

    def get_proposed_tasks(self) -> list[str]:
        return TASK_COLLECTOR.copy()
    
    @property
    def goal_str(self) -> str:
        return dedent("""\
            I am trying to collect a dataset to train a better web browser agent that can perform actions for users in a web browser. For this, I need to first collect tasks that are feasible to perform on the current web page. 
            The tasks should be concrete (e.g., on an amazon product page for product X, an appropriate task could be "Leave a positive review for X" or on a maps website a task could be "Show me driving directions from X to Y." where X and Y are specific locations).
            You may explore by performing actions on this web page if that helps to determine concrete tasks that are feasible.

            Find the tasks that are possible to perform on the current web page itself, without have to navigate to other links/urls. Though, you may find it helpful to navigate through menus on this page to get a better idea of what types of tasks are feasible. If you accidentally go to a new url while trying to navigate items on the page, you can go back to the previous page using the `go_back` function.

            Tasks are usually of three types:
            1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. 
            2. Site navigation: The user wants to navigate to a specific page.
            3. Content modification: The user wants to modify the content of a webpage or configuration.

            Be as specific as you can while creating tasks. The web agent may start from a different web page when asked to complete the task and so may not have the current page context to understand the task. So, for example, avoid creating generic tasks like "Add item to cart" or "Print receipt for this order." Instead you want to create specific tasks like "Add a Sony PS5 to cart" or "Print a receipt for Martha Jone's order of the Nike Velocity Sweatpants from May 21, 2021"

            I recommend the following order to collecting tasks: 
            1. First look for information seeking/extraction tasks that can be answered simply using information on the current page, requiring no additional actions.
            2. Collect navigation tasks that require navigating to another webpage from this current page. You may click to links to try finding other interesting pages to collect tasks from. But if you do navigate to another page, instead of collecting tasks on that page, make sure to navigate back to the previous page using `go_back`. We will collect tasks from these new pages later. When collecting navigation tasks, prioritize those that would likely have interesting/useful tasks on them over ones that likely won't give many useful tasks to collect.
            3. Finally, you can try to find content modification tasks on the current page that require performing actions on the current page itself.

            As you are exploring the page, you may find it helpful to click on buttons, links, and other elements on the page to see if they reveal any additional information or options that could lead to new tasks. You can also hover over elements to see if they provide any tooltips or additional context.         
            
            **Important**:
            When collecting tasks, focus more on the common tasks that a typical user of this webpage would want to perform. Avoid niche tasks that are unlikely to be relevant to the typical user of this website.
            For most common styles of tasks, it may be useful to include a few variants or related tasks to help the web agent learn frequently used skills.
          
            As you are exploring, you can add tasks to the dataset using the `add_tasks_to_dataset` function.

            When you are done exploring, send a message to the user using `send_msg_to_user` confirming this."""
        )
