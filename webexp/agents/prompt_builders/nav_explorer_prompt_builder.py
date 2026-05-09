from .solver_prompt_builder import SolverPromptBuilder

class NavExplorerPromptBuilder(SolverPromptBuilder):

    def cot_examples(self)  -> list[dict]:
        return [
            {"thought": "It seems that we can navigate to different pages including, Reviews, Home, and Recommendations from this page. Before adding these as navigation tasks, I will first try navigating to them to make sure these indeed take me to new webpages. I will start with the Reviews page.", "action": "click('42')"},
            {"thought": "It seems I was successfully able to navigate to the Reviews page and have now returned. I will add this to the list of navigation tasks and then try other places to navigate to.", "action": "add_tasks_to_dataset('[NAV] Navigate to the Reviews page.')"},
            {"thought": "I see a menu item for categories. Perhaps expanding this menu by clicking on it will show us additional places to navigate to.", "action": "click('5')"},
            {"thought": "I have thoroughly explored this web page and found a good variety of tasks for the user to perform on this page. I will now respond to the user confirming that I have finished collecting data on this web page and now am ready to explore a new web page.", "action": "send_msg_to_user('I have finished exploring and collecting a variety of tasks on this web page. We can move on to the next page.')"},
        ]
