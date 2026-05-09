from .solver_prompt_builder import SolverPromptBuilder

class PageExplorerPromptBuilder(SolverPromptBuilder):

    def cot_examples(self) -> list[dict]:
        return [
            {"thought": "I see selectors for choosing the date on this orders page. An example task could be to find the orders in a particular time period (e.g., in Jan 2022). Before we add this as a task to our dataset, we should first try to see if this is possible by trying out the task ourselves. I will click on the date selector.", "action": "click('12')"},
            {"thought": "This page lists information about customers that we can create information extraction tasks about. I will add such tasks to the dataset.", "action": "add_tasks_to_dataset('[INFO] What is the email address of the customer Joe Bloggs?', '[INFO] List the names of all customers from Texas')"},
            {"thought": "It seems that we can navigate to different pages including, Reviews, Home, and Recommendations from this page. Let me add these as navigation tasks to the dataset.", "action": "add_tasks_to_dataset('[NAV] Navigate to the Reviews page.', '[NAV] Go to the Home page.', '[NAV] Visit the Recommendations page.')"},
            {"thought": "I see a menu item for 'Product Information' on this page. Perhaps there are some interesting information extraction tasks based on this. Let me click on the 'Product Information' menu item to explore further to see if I can find some concrete tasks to add to the dataset.", "action": "click('5')"},
            {"thought": "My last action has taken me to a new URL/page. Since my goal is to find tasks the original page, I will first go back to the previous page to continue finding tasks on that page.", "action": "go_back()"},
            {"thought": "This is the product page for the Nintendo Switch. I see that we can perform content modification tasks such as adding the product to the cart or adding a review. I will add these tasks to the dataset.", "action": "add_tasks_to_dataset('[MOD] Add a Nintendo Switch with 256 GB storage to the cart.', '[MOD] Leave a negative review for the Nintendo Switch saying that the joycon controllers started drifting after a month of usage.')"},
            {"thought": "I have thoroughly explored this web page and found a good variety of tasks for the user to perform on this page. I will now respond to the user confirming that I have finished collecting data on this web page and now am ready to explore a new web page.", "action": "send_msg_to_user('I have finished exploring and collecting a variety of tasks on this web page. We can move on to the next page.')"}
        ]
