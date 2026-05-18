from .solver_prompt_builder import SolverPromptBuilder
from textwrap import dedent

class PageExplorerPromptBuilder(SolverPromptBuilder):

    def system_message(self, use_som: bool = False):
        som_note = ""
        if use_som:
            som_note = dedent("""\

                ## Set-of-Mark Visual Grounding
                The screenshot has colored numbered tags overlaid on interactive elements.
                Each tag number corresponds to the [bid] value in the accessibility tree.
                Use the visual tags to locate elements you want to interact with.
                """)

        return {
            "type": "text",
            "text": dedent("""\
                # Instructions
                You are a UI Assistant whose goal is to explore a web page and find tasks that users can perform on it.
                Review the page state and all other information to find and collect diverse, feasible tasks.
                """
            ) + som_note + dedent("""\

                FORMAT: Your entire response must be a single JSON object: {"thought": "...", "action": "..."}
                - Do NOT output any text, explanation, or markdown outside the JSON.
                - All reasoning goes inside the "thought" key.
                - Only ONE action in the "action" key.
                - Use add_tasks_to_dataset() to record tasks you discover.
                - Use send_msg_to_user() when you are done exploring.
                """
            )
        }

    def cot_examples(self) -> list[dict]:
        return [
            {"thought": "I see selectors for choosing the date on this orders page. An example task could be to find the orders in a particular time period. Before adding this as a task, I should verify it's feasible by trying the date selector. Let me click on the date selector to see if it works.", "action": "click('12')"},
            {"thought": "The date selector works and shows orders from specific periods. Now I can add concrete information-seeking tasks with specific details.", "action": "add_tasks_to_dataset('[INFO] Find all orders placed between January 1, 2023 and January 31, 2023 on this orders dashboard')"},
            {"thought": "This page lists customer information. I can create specific information extraction tasks with concrete names and details.", "action": "add_tasks_to_dataset('[INFO] What is the email address and phone number of customer Joe Bloggs from Texas on this CRM?', '[INFO] List the names and order totals of all customers from California on this dashboard')"},
            {"thought": "It seems that we can navigate to different pages including Reviews, Home, and Recommendations from this page. Let me verify one of these links actually leads to the expected page before adding tasks.", "action": "click('23')"},
            {"thought": "The Reviews link works and leads to a product reviews page. Now I can add verified navigation tasks with specific destination pages.", "action": "add_tasks_to_dataset('[NAV] Navigate to the Product Reviews page on this website', '[NAV] Go to the Shopping Cart page on this website')"},
            {"thought": "My last action has taken me to a new URL/page. Since my goal is to find tasks on the original page, I will go back to continue exploring.", "action": "go_back()"},
            {"thought": "This is the product page for the Nintendo Switch OLED model. I see we can perform content modification tasks such as adding the product to the cart or leaving a review. Let me verify the Add to Cart button works first.", "action": "click('45')"},
            {"thought": "The Add to Cart button works and shows the item was added. Now I can add verified content modification tasks with specific product details.", "action": "add_tasks_to_dataset('[MOD] Add a Nintendo Switch OLED with White Joy-Con controllers to the shopping cart on this electronics store', '[MOD] Leave a 3-star review saying \"Great console but the Joy-Con drift issue persists even on the OLED model\" for the Nintendo Switch OLED on this electronics store')"},
            {"thought": "I have thoroughly explored this web page, verified tasks are feasible by testing actions, and found a good variety of specific tasks with concrete details. I will now confirm completion.", "action": "send_msg_to_user('I have finished exploring and collecting a variety of specific, verified tasks on this web page. We can move on to the next page.')"},
        ]
