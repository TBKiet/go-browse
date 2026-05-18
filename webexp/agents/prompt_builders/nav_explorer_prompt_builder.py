from .solver_prompt_builder import SolverPromptBuilder
from textwrap import dedent

class NavExplorerPromptBuilder(SolverPromptBuilder):

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
                You are a UI Assistant whose goal is to explore a web page and find navigation tasks.
                Review the page state to discover where users can navigate to from the current page.
                """
            ) + som_note + dedent("""\

                FORMAT: Your entire response must be a single JSON object: {"thought": "...", "action": "..."}
                - Do NOT output any text, explanation, or markdown outside the JSON.
                - All reasoning goes inside the "thought" key.
                - Only ONE action in the "action" key.
                - Use add_tasks_to_dataset() to record navigation tasks you discover.
                - Use send_msg_to_user() when you are done exploring.
                """
            )
        }

    def cot_examples(self)  -> list[dict]:
        return [
            {"thought": "It seems that we can navigate to different pages including Product Reviews, Shopping Cart, and Help Center from this page. Before adding these as navigation tasks, I will first try navigating to the Product Reviews page to verify the link works.", "action": "click('42')"},
            {"thought": "I successfully navigated to the Product Reviews page and confirmed it shows review content. After returning to the original page, I can now add this as a verified task with a specific description.", "action": "add_tasks_to_dataset('[NAV] Navigate to the Product Reviews section for the current product on this electronics store')"},
            {"thought": "I see a menu item for product categories. Expanding this menu by clicking on it might reveal additional specific pages to navigate to. Let me check what's available.", "action": "click('5')"},
            {"thought": "The menu reveals category pages like 'Laptops', 'Tablets', and 'Accessories'. Let me verify the Laptops link works, then add specific navigation tasks.", "action": "click('15')"},
            {"thought": "The Laptops category page loaded correctly. Now I can add verified navigation tasks with specific destination descriptions.", "action": "add_tasks_to_dataset('[NAV] Navigate to the Laptops category page with Dell and HP filters applied on this electronics store', '[NAV] Go to the Tablets and iPads section on this electronics store')"},
            {"thought": "I have thoroughly explored this web page, verified navigation links by testing them, and found quality navigation tasks with specific destination pages. I will now confirm completion.", "action": "send_msg_to_user('I have finished exploring and collecting verified navigation tasks on this web page. We can move on to the next page.')"},
        ]
