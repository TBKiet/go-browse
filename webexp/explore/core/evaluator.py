from .trajectory import Trajectory
from PIL import Image
from typing import Union, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion
from textwrap import dedent
import base64
import io
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", None))

def extract_content(text, start_tag):
    """
    Extract the content that follows 'Info:' in a given string.

    :param text: A string that may contain lines starting with 'Info:'
    :return: The content that follows 'Info:' or None if not found
    """
    # Split the text into lines
    lines = text.split("\n")

    # Loop through each line to find a line that starts with 'Info:'
    for line in lines:
        if line.startswith(start_tag):
            # Extract and return the content after 'Info:'
            return line[len(start_tag) :].strip()

    # Return None if 'Info:' is not found in any line
    return ""

def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return image_base64


def build_vision_eval_prompt(
    intent, response, last_actions, axtree_txt
) -> tuple[str, str]:
    system_msg = dedent("""\
        You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

        There are three types of tasks:
        1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.
        2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.
        3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

        *IMPORTANT*
        Format your response into two lines as shown below:

        Thoughts: <your thoughts and reasoning process>
        Status: "success" or "failure"
        """
    )
    prompt = (
        f"User Intent: {intent}\n\n"
        "Action History:\n"
        f"{last_actions}\n\n"
        "The final state of the webpage provided as an accessibility tree:\n"
        f"{axtree_txt}\n\n"
        "The last snapshot of the web page is shown in the image."
    )

    return prompt, system_msg

class GPT4V_Client:
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def encode_image(self, path: str):
        if isinstance(path, np.ndarray):
            return image_to_jpg_base64_url(path)
            
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
                         
    def one_step_chat(
        self, text, image: Union[Image.Image, np.ndarray], 
        system_msg: Optional[str] = None,
    ) -> tuple[str, ChatCompletion]:
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"},},
                ],
        }]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content, response


class Evaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = GPT4V_Client(model_name) #LM_Client(model_name)
    
    def evaluate(self, trajectory: Trajectory):
        action_history = ""
        for idx, step in enumerate(trajectory.steps):
            action_history += f"{idx+1}: {step.action}\n"
            
        response = trajectory.response if trajectory.response else "None"
        
        prompt, sys_msg = build_vision_eval_prompt(
            trajectory.goal, response, action_history, trajectory.steps[-1].observation["axtree_txt"]
        )
        img = trajectory.steps[-1].observation["screenshot"]
        msg_str, llm_response_obj = self.client.one_step_chat(text=prompt, image=img, system_msg=sys_msg)
        
        msg_dict = {
            "thoughts": extract_content(msg_str, "Thoughts:"),
            "status": extract_content(msg_str, "Status:").replace('"', ""),
        }
        
        logger.info(f"Evaluating trajectory with goal: {trajectory.goal}")
        logger.info(f"Model Response: {msg_str}")
        
        trajectory.success = msg_dict["status"].lower() == "success"
        trajectory.reward = 1.0 if trajectory.success else 0.0

        evaluation_info = {
            "output": msg_dict,
            "reward": trajectory.reward,
            "model_usage": llm_response_obj.usage.to_dict()
        }

        if trajectory.misc is None:
            trajectory.misc = {}
        
        trajectory.misc["evaluation_info"] = evaluation_info
