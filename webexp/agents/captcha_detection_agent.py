"""CAPTCHA Detection Agent.

Checks whether a web page contains a CAPTCHA challenge, so the exploration
loop can skip such pages instead of wasting LLM calls on unsolvable tasks.
"""

from PIL import Image
from openai import OpenAI
import base64
import io
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _pil_to_b64(image: Image.Image) -> str:
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()


class CaptchaDetectionAgent:
    """Checks if a page screenshot shows a CAPTCHA challenge.

    Uses GPT-4V to visually inspect the screenshot and answer a simple
    yes/no question. Lightweight — only called once per seed page.
    """

    SYSTEM_PROMPT = """You are a CAPTCHA detection system. Your job is to look at a screenshot of a web page and determine whether it contains a CAPTCHA challenge (e.g., reCAPTCHA, hCaptcha, image selection puzzles, "I am not a robot" checkboxes).

Answer ONLY with the word "yes" or "no".

Answer:"""

    def __init__(self, model_name: str = "gpt-4o-2024-05-13"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", None),
        )

    def is_captcha(self, screenshot) -> bool:
        """Check if a screenshot contains a CAPTCHA.

        Args:
            screenshot: PIL Image or numpy array of the page screenshot.

        Returns:
            True if the page is a CAPTCHA, False otherwise.
        """
        try:
            if not isinstance(screenshot, Image.Image):
                screenshot = Image.fromarray(screenshot)

            b64 = _pil_to_b64(screenshot)

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        }
                    ],
                },
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )

            answer = response.choices[0].message.content.strip().lower()
            is_captcha = answer == "yes"
            logger.info(f"CAPTCHA check: answer='{answer}', is_captcha={is_captcha}")
            return is_captcha

        except Exception as e:
            logger.error(f"CAPTCHA detection failed: {e}")
            # On error, assume not a captcha to avoid skipping valid pages
            return False
