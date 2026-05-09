from glob import glob
import os

for module in glob(os.path.join(os.path.dirname(__file__), "*.py")):
    if os.path.basename(module) == "__init__.py":
        continue
    __import__(f"webexp.agents.{os.path.basename(module)[:-3]}")