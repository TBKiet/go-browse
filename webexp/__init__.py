from browsergym.core.registration import register_task
from browsergym.webarena import config, task

import browsergym.webarena

class ExplorationTaskWrapper(task.GenericWebArenaTask):
    def setup(self, page):
        super().setup(page)
        self.evaluator = lambda *args, **kwargs: 0.0
        return None, {}

# TODO: We probably only need to register one task per webarena domain and give each a human-readable name

for task_id in config.TASK_IDS:
    gym_id = f"webarena.exploration.{task_id}"
    register_task(
        gym_id,
        ExplorationTaskWrapper,
        task_kwargs={"task_id": task_id},
    )
