import os
from typing import Optional

import imageio
from rv_eval.evaluator import LiberoEvaluator, get_evaluation_tasks, init_libero_env

class SteeringEvaluator(LiberoEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.env = None
        self.init_state = None
        self.max_steps = None
        self.instruction = None

    def setup(self, task_suite_name: str, task_name: str):
        """Set initial state for the episode."""
        
        tasks_to_evaluate = get_evaluation_tasks(task_suite_name, task_name)

        assert len(tasks_to_evaluate) == 1, "SteeringEvaluator only supports evaluating one task at a time"

        suite_name, task = list(tasks_to_evaluate.items())[0]

        self.env, init_states, self.max_steps, self.instruction = init_libero_env(
            task[0], suite_name, seed=self.seed
        )

        self.init_state = init_states[0]


    def evaluate(
        self,
        save_path: str = "run.mp4",
    ):
        
        if self.env is None or self.init_state is None:
            raise ValueError("Environment and initial state must be set up before evaluation. Call setup() first.")
        
        if self.skip_evaluated:
            if os.path.exists(save_path):
                print(f"Skipping evaluation since {save_path} already exists")
                return
            
        sucess, frames = self.run_episode(
            self.env, self.init_state, self.instruction, self.max_steps,
        )
            
        if self.save_video:
            imageio.mimwrite(save_path, frames, fps=20)
            
        self.env.close()
