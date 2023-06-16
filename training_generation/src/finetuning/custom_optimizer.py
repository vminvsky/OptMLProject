from finetuning.sam import SAM
import torch
from torch.optim import Optimizer

class CustomSAMOptimizer(Optimizer):
    def __init__(self, model_params, lr, momentum):
        self.base_optimizer = torch.optim.AdamW
        self.optimizer = SAM(model_params, self.base_optimizer, lr=lr, rho=0.05)
        self.param_groups = self.optimizer.param_groups
        super().__init__(self.param_groups, {})  # Call the base class's __init__ method

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def first_step(self, zero_grad=False):
        self.optimizer.first_step(zero_grad)

    def second_step(self, zero_grad=False):
        self.optimizer.second_step(zero_grad)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)