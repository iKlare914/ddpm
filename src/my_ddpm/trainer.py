from my_ddpm.load_dataset import getCifarLoader, DataLoader
from my_ddpm.logger import get_logger
from my_ddpm.sampler import DDIMSampler, DDPMSampler, TimestepSampler
from my_ddpm.config import TrainerConfig
import torch as th
import torch.nn as nn
import wandb
from pathlib import Path
from time import time
from datetime import datetime

logger = get_logger(__name__)

class Trainer():
    def __init__(
            self,
            *,
            model: nn.Module,
            data: DataLoader,
            diffusion_sampler: DDIMSampler | DDPMSampler,
            timestep_sampler: TimestepSampler,
            config: TrainerConfig 
    ):
        self.lr = config.lr
        self.dataloader = data
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.epoches = config.epoches
        self.resume = config.resume 
        self.save_dir: Path = config.save_dir if config.save_dir is not None else Path(__file__).parent / 'model-weights' / datetime.now().strftime("%Y %m %d %H %M") 
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.name = self.save_dir.name
        self.model = model
        self.device = model.device
        self.diffusion_sampler = diffusion_sampler
        self.timestep_sampler = timestep_sampler
    
    def train(self):
        pass

    def get_grad_norm(self) -> float:
        """Return the L2 norm of all gradients flattened into one vector.

        Parameters without gradients are skipped. Return 0.0 if none exist.
        """
        grads = [
            param.grad.detach().reshape(-1)
            for param in self.model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return 0.0
        grad_vector = th.cat(grads)
        return th.linalg.vector_norm(grad_vector, ord=2).item()
