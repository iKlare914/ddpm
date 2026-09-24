from pydantic import BaseModel, ConfigDict, PositiveInt
from pathlib import Path

class StrictModel(BaseModel):
    """Base configuration model that rejects undeclared fields."""

    model_config = ConfigDict(extra='forbid')

class TrainerConfig(StrictModel, frozen=True):
    """Training configuration whose fields cannot be reassigned after creation.

    Attributes:
        lr: Optimizer learning rate, currently declared as a positive integer.
        batch_size: Number of samples per training batch. Must be positive.
        weight_decay: Weight decay coefficient for the optimizer.
        epoches: Total number of training epochs. Must be a positive integer.
        resume: Whether to resume training from an existing checkpoint.
            Defaults to False.
        save_dir: Directory in which to save training checkpoints.
    """

    lr: PositiveInt
    batch_size: PositiveInt
    weight_decay: float
    epoches: PositiveInt
    resume: bool = False
    save_dir: Path
