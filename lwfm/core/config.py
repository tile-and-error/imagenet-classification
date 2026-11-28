# Config loading
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class ExperimentConfig(BaseModel):
    name: str
    seed: int
    device: str
    model: dict
    data: dict
    losses: list[dict]
    metrics: list[dict]
    training: dict
    recipe: dict
    outputs: dict
    deployment: dict
