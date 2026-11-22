from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lwfm.core.config import ExperimentConfig
from lwfm.core.graph import ExecutionGraph
from lwfm.results.router import OutputRouter
from lwfm.training.trainer import Trainer
from lwfm.evaluation.evaluator import Evaluator
from lwfm.data.modules import DataModule
from lwfm.core.losses import LossComposer

@dataclass
class RunContext:
    experiment_name: str
    config: ExperimentConfig
    run_id: int
    run_dir: Path
    execution_graph: ExecutionGraph
    trainer: Trainer
    evaluator: Evaluator

    output_router: OutputRouter
    data_module: DataModule
    loss_composer: LossComposer

    started_at: datetime