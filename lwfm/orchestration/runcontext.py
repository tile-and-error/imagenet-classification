from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lwfm.core.config import ExperimentConfig
from lwfm.core.graph import ExecutionGraph
from lwfm.core.losses import LossComposer
from lwfm.data.modules import DataModule
from lwfm.evaluation.evaluator import Evaluator
from lwfm.evaluation.metrics import MetricBinding
from lwfm.results.router import OutputRouter
from lwfm.training.trainer import Trainer


@dataclass
class RunContext:
    experiment_name: str
    config: ExperimentConfig
    run_id: int
    run_dir: Path
    output_dir: Path
    execution_graph: ExecutionGraph
    trainer: Trainer
    evaluator: Evaluator

    output_router: OutputRouter
    data_module: DataModule
    loss_composer: LossComposer
    metric_bindings: list[MetricBinding]

    started_at: datetime
