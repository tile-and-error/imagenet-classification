from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lwfm.core.config import ExperimentConfig
from lwfm.core.graph import ExecutionGraph
from lwfm.results.router import OutputRouter
from lwfm.training.trainer import Trainer
from lwfm.evaluation.evaluator import Evaluator

@dataclass
class RunContext:
    config: ExperimentConfig
    run_id: int
    run_dir: Path
    execution_graph: ExecutionGraph
    trainer: Trainer
    evaluator: Evaluator

    output_router: OutputRouter

    started_at: datetime