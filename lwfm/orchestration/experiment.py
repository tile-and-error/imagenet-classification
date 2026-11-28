import random
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import torch

from lwfm.core.config import ExperimentConfig
from lwfm.core.graph import ExecutionGraph
from lwfm.data.factory import build_data_module
from lwfm.evaluation.evaluator import Evaluator
from lwfm.evaluation.factory import build_metric_bindings
from lwfm.loss.lossfactory import build_loss_composer
from lwfm.models.factory import build_model
from lwfm.orchestration.recipe import Recipe
from lwfm.orchestration.runcontext import RunContext
from lwfm.orchestration.stages import BenchmarkStage, EvaluateStage, FitStage, Stage
from lwfm.results.events import RunEvent, RunStatus
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import ConsoleSink, LocalFileSink, OutputSink
from lwfm.training.trainer import Trainer


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

        batch = {
            "inputs": {"image": torch.rand(8, 3, 96, 96)},  # floats in [0, 1]
            "targets": {"class": torch.randint(0, 10, (8,))},
            "meta": {"sample_id": [f"s{i}" for i in range(8)]},
        }

        models = [build_model(config.model)]
        pipeline = ExecutionGraph(models, preflight_sample=batch)

        time_utc = datetime.now(UTC)
        run_id = random.randint(1000000000, 9999999999)
        experiment_name = self.config.name

        output_name = (
            f"{time_utc.strftime('%Y-%m-%d_%H-%M-%S')}_{experiment_name}_{run_id}"
        )
        output_dir = Path().cwd().parent / config.outputs["root"] / output_name

        self.context = RunContext(
            experiment_name,
            self.config,
            run_id,
            Path.cwd(),
            output_dir,
            pipeline,
            build_trainer(config.training, config.device),
            Evaluator(),
            OutputRouter(*build_routers(config.outputs, output_dir)),
            build_data_module(config.data, config.seed),
            build_loss_composer(config.losses),
            build_metric_bindings(config.metrics),
            time_utc,
        )

        self.recipe = build_recipe(config.recipe)

    def run(self) -> None:
        status: RunStatus = "failed"
        try:
            self.recipe.execute(self.context)
            status = "completed"

        except KeyboardInterrupt:
            status = "interrupted"
            raise

        finally:
            self.context.output_router.write(
                RunEvent(
                    kind="status",
                    name="experiment/status",
                    value=status,
                    stage=None,
                    step=None,
                )
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Safely close the experiment
        context = getattr(self, "context", None)
        output_router = getattr(context, "output_router", None)

        if output_router is not None:
            output_router.close()


def build_trainer(config, device: str) -> Trainer:  # noqa: ANN001
    return Trainer(
        device=device,
        max_epochs=config["max_epochs"],
        max_steps=config["max_steps"],
    )


def build_routers(outputs, file_dir: Path) -> list[OutputSink]:  # noqa: ANN001
    sinks: list[OutputSink] = []
    for sink in outputs["sinks"]:
        if sink == "local":
            sinks.append(LocalFileSink(file_dir))
        elif sink == "console":
            sinks.append(ConsoleSink())
        else:
            raise ValueError("Incorrect Router Type")
    return sinks


STAGE_TYPES = {"fit": FitStage, "evaluate": EvaluateStage, "benchmark": BenchmarkStage}


def build_stage(stage) -> Stage:  # noqa: ANN001
    if stage["type"] not in STAGE_TYPES:
        raise ValueError(f"Incorrect stage type {stage['type']}")
    else:
        return STAGE_TYPES[stage["type"]]()


def build_recipe(recipe) -> Recipe:  # noqa: ANN001
    stages = [build_stage(stage) for stage in recipe["stages"]]
    return Recipe(stages)
