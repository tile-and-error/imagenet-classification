import random
from datetime import UTC, datetime
from pathlib import Path

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
from lwfm.orchestration.stages import BenchmarkStage, EvaluateStage, FitStage
from lwfm.results.events import RunEvent
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import ConsoleSink, LocalFileSink
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
            OutputRouter(build_routers(config.outputs, output_dir)),
            build_data_module(config.data, config.seed),
            build_loss_composer(config.losses),
            build_metric_bindings(config.metrics),
            time_utc,
        )

        self.recipe = build_recipe(config.recipe)

    def run(self):
        try:
            self.recipe.execute(self.context)
            status = "completed"

        except KeyboardInterrupt:
            status = "interrupted"
            raise

        except Exception:
            status = "failed"
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Safely close the experiment
        context = getattr(self, "context", None)
        output_router = getattr(context, "output_router", None)

        if output_router is not None:
            output_router.close()


def build_trainer(config, device):
    return Trainer(
        device=device,
        max_epochs=config["max_epochs"],
        max_steps=config["max_steps"],
    )


def build_routers(outputs, file_dir):
    for i in range(len(outputs["sinks"])):
        if outputs["sinks"][i] == "local":
            return LocalFileSink(file_dir)
        elif outputs["sinks"][i] == "console":
            return ConsoleSink()
        else:
            raise ValueError("Incorrect Router Type")


STAGE_TYPES = {
    "fit": FitStage,
    "evaluate": EvaluateStage,
    "benchmark": BenchmarkStage
}

def build_stage(stage):
    if stage["type"] not in STAGE_TYPES:
        raise ValueError(f"Incorrect stage type {stage["type"]}")
    else:
        return STAGE_TYPES[stage["type"]]()


def build_recipe(recipe):
    stages = [build_stage(stage) for stage in recipe["stages"]]
    return Recipe(stages)
