from pathlib import Path
import torch
from datetime import datetime, timezone
import random

from lwfm.models.factory import build_model
from lwfm.core.config import ExperimentConfig
from lwfm.orchestration.runcontext import RunContext
from lwfm.core.graph import ExecutionGraph
from lwfm.orchestration.recipe import Recipe
from lwfm.orchestration.stages import FitStage, EvaluateStage, BenchmarkStage
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import ConsoleSink, LocalFileSink
from lwfm.evaluation.evaluator import Evaluator
from lwfm.training.trainer import Trainer
from lwfm.core.losses import LossComposer
from lwfm.data.factory import build_data_module
from lwfm.core.lossfactory import build_loss_composer

class Experiment:
    def __init__(
            self,
            config: ExperimentConfig
    ) -> None:
        self.config = config

        batch = {
            "inputs": {"image": torch.rand(8, 3, 96, 96)},  # floats in [0, 1]
            "targets": {"class": torch.randint(0, 10, (8,))},
            "meta": {"sample_id": [f"s{i}" for i in range(8)]},
        }

        models = [build_model(config.model)]
        pipeline = ExecutionGraph(models, preflight_sample=batch)

        time_utc = datetime.now(timezone.utc)
        run_id = random.randint(1000000000, 9999999999)
        experiment_name = self.config.name
        
        self.context = RunContext(
            experiment_name,
            self.config,
            run_id,
            Path.cwd(),
            pipeline,
            build_trainer(config.training, config.device),
            Evaluator(),
            OutputRouter(build_routers(config.outputs, experiment_name, time_utc, run_id)),
            build_data_module(config.data, config.seed),
            build_loss_composer(config.losses),
            time_utc
        )

        self.recipe = build_recipe(config.recipe)

    def run(self):
        self.recipe.execute(self.context)

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

def build_routers(outputs, experiment_name, time_utc, run_id):
    for i in range(len(outputs["sinks"])):
        if outputs["sinks"][i] == "local":
            output_dir = f"{time_utc.strftime("%Y-%m-%d_%H-%M-%S")}_{experiment_name}_{run_id}" 
            file_dir = Path().cwd().parent / outputs["root"] / output_dir
            return LocalFileSink(file_dir)
        elif outputs["sinks"][i] == "console":
            return ConsoleSink()
        else:
            raise ValueError("Incorrect Router Type")

def build_stage(stage):
    if stage["type"] == "fit":
        return FitStage()
    elif stage["type"] == "evaluate":
        return EvaluateStage()
    elif stage["type"] == "benchmark":
        return BenchmarkStage()
    else:
        raise ValueError("Incorrect Stage Type")

def build_recipe(recipe):
    stages = [build_stage(stage) for stage in recipe["stages"]]
    return Recipe(stages)