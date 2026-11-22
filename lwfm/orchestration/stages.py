from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import os
import torch

from lwfm.orchestration.runcontext import RunContext
from lwfm.core.graph import ExecutionGraph
from lwfm.training.optimizer import build_optimizer, build_scheduler
from lwfm.results.events import RunEvent
from lwfm.evaluation.factory import build_metric_binding

@dataclass
class StageResult:
    artifacts: dict[str, str]  # name -> path inside the run directory
    measurements: dict[str, float | int | str]
    status: str = "completed"


class Stage(ABC):
    name: str  # set by each subclass, e.g. FitStage

    @abstractmethod
    def execute(self, ctx: RunContext) -> StageResult: ...

class FitStage(Stage):
    name = "fit"

    def execute(self, ctx: RunContext) -> StageResult:
        # Load the execution Graph
        graph = ctx.execution_graph
        # Call Trainer
        ctx.data_module.setup("fit")
        train_loader = ctx.data_module.train_dataloader()
        loss = ctx.loss_composer
        optimizer = build_optimizer(
            (p for p in graph.parameters() if p.requires_grad),
            ctx.config.training
        )

        extra_config = {}

        if "scheduler" in ctx.config.training:
            if "max_epochs" not in ctx.config.training:
                raise ValueError("Max Epochs must be defined to use scheduler")
            if "scheduler_interval" not in ctx.config.training:
                raise ValueError("Scheduler Interval must be defined to use scheduler")
            
            scheduler = build_scheduler(ctx.config.training["scheduler"], optimizer, ctx.config.training["max_epochs"], len(train_loader))

            extra_config["scheduler"] = scheduler
            extra_config["scheduler_interval"] = ctx.config.training["scheduler_interval"]

        result = ctx.trainer.fit(
            graph, 
            train_loader, 
            loss, 
            optimizer,
            val_loader=ctx.data_module.val_dataloader(),
            # metrics=build_metric_binding() TODO: Which set of metrics??
            output_sink=ctx.output_router,
            *extra_config
        )
        # Save Checkpoint
        save_checkpoint(
            ctx.run_dir,
            graph,
            optimizer,
            result.epoch,
            result.global_step
        )
        # Record Output
        artefacts = {
            "fit": str(ctx.run_dir / "artifacts" / "fit_last.pt"),
            "model": str(ctx.run_dir / "artifacts" / "model_weights.pt")
        }

        return StageResult(
            artefacts,
            result,
            "completed"
        )

class EvaluateStage(Stage):
    name = "evaluate"
    split: str = "val"

    def execute(self, ctx: RunContext) -> StageResult:
        graph = ctx.execution_graph
        metric_bindings = build_metric_binding(ctx.config.metrics)

        if self.split == "val":
            ctx.data_module.setup("val")
            loader = ctx.data_module.val_dataloader()
        elif self.split == "test":
            ctx.data_module.setup("test")
            loader = ctx.data_module.test_dataloader()
        else:
            raise ValueError(f"Incorrect split {self.split}")

        result = ctx.evaluator.evaluate(
            graph,
            loader,
            metric_bindings,
            device = ctx.config.device,
            output_sink=ctx.output_router,
            split_name=self.split
        )

        measuerments = {
            f"{self.split}/{name}": value
            for name, value in result.metric_values.items()
        }

        return StageResult(
            {},
            measuerments=measuerments,
        )

class BenchmarkStage(Stage):
    name = "benchmark"

    def execute(self, ctx: RunContext) -> StageResult:
        pass

def save_checkpoint(
        path: Path,
        graph,
        optimizer,
        epoch: int,
        global_step: int,
        deployable_name
):
    checkpoint = {
        "model_state_dict": graph.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step
    }

    fit_path = path / "artifacts" / "fit_last.pt"
    temp_path = fit_path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, fit_path)

    student = graph.deployable_model(deployable_name)
    student_path = path / "artifacts" / "model_weights.pt"
    temp_path = student_path.with_suffix(".tmp")
    torch.save(student.state_dict(), temp_path)
    os.replace(temp_path, student_path)