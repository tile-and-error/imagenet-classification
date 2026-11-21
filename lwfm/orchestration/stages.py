from abc import ABC, abstractmethod
from dataclasses import dataclass

from lwfm.orchestration.runcontext import RunContext
from lwfm.core.graph import ExecutionGraph

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
        
        # Save Checkpoint
        # Record Output
        # Make Model Available
        pass

class EvaluateStage(Stage):
    name = "evaluate"

    def execute(self, ctx: RunContext) -> StageResult:
        pass

class BenchmarkStage(Stage):
    name = "benchmark"

    def execute(self, ctx: RunContext) -> StageResult:
        pass