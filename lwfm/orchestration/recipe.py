from lwfm.orchestration.stages import Stage, StageResult
from lwfm.orchestration.runcontext import RunContext

class Recipe:
    def __init__(self, stages: list[Stage]):
        names = [stage.name for stage in stages]

        if len(set(names)) < len(names):
            raise ValueError("Stage names must be unique")
        
        self.stages = stages

    def execute(self, ctx: RunContext) -> list[StageResult]:
        results = []

        for stage in self.stages:
            result = stage.execute(ctx)
            results.append(result)

            if result.status != "completed":
                break

        return results