from pathlib import Path

import yaml

from lwfm.core.config import ExperimentConfig


def load_yaml_config(path: Path) -> ExperimentConfig:
    with path.open("r") as yaml_file:
        raw_config = yaml.safe_load(yaml_file)

    return ExperimentConfig.model_validate(raw_config)
