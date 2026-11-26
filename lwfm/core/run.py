import argparse
from pathlib import Path

from lwfm.orchestration.experiment import Experiment
from lwfm.core.yaml import load_yaml_config

def main():
    parser = argparse.ArgumentParser(
        description="LWFM Experiment Runner"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config"
    )

    args = parser.parse_args()

    yaml = args.config

    print(f"Loading experiment: {yaml}")

    yaml_path = Path.cwd().parent / yaml

    config = load_yaml_config(yaml_path)

    experiment = Experiment(config)

    with Experiment(config) as experiment:
        experiment.run()

if __name__ == "__main__":
    main()