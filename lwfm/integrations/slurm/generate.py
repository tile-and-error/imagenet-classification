import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Map defaults to their standard templates
NAME_MAP = {
    "cpu": "cpu",
    "cpu_multi": "cpu_multi",
    "gpu": "gpu",
    "gecko": "gpu",
    "eagle": "gpu",
    "falcon": "gpu",
}

# Get Jinja Templates
env = Environment(loader=FileSystemLoader("templates/"), autoescape=select_autoescape())

# Load script arguments
system = "dcs"
largepath = sys.argv[1]
type = sys.argv[2]
template_name = sys.argv[3]  # "cpu"

username = sys.argv[4]
timestamp = sys.argv[5]

if template_name not in NAME_MAP:
    raise ValueError("Not a valid template")

# Get Template
template_name = f"dcs_{NAME_MAP[template_name]}.j2"
template = env.get_template(template_name)

if type == "default":
    # Use a default template
    experiment_file = sys.argv[6]

    yaml_file_name = f"defaults/{system}_{template_name}.yaml"
    yaml_file = Path(__file__).resolve().parent / yaml_file_name
    yaml_text = yaml_file.read_text(encoding="utf-8")
    context_data = yaml.safe_load(yaml_text) or {}

    context_data["experiment"] = experiment_file

else:
    # Use a custom template
    yaml_file = sys.argv[6]

    yaml_text = Path(yaml_file).read_text(encoding="utf-8")
    context_data = yaml.safe_load(yaml_text) or {}

context_data["largepath"] = largepath
context_data["id"] = f"{context_data['id']}_{username}_{timestamp}"

# Create new sbatch file
output = template.render(**context_data)

file_name = f"configs/slurmjobs/{username}_{timestamp}.sbatch"
result_file = Path.cwd().parent.parent / file_name

result_file.write_text(output, encoding="utf-8")
