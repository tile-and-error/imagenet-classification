# Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python package and project manager.

## Set up the virtual environment

From the repository root:

1. **Create the virtual environment** (uv uses the Python version in `.python-version` and downloads it if needed):
   ```bash
   uv venv
   ```
2. **Install the requirements** from `requirements.txt` into it:
   ```bash
   uv pip install -r requirements.txt
   ```
3. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate        # macOS / Linux (bash, zsh)
   source .venv/bin/activate.fish   # fish
   .venv\Scripts\activate           # Windows
   ```

   Run `deactivate` to leave it.

4. **Verify correct Python binary:**
   ```bash
   which python     # macOS / Linux
   where.exe python # Windows
   ```

5. **(Optional) Install the notebook kernel** to run the notebooks in `examples/`:

   ```bash
   uv sync --extra examples --inexact
   ```

   This installs the project's dependencies from `pyproject.toml` plus `ipykernel` (the `examples` extra), so `.venv` can be picked as the kernel in Jupyter or VS Code. `--inexact` keeps the packages from step 2, which `uv sync` would otherwise remove.

# Datasets

`download_data.py` downloads a dataset, sorts it into a standard folder layout and checks that nothing is missing. Everyone runs the same command, so everyone ends up with the same data.

| Dataset                   | Command name    | Download size | Classes | Train     | Test                             |
| ------------------------- | --------------- | ------------- | ------- | --------- | -------------------------------- |
| ImageNet-1k (ILSVRC 2012) | `imagenet-1k`   | ~150 GB       | 1000    | 1,281,167 | 50,000 (official validation set) |
| Tiny ImageNet             | `tiny-imagenet` | ~160 MB       | 200     | 100,000   | 10,000                           |

Use `tiny-imagenet` to try the script out before downloading ImageNet-1k.

> [!IMPORTANT]
> `imagenet-1k` and `tiny-imagenet` have a different order of labels and don't fully overlap. `download/imagenet_mapping.json` (committed, so there is nothing to run) maps each Tiny ImageNet folder (`000`–`199`) to its WordNet ID, name and ImageNet-1k index: `{"000": {"wnid": ..., "name": ..., "imagenet_1k": ...}}`. `imagenet_1k` is `null` for the 18 Tiny ImageNet classes that are not in ImageNet-1k.


## 1. Set up Hugging Face access (once per person)

1. **Log in** to [huggingface.co](https://huggingface.co), or create a free account.
2. **Create a read token:** go to [Settings → Access Tokens](https://huggingface.co/settings/tokens), click **Create new token**, choose the **Read** role and copy the token.
3. **Install the requirements and log in from the terminal:**
   - NOTE: You should be in the virtual environment for `hf` CLI to be in PATH.
   ```bash
   pip install huggingface_hub pyarrow
   hf auth login        # paste the token when asked
   ```

## 2. Download

> [!IMPORTANT]
> Some datasets are gated. Before downloading, open the dataset's Hugging Face page (e.g. [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)) and accept the terms, otherwise the download fails with an [access error](#huggingface-access-error).

Run these from the repository root:

```bash
python download/download_data.py tiny-imagenet /path/to/data/tiny-imagenet
python download/download_data.py imagenet-1k   /path/to/data/imagenet-1k
```

- ImageNet-1k needs about 150 GB free.
- The test split downloads first. It's small, so any problem shows up before the large training download.
- If all files have been downloaded, running the command again downloads nothing and just re-runs the check. Otherwise, It skips the files that already finished and carries on.

When it's done you should see:

```
OK   test: 1,000/1,000 classes, 50,000/50,000 images
OK   train: 1,000/1,000 classes, 1,281,167/1,281,167 images
imagenet-1k is complete in /path/to/data/imagenet-1k
```

### Output layout

```
<out>/
├── train/
│   ├── 000/  *.JPEG
│   ├── 001/
│   └── ...
└── test/
    ├── 000/
    └── ...
```

Class folders are named by label number.

## Adding a dataset

1. **Write a download function** in `download_data.py` that puts images into `<out>/<split>/<class>/`.
   - **Hugging Face `.parquet` dataset** with `image` and `label` columns: call the shared function with the repo name and the file pattern for each split.
   - **Else:** write the download and extraction yourself.
2. **Register it** in `DATASETS` with the classes and images expected in each split:
   ```python
   "my-dataset": (download_my_dataset, {"train": (10, 8000), "test": (10, 2000)}),
   ```
3. **Test it** with `python download/download_data.py my-dataset /path/to/data/my-dataset`, then commit the change. Everyone else then runs the same command.

# Appendix

## HuggingFace Access Error

Example with `ILSVRC/imagenet-1k`:

```
Cannot access gated repo for url https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/validation-00000-of-00014.parquet.
Access to dataset ILSVRC/imagenet-1k is restricted and you are not in the authorized list. Visit https://huggingface.co/datasets/ILSVRC/imagenet-1k to ask for access.
```

**Fix:** open the dataset page, accept the terms, then run the command again ([back to Download](#2-download)).
