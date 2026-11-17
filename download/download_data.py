#!/usr/bin/env python3
"""Download a dataset into <out>/<split>/<class>/<images>.

    python download_data.py imagenet-1k /path/to/data/imagenet-1k
    python download_data.py tiny-imagenet /path/to/data/tiny-imagenet   # ~160 MB, for
                                                                        # testing

Re-run the same command to resume an interrupted download or to re-check a finished one.

Adding a dataset: write a download_<name>(out, expected) function that produces that
layout, then add it to DATASETS with the classes and images expected per split.

"""

import argparse
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


def download_hf_parquet(out: Path, expected: dict, repo: str, splits: list) -> None:
    """Download a Hugging Face parquet dataset one file at a time into
    <out>/<split>/<label>/.

    splits: [(file pattern in the repo, local split name), ...], downloaded in that
        order.
    Class folders are the zero-padded label numbers: 000, 001, ...
    """
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    tmp = out / "_download"
    done_file = out / ".downloaded"  # shards already extracted, so re-runs resume
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    for pattern, split in splits:
        shards = sorted(f for f in files if fnmatch(f, pattern))
        for i, name in enumerate(shards, 1):
            if name in done:
                continue
            print(f"[{split} {i}/{len(shards)}] {name}")
            path = Path(hf_hub_download(repo, name, repo_type="dataset", local_dir=tmp))
            n = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=256, columns=["image", "label"]
            ):
                for image, label in zip(
                    batch.column("image").to_pylist(),
                    batch.column("label").to_pylist(),
                    strict=False,
                ):
                    class_dir = out / split / f"{label:03d}"
                    class_dir.mkdir(parents=True, exist_ok=True)
                    filename = (
                        Path(image["path"]).name
                        if image["path"]
                        else f"{path.stem}_{n}.JPEG"
                    )
                    (class_dir / filename).write_bytes(image["bytes"])
                    n += 1
            path.unlink()  # keep only one shard on disk at a time
            with done_file.open("a") as f:
                f.write(name + "\n")
        if not check(out, {split: expected[split]}):
            sys.exit(f"{split} doesn't match what was expected, so stopping here.")

    shutil.rmtree(tmp, ignore_errors=True)


def download_imagenet_1k(out: Path, expected: dict) -> None:
    """
    train <- official training set
    test <- official 50k validation set

    The real test set does not have publicly available labels.
    Class folders are 000-999: the same order pretrained ImageNet models output.
    """
    # Test set first - it's small, so a problem shows up before the ~140 GB train 
    # download.
    download_hf_parquet(
        out,
        expected,
        "ILSVRC/imagenet-1k",
        [("data/validation-*.parquet", "test"), ("data/train-*.parquet", "train")],
    )


def download_tiny_imagenet(out: Path, expected: dict) -> None:
    """
    200-class, 64x64 subset of ImageNet (~160 MB). Useful for testing this script.

    train <- 100k training images
    test <- 10k validation images

    Class folders are 000-199 in Tiny ImageNet's label order.
    """
    download_hf_parquet(
        out,
        expected,
        "zh-plus/tiny-imagenet",
        [("data/valid-*.parquet", "test"), ("data/train-*.parquet", "train")],
    )


DATASETS = {
    # name: (download function, {split: (expected classes, expected images)})
    "imagenet-1k": (
        download_imagenet_1k,
        {"train": (1000, 1_281_167), "test": (1000, 50_000)},
    ),
    "tiny-imagenet": (
        download_tiny_imagenet,
        {"train": (200, 100_000), "test": (200, 10_000)},
    ),
}


def check(out: Path, expected: dict) -> bool:
    """Compare class and image counts per split with what the dataset should have."""
    ok = True
    for split, (n_classes, n_images) in expected.items():
        split_dir = out / split
        classes = (
            [d for d in split_dir.iterdir() if d.is_dir()] if split_dir.is_dir() else []
        )
        images = sum(1 for d in classes for f in d.iterdir() if f.is_file())
        good = len(classes) == n_classes and images == n_images
        ok = ok and good
        print(
            f"{'OK  ' if good else 'FAIL'} {split}: "
            f"{len(classes):,}/{n_classes:,} classes, {images:,}/{n_images:,} images"
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    download, expected = DATASETS[args.dataset]
    args.out.mkdir(parents=True, exist_ok=True)
    download(args.out, expected)
    print(f"{args.dataset} is complete in {args.out}")


if __name__ == "__main__":
    main()
