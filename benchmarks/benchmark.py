from functools import partial
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import timeit

from tqdm import tqdm

from benchmarks.make_textgrid import build_files
from benchmarks.impls import *


METHODS = {
    "gridio": {
        "construct": gridio_construct,
        "to_df": gridio_to_df,
    },
    "textgrid": {
        "construct": textgrid_construct,
        "to_df": textgrid_to_df,
    },
    "parselmouth": {
        "construct": parselmouth_construct,
        "to_df": parselmouth_to_df,
    },
}


def run_benchmarks(
    files: list[Path],
    repeats: int,
    tasks: list[str] = ["construct", "to_df"],
    packages: list[str] = ["gridio", "textgrid", "parselmouth"],
) -> pd.DataFrame:
    records = []
    for task, package in tqdm(list(product(tasks, packages))):
        func = METHODS[package][task]
        # Warm up
        _ = func(files)
        timer = timeit.Timer(partial(func, files))
        times = timer.repeat(repeat=repeats, number=1)
        for run_idx, delta in enumerate(times, start=1):
            records.append(
                {
                    "package": package,
                    "task": task,
                    "run": run_idx,
                    "seconds": delta,
                }
            )
    records_df = pd.DataFrame.from_records(records)
    # summary_df = records_df.groupby(["package", "task"], as_index=False).agg(
    #     mean=("seconds", "mean"),
    #     std=("seconds", "std"),
    #     min=("seconds", "min"),
    #     max=("seconds", "max"),
    # )
    return records_df


if __name__ == "__main__":
    temp_dir = TemporaryDirectory()
    output_dir = Path(temp_dir.name)
    files = build_files(
        output_dir=output_dir,
        file_count=5000,
        interval_tiers=2,
        point_tiers=2,
        interval_items=50,
        point_items=50,
        step=0.01,
    )
    try:
        results = run_benchmarks(
            files,
            repeats=5,
            tasks=["construct", "to_df"],
            packages=["gridio", "textgrid", "parselmouth"],
        )
        results.to_csv("benchmarks/results/all.csv", index=False)
    finally:
        temp_dir.cleanup()
