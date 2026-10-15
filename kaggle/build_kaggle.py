"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public]

writes kaggle/dataset/ (one replicate of every scenario, the full results table and the datasheet) and
kaggle/notebook/ (a CPU notebook that embeds the package source and reruns the benchmark).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvbench.harness import SEED, run, summarise  # noqa: E402
from cvbench.scenarios import SCENARIOS  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/cv-honesty-benchmark"
NOTEBOOK_ID = "locdunhat/does-your-cv-tell-the-truth-six-traps"
TITLE = "Does Your CV Tell the Truth? Six Traps"
REPLICATES = 8


def build_dataset(public: bool) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, sc) in enumerate(SCENARIOS.items()):
        d = sc.generate(np.random.default_rng([SEED, 0, i]))
        train = d.X.copy()
        if d.groups is not None:
            train.insert(0, "group", d.groups)
        if d.time is not None:
            train.insert(0, "time", d.time)
        train["y"] = d.y
        test = d.X_test.copy()
        test["y"] = d.y_test
        train.to_csv(out / f"{name}_train.csv", index=False)
        test.to_csv(out / f"{name}_deploy.csv", index=False)
    results = run(replicates=REPLICATES)
    results.to_csv(out / "results.csv", index=False)
    summarise(results).to_csv(out / "summary.csv", index=False)
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "CV Honesty Benchmark",
        "id": DATASET_ID,
        "subtitle": "Synthetic tabular tasks with a known deployment score, to test validation schemes",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["tabular", "benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "cvbench").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + """

import os, sys
ROOT = "/kaggle/working/cv-honesty-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("cv-honesty-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""),
        ("code", f"""import time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn
from cvbench.harness import run, summarise, pivot
from cvbench.scenarios import SCENARIOS
pd.set_option("display.precision", 4)
print("scikit-learn", sklearn.__version__)
t = time.time()
results = run(replicates={REPLICATES})
summary = summarise(results)
print(len(results), "estimates in", round(time.time() - t), "s")
"""),
        ("markdown", "## 1. Bias of each scheme (estimate minus deployment AUC, mean over replicates)\n\n"
                     "Positive means the scheme promised more than the model delivered. Empty cells are schemes "
                     "that need structure the scenario does not have (groups, timestamps, a categorical to encode)."),
        ("code", "pivot(summary, 'bias').style.format('{:+.3f}', na_rep='').background_gradient(cmap='RdBu_r', vmin=-0.1, vmax=0.1, axis=None)"),
        ("markdown", "## 2. RMSE: bias and variance together\n\nThis is the error you should expect from one run "
                     "of the scheme on one dataset. On the small dataset all schemes are unbiased; what separates "
                     "them is variance."),
        ("code", "pivot(summary, 'rmse').style.format('{:.3f}', na_rep='').background_gradient(cmap='Reds', axis=None)"),
        ("markdown", "## 3. Every estimate against the truth"),
        ("code", """fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
for ax, (name, g) in zip(axes.ravel(), results.groupby("scenario", sort=False)):
    for scheme, s in g.groupby("scheme", sort=False):
        ax.scatter(s.truth, s.estimate, s=18, label=scheme, alpha=0.8)
    lo, hi = min(g.truth.min(), g.estimate.min()) - 0.01, max(g.truth.max(), g.estimate.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set_title(name); ax.set_xlabel("deployment AUC"); ax.set_ylabel("CV estimate")
handles, labels = [], []
for ax in axes.ravel():
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels: handles.append(h); labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=4); plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.show()
"""),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
        ("code", "summary.sort_values(['scenario', 'rmse'])"),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for kind, src in _cells():
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "cv-honesty.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "cv-honesty.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "keywords": ["evaluation", "benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public))
    print("wrote", build_notebook(public))
