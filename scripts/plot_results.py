"""Portable redraws from saved manuscript evidence; no new model fits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def read(name):
    return json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))


def save(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    data = defaultdict(list)
    for r in read("missingness.json"):
        data[(r["dataset"], r["severity"])].append(100 * (r["scores"]["repaired"]["macro_f1"] - r["scores"]["observed"]["macro_f1"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True, layout="constrained")
    for ax, task, color in zip(axes, ("adult", "diabetes", "steel"), (BLUE, ORANGE, GREEN)):
        for j, rate in enumerate((.05, .1, .2)):
            values = data[(task, rate)]
            ax.scatter([j - .1, j, j + .1], values, color=color, s=32)
            ax.errorbar(j + .22, mean(values), yerr=stdev(values), fmt="D", color="black", capsize=3, markersize=4)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set(xticks=[0, 1, 2], xticklabels=["5", "10", "20"], title=task.title(), xlabel="Injected missing cells (%)", ylim=(-1.05, .8))
    axes[0].set_ylabel("Repair benefit in macro-F1 (pp)")
    fig.suptitle("27 saved draws; cell detection F1 = 1.000 throughout\nPoints: draws · Diamonds/whiskers: mean and sample SD", fontsize=11)
    save(fig, output, "missingness")

    readiness = read("readiness.json")
    models = ("logistic_regression", "random_forest", "xgboost")
    metrics = ("macro_f1", "average_precision", "roc_auc")
    native = {(r["model"], r["metric"]): 100 * (r["value"] - r["observed"])
              for r in readiness["signed_effects"] if r["case"] == "C01" and r["arm"] == "rules"}
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    for j, metric in enumerate(metrics):
        ax.bar([i + .24 * (j - 1) for i in range(3)], [native[(m, metric)] for m in models], width=.23,
               label={"macro_f1": "Macro-F1", "average_precision": "AP", "roc_auc": "ROC-AUC"}[metric], color=(BLUE, ORANGE, GREEN)[j])
    ax.axhline(0, color="#777777", linewidth=.8)
    ax.set(xticks=[0, 1, 2], xticklabels=["LR", "RF", "XGB"], ylabel="Native repair benefit (pp)", title="One native Diabetes input; fixed held-out encounters")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    save(fig, output, "native_diabetes")

    names = ["Truth", "Rules", "Single", "Multi"]
    codes = {"READY": 0, "REMEDIATE": 1, "NOT_READY": 2, None: 3}
    labels = {"READY": "READY", "REMEDIATE": "REMEDIATE", "NOT_READY": "NOT READY", None: "FAILED"}
    rows = [[r["primary_truth"]["label"], *(r["predictions"][m] for m in ("rules", "single_llm", "multi_agent"))] for r in readiness["cases"]]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.imshow([[codes[v] for v in row] for row in rows], cmap=ListedColormap(["#DDEEDF", "#F9E8CD", "#E5DDEB", "#E8E8E8"]), vmin=0, vmax=3, aspect="auto")
    for i, row in enumerate(rows):
        for j, value in enumerate(row): ax.text(j, i, labels[value], ha="center", va="center", fontsize=9)
    ax.set(xticks=range(4), xticklabels=names, yticks=range(6),
           yticklabels=["C01 Native", "C02 Missing 5%", "C03 Domain 10%", "C04 Domain 20%", "C05 All X missing", "C06 All y missing"],
           title="Six matched readiness cases; failures remain in the denominator")
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    save(fig, output, "readiness")
    print(json.dumps({"figures": 3, "formats": ["png", "svg"], "output": str(output)}))


if __name__ == "__main__":
    main()
