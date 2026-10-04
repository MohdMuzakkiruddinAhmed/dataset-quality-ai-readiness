"""Recreate manuscript numerical summaries from saved measurements, without fitting."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("READY", "REMEDIATE", "NOT_READY")


def read(name):
    return json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))


def summarize():
    draws = read("missingness.json")
    grouped = defaultdict(list)
    by_dataset = defaultdict(list)
    assert len(draws) == 27 and len({r["pair_id"] for r in draws}) == 27
    for row in draws:
        assert row["model"] == "logistic_regression" and row["method"] == "rules"
        assert row["detection"]["metrics"]["f1"] == 1.0
        effect = 100 * (row["scores"]["repaired"]["macro_f1"] - row["scores"]["observed"]["macro_f1"])
        grouped[(row["dataset"], row["severity"])].append(effect)
        by_dataset[row["dataset"]].append(effect)
    assert set(grouped) == {(d, r) for d in ("adult", "diabetes", "steel") for r in (.05, .1, .2)}
    assert all(len(v) == 3 for v in grouped.values())
    severity = [{"dataset": d, "severity": rate, "draws": len(v), "mean_benefit_pp": mean(v),
                 "sample_sd_pp": stdev(v), "positive": sum(x > 0 for x in v), "negative": sum(x < 0 for x in v)}
                for (d, rate), v in sorted(grouped.items())]
    task_means = {d: {"mean_benefit_pp": mean(v), "positive": sum(x > 0 for x in v),
                      "negative": sum(x < 0 for x in v), "draws": len(v)} for d, v in sorted(by_dataset.items())}
    examples = []
    for row in read("adult_examples.json"):
        ref, obs, rep = (row["scores"][arm] for arm in ("reference", "observed", "repaired"))
        loss, benefit = ref - obs, rep - obs
        valid = row["family"] != "exact_target_encoding"
        effect = {"family": row["family"], "unit": row["detection"]["unit"],
                  "detection_f1": row["detection"]["metrics"]["f1"],
                  "benefit_pp": 100 * benefit if valid else None,
                  "recovery": benefit / loss if valid and loss > 1e-12 else None}
        examples.append(effect)
    evidence = read("readiness.json")
    cases = evidence["cases"]
    assert len(cases) == 6
    truth = [r["primary_truth"]["label"] for r in cases]
    methods = []
    for method in ("always_ready", "rules", "single_llm", "multi_agent"):
        predictions = [r["predictions"][method] for r in cases]
        completed = [(t, p) for t, p in zip(truth, predictions) if p in LABELS]
        f1 = []
        for label in LABELS:
            tp = sum(t == label and p == label for t, p in completed)
            fp = sum(t != label and p == label for t, p in completed)
            fn = sum(t == label and p != label for t, p in completed)
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        attempts = [row for key, row in evidence["attempts"].items() if key.endswith("_" + method)]
        result = {"method": method, "assigned": 6, "completed": len(completed),
                  "correct": sum(t == p for t, p in zip(truth, predictions)), "macro_f1_completed": mean(f1),
                  "model_requests": sum(r["model_requests"] for r in attempts) if attempts else None,
                  "tool_calls": sum(r["tool_calls"] for r in attempts) if attempts else None,
                  "mean_seconds": mean(r["duration_seconds"] for r in attempts) if attempts else None}
        saved = evidence["comparisons"][method]
        assert result["correct"] == saved["correct_predictions_with_determinate_truth"]
        assert result["completed"] == saved["completed_predictions_on_determinate_truth"]
        assert abs(result["macro_f1_completed"] - saved["macro_f1_completed_fixed_three_labels"]) < 1e-12
        methods.append(result)
    assert sum(r["model_requests"] or 0 for r in methods) == evidence["model_requests"] == 51
    assert sum(r["tool_calls"] or 0 for r in methods) == evidence["tool_calls"] == 132
    native = [{"model": r["model"], "metric": r["metric"], "benefit_pp": 100 * (r["value"] - r["observed"])}
              for r in evidence["signed_effects"]
              if r["case"] == "C01" and r["arm"] == "rules" and r["metric"] in ("macro_f1", "average_precision", "roc_auc")]
    assert len(native) == 9
    return {"dataset_means": task_means, "severity_summary": severity, "adult_examples": examples,
            "native_diabetes": native, "assessor_summary": methods,
            "readiness_cases": [{"case": r["case"]["id"], "truth": r["primary_truth"]["label"], "predictions": r["predictions"]} for r in cases]}


def markdown(summary):
    lines = ["# Reproduced manuscript summaries", "", "Calculated only from the bundled saved measurements. No new observations or model fits.", "",
             "## Missingness (Table II)", "", "| Dataset | Rate | Mean benefit (pp) | Sample SD (pp) | Positive / negative |",
             "|---|---:|---:|---:|---:|"]
    for r in summary["severity_summary"]:
        lines.append(f"| {r['dataset']} | {100*r['severity']:.0f}% | {r['mean_benefit_pp']:+.3f} | {r['sample_sd_pp']:.3f} | {r['positive']} / {r['negative']} |")
    lines += ["", "## Declared Adult examples (Table III)", "", "| Defect | Unit | Detection F1 | Benefit (pp) | Recovery |", "|---|---|---:|---:|---:|"]
    for r in summary["adult_examples"]:
        b = "Invalid" if r["benefit_pp"] is None else f"{r['benefit_pp']:+.3f}"
        recovery = "Undefined" if r["recovery"] is None else f"{r['recovery']:.3f}"
        lines.append(f"| {r['family']} | {r['unit']} | {r['detection_f1']:.3f} | {b} | {recovery} |")
    lines += ["", "## Native Diabetes (Table IV)", "", "| Learner | Metric | Benefit (pp) |", "|---|---|---:|"]
    lines += [f"| {r['model']} | {r['metric']} | {r['benefit_pp']:+.3f} |" for r in summary["native_diabetes"]]
    lines += ["", "## Readiness and completion (Table V)", "", "| Assessor | Completed | Correct / assigned | F1 completed | Requests | Tools | Mean seconds |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["assessor_summary"]:
        seconds = "—" if r["mean_seconds"] is None else f"{r['mean_seconds']:.1f}"
        lines.append(f"| {r['method']} | {r['completed']}/6 | {r['correct']}/6 | {r['macro_f1_completed']:.3f} | {r['model_requests'] if r['model_requests'] is not None else '—'} | {r['tool_calls'] if r['tool_calls'] is not None else '—'} | {seconds} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    args = parser.parse_args()
    summary = summarize()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "tables.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"verified": True, "missingness_draws": 27, "readiness_cases": 6, "output": str(args.output)}))


if __name__ == "__main__":
    main()
