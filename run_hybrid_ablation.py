"""Run a reproducible ablation study for the optimized hybrid PINN.

The experiment keeps the physical problem, network, loss weights, Adam budget,
and evaluation grid fixed. It changes only the optimization component(s) named
by each configuration and repeats every configuration for multiple seeds.
"""

from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from hybrid_optimization import main as run_hybrid_pinn


@dataclass(frozen=True)
class AblationConfiguration:
    name: str
    label: str
    dynamic_loss_weights: bool = False
    dynamic_collocation: bool = False
    use_lr_scheduler: bool = False
    use_lbfgs: bool = False


CONFIGURATIONS = (
    AblationConfiguration("baseline", "Baseline"),
    AblationConfiguration(
        "optimizer_refinement",
        "Scheduler + L-BFGS",
        use_lr_scheduler=True,
        use_lbfgs=True,
    ),
    AblationConfiguration(
        "dynamic_weighting",
        "Dynamic weighting",
        dynamic_loss_weights=True,
    ),
    AblationConfiguration(
        "adaptive_sampling",
        "Adaptive sampling",
        dynamic_collocation=True,
    ),
    AblationConfiguration(
        "complete_optimized",
        "Complete optimized",
        dynamic_loss_weights=True,
        dynamic_collocation=True,
        use_lr_scheduler=True,
        use_lbfgs=True,
    ),
)


TRIAL_FIELDS = (
    "configuration",
    "label",
    "seed",
    "dynamic_loss_weights",
    "dynamic_collocation",
    "use_lr_scheduler",
    "use_lbfgs",
    "field_relative_l2_error",
    "interface_l2_error",
    "reflectance",
    "transmittance",
    "energy_error",
    "training_time_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234, 2345, 3456],
        help="Random seeds used for every configuration.",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=300)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=[configuration.name for configuration in CONFIGURATIONS],
        default=[configuration.name for configuration in CONFIGURATIONS],
        help="Subset of configurations to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ablation_results",
    )
    return parser.parse_args()


def write_trials(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS, delimiter=",")
        writer.writeheader()
        writer.writerows(rows)


def summarize_trials(
    configurations: list[AblationConfiguration],
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary = []
    metric_names = (
        "field_relative_l2_error",
        "interface_l2_error",
        "reflectance",
        "energy_error",
        "training_time_s",
    )

    for configuration in configurations:
        matching = [row for row in rows if row["configuration"] == configuration.name]
        item: dict[str, object] = {
            "configuration": configuration.name,
            "label": configuration.label,
            "runs": len(matching),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in matching], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)

    return summary


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    means = np.asarray([row["field_relative_l2_error_mean"] for row in rows], dtype=float)
    standard_deviations = np.asarray(
        [row["field_relative_l2_error_std"] for row in rows], dtype=float
    )

    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.errorbar(
        positions,
        means,
        yerr=standard_deviations,
        fmt="o",
        capsize=5,
        markersize=7,
    )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel(r"Relative field $L_2$ error")
    axis.grid(True, which="both", linestyle=":", alpha=0.6)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = set(args.configurations)
    configurations = [
        configuration
        for configuration in CONFIGURATIONS
        if configuration.name in selected_names
    ]

    trials: list[dict[str, object]] = []
    trials_path = args.output_dir / "hybrid_ablation_trials.csv"

    for configuration in configurations:
        for seed in args.seeds:
            print("\n" + "=" * 72)
            print(f"Configuration: {configuration.label} | seed={seed}")
            print("=" * 72)

            checkpoint_name = f"{configuration.name}_seed_{seed}.pt"
            result = run_hybrid_pinn(
                {
                    "seed": seed,
                    "epochs": args.epochs,
                    "dynamic_loss_weights": configuration.dynamic_loss_weights,
                    "dynamic_collocation": configuration.dynamic_collocation,
                    "use_lr_scheduler": configuration.use_lr_scheduler,
                    "use_lbfgs": configuration.use_lbfgs,
                    "lbfgs_max_iter": args.lbfgs_max_iter,
                    "print_every": max(args.epochs, 1),
                    "force_retrain": False,
                    "checkpoint_path": f"checkpoints/ablation/{checkpoint_name}",
                }
            )

            row = {
                "configuration": configuration.name,
                "label": configuration.label,
                "seed": seed,
                "dynamic_loss_weights": configuration.dynamic_loss_weights,
                "dynamic_collocation": configuration.dynamic_collocation,
                "use_lr_scheduler": configuration.use_lr_scheduler,
                "use_lbfgs": configuration.use_lbfgs,
                "field_relative_l2_error": result["field_relative_l2_error"],
                "interface_l2_error": result["interface_l2_error"],
                "reflectance": result["R_pinn"],
                "transmittance": result["T_pinn"],
                "energy_error": abs(result["energy_sum_pinn"] - 1.0),
                "training_time_s": result["training_time_s"],
            }
            trials.append(row)

            # Save after every run so completed trials survive an interrupted study.
            write_trials(trials_path, trials)
            plt.close("all")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_trials(configurations, trials)
    summary_path = args.output_dir / "hybrid_ablation_summary.csv"
    figure_path = args.output_dir / "hybrid_ablation_l2_error.png"
    write_summary(summary_path, summary)
    plot_summary(figure_path, summary)

    print("\nAblation study completed.")
    print(f"Individual trials: {trials_path}")
    print(f"Summary:          {summary_path}")
    print(f"Error plot:       {figure_path}")


if __name__ == "__main__":
    main()
