"""Run a compact off-design frequency study for the two forward PINNs.

The coating refractive index and thickness are always designed at ``f0``.
Only the operating frequency is changed during the sweep. This distinction is
essential: recomputing the layer thickness at every frequency would test a new
optimal coating each time rather than the same coating away from its design
frequency.
"""

from __future__ import annotations

import argparse
import csv
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from full_domain import main as run_full_domain
from hybrid_optimization import main as run_hybrid_optimized
from physics import compute_optimal_layer, solve_single_layer_analytic


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FREQUENCY_RATIOS = (0.90, 0.95, 1.00, 1.05, 1.10)
MODEL_LABELS = {
    "hybrid_optimized": "Optimized hybrid PINN",
    "full_domain": "Full-domain PINN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the optimized hybrid and full-domain PINNs at a small set "
            "of operating frequencies around a fixed coating design frequency."
        )
    )
    parser.add_argument(
        "--design-frequency-ghz",
        type=float,
        default=150.0,
        help="Coating design frequency in GHz (default: 150).",
    )
    parser.add_argument(
        "--frequency-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_FREQUENCY_RATIOS),
        help="Operating frequencies as ratios of f0 (default: 0.90 0.95 1.00 1.05 1.10).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234],
        help="Random seeds. One seed is the lightweight default; use three for final statistics.",
    )
    parser.add_argument(
        "--models",
        choices=tuple(MODEL_LABELS),
        nargs="+",
        default=list(MODEL_LABELS),
        help="PINN formulations to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "frequency_sweep_results",
        help="Directory for CSV files and figures.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even when the frequency-specific checkpoint already exists.",
    )
    args = parser.parse_args()

    if not 2 <= len(args.frequency_ratios) <= 8:
        parser.error("Choose between 2 and 8 operating frequencies.")
    if any(ratio <= 0.0 for ratio in args.frequency_ratios):
        parser.error("All frequency ratios must be positive.")
    if args.design_frequency_ghz <= 0.0:
        parser.error("The design frequency must be positive.")

    args.frequency_ratios = sorted(set(args.frequency_ratios))
    args.seeds = list(dict.fromkeys(args.seeds))
    args.models = list(dict.fromkeys(args.models))
    return args


def checkpoint_name(
    model_name: str,
    design_frequency_hz: float,
    operating_frequency_hz: float,
    seed: int,
) -> str:
    def frequency_tag(value_hz: float) -> str:
        return f"{value_hz / 1e9:.6f}".rstrip("0").rstrip(".").replace(".", "p")

    design_tag = frequency_tag(design_frequency_hz)
    operating_tag = frequency_tag(operating_frequency_hz)
    return (
        "checkpoints/frequency_sweep/"
        f"{model_name}_design{design_tag}GHz_f{operating_tag}GHz_seed{seed}.pt"
    )


def result_to_row(model_name: str, seed: int, ratio: float, result: dict[str, Any]) -> dict[str, Any]:
    r_pinn = complex(result["r_pinn"])
    t_pinn = complex(result["t_pinn"])
    r_reference = complex(result["r_reference"])
    t_reference = complex(result["t_reference"])
    R_pinn = float(result["R_pinn"])
    T_pinn = float(result["T_pinn"])
    R_reference = float(result["R_reference"])
    T_reference = float(result["T_reference"])

    return {
        "model": model_name,
        "label": MODEL_LABELS[model_name],
        "seed": seed,
        "frequency_ratio": ratio,
        "design_frequency_hz": float(result["design_frequency_hz"]),
        "operating_frequency_hz": float(result["operating_frequency_hz"]),
        "operating_frequency_ghz": float(result["operating_frequency_hz"]) / 1e9,
        "field_relative_l2_error": float(result["field_relative_l2_error"]),
        "r_pinn_real": r_pinn.real,
        "r_pinn_imag": r_pinn.imag,
        "r_reference_real": r_reference.real,
        "r_reference_imag": r_reference.imag,
        "r_absolute_error": abs(r_pinn - r_reference),
        "t_pinn_real": t_pinn.real,
        "t_pinn_imag": t_pinn.imag,
        "t_reference_real": t_reference.real,
        "t_reference_imag": t_reference.imag,
        "t_absolute_error": abs(t_pinn - t_reference),
        "reflectance_pinn": R_pinn,
        "reflectance_reference": R_reference,
        "reflectance_absolute_error": abs(R_pinn - R_reference),
        "transmittance_pinn": T_pinn,
        "transmittance_reference": T_reference,
        "transmittance_absolute_error": abs(T_pinn - T_reference),
        "energy_error": abs(R_pinn + T_pinn - 1.0),
        "interface_l2_error": result.get("interface_l2_error", ""),
        "boundary_l2_error": result.get("boundary_l2_error", ""),
        "extraction_l2_error": result.get("extract_l2_error", ""),
        "training_time_s": float(result["training_time_s"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=",")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "field_relative_l2_error",
        "r_absolute_error",
        "t_absolute_error",
        "reflectance_pinn",
        "reflectance_reference",
        "reflectance_absolute_error",
        "transmittance_pinn",
        "transmittance_reference",
        "transmittance_absolute_error",
        "energy_error",
        "training_time_s",
    )
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), float(row["frequency_ratio"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (model_name, ratio), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        item: dict[str, Any] = {
            "model": model_name,
            "label": MODEL_LABELS[model_name],
            "frequency_ratio": ratio,
            "operating_frequency_ghz": float(group[0]["operating_frequency_ghz"]),
            "runs": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_min"] = float(values.min())
            item[f"{metric}_max"] = float(values.max())
        summary.append(item)
    return summary


def model_rows(rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["model"] == model_name],
        key=lambda row: (float(row["frequency_ratio"]), int(row["seed"])),
    )


def grouped_means(rows: list[dict[str, Any]], model_name: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in model_rows(rows, model_name):
        grouped[float(row["frequency_ratio"])].append(float(row[metric]))
    ratios = np.asarray(sorted(grouped), dtype=float)
    means = np.asarray([np.mean(grouped[ratio]) for ratio in ratios], dtype=float)
    return ratios, means


def plot_scattering(
    path: Path,
    rows: list[dict[str, Any]],
    design_frequency_hz: float,
    frequency_ratios: list[float],
) -> None:
    eps_r1, eps_r3, mu_r = 1.0, 11.7, 1.0
    n1 = np.sqrt(eps_r1 * mu_r)
    n3 = np.sqrt(eps_r3 * mu_r)
    n2, d = compute_optimal_layer(design_frequency_hz, eps_r1, eps_r3, mu_r)
    dense_ratios = np.linspace(min(frequency_ratios), max(frequency_ratios), 201)
    analytic = [
        solve_single_layer_analytic(design_frequency_hz * ratio, n1, n2, n3, d)
        for ratio in dense_ratios
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    axes[0].plot(dense_ratios, [item["R"] for item in analytic], "k-", label="Analytical reference")
    axes[1].plot(dense_ratios, [item["T"] for item in analytic], "k-", label="Analytical reference")

    for model_name in MODEL_LABELS:
        if not any(row["model"] == model_name for row in rows):
            continue
        ratios, reflectance = grouped_means(rows, model_name, "reflectance_pinn")
        _, transmittance = grouped_means(rows, model_name, "transmittance_pinn")
        axes[0].plot(ratios, reflectance, "o--", label=MODEL_LABELS[model_name])
        axes[1].plot(ratios, transmittance, "o--", label=MODEL_LABELS[model_name])

    axes[0].set_ylabel("Reflectance $R$")
    axes[1].set_ylabel("Transmittance $T$")
    for axis in axes:
        axis.set_xlabel(r"Operating frequency $f/f_0$")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_errors(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = (
        ("field_relative_l2_error", r"Relative field $L_2$ error"),
        ("reflectance_absolute_error", r"Absolute reflectance error $|R-R_{ref}|$"),
        ("transmittance_absolute_error", r"Absolute transmittance error $|T-T_{ref}|$"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True)

    for model_index, model_name in enumerate(MODEL_LABELS):
        selected = model_rows(rows, model_name)
        if not selected:
            continue
        color = f"C{model_index}"
        for axis, (metric, _) in zip(axes, metrics):
            ratios, means = grouped_means(rows, model_name, metric)
            axis.plot(ratios, np.maximum(means, 1e-16), "o-", color=color, label=MODEL_LABELS[model_name])
            axis.scatter(
                [float(row["frequency_ratio"]) for row in selected],
                [max(float(row[metric]), 1e-16) for row in selected],
                color=color,
                alpha=0.35,
                s=22,
            )

    for axis, (_, ylabel) in zip(axes, metrics):
        axis.set_yscale("log")
        axis.set_xlabel(r"Operating frequency $f/f_0$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", linestyle=":", alpha=0.6)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_field_snapshot(
    path: Path,
    ratio: float,
    snapshots: dict[tuple[str, float], dict[str, Any]],
) -> None:
    available_models = [name for name in MODEL_LABELS if (name, ratio) in snapshots]
    if not available_models:
        return

    figure, axes = plt.subplots(len(available_models), 2, figsize=(11, 4.2 * len(available_models)), squeeze=False)
    for row_index, model_name in enumerate(available_models):
        snapshot = snapshots[(model_name, ratio)]
        x_mm = np.asarray(snapshot["x_m"]) * 1e3
        prediction = np.asarray(snapshot["field_pinn"])
        reference = np.asarray(snapshot["field_reference"])

        axes[row_index, 0].plot(x_mm, np.abs(reference), "k-", label="Analytical reference")
        axes[row_index, 0].plot(x_mm, np.abs(prediction), "--", label=MODEL_LABELS[model_name])
        axes[row_index, 0].set_ylabel(r"$|E|$")

        axes[row_index, 1].plot(x_mm, np.unwrap(np.angle(reference)), "k-", label="Analytical reference")
        axes[row_index, 1].plot(x_mm, np.unwrap(np.angle(prediction)), "--", label=MODEL_LABELS[model_name])
        axes[row_index, 1].set_ylabel("Phase [rad]")

        for axis in axes[row_index]:
            axis.set_xlabel("Position $x$ [mm]")
            axis.grid(True, linestyle=":", alpha=0.6)
            axis.legend()
            axis.set_title(f"{MODEL_LABELS[model_name]}, $f/f_0={ratio:.2f}$")

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    design_frequency_hz = args.design_frequency_ghz * 1e9
    endpoint_ratios = {min(args.frequency_ratios), 1.0, max(args.frequency_ratios)}
    solvers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "hybrid_optimized": run_hybrid_optimized,
        "full_domain": run_full_domain,
    }

    rows: list[dict[str, Any]] = []
    snapshots: dict[tuple[str, float], dict[str, Any]] = {}
    total_runs = len(args.frequency_ratios) * len(args.seeds) * len(args.models)
    run_number = 0
    print(f"Running {total_runs} trainings at {len(args.frequency_ratios)} frequencies.")
    print(f"Design frequency: {args.design_frequency_ghz:.3f} GHz")
    print(f"Frequency ratios: {args.frequency_ratios}")

    for ratio in args.frequency_ratios:
        operating_frequency_hz = design_frequency_hz * ratio
        for model_name in args.models:
            for seed in args.seeds:
                run_number += 1
                print("\n" + "=" * 72)
                print(
                    f"Run {run_number}/{total_runs}: {MODEL_LABELS[model_name]}, "
                    f"f/f0={ratio:.3f}, f={operating_frequency_hz / 1e9:.3f} GHz, seed={seed}"
                )
                config = {
                    "f0": design_frequency_hz,
                    "operating_frequency": operating_frequency_hz,
                    "seed": seed,
                    "checkpoint_path": checkpoint_name(
                        model_name,
                        design_frequency_hz,
                        operating_frequency_hz,
                        seed,
                    ),
                    "force_retrain": args.force_retrain,
                    "show_plots": False,
                }
                result = solvers[model_name](config)
                rows.append(result_to_row(model_name, seed, ratio, result))

                if ratio in endpoint_ratios and seed == args.seeds[0]:
                    snapshots[(model_name, ratio)] = {
                        "x_m": result["x_m"],
                        "field_pinn": result["field_pinn"],
                        "field_reference": result["field_reference"],
                    }

                plt.close("all")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                write_csv(output_dir / "frequency_sweep_trials.csv", rows)

    summary_rows = summarize(rows)
    write_csv(output_dir / "frequency_sweep_summary.csv", summary_rows)
    plot_scattering(
        output_dir / "frequency_sweep_scattering.png",
        rows,
        design_frequency_hz,
        args.frequency_ratios,
    )
    plot_errors(output_dir / "frequency_sweep_errors.png", rows)
    for ratio in sorted(endpoint_ratios):
        frequency_ghz = design_frequency_hz * ratio / 1e9
        plot_field_snapshot(
            output_dir / f"field_comparison_{frequency_ghz:.1f}GHz.png",
            ratio,
            snapshots,
        )

    print("\nFrequency sweep complete.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()