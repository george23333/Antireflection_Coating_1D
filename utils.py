import matplotlib.pyplot as plt
import numpy as np
import torch


def setup_device_and_seed(seed: int = 1234) -> torch.device:
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    return device


def print_summary(results: dict) -> None:
    print("\n=== Forward PINN Metrics (Analytic Exterior + 1 NN Coating) ===")
    print(f"n1={results['n1']:.6f}, n2_opt={results['n2_opt']:.6f}, n3={results['n3']:.6f}")
    print(f"d_opt={results['d_opt_m']:.6e} m")
    print(
        f"PINN: r={results['r_pinn']:.6f}, t={results['t_pinn']:.6f}, "
        f"R={results['R_pinn']:.3e}, T={results['T_pinn']:.3e}, "
        f"R+T={results['energy_sum_pinn']:.6f}"
    )
    print(
        f"ANLT: r={results['r_analytic']:.6f}, t={results['t_analytic']:.6f}, "
        f"R={results['R_analytic']:.3e}, T={results['T_analytic']:.3e}, "
        f"R+T={results['energy_sum_analytic']:.6f}"
    )
    print(f"|R_pinn - R_an| = {results['R_abs_error']:.3e}")
    print(f"Interface continuity error = {results['interface_l2_error']:.3e}")
    print(
        f"Perturbation check: R(d_opt)={results['R_analytic']:.3e}, "
        f"R(1.2*d_opt)={results['R_perturbed_d_1p2']:.3e}"
    )


def plot_results(results: dict, f0: float) -> None:
    x_np = results["x_plot"]
    e_pred_np = results["e_pred_plot"]
    e_an_np = results["e_analytic_plot"]
    d = results["d_opt_m"]

    loss_total_hist = results["loss_history"]["total"]
    loss_pde_hist = results["loss_history"]["pde"]
    loss_if_hist = results["loss_history"]["interface"]
    loss_energy_hist = results["loss_history"]["energy"]

    f_probe = results["f_probe"]
    R_probe = results["R_probe"]
    R_pinn = results["R_pinn"]

    plt.figure(figsize=(10, 7))
    plt.plot(x_np, np.abs(e_an_np), label="|E| Analytic", linewidth=2)
    plt.plot(x_np, np.abs(e_pred_np), "--", label="|E| PINN", linewidth=1.5)
    plt.axvline(0.0, color="k", linestyle=":", linewidth=1)
    plt.axvline(d, color="k", linestyle=":", linewidth=1)
    plt.xlabel("x [m]")
    plt.ylabel("|E(x)|")
    plt.title("Field Magnitude: PINN vs Analytic")
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(x_np, np.unwrap(np.angle(e_an_np)), label="Phase Analytic", linewidth=2)
    plt.plot(x_np, np.unwrap(np.angle(e_pred_np)), "--", label="Phase PINN", linewidth=1.5)
    plt.axvline(0.0, color="k", linestyle=":", linewidth=1)
    plt.axvline(d, color="k", linestyle=":", linewidth=1)
    plt.xlabel("x [m]")
    plt.ylabel("Phase [rad]")
    plt.title("Field Phase: PINN vs Analytic")
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(loss_total_hist, label="Total Loss")
    plt.plot(loss_pde_hist, label="PDE Loss")
    plt.plot(loss_if_hist, label="Interface Loss")
    plt.plot(loss_energy_hist, label="Energy Loss")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(f_probe / 1e9, R_probe, label="Analytic Reflectance (fixed d_opt)")
    plt.scatter([f0 / 1e9], [R_pinn], color="red", s=40, label="PINN @ f0")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Reflectance R")
    plt.title("Frequency Characteristic")
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()