import time
from dataclasses import dataclass, replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from pydoe import lhs

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

C0 = 299_792_458.0


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class PINNConfig:
    # Physics
    f0: float = 150e9
    eps_r1: float = 1.0
    eps_r3: float = 11.7
    mu_r: float = 1.0
    Ei: float = 1.0

    # Spatial domain
    left_wavelengths: float = 1.0
    right_wavelengths: float = 1.0

    # Network / training
    layers: tuple[int, ...] = (1, 50, 50, 50, 50, 2)
    epochs: int = 15000
    lr: float = 1e-3
    n_collocation: int = 1200
    resample_every: int = 10

    # Fixed loss weights
    w_pde: float = 1.0
    w_bc: float = 20.0
    w_energy: float = 5.0
    w_extract: float = 0.0

    # Output
    print_every: int = 200
    n_plot: int = 1500
    freq_probe_points: int = 101


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def complex_abs2(z: torch.Tensor) -> torch.Tensor:
    return z.real.square() + z.imag.square()


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return complex_abs2(z).mean()


def plot_curve(x, y, *, label, xlabel, ylabel, title, style="-", vlines=None):
    plt.plot(x, y, style, label=label)
    if vlines:
        for xv in vlines:
            plt.axvline(xv, color="k", linestyle=":", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()


def sample_interval_lhs(n_points: int, x_min: float, x_max: float) -> torch.Tensor:
    points = x_min + lhs(1, n_points) * (x_max - x_min)
    return torch.tensor(points, dtype=torch.float64, device=device)


# def split_point_counts(total_points: int, weights: list[float]) -> list[int]:
#     weights_arr = np.asarray(weights, dtype=np.float64)
#     active = weights_arr > 0.0
#     counts = np.zeros_like(weights_arr, dtype=int)

#     if total_points <= 0 or not np.any(active):
#         return counts.tolist()

#     n_active = int(np.count_nonzero(active))
#     remaining = int(total_points)

#     if remaining >= n_active:
#         counts[active] = 1
#         remaining -= n_active

#     if remaining <= 0:
#         return counts.tolist()

#     normalized = np.zeros_like(weights_arr, dtype=np.float64)
#     normalized[active] = weights_arr[active] / weights_arr[active].sum()

#     raw_extra = remaining * normalized
#     extra = np.floor(raw_extra).astype(int)
#     counts += extra

#     leftover = remaining - int(extra.sum())
#     if leftover > 0:
#         order = np.argsort(-(raw_extra - extra))
#         for idx in order[:leftover]:
#             counts[idx] += 1

#     return counts.tolist()


# ---------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------

def compute_optimal_layer(
    f0: float,
    eps_r1: float = 1.0,
    eps_r3: float = 11.7,
    mu_r: float = 1.0,
) -> tuple[float, float]:
    eps_r2 = np.sqrt(eps_r1 * eps_r3)
    d = C0 / (4.0 * f0 * np.sqrt(eps_r2 * mu_r))
    n2 = np.sqrt(eps_r2 * mu_r)
    return float(n2), float(d)


def solve_single_layer_analytic(
    f: float,
    n1: float,
    n2: float,
    n3: float,
    d: float,
    Ei: complex = 1.0,
) -> dict[str, complex]:
    k0 = 2.0 * np.pi * f / C0
    k1, k2, k3 = k0 * n1, k0 * n2, k0 * n3

    e2m = np.exp(-1j * k2 * d)
    e2p = np.exp(1j * k2 * d)
    e3m = np.exp(-1j * k3 * d)

    M = np.array([[-1.0,      1.0,        1.0,        0.0],
                  [ k1,       k2,        -k2,         0.0],
                  [ 0.0,      e2m,        e2p,       -e3m],
                  [ 0.0,   k2 * e2m,  -k2 * e2p,  -k3 * e3m]], dtype=np.complex128)

    b = np.array([Ei, k1 * Ei, 0.0, 0.0], dtype=np.complex128)

    Er, A, B, Et = np.linalg.solve(M, b)

    r = Er / Ei
    t = Et / Ei
    R = np.abs(r) ** 2
    T = (n3 / n1) * np.abs(t) ** 2

    return {"Er": Er, "A": A, "B": B, "Et": Et, "r": r, "t": t, "R": R, "T": T}


def analytic_field(
    x: np.ndarray,
    f: float,
    n1: float,
    n2: float,
    n3: float,
    d: float,
    Ei: complex,
    amps: dict[str, complex],
) -> np.ndarray:
    k0 = 2.0 * np.pi * f / C0
    k1, k2, k3 = k0 * n1, k0 * n2, k0 * n3

    Er = amps["Er"]
    A = amps["A"]
    B = amps["B"]
    Et = amps["Et"]

    E = np.zeros_like(x, dtype=np.complex128)

    m1 = x < 0.0
    m2 = (x >= 0.0) & (x <= d)
    m3 = x > d

    E[m1] = Ei * np.exp(-1j * k1 * x[m1]) + Er * np.exp(1j * k1 * x[m1])
    E[m2] = A * np.exp(-1j * k2 * x[m2]) + B * np.exp(1j * k2 * x[m2])
    E[m3] = Et * np.exp(-1j * k3 * x[m3])

    return E


# ---------------------------------------------------------------------
# PINN model
# ---------------------------------------------------------------------

class FCNComplex(nn.Module):
    def __init__(self, layers: tuple[int, ...]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float64)
        for layer in self.linears[:-1]:
            y = self.activation(layer(y))
        return self.linears[-1](y)
    
    def complex_field(self, x: torch.Tensor) -> torch.Tensor:
        y = self(x)
        return torch.complex(y[:, 0], y[:, 1]).unsqueeze(1)
    

# ---------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------

def refractive_index_profile(x: torch.Tensor, phys: dict[str, float]) -> torch.Tensor:
    n = x.new_full(x.shape, phys["n2"])
    n = torch.where(x < 0.0, x.new_full(x.shape, phys["n1"]), n)
    n = torch.where(x > phys["d"], x.new_full(x.shape, phys["n3"]), n)
    return n


# def domain_segments(phys: dict[str, float]) -> list[tuple[float, float, float]]:
#     return [
#         (phys["x_left"], 0.0, phys["n1"]),
#         (0.0, phys["d"], phys["n2"]),
#         (phys["d"], phys["x_right"], phys["n3"]),
#     ]


# def sample_domain_points_by_optical_length(n_points: int, phys: dict[str, float]) -> torch.Tensor:
#     segments = domain_segments(phys)
#     optical_weights = [max((x1 - x0) * n_seg, 0.0) for x0, x1, n_seg in segments]
#     counts = split_point_counts(n_points, optical_weights)

#     points = [
#         sample_interval_lhs(count, x0, x1)
#         for count, (x0, x1, _) in zip(counts, segments)
#         if count > 0
#     ]

#     if not points:
#         return torch.empty((0, 1), dtype=torch.float64, device=device)

#     x = torch.cat(points, dim=0)
#     if x.shape[0] > 1:
#         x = x[torch.randperm(x.shape[0], device=device)]
#     return x


def total_field_and_derivatives(
    model: nn.Module,
    x: torch.Tensor,
    cfg: PINNConfig,
    phys: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)

    j = torch.tensor(1j, dtype=torch.complex128, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex128, device=device)

    # Neural correction
    y = model(x / phys["l_ref"])
    u = torch.complex(y[:, 0:1], y[:, 1:2])

    # Known incoming wave
    e_inc = ei * torch.exp(-j * phys["k1"] * x)

    # Total field
    e_total = e_inc + u

    def grad(u_scalar: torch.Tensor) -> torch.Tensor:
        return autograd.grad(
            u_scalar,
            x,
            grad_outputs=torch.ones_like(u_scalar),
            create_graph=True,
            retain_graph=True,
        )[0]

    de_re = grad(e_total.real)
    de_im = grad(e_total.imag)
    d2e_re = grad(de_re)
    d2e_im = grad(de_im)

    de_total = torch.complex(de_re, de_im)
    d2e_total = torch.complex(d2e_re, d2e_im)

    return e_total, de_total, d2e_total


# ---------------------------------------------------------------------
# Boundary diagnostics and losses
# ---------------------------------------------------------------------

def boundary_diagnostics(
    model: nn.Module,
    cfg: PINNConfig,
    phys: dict[str, float],
) -> dict[str, torch.Tensor]:
    j = torch.tensor(1j, dtype=torch.complex128, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex128, device=device)

    x_left = torch.tensor([[phys["x_left"]]], dtype=torch.float64, device=device)
    x_right = torch.tensor([[phys["x_right"]]], dtype=torch.float64, device=device)

    e_left, de_left, _ = total_field_and_derivatives(model, x_left, cfg, phys)
    e_right, de_right, _ = total_field_and_derivatives(model, x_right, cfg, phys)

    k0 = phys["k0"]
    k1 = phys["k1"]
    k3 = phys["k3"]
    n1 = phys["n1"]
    n3 = phys["n3"]

    # Total-field ABCs
    bc_left = (de_left - j * k1 * e_left + 2.0 * j * k1 * ei * torch.exp(-j * k1 * x_left)) / k0
    bc_right = (de_right + j * k3 * e_right) / k0

    exp_in = torch.exp(-j * k1 * x_left)
    exp_ref = torch.exp(j * k1 * x_left)
    exp_tr = torch.exp(-j * k3 * x_right)

    r_field = (e_left / ei - exp_in) / exp_ref
    r_deriv = (de_left / (j * k1 * ei) + exp_in) / exp_ref

    t_field = e_right / (ei * exp_tr)
    t_deriv = de_right / (-j * k3 * ei * exp_tr)

    r = 0.5 * (r_field + r_deriv)
    t = 0.5 * (t_field + t_deriv)

    energy_balance = complex_abs2(r) + (n3 / n1) * complex_abs2(t) - 1.0

    return {
        "bc_left": bc_left,
        "bc_right": bc_right,
        "r_field": r_field,
        "r_deriv": r_deriv,
        "t_field": t_field,
        "t_deriv": t_deriv,
        "r": r,
        "t": t,
        "energy_balance": energy_balance,
    }


def compute_losses(
    model: nn.Module,
    cfg: PINNConfig,
    collocation: torch.Tensor,
    phys: dict[str, float],
) -> dict[str, torch.Tensor]:
    e, _, d2e = total_field_and_derivatives(model, collocation, cfg, phys)
    n_profile = refractive_index_profile(collocation, phys)

    # Physical-coordinate Helmholtz equation:
    # d²E/dx² + (k0 n)^2 E = 0
    residual = d2e + (phys["k0"] * n_profile).square() * e
    l_pde = complex_mse(residual / (phys["k0"] ** 2))

    boundary = boundary_diagnostics(model, cfg, phys)
    l_bc = complex_mse(boundary["bc_left"]) + complex_mse(boundary["bc_right"])
    l_energy = boundary["energy_balance"].square().mean()
    l_extract = complex_mse(boundary["r_field"] - boundary["r_deriv"]) + complex_mse(
        boundary["t_field"] - boundary["t_deriv"]
    )

    l_total = (
        cfg.w_pde * l_pde
        + cfg.w_bc * l_bc
        + cfg.w_energy * l_energy
        + cfg.w_extract * l_extract
    )

    return {
        "total": l_total,
        "pde": l_pde,
        "bc": l_bc,
        "energy": l_energy,
        "extract": l_extract,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_field(
    model: nn.Module,
    x_grid: torch.Tensor,
    cfg: PINNConfig,
    phys: dict[str, float],
) -> torch.Tensor:
    with torch.enable_grad():
        e_total, _, _ = total_field_and_derivatives(model, x_grid, cfg, phys)
    return e_total.detach()


def extract_scattering_coeffs(
    model: nn.Module,
    cfg: PINNConfig,
    phys: dict[str, float],
) -> tuple[complex, complex]:
    with torch.enable_grad():
        boundary = boundary_diagnostics(model, cfg, phys)

    r = complex(boundary["r"].detach().cpu().numpy().item())
    t = complex(boundary["t"].detach().cpu().numpy().item())
    return r, t


# ---------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------

def run_forward_pinn(cfg: PINNConfig) -> dict[str, Any]:
    n1 = np.sqrt(cfg.eps_r1 * cfg.mu_r)
    n3 = np.sqrt(cfg.eps_r3 * cfg.mu_r)
    n2, d = compute_optimal_layer(cfg.f0, cfg.eps_r1, cfg.eps_r3, cfg.mu_r)

    lambda1 = C0 / (cfg.f0 * n1)
    lambda3 = C0 / (cfg.f0 * n3)

    x_left = -cfg.left_wavelengths * lambda1
    x_right = d + cfg.right_wavelengths * lambda3

    k0 = 2.0 * np.pi * cfg.f0 / C0
    k1, k2, k3 = k0 * n1, k0 * n2, k0 * n3

    phys = {
        "n1": float(n1),
        "n2": float(n2),
        "n3": float(n3),
        "k0": float(k0),
        "k1": float(k1),
        "k2": float(k2),
        "k3": float(k3),
        "d": float(d),
        "l_ref": float(C0 / cfg.f0),
        "x_left": float(x_left),
        "x_right": float(x_right),
    }

    model = FCNComplex(cfg.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    loss_hist = {"total": [], "pde": [], "bc": [], "energy": [], "extract": []}

    start_time = time.time()
    model.train()

    #x_collocation = sample_domain_points_by_optical_length(cfg.n_collocation, phys)
    x_collocation = sample_interval_lhs(cfg.n_collocation, x_left, x_right)

    for ep in range(1, cfg.epochs + 1):
        if ep > 1 and ep % cfg.resample_every == 0:
            #x_collocation = sample_domain_points_by_optical_length(cfg.n_collocation, phys)
            x_collocation = sample_interval_lhs(cfg.n_collocation, x_left, x_right)

        losses = compute_losses(model, cfg, x_collocation, phys)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        for key in loss_hist:
            loss_hist[key].append(losses[key].item())

        if ep == 1 or ep % cfg.print_every == 0:
            print(
                f"iter={ep:5d}, "
                f"total={losses['total'].item():.3e}, "
                f"pde={losses['pde'].item():.3e}, "
                f"bc={losses['bc'].item():.3e}, "
                f"en={losses['energy'].item():.3e}, "
                f"ext={losses['extract'].item():.3e}"
            )

    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    model.eval()
    x_grid = torch.linspace(x_left, x_right, cfg.n_plot, device=device).unsqueeze(1)
    e_pred = evaluate_field(model, x_grid, cfg, phys)

    r_pinn, t_pinn = extract_scattering_coeffs(model, cfg, phys)
    R_pinn = abs(r_pinn) ** 2
    T_pinn = (n3 / n1) * abs(t_pinn) ** 2

    amps_an = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d, cfg.Ei)
    r_an = complex(amps_an["r"])
    t_an = complex(amps_an["t"])
    R_an = float(amps_an["R"])
    T_an = float(amps_an["T"])

    with torch.enable_grad():
        boundary = boundary_diagnostics(model, cfg, phys)

    bc_l2 = float(
        abs(complex(boundary["bc_left"].detach().cpu().numpy().item())) ** 2
        + abs(complex(boundary["bc_right"].detach().cpu().numpy().item())) ** 2
    )
    extract_l2 = float(
        abs(complex((boundary["r_field"] - boundary["r_deriv"]).detach().cpu().numpy().item())) ** 2
        + abs(complex((boundary["t_field"] - boundary["t_deriv"]).detach().cpu().numpy().item())) ** 2
    )

    d_pert = 1.2 * d
    R_pert = float(solve_single_layer_analytic(cfg.f0, n1, n2, n3, d_pert)["R"])

    f_probe = np.linspace(0.8 * cfg.f0, 1.2 * cfg.f0, cfg.freq_probe_points)
    R_probe = np.array([solve_single_layer_analytic(f, n1, n2, n3, d)["R"] for f in f_probe])

    x_np = x_grid.detach().cpu().numpy().squeeze()
    e_pred_np = e_pred.detach().cpu().numpy().squeeze()
    e_an_np = analytic_field(x_np, cfg.f0, n1, n2, n3, d, cfg.Ei, amps_an)

    print("\n=== Forward PINN Metrics ===")
    print(f"n1={n1:.6f}, n2_opt={n2:.6f}, n3={n3:.6f}")
    print(f"d_opt={d:.6e} m")
    print(f"PINN: r={r_pinn:.6f}, t={t_pinn:.6f}, R={R_pinn:.3e}, T={T_pinn:.3e}, R+T={R_pinn + T_pinn:.6e}")
    print(f"ANLT: r={r_an:.6f}, t={t_an:.6f}, R={R_an:.3e}, T={T_an:.3e}, R+T={R_an + T_an:.6e}")
    print(f"Boundary condition error = {bc_l2:.3e}")
    print(f"Extraction consistency error = {extract_l2:.3e}")
    print(f"Perturbation check: R(d_opt)={R_an:.3e}, R(1.2*d_opt)={R_pert:.3e}")

    plt.figure(figsize=(10, 7))
    plot_curve(
        x_np,
        np.abs(e_an_np),
        label="|E| Analytic",
        xlabel="x [m]",
        ylabel="|E(x)|",
        title="Field Magnitude: PINN vs Analytic",
        vlines=[0.0, d],
    )
    plt.plot(x_np, np.abs(e_pred_np), "--", label="|E| PINN")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 7))
    plot_curve(
        x_np,
        np.unwrap(np.angle(e_an_np)),
        label="Phase Analytic",
        xlabel="x [m]",
        ylabel="Phase [rad]",
        title="Field Phase: PINN vs Analytic",
        vlines=[0.0, d],
    )
    plt.plot(x_np, np.unwrap(np.angle(e_pred_np)), "--", label="Phase PINN")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 7))
    for key, label in [
        ("total", "Total Loss"),
        ("pde", "PDE Loss"),
        ("bc", "Boundary Loss"),
        ("energy", "Energy Loss"),
        ("extract", "Extraction Consistency"),
    ]:
        plt.plot(loss_hist[key], label=label)
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
    plt.scatter([cfg.f0 / 1e9], [R_pinn], color="red", s=40, label="PINN @ f0")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Reflectance R")
    plt.title("Frequency Characteristic")
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "n1": float(n1),
        "n2_opt": float(n2),
        "n3": float(n3),
        "d_opt_m": float(d),
        "r_pinn": r_pinn,
        "t_pinn": t_pinn,
        "R_pinn": float(R_pinn),
        "T_pinn": float(T_pinn),
        "energy_sum_pinn": float(R_pinn + T_pinn),
        "boundary_l2_error": float(bc_l2),
        "extract_l2_error": float(extract_l2),
        "loss_history": loss_hist,
    }


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = PINNConfig()
    if config:
        cfg = replace(cfg, **config)
    return run_forward_pinn(cfg)


if __name__ == "__main__":
    main()