import time
from dataclasses import asdict, dataclass, replace
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

torch.set_default_dtype(torch.float32)
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

    # Plot domain
    left_wavelengths: float = 1.0
    right_wavelengths: float = 1.0

    # Network / training
    layers: tuple[int, ...] = (1, 64, 64, 64, 64, 2)
    epochs: int = 5000
    lr: float = 1e-3
    #init_rt_from_analytic: bool = False
    n_collocation_coating: int = 400

    # Loss weights
    w_pde: float = 1.0
    w_if: float = 50.0
    w_energy: float = 2.0

    # Output
    print_every: int = 200
    n_plot: int = 1200
    freq_probe_points: int = 101
    plot_results: bool = True


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
# PINN modules
# ---------------------------------------------------------------------

class FCNComplex(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep input dtype consistent with layer weights (float32/float32).
        y = x.to(dtype=self.linears[0].weight.dtype)
        for layer in self.linears[:-1]:
            y = self.activation(layer(y))
        return self.linears[-1](y)

    def complex_field(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return torch.complex(y[:, 0], y[:, 1]).unsqueeze(1)


class ScatteringCoeffs(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_re = nn.Parameter(torch.tensor(0.0))
        self.r_im = nn.Parameter(torch.tensor(0.0))
        self.t_re = nn.Parameter(torch.tensor(0.0))
        self.t_im = nn.Parameter(torch.tensor(0.0))

    @property
    def r(self) -> torch.Tensor:
        return torch.complex(self.r_re, self.r_im)

    @property
    def t(self) -> torch.Tensor:
        return torch.complex(self.t_re, self.t_im)


# ---------------------------------------------------------------------
# Derivatives / losses
# ---------------------------------------------------------------------

def field_and_derivatives(
    model: FCNComplex,
    x: torch.Tensor,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)
    y = model(x)

    e_re, e_im = y[:, :1], y[:, 1:2]

    def grad(u):
        return autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=create_graph,
            retain_graph=True,
        )[0]

    de_re, de_im = grad(e_re), grad(e_im)
    d2e_re, d2e_im = grad(de_re), grad(de_im)

    e = torch.complex(e_re, e_im)
    de = torch.complex(de_re, de_im)
    d2e = torch.complex(d2e_re, d2e_im)
    return e, de, d2e


def coating_pde_loss(model: FCNComplex, x_phys: torch.Tensor, n2: float, l_ref: float) -> torch.Tensor:
    x_hat = x_phys / l_ref
    e, _, d2e = field_and_derivatives(model, x_hat, create_graph=True)
    residual = d2e + (2.0 * np.pi * n2) ** 2 * e
    return complex_mse(residual)


def compute_losses(
    model: FCNComplex,
    scat: ScatteringCoeffs,
    cfg: PINNConfig,
    collocation: torch.Tensor,
    phys: dict[str, float],
) -> dict[str, torch.Tensor]:
    k0, k1, k3 = phys["k0"], phys["k1"], phys["k3"]
    n1, n3 = phys["n1"], phys["n3"]
    n2, d, l_ref = phys["n2"], phys["d"], phys["l_ref"]

    #j = to_torch_complex(1j, device)
    #ei = to_torch_complex(cfg.Ei, device)
    j = torch.tensor(1j, dtype=torch.complex64, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex64, device=device)


    l_pde = coating_pde_loss(model, collocation, n2, l_ref)

    x0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    xd = torch.tensor([[d]], dtype=torch.float32, device=device)

    e2_0, de2_0_hat, _ = field_and_derivatives(model, x0 / l_ref, create_graph=True)
    e2_d, de2_d_hat, _ = field_and_derivatives(model, xd / l_ref, create_graph=True)

    de2_0 = de2_0_hat / l_ref
    de2_d = de2_d_hat / l_ref

    e1_0 = ei * (1.0 + scat.r)
    de1_0 = ei * (-j * k1 + j * k1 * scat.r)

    e3_d = ei * scat.t
    de3_d = -j * k3 * e3_d

    l_if = (
        complex_mse(e2_0 - e1_0)
        + complex_mse((de2_0 - de1_0) / k0)
        + complex_mse(e2_d - e3_d)
        + complex_mse((de2_d - de3_d) / k0)
    )

    energy_balance = complex_abs2(scat.r) + (n3 / n1) * complex_abs2(scat.t) - 1.0
    l_energy = energy_balance.square().mean()

    l_total = cfg.w_pde * l_pde + cfg.w_if * l_if + cfg.w_energy * l_energy

    return {
        "total": l_total,
        "pde": l_pde,
        "if": l_if,
        "energy": l_energy,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_piecewise_field(
    model: FCNComplex,
    scat: ScatteringCoeffs,
    cfg: PINNConfig,
    x_grid: torch.Tensor,
    phys: dict[str, float],
) -> torch.Tensor:
    k1, k3 = phys["k1"], phys["k3"]
    d, l_ref = phys["d"], phys["l_ref"]

    # j = to_torch_complex(1j, x_grid.device)
    # ei = to_torch_complex(cfg.Ei, x_grid.device)
    j = torch.tensor(1j, dtype=torch.complex64, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex64, device=device)

    x = x_grid
    out = torch.zeros((x.shape[0], 1), dtype=torch.complex64, device=x.device)

    left = x[:, 0] < 0.0
    coat = (x[:, 0] >= 0.0) & (x[:, 0] <= d)
    right = x[:, 0] > d

    with torch.inference_mode():
        if left.any():
            xx = x[left]
            out[left] = ei * (torch.exp(-j * k1 * xx) + scat.r * torch.exp(j * k1 * xx))

        if coat.any():
            out[coat] = model.complex_field(x[coat] / l_ref)

        if right.any():
            xx = x[right]
            out[right] = ei * scat.t * torch.exp(-j * k3 * (xx - d))

    return out


# ---------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------

def run_forward_pinn(cfg: PINNConfig) -> dict[str, Any]:
    n1 = np.sqrt(cfg.eps_r1 * cfg.mu_r)
    n3 = np.sqrt(cfg.eps_r3 * cfg.mu_r)
    n2, d = compute_optimal_layer(cfg.f0, cfg.eps_r1, cfg.eps_r3, cfg.mu_r)

    lambda1 = C0 / (cfg.f0 * n1)
    lambda3 = C0 / (cfg.f0 * n3)
    l_ref = C0 / cfg.f0

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
        "l_ref": float(l_ref),
        "x_left": float(x_left),
        "x_right": float(x_right),
    }

    model = FCNComplex(cfg.layers).to(device)
    scat = ScatteringCoeffs().to(device)

    # if cfg.init_rt_from_analytic:
    #     amps0 = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d)
    #     with torch.no_grad():
    #         scat.r_re.copy_(torch.tensor(np.real(amps0["r"]), device=device))
    #         scat.r_im.copy_(torch.tensor(np.imag(amps0["r"]), device=device))
    #         scat.t_re.copy_(torch.tensor(np.real(amps0["t"]), device=device))
    #         scat.t_im.copy_(torch.tensor(np.imag(amps0["t"]), device=device))

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(scat.parameters()),
        lr=cfg.lr,
    )

    loss_hist = {"total": [], "pde": [], "if": [], "energy": []}

    t0 = time.time()
    model.train()
    scat.train()

    x_f = lhs(1, cfg.n_collocation_coating) * d
    x_f = torch.tensor(x_f, dtype=torch.float32, device=device)

    for ep in range(1, cfg.epochs + 1):
        if ep % 10 ==0:
            x_f = lhs(1, cfg.n_collocation_coating) * d
            x_f = torch.tensor(x_f, dtype=torch.float32, device=device)
            
        losses = compute_losses(model, scat, cfg, x_f, phys)

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
                f"if={losses['if'].item():.3e}, "
                f"en={losses['energy'].item():.3e}"
            )

    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f} seconds")

    # Evaluation
    model.eval()
    scat.eval()

    x_grid = torch.linspace(x_left, x_right, cfg.n_plot, device=device).unsqueeze(1)
    e_pred = evaluate_piecewise_field(model, scat, cfg, x_grid, phys)

    r_pinn = complex(scat.r.detach().cpu().numpy().item())
    t_pinn = complex(scat.t.detach().cpu().numpy().item())
    R_pinn = abs(r_pinn) ** 2
    T_pinn = (n3 / n1) * abs(t_pinn) ** 2

    amps_an = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d)
    r_an = complex(amps_an["r"])
    t_an = complex(amps_an["t"])
    R_an = float(amps_an["R"])
    T_an = float(amps_an["T"])

    # Interface diagnostic
    with torch.enable_grad():
        x0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        xd = torch.tensor([[d]], dtype=torch.float32, device=device)

        e2_0, de2_0_hat, _ = field_and_derivatives(model, x0 / l_ref, create_graph=True)
        e2_d, de2_d_hat, _ = field_and_derivatives(model, xd / l_ref, create_graph=True)

    de2_0 = de2_0_hat / l_ref
    de2_d = de2_d_hat / l_ref

    e1_0 = cfg.Ei * (1.0 + r_pinn)
    de1_0 = cfg.Ei * (-1j * k1 + 1j * k1 * r_pinn)
    e3_d = cfg.Ei * t_pinn
    de3_d = -1j * k3 * e3_d

    if_l2 = float(
        abs(complex(e2_0.detach().cpu().numpy().item()) - e1_0) ** 2
        + abs(complex((de2_0.detach().cpu().numpy().item() - de1_0) / k0)) ** 2
        + abs(complex(e2_d.detach().cpu().numpy().item()) - e3_d) ** 2
        + abs(complex((de2_d.detach().cpu().numpy().item() - de3_d) / k0)) ** 2
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
    print(f"PINN: r={r_pinn:.6f}, t={t_pinn:.6f}, R={R_pinn:.3e}, T={T_pinn:.3e}, R+T={R_pinn+T_pinn:.6f}")
    print(f"ANLT: r={r_an:.6f}, t={t_an:.6f}, R={R_an:.3e}, T={T_an:.3e}, R+T={R_an+T_an:.6f}")
    print(f"|R_pinn - R_an| = {abs(R_pinn - R_an):.3e}")
    print(f"Interface continuity error = {if_l2:.3e}")
    print(f"Perturbation check: R(d_opt)={R_an:.3e}, R(1.2*d_opt)={R_pert:.3e}")

    if cfg.plot_results:
        plt.figure(figsize=(10, 7))
        plot_curve(
            x_np, np.abs(e_an_np),
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
            x_np, np.unwrap(np.angle(e_an_np)),
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
            ("if", "Interface Loss"),
            ("energy", "Energy Loss"),
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
        "config": asdict(cfg),
        "n1": float(n1),
        "n2_opt": float(n2),
        "n3": float(n3),
        "d_opt_m": float(d),
        "r_pinn": r_pinn,
        "t_pinn": t_pinn,
        "R_pinn": float(R_pinn),
        "T_pinn": float(T_pinn),
        "energy_sum_pinn": float(R_pinn + T_pinn),
        "r_analytic": r_an,
        "t_analytic": t_an,
        "R_analytic": float(R_an),
        "T_analytic": float(T_an),
        "energy_sum_analytic": float(R_an + T_an),
        "R_abs_error": float(abs(R_pinn - R_an)),
        "interface_l2_error": float(if_l2),
        "R_perturbed_d_1p2": float(R_pert),
        "train_time_s": float(train_time),
        "final_loss_total": float(loss_hist["total"][-1]),
        "final_loss_pde": float(loss_hist["pde"][-1]),
        "final_loss_if": float(loss_hist["if"][-1]),
        "final_loss_energy": float(loss_hist["energy"][-1]),
        "loss_history": loss_hist,
    }


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = PINNConfig()
    if config:
        cfg = replace(cfg, **config)
    return run_forward_pinn(cfg)


if __name__ == "__main__":
    main()
