import time
from dataclasses import dataclass, asdict, replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

torch.set_default_dtype(torch.float32)
torch.manual_seed(1234)
np.random.seed(1234)

C0 = 299_792_458.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class FullDomainPINNConfig:
    # Physics
    f0: float = 150e9
    eps_r1: float = 1.0
    eps_r3: float = 11.7
    mu_r: float = 1.0
    Ei: float = 1.0

    # Domain
    left_wavelengths: float = 1.0
    right_wavelengths: float = 1.0

    # Network / training
    layers: tuple[int, ...] = (1, 64, 64, 64, 64, 2)
    epochs: int = 8000
    lr: float = 1e-3
    n_f_total: int = 1200

    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 50.0
    w_energy: float = 1.0

    # Logging / plotting
    print_every: int = 200
    n_plot: int = 1500
    plot_results: bool = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def complex_abs2(z: torch.Tensor) -> torch.Tensor:
    return z.real.square() + z.imag.square()


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return complex_abs2(z).mean()


def sample_uniform(low: float, high: float, n: int, device: torch.device) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, 1, device=device)


def to_complex_torch(x: complex | float, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.complex64, device=device)


def compute_optimal_layer(
    f0: float,
    eps_r1: float = 1.0,
    eps_r3: float = 11.7,
    mu_r: float = 1.0,
) -> tuple[float, float]:
    """
    Quarter-wave AR coating:
        eps_r2 = sqrt(eps_r1 * eps_r3)
        n2 = sqrt(eps_r2 * mu_r)
        d = c0 / (4 f0 n2)
    """
    eps_r2 = np.sqrt(eps_r1 * eps_r3)
    n2 = np.sqrt(eps_r2 * mu_r)
    d = C0 / (4.0 * f0 * n2)
    return float(n2), float(d)


def solve_single_layer_analytic(
    f: float,
    n1: float,
    n2: float,
    n3: float,
    d: float,
    Ei: complex = 1.0,
) -> dict[str, complex]:
    """
    Analytic reference solution for comparison only.
    Convention:
        x < 0:   E1 = Ei exp(-j k1 x) + Er exp(+j k1 x)
        0<=x<=d: E2 = A exp(-j k2 x) + B exp(+j k2 x)
        x > d:   E3 = Et exp(-j k3 x)
    """
    k0 = 2.0 * np.pi * f / C0
    k1, k2, k3 = k0 * n1, k0 * n2, k0 * n3

    e2m = np.exp(-1j * k2 * d)
    e2p = np.exp(1j * k2 * d)
    e3m = np.exp(-1j * k3 * d)

    M = np.array(
        [
            [-1.0,      1.0,        1.0,        0.0],
            [ k1,       k2,        -k2,         0.0],
            [ 0.0,      e2m,        e2p,       -e3m],
            [ 0.0,   k2 * e2m,  -k2 * e2p,  -k3 * e3m],
        ],
        dtype=np.complex128,
    )
    b = np.array([Ei, k1 * Ei, 0.0, 0.0], dtype=np.complex128)

    Er, A, B, Et = np.linalg.solve(M, b)

    r = Er / Ei
    t = Et / Ei
    R = np.abs(r) ** 2
    T = (n3 / n1) * np.abs(t) ** 2

    return {
        "Er": Er,
        "A": A,
        "B": B,
        "Et": Et,
        "r": r,
        "t": t,
        "R": R,
        "T": T,
    }


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

    x = np.asarray(x)
    E = np.zeros_like(x, dtype=np.complex128)

    left = x < 0.0
    coating = (x >= 0.0) & (x <= d)
    right = x > d

    E[left] = Ei * np.exp(-1j * k1 * x[left]) + Er * np.exp(1j * k1 * x[left])
    E[coating] = A * np.exp(-1j * k2 * x[coating]) + B * np.exp(1j * k2 * x[coating])
    E[right] = Et * np.exp(-1j * k3 * x[right])

    return E


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class FCNComplex(nn.Module):
    def __init__(self, layers: tuple[int, ...]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.float()
        for layer in self.linears[:-1]:
            a = self.activation(layer(a))
        return self.linears[-1](a)

    def field_complex(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return torch.complex(out[:, 0:1], out[:, 1:2])


class ScatteringCoeffs(nn.Module):
    """
    We still learn r and t, but only to define boundary conditions and
    diagnostics. The field itself is represented by the global NN.
    """
    def __init__(self):
        super().__init__()
        self.r_re = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.r_im = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.t_re = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.t_im = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

    @property
    def r(self) -> torch.Tensor:
        return torch.complex(self.r_re, self.r_im)

    @property
    def t(self) -> torch.Tensor:
        return torch.complex(self.t_re, self.t_im)


# ---------------------------------------------------------------------
# Derivatives / material profile
# ---------------------------------------------------------------------

def field_and_derivatives(
    model: FCNComplex,
    x_hat: torch.Tensor,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x_hat.clone().detach().requires_grad_(True)
    out = model(x_req)

    e_re = out[:, 0:1]
    e_im = out[:, 1:2]

    de_re = autograd.grad(
        e_re, x_req,
        grad_outputs=torch.ones_like(e_re),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    de_im = autograd.grad(
        e_im, x_req,
        grad_outputs=torch.ones_like(e_im),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    d2e_re = autograd.grad(
        de_re, x_req,
        grad_outputs=torch.ones_like(de_re),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    d2e_im = autograd.grad(
        de_im, x_req,
        grad_outputs=torch.ones_like(de_im),
        create_graph=create_graph,
    )[0]

    e = torch.complex(e_re, e_im)
    de = torch.complex(de_re, de_im)
    d2e = torch.complex(d2e_re, d2e_im)
    return e, de, d2e


def refractive_index_piecewise(x_phys: torch.Tensor, d: float, n1: float, n2: float, n3: float) -> torch.Tensor:
    """
    Piecewise-constant refractive index profile over the whole domain.
    """
    x = x_phys[:, 0]
    n = torch.full_like(x, fill_value=n3)
    n = torch.where(x < 0.0, torch.full_like(x, fill_value=n1), n)
    n = torch.where((x >= 0.0) & (x <= d), torch.full_like(x, fill_value=n2), n)
    return n.unsqueeze(1)


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

def full_domain_pde_loss(
    model: FCNComplex,
    x_phys: torch.Tensor,
    d: float,
    n1: float,
    n2: float,
    n3: float,
    l_ref: float,
) -> torch.Tensor:
    """
    Solve on normalized coordinate x_hat = x / l_ref.
    Then:
        d²E/dx_hat² + (2*pi*n(x))² E = 0
    over the full domain.
    """
    x_hat = x_phys / l_ref
    e, _, d2e = field_and_derivatives(model, x_hat, create_graph=True)
    n_of_x = refractive_index_piecewise(x_phys, d, n1, n2, n3)
    residual = d2e + (2.0 * np.pi * n_of_x) ** 2 * e
    return complex_mse(residual)


def compute_losses(
    model: FCNComplex,
    scat: ScatteringCoeffs,
    cfg: FullDomainPINNConfig,
    collocation: torch.Tensor,
    phys: dict[str, float],
) -> dict[str, torch.Tensor]:
    n1, n2, n3 = phys["n1"], phys["n2"], phys["n3"]
    k1, k3 = phys["k1"], phys["k3"]
    d = phys["d"]
    l_ref = phys["l_ref"]
    x_left = phys["x_left"]
    x_right = phys["x_right"]

    j = to_complex_torch(1j, DEVICE)
    ei = to_complex_torch(cfg.Ei, DEVICE)

    # PDE loss over the whole domain
    l_pde = full_domain_pde_loss(model, collocation, d, n1, n2, n3, l_ref)

    # Boundary points
    xL = torch.tensor([[x_left]], dtype=torch.float32, device=DEVICE)
    xR = torch.tensor([[x_right]], dtype=torch.float32, device=DEVICE)

    # NN field and derivatives at boundaries
    eL, deL_hat, _ = field_and_derivatives(model, xL / l_ref, create_graph=True)
    eR, deR_hat, _ = field_and_derivatives(model, xR / l_ref, create_graph=True)

    deL = deL_hat / l_ref
    deR = deR_hat / l_ref

    # Left boundary: known incident + unknown reflection
    eL_target = ei * (
        torch.exp(-j * k1 * xL) + scat.r * torch.exp(j * k1 * xL)
    )
    deL_target = ei * (
        -j * k1 * torch.exp(-j * k1 * xL) + j * k1 * scat.r * torch.exp(j * k1 * xL)
    )

    # Right boundary: outgoing transmitted wave only
    eR_target = ei * scat.t * torch.exp(-j * k3 * xR)
    deR_target = -j * k3 * eR_target

    l_bc = (
        complex_mse(eL - eL_target)
        + complex_mse(deL - deL_target)
        + complex_mse(eR - eR_target)
        + complex_mse(deR - deR_target)
    )

    # Energy conservation
    energy_balance = complex_abs2(scat.r) + (n3 / n1) * complex_abs2(scat.t) - 1.0
    l_energy = energy_balance.square().mean()

    l_total = cfg.w_pde * l_pde + cfg.w_bc * l_bc + cfg.w_energy * l_energy

    return {
        "total": l_total,
        "pde": l_pde,
        "bc": l_bc,
        "energy": l_energy,
    }


# ---------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------

def run_full_domain_pinn(cfg: FullDomainPINNConfig) -> dict[str, Any]:
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

    x_f = sample_uniform(x_left, x_right, cfg.n_f_total, DEVICE)

    model = FCNComplex(cfg.layers).to(DEVICE)
    scat = ScatteringCoeffs().to(DEVICE)

    # optional analytic initialization of r,t only
    amps0 = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d, Ei=cfg.Ei)
    with torch.no_grad():
        scat.r_re.copy_(torch.tensor(np.real(amps0["r"]), dtype=torch.float32, device=DEVICE))
        scat.r_im.copy_(torch.tensor(np.imag(amps0["r"]), dtype=torch.float32, device=DEVICE))
        scat.t_re.copy_(torch.tensor(np.real(amps0["t"]), dtype=torch.float32, device=DEVICE))
        scat.t_im.copy_(torch.tensor(np.imag(amps0["t"]), dtype=torch.float32, device=DEVICE))

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(scat.parameters()),
        lr=cfg.lr,
    )

    loss_hist = {"total": [], "pde": [], "bc": [], "energy": []}

    model.train()
    scat.train()

    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
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
                f"bc={losses['bc'].item():.3e}, "
                f"en={losses['energy'].item():.3e}"
            )

    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f} seconds")

    # Evaluation
    model.eval()
    scat.eval()

    x_grid = torch.linspace(x_left, x_right, cfg.n_plot, device=DEVICE).unsqueeze(1)
    with torch.inference_mode():
        e_pred = model.field_complex(x_grid / l_ref)

    r_pinn = complex(scat.r.detach().cpu().numpy().item())
    t_pinn = complex(scat.t.detach().cpu().numpy().item())
    R_pinn = float(np.abs(r_pinn) ** 2)
    T_pinn = float((n3 / n1) * np.abs(t_pinn) ** 2)

    amps_an = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d, Ei=cfg.Ei)
    r_an = complex(amps_an["r"])
    t_an = complex(amps_an["t"])
    R_an = float(amps_an["R"])
    T_an = float(amps_an["T"])

    x_np = x_grid.detach().cpu().numpy().squeeze()
    e_pred_np = e_pred.detach().cpu().numpy().squeeze()
    e_an_np = analytic_field(x_np, cfg.f0, n1, n2, n3, d, cfg.Ei, amps_an)

    rel_l2_field = float(
        np.linalg.norm(e_pred_np - e_an_np) / np.linalg.norm(e_an_np)
    )

    print("\n=== Full-domain PINN Metrics (no ansatz in regions 1,2,3) ===")
    print(f"n1={n1:.6f}, n2={n2:.6f}, n3={n3:.6f}")
    print(f"d_opt={d:.6e} m")
    print(f"PINN: r={r_pinn:.6f}, t={t_pinn:.6f}, R={R_pinn:.3e}, T={T_pinn:.3e}, R+T={R_pinn+T_pinn:.6f}")
    print(f"ANLT: r={r_an:.6f}, t={t_an:.6f}, R={R_an:.3e}, T={T_an:.3e}, R+T={R_an+T_an:.6f}")
    print(f"|r_pinn-r_an| = {abs(r_pinn-r_an):.3e}")
    print(f"|t_pinn-t_an| = {abs(t_pinn-t_an):.3e}")
    print(f"relative field L2 error = {rel_l2_field:.3e}")

    if cfg.plot_results:
        plt.figure(figsize=(10, 7))
        plt.plot(x_np, np.abs(e_an_np), label="|E| Analytic", linewidth=2)
        plt.plot(x_np, np.abs(e_pred_np), "--", label="|E| Full-domain PINN", linewidth=1.5)
        plt.axvline(0.0, color="k", linestyle=":", linewidth=1)
        plt.axvline(d, color="k", linestyle=":", linewidth=1)
        plt.xlabel("x [m]")
        plt.ylabel("|E(x)|")
        plt.title("Field Magnitude")
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.plot(x_np, np.unwrap(np.angle(e_an_np)), label="Phase Analytic", linewidth=2)
        plt.plot(x_np, np.unwrap(np.angle(e_pred_np)), "--", label="Phase Full-domain PINN", linewidth=1.5)
        plt.axvline(0.0, color="k", linestyle=":", linewidth=1)
        plt.axvline(d, color="k", linestyle=":", linewidth=1)
        plt.xlabel("x [m]")
        plt.ylabel("Phase [rad]")
        plt.title("Field Phase")
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.plot(loss_hist["total"], label="Total Loss")
        plt.plot(loss_hist["pde"], label="PDE Loss")
        plt.plot(loss_hist["bc"], label="Boundary Loss")
        plt.plot(loss_hist["energy"], label="Energy Loss")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "config": asdict(cfg),
        "n1": float(n1),
        "n2": float(n2),
        "n3": float(n3),
        "d_opt_m": float(d),
        "r_pinn": r_pinn,
        "t_pinn": t_pinn,
        "R_pinn": R_pinn,
        "T_pinn": T_pinn,
        "energy_sum_pinn": float(R_pinn + T_pinn),
        "r_analytic": r_an,
        "t_analytic": t_an,
        "R_analytic": R_an,
        "T_analytic": T_an,
        "energy_sum_analytic": float(R_an + T_an),
        "abs_error_r": float(abs(r_pinn - r_an)),
        "abs_error_t": float(abs(t_pinn - t_an)),
        "relative_field_l2_error": rel_l2_field,
        "train_time_s": float(train_time),
        "final_loss_total": float(loss_hist["total"][-1]),
        "final_loss_pde": float(loss_hist["pde"][-1]),
        "final_loss_bc": float(loss_hist["bc"][-1]),
        "final_loss_energy": float(loss_hist["energy"][-1]),
        "loss_history": loss_hist,
    }


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = FullDomainPINNConfig()
    if config:
        cfg = replace(cfg, **config)
    return run_full_domain_pinn(cfg)


if __name__ == "__main__":
    main()