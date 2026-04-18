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

    # Network and training
    layers: tuple[int, ...] = (1, 50, 50, 50, 50, 2)
    epochs: int = 5000
    lr: float = 1e-3
    n_collocation_coating: int = 500

    # Loss weights
    w_pde: float = 1.0
    w_if: float = 50.0
    w_energy: float = 2.0

    # Dynamic loss weighting
    dynamic_loss_weights: bool = False
    dynamic_weight_ema: float = 0.95
    dynamic_weight_min_ratio: float = 0.2
    dynamic_weight_max_ratio: float = 5.0

    # Dynamic collocation sampling
    dynamic_collocation: bool = False
    collocation_resample_every: int = 10
    collocation_hard_ratio: float = 0.3
    collocation_hard_start_epoch: int = 1000
    collocation_candidate_factor: int = 5

    # Learning-rate scheduler
    use_lr_scheduler: bool = False
    scheduler_type: str = "cosine"  # "cosine" | "step"
    scheduler_eta_min: float = 1e-5
    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.5

    # Second-order optimizer (Adam -> LBFGS)
    use_lbfgs: bool = False
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 300
    lbfgs_history_size: int = 100
    lbfgs_line_search_fn: str = "strong_wolfe"

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


class DynamicLossWeights:
    def __init__(self, cfg: PINNConfig):
        self.names = ("pde", "if", "energy")
        self.base = {
            "pde": float(cfg.w_pde),
            "if": float(cfg.w_if),
            "energy": float(cfg.w_energy),
        }
        self.current = self.base.copy()
        self.ema = cfg.dynamic_weight_ema
        self.min_ratio = cfg.dynamic_weight_min_ratio
        self.max_ratio = cfg.dynamic_weight_max_ratio

    def weights(self) -> dict[str, float]:
        return self.current.copy()

    def update(self, losses: dict[str, torch.Tensor]) -> dict[str, float]:
        values = {
            name: max(losses[name].item(), 1e-12)
            for name in self.names
        }

        mean_loss = sum(values.values()) / len(values)

        new_weights = {}
        for name in self.names:
            ratio = mean_loss / values[name]
            ratio = np.clip(ratio, self.min_ratio, self.max_ratio)   # optional clipping
            new_weights[name] = self.base[name] * ratio

        # optional smoothing
        for name in self.names:
            self.current[name] = (
                self.ema * self.current[name]
                + (1 - self.ema) * new_weights[name]
            )

        return self.current


def default_loss_weights(cfg: PINNConfig) -> dict[str, float]:
    return {"pde": float(cfg.w_pde), "if": float(cfg.w_if), "energy": float(cfg.w_energy)}


def pde_residual_abs2(model: "FCNComplex", x: torch.Tensor, n2: float, l_ref: float) -> torch.Tensor:
    with torch.enable_grad():
        e, _, d2e = field_and_derivatives(model, x / l_ref)
        residual = d2e + (2.0 * np.pi * n2) ** 2 * e
    return complex_abs2(residual).detach().squeeze(1)


def sample_collocation_points(
    model: "FCNComplex | None",
    cfg: PINNConfig,
    phys: dict[str, float],
    epoch: int,
) -> torch.Tensor:
    d = phys["d"]
    n_total = cfg.n_collocation_coating

    x = lhs(1, n_total) * d
    x = torch.tensor(x, dtype=torch.float32, device=device)

    if not cfg.dynamic_collocation:
        return x

    # Number of hard points to insert
    n_hard = int(cfg.collocation_hard_ratio * n_total)

    # Add hard points only after a chosen epoch
    if model is not None and epoch >= cfg.collocation_hard_start_epoch:
        n_candidates = max(n_total, int(cfg.collocation_candidate_factor * n_total))

        # Large candidate pool
        candidates = lhs(1, n_candidates) * d
        candidates = torch.tensor(candidates, dtype=torch.float32, device=device)

        # Score candidates by PDE residual
        residual_score = pde_residual_abs2(model, candidates, phys["n2"], phys["l_ref"])

        # Pick the hardest ones
        topk = torch.topk(
            residual_score,
            k=min(n_hard, candidates.shape[0]),
            largest=True
        ).indices
        hard_points = candidates[topk]

        # Keep some uniform points, replace the rest by hard points
        n_keep = n_total - hard_points.shape[0]
        if n_keep > 0:
            keep_idx = torch.randperm(x.shape[0], device=device)[:n_keep]
            x = torch.cat([x[keep_idx], hard_points], dim=0)
        else:
            x = hard_points

    # Shuffle points
    x = x[torch.randperm(x.shape[0], device=device)]

    return x


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: PINNConfig):
    if not cfg.use_lr_scheduler:
        return None

    scheduler_type = cfg.scheduler_type
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.scheduler_eta_min,
        )
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler_step_size,
            gamma=cfg.scheduler_gamma,
        )

    raise ValueError(f"Unsupported scheduler_type='{cfg.scheduler_type}'")


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
        y = x.to(torch.float32)
        for layer in self.linears[:-1]:
            y = self.activation(layer(y))
        return self.linears[-1](y)

    def complex_field(self, x: torch.Tensor) -> torch.Tensor:
        y = self(x)
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

def field_and_derivatives(model: FCNComplex, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)
    y = model(x)
    e_re, e_im = y[:, :1], y[:, 1:2]

    def grad(u):
        return autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

    de_re, de_im = grad(e_re), grad(e_im)
    d2e_re, d2e_im = grad(de_re), grad(de_im)

    e = torch.complex(e_re, e_im)
    de = torch.complex(de_re, de_im)
    d2e = torch.complex(d2e_re, d2e_im)
    
    return e, de, d2e


def coating_pde_loss(model: FCNComplex, x: torch.Tensor, n2: float, l_ref: float) -> torch.Tensor:
    x_hat = x / l_ref  # l_ref is the normalization length
    e, _, d2e = field_and_derivatives(model, x_hat)
    residual = d2e + (2.0 * np.pi * n2) ** 2 * e
    return complex_mse(residual)


def compute_losses(
    model: FCNComplex,
    scat: ScatteringCoeffs,
    cfg: PINNConfig,
    collocation: torch.Tensor,
    phys: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    k0, k1, k3 = phys["k0"], phys["k1"], phys["k3"]
    n1, n2, n3 = phys["n1"], phys["n2"], phys["n3"]
    d, l_ref = phys["d"], phys["l_ref"]

    j = torch.tensor(1j, dtype=torch.complex64, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex64, device=device)

    # PDE loss
    l_pde = coating_pde_loss(model, collocation, n2, l_ref)

    # Define the interface locations
    x0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    xd = torch.tensor([[d]], dtype=torch.float32, device=device)

    e2_0, de2_dx_hat_0, _ = field_and_derivatives(model, x0 / l_ref)    # E2(0), dE2/dx_hat(0)
    e2_d, de2_dx_hat_d, _ = field_and_derivatives(model, xd / l_ref)    # E2(d), dE2/dx_hat(d)

    de2_0 = de2_dx_hat_0 / l_ref                                        # dE2/dx(0)
    de2_d = de2_dx_hat_d / l_ref                                        # dE2/dx(d)

    e1_0 = ei * (1.0 + scat.r)
    de1_0 = ei * (-j * k1 + j * k1 * scat.r)

    e3_d = ei * scat.t * torch.exp(-j * k3 * d)
    de3_d = -j * k3 * e3_d

    # Interface loss
    l_if = (
        complex_mse(e2_0 - e1_0)
        + complex_mse((de2_0 - de1_0) / k0)
        + complex_mse(e2_d - e3_d)
        + complex_mse((de2_d - de3_d) / k0)
    )

    # Energy conservation loss
    energy_balance = complex_abs2(scat.r) + (n3 / n1) * complex_abs2(scat.t) - 1.0
    l_energy = energy_balance.square().mean()

    if weights is None:
        weights = default_loss_weights(cfg)
    l_total = weights["pde"] * l_pde + weights["if"] * l_if + weights["energy"] * l_energy

    return {"total": l_total, "pde": l_pde, "if": l_if, "energy": l_energy,}


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

    j = torch.tensor(1j, dtype=torch.complex64, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex64, device=device)

    x = x_grid
    out = torch.zeros((x.shape[0], 1), dtype=torch.complex64, device=x.device)

    left = x[:, 0] < 0.0
    coat = (x[:, 0] >= 0.0) & (x[:, 0] <= d)
    right = x[:, 0] > d

    with torch.inference_mode():
        out[left] = ei * (torch.exp(-j * k1 * x[left]) + scat.r * torch.exp(j * k1 * x[left]))
        out[coat] = model.complex_field(x[coat] / l_ref)
        out[right] = ei * scat.t * torch.exp(-j * k3 * x[right])

    return out


# ---------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------

def run_forward_pinn(cfg: PINNConfig) -> dict[str, Any]:
    n1 = np.sqrt(cfg.eps_r1 * cfg.mu_r)
    n3 = np.sqrt(cfg.eps_r3 * cfg.mu_r)
    n2, d = compute_optimal_layer(cfg.f0, cfg.eps_r1, cfg.eps_r3, cfg.mu_r)

    d = d * 1.00 # pertubation

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

    params = list(model.parameters()) + list(scat.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = build_lr_scheduler(optimizer, cfg)

    loss_hist = {"total": [], "pde": [], "if": [], "energy": [], "lr": []}
    weight_hist = {"pde": [], "if": [], "energy": []}

    weight_adapter = DynamicLossWeights(cfg) if cfg.dynamic_loss_weights else None
    current_weights = weight_adapter.weights() if weight_adapter is not None else default_loss_weights(cfg)

    start_time = time.time()
    model.train()
    scat.train()

    resample_every = max(1, int(cfg.collocation_resample_every))
    x_collocation = sample_collocation_points(
        model if cfg.dynamic_collocation else None,
        cfg,
        phys,
        epoch=1,
    )

    # Adam training
    for ep in range(1, cfg.epochs + 1):
        if ep > 1 and ep % resample_every == 0:
            x_collocation = sample_collocation_points(
                model if cfg.dynamic_collocation else None,
                cfg,
                phys,
                epoch=ep,
            )

        losses = compute_losses(model, scat, cfg, x_collocation, phys, weights=current_weights)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if weight_adapter is not None:
            current_weights = weight_adapter.update(losses)

        for key in ("total", "pde", "if", "energy"):
            loss_hist[key].append(losses[key].item())
        loss_hist["lr"].append(float(optimizer.param_groups[0]["lr"]))

        for key in ("pde", "if", "energy"):
            weight_hist[key].append(float(current_weights[key]))

        if ep == 1 or ep % cfg.print_every == 0:
            print(
                f"iter={ep:5d}, "
                f"total={losses['total'].item():.3e}, "
                f"pde={losses['pde'].item():.3e}, "
                f"if={losses['if'].item():.3e}, "
                f"en={losses['energy'].item():.3e}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}, "
                f"w_pde={current_weights['pde']:.2f}, "
                f"w_if={current_weights['if']:.2f}, "
                f"w_en={current_weights['energy']:.2f}"
            )

    if cfg.use_lbfgs:
        print("\nStarting LBFGS refinement...")
        x_lbfgs = sample_collocation_points(
            model if cfg.dynamic_collocation else None,
            cfg,
            phys,
            epoch=cfg.epochs + 1,
        )

        lbfgs = torch.optim.LBFGS(
            params,
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size,
            line_search_fn=cfg.lbfgs_line_search_fn,
        )

        def closure():
            lbfgs.zero_grad()
            lbfgs_losses = compute_losses(model, scat, cfg, x_lbfgs, phys, weights=current_weights)
            lbfgs_losses["total"].backward()
            return lbfgs_losses["total"]

        lbfgs.step(closure)
        lbfgs_eval = compute_losses(model, scat, cfg, x_lbfgs, phys, weights=current_weights)
        for key in ("total", "pde", "if", "energy"):
            loss_hist[key].append(lbfgs_eval[key].item())
        for key in ("pde", "if", "energy"):
            weight_hist[key].append(float(current_weights[key]))

        print(
            "LBFGS done: "
            f"total={lbfgs_eval['total'].item():.3e}, "
            f"pde={lbfgs_eval['pde'].item():.3e}, "
            f"if={lbfgs_eval['if'].item():.3e}, "
            f"en={lbfgs_eval['energy'].item():.3e}"
        )

    train_time = time.time() - start_time
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

        e2_0, de2_dx_hat_0, _ = field_and_derivatives(model, x0 / l_ref)    # E2(0), dE2/dx_hat(0)
        e2_d, de2_dx_hat_d, _ = field_and_derivatives(model, xd / l_ref)    # E2(d), dE2/dx_hat(d)

    de2_0 = de2_dx_hat_0 / l_ref                                            # dE2/dx(0)
    de2_d = de2_dx_hat_d / l_ref                                            # dE2/dx(d)

    e1_0 = cfg.Ei * (1.0 + r_pinn)                                          # E1(0)
    de1_0 = cfg.Ei * (-1j * k1 + 1j * k1 * r_pinn)                          # dE1/dx(0)
    e3_d = cfg.Ei * t_pinn * np.exp(-1j * k3 * d)                            # E3(d)
    de3_d = -1j * k3 * e3_d                                                 # dE3/dx(d)

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
    print(f"PINN: r={r_pinn:.6f}, t={t_pinn:.6f}, R={R_pinn:.3e}, T={T_pinn:.3e}, R+T={R_pinn+T_pinn:.6e}")
    print(f"ANLT: r={r_an:.6f}, t={t_an:.6f}, R={R_an:.3e}, T={T_an:.3e}, R+T={R_an+T_an:.6e}")
    print(f"Interface continuity error = {if_l2:.3e}")
    print(f"Perturbation check: R(d_opt)={R_an:.3e}, R(1.2*d_opt)={R_pert:.3e}")

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

    if cfg.use_lr_scheduler:
        plt.figure(figsize=(10, 4))
        plt.plot(loss_hist["lr"], label="Learning Rate")
        plt.xlabel("Iteration")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.show()

    if cfg.dynamic_loss_weights:
        plt.figure(figsize=(10, 4))
        plt.plot(weight_hist["pde"], label="w_pde")
        plt.plot(weight_hist["if"], label="w_if")
        plt.plot(weight_hist["energy"], label="w_energy")
        plt.xlabel("Iteration")
        plt.ylabel("Weight")
        plt.title("Dynamic Loss Weights")
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
        "interface_l2_error": float(if_l2),
        "loss_history": loss_hist,
    }


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = PINNConfig()
    if config:
        cfg = replace(cfg, **config)
    return run_forward_pinn(cfg)


if __name__ == "__main__":
    main()