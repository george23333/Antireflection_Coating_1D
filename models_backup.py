import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from physics import compute_optimal_layer, solve_single_layer_analytic, analytic_field


@dataclass
class ForwardPINNConfig:
    f0: float = 150e9
    eps_r1: float = 1.0
    eps_r3: float = 11.7
    mu_r: float = 1.0
    Ei: float = 1.0

    left_wavelengths: float = 1.0
    right_wavelengths: float = 1.0

    layers: tuple[int, ...] = (1, 64, 64, 64, 64, 2)
    epochs: int = 5000
    lr: float = 1e-3
    init_rt_from_analytic: bool = True

    n_f_coating: int = 400

    w_pde: float = 1.0
    w_if: float = 50.0
    w_energy: float = 2.0

    print_every: int = 200
    n_plot: int = 1200
    freq_probe_points: int = 101


def complex_abs2(z: torch.Tensor) -> torch.Tensor:
    return z.real**2 + z.imag**2


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(complex_abs2(z))


def sample_uniform(low: float, high: float, n: int, device: torch.device) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, 1, device=device)


class FCNComplex(nn.Module):
    def __init__(self, layers: tuple[int, ...]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight, gain=1.0)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.float()
        for lin in self.linears[:-1]:
            a = self.activation(lin(a))
        return self.linears[-1](a)

    def field_complex(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return torch.complex(out[:, 0:1], out[:, 1:2])


class ScatteringCoeffs(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_re = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.r_im = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.t_re = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.t_im = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

    def reflection(self) -> torch.Tensor:
        return torch.complex(self.r_re, self.r_im)

    def transmission(self) -> torch.Tensor:
        return torch.complex(self.t_re, self.t_im)


def field_and_derivatives(
    model: FCNComplex,
    x: torch.Tensor,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.clone().detach().requires_grad_(True)
    out = model(x_req)

    e_re = out[:, 0:1]
    e_im = out[:, 1:2]

    de_re = autograd.grad(
        e_re,
        x_req,
        grad_outputs=torch.ones_like(e_re),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    de_im = autograd.grad(
        e_im,
        x_req,
        grad_outputs=torch.ones_like(e_im),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    d2e_re = autograd.grad(
        de_re,
        x_req,
        grad_outputs=torch.ones_like(de_re),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    d2e_im = autograd.grad(
        de_im,
        x_req,
        grad_outputs=torch.ones_like(de_im),
        create_graph=create_graph,
    )[0]

    e = torch.complex(e_re, e_im)
    de = torch.complex(de_re, de_im)
    d2e = torch.complex(d2e_re, d2e_im)
    return e, de, d2e


def coating_pde_loss(
    model: FCNComplex,
    x_phys: torch.Tensor,
    n2: float,
    l_ref: float,
) -> torch.Tensor:
    x_hat = x_phys / l_ref
    e, _, d2e = field_and_derivatives(model, x_hat, create_graph=True)
    res = d2e + ((2.0 * np.pi * n2) ** 2) * e
    return complex_mse(res)


def compute_losses(
    model_coating: FCNComplex,
    scat: ScatteringCoeffs,
    cfg: ForwardPINNConfig,
    collocation: dict[str, torch.Tensor],
    phys: dict[str, float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n1 = phys["n1"]
    n2 = phys["n2"]
    n3 = phys["n3"]
    k0 = phys["k0"]
    k1 = phys["k1"]
    k3 = phys["k3"]
    d = phys["d"]
    l_ref = phys["l_ref"]

    j = torch.complex(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    ei = torch.complex(
        torch.tensor(cfg.Ei, dtype=torch.float32, device=device),
        torch.tensor(0.0, dtype=torch.float32, device=device),
    )

    r = scat.reflection()
    t = scat.transmission()

    l_pde = coating_pde_loss(model_coating, collocation["coating"], n2, l_ref)

    x0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    e2_0, de2_0_hat, _ = field_and_derivatives(model_coating, x0 / l_ref, create_graph=True)
    de2_0 = de2_0_hat / l_ref

    e1_0 = ei * (1.0 + r)
    de1_0 = ei * (-j * k1 + j * k1 * r)

    xd = torch.tensor([[d]], dtype=torch.float32, device=device)
    e2_d, de2_d_hat, _ = field_and_derivatives(model_coating, xd / l_ref, create_graph=True)
    de2_d = de2_d_hat / l_ref

    e3_d = ei * t
    de3_d = -j * k3 * e3_d

    l_if = (
        complex_mse(e2_0 - e1_0)
        + complex_mse((de2_0 - de1_0) / k0)
        + complex_mse(e2_d - e3_d)
        + complex_mse((de2_d - de3_d) / k0)
    )

    energy_balance = complex_abs2(r) + (n3 / n1) * complex_abs2(t) - 1.0
    l_energy = torch.mean(energy_balance**2)

    l_total = cfg.w_pde * l_pde + cfg.w_if * l_if + cfg.w_energy * l_energy

    return {
        "total": l_total,
        "pde": l_pde,
        "if": l_if,
        "energy": l_energy,
    }


def evaluate_piecewise_field(
    model_coating: FCNComplex,
    scat: ScatteringCoeffs,
    cfg: ForwardPINNConfig,
    x_grid: torch.Tensor,
    phys: dict[str, float],
) -> torch.Tensor:
    k1 = phys["k1"]
    k3 = phys["k3"]
    d = phys["d"]
    l_ref = phys["l_ref"]

    j = torch.complex(torch.tensor(0.0, device=x_grid.device), torch.tensor(1.0, device=x_grid.device))
    ei = torch.complex(
        torch.tensor(cfg.Ei, dtype=torch.float32, device=x_grid.device),
        torch.tensor(0.0, dtype=torch.float32, device=x_grid.device),
    )

    r = scat.reflection()
    t = scat.transmission()

    x = x_grid
    e_out = torch.zeros((x.shape[0], 1), dtype=torch.complex64, device=x.device)

    mask1 = x[:, 0] < 0.0
    mask2 = (x[:, 0] >= 0.0) & (x[:, 0] <= d)
    mask3 = x[:, 0] > d

    with torch.inference_mode():
        if torch.any(mask1):
            x1 = x[mask1]
            e_out[mask1] = ei * (torch.exp(-j * k1 * x1) + r * torch.exp(j * k1 * x1))

        if torch.any(mask2):
            x2 = x[mask2] / l_ref
            e_out[mask2] = model_coating.field_complex(x2)

        if torch.any(mask3):
            x3 = x[mask3]
            e_out[mask3] = ei * t * torch.exp(-j * k3 * (x3 - d))

    return e_out


def run_forward_pinn(cfg: ForwardPINNConfig, device: torch.device) -> dict[str, Any]:
    c0 = 299_792_458.0

    n1 = np.sqrt(cfg.eps_r1 * cfg.mu_r)
    n3 = np.sqrt(cfg.eps_r3 * cfg.mu_r)
    n2, d = compute_optimal_layer(cfg.f0, cfg.eps_r1, cfg.eps_r3, cfg.mu_r)

    lambda1 = c0 / (cfg.f0 * n1)
    lambda3 = c0 / (cfg.f0 * n3)
    l_ref = c0 / cfg.f0

    x_left = -cfg.left_wavelengths * lambda1
    x_right = d + cfg.right_wavelengths * lambda3

    k0 = 2.0 * np.pi * cfg.f0 / c0
    k1, k2, k3 = k0 * n1, k0 * n2, k0 * n3

    collocation = {
        "coating": sample_uniform(0.0, d, cfg.n_f_coating, device),
    }

    model_coating = FCNComplex(cfg.layers).to(device)
    scat = ScatteringCoeffs().to(device)

    if cfg.init_rt_from_analytic:
        amps0 = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d)
        r0 = amps0["r"]
        t0 = amps0["t"]
        with torch.no_grad():
            scat.r_re.copy_(torch.tensor(np.real(r0), dtype=torch.float32, device=device))
            scat.r_im.copy_(torch.tensor(np.imag(r0), dtype=torch.float32, device=device))
            scat.t_re.copy_(torch.tensor(np.real(t0), dtype=torch.float32, device=device))
            scat.t_im.copy_(torch.tensor(np.imag(t0), dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        list(model_coating.parameters()) + list(scat.parameters()),
        lr=cfg.lr,
    )

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

    loss_total_hist = []
    loss_pde_hist = []
    loss_if_hist = []
    loss_energy_hist = []

    model_coating.train()
    scat.train()

    t_start = time.time()
    for ep in range(1, cfg.epochs + 1):
        losses = compute_losses(model_coating, scat, cfg, collocation, phys, device)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        loss_total_hist.append(losses["total"].item())
        loss_pde_hist.append(losses["pde"].item())
        loss_if_hist.append(losses["if"].item())
        loss_energy_hist.append(losses["energy"].item())

        if ep % cfg.print_every == 0 or ep == 1:
            print(
                f"iter={ep:5d}, total={losses['total'].item():.3e}, "
                f"pde={losses['pde'].item():.3e}, "
                f"if={losses['if'].item():.3e}, "
                f"en={losses['energy'].item():.3e}"
            )

    train_time = time.time() - t_start
    print(f"Training time: {train_time:.2f} seconds")

    model_coating.eval()
    scat.eval()

    x_grid = torch.linspace(x_left, x_right, cfg.n_plot, device=device).unsqueeze(1)
    e_pred = evaluate_piecewise_field(model_coating, scat, cfg, x_grid, phys)

    r_pinn = complex(scat.reflection().detach().cpu().numpy().item())
    t_pinn = complex(scat.transmission().detach().cpu().numpy().item())
    R_pinn = float(np.abs(r_pinn) ** 2)
    T_pinn = float((n3 / n1) * np.abs(t_pinn) ** 2)

    with torch.enable_grad():
        x0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        xd = torch.tensor([[d]], dtype=torch.float32, device=device)

        e2_0, de2_0_hat, _ = field_and_derivatives(model_coating, x0 / l_ref, create_graph=True)
        e2_d, de2_d_hat, _ = field_and_derivatives(model_coating, xd / l_ref, create_graph=True)
        de2_0 = de2_0_hat / l_ref
        de2_d = de2_d_hat / l_ref

    e1_0 = cfg.Ei * (1.0 + r_pinn)
    de1_0 = cfg.Ei * (-1j * k1 + 1j * k1 * r_pinn)
    e3_d = cfg.Ei * t_pinn
    de3_d = -1j * k3 * e3_d

    if_l2 = (
        np.abs(complex((e2_0.detach().cpu().numpy().item()) - e1_0)) ** 2
        + np.abs(complex(((de2_0.detach().cpu().numpy().item()) - de1_0) / k0)) ** 2
        + np.abs(complex((e2_d.detach().cpu().numpy().item()) - e3_d)) ** 2
        + np.abs(complex(((de2_d.detach().cpu().numpy().item()) - de3_d) / k0)) ** 2
    )
    if_l2 = float(if_l2)

    amps_analytic = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d)
    r_an = complex(amps_analytic["r"])
    t_an = complex(amps_analytic["t"])
    R_an = float(amps_analytic["R"])
    T_an = float(amps_analytic["T"])

    d_pert = 1.2 * d
    amps_pert = solve_single_layer_analytic(cfg.f0, n1, n2, n3, d_pert)
    R_pert = float(amps_pert["R"])

    f_probe = np.linspace(0.8 * cfg.f0, 1.2 * cfg.f0, cfg.freq_probe_points)
    R_probe = np.array([solve_single_layer_analytic(fi, n1, n2, n3, d)["R"] for fi in f_probe])

    x_np = x_grid.detach().cpu().numpy().squeeze()
    e_pred_np = e_pred.detach().cpu().numpy().squeeze()
    e_an_np = analytic_field(x_np, cfg.f0, n1, n2, n3, d, cfg.Ei, amps_analytic)

    return {
        "config": asdict(cfg),
        "n1": float(n1),
        "n2_opt": float(n2),
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
        "R_abs_error": float(abs(R_pinn - R_an)),
        "interface_l2_error": float(if_l2),
        "R_perturbed_d_1p2": float(R_pert),
        "train_time_s": float(train_time),
        "final_loss_total": float(loss_total_hist[-1]),
        "final_loss_pde": float(loss_pde_hist[-1]),
        "final_loss_if": float(loss_if_hist[-1]),
        "final_loss_energy": float(loss_energy_hist[-1]),
        "loss_history": {
            "total": loss_total_hist,
            "pde": loss_pde_hist,
            "interface": loss_if_hist,
            "energy": loss_energy_hist,
        },
        "x_plot": x_np,
        "e_pred_plot": e_pred_np,
        "e_analytic_plot": e_an_np,
        "f_probe": f_probe,
        "R_probe": R_probe,
    }