import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from physics import analytic_field, compute_optimal_layer, solve_single_layer_analytic


torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

C0 = 299_792_458.0


@dataclass
class InversePINNConfig:
    # Known physics.
    f0: float = 150e9
    eps_r1: float = 1.0
    eps_r3: float = 11.7
    mu_r: float = 1.0
    Ei: float = 1.0

    # Synthetic target used to demonstrate the inverse problem.
    true_n2_scale: float = 1.0
    true_d_scale: float = 1.12

    # Unknown-parameter bounds. Keep them reasonably tight for identifiability.
    n2_min: float = 1.05
    n2_max: float = 3.6
    d_min_scale: float = 0.55
    d_max_scale: float = 1.55

    # Initial guess as scale factors relative to the quarter-wave design.
    init_n2_scale: float = 0.8
    init_d_scale: float = 0.8

    # Unknown left interface x0. Values are scaled by the wavelength in region 1.
    true_x0_wavelengths: float = 0.20
    init_x0_wavelengths: float = -0.10
    x0_min_wavelengths: float = -0.50
    x0_max_wavelengths: float = 0.50

    # Network / training.
    layers: tuple[int, ...] = (1, 50, 50, 50, 50, 2)
    epochs: int = 8000
    lr: float = 2e-3
    n_collocation: int = 300
    n_data_left: int = 10
    n_data_coating: int = 16
    n_data_right: int = 10
    noise_std: float = 0.01

    # Loss weights.
    w_pde: float = 1.0
    w_interface: float = 50.0
    w_data: float = 100.0
    w_energy: float = 1.0

    # Optimization refinements, following hybrid_optimization.py.
    use_lr_scheduler: bool = True
    scheduler_eta_min: float = 1e-5
    use_lbfgs: bool = True
    lbfgs_lr: float = 0.8
    lbfgs_max_iter: int = 250
    lbfgs_history_size: int = 100
    lbfgs_line_search_fn: str = "strong_wolfe"

    print_every: int = 500
    n_plot: int = 1200
    make_plots: bool = True

    # Saved model and recovered inverse parameters.
    checkpoint_path: str = "checkpoints/inverse_two_unknown_boundaries.pt"
    force_retrain: bool = False


def complex_abs2(z: torch.Tensor) -> torch.Tensor:
    return z.real.square() + z.imag.square()


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return complex_abs2(z).mean()


def inverse_sigmoid(y: float) -> float:
    #y = min(max(y, 1e-6), 1.0 - 1e-6)
    return float(np.log(y / (1.0 - y)))


def bounded_parameter(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def sample_uniform(n_points: int, x_min: float, x_max: float) -> torch.Tensor:
    x = x_min + (x_max - x_min) * torch.rand((n_points, 1), device=device)
    return x.to(torch.float64)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: InversePINNConfig):
    if not cfg.use_lr_scheduler:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.scheduler_eta_min,
    )


class CoatingFieldNet(nn.Module):
    def __init__(self, layers: tuple[int, ...]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        y = 2.0 * s.to(torch.float64) - 1.0             # Scale input from [0, 1] to [-1, 1]
        for layer in self.linears[:-1]:
            y = self.activation(layer(y))
        return self.linears[-1](y)

    def complex_field(self, s: torch.Tensor) -> torch.Tensor:
        y = self(s)
        return torch.complex(y[:, :1], y[:, 1:2])


class InverseCoatingParams(nn.Module):
    def __init__(
        self,
        cfg: InversePINNConfig,
        d_design: float,
        n2_design: float,
        lambda1: float,
    ):
        super().__init__()
        d_min = cfg.d_min_scale * d_design
        d_max = cfg.d_max_scale * d_design

        init_n2 = np.clip(cfg.init_n2_scale * n2_design, cfg.n2_min, cfg.n2_max)     # ensure initial guess is within bounds
        init_d = np.clip(cfg.init_d_scale * d_design, d_min, d_max)

        x0_min = cfg.x0_min_wavelengths * lambda1
        x0_max = cfg.x0_max_wavelengths * lambda1
        init_x0 = np.clip(cfg.init_x0_wavelengths * lambda1, x0_min, x0_max)

        raw_n2 = inverse_sigmoid((init_n2 - cfg.n2_min) / (cfg.n2_max - cfg.n2_min)) # normalize initial guess to [0, 1] and then apply inverse sigmoid
        raw_d = inverse_sigmoid((init_d - d_min) / (d_max - d_min))
        raw_x0 = inverse_sigmoid((init_x0 - x0_min) / (x0_max - x0_min))

        self.raw_n2 = nn.Parameter(torch.tensor(raw_n2, dtype=torch.float64))
        self.raw_d = nn.Parameter(torch.tensor(raw_d, dtype=torch.float64))
        self.raw_x0 = nn.Parameter(torch.tensor(raw_x0, dtype=torch.float64))
        self.r_re = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.r_im = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.t_re = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.t_im = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.n2_min = float(cfg.n2_min)
        self.n2_max = float(cfg.n2_max)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.x0_min = float(x0_min)
        self.x0_max = float(x0_max)

    @property
    def n2(self) -> torch.Tensor:
        return bounded_parameter(self.raw_n2, self.n2_min, self.n2_max)

    @property
    def d(self) -> torch.Tensor:
        return bounded_parameter(self.raw_d, self.d_min, self.d_max)

    @property
    def x0(self) -> torch.Tensor:
        return bounded_parameter(self.raw_x0, self.x0_min, self.x0_max)

    @property
    def r(self) -> torch.Tensor:
        return torch.complex(self.r_re, self.r_im)

    @property
    def t(self) -> torch.Tensor:
        return torch.complex(self.t_re, self.t_im)


def coating_field_and_derivatives(
    model: CoatingFieldNet, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = s.clone().detach().requires_grad_(True)
    y = model(s)
    e_re, e_im = y[:, :1], y[:, 1:2]

    def grad(u: torch.Tensor) -> torch.Tensor:
        return autograd.grad(
            u,
            s,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

    de_re, de_im = grad(e_re), grad(e_im)
    d2e_re, d2e_im = grad(de_re), grad(de_im)

    e = torch.complex(e_re, e_im)
    de_ds = torch.complex(de_re, de_im)
    d2e_ds2 = torch.complex(d2e_re, d2e_im)
    return e, de_ds, d2e_ds2


def predict_field(
    model: CoatingFieldNet,
    params: InverseCoatingParams,
    x: torch.Tensor,
    phys: dict[str, float],
    cfg: InversePINNConfig,
) -> torch.Tensor:
    j = torch.tensor(1j, dtype=torch.complex128, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex128, device=device)
    # Local coordinate measured from the current estimated left interface.
    xi = x - params.x0

    # Hard region assignment. detach() is intentional because Boolean masks
    # are not differentiable with respect to x0 or d.
    xi_mask = x[:, 0] - params.x0.detach()
    left = xi_mask < 0.0
    right = xi_mask > params.d.detach()
    coating = ~(left | right)

    out = torch.zeros((x.shape[0], 1), dtype=torch.complex128, device=device)
    if left.any():
        xl = xi[left]
        out[left] = ei * (torch.exp(-j * phys["k1"] * xl) + params.r * torch.exp(j * phys["k1"] * xl))
    if coating.any():
        out[coating] = model.complex_field(xi[coating] / params.d)
    if right.any():
        xr = xi[right]
        out[right] = ei * params.t * torch.exp(-j * phys["k3"] * xr)
    return out


def make_synthetic_observations(
    cfg: InversePINNConfig,
    phys_true: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = phys_true["d"]
    lambda1 = C0 / (cfg.f0 * phys_true["n1"])
    lambda3 = C0 / (cfg.f0 * phys_true["n3"])

    x_left = np.linspace(-0.85 * lambda1, -0.15 * lambda1, cfg.n_data_left)
    x_coat = np.linspace(0.08 * d, 0.92 * d, cfg.n_data_coating)
    x_right = np.linspace(d + 0.15 * lambda3, d + 0.85 * lambda3, cfg.n_data_right)
    x_local_np = np.concatenate([x_left, x_coat, x_right])[:, None]
    x_data_np = x_local_np + phys_true["x0"]
    region_np = np.concatenate(
        [
            -np.ones(cfg.n_data_left, dtype=np.int64),
            np.zeros(cfg.n_data_coating, dtype=np.int64),
            np.ones(cfg.n_data_right, dtype=np.int64),
        ]
    )

    amps = solve_single_layer_analytic(
        cfg.f0, phys_true["n1"], phys_true["n2"], phys_true["n3"], phys_true["d"], cfg.Ei
    )
    e_data_np = analytic_field(
        x_local_np[:, 0],
        cfg.f0,
        phys_true["n1"],
        phys_true["n2"],
        phys_true["n3"],
        phys_true["d"],
        cfg.Ei,
        amps,
    )[:, None]

    if cfg.noise_std > 0.0:
        scale = np.maximum(np.abs(e_data_np).mean(), 1e-12)
        noise = cfg.noise_std * scale * (
            np.random.randn(*e_data_np.shape) + 1j * np.random.randn(*e_data_np.shape)
        )
        e_data_np = e_data_np + noise

    x_data = torch.tensor(x_data_np, dtype=torch.float64, device=device)
    e_data = torch.tensor(e_data_np, dtype=torch.complex128, device=device)
    region = torch.tensor(region_np, dtype=torch.int64, device=device)
    return x_data, e_data, region


def compute_losses(
    model: CoatingFieldNet,
    params: InverseCoatingParams,
    cfg: InversePINNConfig,
    s_collocation: torch.Tensor,
    x_data: torch.Tensor,
    e_data: torch.Tensor,
    region_data: torch.Tensor,
    phys: dict[str, float],
) -> dict[str, torch.Tensor]:
    j = torch.tensor(1j, dtype=torch.complex128, device=device)
    ei = torch.tensor(cfg.Ei, dtype=torch.complex128, device=device)

    e, de_ds, d2e_ds2 = coating_field_and_derivatives(model, s_collocation)
    k0n2d = phys["k0"] * params.n2 * params.d
    residual = d2e_ds2 + k0n2d.square() * e
    l_pde = complex_mse(residual / k0n2d.square().detach())

    s0 = torch.zeros((1, 1), dtype=torch.float64, device=device)
    s1 = torch.ones((1, 1), dtype=torch.float64, device=device)
    e0, de0_ds, _ = coating_field_and_derivatives(model, s0)
    e1, de1_ds, _ = coating_field_and_derivatives(model, s1)
    de0_dx = de0_ds / params.d
    de1_dx = de1_ds / params.d

    e_left_0 = ei * (1.0 + params.r)
    de_left_0 = ei * (-j * phys["k1"] + j * phys["k1"] * params.r)
    e_right_d = ei * params.t * torch.exp(-j * phys["k3"] * params.d)
    de_right_d = -j * phys["k3"] * e_right_d

    l_interface = (
        complex_mse(e0 - e_left_0)
        + complex_mse((de0_dx - de_left_0) / phys["k0"])
        + complex_mse(e1 - e_right_d)
        + complex_mse((de1_dx - de_right_d) / phys["k0"])
    )

    # Ignore the synthetic labels and classify using the current hard boundaries.
    e_pred = predict_field(model, params, x_data, phys, cfg)
    l_data = complex_mse(e_pred - e_data)

    energy = complex_abs2(params.r) + (phys["n3"] / phys["n1"]) * complex_abs2(params.t) - 1.0
    l_energy = energy.square().mean()

    l_total = (
        cfg.w_pde * l_pde
        + cfg.w_interface * l_interface
        + cfg.w_data * l_data
        + cfg.w_energy * l_energy
    )
    return {
        "total": l_total,
        "pde": l_pde,
        "interface": l_interface,
        "data": l_data,
        "energy": l_energy,
    }


def run_inverse_pinn(cfg: InversePINNConfig) -> dict[str, Any]:
    n1 = float(np.sqrt(cfg.eps_r1 * cfg.mu_r))
    n3 = float(np.sqrt(cfg.eps_r3 * cfg.mu_r))
    n2_design, d_design = compute_optimal_layer(cfg.f0, cfg.eps_r1, cfg.eps_r3, cfg.mu_r)
    n2_true = float(cfg.true_n2_scale * n2_design)
    d_true = float(cfg.true_d_scale * d_design)
    lambda1 = C0 / (cfg.f0 * n1)
    x0_true = float(cfg.true_x0_wavelengths * lambda1)

    k0 = 2.0 * np.pi * cfg.f0 / C0
    phys = {
        "n1": n1,
        "n3": n3,
        "k0": float(k0),
        "k1": float(k0 * n1),
        "k3": float(k0 * n3),
    }
    phys_true = dict(phys, n2=n2_true, d=d_true, x0=x0_true)

    x_data, e_data, region_data = make_synthetic_observations(cfg, phys_true)
    model = CoatingFieldNet(cfg.layers).to(device)
    params = InverseCoatingParams(cfg, d_design, n2_design, lambda1).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(params.parameters()), lr=cfg.lr)
    scheduler = build_lr_scheduler(optimizer, cfg)

    loss_history: dict[str, list[float]] = {
        "total": [],
        "pde": [],
        "interface": [],
        "data": [],
        "energy": [],
        "n2": [],
        "d": [],
        "x0": [],
    }

    checkpoint_path = Path(__file__).resolve().parent / cfg.checkpoint_path
    load_checkpoint = checkpoint_path.exists() and not cfg.force_retrain
    if load_checkpoint:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
        except TypeError:  # Compatibility with older PyTorch versions.
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        params.load_state_dict(checkpoint["params_state_dict"])
        loss_history = checkpoint.get("loss_history", loss_history)
        print(f"Loaded trained inverse model: {checkpoint_path}")

    start = time.time()
    for epoch in range(1, 0 if load_checkpoint else cfg.epochs + 1):
        s_collocation = sample_uniform(cfg.n_collocation, 0.0, 1.0)
        losses = compute_losses(model, params, cfg, s_collocation, x_data, e_data, region_data, phys)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for key in ("total", "pde", "interface", "data", "energy"):
            loss_history[key].append(float(losses[key].detach().cpu()))
        loss_history["n2"].append(float(params.n2.detach().cpu()))
        loss_history["d"].append(float(params.d.detach().cpu()))
        loss_history["x0"].append(float(params.x0.detach().cpu()))

        if epoch == 1 or epoch % cfg.print_every == 0:
            print(
                f"epoch {epoch:5d} | total={losses['total'].item():.3e} "
                f"pde={losses['pde'].item():.3e} if={losses['interface'].item():.3e} "
                f"data={losses['data'].item():.3e} | "
                f"n2={params.n2.item():.6f} d={params.d.item():.6e} "
                f"x0={params.x0.item():.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    if cfg.use_lbfgs and not load_checkpoint:
        print("\nStarting LBFGS refinement...")
        s_lbfgs = sample_uniform(cfg.n_collocation, 0.0, 1.0)
        lbfgs = torch.optim.LBFGS(
            list(model.parameters()) + list(params.parameters()),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size,
            line_search_fn=cfg.lbfgs_line_search_fn,
        )

        def closure():
            lbfgs.zero_grad()
            lbfgs_losses = compute_losses(
                model, params, cfg, s_lbfgs, x_data, e_data, region_data, phys
            )
            lbfgs_losses["total"].backward()
            return lbfgs_losses["total"]

        lbfgs.step(closure)
        lbfgs_losses = compute_losses(
            model, params, cfg, s_lbfgs, x_data, e_data, region_data, phys
        )
        for key in ("total", "pde", "interface", "data", "energy"):
            loss_history[key].append(float(lbfgs_losses[key].detach().cpu()))
        loss_history["n2"].append(float(params.n2.detach().cpu()))
        loss_history["d"].append(float(params.d.detach().cpu()))
        loss_history["x0"].append(float(params.x0.detach().cpu()))
        print(
            "LBFGS done: "
            f"total={lbfgs_losses['total'].item():.3e}, "
            f"pde={lbfgs_losses['pde'].item():.3e}, "
            f"if={lbfgs_losses['interface'].item():.3e}, "
            f"data={lbfgs_losses['data'].item():.3e}, "
            f"n2={params.n2.item():.6f}, d={params.d.item():.6e}"
        )

    train_time = time.time() - start
    if not load_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params_state_dict": params.state_dict(),
                "loss_history": loss_history,
            },
            checkpoint_path,
        )
        print(f"Saved trained inverse model: {checkpoint_path}")

    n2_est = float(params.n2.detach().cpu())
    d_est = float(params.d.detach().cpu())
    x0_est = float(params.x0.detach().cpu())
    r_est = complex(params.r.detach().cpu().numpy().item())
    t_est = complex(params.t.detach().cpu().numpy().item())

    print("\n=== Inverse result ===")
    print(f"true n2 = {n2_true:.8f}, estimated n2 = {n2_est:.8f}")
    print(f"true d  = {d_true:.8e} m, estimated d  = {d_est:.8e} m")
    print(f"true x0 = {x0_true:.8e} m, estimated x0 = {x0_est:.8e} m")
    print(f"true x1 = {x0_true + d_true:.8e} m, estimated x1 = {x0_est + d_est:.8e} m")
    print(f"r = {r_est:.6e}, R = {abs(r_est) ** 2:.6e}")
    print(f"t = {t_est:.6e}, T = {(n3 / n1) * abs(t_est) ** 2:.6e}")
    if not load_checkpoint:
        print(f"Training time: {train_time:.2f} seconds")

    if cfg.make_plots:
        lambda3 = C0 / (cfg.f0 * n3)
        x_min = x0_true - 1.0 * lambda1
        x_max = x0_true + d_true + 1.0 * lambda3
        x_plot = torch.linspace(x_min, x_max, cfg.n_plot, device=device).unsqueeze(1)
        e_pred = predict_field(model, params, x_plot, phys, cfg).detach().cpu().numpy()[:, 0]

        amps_true = solve_single_layer_analytic(cfg.f0, n1, n2_true, n3, d_true, cfg.Ei)
        e_true = analytic_field(
            x_plot.detach().cpu().numpy()[:, 0] - x0_true,
            cfg.f0, n1, n2_true, n3, d_true, cfg.Ei, amps_true
        )

        plt.figure(figsize=(6, 5))
        plt.plot(x_plot.cpu().numpy()[:, 0], np.abs(e_true), label="true |E|")
        plt.plot(x_plot.cpu().numpy()[:, 0], np.abs(e_pred), "--", label="PINN |E|")
        plt.scatter(
            x_data.detach().cpu().numpy()[:, 0],
            np.abs(e_data.detach().cpu().numpy()[:, 0]),
            s=18,
            label="data",
        )
        plt.axvline(x0_true, color="k", linestyle=":", linewidth=1, label="true interfaces")
        plt.axvline(x0_true + d_true, color="k", linestyle=":", linewidth=1)
        plt.axvline(x0_est, color="r", linestyle=":", linewidth=1, label="estimated interfaces")
        plt.axvline(x0_est + d_est, color="r", linestyle=":", linewidth=1)
        plt.xlabel("x [m]")
        plt.ylabel("|E(x)|")
        #plt.title("Inverse PINN Field Reconstruction")
        plt.title("(a)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(6, 5))
        plt.semilogy(loss_history["total"], label="total")
        plt.semilogy(loss_history["pde"], label="pde")
        plt.semilogy(loss_history["interface"], label="interface")
        plt.semilogy(loss_history["data"], label="data")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Inverse PINN Loss History")
        plt.title("(b)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(6, 5))
        plt.plot(loss_history["n2"], label="estimated n2")
        plt.axhline(n2_true, color="k", linestyle=":", label="true n2")
        plt.xlabel("Epoch")
        plt.ylabel("n2")
        #plt.title("Recovered Refractive Index")
        plt.title("(c)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(6, 5))
        plt.plot(np.array(loss_history["d"]) * 1e6, label="estimated d")
        plt.axhline(d_true * 1e6, color="k", linestyle=":", label="true d")
        plt.xlabel("Epoch")
        plt.ylabel("d [um]")
        #plt.title("Recovered Thickness")
        plt.title("(d)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(6, 5))
        plt.plot(np.array(loss_history["x0"]) * 1e6, label="estimated x0")
        plt.axhline(x0_true * 1e6, color="k", linestyle=":", label="true x0")
        plt.xlabel("Epoch")
        plt.ylabel("x0 [um]")
        #plt.title("Recovered Left Interface")
        plt.title("(e)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "model": model,
        "params": params,
        "config": cfg,
        "loss_history": loss_history,
        "true": {"n2": n2_true, "d": d_true, "x0": x0_true},
        "estimated": {
            "n2": n2_est,
            "d": d_est,
            "x0": x0_est,
            "r": r_est,
            "t": t_est,
        },
        "x_data": x_data.detach().cpu().numpy(),
        "e_data": e_data.detach().cpu().numpy(),
        "region_data": region_data.detach().cpu().numpy(),
    }


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = InversePINNConfig()
    if config:
        cfg = replace(cfg, **config)
    return run_inverse_pinn(cfg)


if __name__ == "__main__":
    main()
    #main({"force_retrain": True})
