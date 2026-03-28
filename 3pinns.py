import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Setup
# -------------------------------------------------

torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C0 = 299_792_458.0


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def complex_abs2(z):
    return z.real**2 + z.imag**2


def complex_mse(z):
    return torch.mean(complex_abs2(z))


def to_complex(x):
    return torch.tensor(x, dtype=torch.complex64, device=DEVICE)


# -------------------------------------------------
# Model
# -------------------------------------------------

class FCNComplex(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.act = nn.Tanh()

        for l in self.layers:
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)

    def forward(self, x):
        y = x.float()   # IMPORTANT: force float32
        for l in self.layers[:-1]:
            y = self.act(l(y))
        y = self.layers[-1](y)
        return y

    def complex_field(self, x):
        out = self.forward(x)
        return torch.complex(out[:, :1], out[:, 1:2])


class Scattering(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_re = nn.Parameter(torch.tensor(0.0))
        self.r_im = nn.Parameter(torch.tensor(0.0))
        self.t_re = nn.Parameter(torch.tensor(0.0))
        self.t_im = nn.Parameter(torch.tensor(-0.5))

    def r(self):
        return torch.complex(self.r_re, self.r_im)

    def t(self):
        return torch.complex(self.t_re, self.t_im)


# -------------------------------------------------
# Derivatives (safe)
# -------------------------------------------------

def derivatives(model, x):
    x = x.clone().detach().float().requires_grad_(True)

    out = model(x)

    re = out[:, :1]
    im = out[:, 1:2]

    dre = autograd.grad(re, x, torch.ones_like(re), create_graph=True)[0]
    dim = autograd.grad(im, x, torch.ones_like(im), create_graph=True)[0]

    d2re = autograd.grad(dre, x, torch.ones_like(dre), create_graph=True)[0]
    d2im = autograd.grad(dim, x, torch.ones_like(dim), create_graph=True)[0]

    e = torch.complex(re, im)
    de = torch.complex(dre, dim)
    d2e = torch.complex(d2re, d2im)

    return e, de, d2e


# -------------------------------------------------
# PDE (normalized)
# -------------------------------------------------

def pde_loss(model, x_phys, n, l_ref):
    x_hat = x_phys / l_ref
    e, _, d2e = derivatives(model, x_hat)

    k_hat = 2 * np.pi * n
    return complex_mse(d2e + k_hat**2 * e)


# -------------------------------------------------
# Main
# -------------------------------------------------

def run():

    # Physics
    f0 = 150e9
    eps1, eps3 = 1.0, 11.7

    n1 = np.sqrt(eps1)
    n3 = np.sqrt(eps3)
    n2 = np.sqrt(n1 * n3)

    k0 = 2 * np.pi * f0 / C0
    k1, k2, k3 = k0 * n1, k0 * n2, k0 * n3

    d = C0 / (4 * f0 * n2)
    l_ref = C0 / f0

    lambda1 = C0 / (f0 * n1)
    lambda3 = C0 / (f0 * n3)

    xL = -2 * lambda1
    xR = d + 2 * lambda3

    # Models
    layers = (1, 64, 64, 64, 2)

    model1 = FCNComplex(layers).to(DEVICE)
    model2 = FCNComplex(layers).to(DEVICE)
    model3 = FCNComplex(layers).to(DEVICE)

    scat = Scattering().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model1.parameters())
        + list(model2.parameters())
        + list(model3.parameters())
        + list(scat.parameters()),
        lr=1e-3,
    )

    # Training
    for ep in range(9000):

        x1 = (torch.rand(200, 1, device=DEVICE) * (0 - xL) + xL).float()
        x2 = (torch.rand(200, 1, device=DEVICE) * d).float()
        x3 = (torch.rand(200, 1, device=DEVICE) * (xR - d) + d).float()

        # PDE
        l_pde = (
            pde_loss(model1, x1, n1, l_ref)
            + pde_loss(model2, x2, n2, l_ref)
            + pde_loss(model3, x3, n3, l_ref)
        )

        # Interfaces
        x0 = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)
        xd = torch.tensor([[d]], dtype=torch.float32, device=DEVICE)

        e1_0, de1_0, _ = derivatives(model1, x0 / l_ref)
        e2_0, de2_0, _ = derivatives(model2, x0 / l_ref)

        e2_d, de2_d, _ = derivatives(model2, xd / l_ref)
        e3_d, de3_d, _ = derivatives(model3, xd / l_ref)

        de1_0 /= l_ref
        de2_0 /= l_ref
        de2_d /= l_ref
        de3_d /= l_ref

        l_if = (
            complex_mse(e1_0 - e2_0)
            + complex_mse(de1_0 - de2_0)
            + complex_mse(e2_d - e3_d)
            + complex_mse(de2_d - de3_d)
        )

        # Boundary
        Ei = to_complex(1.0)
        j = to_complex(1j)

        xL_t = torch.tensor([[xL]], dtype=torch.float32, device=DEVICE)
        xR_t = torch.tensor([[xR]], dtype=torch.float32, device=DEVICE)

        eL, deL, _ = derivatives(model1, xL_t / l_ref)
        eR, deR, _ = derivatives(model3, xR_t / l_ref)

        deL /= l_ref
        deR /= l_ref

        r = scat.r()
        t = scat.t()

        eL_target = Ei * (
            torch.exp(-j * k1 * xL_t)
            + r * torch.exp(j * k1 * xL_t)
        )

        deL_target = Ei * (
            -j * k1 * torch.exp(-j * k1 * xL_t)
            + j * k1 * r * torch.exp(j * k1 * xL_t)
        )

        eR_target = Ei * t * torch.exp(-j * k3 * xR_t)
        deR_target = -j * k3 * eR_target

        l_bc = (
            complex_mse(eL - eL_target)
            + complex_mse(deL - deL_target)
            + complex_mse(eR - eR_target)
            + complex_mse(deR - deR_target)
        )

        # Energy
        l_energy = (complex_abs2(r) + (n3/n1)*complex_abs2(t) - 1)**2

        loss = l_pde + 20*l_if + 10*l_bc + l_energy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 500 == 0:
            print(f"{ep}: loss={loss.item():.3e}")

    # Evaluation
    x = torch.linspace(xL, xR, 1200, device=DEVICE).unsqueeze(1)

    with torch.no_grad():
        E = torch.zeros((x.shape[0], 1), dtype=torch.complex64, device=DEVICE)

        m1 = x[:, 0] < 0
        m2 = (x[:, 0] >= 0) & (x[:, 0] <= d)
        m3 = x[:, 0] > d

        E[m1] = model1.complex_field(x[m1] / l_ref)
        E[m2] = model2.complex_field(x[m2] / l_ref)
        E[m3] = model3.complex_field(x[m3] / l_ref)

    x_np = x.cpu().numpy().squeeze()
    E_np = E.cpu().numpy().squeeze()

    plt.figure(figsize=(8,5))
    plt.plot(x_np, np.abs(E_np))
    plt.axvline(0, linestyle="--")
    plt.axvline(d, linestyle="--")
    plt.title("|E(x)| (3 PINNs)")
    plt.show()


if __name__ == "__main__":
    run()