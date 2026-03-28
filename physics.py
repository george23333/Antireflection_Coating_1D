import numpy as np

C0 = 299_792_458.0


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

    E = np.zeros_like(x, dtype=np.complex128)

    m1 = x < 0.0
    m2 = (x >= 0.0) & (x <= d)
    m3 = x > d

    E[m1] = Ei * np.exp(-1j * k1 * x[m1]) + Er * np.exp(1j * k1 * x[m1])
    E[m2] = A * np.exp(-1j * k2 * x[m2]) + B * np.exp(1j * k2 * x[m2])
    E[m3] = Et * np.exp(-1j * k3 * x[m3])

    return E