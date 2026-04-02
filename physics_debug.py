import numpy as np
import matplotlib.pyplot as plt

from physics import (
    compute_optimal_layer,
    solve_single_layer_analytic,
    analytic_field,
)

C0 = 299_792_458.0


def main():

    f0 = 150e9
    eps_r1 = 1.0
    eps_r3 = 11.7
    mu_r = 1.0
    Ei = 1.0

    n1 = np.sqrt(eps_r1 * mu_r)
    n3 = np.sqrt(eps_r3 * mu_r)

    n2, d = compute_optimal_layer(f0, eps_r1, eps_r3, mu_r)

    # pertubed d
    d = d * 1.0

    print(d)

    print("=== Parameters ===")
    print(f"n1 = {n1:.4f}, n2 = {n2:.4f}, n3 = {n3:.4f}")
    print(f"d  = {d:.4e} m")

    amps = solve_single_layer_analytic(f0, n1, n2, n3, d, Ei)

    print("\n=== Coefficients ===")
    print()
    print(f"Er = {amps['Er']}")
    print(f"Et = {amps['Et']}")
    print(f"R  = {amps['R']:.3e}")
    print(f"T  = {amps['T']:.3e}")
    print(f"R+T = {amps['R'] + amps['T']:.6f}")
    print(f"r = {amps['r']:.3e}")
    print(f"t = {amps['t']:.3e}")

    # -------------------------
    # Spatial grid
    # -------------------------
    lambda1 = C0 / (f0 * n1)
    lambda3 = C0 / (f0 * n3)

    x_left = -1.5 * lambda1
    x_right = d + 1.5 * lambda3

    x = np.linspace(x_left, x_right, 1500)

    E = analytic_field(x, f0, n1, n2, n3, d, Ei, amps)

    # -------------------------
    # Plot magnitude
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.abs(E), label="|E(x)|")
    plt.axvline(0.0, linestyle=":", label="x=0")
    plt.axvline(d, linestyle=":", label="x=d")
    plt.xlabel("x [m]")
    plt.ylabel("|E(x)|")
    plt.title("Reference Solution: Field Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Plot phase
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.unwrap(np.angle(E)))
    plt.axvline(0.0, linestyle=":")
    plt.axvline(d, linestyle=":")
    plt.xlabel("x [m]")
    plt.ylabel("Phase [rad]")
    plt.title("Reference Solution: Phase")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Frequency sweep
    # -------------------------
    f_vals = np.linspace(0.8 * f0, 1.2 * f0, 200)
    R_vals = []

    for f in f_vals:
        amps_f = solve_single_layer_analytic(f, n1, n2, n3, d, Ei)
        R_vals.append(amps_f["R"])

    R_vals = np.array(R_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(f_vals / 1e9, R_vals)
    plt.axvline(f0 / 1e9, linestyle=":", label="f0")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Reflectance R")
    plt.title("Reference Reflectance vs Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()