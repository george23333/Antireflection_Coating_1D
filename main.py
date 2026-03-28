from models import ForwardPINNConfig, run_forward_pinn
from utils import plot_results, print_summary, setup_device_and_seed


def main() -> dict:
    device = setup_device_and_seed(1234)

    cfg = ForwardPINNConfig(
        f0=150e9,
        eps_r1=1.0,
        eps_r3=11.7,
        mu_r=1.0,
        Ei=1.0,
        epochs=5000,
        lr=1e-3,
        n_f_coating=400,
        w_pde=1.0,
        w_if=50.0,
        w_energy=2.0,
    )

    results = run_forward_pinn(cfg, device)
    print_summary(results)
    plot_results(results, cfg.f0)
    return results


if __name__ == "__main__":
    main()