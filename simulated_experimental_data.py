from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FunctionCallable = Callable[[np.ndarray, float, float, float], np.ndarray]


def gaussian_derivative(
    x: np.ndarray,
    center_index: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    """Return the derivative of a Gaussian curve evaluated at x values."""
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    x = np.asarray(x, dtype=float)
    exponent = -((x - center_index) ** 2) / (2.0 * sigma**2)
    return -amplitude * ((x - center_index) / sigma**2) * np.exp(exponent)


def _gaussian_derivative_function(
    x: np.ndarray,
    center_index: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    return gaussian_derivative(x, center_index=center_index, sigma=sigma, amplitude=amplitude)


def lorentzian(
    x: np.ndarray,
    center_index: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    """Return a Lorentzian curve evaluated at x values."""
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    x = np.asarray(x, dtype=float)
    denominator = 1.0 + ((x - center_index) / sigma) ** 2
    return amplitude / denominator


def _lorentzian_function(
    x: np.ndarray,
    center_index: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    return lorentzian(x, center_index=center_index, sigma=sigma, amplitude=amplitude)


FUNCTION_REGISTRY: dict[str, FunctionCallable] = {
    "gaussian_derivative": _gaussian_derivative_function,
    "lorentzian": _lorentzian_function,
}


def register_function(name: str, func: FunctionCallable) -> None:
    """Register a new base-function implementation for simulated data generation."""
    FUNCTION_REGISTRY[name] = func


def get_available_functions() -> list[str]:
    """Return the currently available function names."""
    return sorted(FUNCTION_REGISTRY.keys())


def generate_simulated_data(
    output_path: str | Path = "simulated_data.csv",
    function_name: str = "gaussian_derivative",
    num_points: int = 1000,
    center_index: float = 0.0,
    sigma: float = 1.0,
    amplitude: float = 1.0,
    noise_std: float = 0.05,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Generate a simulated experimental dataset and save it to CSV.

    The generated DataFrame contains the integer point index, the noisy voltage,
    the pure underlying signal, and the noise deviation that was added.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if function_name not in FUNCTION_REGISTRY:
        available = ", ".join(get_available_functions())
        raise ValueError(f"Unknown function '{function_name}'. Available: {available}")

    points = np.arange(num_points, dtype=int)
    x_values = np.linspace(center_index - 3.0 * sigma, center_index + 3.0 * sigma, num_points)

    base_function = FUNCTION_REGISTRY[function_name]
    v_orig = base_function(x_values, center_index=center_index, sigma=sigma, amplitude=amplitude)

    rng = np.random.default_rng(random_seed)
    v_noise = rng.normal(loc=0.0, scale=noise_std, size=num_points)
    voltage = v_orig + v_noise

    dataframe = pd.DataFrame(
        {
            "point": points,
            "voltage": voltage,
            "v_orig": v_orig,
            "v_noise": v_noise,
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    return dataframe


def load_and_plot_simulated_data(
    csv_path: str | Path,
    show: bool = True,
    output_png: str | Path | None = None,
) -> tuple[pd.DataFrame, plt.Figure, list[plt.Axes]]:
    """Load simulated data from CSV and create a three-panel matplotlib plot."""
    dataframe = pd.read_csv(csv_path)

    if not {"point", "voltage", "v_orig", "v_noise"}.issubset(dataframe.columns):
        missing = {"point", "voltage", "v_orig", "v_noise"} - set(dataframe.columns)
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

    axes[0].plot(dataframe["point"], dataframe["voltage"], marker="o", linestyle="-", color="tab:blue")
    axes[0].set_ylabel("voltage")
    axes[0].set_title("Simulated experimental data")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dataframe["point"], dataframe["v_orig"], color="tab:orange", linewidth=1.8)
    axes[1].set_ylabel("v_orig")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dataframe["point"], dataframe["v_noise"], color="tab:green", linewidth=1.8)
    axes[2].set_ylabel("v_noise")
    axes[2].set_xlabel("point")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if output_png is None:
        output_png = Path(csv_path).with_suffix(".png")
    else:
        output_png = Path(output_png)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)

    if show:
        plt.show()

    return dataframe, fig, axes


if __name__ == "__main__":
    generate_simulated_data(output_path="simulated_data.csv")
