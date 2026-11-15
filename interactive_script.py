from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython import get_ipython
from IPython.display import display

from simulated_experimental_data import (
    generate_simulated_data,
    load_and_plot_simulated_data,
)


sim_dict: dict[str, dict[str, Any]] = {
    "gaussian_example": {
        "function_type": "Gaussian",
        "num_points": 1000,
        "center_index": 0.0,
        "sigma": 1.0,
        "amplitude": 1.0,
        "noise_std": 0.1,
        "random_seed": 42,
    },
    "lorentzian_example": {
        "function_type": "Lorentzian",
        "num_points": 800,
        "center_index": 0.0,
        "sigma": 0.7,
        "amplitude": 1.5,
        "noise_std": 0.08,
        "random_seed": 7,
    },
}


def _normalize_function_type(function_type: str) -> str:
    normalized = function_type.strip().lower().replace(" ", "_")
    mapping = {
        "gaussian": "gaussian_derivative",
        "gaussian_derivative": "gaussian_derivative",
        "lorentzian": "lorentzian",
        "lorentz": "lorentzian",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported function_type: {function_type}")
    return mapping[normalized]


def _display_table(frame: pd.DataFrame) -> None:
    try:
        display(frame)
    except Exception:
        print(frame.to_string(index=False))


def _display_figure(fig: plt.Figure) -> None:
    if get_ipython() is not None:
        display(fig)
    else:
        plt.show()


def run_simulations(
    sim_specs: dict[str, dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Run each simulation defined in sim_dict and save timestamped CSV/PNG outputs."""
    specs = sim_specs or sim_dict
    output_path = Path(output_dir or ".")
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for name, spec in specs.items():
        simulation_params = dict(spec)
        function_type = simulation_params.pop("function_type")
        function_name = _normalize_function_type(function_type)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{name}_{timestamp}"
        csv_path = output_path / f"{base_name}.csv"
        png_path = output_path / f"{base_name}.png"

        generate_simulated_data(
            output_path=csv_path,
            function_name=function_name,
            **simulation_params,
        )
        _, fig, _ = load_and_plot_simulated_data(csv_path, show=False, output_png=png_path)

        if show:
            _display_figure(fig)

        result_row = {
            "simulation": name,
            "function_type": function_type,
            "function_name": function_name,
            "csv_file": str(csv_path.name),
            "png_file": str(png_path.name),
            **simulation_params,
        }
        results.append(result_row)

    summary = pd.DataFrame(results)
    print("\nSimulation parameter summary")
    print("=" * 80)
    _display_table(summary)
    return summary


if __name__ == "__main__":
    run_simulations()
