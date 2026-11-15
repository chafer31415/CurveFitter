import os
import tempfile
import unittest

from interactive_script import run_simulations
from simulated_experimental_data import (
    generate_simulated_data,
    get_available_functions,
    load_and_plot_simulated_data,
)


class SimulatedDataTests(unittest.TestCase):
    def test_generates_expected_columns_and_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "simulated.csv")
            df = generate_simulated_data(
                output_path=output_path,
                function_name="gaussian_derivative",
                num_points=20,
                center_index=10.0,
                sigma=1.0,
                amplitude=2.0,
                noise_std=0.1,
                random_seed=7,
            )

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(list(df.columns), ["point", "voltage", "v_orig", "v_noise"])
            self.assertEqual(df["point"].iloc[0], 0)
            self.assertEqual(df["point"].iloc[-1], 19)
            self.assertIn("gaussian_derivative", get_available_functions())

    def test_lorentzian_function_is_available(self):
        self.assertIn("lorentzian", get_available_functions())

        df = generate_simulated_data(
            output_path="/tmp/lorentzian.csv",
            function_name="lorentzian",
            num_points=12,
            center_index=6.0,
            sigma=0.8,
            amplitude=1.5,
            noise_std=0.0,
            random_seed=3,
        )

        self.assertEqual(list(df.columns), ["point", "voltage", "v_orig", "v_noise"])
        self.assertEqual(len(df), 12)

    def test_load_and_plot_simulated_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "simulated.csv")
            generate_simulated_data(
                output_path=output_path,
                function_name="gaussian_derivative",
                num_points=10,
                center_index=5.0,
                sigma=1.0,
                amplitude=1.0,
                noise_std=0.2,
                random_seed=4,
            )

            df, fig, axes = load_and_plot_simulated_data(output_path, show=False)

            self.assertEqual(list(df.columns), ["point", "voltage", "v_orig", "v_noise"])
            self.assertEqual(len(axes), 3)
            self.assertEqual(fig.axes[0].get_ylabel(), "voltage")
            self.assertEqual(fig.axes[1].get_ylabel(), "v_orig")
            self.assertEqual(fig.axes[2].get_ylabel(), "v_noise")

    def test_run_simulations_saves_timestamped_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_simulations(
                sim_specs={
                    "demo": {
                        "function_type": "Gaussian",
                        "num_points": 8,
                        "center_index": 0.0,
                        "sigma": 1.0,
                        "amplitude": 1.0,
                        "noise_std": 0.0,
                        "random_seed": 1,
                    }
                },
                output_dir=temp_dir,
                show=False,
            )

            output_files = os.listdir(temp_dir)
            self.assertGreater(len(summary), 0)
            self.assertTrue(any(name.endswith(".csv") for name in output_files))
            self.assertTrue(any(name.endswith(".png") for name in output_files))


if __name__ == "__main__":
    unittest.main()
