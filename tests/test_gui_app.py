import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import pandas as pd
from PyQt6 import QtWidgets

from gui_app import MainWindow, build_simulation_spec


class GuiAppTests(unittest.TestCase):
    def test_build_simulation_spec_uses_widget_values(self):
        values = {
            "function_type": "Lorentzian",
            "num_points": "120",
            "center_index": "3.5",
            "sigma": "0.8",
            "amplitude": "1.2",
            "noise_std": "0.05",
            "random_seed": "9",
        }

        spec = build_simulation_spec(values)

        self.assertEqual(spec["function_type"], "Lorentzian")
        self.assertEqual(spec["num_points"], 120)
        self.assertEqual(spec["center_index"], 3.5)
        self.assertEqual(spec["sigma"], 0.8)
        self.assertEqual(spec["amplitude"], 1.2)
        self.assertEqual(spec["noise_std"], 0.05)
        self.assertEqual(spec["random_seed"], 9)

    def test_save_and_export_controls_write_files(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = MainWindow()
        dataframe = pd.DataFrame(
            {
                "point": [0, 1, 2],
                "voltage": [1.0, 1.5, 2.0],
                "v_orig": [1.0, 1.2, 1.8],
                "v_noise": [0.0, 0.3, 0.2],
            }
        )
        window.current_dataframe = dataframe
        window.graph_panel.set_plot(dataframe)

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = os.path.join(temp_dir, "plot.png")
            export_path = os.path.join(temp_dir, "export.csv")

            window.save_current_plot(plot_path)
            window.export_current_data(export_path)

            self.assertTrue(os.path.exists(plot_path))
            self.assertTrue(os.path.exists(export_path))


if __name__ == "__main__":
    unittest.main()
