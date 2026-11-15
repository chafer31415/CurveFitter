from __future__ import annotations

import os
import sys

# Tell Qt exactly where its platform plugins live
plugin_root = "/Users/cfertig/Documents/GitHub/testrepo/.venv/lib/python3.14/site-packages/PyQt6/Qt6/plugins"
qt_lib_root = "/Users/cfertig/Documents/GitHub/testrepo/.venv/lib/python3.14/site-packages/PyQt6/Qt6/lib"
os.environ["QT_PLUGIN_PATH"] = plugin_root
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{plugin_root}/platforms"
os.environ.setdefault("QT_QPA_PLATFORM", "cocoa" if sys.platform == "darwin" else "offscreen")
if sys.platform == "darwin":
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = qt_lib_root
    os.environ["DYLD_FRAMEWORK_PATH"] = qt_lib_root
    os.environ["DYLD_LIBRARY_PATH"] = qt_lib_root

from pathlib import Path
from typing import Any

import PyQt6

pyqt_root = Path(PyQt6.__file__).resolve().parent
qt_plugins = pyqt_root / "Qt6" / "plugins"
qt_platforms = qt_plugins / "platforms"
if qt_platforms.exists():
    os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_platforms))

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PyQt6 import QtWidgets

from simulated_experimental_data import generate_simulated_data, load_and_plot_simulated_data


DEFAULT_SPEC = {
    "function_type": "Gaussian",
    "num_points": 1000,
    "center_index": 0.0,
    "sigma": 1.0,
    "amplitude": 1.0,
    "noise_std": 0.1,
    "random_seed": 42,
}


def build_simulation_spec(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "function_type": str(values["function_type"]),
        "num_points": int(values["num_points"]),
        "center_index": float(values["center_index"]),
        "sigma": float(values["sigma"]),
        "amplitude": float(values["amplitude"]),
        "noise_std": float(values["noise_std"]),
        "random_seed": int(values["random_seed"]),
    }


class SimulationPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Simulation Management", parent)
        self.setLayout(QtWidgets.QFormLayout())

        self.function_type = QtWidgets.QComboBox()
        self.function_type.addItems(["Gaussian", "Lorentzian"])
        self.layout().addRow("Function", self.function_type)

        self.num_points = QtWidgets.QLineEdit("1000")
        self.layout().addRow("Num points", self.num_points)

        self.center_index = QtWidgets.QLineEdit("0.0")
        self.layout().addRow("Center index", self.center_index)

        self.sigma = QtWidgets.QLineEdit("1.0")
        self.layout().addRow("Sigma", self.sigma)

        self.amplitude = QtWidgets.QLineEdit("1.0")
        self.layout().addRow("Amplitude", self.amplitude)

        self.noise_std = QtWidgets.QLineEdit("0.1")
        self.layout().addRow("Noise std", self.noise_std)

        self.random_seed = QtWidgets.QLineEdit("42")
        self.layout().addRow("Random seed", self.random_seed)

        self.run_button = QtWidgets.QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_requested)
        self.layout().addRow(self.run_button)

        self.status_label = QtWidgets.QLabel("Ready")
        self.layout().addRow(self.status_label)

    def values(self) -> dict[str, Any]:
        return {
            "function_type": self.function_type.currentText(),
            "num_points": self.num_points.text(),
            "center_index": self.center_index.text(),
            "sigma": self.sigma.text(),
            "amplitude": self.amplitude.text(),
            "noise_std": self.noise_std.text(),
            "random_seed": self.random_seed.text(),
        }

    def run_requested(self) -> None:
        self.status_label.setText("Running...")
        self.parent().run_simulation_from_controls()


class CsvDirectoryPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("CSV Directory", parent)
        self.setLayout(QtWidgets.QVBoxLayout())

        self.path_edit = QtWidgets.QLineEdit(str(Path.cwd()))
        self.layout().addWidget(self.path_edit)

        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_directory)
        self.layout().addWidget(self.browse_button)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.layout().addWidget(self.file_list)

        self.load_button = QtWidgets.QPushButton("Load Selected CSVs")
        self.load_button.clicked.connect(self.parent().load_selected_csvs)
        self.layout().addWidget(self.load_button)

    def choose_directory(self) -> None:
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose CSV directory")
        if selected_dir:
            self.path_edit.setText(selected_dir)
            self.refresh_files(selected_dir)

    def refresh_files(self, directory: str) -> None:
        self.file_list.clear()
        path = Path(directory)
        if not path.exists():
            return
        for csv_file in sorted(path.glob("*.csv")):
            self.file_list.addItem(str(csv_file.name))


class GraphPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Graph", parent)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout().addWidget(self.canvas)

        self.control_row = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Plot")
        self.save_button.clicked.connect(self.parent().save_current_plot)
        self.control_row.addWidget(self.save_button)

        self.export_button = QtWidgets.QPushButton("Export Data")
        self.export_button.clicked.connect(self.parent().export_current_data)
        self.control_row.addWidget(self.export_button)

        self.layout().addLayout(self.control_row)

    def set_plot(self, dataframe: pd.DataFrame) -> None:
        self.parent().current_dataframe = dataframe
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(dataframe["point"], dataframe["voltage"], marker="o", linestyle="-", color="tab:blue")
        ax.set_xlabel("point")
        ax.set_ylabel("voltage")
        ax.set_title("Loaded Simulation Data")
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation GUI")
        self.resize(1200, 800)

        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.left_panel = QtWidgets.QVBoxLayout()
        self.csv_panel = CsvDirectoryPanel(self)
        self.left_panel.addWidget(self.csv_panel)
        self.sim_panel = SimulationPanel(self)
        self.left_panel.addWidget(self.sim_panel)
        self.layout.addLayout(self.left_panel, 1)

        self.current_dataframe: pd.DataFrame | None = None
        self.graph_panel = GraphPanel(self)
        self.layout.addWidget(self.graph_panel, 3)

    def run_simulation_from_controls(self) -> None:
        spec = build_simulation_spec(self.sim_panel.values())
        self.sim_panel.status_label.setText(f"Running {spec['function_type']}...")
        output_path = Path.cwd() / f"simulated_{spec['function_type'].lower()}.csv"
        dataframe = generate_simulated_data(output_path=output_path, **spec)
        self.graph_panel.set_plot(dataframe)
        self.sim_panel.status_label.setText("Simulation complete")

    def save_current_plot(self, output_path: str | Path | None = None) -> None:
        if output_path is None:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Plot",
                str(Path.cwd() / "simulation_plot.png"),
                "PNG Files (*.png)",
            )
            if not output_path:
                return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_panel.figure.savefig(path, dpi=200)
        self.sim_panel.status_label.setText(f"Saved plot to {path}")

    def export_current_data(self, output_path: str | Path | None = None) -> None:
        if self.current_dataframe is None:
            self.sim_panel.status_label.setText("No data available to export")
            return

        if output_path is None:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export Data",
                str(Path.cwd() / "simulation_data.csv"),
                "CSV Files (*.csv)",
            )
            if not output_path:
                return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.current_dataframe.to_csv(path, index=False)
        self.sim_panel.status_label.setText(f"Exported data to {path}")

    def load_selected_csvs(self) -> None:
        selected_items = self.csv_panel.file_list.selectedItems()
        if not selected_items:
            return
        csv_files = [Path(self.csv_panel.path_edit.text()) / item.text() for item in selected_items]
        data_frames = []
        for csv_file in csv_files:
            if csv_file.exists():
                data_frames.append(pd.read_csv(csv_file))
        if data_frames:
            combined = pd.concat(data_frames, ignore_index=True)
            self.graph_panel.set_plot(combined)


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
