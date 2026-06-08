import csv
import numpy as np

from mpc_python.mpc_demo_nosim import save_history_csv


def test_save_history_csv(tmp_path):
    filename = tmp_path / "mpc_history.csv"
    history = {
        "x": [np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
        "u": [np.array([0.1, 0.01])],
        "xref": [np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
        "error": [0.0],
    }

    save_history_csv(history, str(filename))
    assert filename.exists()

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0][0] == "step"
    assert len(rows) >= 2
