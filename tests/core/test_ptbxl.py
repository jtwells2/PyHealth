"""Unit tests for PTBXLDataset and PTBXLMultilabelClassification.

Test strategy
-------------
* All tests are self-contained and run fully offline — no network calls, no
  real ECG data required.
* ``TestPTBXLDataset`` exercises ``prepare_metadata`` in isolation by creating
  a minimal temporary filesystem (tiny ``.hea`` header stubs + zero-byte
  ``.mat`` placeholders) and verifying the CSV produced.
* ``TestPTBXLMultilabelClassification`` exercises the task's ``__call__``
  method with synthetic in-memory ECG arrays, bypassing the dataset loading
  machinery entirely.  Both ``label_type`` variants and both ``sampling_rate``
  values are tested.

Author:
    CS-598 DLH Project Team
"""

import io
import os
import struct
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyhealth.datasets.ptbxl import PTBXLDataset
from pyhealth.tasks.ptbxl_multilabel_classification import (
    CHALLENGE_SNOMED_CLASSES,
    SNOMED_TO_SUPERDIAG,
    SUPERDIAG_CLASSES,
    PTBXLMultilabelClassification,
)


# ---------------------------------------------------------------------------
# Helpers for constructing a minimal fake filesystem
# ---------------------------------------------------------------------------

def _write_hea(path: Path, record_id: str, age: int, sex: str, dx: str) -> None:
    """Write a minimal WFDB-style .hea file with the required comment lines."""
    header = (
        f"{record_id} 12 500 5000\n"
        f"# Age: {age}\n"
        f"# Sex: {sex}\n"
        f"# Dx: {dx}\n"
    )
    path.write_text(header, encoding="utf-8")


def _write_mat(path: Path) -> None:
    """Write a zero-byte placeholder .mat file (sufficient for metadata tests)."""
    path.write_bytes(b"")


def _make_mat_bytes(signal: np.ndarray) -> bytes:
    """Produce a minimal scipy.io.savemat-compatible bytes object in memory.

    We use scipy.io.savemat rather than a home-rolled format so that loadmat
    can round-trip the data correctly.
    """
    import scipy.io
    buf = io.BytesIO()
    scipy.io.savemat(buf, {"val": signal})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fake Patient / Event for task tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeEvent:
    signal_file: str = ""
    scp_codes: str = ""
    age: int = 50
    sex: str = "Male"


class _FakePatient:
    def __init__(self, patient_id: str, events: List[_FakeEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type: str = None) -> List[_FakeEvent]:
        return self._events


# ---------------------------------------------------------------------------
# Dataset metadata tests
# ---------------------------------------------------------------------------

class TestPTBXLDataset(unittest.TestCase):
    """Test PTBXLDataset.prepare_metadata without touching BaseDataset.__init__."""

    def _make_ds(self, root: str) -> PTBXLDataset:
        """Instantiate PTBXLDataset bypassing BaseDataset initialisation."""
        ds = PTBXLDataset.__new__(PTBXLDataset)
        ds.root = root
        return ds

    # ------------------------------------------------------------------
    # Baseline: single group directory
    # ------------------------------------------------------------------

    def test_prepare_metadata_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g1 = root / "g1"
            g1.mkdir()

            _write_hea(g1 / "HR00001.hea", "HR00001", 56, "Female", "426783006,251146004")
            _write_mat(g1 / "HR00001.mat")
            _write_hea(g1 / "HR00002.hea", "HR00002", 42, "Male", "270492004")
            _write_mat(g1 / "HR00002.mat")

            ds = self._make_ds(tmp)
            ds.prepare_metadata()

            csv = root / "ptbxl-pyhealth.csv"
            self.assertTrue(csv.exists(), "ptbxl-pyhealth.csv should be written")

            df = pd.read_csv(csv)
            self.assertEqual(len(df), 2)
            self.assertIn("patient_id", df.columns)
            self.assertIn("record_id", df.columns)
            self.assertIn("signal_file", df.columns)
            self.assertIn("age", df.columns)
            self.assertIn("sex", df.columns)
            self.assertIn("scp_codes", df.columns)

            row = df[df["patient_id"] == "HR00001"].iloc[0]
            self.assertEqual(row["age"], 56)
            self.assertEqual(row["sex"], "Female")
            self.assertEqual(row["scp_codes"], "426783006,251146004")
            self.assertTrue(str(row["signal_file"]).endswith("HR00001.mat"))

    # ------------------------------------------------------------------
    # Multiple group directories
    # ------------------------------------------------------------------

    def test_prepare_metadata_multiple_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for g, rec_id in [("g1", "HR00001"), ("g2", "HR01001"), ("g3", "HR02001")]:
                gdir = root / g
                gdir.mkdir()
                _write_hea(gdir / f"{rec_id}.hea", rec_id, 30, "Male", "426783006")
                _write_mat(gdir / f"{rec_id}.mat")

            ds = self._make_ds(tmp)
            ds.prepare_metadata()

            df = pd.read_csv(root / "ptbxl-pyhealth.csv")
            self.assertEqual(len(df), 3)
            self.assertEqual(sorted(df["patient_id"].tolist()), sorted(["HR00001", "HR01001", "HR02001"]))

    # ------------------------------------------------------------------
    # Missing .mat → row is skipped
    # ------------------------------------------------------------------

    def test_prepare_metadata_skips_missing_mat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g1 = root / "g1"
            g1.mkdir()

            _write_hea(g1 / "HR00001.hea", "HR00001", 45, "Male", "426783006")
            # deliberately omit HR00001.mat
            _write_hea(g1 / "HR00002.hea", "HR00002", 30, "Female", "270492004")
            _write_mat(g1 / "HR00002.mat")

            ds = self._make_ds(tmp)
            ds.prepare_metadata()

            df = pd.read_csv(root / "ptbxl-pyhealth.csv")
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["patient_id"], "HR00002")

    # ------------------------------------------------------------------
    # Idempotency: calling prepare_metadata twice should not raise
    # ------------------------------------------------------------------

    def test_prepare_metadata_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g1 = root / "g1"
            g1.mkdir()
            _write_hea(g1 / "HR00001.hea", "HR00001", 50, "Male", "426783006")
            _write_mat(g1 / "HR00001.mat")

            ds = self._make_ds(tmp)
            ds.prepare_metadata()
            ds.prepare_metadata()  # second call should be a no-op

            df = pd.read_csv(root / "ptbxl-pyhealth.csv")
            self.assertEqual(len(df), 1)

    # ------------------------------------------------------------------
    # No .hea files → RuntimeError
    # ------------------------------------------------------------------

    def test_prepare_metadata_no_records_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "g1").mkdir()  # empty group dir

            ds = self._make_ds(tmp)
            with self.assertRaises(RuntimeError):
                ds.prepare_metadata()

    # ------------------------------------------------------------------
    # default_task property
    # ------------------------------------------------------------------

    def test_default_task_returns_superdiagnostic_instance(self):
        ds = PTBXLDataset.__new__(PTBXLDataset)
        task = ds.default_task
        self.assertIsInstance(task, PTBXLMultilabelClassification)
        self.assertEqual(task.label_type, "superdiagnostic")
        self.assertEqual(task.sampling_rate, 100)


# ---------------------------------------------------------------------------
# Task unit tests
# ---------------------------------------------------------------------------

class TestPTBXLMultilabelClassification(unittest.TestCase):
    """Test PTBXLMultilabelClassification.__call__ with synthetic ECG data."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_mat_file(self, tmp_dir: Path, name: str, signal: np.ndarray) -> str:
        """Write a scipy .mat file containing 'val' and return its path."""
        import scipy.io
        path = tmp_dir / name
        scipy.io.savemat(str(path), {"val": signal})
        return str(path)

    def _make_patient(self, signal_file: str, scp_codes: str) -> _FakePatient:
        event = _FakeEvent(signal_file=signal_file, scp_codes=scp_codes)
        return _FakePatient("p001", [event])

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------

    def test_invalid_sampling_rate_raises(self):
        with self.assertRaises(ValueError):
            PTBXLMultilabelClassification(sampling_rate=250)

    def test_invalid_label_type_raises(self):
        with self.assertRaises(ValueError):
            PTBXLMultilabelClassification(label_type="morphological")

    def test_task_names_are_unique(self):
        t_a = PTBXLMultilabelClassification(sampling_rate=100, label_type="superdiagnostic")
        t_b = PTBXLMultilabelClassification(sampling_rate=500, label_type="superdiagnostic")
        t_c = PTBXLMultilabelClassification(sampling_rate=100, label_type="diagnostic")
        t_d = PTBXLMultilabelClassification(sampling_rate=500, label_type="diagnostic")
        names = {t_a.task_name, t_b.task_name, t_c.task_name, t_d.task_name}
        self.assertEqual(len(names), 4, "All four configurations should have distinct task names.")

    # ------------------------------------------------------------------
    # Signal loading and decimation
    # ------------------------------------------------------------------

    def test_superdiagnostic_100hz_signal_shape(self):
        """Superdiagnostic task at 100 Hz should yield (12, 1000) signals."""
        with tempfile.TemporaryDirectory() as tmp:
            signal_500 = np.random.randn(12, 5000).astype(np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal_500)
            patient = self._make_patient(mat_path, "426783006")  # NORM

            task = PTBXLMultilabelClassification(sampling_rate=100, label_type="superdiagnostic")
            samples = task(patient)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["signal"].shape, (12, 1000))

    def test_superdiagnostic_500hz_signal_shape(self):
        """Superdiagnostic task at 500 Hz should yield (12, 5000) signals (no decimation)."""
        with tempfile.TemporaryDirectory() as tmp:
            signal_500 = np.random.randn(12, 5000).astype(np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal_500)
            patient = self._make_patient(mat_path, "426783006")  # NORM

            task = PTBXLMultilabelClassification(sampling_rate=500, label_type="superdiagnostic")
            samples = task(patient)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["signal"].shape, (12, 5000))

    def test_signal_dtype_is_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            signal_500 = np.random.randn(12, 5000).astype(np.float64)  # 64-bit input
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal_500)
            patient = self._make_patient(mat_path, "426783006")

            task = PTBXLMultilabelClassification()
            samples = task(patient)
            self.assertEqual(samples[0]["signal"].dtype, np.float32)

    # ------------------------------------------------------------------
    # Superdiagnostic label mapping
    # ------------------------------------------------------------------

    def test_superdiagnostic_normal_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            signal = np.zeros((12, 5000), dtype=np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            patient = self._make_patient(mat_path, "426783006")  # → NORM

            task = PTBXLMultilabelClassification(label_type="superdiagnostic")
            samples = task(patient)

            self.assertEqual(len(samples), 1)
            self.assertIn("NORM", samples[0]["labels"])

    def test_superdiagnostic_multilabel(self):
        """A recording with both AF (CD) and low QRS voltage (HYP) codes."""
        with tempfile.TemporaryDirectory() as tmp:
            signal = np.zeros((12, 5000), dtype=np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            # 164889003 → CD (atrial fibrillation), 251146004 → HYP (low QRS voltage)
            patient = self._make_patient(mat_path, "164889003,251146004")

            task = PTBXLMultilabelClassification(label_type="superdiagnostic")
            samples = task(patient)

            self.assertEqual(len(samples), 1)
            label_set = set(samples[0]["labels"])
            self.assertIn("CD", label_set)
            self.assertIn("HYP", label_set)

    def test_superdiagnostic_no_known_codes_skipped(self):
        """Recordings with no recognised superdiagnostic codes should be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            signal = np.zeros((12, 5000), dtype=np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            patient = self._make_patient(mat_path, "999999999")  # unknown code

            task = PTBXLMultilabelClassification(label_type="superdiagnostic")
            samples = task(patient)
            self.assertEqual(samples, [])

    # ------------------------------------------------------------------
    # Diagnostic (27-class) label mapping
    # ------------------------------------------------------------------

    def test_diagnostic_known_challenge_code(self):
        # 270492004 = First-degree AV block, part of Challenge 2020 scoring set
        with tempfile.TemporaryDirectory() as tmp:
            signal = np.zeros((12, 5000), dtype=np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            patient = self._make_patient(mat_path, "270492004")

            task = PTBXLMultilabelClassification(label_type="diagnostic")
            samples = task(patient)

            self.assertEqual(len(samples), 1)
            self.assertIn("270492004", samples[0]["labels"])

    def test_diagnostic_non_challenge_code_skipped(self):
        """A code not in the 27-class Challenge vocabulary should be filtered out."""
        with tempfile.TemporaryDirectory() as tmp:
            signal = np.zeros((12, 5000), dtype=np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            patient = self._make_patient(mat_path, "999999999")

            task = PTBXLMultilabelClassification(label_type="diagnostic")
            samples = task(patient)
            self.assertEqual(samples, [])

    def test_diagnostic_multiple_valid_codes(self):
        """Multiple Challenge codes in one recording should all appear as labels."""
        with tempfile.TemporaryDirectory() as tmp:
            signal = np.zeros((12, 5000), dtype=np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            # 164889003 (AF) and 426783006 (NSR) are both in Challenge vocabulary
            patient = self._make_patient(mat_path, "164889003,426783006")

            task = PTBXLMultilabelClassification(label_type="diagnostic")
            samples = task(patient)

            self.assertEqual(len(samples), 1)
            label_set = set(samples[0]["labels"])
            self.assertIn("164889003", label_set)
            self.assertIn("426783006", label_set)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_missing_signal_file_returns_empty(self):
        event = _FakeEvent(signal_file="", scp_codes="426783006")
        patient = _FakePatient("p001", [event])
        task = PTBXLMultilabelClassification()
        self.assertEqual(task(patient), [])

    def test_nonexistent_signal_file_returns_empty(self):
        event = _FakeEvent(signal_file="/nonexistent/path/to/rec.mat", scp_codes="426783006")
        patient = _FakePatient("p001", [event])
        task = PTBXLMultilabelClassification()
        self.assertEqual(task(patient), [])

    def test_empty_patient_no_events(self):
        patient = _FakePatient("p001", [])
        task = PTBXLMultilabelClassification()
        self.assertEqual(task(patient), [])

    def test_wrong_signal_shape_skipped(self):
        """Signals that are not 2-D with 12 channels should be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            # Write a single-channel signal (shape 1×5000)
            signal = np.zeros((1, 5000), dtype=np.float32)
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            event = _FakeEvent(signal_file=mat_path, scp_codes="426783006")
            patient = _FakePatient("p001", [event])

            task = PTBXLMultilabelClassification(label_type="superdiagnostic")
            samples = task(patient)
            self.assertEqual(samples, [])

    # ------------------------------------------------------------------
    # Label-space constants sanity checks
    # ------------------------------------------------------------------

    def test_superdiag_classes_count(self):
        self.assertEqual(len(SUPERDIAG_CLASSES), 5)
        self.assertEqual(set(SUPERDIAG_CLASSES), {"NORM", "MI", "STTC", "CD", "HYP"})

    def test_challenge_classes_count(self):
        self.assertEqual(len(CHALLENGE_SNOMED_CLASSES), 27)

    def test_snomed_to_superdiag_values(self):
        valid_classes = set(SUPERDIAG_CLASSES)
        for code, cls in SNOMED_TO_SUPERDIAG.items():
            self.assertIn(
                cls,
                valid_classes,
                f"SNOMED code {code} maps to unknown class '{cls}'",
            )


if __name__ == "__main__":
    unittest.main()
