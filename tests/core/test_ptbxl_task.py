"""Unit tests for PTBXLMultilabelClassification task.

Test strategy
-------------
* All tests are self-contained and run fully offline — no network calls, no
  real ECG data required.
* Synthetic in-memory ECG arrays are written to temporary ``.mat`` files via
  ``scipy.io.savemat`` so that the task's ``loadmat`` call round-trips cleanly.
* ``_FakeEvent`` mirrors the event attributes produced by
  ``PTBXLDataset.load_data()``: ``mat`` (file path) and ``dx_codes``
  (SNOMED-CT codes joined by ``"."``).
* Both ``label_type`` variants (``"superdiagnostic"`` / ``"diagnostic"``) and
  both ``sampling_rate`` values (100 / 500) are exercised.

Author:
    CS-598 DLH Project Team
"""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from pyhealth.tasks.ptbxl_multilabel_classification import (
    CHALLENGE_SNOMED_CLASSES,
    SNOMED_TO_SUPERDIAG,
    SUPERDIAG_CLASSES,
    PTBXLMultilabelClassification,
)


# ---------------------------------------------------------------------------
# Fake Patient / Event
# ---------------------------------------------------------------------------

@dataclass
class _FakeEvent:
    """Minimal stand-in for a PyHealth Event with PTB-XL attributes.

    Attribute names match what PTBXLDataset.load_data() produces after the
    ``ptbxl/`` table-prefix is stripped by BaseDataset:
        - ``mat``      ← column ``ptbxl/mat``   (path to .mat signal file)
        - ``dx_codes`` ← column ``ptbxl/dx_codes`` (SNOMED codes, dot-joined)
    """
    mat: str = ""
    dx_codes: str = ""
    age: int = 50
    sex: str = "Male"


class _FakePatient:
    def __init__(self, patient_id: str, events: List[_FakeEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type: str = None) -> List[_FakeEvent]:
        return self._events


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

    def _make_patient(self, mat_path: str, dx_codes: str) -> _FakePatient:
        event = _FakeEvent(mat=mat_path, dx_codes=dx_codes)
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
            # dx_codes are dot-joined (load_data uses ".".join(dx))
            patient = self._make_patient(mat_path, "164889003.251146004")

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
            # dot-joined as produced by load_data()
            patient = self._make_patient(mat_path, "164889003.426783006")

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
        event = _FakeEvent(mat="", dx_codes="426783006")
        patient = _FakePatient("p001", [event])
        task = PTBXLMultilabelClassification()
        self.assertEqual(task(patient), [])

    def test_nonexistent_signal_file_returns_empty(self):
        event = _FakeEvent(mat="/nonexistent/path/to/rec.mat", dx_codes="426783006")
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
            signal = np.zeros((1, 5000), dtype=np.float32)  # single-channel, not 12
            mat_path = self._make_mat_file(Path(tmp), "rec.mat", signal)
            event = _FakeEvent(mat=mat_path, dx_codes="426783006")
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
