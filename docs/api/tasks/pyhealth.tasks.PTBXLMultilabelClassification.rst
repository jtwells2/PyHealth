pyhealth.tasks.PTBXLMultilabelClassification
============================================

PTB-XL is a large publicly available 12-lead ECG dataset annotated with SNOMED-CT codes.
This task turns a :class:`~pyhealth.datasets.PTBXLDataset` into a **multi-label classification** problem.

Two label spaces are supported via the ``label_type`` argument:

- ``"superdiagnostic"`` — 5 coarse diagnostic classes (NORM, MI, STTC, CD, HYP)
- ``"diagnostic"`` — 27 SNOMED-CT classes scored in the PhysioNet / CinC Challenge 2020

The ``sampling_rate`` argument (100 or 500 Hz) controls temporal resolution, enabling
an ablation study across both axes.

.. autoclass:: pyhealth.tasks.PTBXLMultilabelClassification
    :members:
    :undoc-members:
    :show-inheritance:
