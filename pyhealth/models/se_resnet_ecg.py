"""SE-ResNet-50 for 12-lead ECG multi-label classification.

This module provides :class:`SEResNetECG`, a :class:`~pyhealth.models.BaseModel`
subclass implementing the Squeeze-Excitation ResNet (SE-ResNet-50) benchmarked in:

    Nonaka, N. & Seita, J. (2021). *In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis.* Proceedings of Machine Learning
    Research 126:1–19, MLHC 2021.

From the paper (Appendix A.3):
    "SE-ResNet is a ResNet with Squeeze-Excitation module (Hu et al., 2018).
    Akin to ResNet and ResNeXt architecture, we implemented the same structure
    to architecture used for image classification and replaced convolution and
    batch normalization layer to 1d."

Best hyperparameters from paper grid search (SE-ResNet-50, PTB-XL "all" task):
    batch_size = 64, learning_rate = 0.01  →  ROC-AUC = 0.9082

Architecture overview
---------------------
The SE block (Hu et al., 2018) adds channel-wise recalibration to every
residual block:

1. **Squeeze**: Global average pooling collapses spatial / temporal dimension
   ``(B, C, T) → (B, C)``.
2. **Excitation**: Two-layer MLP with reduction ratio ``r=16`` learns
   channel weights ``(B, C) → (B, C//r) → (B, C)``.
3. **Scale**: Element-wise multiply weights back to the feature map.

The backbone follows ResNet-50 (layers = [3, 4, 6, 3]) with all 2D Conv /
BatchNorm replaced by 1D equivalents for ECG signals.

::

    Input  (B, 12, T)                             (12 leads × T samples)
      → Conv1d(12, 64, 7, s=2) → BN → ReLU → MaxPool1d(3, s=2)
      → Layer 1 : 3 × Bottleneck1D(64,  64,  256)   [no stride]
      → Layer 2 : 4 × Bottleneck1D(256, 128, 512)   [stride=2 on first]
      → Layer 3 : 6 × Bottleneck1D(512, 256, 1024)  [stride=2 on first]
      → Layer 4 : 3 × Bottleneck1D(1024,512, 2048)  [stride=2 on first]
      → AdaptiveAvgPool1d(1) → Flatten → (B, 2048)
      → neck  : Linear(2048, backbone_out=256)       [paper: all backbones output 256]
      → head  : Linear(256, 256) → ReLU → BN1d → Dropout(0.5) → Linear(256, K)
      → BCEWithLogitsLoss (multi-label sigmoid per class)

Bottleneck SE block structure
-------------------------------
Each Bottleneck1D block::

    identity = x  (or downsampled if stride > 1 / dim mismatch)
    x  →  Conv1d(in, planes, 1) → BN → ReLU
       →  Conv1d(planes, planes, 3, padding=1) → BN → ReLU
       →  Conv1d(planes, planes*4, 1) → BN
       →  SEBlock1D(planes*4, r=se_reduction)   ← channel recalibration
       →  + identity  → ReLU

Signal format expected
----------------------
``feature_keys=["signal"]`` → each element is ``np.ndarray`` of shape
``(12, T)`` loaded from a ``.pkl`` file by ``SampleSignalDataset``.
T = 1000 at 100 Hz  |  T = 5000 at 500 Hz.

PyHealth integration
--------------------
``SEResNetECG`` is a drop-in replacement for ``SparcNet`` and ``BiLSTMECG``
in any PyHealth ``Trainer``-based pipeline::

    model = SEResNetECG(
        dataset      = sample_dataset,
        feature_keys = ["signal"],
        label_key    = "labels",
        mode         = "multilabel",
    )
    trainer = Trainer(model=model, ...)

Author:
    CS-598 DLH Project Team — PyHealth contribution  (April 2026)

References:
    - Nonaka & Seita (2021). In-depth Benchmarking of Deep Neural Network
      Architectures for ECG Diagnosis. MLHC 2021.
    - Hu et al. (2018). Squeeze-and-Excitation Networks. CVPR 2018.
    - He et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


# ── Squeeze-and-Excitation module (Hu et al. 2018) ────────────────────────────

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D temporal signals.

    Performs channel recalibration via global pooling + two-layer MLP:

    1. **Squeeze**: ``AdaptiveAvgPool1d(1)  →  (B, C)``
    2. **Excitation**: ``Linear(C, C//r) → ReLU → Linear(C//r, C) → Sigmoid``
    3. **Scale**: element-wise multiply back onto the input feature map.

    Args:
        channels (int): Number of input/output channels ``C``.
        reduction (int): Reduction ratio ``r`` for the MLP bottleneck.
            Default ``16`` (as in the original SE-Net paper).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(channels, mid)
        self.fc2     = nn.Linear(mid, channels)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, T)`` feature map.

        Returns:
            Channel-recalibrated feature map ``(B, C, T)``.
        """
        b, c, _ = x.shape
        # Squeeze: global average pool → (B, C)
        z = self.pool(x).view(b, c)
        # Excitation: MLP → channel weights in [0, 1]
        z = self.relu(self.fc1(z))
        z = self.sigmoid(self.fc2(z))
        # Scale: broadcast back to (B, C, T)
        return x * z.unsqueeze(-1)


# ── ResNet-50 Bottleneck block (1D, with SE) ───────────────────────────────────

class Bottleneck1D(nn.Module):
    """ResNet-50 Bottleneck residual block adapted to 1D signals with SE.

    Bottleneck design (He et al., 2016):
        Conv1d(in→planes, k=1) → BN → ReLU    # reduce channels
        Conv1d(planes→planes, k=3) → BN → ReLU # spatial mix
        Conv1d(planes→planes*4, k=1) → BN      # expand channels
        SEBlock1D(planes*4)                     # channel recalibration (Hu 2018)
        + identity (possibly downsampled) → ReLU

    Args:
        in_channels (int): Input channel count.
        planes (int): Bottleneck width (output = ``planes * expansion = planes * 4``).
        stride (int): Stride for the 3×1 convolution. ``>1`` downsamples T.
        downsample (nn.Module, optional): 1×1 conv to align identity when
            ``in_channels != planes * 4`` or ``stride != 1``.
        se_reduction (int): SE block reduction ratio. Default ``16``.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels:  int,
        planes:       int,
        stride:       int = 1,
        downsample:   Optional[nn.Module] = None,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # 1×1 channel reduce
        self.conv1 = nn.Conv1d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        # 3×1 temporal mix (stride here for downsampling)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        # 1×1 channel expand
        self.conv3 = nn.Conv1d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(out_channels)
        # Channel recalibration
        self.se    = SEBlock1D(out_channels, reduction=se_reduction)
        # Residual shortcut
        self.downsample = downsample
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)                       # squeeze-and-excite

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


# ── SEResNetECG — PyHealth BaseModel wrapper ───────────────────────────────────

class SEResNetECG(BaseModel):
    """SE-ResNet-50 ECG classifier (Nonaka & Seita 2021, Appendix A.3).

    Backone: ResNet-50 (layers = [3, 4, 6, 3]) with all 2D Conv/BN replaced
    by 1D equivalents and Squeeze-Excitation blocks after every Bottleneck.
    A neck projects the 2048-dim pooled feature to ``backbone_out = 256``
    (matching the paper: "We set output size of all backbone module to 256").
    The prediction head (paper: FC → ReLU → BN → Dropout → FC) maps to ``K``
    output logits for multi-label classification.

    Args:
        dataset: A PyHealth ``SampleSignalDataset`` (or ``Subset``) that
            exposes ``input_info["signal"]["n_channels"]``.
        feature_keys (List[str]): Must be ``["signal"]``.
        label_key (str): Sample key holding the label list (``"labels"`` for
            the jtwells2 / PTBXLMultilabelClassification schema).
        mode (str): ``"multilabel"`` applies ``BCEWithLogitsLoss``.
        layers (List[int]): Block counts per stage. Default ``[3, 4, 6, 3]``
            (ResNet-50).  Use ``[2, 2, 2, 2]`` for a lighter SE-ResNet-18.
        se_reduction (int): SE block reduction ratio. Paper uses ``16``.
        backbone_out (int): Dimension of the neck projection. Paper uses
            ``256`` for all backbone architectures.
        dropout (float): Dropout probability in the prediction head. ``0.5``
            matches the paper's training setting.

    Examples:
        Paper-aligned SE-ResNet-50::

            >>> from se_resnet_ecg import SEResNetECG
            >>> model = SEResNetECG(
            ...     dataset      = sample_dataset,
            ...     feature_keys = ["signal"],
            ...     label_key    = "labels",
            ...     mode         = "multilabel",
            ... )

        Lighter SE-ResNet-18 variant::

            >>> model = SEResNetECG(
            ...     dataset      = sample_dataset,
            ...     feature_keys = ["signal"],
            ...     label_key    = "labels",
            ...     mode         = "multilabel",
            ...     layers       = [2, 2, 2, 2],
            ... )
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key:    str,
        mode:         str,
        layers:       List[int] = None,
        se_reduction: int       = 16,
        backbone_out: int       = 256,
        dropout:      float     = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset      = dataset,
            feature_keys = feature_keys,
            label_key    = label_key,
            mode         = mode,
        )

        if layers is None:
            layers = [3, 4, 6, 3]   # ResNet-50 default

        sig_info    = self.dataset.input_info["signal"]
        in_channels = sig_info["n_channels"]  # 12 leads

        self.label_tokenizer = self.get_label_tokenizer()
        output_size          = self.get_output_size(self.label_tokenizer)

        # ── Stem (ResNet conv1) ────────────────────────────────────────────
        self._in_channels = 64
        self.conv1   = nn.Conv1d(in_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm1d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ── Residual stages ────────────────────────────────────────────────
        self.layer1 = self._make_layer(64,  layers[0], se_reduction=se_reduction)
        self.layer2 = self._make_layer(128, layers[1], stride=2, se_reduction=se_reduction)
        self.layer3 = self._make_layer(256, layers[2], stride=2, se_reduction=se_reduction)
        self.layer4 = self._make_layer(512, layers[3], stride=2, se_reduction=se_reduction)

        # ── Neck: project 2048 → backbone_out (256 per paper) ─────────────
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.neck    = nn.Linear(512 * Bottleneck1D.expansion, backbone_out)

        # ── Prediction head (paper: FC → ReLU → BN → Dropout → FC) ────────
        self.head = nn.Sequential(
            nn.Linear(backbone_out, backbone_out),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(backbone_out),
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out, output_size),
        )

        # Weight initialisation (Kaiming uniform for conv, constant for BN)
        self._init_weights()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_layer(
        self,
        planes:       int,
        num_blocks:   int,
        stride:       int = 1,
        se_reduction: int = 16,
    ) -> nn.Sequential:
        """Build one ResNet stage with ``num_blocks`` Bottleneck1D blocks."""
        downsample = None
        out_channels = planes * Bottleneck1D.expansion
        if stride != 1 or self._in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self._in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers_list = [
            Bottleneck1D(self._in_channels, planes, stride, downsample, se_reduction)
        ]
        self._in_channels = out_channels
        for _ in range(1, num_blocks):
            layers_list.append(
                Bottleneck1D(self._in_channels, planes,
                             se_reduction=se_reduction)
            )
        return nn.Sequential(*layers_list)

    def _init_weights(self) -> None:
        """Kaiming normal init for Conv1d; constant init for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, **kwargs) -> dict:
        """Run inference and compute loss.

        Keyword arguments are the batch dict produced by PyHealth's
        ``collate_fn_dict`` (keys: ``"signal"``, ``"labels"``, etc.).

        Returns:
            dict with keys:
                - ``"loss"``   – scalar ``BCEWithLogitsLoss``
                - ``"y_prob"`` – sigmoid probabilities  ``(B, K)``
                - ``"y_true"`` – multi-hot ground truth  ``(B, K)``
                - ``"logit"``  – raw logits              ``(B, K)``
        """
        # Load signal: List[np.ndarray(12, T)] → Tensor(B, 12, T)
        x = torch.tensor(
            np.array(kwargs[self.feature_keys[0]]),
            device=self.device,
        ).float()

        # ── Stem ────────────────────────────────────────────────────────────
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # (B, 64, T//4)

        # ── Residual stages ─────────────────────────────────────────────────
        x = self.layer1(x)   # (B, 256,  T//4)
        x = self.layer2(x)   # (B, 512,  T//8)
        x = self.layer3(x)   # (B, 1024, T//16)
        x = self.layer4(x)   # (B, 2048, T//32)

        # ── Pooling + neck ──────────────────────────────────────────────────
        x = self.avgpool(x).squeeze(-1)   # (B, 2048)
        x = self.neck(x)                  # (B, 256)

        # ── Prediction head ─────────────────────────────────────────────────
        logits = self.head(x)             # (B, K)

        # ── Loss and probabilities (handled by BaseModel) ───────────────────
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss   = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
