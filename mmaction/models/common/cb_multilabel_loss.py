import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from mmaction.registry import MODELS
from mmengine.model import BaseModule


class CBFocalLossMultilabel(nn.Module):
    """
    Class-Balanced Focal Loss for multi-label classification.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    Cui et al., CVPR 2019.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: Tensor, labels: Tensor, alpha: Tensor) -> Tensor:
        """
        Args:
            logits: [B, C] raw logits
            labels: [B, C] multi-hot float targets
            alpha:  [B, C] per-class CB weights
        Returns:
            [B, C] weighted focal loss (unreduced)
        """
        bce = F.binary_cross_entropy_with_logits(
            input=logits, target=labels, reduction="none"
        )

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(
                -self.gamma * labels * logits
                - self.gamma * torch.log(1 + torch.exp(-logits))
            )

        return alpha * modulator * bce


@MODELS.register_module()
class CBMultilabelCrossEntropy(BaseModule):
    """
    Drop-in replacement for MultilableCrossEntropy that uses
    Class-Balanced Focal Loss for the behaviour classification head.

    The objectness branch (column 0 when present) remains standard
    BCE so that detection is unaffected. Only the behaviour columns
    get CB reweighting.

    Config example:
        loss_cls=dict(
            type='CBMultilabelCrossEntropy',
            loss_weight=2.0,
            num_classes=17,
            mask_cls=False,
            no_obj_mode=True,
            # --- new CB params ---
            samples_per_cls=[
                5000,   # camera_interaction
                800,    # climbing_down
                3000,   # climbing_up
                12000,  # walking
                1500,   # running
                40000,  # sitting
                600,    # sitting_on_back
                25000,  # standing
                4000,   # hanging
                8000,   # grooming
                7000,   # being_groomed
                2000,   # touch
                200,    # leading
                150,    # following
                1000,   # chimp_carrying
                900,    # being_carried
            ],
            beta=0.99,
            gamma=2.0,
        ),
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        loss_weight: float = 1.0,
        num_classes: int = 17,
        mask_cls: bool = False,
        no_obj_mode: bool = False,
        extra_obj_mode: bool = False,
        focal: bool = True,
        # ── CB-specific parameters ──
        samples_per_cls: Optional[List[int]] = None,
        beta: float = 0.99,
        gamma: float = 2.0,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.mask_cls = mask_cls
        self.no_obj_mode = no_obj_mode
        self.extra_obj_mode = extra_obj_mode
        self.cls_mask_prob = 0.1
        self.focal = focal

        # Standard BCE for the objectness branch (unchanged)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        # CB focal loss for behaviour branch
        self.beta = beta
        self.gamma = gamma
        self.cb_focal = CBFocalLossMultilabel(gamma=gamma)

        # Precompute CB weights
        if samples_per_cls is not None:
            self._register_cb_weights(samples_per_cls)
        else:
            # Fallback: uniform weights (equivalent to standard focal loss)
            self.register_buffer("cb_weights", None)

    def _register_cb_weights(self, samples_per_cls: List[int]):
        """Compute and register effective-number class-balanced weights."""
        samples = np.array(samples_per_cls, dtype=np.float64)
        num_cls = len(samples)

        effective_num = 1.0 - np.power(self.beta, samples)
        weights = (1.0 - self.beta) / effective_num
        # Normalise so weights average to 1 (loss magnitude stays comparable)
        weights = weights / np.sum(weights) * num_cls

        # Store as buffer so it moves with .to(device) automatically
        self.register_buffer(
            "cb_weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    def _compute_cls_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Behaviour classification loss with CB focal weighting.

        Args:
            logits: [B, num_behaviour_classes]
            labels: [B, num_behaviour_classes] multi-hot
        Returns:
            [B, num_behaviour_classes] unreduced loss
        """
        if self.cb_weights is not None:
            # [1, C] -> broadcast to [B, C]
            alpha = self.cb_weights.unsqueeze(0).expand_as(logits)
        else:
            alpha = torch.ones_like(logits)

        if self.focal:
            return self.cb_focal(logits, labels, alpha)
        else:
            # CB-weighted BCE without focal modulation
            bce = self.criterion(logits, labels)
            return alpha * bce

    def forward(
        self,
        x: Tensor,
        target: Union[Tuple[Tensor, Tensor], Tensor],
        obj_weight=None,
        cls_weight=None,
        avg_factor=None,
        cls_masks=None,
        **kwargs,
    ):
        if isinstance(target, tuple):
            labels, scores = target
        else:
            labels = target

        x = x.float()
        labels = labels.float()

        # ── Split objectness (col 0) from behaviour classes ──
        if self.extra_obj_mode:
            loss_obj = self.criterion(x[:, 0], labels[:, 0])
            loss_cls = self._compute_cls_loss(x[:, 1:], labels[:, 1:])

        else:
            if x.size(1) != self.num_classes:
                # Column 0 = objectness, columns 1: = behaviours
                loss_obj = self.criterion(x[:, 0], labels[:, 0])
                loss_cls = self._compute_cls_loss(x[:, 1:], labels[:, 1:])
            else:
                # No objectness column, all columns are behaviours
                loss_obj = 0
                loss_cls = self._compute_cls_loss(x, labels)

        # ── Apply per-sample obj / cls weighting (from matcher) ──
        if obj_weight is not None and cls_weight is not None:
            if not self.no_obj_mode:
                obj_weight = obj_weight.reshape(x.size(0),).float()
                cls_weight = cls_weight.reshape(x.size(0), 1).float()
            else:
                obj_weight = 0 if not self.extra_obj_mode else 1
                cls_weight = 1
            loss_obj = loss_obj * obj_weight
            loss_cls = loss_cls * cls_weight

        # ── Optional class masking ──
        if cls_masks is not None and self.mask_cls:
            cls_masks = cls_masks.reshape(len(loss_cls), 1)
            cls_masks = 1 - (
                (torch.rand_like(cls_masks) * (1 - cls_masks)) >= self.cls_mask_prob
            ).float()
            loss_cls = loss_cls * cls_masks

        # ── Reduce ──
        loss_cls = loss_cls.mean(dim=1)

        if avg_factor is not None:
            loss = (loss_cls + loss_obj).sum() * avg_factor
        else:
            loss = (loss_cls + loss_obj).mean()

        return loss * self.loss_weight
