"""Abstract detector interface."""

from __future__ import annotations

import abc

import numpy as np

from src.utils.types import DetectionResult


class BaseDetector(abc.ABC):

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights and initialize inference runtime."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        Run detection on a BGR image.

        Returns list of DetectionResult, each with a pixel-level mask.
        Empty list when no detections are found.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Release GPU/CPU memory."""

    def __enter__(self) -> "BaseDetector":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
