from __future__ import annotations

import numpy as np


class Normalizer:
    """A normalizer designed for policy observation normalization."""

    def __init__(
        self,
        load_path: str | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        eps: np.ndarray | None = None,
    ):
        """
        NOTE: either load_path or mean, std, eps should be provided.
            No error checking is done here.
        """
        if load_path is not None:
            self.load_normalizer(load_path)
        else:
            self.mean = mean
            self.std = std
            self.eps = eps

    def load_normalizer(self, load_path: str):
        data = np.load(load_path)
        self.mean = data["mean"].squeeze(0)
        self.std = data["std"].squeeze(0)
        self.eps = data["eps"].squeeze(0)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    def split(self, slices: list[slice]) -> Normalizer:
        """Split the normalizer into a smaller normalizer.
        Args:
            slices: A list of slices, each slice is the index of the normalizer to split.
        Returns:
            A smaller normalizer that concatenates the normalizer along the given slices.
        """
        mean = np.concatenate([self.mean[s] for s in slices])
        std = np.concatenate([self.std[s] for s in slices])
        eps = np.concatenate([self.eps[s] for s in slices])
        return Normalizer(mean=mean, std=std, eps=eps)
