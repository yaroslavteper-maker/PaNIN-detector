"""
Dependency-free reimplementation of the two color-normalization steps used by
the MicePan / VISTA pipeline, so we don't need staintools/spams (which pull in
an old, hard-to-build native stack).

1. luminosity_standardize()  -- port of staintools 2.1.2 LuminosityStandardizer
2. ReinhardColorNormalizer   -- copied verbatim from the repo's ProcessImages.py

Both operate on RGB uint8 arrays and use OpenCV's LAB conversion, matching the
original exactly.
"""
import numpy as np
import cv2 as cv


def _is_uint8(I):
    return isinstance(I, np.ndarray) and I.dtype == np.uint8


def luminosity_standardize(I, percentile=95):
    """staintools.LuminosityStandardizer.standardize, reimplemented."""
    assert _is_uint8(I), "Should be an RGB uint8 image"
    lab = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(float)
    p = np.percentile(L, percentile)
    lab[:, :, 0] = np.clip(255.0 * (L / p), 0, 255).astype(np.uint8)
    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)


class ReinhardColorNormalizer(object):
    """Verbatim from ProcessImages.py (Reinhard color transfer in LAB)."""

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target, mask):
        self.target_means, self.target_stds = self.get_mean_std(target, mask)

    def transform(self, I, mask):
        I1, I2, I3 = self.lab_split(I)
        means, stds = self.get_mean_std(I, mask)
        n1 = ((I1 - means[0]) * ((self.target_stds[0] + 1e-6) / (stds[0] + 1e-6))) + self.target_means[0]
        n2 = ((I2 - means[1]) * ((self.target_stds[1] + 1e-6) / (stds[1] + 1e-6))) + self.target_means[1]
        n3 = ((I3 - means[2]) * ((self.target_stds[2] + 1e-6) / (stds[2] + 1e-6))) + self.target_means[2]
        return self.merge_back(n1, n2, n3)

    @staticmethod
    def lab_split(I):
        assert _is_uint8(I), "Should be an RGB uint8 image"
        I = cv.cvtColor(I, cv.COLOR_RGB2LAB).astype(np.float32)
        I1, I2, I3 = cv.split(I)
        I1 /= 2.55      # -> [0,100]
        I2 -= 128.0     # -> [-127,127]
        I3 -= 128.0
        return I1, I2, I3

    @staticmethod
    def merge_back(I1, I2, I3):
        I1 = I1 * 2.55
        I2 = I2 + 128.0
        I3 = I3 + 128.0
        I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv.cvtColor(I, cv.COLOR_LAB2RGB)

    def get_mean_std(self, I, mask):
        assert _is_uint8(I), "Should be an RGB uint8 image"
        I1, I2, I3 = self.lab_split(I)
        means = (np.mean(I1[mask]), np.mean(I2[mask]), np.mean(I3[mask]))
        stds = (np.std(I1[mask]), np.std(I2[mask]), np.std(I3[mask]))
        return means, stds


def tissue_mask(I):
    """Non-white mask used by the pipeline: any channel <= 200."""
    return (I[:, :, 0] <= 200) | (I[:, :, 1] <= 200) | (I[:, :, 2] <= 200)
