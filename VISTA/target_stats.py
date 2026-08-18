"""
Compute the Reinhard target LAB mean/std using a from-scratch float CIELAB
(D65, OpenCV sRGB coefficients) that will be mirrored exactly in Swift
(VISTAColor.swift). Baking these numbers keeps in-app normalization
self-consistent: target stats and per-tile stats use identical color math.

L in [0,100], a/b centered at 0 (matches the repo's lab_split ranges).
"""
import numpy as np
from skimage.io import imread

# OpenCV sRGB->XYZ (D65) matrix and white point
M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])
Xn, Yn, Zn = 0.950456, 1.0, 1.088754


def srgb_to_linear(c):
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def rgb_to_lab(rgb):
    c = srgb_to_linear(rgb.astype(np.float64) / 255.0)
    xyz = c @ M.T
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def main():
    target = imread("norm/TargetForNormalization-Copy1.tif")[:, :, :3]
    mask = (target[:, :, 0] <= 200) | (target[:, :, 1] <= 200) | (target[:, :, 2] <= 200)
    L, a, b = rgb_to_lab(target)
    means = (L[mask].mean(), a[mask].mean(), b[mask].mean())
    stds = (L[mask].std(), a[mask].std(), b[mask].std())
    print("// Reference float-LAB target stats (bake into VISTAColor.swift)")
    print("means L,a,b = (%.6f, %.6f, %.6f)" % means)
    print("stds  L,a,b = (%.6f, %.6f, %.6f)" % stds)


if __name__ == "__main__":
    main()
