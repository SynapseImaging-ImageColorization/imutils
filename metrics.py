import numpy as np


def ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    """Computes the structural similarity index between two images.

    Args:
        im1 (np.ndarray): First image.
        im2 (np.ndarray): Second image.

    Returns:
        float: Structural similarity index.
    """
    return 0


def psnr(im1: np.ndarray, im2: np.ndarray) -> float:
    """
    Computes the peak signal-to-noise ratio between two images.

    Args:
        im1 (np.ndarray): First image.
        im2 (np.ndarray): Second image.

    Returns:
        float: Peak signal-to-noise ratio.
    """
    return 0


def fid(im1: np.ndarray, im2: np.ndarray) -> float:
    """
    Computes the Frechet Inception Distance between two images.

    Args:
        im1 (np.ndarray): First image.
        im2 (np.ndarray): Second image.

    Returns:
        float: Frechet Inception Distance.
    """
    return 0
