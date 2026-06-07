from scipy.signal import savgol_filter
import numpy as np
from math import factorial

def savgol_denoise(signal_1d, window_length, order):
    """Denoise a 1D signal using the Savitzky-Golay smoothing filter.

    The function fits a local polynomial of the given order within a sliding
    window and uses the central polynomial coefficient to smooth the signal.
    Window length must be odd and greater than the polynomial order.
    """
    if window_length % 2 == 0:
        raise ValueError("Rozmiar okna (window_length) musi być liczbą nieparzystą.")
    if window_length <= order:
        raise ValueError("Rozmiar okna musi być większy niż rząd wielomianu (order).")
        
    half_window = window_length // 2

    z = np.arange(-half_window, half_window + 1)

    J = np.vstack([z**j for j in range(order + 1)]).T

    J_pinv = np.linalg.pinv(J)

    coeffs = J_pinv[0]

    signal_padded = np.pad(signal_1d, half_window, mode='reflect')

    signal_denoised = np.convolve(signal_padded, coeffs[::-1], mode='valid')
    
    return signal_denoised
