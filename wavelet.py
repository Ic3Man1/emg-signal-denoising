import pywt
import numpy as np

def wavelet_denoise(signal_1d, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = 0.2 * sigma * np.sqrt(2 * np.log(len(signal_1d)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_thresh, wavelet)[:len(signal_1d)]