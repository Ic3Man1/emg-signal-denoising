import numpy as np
from scipy.signal import butter, filtfilt

def butterworth_denoise(signal_1d, fs=2048.0, lowcut=20.0, highcut=400.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    denoised_signal = filtfilt(b, a, signal_1d)
    
    return denoised_signal