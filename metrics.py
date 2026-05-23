import numpy as np
import scipy

def snr(clean, denoised):
    noise = clean - denoised
    return 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))

def median_frequency(signal_1d, fs=2048):
    freqs, psd = scipy.signal.welch(signal_1d, fs=fs, nperseg=512)
    
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    
    mf_index = np.searchsorted(cumulative_power, total_power / 2)
    return freqs[mf_index]