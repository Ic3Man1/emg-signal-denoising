import numpy as np
import scipy.signal


def snr(clean, denoised):
    """Compute signal-to-noise ratio (SNR) in decibels for a denoised signal."""
    noise = clean - denoised
    return 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))


def median_frequency(signal_1d, fs=2048):
    """Compute the median frequency of the PSD for the signal.

    The median frequency is useful for assessing muscle fatigue and denoising
    quality in EMG signals.
    """
    freqs, psd = scipy.signal.welch(signal_1d, fs=fs, nperseg=512)
    cumulative_power = np.cumsum(psd) / np.sum(psd)
    mf_index = np.searchsorted(cumulative_power, 0.5)
    return freqs[min(mf_index, len(freqs) - 1)]

def prd(clean, denoised):
    """Compute the percent root-mean-square difference between clean and denoised signals.

    This normalized RMSE metric is amplitude-independent and commonly used in
    EMG/ECG denoising literature. Lower values indicate better fidelity.
    """
    return 100.0 * np.sqrt(np.sum((clean - denoised) ** 2) / np.sum(clean ** 2))


def correlation_coefficient(clean, denoised):
    """Compute Pearson correlation coefficient between clean and denoised signals.

    This measures waveform similarity independently of amplitude scaling.
    Values near 1 mean the denoised signal preserves the original shape.
    """
    return np.corrcoef(clean, denoised)[0, 1]


def snr_improvement(clean, noisy, denoised):
    """Compute the SNR improvement in decibels after denoising."""
    return snr(clean, denoised) - snr(clean, noisy)

def spectral_distortion(clean, denoised, fs=2048, f_min=10, f_max=500):
    """Compute the average spectral distortion between clean and denoised PSDs
    within the useful EMG frequency range (default: 10 - 500 Hz).

    The metric measures how much the denoising changes the PSD shape. Values
    closer to zero indicate lower spectral distortion.
    """
    freqs, psd_clean    = scipy.signal.welch(clean,    fs=fs, nperseg=512)
    _,     psd_denoised = scipy.signal.welch(denoised, fs=fs, nperseg=512)

    band_mask = (freqs >= f_min) & (freqs <= f_max)
    
    psd_clean_band = psd_clean[band_mask]
    psd_denoised_band = psd_denoised[band_mask]

    eps = 1e-12
    log_ratio = 10 * np.log10((psd_denoised_band + eps) / (psd_clean_band + eps))
    
    return np.sqrt(np.mean(log_ratio ** 2))