import numpy as np
import scipy.signal


def snr(clean, denoised):
    noise = clean - denoised
    return 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))


def median_frequency(signal_1d, fs=2048):
    """Medianowa częstotliwość PSD — dobry wskaźnik zmęczenia mięśni i jakości odszumiania."""
    freqs, psd = scipy.signal.welch(signal_1d, fs=fs, nperseg=512)
    cumulative_power = np.cumsum(psd) / np.sum(psd)  # fix: normalizacja
    mf_index = np.searchsorted(cumulative_power, 0.5)
    return freqs[min(mf_index, len(freqs) - 1)]

def prd(clean, denoised):
    """Percent Root-mean-square Difference — znormalizowany RMSE (%).
    Niezależny od amplitudy sygnału, standard w literaturze EMG/ECG.
    < 1% : doskonałe, 1–5% : dobre, > 10% : słabe."""
    return 100.0 * np.sqrt(np.sum((clean - denoised) ** 2) / np.sum(clean ** 2))


def correlation_coefficient(clean, denoised):
    """Współczynnik korelacji Pearsona [0..1].
    Mierzy zachowanie kształtu sygnału niezależnie od amplitudy.
    > 0.99 : bardzo dobry, < 0.95 : odszumianie zniekształca sygnał."""
    return np.corrcoef(clean, denoised)[0, 1]


def snr_improvement(clean, noisy, denoised):
    """Przyrost SNR po odszumianiu [dB] — syntetyczna metryka efektywności filtru.
    To właśnie ten delta SNR który już drukujesz, ale tu jako osobna funkcja."""
    return snr(clean, denoised) - snr(clean, noisy)


def spectral_distortion(clean, denoised, fs=2048):
    """Średnie odchylenie widmowe [dB] — jak bardzo filtr zmienia kształt widma PSD.
    Ważne dla EMG, gdzie rozkład częstotliwości niesie info o typie skurczu.
    Bliżej 0 = mniejsze zniekształcenie widma."""
    _, psd_clean    = scipy.signal.welch(clean,    fs=fs, nperseg=512)
    _, psd_denoised = scipy.signal.welch(denoised, fs=fs, nperseg=512)

    # unikamy log(0)
    eps = 1e-12
    log_ratio = 10 * np.log10((psd_denoised + eps) / (psd_clean + eps))
    return np.sqrt(np.mean(log_ratio ** 2))  # RMS odchylenia widmowego