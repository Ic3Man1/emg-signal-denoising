import wfdb
import numpy as np

from plots import plot_comparison, plot_three_signals, plot_many
from noise import add_noise, add_powerline, add_crosstalk, add_motion_artifacts
from wavelet import wavelet_denoise
from metrics import snr, prd, correlation_coefficient, snr_improvement, spectral_distortion, median_frequency
from savgol import custom_savgol_denoise
from pca import pca_multichannel_denoise

RECORD = wfdb.rdrecord("data_set/session3_participant1_gesture10_trial1/session3_participant1_gesture10_trial1")
SIGNAL = RECORD.p_signal

EMG_INDICIES = [i for i, name in enumerate(RECORD.sig_name) if not('U' in name.upper())]
DATA = SIGNAL[:, EMG_INDICIES]

def main():
    noisy_data = add_noise(DATA, 15)

    signal_power = np.sqrt(np.mean(DATA**2))
    noisy_data = add_powerline(noisy_data, amplitude=0.05 * signal_power)
    noisy_data = add_crosstalk(noisy_data, strength=0.10)
    noisy_data = add_motion_artifacts(noisy_data, amplitude=0.05 * signal_power)

    # plot_comparison(data, noisy_data)

    denoised = np.zeros_like(noisy_data)
    denoised1 = np.zeros_like(noisy_data)
    denoised2 = np.zeros_like(noisy_data)
    for i in range(noisy_data.shape[1]):
        denoised[:, i] = wavelet_denoise(noisy_data[:, i])
        denoised1[:, i] = custom_savgol_denoise(noisy_data[:, i], window_length=17, order=11)
        denoised2[:, i] = pca_multichannel_denoise(noisy_data, variance_to_keep=0.995)[:, i]

    snr_before_all = []
    snr_after_all  = []
    prd_all        = []
    corr_all       = []
    sd_all         = []

    for i in range(DATA.shape[1]):
        clean_ch    = DATA[:, i]
        noisy_ch    = noisy_data[:, i]
        denoised_ch = denoised[:, i]

        sb   = snr(clean_ch, noisy_ch)
        sa   = snr(clean_ch, denoised_ch)
        dsnr = snr_improvement(clean_ch, noisy_ch, denoised_ch)
        p    = prd(clean_ch, denoised_ch)
        r    = correlation_coefficient(clean_ch, denoised_ch)
        sd   = spectral_distortion(clean_ch, denoised_ch)

        snr_before_all.append(sb)
        snr_after_all.append(sa)
        prd_all.append(p)
        corr_all.append(r)
        sd_all.append(sd)

        print(f"Kanał {i:2d} | "
            f"SNR: {sb:5.2f} → {sa:5.2f} dB (Δ{dsnr:+.2f}) | "
            f"PRD: {p:5.2f}% | "
            f"r: {r:.4f} | "
            f"SD: {sd:.2f} dB")
        
    print("\n--- Średnie po wszystkich kanałach ---")
    print(f"SNR before : {np.mean(snr_before_all):.2f} dB")
    print(f"SNR after  : {np.mean(snr_after_all):.2f} dB")
    print(f"ΔSNR       : {np.mean(snr_after_all) - np.mean(snr_before_all):+.2f} dB")
    print(f"PRD        : {np.mean(prd_all):.2f}%")
    print(f"r          : {np.mean(corr_all):.4f}")
    print(f"SD         : {np.mean(sd_all):.2f} dB")

    # plot_comparison(data, denoised)
    # plot_comparison(denoised, noisy_data)
    plot_three_signals(DATA, noisy_data, denoised, channel_idx=0)
    n_samples=1024
    channel_idx=0
    data = {
        "savgol": denoised[:n_samples, channel_idx],
        "wavelet": denoised1[:n_samples, channel_idx],
        "pca": denoised2[:n_samples, channel_idx],
        "oryginal": DATA[:n_samples, channel_idx],
    }
    plot_many(data)

if __name__ == "__main__":
    main()